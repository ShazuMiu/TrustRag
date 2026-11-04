"""Second-stage reranking. Produces a calibrated relevance score in [0, 1]
that the retrieval gate (CRAG-style evaluator) can threshold.

* CrossEncoderReranker — e.g. BAAI/bge-reranker-base (full mode)
* LexicalReranker — IDF-weighted query-term coverage + dense cosine (offline)
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from ..index.store import CorpusIndex
from ..schemas import Passage
from ..text import tokenize


class LexicalReranker:
    name = "lexical"

    def __init__(self, index: CorpusIndex):
        self.index = index

    def _idf(self, term: str) -> float:
        return self.index.bm25.idf.get(term, math.log(1 + len(self.index) + 0.5))

    def score(self, query: str, passages: List[Passage]) -> List[float]:
        q_terms = list(dict.fromkeys(tokenize(query)))
        if not q_terms:
            return [0.0] * len(passages)
        weights = np.array([self._idf(t) for t in q_terms])
        qv = self.index.embedder.encode_query(query)
        q_bigrams = set(zip(q_terms, q_terms[1:]))
        out = []
        for p in passages:
            doc = f"{p.chunk.title}. {p.chunk.text}"
            toks = tokenize(doc)
            tset = set(toks)
            cov = float(sum(w for t, w in zip(q_terms, weights) if t in tset) / weights.sum())
            bigram = 0.0
            if q_bigrams:
                d_bigrams = set(zip(toks, toks[1:]))
                bigram = len(q_bigrams & d_bigrams) / len(q_bigrams)
            dv = self.index.embedder.encode_docs([doc])[0]
            cos = max(0.0, float(dv @ qv))
            out.append(min(1.0, 0.6 * cov ** 1.5 + 0.25 * cos + 0.15 * bigram))
        return out


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, device=device, max_length=512)

    def score(self, query: str, passages: List[Passage]) -> List[float]:
        if not passages:
            return []
        pairs = [(query, f"{p.chunk.title}. {p.chunk.text}") for p in passages]
        s = np.asarray(self.model.predict(pairs, show_progress_bar=False), dtype=float)
        if s.min() < 0 or s.max() > 1:  # raw logits -> probability
            s = 1 / (1 + np.exp(-s))
        return s.tolist()


class Reranker:
    def __init__(self, index: CorpusIndex, cfg: dict):
        rc = cfg["reranker"]
        self.enabled = rc.get("enabled", True)
        self.top_n = rc.get("top_n", 20)
        if rc.get("backend") == "cross-encoder":
            self.impl = CrossEncoderReranker(rc["model"], cfg["embeddings"].get("device", "cpu"))
        else:
            self.impl = LexicalReranker(index)

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        cands = passages[: self.top_n]
        scores = self.impl.score(query, cands)
        for p, s in zip(cands, scores):
            p.scores["rerank"] = float(s)
            p.score = float(s)
        if self.enabled:
            cands = sorted(cands, key=lambda p: -p.score)
        return cands
