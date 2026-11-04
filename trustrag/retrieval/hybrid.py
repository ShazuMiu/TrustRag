"""Hybrid retrieval: BM25 (lexical) + dense (semantic) fused with Reciprocal Rank Fusion.

RRF(d) = sum_r 1 / (k + rank_r(d))   — robust, needs no score normalisation,
and recovers documents that either retriever alone would miss (exact terms,
numbers and names via BM25; paraphrases via dense vectors).
"""

from __future__ import annotations

from typing import Dict, List

from ..index.store import CorpusIndex
from ..schemas import Passage


class HybridRetriever:
    def __init__(self, index: CorpusIndex, cfg: dict):
        self.index = index
        rc = cfg["retrieval"]
        self.n = rc.get("candidates_per_retriever", 30)
        self.rrf_k = rc.get("rrf_k", 60)
        self.use_bm25 = rc.get("use_bm25", True)
        self.use_dense = rc.get("use_dense", True)
        if not (self.use_bm25 or self.use_dense):
            raise ValueError("At least one of use_bm25 / use_dense must be true")

    def search(self, query: str, n: int | None = None) -> List[Passage]:
        n = n or self.n
        fused: Dict[int, float] = {}
        raw: Dict[int, Dict[str, float]] = {}

        if self.use_bm25:
            for rank, (i, s) in enumerate(self.index.bm25.search(query, n)):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                raw.setdefault(i, {})["bm25"] = s
        if self.use_dense:
            qv = self.index.embedder.encode_query(query)
            for rank, (i, s) in enumerate(self.index.vindex.search(qv, n)):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                raw.setdefault(i, {})["dense"] = s

        order = sorted(fused, key=lambda i: -fused[i])
        out = []
        for i in order:
            sc = dict(raw[i])
            sc["rrf"] = fused[i]
            out.append(Passage(chunk=self.index.chunks[i], score=fused[i], scores=sc))
        return out
