"""Okapi BM25 (sparse / lexical retrieval), implemented with numpy only."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from ..text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.doc_len = np.zeros(0)
        self.avgdl = 0.0
        self._postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, texts: List[str]) -> "BM25":
        self.doc_freqs = [Counter(tokenize(t)) for t in texts]
        self.doc_len = np.array([sum(c.values()) for c in self.doc_freqs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        df: Counter = Counter()
        for c in self.doc_freqs:
            df.update(c.keys())
        n = len(texts)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._build_postings()
        return self

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.doc_freqs))
        if not q or not len(out):
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in set(q):
            post = self._postings.get(term)
            if post is None:
                continue
            idx, tf = post
            out[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[idx])
        return out

    def _build_postings(self) -> None:
        tmp: Dict[str, Tuple[list, list]] = {}
        for i, c in enumerate(self.doc_freqs):
            for t, f in c.items():
                a, b = tmp.setdefault(t, ([], []))
                a.append(i)
                b.append(f)
        self._postings = {t: (np.array(a), np.array(b, dtype=float)) for t, (a, b) in tmp.items()}

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        s = self.scores(query)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]

    # persistence helpers
    def state(self) -> dict:
        return {"k1": self.k1, "b": self.b, "doc_freqs": [dict(c) for c in self.doc_freqs]}

    @staticmethod
    def from_state(st: dict) -> "BM25":
        bm = BM25(st["k1"], st["b"])
        bm.doc_freqs = [Counter(d) for d in st["doc_freqs"]]
        bm.doc_len = np.array([sum(c.values()) for c in bm.doc_freqs], dtype=float)
        bm.avgdl = float(bm.doc_len.mean()) if len(bm.doc_len) else 0.0
        df: Counter = Counter()
        for c in bm.doc_freqs:
            df.update(c.keys())
        n = len(bm.doc_freqs)
        bm.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        bm._build_postings()
        return bm
