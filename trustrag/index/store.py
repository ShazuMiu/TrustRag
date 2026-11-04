"""The on-disk index: chunks + BM25 + dense vectors (+ optional FAISS).

Layout of an index directory:
    chunks.jsonl     one Chunk per line
    vectors.npy      float32 [n_chunks, dim], L2-normalised
    bm25.pkl         BM25 statistics
    lsa.pkl          (lite mode only) fitted LSA embedder
    meta.json        embedder backend/model, build time, counts
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..schemas import Chunk
from .bm25 import BM25
from .embeddings import Embedder, build_embedder


class VectorIndex:
    """Exact cosine search with numpy; uses FAISS IndexFlatIP when installed and large."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype(np.float32)
        self._faiss = None
        if len(vectors) > 20000:
            try:
                import faiss

                self._faiss = faiss.IndexFlatIP(vectors.shape[1])
                self._faiss.add(self.vectors)
            except ImportError:
                pass

    def search(self, q: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if not len(self.vectors):
            return []
        if self._faiss is not None:
            D, I = self._faiss.search(q.reshape(1, -1).astype(np.float32), k)
            return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
        sims = self.vectors @ q.astype(np.float32)
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]


class CorpusIndex:
    def __init__(self, chunks: List[Chunk], bm25: BM25, vectors: np.ndarray, embedder: Embedder):
        self.chunks = chunks
        self.bm25 = bm25
        self.vindex = VectorIndex(vectors)
        self.embedder = embedder
        self._by_id = {c.chunk_id: i for i, c in enumerate(chunks)}

    def __len__(self) -> int:
        return len(self.chunks)

    @staticmethod
    def build(chunks: List[Chunk], cfg: dict, index_dir: Optional[Path] = None) -> "CorpusIndex":
        texts = [f"{c.title}. {c.text}" if c.title else c.text for c in chunks]
        embedder = build_embedder(cfg)
        if embedder.needs_fit:
            embedder.fit(texts)
        vectors = embedder.encode_docs(texts)
        bm25 = BM25().fit(texts)
        idx = CorpusIndex(chunks, bm25, vectors, embedder)
        if index_dir is not None:
            idx.save(Path(index_dir), cfg)
        return idx

    def save(self, directory: Path, cfg: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(directory / "vectors.npy", self.vindex.vectors)
        with open(directory / "bm25.pkl", "wb") as fh:
            pickle.dump(self.bm25.state(), fh)
        self.embedder.save(directory)
        meta = {
            "embeddings_backend": cfg["embeddings"]["backend"],
            "embeddings_model": cfg["embeddings"].get("model"),
            "n_chunks": len(self.chunks),
            "n_docs": len({c.doc_id for c in self.chunks}),
            "dim": int(self.vindex.vectors.shape[1]) if len(self.chunks) else 0,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @staticmethod
    def load(directory: Path, cfg: dict) -> "CorpusIndex":
        directory = Path(directory)
        if not (directory / "chunks.jsonl").exists():
            raise FileNotFoundError(f"No index at {directory}. Run: python -m trustrag.cli ingest ...")
        meta = json.loads((directory / "meta.json").read_text())
        if meta["embeddings_backend"] != cfg["embeddings"]["backend"]:
            raise ValueError(
                f"Index was built with embeddings backend '{meta['embeddings_backend']}' but config uses "
                f"'{cfg['embeddings']['backend']}'. Re-run ingest with this config.")
        with open(directory / "chunks.jsonl", encoding="utf-8") as fh:
            chunks = [Chunk.from_dict(json.loads(l)) for l in fh if l.strip()]
        vectors = np.load(directory / "vectors.npy")
        with open(directory / "bm25.pkl", "rb") as fh:
            bm25 = BM25.from_state(pickle.load(fh))
        embedder = build_embedder(cfg, directory)
        return CorpusIndex(chunks, bm25, vectors, embedder)
