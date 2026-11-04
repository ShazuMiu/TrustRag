"""Dense text embeddings.

* SentenceTransformerEmbedder — neural (BGE / E5 / MiniLM ...). Full mode.
* LSAEmbedder — TF-IDF (word + char n-grams) + truncated SVD. Offline lite mode;
  needs to be fitted on the corpus at index time and is saved with the index.
All embedders return L2-normalised float32 matrices, so dot product == cosine.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class Embedder:
    name = "base"
    needs_fit = False

    def fit(self, texts: List[str]) -> "Embedder":
        return self

    def encode_docs(self, texts: List[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        pass


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str, device: str = "cpu", batch_size: int = 32, query_instruction: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size
        # BGE models expect an instruction prefix on queries (not on passages)
        self.query_instruction = query_instruction if "bge" in model.lower() else ""
        if "e5" in model.lower():
            self.query_instruction, self._doc_prefix = "query: ", "passage: "
        else:
            self._doc_prefix = ""

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        texts = [self._doc_prefix + t for t in texts]
        return _normalize(self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=len(texts) > 200,
                                            normalize_embeddings=True))

    def encode_query(self, text: str) -> np.ndarray:
        return _normalize(self.model.encode([self.query_instruction + text], normalize_embeddings=True))[0]


class LSAEmbedder(Embedder):
    """Latent Semantic Analysis embeddings — a reasonable offline stand-in."""

    name = "lsa"
    needs_fit = True

    def __init__(self, dims: int = 256):
        self.dims = dims
        self.word_vec = None
        self.char_vec = None
        self.svd = None

    def fit(self, texts: List[str]) -> "LSAEmbedder":
        from scipy.sparse import hstack
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english")
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        X = hstack([self.word_vec.fit_transform(texts), self.char_vec.fit_transform(texts)]).tocsr()
        k = max(2, min(self.dims, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        return self

    def _transform(self, texts: List[str]) -> np.ndarray:
        from scipy.sparse import hstack

        X = hstack([self.word_vec.transform(texts), self.char_vec.transform(texts)]).tocsr()
        return _normalize(self.svd.transform(X))

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        return self._transform(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._transform([text])[0]

    def save(self, directory: Path) -> None:
        with open(Path(directory) / "lsa.pkl", "wb") as fh:
            pickle.dump({"dims": self.dims, "word": self.word_vec, "char": self.char_vec, "svd": self.svd}, fh)

    @staticmethod
    def load(directory: Path) -> "LSAEmbedder":
        with open(Path(directory) / "lsa.pkl", "rb") as fh:
            st = pickle.load(fh)
        e = LSAEmbedder(st["dims"])
        e.word_vec, e.char_vec, e.svd = st["word"], st["char"], st["svd"]
        return e


def build_embedder(cfg: dict, index_dir: Optional[Path] = None) -> Embedder:
    ecfg = cfg["embeddings"]
    if ecfg["backend"] == "sentence-transformers":
        return SentenceTransformerEmbedder(ecfg["model"], ecfg.get("device", "cpu"), ecfg.get("batch_size", 32),
                                           ecfg.get("query_instruction", ""))
    if ecfg["backend"] == "lsa":
        if index_dir is not None and (Path(index_dir) / "lsa.pkl").exists():
            return LSAEmbedder.load(Path(index_dir))
        return LSAEmbedder(ecfg.get("lsa_dims", 256))
    raise ValueError(f"Unknown embeddings backend: {ecfg['backend']}")
