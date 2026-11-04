"""Natural Language Inference backends: P(entailment), P(contradiction) for (premise, hypothesis).

* TransformersNLI — e.g. MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (full mode).
  Label order is read from the model config, never hard-coded (a frequent bug).
* LexicalNLI — transparent offline heuristic: content-word & bigram coverage of the
  hypothesis by the premise, with negation-mismatch -> contradiction.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..text import has_negation, tokenize

Pair = Tuple[str, str]  # (premise, hypothesis)


class NLIModel:
    name = "base"

    def predict(self, pairs: Sequence[Pair]) -> np.ndarray:  # -> [n, 2] columns (entail, contra)
        raise NotImplementedError


class LexicalNLI(NLIModel):
    name = "lexical"

    def predict(self, pairs: Sequence[Pair]) -> np.ndarray:
        out = np.zeros((len(pairs), 2))
        for i, (prem, hyp) in enumerate(pairs):
            h = tokenize(hyp)
            if not h:
                continue
            p = tokenize(prem)
            pset = set(p)
            cov = sum(t in pset for t in h) / len(h)
            hb = set(zip(h, h[1:]))
            bcov = len(hb & set(zip(p, p[1:]))) / len(hb) if hb else cov
            entail = (cov ** 1.5) * (0.6 + 0.4 * bcov)
            contra = 0.0
            if cov >= 0.6 and has_negation(hyp) != has_negation(prem):
                contra = cov
                entail *= 0.2
            out[i] = (min(entail, 1.0), min(contra, 1.0))
        return out


class TransformersNLI(NLIModel):
    def __init__(self, model: str, device: str = "cpu", batch_size: int = 16, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.name = model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.ent_idx = next(i for i, l in labels.items() if l.startswith("entail"))
        self.con_idx = next((i for i, l in labels.items() if l.startswith("contra")), None)

    def predict(self, pairs: Sequence[Pair]) -> np.ndarray:
        out = np.zeros((len(pairs), 2))
        for s in range(0, len(pairs), self.batch_size):
            batch = pairs[s:s + self.batch_size]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
            out[s:s + len(batch), 0] = probs[:, self.ent_idx]
            if self.con_idx is not None:
                out[s:s + len(batch), 1] = probs[:, self.con_idx]
        return out


def build_nli(backend: str, model: str = "", device: str = "cpu", batch_size: int = 16,
              max_length: int = 512) -> NLIModel:
    if backend == "transformers":
        return TransformersNLI(model, device, batch_size, max_length)
    if backend == "lexical":
        return LexicalNLI()
    raise ValueError(f"Unknown NLI backend: {backend}")
