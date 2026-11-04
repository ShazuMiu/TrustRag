"""Claim-level citation verification.

Key differences from TrustRAG v1:
  * The claim is checked against the passage(s) it ACTUALLY CITES (citation
    correctness), and separately against all retrieved passages (to detect
    misattribution and to find the right source for correction).
  * Premises are sentence windows (1-3 sentences) instead of whole abstracts, so
    nothing is silently truncated at the NLI model's 512-token limit and the
    exact supporting evidence is returned.
  * Multi-source claims: if no single cited passage entails the claim, the best
    windows of all cited passages are concatenated and re-tested.
  * Numeric guard: a claim whose numbers do not appear in the evidence window
    cannot be "supported" by it (catches distorted numbers / dates).

Verdict labels:
  supported      cited source(s) entail the claim
  misattributed  cited source(s) do not, but another retrieved passage does
  contradicted   evidence contradicts the claim (or numbers disagree)
  unsupported    nothing retrieved supports it  -> hallucination candidate
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np

from ..schemas import Claim, ClaimVerdict, Passage
from ..text import extract_numbers, split_sentences
from .nli import NLIModel

SUPPORTED, MISATTRIBUTED, CONTRADICTED, UNSUPPORTED = "supported", "misattributed", "contradicted", "unsupported"


def evidence_windows(text: str, sizes=(1, 2, 3), full_if_words_under: int = 300) -> List[str]:
    sents = split_sentences(text)
    wins: List[str] = []
    for w in sizes:
        if w > len(sents):
            continue
        for i in range(len(sents) - w + 1):
            wins.append(" ".join(sents[i:i + w]))
    if len(text.split()) < full_if_words_under:
        wins.append(" ".join(sents))
    return list(dict.fromkeys(x for x in wins if x)) or [text]


_CODE = re.compile(r"\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]+\b")


def numbers_ok(claim: str, evidence: str) -> bool:
    """Exact-value guard: every number (8,611 / 6.15 / 2022) and alphanumeric code
    (BM25, K2, T5, P100) in the claim must literally appear in the evidence."""
    cn = extract_numbers(claim)
    if cn:
        en = set(extract_numbers(evidence))
        if not all(n in en for n in cn):
            return False
    codes = {c.lower() for c in _CODE.findall(claim)}
    if codes:
        ev = {c.lower() for c in _CODE.findall(evidence)}
        if not codes <= ev:
            return False
    return True


class Verifier:
    def __init__(self, nli: NLIModel, cfg: dict):
        v = cfg["verification"]
        self.nli = nli
        self.tau_e = v.get("entail_threshold", 0.5)
        self.tau_c = v.get("contradiction_threshold", 0.6)
        self.numeric_guard = v.get("numeric_guard", True)
        self.window_sizes = tuple(cfg["nli"].get("window_sizes", [1, 2, 3]))

    # ---------------------------------------------------------------- scoring
    def score_passages(self, claim_text: str, passages: List[Passage]) -> Dict[str, dict]:
        """For each passage: best entailment window, max contradiction, numeric flag."""
        pairs: List[Tuple[str, str]] = []
        owners: List[Tuple[str, str]] = []
        for p in passages:
            for w in evidence_windows(p.chunk.text, self.window_sizes):
                pairs.append((w, claim_text))
                owners.append((p.label, w))
        probs = self.nli.predict(pairs) if pairs else np.zeros((0, 2))
        res: Dict[str, dict] = {p.label: {"entail": 0.0, "raw_entail": 0.0, "contra": 0.0, "window": "",
                                          "numeric_mismatch": False} for p in passages}
        for (lab, win), (e, c) in zip(owners, probs):
            r = res[lab]
            raw_e = float(e)
            ok = (not self.numeric_guard) or numbers_ok(claim_text, win)
            eff = raw_e if ok else min(raw_e, 0.05)
            if eff > r["entail"] or (not r["window"]):
                r["entail"], r["window"] = eff, win
            if raw_e > r["raw_entail"]:
                r["raw_entail"] = raw_e
                r["numeric_mismatch"] = not ok
            r["contra"] = max(r["contra"], float(c))
        return res

    # ---------------------------------------------------------------- verdict
    def verify(self, claim: Claim, passages: List[Passage]) -> Tuple[ClaimVerdict, Dict[str, dict]]:
        scores = self.score_passages(claim.text, passages)
        cited = [l for l in claim.citations if l in scores]
        v = ClaimVerdict(claim=claim)

        cited_e, cited_win = 0.0, ""
        for l in cited:
            if scores[l]["entail"] > cited_e:
                cited_e, cited_win = scores[l]["entail"], scores[l]["window"]

        # multi-source synthesis: concatenate best windows of the cited passages
        if len(cited) > 1 and cited_e < self.tau_e:
            combo = " ".join(scores[l]["window"] for l in cited)
            e, c = self.nli.predict([(combo, claim.text)])[0]
            if (not self.numeric_guard) or numbers_ok(claim.text, combo):
                if e > cited_e:
                    cited_e, cited_win = float(e), combo

        best_label = max(scores, key=lambda l: scores[l]["entail"]) if scores else ""
        best_e = scores[best_label]["entail"] if best_label else 0.0
        max_contra = max((s["contra"] for s in scores.values()), default=0.0)
        num_conflict = bool(best_label) and scores[best_label]["numeric_mismatch"] and \
            max(s["raw_entail"] for s in scores.values()) >= self.tau_e

        v.entailment, v.best_label, v.best_entailment = cited_e, best_label, best_e
        v.contradiction = max((scores[l]["contra"] for l in cited), default=max_contra)
        v.numeric_mismatch = any(s["numeric_mismatch"] for s in scores.values()) and best_e < self.tau_e

        if cited and cited_e >= self.tau_e:
            v.label, v.evidence = SUPPORTED, cited_win
        elif best_e >= self.tau_e:
            v.label, v.evidence = MISATTRIBUTED, scores[best_label]["window"]
        elif max_contra >= self.tau_c or num_conflict:
            v.label = CONTRADICTED
            v.evidence = max(scores.values(), key=lambda s: (s["contra"], s["raw_entail"]))["window"]
        else:
            v.label = UNSUPPORTED
            v.evidence = scores[best_label]["window"] if best_label else ""
        return v, scores
