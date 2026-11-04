"""Citation correction + hallucination repair (CiteFix / VeriCite / HALT-RAG inspired).

Per claim:
  ghost citation (label not in sources)      -> removed
  supported                                   -> keep; prune cited sources that add no support
  misattributed                               -> re-cite to the passage(s) that do entail it
  unsupported / contradicted                  -> rewrite from the best evidence with the LLM and
                                                 re-verify (one attempt); otherwise remove the claim
Then a HALT-style policy decides: answered | partial | abstained.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..generation.prompts import REWRITE_CLAIM_SYSTEM, rewrite_claim_user
from ..schemas import Claim, ClaimVerdict, Passage
from .claims import extract_claims, render_claim
from .verifier import CONTRADICTED, MISATTRIBUTED, SUPPORTED, UNSUPPORTED, Verifier


class Corrector:
    def __init__(self, verifier: Verifier, cfg: dict, llm=None):
        c = cfg["correction"]
        self.verifier = verifier
        self.enabled = c.get("enabled", True)
        self.recite = c.get("recite", True)
        self.prune = c.get("prune_citations", True)
        self.regenerate = c.get("regenerate", True)
        self.max_cites = c.get("max_citations_per_claim", 2)
        self.llm = llm
        self.tau_e = verifier.tau_e

    def _supporting(self, scores: Dict[str, dict], labels: Optional[List[str]] = None) -> List[str]:
        labels = labels if labels is not None else list(scores)
        good = [l for l in labels if scores[l]["entail"] >= self.tau_e]
        return sorted(good, key=lambda l: -scores[l]["entail"])[: self.max_cites]

    def correct(self, question: str, verdict: ClaimVerdict, scores: Dict[str, dict],
                passages: List[Passage]) -> ClaimVerdict:
        v = verdict
        cl = v.claim
        v.final_text = cl.text
        v.final_citations = list(cl.citations)
        if not self.enabled:
            v.action = "kept"
            return v

        if v.label == SUPPORTED:
            if self.prune:
                individually = self._supporting(scores, cl.citations)
                if individually and set(individually) != set(cl.citations):
                    v.final_citations = individually
                    v.action = "pruned_citations"
                    return v
            v.action = "pruned_citations" if cl.ghost_citations else "kept"
            return v

        if v.label == MISATTRIBUTED and self.recite:
            v.final_citations = self._supporting(scores) or [v.best_label]
            v.action = "recited"
            return v

        # unsupported / contradicted (or misattributed with recite disabled)
        if self.regenerate and self.llm is not None and getattr(self.llm, "is_generative", False):
            ranked = sorted(scores, key=lambda l: -scores[l]["entail"])[:2]
            evidence = [p for p in passages if p.label in ranked]
            if evidence:
                try:
                    out = self.llm.complete(REWRITE_CLAIM_SYSTEM, rewrite_claim_user(question, cl.text, evidence),
                                            max_tokens=120).strip()
                except Exception:
                    out = "DROP"
                if out and "DROP" not in out.upper()[:10]:
                    new = extract_claims(out, [p.label for p in passages])
                    if new:
                        nc = new[0]
                        if not nc.citations:
                            nc.citations = [p.label for p in evidence]
                        nv, nscores = self.verifier.verify(nc, passages)
                        if nv.label == SUPPORTED:
                            v.final_text = nc.text
                            v.final_citations = self._supporting(nscores, nc.citations) or nc.citations
                            v.action = "rewritten"
                            return v
        v.action = "removed"
        v.final_citations = []
        return v


def apply_policy(verdicts: List[ClaimVerdict], cfg: dict) -> Dict[str, object]:
    """HALT-style answer policy. Returns status, final answer text and confidence."""
    pol = cfg["policy"]
    kept = [v for v in verdicts if v.action != "removed"]
    if not verdicts or not kept:
        return {"status": "abstained", "answer": pol["abstain_message"], "confidence": 0.0}
    ratio = len(kept) / len(verdicts)
    answer = " ".join(render_claim(v.final_text, v.final_citations) for v in kept)
    status = "answered" if ratio >= pol.get("min_kept_ratio", 0.5) else "partial"
    if status == "partial":
        answer += " (Note: some statements were removed because the sources did not support them.)"
    support = [max(v.entailment, v.best_entailment) if v.action != "rewritten" else 1.0 for v in kept]
    confidence = (sum(support) / len(support)) * ratio
    return {"status": status, "answer": answer, "confidence": float(min(1.0, confidence))}
