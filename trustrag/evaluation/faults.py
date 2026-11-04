"""Controlled fault-injection test for the verification + correction layer.

Why: on real data we never know exactly which generated sentences are wrong.
Here we take answers whose sentences the judge confirms as supported, then inject
KNOWN errors (the error taxonomy of the review, RQ1):

    wrong_citation  citation swapped to a source that does not support the claim
    ghost_citation  citation to a source label that does not exist  (fabricated attribution)
    number          a number in the claim is changed                (numeric distortion)
    negation        the claim's meaning is flipped with "not"       (contradiction)
    fabricated      an extra sentence copied from an UNRETRIEVED doc, cited to a real source

Then we measure what the naive pipeline (passes everything through) and TrustRAG
(verify + correct) deliver. Ground truth comes from the injection itself.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List

import numpy as np

from ..generation.prompts import GENERATION_SYSTEM, generation_user
from ..verification.claims import extract_claims, render_claim
from ..verification.corrector import apply_policy

ERR_TYPES = ["wrong_citation", "ghost_citation", "number", "negation", "fabricated"]
CITATION_ERRORS = {"wrong_citation", "ghost_citation"}
HALLUCINATIONS = {"number", "negation", "fabricated"}

_AUX = re.compile(r"\b(is|are|was|were|can|does|do|has|have|will|uses|use)\b")


def _perturb_number(text: str, rng: random.Random):
    m = list(re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", text))  # standalone numbers only
    if not m:
        return None
    t = rng.choice(m)
    val = t.group(0)
    if "." in val:
        new = f"{float(val) * 1.5:.{len(val.split('.')[1])}f}"
    else:
        n = int(val)
        new = str(n + rng.choice([1, 2, 3, 7]) if n >= 1000 else max(2, n * 2 + 1))
    return text[: t.start()] + new + text[t.end():]


def _negate(text: str):
    m = _AUX.search(text)
    if not m:
        return None
    word = m.group(1)
    if word in ("uses", "use"):
        return text[: m.start()] + ("does not use" if word == "uses" else "do not use") + text[m.end():]
    return text[: m.end()] + " not" + text[m.end():]


def run_fault_injection(base, examples, judge, seed: int = 13, error_rate: float = 0.6) -> Dict:
    rng = random.Random(seed)
    tau = judge.tau_e
    all_chunks = base.index.chunks
    records: List[dict] = []
    for e in examples:
        if not e.answerable:
            continue
        timings: Dict[str, float] = {}
        passages, decision, _ = base.retrieve(e.question, timings)
        if len(passages) < 2:
            continue
        labels = [p.label for p in passages]
        raw = base.llm.complete(GENERATION_SYSTEM.format(max_sentences=6), generation_user(e.question, passages))
        clean = []
        for c in extract_claims(raw, labels):
            if not c.citations:
                continue
            sc = judge.score_passages(c.text, passages)
            if max(sc[l]["entail"] for l in c.citations) >= tau:
                clean.append((c, sc))
        if not clean:
            continue

        noisy = []  # (text, citations(list incl. ghost), error_type or None)
        for c, sc in clean:
            err = rng.choice(ERR_TYPES[:4]) if rng.random() < error_rate else None
            text, cites = c.text, list(c.citations)
            if err == "wrong_citation":
                wrong = [l for l in labels if sc[l]["entail"] < 0.2 and l not in cites]
                if wrong:
                    cites = [rng.choice(wrong)]
                else:
                    err = None
            elif err == "ghost_citation":
                cites = [f"S{len(labels) + rng.randint(1, 4)}"]
            elif err == "number":
                t2 = _perturb_number(text, rng)
                if t2:
                    text = t2
                else:
                    err = None
            elif err == "negation":
                t2 = _negate(text)
                if t2:
                    text = t2
                else:
                    err = None
            noisy.append((text, cites, err))
        if rng.random() < 0.5:  # fabricated sentence from a document that was NOT retrieved
            used = {p.chunk.doc_id for p in passages}
            pool = [ch for ch in all_chunks if ch.doc_id not in used]
            if pool:
                from ..text import split_sentences

                sents = [s for s in split_sentences(rng.choice(pool).text) if len(s.split()) >= 6]
                if sents:
                    noisy.append((rng.choice(sents), [rng.choice(labels)], "fabricated"))
        noisy_answer = " ".join(render_claim(t, c) for t, c, _ in noisy)

        # --- TrustRAG verification + correction on the noisy answer
        claims = extract_claims(noisy_answer, labels)
        if len(claims) != len(noisy):  # sentence split mismatch -> skip example
            continue
        verdicts = []
        for cl in claims:
            v, scores = base.verifier.verify(cl, passages)
            base.corrector.correct(e.question, v, scores, passages)
            verdicts.append(v)
        apply_policy(verdicts, base.cfg)

        for (t, c, err), v in zip(noisy, verdicts):
            kept = v.action != "removed"
            # does the claim as finally delivered have a supporting citation?
            final_ok = False
            if kept:
                sc = judge.score_passages(v.final_text, passages)
                final_ok = any(sc[l]["entail"] >= tau for l in v.final_citations if l in sc)
            naive_ok = False
            sc0 = judge.score_passages(t, passages)
            naive_ok = any(sc0[l]["entail"] >= tau for l in c if l in sc0) and err not in HALLUCINATIONS
            records.append({"qid": e.qid, "error": err or "clean", "text": t, "citations": ",".join(c),
                            "final_text": v.final_text if kept else "", "final_citations": ",".join(v.final_citations),
                            "label": v.label, "action": v.action,
                            "kept": kept, "final_ok": final_ok, "naive_ok": naive_ok})

    return summarize_faults(records)


def _hall_fixed(r: dict) -> bool:
    """A hallucinated claim is fixed if removed, or rewritten into a supported claim."""
    return (not r["kept"]) or (r["action"] == "rewritten" and r["final_ok"])


def summarize_faults(records: List[dict]) -> Dict:
    if not records:
        return {"n_claims": 0}
    out: Dict = {"n_claims": len(records), "by_type": {}, "records": records}
    for t in ERR_TYPES + ["clean"]:
        rs = [r for r in records if r["error"] == t]
        if not rs:
            continue
        d = {"n": len(rs)}
        if t == "clean":
            d["kept_correct"] = float(np.mean([r["final_ok"] for r in rs]))
            d["false_removal"] = float(np.mean([not r["kept"] for r in rs]))
        elif t in CITATION_ERRORS:
            d["detected"] = float(np.mean([r["label"] != "supported" for r in rs]))
            d["fixed"] = float(np.mean([r["final_ok"] for r in rs]))
        else:
            d["detected"] = float(np.mean([r["label"] != "supported" for r in rs]))
            d["fixed"] = float(np.mean([_hall_fixed(r) for r in rs]))
        out["by_type"][t] = d
    errs = [r for r in records if r["error"] != "clean"]
    hall = [r for r in records if r["error"] in HALLUCINATIONS]
    cit = [r for r in records if r["error"] in CITATION_ERRORS]
    delivered = [r for r in records if r["kept"]]
    out["naive"] = {
        "hallucinated_claims_in_answer": len(hall) / len(records),
        "claims_with_correct_citation": float(np.mean([r["naive_ok"] for r in records])),
    }
    out["trustrag"] = {
        "hallucinated_claims_in_answer": (sum(not _hall_fixed(r) for r in hall) / len(delivered))
        if delivered else 0.0,
        "claims_with_correct_citation": float(np.mean([r["final_ok"] for r in delivered])) if delivered else 0.0,
        "error_detection_recall": float(np.mean([r["label"] != "supported" for r in errs])) if errs else 0.0,
        "citation_errors_fixed": float(np.mean([r["final_ok"] for r in cit])) if cit else 0.0,
        "hallucinations_removed": float(np.mean([_hall_fixed(r) for r in hall])) if hall else 0.0,
        "clean_claims_falsely_removed": out["by_type"].get("clean", {}).get("false_removal", 0.0),
    }
    return out
