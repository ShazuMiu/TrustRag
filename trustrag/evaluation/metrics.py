"""Evaluation metrics.

Retrieval    : Recall@k, Precision@k, Hit@k, MRR, nDCG@k        (needs gold doc ids)
Answer       : Exact-match (contains), token F1, ROUGE-L          (needs reference answers)
             : answer relevance (embedding cosine query<->answer)
Grounding    : faithfulness, hallucination rate                   (NLI judge, no labels needed)
Citations    : citation recall / precision / F1 (ALCE definition), ghost-citation rate,
               uncited-claim rate
Behaviour    : abstention rate, correct-abstention on unanswerable questions, latency
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from ..schemas import Passage
from ..verification.claims import extract_claims
from ..verification.verifier import Verifier

# ----------------------------------------------------------------- retrieval


def _dedup(seq: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(seq))


def retrieval_metrics(ranked_doc_ids: Sequence[str], gold: Sequence[str], ks=(1, 3, 5)) -> Dict[str, float]:
    ranked = _dedup(ranked_doc_ids)
    gold_set = set(gold)
    out: Dict[str, float] = {}
    if not gold_set:
        return out
    for k in ks:
        top = ranked[:k]
        hits = sum(d in gold_set for d in top)
        out[f"recall@{k}"] = hits / len(gold_set)
        out[f"precision@{k}"] = hits / k
        out[f"hit@{k}"] = float(hits > 0)
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(top) if d in gold_set)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold_set))))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    rr = 0.0
    for i, d in enumerate(ranked):
        if d in gold_set:
            rr = 1 / (i + 1)
            break
    out["mrr"] = rr
    return out


# ----------------------------------------------------------------- answer quality


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_answer(prediction: str, answers: Sequence[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(normalize_answer(a) and normalize_answer(a) in p for a in answers))


def token_f1(prediction: str, reference: str) -> float:
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(prediction: str, reference: str) -> float:
    a, b = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def answer_relevance(query: str, answer: str, embedder) -> float:
    try:
        q = embedder.encode_query(query)
        a = embedder.encode_docs([answer])[0]
        return float(max(0.0, q @ a))
    except Exception:
        return 0.0


# ----------------------------------------------------------------- grounding & citations


def citation_metrics(answer: str, passages: List[Passage], judge: Verifier) -> Dict[str, float]:
    """ALCE-style citation recall/precision computed with an NLI judge.

    citation recall    = fraction of claims whose cited passages (jointly) entail the claim
    citation precision = fraction of citations that are needed: a citation is precise if it
                         entails the claim alone, or if removing it breaks a supported claim
    faithfulness       = fraction of claims entailed by ANY retrieved passage
    """
    labels = [p.label for p in passages]
    claims = extract_claims(answer, labels)
    if not claims:
        return {"n_claims": 0}
    tau = judge.tau_e
    by_label = {p.label: p for p in passages}
    supported_by_cites = 0
    faithful = 0
    n_cites = precise = 0
    ghost = sum(len(c.ghost_citations) for c in claims)
    uncited = 0
    for c in claims:
        scores = judge.score_passages(c.text, passages)
        if any(s["entail"] >= tau for s in scores.values()):
            faithful += 1
        cites = [l for l in c.citations if l in by_label]
        if not cites:
            uncited += 1
            n_cites += len(c.ghost_citations)
            continue

        def joint(ls: List[str]) -> float:
            if not ls:
                return 0.0
            best = max(scores[l]["entail"] for l in ls)
            if best >= tau or len(ls) == 1:
                return best
            combo = " ".join(scores[l]["window"] for l in ls)
            return float(judge.nli.predict([(combo, c.text)])[0][0])

        set_ok = joint(cites) >= tau
        supported_by_cites += int(set_ok)
        n_cites += len(cites) + len(c.ghost_citations)
        for l in cites:
            alone = scores[l]["entail"] >= tau
            if alone:
                precise += 1
            elif set_ok and joint([x for x in cites if x != l]) < tau:
                precise += 1
    n = len(claims)
    rec = supported_by_cites / n
    prec = precise / n_cites if n_cites else 0.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    return {
        "n_claims": n,
        "faithfulness": faithful / n,
        "hallucination_rate": 1 - faithful / n,
        "citation_recall": rec,
        "citation_precision": prec,
        "citation_f1": f1,
        "ghost_citation_rate": ghost / n_cites if n_cites else 0.0,
        "uncited_claim_rate": uncited / n,
    }


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if not len(arr):
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}
