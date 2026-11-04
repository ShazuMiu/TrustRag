"""Split an answer into sentence-level claims and parse their citation markers.

Accepts [S1], [1], [S1, S3], [S1][S2]; citations placed after the full stop
("... text. [S1] Next ...") are moved back onto the sentence they belong to.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from ..schemas import Claim
from ..text import normalize_ws, split_sentences

CITE_GROUP = re.compile(r"\[\s*(?:S|s|Source\s*)?\d+(?:\s*[,;]\s*(?:S|s|Source\s*)?\d+)*\s*\]")
CITE_NUM = re.compile(r"\d+")
ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"


def is_abstention(answer: str) -> bool:
    a = (answer or "").strip().upper()
    return not a or ABSTAIN_TOKEN in a


def _labels(group: str) -> List[str]:
    return [f"S{n}" for n in CITE_NUM.findall(group)]


def strip_citations(text: str) -> str:
    return normalize_ws(re.sub(r"\s+([.!?,;:])", r"\1", CITE_GROUP.sub("", text)))


def extract_claims(answer: str, valid_labels: Iterable[str] = ()) -> List[Claim]:
    if is_abstention(answer):
        return []
    valid = set(valid_labels)
    text = normalize_ws(answer)
    # move trailing citations before the sentence-final punctuation
    text = re.sub(r"([.!?])\s*((?:" + CITE_GROUP.pattern + r"\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", text)
    claims: List[Claim] = []
    for sent in split_sentences(text):
        labels: List[str] = []
        for g in CITE_GROUP.findall(sent):
            for lab in _labels(g):
                if lab not in labels:
                    labels.append(lab)
        body = strip_citations(sent)
        if len(body.split()) < 3:   # fragments such as "[S1]." or headings
            continue
        good = [l for l in labels if not valid or l in valid]
        ghost = [l for l in labels if valid and l not in valid]
        claims.append(Claim(claim_id=len(claims), text=body, citations=good, ghost_citations=ghost))
    return claims


def render_claim(text: str, citations: List[str]) -> str:
    body = text.rstrip()
    end = body[-1] if body and body[-1] in ".!?" else "."
    body = body.rstrip(".!?").rstrip()
    cites = "".join(f"[{c}]" for c in citations)
    return f"{body} {cites}{end}" if cites else f"{body}{end}"
