"""Small, dependency-free text utilities (tokenising, stemming, sentence split)."""

from __future__ import annotations

import re
from typing import List

STOPWORDS = frozenset(
    """a an the and or but if then else of in on at to for from by with without
    as is are was were be been being am do does did doing have has had having
    this that these those it its it's they them their there here what which who
    whom whose when where why how all any both each few more most other some such
    only own same so than too very can will just should would could may might must
    shall into over under again further once about above below between through
    during before after up down out off i me my we our you your he him his she her
    also using use used based via per et al s t""".split()
)

NEGATIONS = frozenset(
    "not no never none nobody nothing neither nor cannot can't don't doesn't didn't "
    "isn't aren't wasn't weren't won't wouldn't shouldn't hasn't haven't hadn't without "
    "fails failed lack lacks".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?", re.I)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%|percent)?", re.I)

# Abbreviations that should not end a sentence.
_ABBREV = ("e.g.", "i.e.", "et al.", "vs.", "Dr.", "Prof.", "Fig.", "No.", "approx.", "U.S.")


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def stem(word: str) -> str:
    """Very small suffix stripper (Porter-lite). Good enough for lexical matching."""
    w = word
    for suf in ("ational", "ization", "fulness", "ousness", "iveness",
                "ations", "ation", "ments", "ment", "ness", "ings", "ing",
                "ies", "ied", "edly", "ers", "er", "ed", "ly", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            if suf in ("ies", "ied"):
                return w[: -len(suf)] + "y"
            return w[: -len(suf)]
    return w


def tokenize(text: str, remove_stopwords: bool = True, do_stem: bool = True) -> List[str]:
    toks = [t.lower().replace(",", "") for t in _TOKEN_RE.findall(text or "")]
    out = []
    for t in toks:
        if remove_stopwords and t in STOPWORDS:
            continue
        out.append(stem(t) if do_stem and not t[0].isdigit() else t)
    return out


def split_sentences(text: str) -> List[str]:
    """Rule-based sentence splitter that protects common abbreviations."""
    text = normalize_ws(text)
    if not text:
        return []
    protected = text
    for i, ab in enumerate(_ABBREV):
        protected = protected.replace(ab, ab.replace(".", f"§{i}§"))
    parts = _SENT_RE.split(protected)
    out = []
    for p in parts:
        for i, ab in enumerate(_ABBREV):
            p = p.replace(ab.replace(".", f"§{i}§"), ab)
        p = p.strip()
        if p:
            out.append(p)
    return out


def extract_numbers(text: str) -> List[str]:
    """Return normalised numbers found in text ('8,848.86' -> '8848.86', '57 %' -> '57')."""
    nums = []
    for m in _NUM_RE.finditer(text or ""):
        whole = m.group(1).replace(",", "")
        frac = m.group(2)
        nums.append(f"{whole}.{frac}" if frac else whole)
    return nums


def has_negation(text: str) -> bool:
    words = re.findall(r"[a-z']+", (text or "").lower())
    return any(w in NEGATIONS for w in words)
