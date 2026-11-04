"""Retrieval quality gate (CRAG-inspired) + corrective fallback actions.

decision = CORRECT    if best relevance >= upper
         = INCORRECT  if best relevance <  lower
         = AMBIGUOUS  otherwise

On AMBIGUOUS / INCORRECT the pipeline tries corrective actions in order:
  1. query rewrite (LLM rewrite, or pseudo-relevance-feedback expansion offline)
     + wider candidate pool
  2. external search (Semantic Scholar / DuckDuckGo), only if enabled
If the evidence is still INCORRECT, the pipeline ABSTAINS instead of letting
the LLM answer from parametric memory (the main source of hallucination).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

from ..schemas import Passage
from ..text import tokenize

CORRECT, AMBIGUOUS, INCORRECT = "correct", "ambiguous", "incorrect"


class RetrievalGate:
    def __init__(self, cfg: dict):
        g = cfg["retrieval_gate"]
        self.enabled = g.get("enabled", True)
        self.upper = g.get("upper", 0.55)
        self.lower = g.get("lower", 0.25)
        self.keep_min = g.get("keep_min_score", 0.15)
        self.max_rounds = g.get("max_fallback_rounds", 2)
        self.abstain_if_incorrect = g.get("abstain_if_incorrect", True)
        self.entity_check = g.get("entity_check", True)

    def missing_entities(self, query: str, passages: List[Passage], top: int = 5) -> List[str]:
        """Salient query terms (names, acronyms, numbers) that appear in NONE of the top passages.
        If the question is about 'Jamuna Bridge' and only 'Padma Bridge' was retrieved, the
        evidence cannot answer it, however fluent an answer the LLM could write."""
        if not self.entity_check or not passages:
            return []
        text = " ".join(f"{p.chunk.title} {p.chunk.text}" for p in passages[:top])
        have = set(tokenize(text)) | set(re.findall(r"[a-z0-9]+", text.lower()))
        missing = []
        for term in salient_terms(query):
            toks = tokenize(term) or re.findall(r"[a-z0-9]+", term.lower())
            if toks and not all(t in have for t in toks):
                missing.append(term)
        # multi-word names must co-occur closely ("Padma ... Bridge" yes, "Jamuna ... Bridge" no)
        words = [re.findall(r"[a-z0-9]+", f"{p.chunk.title} {p.chunk.text}".lower()) for p in passages[:top]]
        for phrase in salient_phrases(query):
            parts = [w.lower() for w in phrase]
            if any(w in missing for w in phrase):
                continue
            if not any(_near(ws, parts, len(parts) + 2) for ws in words):
                missing.append(" ".join(phrase))
        return missing

    def assess(self, passages: List[Passage], query: str = "") -> str:
        if not passages:
            return INCORRECT
        if query and self.missing_entities(query, passages):
            return INCORRECT
        top = max(p.score for p in passages)
        if top >= self.upper:
            return CORRECT
        if top < self.lower:
            return INCORRECT
        return AMBIGUOUS

    def filter(self, passages: List[Passage]) -> List[Passage]:
        return [p for p in passages if p.score >= self.keep_min]


_QWORDS = {"what", "which", "who", "whom", "whose", "how", "when", "where", "why", "is", "are", "was", "were",
           "does", "do", "did", "in", "on", "the", "a", "an", "can", "could", "should", "would", "will", "name",
           "list", "describe", "explain", "tell", "give", "compare", "according"}


def salient_terms(query: str) -> List[str]:
    """Capitalised words / acronyms / tokens with digits, split on hyphens ('English-to-Japanese')."""
    out = []
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.,]*[A-Za-z0-9]|[A-Za-z0-9]", query):
        for part in re.split(r"-", raw):
            part = part.strip(".,")
            if not part or part.lower() in _QWORDS:
                continue
            if part[0].isupper() or any(ch.isdigit() for ch in part):
                out.append(part)
    return list(dict.fromkeys(out))


def salient_phrases(query: str) -> List[List[str]]:
    """Runs of >= 2 consecutive capitalised words, e.g. ['Jamuna', 'Bridge'], ['Mount', 'Everest']."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", query)
    out, run = [], []
    for i, w in enumerate(words):
        if w[0].isupper() and not (i == 0 and w.lower() in _QWORDS) and w.lower() not in _QWORDS:
            run.append(w)
        else:
            if len(run) >= 2:
                out.append(run)
            run = []
    if len(run) >= 2:
        out.append(run)
    return out


def _near(words: List[str], parts: List[str], span: int) -> bool:
    first = [i for i, w in enumerate(words) if w == parts[0]]
    for i in first:
        window = words[i:i + span]
        if all(p in window for p in parts[1:]):
            return True
    return False


def prf_expand(query: str, passages: List[Passage], n_docs: int = 3, n_terms: int = 5) -> str:
    """Pseudo-relevance feedback: add frequent informative terms from top passages."""
    q = set(tokenize(query))
    counts: Counter = Counter()
    for p in passages[:n_docs]:
        counts.update(t for t in tokenize(p.chunk.text) if t not in q and len(t) > 3 and not t.isdigit())
    extra = [t for t, _ in counts.most_common(n_terms)]
    return query + (" " + " ".join(extra) if extra else "")


REWRITE_SYSTEM = (
    "You rewrite search queries. Return ONE improved search query that keeps every entity, "
    "number and constraint of the original and adds useful synonyms. Output only the query."
)


def rewrite_query(query: str, passages: List[Passage], llm=None) -> str:
    if llm is not None and getattr(llm, "is_generative", False):
        try:
            out = llm.complete(REWRITE_SYSTEM, f"Original query: {query}", max_tokens=60).strip().strip('"')
            if out:
                return out.splitlines()[0]
        except Exception:
            pass
    return prf_expand(query, passages)
