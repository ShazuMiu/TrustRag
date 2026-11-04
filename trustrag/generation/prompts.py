"""Prompts. The generation prompt enforces sentence-level citations and gives the
model an explicit way to abstain — two of the cheapest, most effective
hallucination controls ("active grounding" in the review)."""

from __future__ import annotations

from typing import List

from ..schemas import Passage

GENERATION_SYSTEM = """You are TrustRAG, a careful research assistant.
Answer ONLY with information stated in the numbered SOURCES. Rules:
1. Every sentence must end with the citation(s) of the source(s) that directly support it, e.g. "... [S2]." or "... [S1][S3]."
2. Cite a source only if it explicitly contains the information in that sentence. Never cite a label that is not listed.
3. Do not use outside knowledge. Do not guess numbers, names or dates; copy them exactly from the sources.
4. Keep the answer concise: at most {max_sentences} sentences, one fact per sentence.
5. If the sources do not contain the answer, reply exactly: INSUFFICIENT_EVIDENCE"""

NAIVE_SYSTEM = """You are a helpful assistant. Use the context to answer the question and cite sources like [S1]."""

REWRITE_CLAIM_SYSTEM = """You fix one unsupported sentence of an answer.
Rewrite the SENTENCE so that it states only what the EVIDENCE explicitly says, keeping it relevant to the QUESTION.
End it with the citation label(s) given. If the evidence says nothing relevant, reply exactly: DROP"""


def format_sources(passages: List[Passage]) -> str:
    blocks = []
    for p in passages:
        meta = ", ".join(str(x) for x in (p.chunk.year, p.chunk.source) if x)
        head = f"[{p.label}] {p.chunk.title}" + (f" ({meta})" if meta else "")
        blocks.append(f"{head}\n{p.chunk.text}")
    return "\n\n".join(blocks)


def generation_user(question: str, passages: List[Passage]) -> str:
    return f"SOURCES:\n\n{format_sources(passages)}\n\nQUESTION: {question}\n\nANSWER:"


def rewrite_claim_user(question: str, sentence: str, evidence: List[Passage]) -> str:
    labels = "".join(f"[{p.label}]" for p in evidence)
    ev = "\n".join(f"[{p.label}] {p.chunk.text}" for p in evidence)
    return f"QUESTION: {question}\nSENTENCE: {sentence}\nEVIDENCE:\n{ev}\nCITATION LABELS TO USE: {labels}\nREWRITTEN SENTENCE:"
