"""Shared data structures used by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A source document before chunking."""

    doc_id: str
    title: str
    text: str
    source: str = "local"
    url: str = ""
    year: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit (a few sentences of one document)."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int = 0
    source: str = "local"
    url: str = ""
    year: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Chunk":
        return Chunk(**d)


@dataclass
class Passage:
    """A retrieved chunk plus its scores and its citation label (S1, S2...)."""

    chunk: Chunk
    score: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    label: str = ""

    @property
    def text(self) -> str:
        return self.chunk.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "text": self.chunk.text,
            "source": self.chunk.source,
            "url": self.chunk.url,
            "year": self.chunk.year,
            "score": round(float(self.score), 4),
            "scores": {k: round(float(v), 4) for k, v in self.scores.items()},
        }


@dataclass
class Claim:
    """One sentence of a generated answer and the citations attached to it."""

    claim_id: int
    text: str                      # sentence without citation markers
    citations: List[str] = field(default_factory=list)   # e.g. ["S1", "S3"]
    ghost_citations: List[str] = field(default_factory=list)  # cited labels that do not exist


@dataclass
class ClaimVerdict:
    """Verification + correction outcome for one claim."""

    claim: Claim
    label: str = "unsupported"     # supported | misattributed | unsupported | contradicted
    entailment: float = 0.0        # best entailment using the CITED sources
    contradiction: float = 0.0
    best_label: str = ""           # best supporting passage among ALL retrieved
    best_entailment: float = 0.0
    evidence: str = ""             # evidence window that decided the verdict
    numeric_mismatch: bool = False
    action: str = "kept"           # kept | recited | pruned_citations | rewritten | removed
    final_text: str = ""
    final_citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim.claim_id,
            "text": self.claim.text,
            "citations": self.claim.citations,
            "ghost_citations": self.claim.ghost_citations,
            "label": self.label,
            "entailment": round(self.entailment, 4),
            "contradiction": round(self.contradiction, 4),
            "best_label": self.best_label,
            "best_entailment": round(self.best_entailment, 4),
            "evidence": self.evidence,
            "numeric_mismatch": self.numeric_mismatch,
            "action": self.action,
            "final_text": self.final_text,
            "final_citations": self.final_citations,
        }


@dataclass
class PipelineResult:
    query: str
    answer: str
    status: str                    # answered | partial | abstained
    original_answer: str = ""
    passages: List[Passage] = field(default_factory=list)
    verdicts: List[ClaimVerdict] = field(default_factory=list)
    retrieval_decision: str = ""   # correct | ambiguous | incorrect
    retrieval_actions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    system: str = "trustrag"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system,
            "query": self.query,
            "answer": self.answer,
            "status": self.status,
            "original_answer": self.original_answer,
            "retrieval_decision": self.retrieval_decision,
            "retrieval_actions": self.retrieval_actions,
            "confidence": round(self.confidence, 4),
            "passages": [p.to_dict() for p in self.passages],
            "claims": [v.to_dict() for v in self.verdicts],
            "timings": {k: round(v, 4) for k, v in self.timings.items()},
            "stats": self.stats,
        }
