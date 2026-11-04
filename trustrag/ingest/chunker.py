"""Sentence-aware chunking with overlap and stable chunk IDs.

Chunks never cut a sentence in half (a common cause of citation mismatch:
the claim's evidence ends up split across two chunks).
"""

from __future__ import annotations

from typing import List

from ..schemas import Chunk, Document
from ..text import split_sentences


def chunk_document(doc: Document, max_words: int = 120, overlap_sentences: int = 1) -> List[Chunk]:
    sentences = split_sentences(doc.text)
    chunks: List[Chunk] = []
    if not sentences:
        return chunks

    current: List[str] = []
    count = 0
    position = 0

    def flush():
        nonlocal position
        if current:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::c{position}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=" ".join(current),
                    position=position,
                    source=doc.source,
                    url=doc.url,
                    year=doc.year,
                    metadata=dict(doc.metadata),
                )
            )
            position += 1

    fresh = False  # True when `current` holds sentences not yet emitted
    for sent in sentences:
        n = len(sent.split())
        if current and fresh and count + n > max_words:
            flush()
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            count = sum(len(s.split()) for s in current)
        current.append(sent)
        count += n
        fresh = True
    if fresh:  # never emit a chunk that is only the overlap of the previous one
        flush()
    return chunks


def chunk_documents(docs: List[Document], max_words: int = 120, overlap_sentences: int = 1) -> List[Chunk]:
    out: List[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, max_words, overlap_sentences))
    return out
