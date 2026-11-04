"""Load documents from a folder (.txt/.md/.pdf/.docx) or a .jsonl file.

Markdown/text files may start with a small front-matter block:

    ---
    title: My title
    source: Wikipedia
    year: 2023
    url: https://...
    ---
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import yaml

from ..schemas import Document

_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def _slug(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", path.stem).strip("_").lower()


def _read_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF

        with fitz.open(path) as pdf:
            return "\n".join(page.get_text() for page in pdf)
    except ImportError:
        try:
            from pypdf import PdfReader

            return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install pymupdf or pypdf to ingest PDFs") from exc


def _read_docx(path: Path) -> str:
    from docx import Document as Docx  # python-docx

    return "\n".join(p.text for p in Docx(str(path)).paragraphs)


def load_file(path: Path) -> Document:
    suffix = path.suffix.lower()
    meta = {}
    if suffix in (".txt", ".md"):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        m = _FRONT.match(raw)
        if m:
            meta = yaml.safe_load(m.group(1)) or {}
            raw = raw[m.end():]
        # strip markdown headings/bullets but keep their text
        text = re.sub(r"^#+\s*", "", raw, flags=re.M)
        text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M)
    elif suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return Document(
        doc_id=str(meta.get("id") or _slug(path)),
        title=str(meta.get("title") or path.stem.replace("_", " ")),
        text=text,
        source=str(meta.get("source") or "local"),
        url=str(meta.get("url") or ""),
        year=meta.get("year"),
    )


def load_jsonl(path: Path) -> List[Document]:
    docs = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            docs.append(
                Document(
                    doc_id=str(d.get("doc_id") or d.get("id") or f"doc{i}"),
                    title=str(d.get("title") or ""),
                    text=str(d.get("text") or d.get("abstract") or ""),
                    source=str(d.get("source") or "jsonl"),
                    url=str(d.get("url") or ""),
                    year=d.get("year"),
                )
            )
    return docs


def load_path(path: str) -> List[Document]:
    p = Path(path)
    if p.is_file():
        return load_jsonl(p) if p.suffix == ".jsonl" else [load_file(p)]
    docs = []
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".txt", ".md", ".pdf", ".docx"):
            docs.append(load_file(f))
        elif f.is_file() and f.suffix == ".jsonl":
            docs.extend(load_jsonl(f))
    return docs
