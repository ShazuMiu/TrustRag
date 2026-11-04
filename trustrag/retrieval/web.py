"""Optional external fallback retrieval (disabled by default).

Privacy (NFR4): only the user's query string is sent to the external service;
local documents are never uploaded.
"""

from __future__ import annotations

import os
from typing import List

from ..schemas import Chunk, Passage


def semantic_scholar_search(query: str, limit: int = 5) -> List[Passage]:
    import requests

    headers = {}
    if os.getenv("S2_API_KEY"):
        headers["x-api-key"] = os.environ["S2_API_KEY"]
    r = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": limit, "fields": "title,abstract,year,url"},
        headers=headers,
        timeout=20,
    )
    r.raise_for_status()
    out = []
    for i, paper in enumerate(r.json().get("data", [])):
        if not paper.get("abstract"):
            continue
        out.append(Passage(chunk=Chunk(
            chunk_id=f"s2::{paper.get('paperId', i)}", doc_id=f"s2::{paper.get('paperId', i)}",
            title=paper.get("title") or "", text=paper["abstract"], source="semantic_scholar",
            url=paper.get("url") or "", year=paper.get("year"))))
    return out


def duckduckgo_search(query: str, limit: int = 5) -> List[Passage]:
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS
    out = []
    with DDGS() as ddg:
        for i, hit in enumerate(ddg.text(query, max_results=limit)):
            out.append(Passage(chunk=Chunk(
                chunk_id=f"web::{i}::{hit.get('href', '')}", doc_id=f"web::{hit.get('href', i)}",
                title=hit.get("title", ""), text=hit.get("body", ""), source="web", url=hit.get("href", ""))))
    return out


def web_search(query: str, provider: str = "semantic_scholar", limit: int = 5) -> List[Passage]:
    if provider == "duckduckgo":
        return duckduckgo_search(query, limit)
    return semantic_scholar_search(query, limit)
