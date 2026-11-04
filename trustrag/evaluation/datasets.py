"""Benchmark loaders. Every loader returns (documents, examples).

For RAGBench and HotpotQA, the per-question documents of all sampled questions are
POOLED into one corpus, so the retriever has to find the right documents among
hundreds of distractors (a realistic test of retrieval, unlike oracle contexts).

    demo       built-in corpus (data/demo) with gold docs + unanswerable questions
    jsonl      your own: --qa questions.jsonl --corpus folder_or_jsonl
    ragbench   galileo-ai/ragbench (subset: pubmedqa, covidqa, hotpotqa, techqa, ...)
    hotpotqa   hotpot_qa distractor (multi-hop; gold = supporting-fact titles)
    asqa       ALCE ASQA json file (e.g. asqa_eval_gtr_top100.json), docs pooled
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import ROOT
from ..ingest.loaders import load_path
from ..schemas import Document


@dataclass
class EvalExample:
    qid: str
    question: str
    answers: List[str] = field(default_factory=list)   # short gold answers (any match counts)
    reference: str = ""                                # long-form reference answer
    gold_doc_ids: List[str] = field(default_factory=list)
    answerable: bool = True


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _examples_from_rows(rows: List[dict]) -> List[EvalExample]:
    out = []
    for i, r in enumerate(rows):
        ans = r.get("answers") or ([r["answer"]] if r.get("answer") else [])
        out.append(EvalExample(
            qid=str(r.get("id", i)), question=r["question"], answers=list(ans),
            reference=r.get("reference") or (ans[0] if ans else ""),
            gold_doc_ids=list(r.get("gold_doc_ids", [])), answerable=bool(r.get("answerable", True))))
    return out


def load_demo() -> Tuple[List[Document], List[EvalExample]]:
    base = ROOT / "data" / "demo"
    return load_path(str(base / "corpus")), _examples_from_rows(_read_jsonl(base / "qa.jsonl"))


def load_jsonl_benchmark(qa_path: str, corpus_path: Optional[str]) -> Tuple[Optional[List[Document]], List[EvalExample]]:
    docs = load_path(corpus_path) if corpus_path else None
    return docs, _examples_from_rows(_read_jsonl(Path(qa_path)))


def load_ragbench(subset: str = "pubmedqa", split: str = "test", limit: int = 100):
    from datasets import load_dataset

    ds = load_dataset("galileo-ai/ragbench", subset, split=split)
    docs, exs = [], []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        qid = str(row.get("id", i))
        gold = []
        for j, text in enumerate(row["documents"]):
            did = f"{qid}_d{j}"
            gold.append(did)
            docs.append(Document(doc_id=did, title="", text=text, source=f"ragbench/{subset}"))
        ref = row.get("response", "")
        exs.append(EvalExample(qid=qid, question=row["question"], answers=[], reference=ref, gold_doc_ids=gold))
    return docs, exs


def load_hotpotqa(limit: int = 100, split: str = "validation"):
    from datasets import load_dataset

    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    except Exception:
        ds = load_dataset("hotpot_qa", "distractor", split=split)
    docs, seen, exs = [], set(), []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            did = title
            if did not in seen:
                seen.add(did)
                docs.append(Document(doc_id=did, title=title, text=" ".join(sents), source="wikipedia"))
        gold = list(dict.fromkeys(row["supporting_facts"]["title"]))
        exs.append(EvalExample(qid=row["id"], question=row["question"], answers=[row["answer"]],
                               reference=row["answer"], gold_doc_ids=gold))
    return docs, exs


def load_asqa(path: str, limit: int = 100, docs_per_question: int = 10):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = list(data.values())
    docs, seen, exs = [], set(), []
    for i, row in enumerate(data[:limit]):
        gold = []
        for j, d in enumerate(row.get("docs", [])[:docs_per_question]):
            did = str(d.get("id") or f"{i}_{j}")
            if did not in seen:
                seen.add(did)
                docs.append(Document(doc_id=did, title=d.get("title", ""), text=d.get("text", ""), source="wikipedia"))
        short = []
        for qa in row.get("qa_pairs", []):
            short.extend(qa.get("short_answers", []))
        exs.append(EvalExample(qid=str(row.get("sample_id", i)), question=row["question"], answers=short,
                               reference=row.get("answer", ""), gold_doc_ids=gold))
    return docs, exs


def load_benchmark(name: str, limit: int = 100, subset: str = "pubmedqa", qa: str = "", corpus: str = "",
                   path: str = ""):
    if name == "demo":
        docs, exs = load_demo()
        return docs, exs[:limit]
    if name == "jsonl":
        docs, exs = load_jsonl_benchmark(qa, corpus or None)
        return docs, exs[:limit]
    if name == "ragbench":
        return load_ragbench(subset, limit=limit)
    if name == "hotpotqa":
        return load_hotpotqa(limit)
    if name == "asqa":
        return load_asqa(path, limit)
    raise ValueError(f"Unknown benchmark {name}")
