"""Benchmark runner: retrieval comparison, end-to-end system comparison
(naive baseline vs TrustRAG vs ablations) and the controlled fault-injection test.
Writes summary.json, per_question.csv, charts and a self-contained report.html.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import resolve_path
from ..index.store import CorpusIndex
from ..ingest.chunker import chunk_documents
from ..pipeline import TrustRAGPipeline
from ..schemas import Passage
from ..verification.nli import build_nli
from ..verification.verifier import Verifier
from .datasets import EvalExample, load_benchmark
from .metrics import (answer_relevance, citation_metrics, contains_answer, retrieval_metrics, rouge_l,
                      summarize, token_f1)

RETRIEVERS = {
    "bm25": {"retrieval": {"use_bm25": True, "use_dense": False}, "reranker": {"enabled": False}},
    "dense": {"retrieval": {"use_bm25": False, "use_dense": True}, "reranker": {"enabled": False}},
    "hybrid_rrf": {"retrieval": {"use_bm25": True, "use_dense": True}, "reranker": {"enabled": False}},
    "hybrid_rrf+rerank": {"retrieval": {"use_bm25": True, "use_dense": True}, "reranker": {"enabled": True}},
}


def build_judge(cfg: dict, pipeline_nli) -> Verifier:
    ev = cfg["evaluation"]
    backend = ev.get("judge_nli_backend", "same")
    if backend == "same":
        nli = pipeline_nli
    else:
        nli = build_nli(backend, ev.get("judge_nli_model", ""), cfg["nli"].get("device", "cpu"))
    return Verifier(nli, cfg)


def prepare_index(cfg: dict, docs, name: str, rebuild: bool = False) -> CorpusIndex:
    idx_dir = resolve_path(f"indexes/{name}_{cfg['embeddings']['backend']}")
    if not rebuild and (idx_dir / "chunks.jsonl").exists():
        return CorpusIndex.load(idx_dir, cfg)
    chunks = chunk_documents(docs, cfg["chunking"]["max_words"], cfg["chunking"]["overlap_sentences"])
    return CorpusIndex.build(chunks, cfg, idx_dir)


# ------------------------------------------------------------------ retrieval benchmark
def retrieval_benchmark(base: TrustRAGPipeline, examples: List[EvalExample], ks=(1, 3, 5)) -> Dict[str, dict]:
    from ..config import deep_merge
    from ..retrieval.hybrid import HybridRetriever
    from ..retrieval.reranker import Reranker

    out = {}
    exs = [e for e in examples if e.answerable and e.gold_doc_ids]
    for name, ov in RETRIEVERS.items():
        cfg = deep_merge(base.base_cfg, ov)
        retr, rer = HybridRetriever(base.index, cfg), Reranker(base.index, cfg)
        rows, lat = [], []
        for e in exs:
            t = time.perf_counter()
            ranked = rer.rerank(e.question, retr.search(e.question))
            lat.append(time.perf_counter() - t)
            rows.append(retrieval_metrics([p.chunk.doc_id for p in ranked], e.gold_doc_ids, ks))
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}
        agg["latency_ms"] = 1000 * float(np.mean(lat)) if lat else 0.0
        out[name] = agg
    return out


# ------------------------------------------------------------------ end-to-end evaluation
def evaluate_systems(base: TrustRAGPipeline, examples: List[EvalExample], systems: List[str], judge: Verifier,
                     ks=(1, 3, 5), progress: bool = True):
    per_q: List[dict] = []
    summary: Dict[str, dict] = {}
    for sysname in systems:
        pipe = base if sysname == base.system else base.with_system(sysname)
        rows = []
        for i, e in enumerate(examples):
            if progress:
                print(f"  [{sysname}] {i + 1}/{len(examples)}: {e.question[:70]}", flush=True)
            res = pipe.run(e.question)
            row = {"system": sysname, "qid": e.qid, "question": e.question, "answerable": e.answerable,
                   "status": res.status, "answer": res.answer, "retrieval_decision": res.retrieval_decision,
                   "latency_s": res.timings.get("total", 0.0)}
            for st in ("retrieve", "rerank", "gate", "generate", "verify", "correct"):
                row[f"t_{st}"] = res.timings.get(st, 0.0)
            abstained = res.status == "abstained"
            row["abstained"] = abstained
            if e.answerable and e.gold_doc_ids:
                row.update(retrieval_metrics([p.chunk.doc_id for p in res.passages], e.gold_doc_ids, ks))
            if not abstained:
                row.update(citation_metrics(res.answer, res.passages, judge))
                row["answer_relevance"] = answer_relevance(e.question, res.answer, base.index.embedder)
                if e.answers:
                    row["answer_contains"] = contains_answer(res.answer, e.answers)
                if e.reference:
                    row["token_f1"] = token_f1(res.answer, e.reference)
                    row["rouge_l"] = rouge_l(res.answer, e.reference)
            row["stats"] = json.dumps(res.stats)
            rows.append(row)
        per_q.extend(rows)
        summary[sysname] = aggregate(rows)
    return summary, per_q


METRIC_KEYS = ["faithfulness", "hallucination_rate", "citation_recall", "citation_precision", "citation_f1",
               "ghost_citation_rate", "uncited_claim_rate", "answer_relevance", "answer_contains", "token_f1",
               "rouge_l", "recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]


def aggregate(rows: List[dict]) -> dict:
    out: Dict[str, float] = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows if k in r and r[k] is not None and r.get("answerable", True)]
        if vals:
            out[k] = float(np.mean(vals))
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    out["n_questions"] = len(rows)
    out["abstention_rate_answerable"] = float(np.mean([r["abstained"] for r in ans])) if ans else 0.0
    if unans:
        out["correct_abstention_unanswerable"] = float(np.mean([r["abstained"] for r in unans]))
    lat = summarize([r["latency_s"] for r in rows])
    out["latency_mean_s"], out["latency_p90_s"] = lat["mean"], lat["p90"]
    for st in ("retrieve", "rerank", "gate", "generate", "verify", "correct"):
        out[f"t_{st}_ms"] = 1000 * float(np.mean([r.get(f"t_{st}", 0.0) for r in rows]))
    return out


# ------------------------------------------------------------------ orchestration
def run_benchmark(cfg: dict, dataset: str = "demo", systems: Optional[List[str]] = None, limit: int = 100,
                  out_dir: str = "results/latest", subset: str = "pubmedqa", qa: str = "", corpus: str = "",
                  path: str = "", faults: bool = True, rebuild_index: bool = False, fault_seeds: int = 5) -> dict:
    systems = systems or ["naive", "trustrag", "no_correction", "no_gate", "no_rerank"]
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"Loading benchmark '{dataset}' ...")
    docs, examples = load_benchmark(dataset, limit=limit, subset=subset, qa=qa, corpus=corpus, path=path)
    name = dataset if dataset != "ragbench" else f"ragbench_{subset}_{limit}"
    if docs is not None:
        index = prepare_index(cfg, docs, name, rebuild_index)
    else:
        index = CorpusIndex.load(resolve_path(cfg["index_dir"]), cfg)
    print(f"Index: {len(index)} chunks from {len({c.doc_id for c in index.chunks})} documents; "
          f"{len(examples)} questions")
    base = TrustRAGPipeline(cfg, index=index, log=False)
    judge = build_judge(cfg, base.nli)
    ks = tuple(cfg["evaluation"].get("k_values", [1, 3, 5]))

    print("Retrieval comparison ...")
    retr = retrieval_benchmark(base, examples, ks)
    print("End-to-end systems ...")
    summary, per_q = evaluate_systems(base, examples, systems, judge, ks)
    fault = None
    if faults:
        from .faults import run_fault_injection, summarize_faults

        print(f"Fault-injection test ({fault_seeds} seeds) ...")
        records = []
        for sd in range(1, fault_seeds + 1):
            records += run_fault_injection(base, examples, judge, seed=sd)["records"]
        fault = summarize_faults(records)
        fault["seeds"] = fault_seeds
        with open(out / "fault_records.csv", "w", newline="", encoding="utf-8") as fh:
            if records:
                w = csv.DictWriter(fh, fieldnames=list(records[0]))
                w.writeheader()
                w.writerows(records)
        fault.pop("records", None)

    result = {
        "dataset": dataset if dataset != "ragbench" else f"ragbench/{subset}",
        "n_questions": len(examples),
        "n_unanswerable": sum(not e.answerable for e in examples),
        "config": {
            "mode": cfg.get("mode"), "embeddings": cfg["embeddings"]["backend"] + ":" + str(
                cfg["embeddings"].get("model") if cfg["embeddings"]["backend"] != "lsa" else "tfidf+svd"),
            "reranker": cfg["reranker"]["backend"] + (":" + cfg["reranker"]["model"]
                                                      if cfg["reranker"]["backend"] != "lexical" else ""),
            "llm": base.llm.name, "nli": base.nli.name, "judge_nli": judge.nli.name,
        },
        "retrieval": retr,
        "systems": summary,
        "fault_injection": fault,
        "runtime_s": time.time() - t0,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    keys = sorted({k for r in per_q for k in r})
    with open(out / "per_question.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(per_q)
    from .report import write_report

    write_report(result, out)
    print(f"\nDone in {result['runtime_s']:.1f}s -> {out}/report.html")
    return result
