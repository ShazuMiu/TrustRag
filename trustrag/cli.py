"""Command line interface.

  python -m trustrag.cli ingest   --config config/lite.yaml --input data/demo/corpus
  python -m trustrag.cli ask      --config config/lite.yaml "How long is the Padma Bridge?"
  python -m trustrag.cli eval     --config config/lite.yaml --dataset demo
  python -m trustrag.cli serve    --config config/lite.yaml --port 8000
  python -m trustrag.cli logs     --config config/lite.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

from .config import load_config, resolve_path


def cmd_ingest(args, cfg):
    from .index.store import CorpusIndex
    from .ingest.chunker import chunk_documents
    from .ingest.loaders import load_path

    docs = load_path(args.input)
    if not docs:
        sys.exit(f"No documents found in {args.input}")
    chunks = chunk_documents(docs, cfg["chunking"]["max_words"], cfg["chunking"]["overlap_sentences"])
    out = resolve_path(args.index or cfg["index_dir"])
    CorpusIndex.build(chunks, cfg, out)
    print(f"Indexed {len(docs)} documents -> {len(chunks)} chunks at {out}")


def cmd_ask(args, cfg):
    from .pipeline import TrustRAGPipeline

    pipe = TrustRAGPipeline(cfg, system=args.system)
    res = pipe.run(args.question)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
        return
    w = textwrap.TextWrapper(width=100)
    print("\n" + "=" * 100)
    print(f"Q: {res.query}\nStatus: {res.status}   retrieval: {res.retrieval_decision}   "
          f"confidence: {res.confidence:.2f}   time: {res.timings.get('total', 0):.2f}s")
    print("-" * 100)
    print(w.fill(res.answer))
    print("-" * 100 + "\nSources:")
    for p in res.passages:
        print(f"  [{p.label}] {p.chunk.title} ({p.chunk.doc_id})  rel={p.score:.2f}")
    if res.verdicts:
        print("Claim checks:")
        for v in res.verdicts:
            print(f"  - {v.label:<13} -> {v.action:<16} cited={v.claim.citations} final={v.final_citations} "
                  f"entail={v.entailment:.2f} | {v.claim.text[:70]}")
    if res.retrieval_actions:
        print("Retrieval actions:", "; ".join(res.retrieval_actions))


def cmd_eval(args, cfg):
    from .evaluation.runner import run_benchmark

    systems = args.systems.split(",") if args.systems else None
    out = args.out or f"results/{args.dataset}_{cfg.get('mode', 'run')}"
    res = run_benchmark(cfg, dataset=args.dataset, systems=systems, limit=args.limit, out_dir=out,
                        subset=args.subset, qa=args.qa, corpus=args.corpus, path=args.path,
                        faults=not args.no_faults, rebuild_index=args.rebuild, fault_seeds=args.fault_seeds)
    print("\nKey results:")
    for s, m in res["systems"].items():
        print(f"  {s:<14} faithfulness={m.get('faithfulness', float('nan')):.3f}  "
              f"cit_P={m.get('citation_precision', float('nan')):.3f}  "
              f"cit_R={m.get('citation_recall', float('nan')):.3f}  "
              f"latency={m.get('latency_mean_s', 0):.2f}s")


def cmd_serve(args, cfg):
    from app.server import create_app

    app = create_app(cfg)
    app.run(host=args.host, port=args.port, debug=False)


def cmd_logs(args, cfg):
    from .logging_db import RunLogger

    for r in RunLogger(resolve_path(cfg["logging"]["sqlite_path"])).recent(args.n):
        print(r)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trustrag", description="TrustRAG v2")
    ap.add_argument("--config", default="config/lite.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="chunk + index documents")
    p.add_argument("--input", required=True, help="folder (.md/.txt/.pdf/.docx) or .jsonl")
    p.add_argument("--index", default="", help="index directory (default: index_dir from config)")

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("--system", default="trustrag")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("eval", help="run the benchmark and write a report")
    p.add_argument("--dataset", default="demo", choices=["demo", "jsonl", "ragbench", "hotpotqa", "asqa"])
    p.add_argument("--subset", default="pubmedqa", help="RAGBench subset")
    p.add_argument("--qa", default="", help="jsonl questions (dataset=jsonl)")
    p.add_argument("--corpus", default="", help="corpus folder/jsonl (dataset=jsonl)")
    p.add_argument("--path", default="", help="ALCE ASQA json file (dataset=asqa)")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--systems", default="", help="comma list, e.g. naive,trustrag,no_correction")
    p.add_argument("--out", default="")
    p.add_argument("--no-faults", action="store_true")
    p.add_argument("--fault-seeds", type=int, default=5, help="random seeds pooled in the fault-injection test")
    p.add_argument("--rebuild", action="store_true", help="rebuild the benchmark index")

    p = sub.add_parser("serve", help="start the web UI + REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("logs", help="show recent logged runs")
    p.add_argument("-n", type=int, default=20)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval, "serve": cmd_serve, "logs": cmd_logs}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
