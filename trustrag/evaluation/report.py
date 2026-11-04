"""Charts (PNG) + a self-contained HTML report you can open or present."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (fixed order); the baseline is a neutral gray.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948"]
BASELINE = "#a3a29c"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(groups: List[str], series: Dict[str, List[float]], title: str, ylabel: str = "",
                 ylim=(0, 1.05)) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=130)
    n = len(series)
    width = 0.8 / max(1, n)
    colors, j = {}, 0
    for name in series:
        if name in ("naive", "bm25"):
            colors[name] = BASELINE
        else:
            colors[name] = SERIES[j % len(SERIES)]
            j += 1
    for i, (name, vals) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, vals, width * 0.92, label=name, color=colors[name],
                      edgecolor="white", linewidth=1)
        if n <= 3:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=9, color=INK)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=min(n, 6), loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    return _png(fig)


def stacked_latency(summary: Dict[str, dict]) -> str:
    stages = ["retrieve", "rerank", "gate", "generate", "verify", "correct"]
    systems = list(summary)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(systems) + 1.2), dpi=130)
    left = [0.0] * len(systems)
    for i, st in enumerate(stages):
        vals = [summary[s].get(f"t_{st}_ms", 0.0) for s in systems]
        ax.barh(systems, vals, left=left, color=SERIES[i % len(SERIES)], label=st, edgecolor="white", linewidth=1.5)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("mean latency per question (ms)", color=INK2, fontsize=9)
    ax.set_title("Where the time goes", loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.invert_yaxis()
    _style(ax)
    ax.xaxis.grid(True, color=GRID)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=8, ncol=6, loc="upper left", bbox_to_anchor=(0, -0.25))
    fig.tight_layout()
    return _png(fig)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _table(rows: List[List[str]], header: List[str]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _f(v, pct=False):
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{100 * v:.1f}%" if pct else f"{v:.3f}"


SYSTEM_METRICS = [("faithfulness", "Faithfulness ↑"), ("hallucination_rate", "Hallucination rate ↓"),
                  ("citation_precision", "Citation precision ↑"), ("citation_recall", "Citation recall ↑"),
                  ("citation_f1", "Citation F1 ↑"), ("ghost_citation_rate", "Ghost citations ↓"),
                  ("answer_relevance", "Answer relevance ↑"), ("answer_contains", "Answer EM (contains) ↑"),
                  ("token_f1", "Token F1 ↑"), ("rouge_l", "ROUGE-L ↑"), ("recall@5", "Context Recall@5 ↑"),
                  ("abstention_rate_answerable", "False abstention ↓"),
                  ("correct_abstention_unanswerable", "Correct abstention (unanswerable) ↑"),
                  ("latency_mean_s", "Latency mean (s) ↓"), ("latency_p90_s", "Latency p90 (s) ↓")]


def write_report(result: dict, out: Path) -> Path:
    out = Path(out)
    systems = result["systems"]
    imgs = {}
    key = ["faithfulness", "citation_precision", "citation_recall", "citation_f1", "answer_relevance"]
    imgs["systems"] = grouped_bars([k.replace("_", " ") for k in key],
                                   {s: [systems[s].get(k, 0.0) for k in key] for s in systems},
                                   "End-to-end quality by system (higher is better)")
    r = result["retrieval"]
    rk = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]
    if r and any(r.values()):
        imgs["retrieval"] = grouped_bars(rk, {n: [r[n].get(k, 0.0) for k in rk] for n in r},
                                         "Retrieval quality by retriever")
    imgs["latency"] = stacked_latency(systems)
    fi = result.get("fault_injection") or {}
    if fi.get("by_type"):
        types = [t for t in fi["by_type"] if t != "clean"]
        imgs["faults"] = grouped_bars(types, {
            "detected": [fi["by_type"][t].get("detected", 0) for t in types],
            "fixed": [fi["by_type"][t].get("fixed", 0) for t in types]},
            "Fault injection: injected errors caught and fixed by TrustRAG")
    for k, b64 in imgs.items():
        (out / f"chart_{k}.png").write_bytes(base64.b64decode(b64))

    sys_rows = [[html.escape(label)] + [_f(systems[s].get(k)) for s in systems] for k, label in SYSTEM_METRICS]
    sys_table = _table(sys_rows, ["Metric"] + list(systems))
    retr_rows = [[html.escape(n)] + [_f(r[n].get(k)) for k in rk] + [f"{r[n].get('latency_ms', 0):.1f}"] for n in r]
    retr_table = _table(retr_rows, ["Retriever"] + rk + ["ms/query"])
    fault_html = ""
    if fi.get("by_type"):
        rows = [[t, str(d["n"]), _f(d.get("detected"), True), _f(d.get("fixed"), True),
                 _f(d.get("false_removal"), True) if t == "clean" else "–"] for t, d in fi["by_type"].items()]
        n, tr = fi["naive"], fi["trustrag"]
        fault_html = f"""
<h2>Controlled fault injection</h2>
<p>Known errors were injected into verified answers (ground truth is known), then passed through each system.
{fi['n_claims']} claims pooled over {fi.get('seeds', 1)} random seed(s). Per-claim records: <code>fault_records.csv</code>.</p>
<img src="data:image/png;base64,{imgs['faults']}">
{_table(rows, ["Injected error", "claims", "detected", "fixed", "clean claims falsely removed"])}
{_table([["Hallucinated claims in delivered answer", _f(n['hallucinated_claims_in_answer'], True), _f(tr['hallucinated_claims_in_answer'], True)],
         ["Delivered claims with a correct citation", _f(n['claims_with_correct_citation'], True), _f(tr['claims_with_correct_citation'], True)],
         ["Error detection recall", "0.0%", _f(tr['error_detection_recall'], True)]],
        ["Fault-injection outcome", "naive RAG", "TrustRAG"])}"""

    cfg = result["config"]
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>TrustRAG Benchmark Report</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e0;--accent:#2a78d6}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:32px 16px 64px}} h1{{font-size:26px;margin:0 0 4px}}
h2{{font-size:19px;margin:36px 0 10px;border-top:1px solid var(--line);padding-top:24px}}
.meta{{color:var(--ink2);font-size:13px}} img{{max-width:100%;height:auto;margin:8px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{color:var(--ink2);font-weight:600}}
code{{background:#f1f0ec;padding:1px 5px;border-radius:4px}}
</style></head><body><main>
<h1>TrustRAG benchmark report</h1>
<div class="meta">Dataset <b>{html.escape(result['dataset'])}</b> · {result['n_questions']} questions
({result['n_unanswerable']} unanswerable) · generated {result['generated_at']} · runtime {result['runtime_s']:.1f}s</div>
<div class="meta">LLM <code>{html.escape(cfg['llm'])}</code> · embeddings <code>{html.escape(cfg['embeddings'])}</code> ·
reranker <code>{html.escape(cfg['reranker'])}</code> · verifier NLI <code>{html.escape(cfg['nli'])}</code> ·
judge NLI <code>{html.escape(cfg['judge_nli'])}</code></div>
<h2>End-to-end comparison</h2>
<img src="data:image/png;base64,{imgs['systems']}">
{sys_table}
<h2>Retrieval comparison</h2>
{'<img src="data:image/png;base64,' + imgs['retrieval'] + '">' if 'retrieval' in imgs else ''}
{retr_table}
<h2>Latency</h2>
<img src="data:image/png;base64,{imgs['latency']}">
{fault_html}
<h2>How to read this</h2>
<p><b>Faithfulness</b>: share of answer sentences entailed by at least one retrieved passage (1 − hallucination rate).
<b>Citation recall</b>: share of sentences whose cited passages entail them. <b>Citation precision</b>: share of citations
that are actually needed (ALCE definition). <b>Ghost citations</b>: citations to sources that do not exist.
If the judge NLI is the same model as the verifier, grounding metrics for TrustRAG are optimistic. Set
<code>evaluation.judge_nli_backend</code> to a different model for an independent judge.</p>
</main></body></html>"""
    path = out / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path
