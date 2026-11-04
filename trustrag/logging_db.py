"""SQLite audit log (FR10 / NFR8): every query, answer, verdict and timing."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .schemas import PipelineResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, system TEXT, query TEXT, status TEXT, answer TEXT,
    retrieval_decision TEXT, confidence REAL, n_claims INTEGER,
    n_supported INTEGER, n_removed INTEGER, latency REAL, payload TEXT
)"""


class RunLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path) as con:
            con.execute(SCHEMA)

    def log(self, r: PipelineResult) -> None:
        n_sup = sum(v.label == "supported" for v in r.verdicts)
        n_rem = sum(v.action == "removed" for v in r.verdicts)
        with sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT INTO runs (ts, system, query, status, answer, retrieval_decision, confidence, n_claims,"
                " n_supported, n_removed, latency, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), r.system, r.query, r.status, r.answer, r.retrieval_decision,
                 r.confidence, len(r.verdicts), n_sup, n_rem, r.timings.get("total", 0.0),
                 json.dumps(r.to_dict(), ensure_ascii=False)),
            )

    def recent(self, n: int = 20):
        with sqlite3.connect(self.path) as con:
            con.row_factory = sqlite3.Row
            return [dict(x) for x in con.execute(
                "SELECT id, ts, system, query, status, confidence, n_claims, n_supported, n_removed, latency "
                "FROM runs ORDER BY id DESC LIMIT ?", (n,))]
