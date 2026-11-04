"""Flask web UI + REST API.

  GET  /                 chat-style UI (app/static/index.html)
  POST /api/ask          {"question": "...", "system": "trustrag"|"naive"|...}  -> full result JSON
  GET  /api/health       index + model info
  GET  /api/logs?n=20    recent logged runs
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from trustrag.config import load_config, resolve_path
from trustrag.logging_db import RunLogger
from trustrag.pipeline import SYSTEM_PRESETS, TrustRAGPipeline

STATIC = Path(__file__).parent / "static"


def create_app(cfg: dict | None = None) -> Flask:
    cfg = cfg or load_config("config/lite.yaml")
    app = Flask(__name__, static_folder=str(STATIC))
    base = TrustRAGPipeline(cfg)
    pipes = {"trustrag": base}

    def get(system: str) -> TrustRAGPipeline:
        if system not in SYSTEM_PRESETS:
            raise ValueError(f"unknown system {system}")
        if system not in pipes:
            p = base.with_system(system)
            p.logger = base.logger
            pipes[system] = p
        return pipes[system]

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "chunks": len(base.index), "llm": base.llm.name, "nli": base.nli.name,
                        "mode": cfg.get("mode"), "systems": list(SYSTEM_PRESETS)})

    @app.post("/api/ask")
    def ask():
        body = request.get_json(force=True, silent=True) or {}
        q = str(body.get("question", "")).strip()
        if not q:
            return jsonify({"error": "question is required"}), 400
        try:
            res = get(body.get("system", "trustrag")).run(q)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(res.to_dict())

    @app.get("/api/logs")
    def logs():
        n = int(request.args.get("n", 20))
        return jsonify(RunLogger(resolve_path(cfg["logging"]["sqlite_path"])).recent(n))

    return app


if __name__ == "__main__":
    create_app().run(port=8000)
