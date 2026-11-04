"""Configuration loading: YAML file -> nested dict, with environment overrides.

Any key can be overridden with an env var of the form
TRUSTRAG__SECTION__KEY=value, e.g. TRUSTRAG__LLM__PROVIDER=groq
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: Dict[str, Any] = {
    "mode": "lite",
    "index_dir": "indexes/demo",
    "chunking": {"max_words": 120, "overlap_sentences": 1},
    "embeddings": {
        "backend": "lsa",                # lsa (offline) | sentence-transformers
        "model": "BAAI/bge-base-en-v1.5",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "lsa_dims": 256,
        "device": "cpu",
        "batch_size": 32,
    },
    "retrieval": {
        "candidates_per_retriever": 30,
        "rrf_k": 60,
        "use_bm25": True,
        "use_dense": True,
        "final_k": 5,
        "dedup_threshold": 0.92,
    },
    "reranker": {
        "enabled": True,
        "backend": "lexical",            # lexical (offline) | cross-encoder
        "model": "BAAI/bge-reranker-base",
        "top_n": 20,
    },
    "retrieval_gate": {                  # CRAG-style evaluator
        "enabled": True,
        "upper": 0.55,                   # >= upper  -> CORRECT
        "lower": 0.25,                   # <  lower  -> INCORRECT
        "keep_min_score": 0.15,
        "max_fallback_rounds": 2,
        "abstain_if_incorrect": True,
    },
    "web_fallback": {"enabled": False, "provider": "semantic_scholar", "max_results": 5},
    "llm": {
        "provider": "extractive",        # extractive (offline) | groq | openai | ollama | gemini | openai_compatible
        "model": "llama-3.3-70b-versatile",
        "base_url": "",
        "api_key_env": "GROQ_API_KEY",
        "temperature": 0.0,
        "max_tokens": 600,
        "timeout": 60,
        "max_retries": 4,
    },
    "generation": {"max_sentences": 6},
    "nli": {
        "backend": "lexical",            # lexical (offline) | transformers
        "model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "device": "cpu",
        "batch_size": 16,
        "max_length": 512,
        "window_sizes": [1, 2, 3],
    },
    "verification": {
        "entail_threshold": 0.5,
        "contradiction_threshold": 0.6,
        "numeric_guard": True,
    },
    "correction": {
        "enabled": True,
        "recite": True,
        "prune_citations": True,
        "regenerate": True,              # needs a generative LLM; otherwise unsupported claims are removed
        "max_citations_per_claim": 2,
    },
    "policy": {"min_kept_ratio": 0.5, "abstain_message": "I could not find enough reliable evidence in the sources to answer this question."},
    "evaluation": {
        "judge_nli_backend": "same",     # same | lexical | transformers (use a DIFFERENT model to avoid circularity)
        "judge_nli_model": "cross-encoder/nli-deberta-v3-base",
        "k_values": [1, 3, 5],
    },
    "logging": {"sqlite_path": "logs/trustrag.sqlite", "enabled": True},
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except Exception:
        return value


def _apply_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in os.environ.items():
        if not key.startswith("TRUSTRAG__"):
            continue
        path = [p.lower() for p in key.split("__")[1:]]
        node = cfg
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = _coerce(value)
    return cfg


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path)
        if not p.exists():
            p = ROOT / path
        with open(p, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    cfg = _apply_env(cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path
