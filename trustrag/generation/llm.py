"""LLM providers behind one tiny interface: complete(system, user) -> str.

Most hosted and local servers speak the OpenAI chat-completions protocol, so a
single HTTP client covers Groq, OpenAI, Gemini (OpenAI-compatible endpoint),
Ollama, LM Studio, vLLM, Together, OpenRouter ...

`extractive` is an offline, non-generative stand-in used for tests/lite mode:
it copies the most query-relevant evidence sentences and cites them.
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import List, Optional

from ..text import split_sentences, tokenize

PROVIDER_DEFAULTS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GOOGLE_API_KEY"),
    "ollama": ("http://localhost:11434/v1", ""),
    "openai_compatible": ("", "OPENAI_API_KEY"),
}


class LLM:
    is_generative = True
    name = "base"

    def complete(self, system: str, user: str, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:  # pragma: no cover
        raise NotImplementedError


class OpenAICompatibleLLM(LLM):
    def __init__(self, provider: str, model: str, base_url: str = "", api_key_env: str = "",
                 temperature: float = 0.0, max_tokens: int = 600, timeout: int = 60, max_retries: int = 4):
        default_url, default_env = PROVIDER_DEFAULTS.get(provider, ("", "OPENAI_API_KEY"))
        self.base_url = (base_url or default_url).rstrip("/")
        if not self.base_url:
            raise ValueError("llm.base_url is required for provider openai_compatible")
        env = api_key_env or default_env
        if not os.getenv(env or "_") and default_env and os.getenv(default_env):
            env = default_env  # e.g. provider changed but api_key_env left at GROQ_API_KEY
        self.api_key = os.getenv(env, "") if env else ""
        if provider not in ("ollama",) and not self.api_key:
            raise ValueError(f"Environment variable {env} is not set (needed for provider '{provider}'). "
                             f"Put it in your .env file.")
        self.name = f"{provider}:{model}"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(self, system: str, user: str, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers,
                                  timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return (r.json()["choices"][0]["message"]["content"] or "").strip()
            except Exception as exc:  # retry with exponential backoff + jitter
                last = exc
                if attempt < self.max_retries:
                    time.sleep(min(30, 2 ** attempt * 2) + random.random())
        raise RuntimeError(f"LLM call failed after retries: {last}")


class ExtractiveLLM(LLM):
    """Offline stand-in. Understands the TrustRAG generation prompt format:
    it reads the numbered sources and the question, and returns the most relevant
    sentences, each followed by the label of the source it came from."""

    is_generative = False
    name = "extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    @staticmethod
    def parse_prompt(user: str):
        q = re.search(r"QUESTION:\s*(.+?)\s*(?:\n|$)", user)
        question = q.group(1) if q else user
        sources = re.findall(r"^\[(S\d+)\][^\n]*\n(.+?)(?=\n\[S\d+\]|\nQUESTION:|\Z)", user, re.S | re.M)
        return question, [(lab, " ".join(txt.split())) for lab, txt in sources]

    def complete(self, system: str, user: str, temperature=None, max_tokens=None) -> str:
        question, sources = self.parse_prompt(user)
        if not sources:
            return "INSUFFICIENT_EVIDENCE"
        q = set(tokenize(question))
        scored = []
        for rank, (lab, txt) in enumerate(sources):
            for j, sent in enumerate(split_sentences(txt)):
                toks = set(tokenize(sent))
                if not toks:
                    continue
                overlap = len(q & toks) / max(1, len(q))
                scored.append((overlap - 0.03 * rank - 0.01 * j, sent, lab))
        scored.sort(key=lambda x: -x[0])
        picked, seen = [], set()
        for s, sent, lab in scored:
            if s <= 0.1 or sent in seen:
                continue
            seen.add(sent)
            body = sent.rstrip()
            end = body[-1] if body[-1] in ".!?" else "."
            body = body.rstrip(".!?")
            picked.append(f"{body} [{lab}]{end}")
            if len(picked) >= self.max_sentences:
                break
        return " ".join(picked) if picked else "INSUFFICIENT_EVIDENCE"


def build_llm(cfg: dict) -> LLM:
    lc = cfg["llm"]
    provider = lc.get("provider", "extractive")
    if provider == "extractive":
        return ExtractiveLLM(max_sentences=min(3, cfg["generation"].get("max_sentences", 3)))
    return OpenAICompatibleLLM(provider, lc["model"], lc.get("base_url", ""), lc.get("api_key_env", ""),
                               lc.get("temperature", 0.0), lc.get("max_tokens", 600), lc.get("timeout", 60),
                               lc.get("max_retries", 4))
