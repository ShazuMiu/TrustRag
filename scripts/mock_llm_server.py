"""A deliberately unreliable OpenAI-compatible LLM server, for testing without API keys.

It answers the TrustRAG generation prompt extractively but INJECTS errors:
a wrong citation, a ghost citation, a changed number and an invented sentence.
For the claim-rewrite prompt it returns the best evidence sentence, so the
rewrite branch of the corrector can be exercised.

    python scripts/mock_llm_server.py --port 8765
    TRUSTRAG__LLM__PROVIDER=openai_compatible TRUSTRAG__LLM__BASE_URL=http://127.0.0.1:8765/v1 \
    TRUSTRAG__LLM__API_KEY_ENV=MOCK_KEY MOCK_KEY=x python -m trustrag.cli ask "How long is the Padma Bridge?"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request  # noqa: E402

from trustrag.generation.llm import ExtractiveLLM  # noqa: E402
from trustrag.text import split_sentences, tokenize  # noqa: E402

app = Flask(__name__)
extractive = ExtractiveLLM(max_sentences=3)


def hallucinate(answer: str, n_sources: int) -> str:
    sents = split_sentences(answer)
    out = []
    for i, s in enumerate(sents):
        if i == 0 and n_sources > 1:          # wrong citation
            s = re.sub(r"\[S(\d+)\]", lambda m: f"[S{1 if m.group(1) != '1' else 2}]", s)
        elif i == 1:                            # numeric distortion, or ghost citation if no number
            t = re.sub(r"(?<![\w.])(\d+)(?![\w.])", lambda m: str(int(m.group(1)) * 2 + 1), s, count=1)
            s = t if t != s else re.sub(r"\[S\d+\]", f"[S{n_sources + 3}]", s)
        out.append(s)
    out.append("Independent studies later confirmed these results with a 99 percent success rate [S1].")
    return " ".join(out)


@app.post("/v1/chat/completions")
def chat():
    body = request.get_json(force=True)
    system = body["messages"][0]["content"]
    user = body["messages"][-1]["content"]
    if "fix one unsupported sentence" in system:
        sent = re.search(r"SENTENCE:\s*(.+)", user).group(1)
        ev = re.findall(r"^\[(S\d+)\]\s*(.+)$", user, re.M)
        q = set(tokenize(sent))
        best = None
        for lab, text in ev:
            for s in split_sentences(text):
                score = len(q & set(tokenize(s)))
                if best is None or score > best[0]:
                    best = (score, s, lab)
        reply = "DROP" if not best or best[0] < 2 else f"{best[1].rstrip('.')} [{best[2]}]."
    elif "rewrite search queries" in system:
        reply = re.sub(r"^Original query:\s*", "", user)
    else:
        n = len(re.findall(r"^\[S\d+\]", user, re.M))
        reply = extractive.complete(system, user)
        if reply != "INSUFFICIENT_EVIDENCE":
            reply = hallucinate(reply, n)
    return jsonify({"choices": [{"message": {"role": "assistant", "content": reply}}]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    app.run(port=ap.parse_args().port)
