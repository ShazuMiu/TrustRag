# TrustRAG v2: Trustworthy Retrieval, Attribution and Generation

A RAG system built to (1) retrieve better evidence, (2) produce fewer hallucinated
claims, and (3) give correct citations. The full guide (architecture, every module,
how to run and test it, results) is in **TrustRAG_v2_Guide.html**.

## Quick start (offline, about 2 minutes, no API key)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-lite.txt
python -m trustrag.cli --config config/lite.yaml ingest --input data/demo/corpus
python -m trustrag.cli --config config/lite.yaml ask "How long is the Padma Bridge?"
python -m unittest discover -s tests -v
python -m trustrag.cli --config config/lite.yaml eval --dataset demo --out results/demo_lite
python -m trustrag.cli --config config/lite.yaml serve        # http://127.0.0.1:8000
```

## Full neural mode (for thesis results)
```bash
pip install -r requirements.txt
cp .env.example .env            # put GROQ_API_KEY=... (or use Ollama, see config/full.yaml)
python -m trustrag.cli --config config/full.yaml ingest --input data/demo/corpus
python -m trustrag.cli --config config/full.yaml eval --dataset ragbench --subset pubmedqa --limit 100
```

## Layout
```
trustrag/ingest        loaders (.md/.txt/.pdf/.docx/.jsonl) + sentence-aware chunker
trustrag/index         BM25, embeddings (BGE or offline LSA), vector index, on-disk store
trustrag/retrieval     hybrid RRF retriever, reranker, CRAG-style gate + fallback, web fallback
trustrag/generation    LLM providers (Groq/OpenAI/Gemini/Ollama/extractive), prompts
trustrag/verification  claim parsing, NLI backends, verifier, corrector + answer policy
trustrag/evaluation    metrics, datasets, benchmark runner, fault injection, HTML report
trustrag/pipeline.py   end-to-end pipeline + baseline/ablation presets
app/                   Flask REST API + web UI
scripts/               run_all scripts, mock (hallucinating) LLM server
tests/                 unittest suite (29 tests)
```
