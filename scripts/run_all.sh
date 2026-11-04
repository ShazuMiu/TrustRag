#!/usr/bin/env bash
# Offline end-to-end check: ingest -> tests -> benchmark -> report
set -e
cd "$(dirname "$0")/.."
CFG=${1:-config/lite.yaml}
python -m trustrag.cli --config "$CFG" ingest --input data/demo/corpus
python -m unittest discover -s tests
python -m trustrag.cli --config "$CFG" eval --dataset demo --out results/demo_$(basename "$CFG" .yaml)
echo "Open results/demo_$(basename "$CFG" .yaml)/report.html"
