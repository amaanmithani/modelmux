#!/usr/bin/env bash
# Regenerate every committed benchmark result, then the README table.
# Needs: Go, k6, Ollama with nomic-embed-text, all-minilm, mxbai-embed-large; uv.
set -euo pipefail
cd "$(dirname "$0")/.."
./bench/overhead.sh
go run ./bench/faults
[ -f bench/data/qqp_heldout.jsonl ] || uv run bench/semantic/fetch_qqp.py
go run ./bench/semantic
go test -run xxx -bench SemanticLookup -benchmem -count 3 ./internal/cache/ > bench/results/semantic-lookup.txt
python3 bench/report.py
