#!/usr/bin/env bash
# Gateway overhead: p99(via ModelMux) - p99(direct to upstream), same machine,
# same fixed-latency stub upstream, caches off. Alternates direct/gateway runs
# ROUNDS times and reports the median of each.
set -euo pipefail
cd "$(dirname "$0")/.."
RATE=${RATE:-2000} DURATION=${DURATION:-30s} ROUNDS=${ROUNDS:-3} LATENCY=${LATENCY:-50ms}
BIN=$(mktemp -d)
go build -o "$BIN/stubllm" ./cmd/stubllm
go build -o "$BIN/modelmux" ./cmd/modelmux
ulimit -n 65536 2>/dev/null || ulimit -n 10240 || true
"$BIN/stubllm" -addr 127.0.0.1:9090 -latency "$LATENCY" >/dev/null 2>&1 & STUB=$!
MODELMUX_CONFIG=configs/bench.yaml "$BIN/modelmux" >/dev/null 2>"$BIN/mm.log" & MM=$!
trap 'kill $STUB $MM 2>/dev/null; rm -rf "$BIN"' EXIT
for _ in $(seq 50); do curl -sf 127.0.0.1:8080/healthz >/dev/null && curl -s 127.0.0.1:9090 >/dev/null && break; sleep 0.1; done
# Warm-up both paths so connection pools are established.
k6 run -q -e TARGET=http://127.0.0.1:9090/v1/chat/completions -e RATE=200 -e DURATION=5s -e OUT=/dev/null bench/overhead.js >/dev/null
k6 run -q -e TARGET=http://127.0.0.1:8080/v1/chat/completions -e RATE=200 -e DURATION=5s -e OUT=/dev/null bench/overhead.js >/dev/null
for r in $(seq "$ROUNDS"); do
  for target in direct gateway; do
    url=http://127.0.0.1:9090/v1/chat/completions
    [ "$target" = gateway ] && url=http://127.0.0.1:8080/v1/chat/completions
    echo "round $r: $target"
    if [ "$target" = gateway ]; then
      ( while kill -0 $MM 2>/dev/null; do ps -o rss=,%cpu= -p $MM >> bench/results/overhead-proc.tmp; sleep 1; done ) & SAMPLER=$!
    fi
    k6 run -q -e TARGET=$url -e RATE="$RATE" -e DURATION="$DURATION" -e LABEL="$target-$r" \
      -e OUT="bench/results/overhead-$target-$r.tmp.json" bench/overhead.js >/dev/null
    if [ "$target" = gateway ]; then kill $SAMPLER 2>/dev/null || true; fi
  done
done
python3 bench/aggregate_overhead.py "$ROUNDS" "$RATE" "$DURATION" "$LATENCY" > bench/results/overhead.json
rm -f bench/results/overhead-*.tmp.json bench/results/overhead-proc.tmp
cat bench/results/overhead.json
