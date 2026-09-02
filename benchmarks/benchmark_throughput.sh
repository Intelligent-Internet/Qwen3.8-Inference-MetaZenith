#!/usr/bin/env bash
# Run a reproducible prompt/decode throughput sweep against a running server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

benchmark_init

PP="${PP:-2048}"
TG="${TG:-128}"
DEPTH="${DEPTH:-0,4096,8192,16384,32768,65536}"
CONCURRENCY="${CONCURRENCY:-1,2,4}"
TIMEOUT="${TIMEOUT:-300}"
LABEL="${LABEL:-qwen38-throughput}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/throughput}"

mkdir -p "$OUTPUT_DIR"
cd "$SCRIPT_DIR"

benchmark_log "pp=$PP tg=$TG depth=$DEPTH concurrency=$CONCURRENCY"

tool-eval-bench bench \
  --perf-legacy-only \
  --backend vllm \
  --format openai \
  --no-probe-engine \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --pp "$PP" \
  --tg "$TG" \
  --depth "$DEPTH" \
  --concurrency "$CONCURRENCY" \
  --timeout "$TIMEOUT" \
  --label "$LABEL" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

benchmark_log "reports=$OUTPUT_DIR"
