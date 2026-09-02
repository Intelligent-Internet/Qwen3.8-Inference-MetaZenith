#!/usr/bin/env bash
# Evaluate tool-call quality against a running OpenAI-compatible server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

benchmark_init

SUITE="${SUITE:-short}"
SEED="${SEED:-42}"
TRIALS="${TRIALS:-1}"
PARALLEL="${PARALLEL:-1}"
TIMEOUT="${TIMEOUT:-300}"
LABEL="${LABEL:-qwen38-tool-calls}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/tool-calls}"

suite_args=()
case "$SUITE" in
  short) suite_args=(--short) ;;
  standard) ;;
  hardmode) suite_args=(--hardmode) ;;
  *) benchmark_die "SUITE must be short, standard, or hardmode" ;;
esac

mkdir -p "$OUTPUT_DIR"
cd "$SCRIPT_DIR"

benchmark_log \
  "suite=$SUITE seed=$SEED trials=$TRIALS parallel=$PARALLEL timeout=${TIMEOUT}s"

tool-eval-bench run \
  "${suite_args[@]}" \
  --backend vllm \
  --format openai \
  --no-probe-engine \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --seed "$SEED" \
  --trials "$TRIALS" \
  --parallel "$PARALLEL" \
  --timeout "$TIMEOUT" \
  --label "$LABEL" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

benchmark_log "reports=$OUTPUT_DIR"
