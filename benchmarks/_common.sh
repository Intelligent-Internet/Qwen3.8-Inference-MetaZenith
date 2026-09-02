#!/usr/bin/env bash

benchmark_die() {
  printf '[benchmark][error] %s\n' "$*" >&2
  exit 1
}

benchmark_log() {
  printf '[benchmark] %s\n' "$*" >&2
}

benchmark_init() {
  BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
  MODEL="${MODEL:-}"

  command -v curl >/dev/null 2>&1 || benchmark_die "curl is required"
  command -v python3 >/dev/null 2>&1 || benchmark_die "python3 is required"
  command -v tool-eval-bench >/dev/null 2>&1 || benchmark_die \
    "tool-eval-bench is not installed; see benchmarks/README.md"

  if [ -z "${TOOL_EVAL_API_KEY:-}" ] && [ -n "${OPENAI_API_KEY:-}" ]; then
    export TOOL_EVAL_API_KEY="$OPENAI_API_KEY"
  fi

  local curl_auth=()
  if [ -n "${TOOL_EVAL_API_KEY:-}" ]; then
    curl_auth=(-H "Authorization: Bearer $TOOL_EVAL_API_KEY")
  fi

  curl -fsS -m 5 "${curl_auth[@]}" "${BASE_URL%/}/health" >/dev/null \
    || benchmark_die "server is not healthy at ${BASE_URL%/}/health"

  if [ -z "$MODEL" ]; then
    MODEL=$(curl -fsS -m 10 "${curl_auth[@]}" "${BASE_URL%/}/v1/models" \
      | python3 -c \
        'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])') \
      || benchmark_die "could not discover MODEL from ${BASE_URL%/}/v1/models"
  fi

  benchmark_log "endpoint=$BASE_URL model=$MODEL"
}
