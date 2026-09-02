# Benchmarks

These scripts measure an already running OpenAI-compatible inference server.
They never start, stop, or reconfigure the server.

## Install the benchmark client

The wrappers use
[`tool-eval-bench`](https://github.com/SeraphimSerapis/tool-eval-bench). The
installation is pinned to the exact revision used for the recorded r107 runs,
so different machines use the same scenarios and scoring logic:

```bash
uv tool install \
  'tool-eval-bench[perf] @ git+https://github.com/SeraphimSerapis/tool-eval-bench.git@d4381ce5c'
tool-eval-bench --version
```

The expected version is `2.5.1.dev46+gd4381ce5c`.

The server must expose `/health`, `/v1/models`, and the OpenAI-compatible chat
completion API. Tool-call evaluation additionally requires the server to be
started with tool parsing enabled. For the target model, use the equivalent of:

```text
--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
```

For an authenticated endpoint, set `TOOL_EVAL_API_KEY` or `OPENAI_API_KEY` in
the environment. Secrets are not written to benchmark reports by these
wrappers.

## Throughput

Run the default prompt/decode grid:

```bash
./benchmarks/benchmark_throughput.sh
```

The default workload uses 2,048 prompt tokens, 128 generated tokens, context
depths from 0 to 65,536, and concurrency levels 1, 2, and 4. Override any
dimension with environment variables:

```bash
PP=4096 TG=256 DEPTH=0,32768,65536 CONCURRENCY=1,4 \
  ./benchmarks/benchmark_throughput.sh
```

Repeated prompts can benefit from prefix caching. Disable prefix caching on
the server or restart it between comparable runs when measuring cold-prefill
performance.

## Tool-call quality

Run the deterministic core 15-scenario suite:

```bash
./benchmarks/benchmark_tool_calls.sh
```

Use `SUITE=standard` for the full standard suite or `SUITE=hardmode` to include
the hard-mode scenarios. More trials provide a stronger comparison:

```bash
SUITE=standard TRIALS=3 PARALLEL=4 \
  ./benchmarks/benchmark_tool_calls.sh
```

## Common configuration

Both scripts accept these environment variables:

- `BASE_URL`: server root URL; default `http://127.0.0.1:8000`.
- `MODEL`: served model name; auto-detected from `/v1/models` when omitted.
- `TIMEOUT`: per-request timeout in seconds; default 300.
- `LABEL`: label stored in generated reports.
- `OUTPUT_DIR`: output directory for reports.

Additional command-line arguments are forwarded to `tool-eval-bench`. Generated
reports and its local benchmark database are ignored by Git.

## Recorded release profile

[`profiles/r107-sm120.md`](profiles/r107-sm120.md) records the exact software,
server flags, throughput grid, tool-call settings, and comparison methodology
used for the r107 RTX 5090 evaluation. Use that profile when reproducing the
published optimization setup; use environment overrides only for exploratory
runs.
