# Optimized Qwen3.8 NVFP4 Inference on RTX 5090

This repository contains a specialized vLLM source tree for serving
`RadixArk/Qwen3.8-27B-NVFP4` on a single NVIDIA GeForce RTX 5090 (`sm_120`).
It is the result of an iterative optimization effort focused on exact
inference, long-context serving, speculative decoding, and the model's
TurboQuant and Gated Delta Network (GDN) execution paths.

This is not a general replacement for upstream vLLM. The current code is a
hardware- and workload-specific fork whose primary target is:

- Model: `RadixArk/Qwen3.8-27B-NVFP4`
- GPU: one NVIDIA GeForce RTX 5090
- CUDA architecture: `sm_120`
- Quantization: NVFP4 weights with TurboQuant 4-bit KV cache
- Speculative decoding: the model's MTP path
- Long-context inference: up to 262,144 tokens in the optimized workload

Serving, validation, and benchmark documentation will be added after the
source and release interface have been reviewed.

## Starting point

The initial baseline is recorded by the first commit in this repository. It
was built from:

- [vLLM](https://github.com/vllm-project/vllm) tag `v0.27.1`
- Upstream commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`
- The TurboQuant MTP K+1 verification-routing correction from upstream
  [PR #40914](https://github.com/vllm-project/vllm/pull/40914)

That routing correction was the only source-level deviation from the vLLM tag
in the baseline. It prevents context-blind CUDA Graph capture and incorrect
outputs when MTP is used with the 4-bit KV cache.

The baseline used one editable local vLLM build and a fixed serving workload so
that later experiments changed the inference implementation without silently
changing the model, server configuration, or dependency environment.

## Optimization journey

The commits after the baseline preserve the progression from isolated decode
changes to a composed inference pipeline. The main stages are summarized below
in chronological order.

### 1. TurboQuant decode working set and context-aware dispatch

The first changes reduced the token-tile working set of the Triton TurboQuant
decode kernel and selected different decode layouts according to context
length. CUDA Graph dispatch was then made aware of the same context regimes so
that short- and long-context paths could be captured safely.

This stage introduced:

- Smaller decode tiles to reduce per-launch working data.
- Context-dependent TurboQuant kernel selection.
- Full CUDA Graph capture boundaries aligned with kernel dispatch boundaries.
- Composition of the TurboQuant GQA decode path with full graphs.
- A specialized six-head KV reduction path for the target model layout.

### 2. NVFP4 output projection and speculative GDN input handling

The next stage optimized work around the attention kernels:

- Small language-model-head workloads are dispatched to the FlashInfer B12x
  NVFP4 GEMM path when supported.
- The speculative GDN update reads Q, K, and V directly from the packed QKV
  representation instead of materializing equivalent intermediate tensors.

These changes reduce conversion and materialization overhead in the small,
latency-sensitive operations surrounding decode.

### 3. Native exact TurboQuant decode kernels

Two CUDA implementations were added for exact TurboQuant decode:

- A head-parallel kernel for context regimes where independent head work is
  preferable.
- A shared high-batch kernel that reuses dequantized KV data across compatible
  requests and heads.

The dispatcher selects between these implementations and the Triton paths
using batch shape, context length, KV layout, and speculative-decoding state.
The kernels were subsequently composed with the full CUDA Graph pipeline.

### 4. Direct-output paths and GDN prefill on Blackwell

Several intermediate output copies were removed by allowing attention and GDN
operations to write directly into their final output buffers. The pure-prefill
GDN path was enabled through FlashInfer on `sm_120`, while speculative GDN
updates retained an exact packed-input implementation.

This stage focused on eliminating redundant memory traffic while preserving the
same inference results.

### 5. Batched-prefill scheduling and KV-cache preparation

The serving pipeline was adjusted so optimized kernels receive suitable memory
and scheduling conditions:

- Hybrid KV-cache capacity is reserved for batched prefills.
- FlashInfer tuning runs before KV-cache allocation where supported.
- Long prefill chunks adapt to the active requests' fair share of the token
  budget instead of relying only on a fixed threshold.

These changes target mixed decode/prefill workloads and reduce avoidable
contention between long prompts.

### 6. Exact prefill fusion

The GDN prefill path was progressively fused into the surrounding operations:

- Convolution outputs are routed directly into the GDN prefill computation.
- Q/K normalization is performed inside the fused path.
- NVFP4 activation quantization and GDN prefill outputs are composed into the
  same optimized pipeline where the target shapes allow it.

The goal is to avoid round trips through temporary tensors between convolution,
normalization, quantization, and recurrent-attention preparation.

### 7. TurboQuant layout and high-batch refinements

The final optimization stage tightened the specialized TurboQuant paths:

- Reused layout proofs and cached norm information outside repeated inner work.
- Rebalanced long-context B4 splits.
- Vectorized packed B4 KV loads.
- Warmed the continuation-prefill dequantization and inverse-rotation path.
- Shortened score lifetimes in the B16 shared kernel to reduce register
  pressure.
- Aligned the packed value payload and slot ABI across store, dispatch, Triton,
  and CUDA implementations.

At the current revision, the optimization delta relative to the recorded
baseline touches 24 upstream source files, with approximately 3,309 inserted
lines and 186 removed lines. Most of that delta is concentrated in TurboQuant
decode, GDN prefill/speculative decode, CUDA Graph dispatch, and scheduler/KV
cache integration.

## Repository layout

- `upstream/` — the vendored vLLM source tree and all optimized runtime code.
- `benchmarks/` — reproducible throughput and tool-call benchmark wrappers.
- `pyproject.toml` — the local editable-build environment definition.
- `uv.lock` — the frozen Python dependency resolution used during development.
- `LICENSE` — the retained Apache License 2.0.

## Benchmarks

The release includes client-side scripts for measuring serving throughput and
tool-call quality against a running OpenAI-compatible endpoint. See
[`benchmarks/README.md`](benchmarks/README.md) for the pinned benchmark client,
server requirements, workload controls, and result locations. The exact r107
server and workload configuration is preserved in
[`benchmarks/profiles/r107-sm120.md`](benchmarks/profiles/r107-sm120.md).
Plain-text terminal logs for R000 and R107 are available in
[`benchmarks/results/`](benchmarks/results/README.md).

## Installation

The first supported build profile is intentionally narrow:

- Ubuntu 24.04 or another `x86_64` Linux distribution with glibc 2.38 or
  newer.
- Python 3.12.
- NVIDIA GeForce RTX 5090, compute capability 12.0 (`sm_120`).
- An NVIDIA driver compatible with CUDA 13.0.
- CUDA Toolkit 13.0 with `nvcc` only when building from source.
- The dependency versions recorded in `requirements-prebuilt.txt` for the
  wheel, or `uv.lock` for a source build.

Other environments may work, but they have not yet been included in the
release validation matrix.

### Pre-built artifacts

The pre-built wheel is the recommended installation method because compiling
the vendored vLLM CUDA extensions can take a long time. The first validated
artifact is:

```text
vllm-0.27.1+qwen38.r107.cu130.sm120-cp312-cp312-linux_x86_64.whl
```

It targets CPython 3.12, CUDA 13.0, and `sm_120`. It was built on Ubuntu 24.04
and requires glibc 2.38 or newer. Installing the wheel does not require the
CUDA Toolkit or `nvcc`; the machine still needs a compatible NVIDIA driver.

Install Git, `curl`, and [uv](https://docs.astral.sh/uv/), then run the commands
below. Cloning the release tag ensures that `requirements-prebuilt.txt` matches
the wheel:

```bash
RELEASE_TAG=v0.27.1-qwen38-r107-cu130-sm120
WHEEL=vllm-0.27.1+qwen38.r107.cu130.sm120-cp312-cp312-linux_x86_64.whl

git clone --branch "$RELEASE_TAG" --depth 1 \
  https://github.com/Intelligent-Internet/Qwen3.8-Inference-AutoResearch.git
cd Qwen3.8-Inference-AutoResearch

mkdir -p release-assets
curl -fL \
  "https://github.com/Intelligent-Internet/Qwen3.8-Inference-AutoResearch/releases/download/$RELEASE_TAG/vllm-0.27.1%2Bqwen38.r107.cu130.sm120-cp312-cp312-linux_x86_64.whl" \
  -o "release-assets/$WHEEL"
curl -fL \
  "https://github.com/Intelligent-Internet/Qwen3.8-Inference-AutoResearch/releases/download/$RELEASE_TAG/SHA256SUMS" \
  -o release-assets/SHA256SUMS

(cd release-assets && sha256sum -c SHA256SUMS)

uv venv --python 3.12
uv pip sync requirements-prebuilt.txt
uv pip install --no-deps "release-assets/$WHEEL"
```

Use the virtual environment directly so that the project manager does not try
to replace the wheel with the editable source dependency:

```bash
.venv/bin/python - <<'PY'
import torch
import vllm

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("Compute capability:", torch.cuda.get_device_capability())
PY
```

The expected vLLM version is
`0.27.1+qwen38.r107.cu130.sm120`, and the compute capability must be `(12, 0)`.
For subsequent commands, call executables through `.venv/bin/`, or use
`uv run --no-sync`; a plain `uv run` may synchronize the editable source build.

The `sha256sum` step must report the wheel as `OK`. Do not substitute an
upstream vLLM wheel: this fork changes C++ and CUDA code, so upstream pre-built
extensions do not contain the optimized kernels.

### Build from source

Install the following system prerequisites:

- A recent NVIDIA driver that supports the installed CUDA 13 runtime.
- CUDA Toolkit 13.0, including `nvcc`.
- GCC and G++ 11.3 or newer.
- Git and [uv](https://docs.astral.sh/uv/).

Confirm that the GPU and compiler are visible:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
nvcc --version
```

Clone the repository, then build the locked environment:

```bash
git clone https://github.com/Intelligent-Internet/Qwen3.8-Inference-AutoResearch.git
cd Qwen3.8-Inference-AutoResearch

export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8
export NVCC_THREADS=2
export VLLM_VERSION_OVERRIDE=0.27.1

uv sync --frozen
```

`uv sync --frozen` creates `.venv`, installs the exact dependency resolution
from `uv.lock`, and builds the optimized vLLM source in `upstream/` as an
editable local dependency. Do not run a separate `pip install vllm`, because
that can replace the optimized local build with an upstream package.

`MAX_JOBS` and `NVCC_THREADS` control build parallelism. The values above were
used on the development machine; reduce them if compilation exhausts system
memory.

Verify the resulting environment:

```bash
uv run python - <<'PY'
import torch
import vllm

print("vLLM:", vllm.__file__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("Compute capability:", torch.cuda.get_device_capability())
PY
```

The vLLM module path must resolve inside this checkout's `upstream/` directory,
and the reported compute capability must be `(12, 0)` for the validated RTX
5090 profile.

### CUDA version versus compute capability

`CUDA 13.0` and `sm_120` describe different things:

- **CUDA 13.0** is the compiler/toolkit and runtime generation.
- **12.0** in `TORCH_CUDA_ARCH_LIST=12.0` is the GPU compute capability and
  produces code for `sm_120`.

The current CUDA 13.0 compiler and this vLLM source tree support Blackwell
targets through `sm_121`; they do not define an `sm_130` target. Therefore,
`TORCH_CUDA_ARCH_LIST="12.0 13.0"` is not a valid build configuration for this
release.

A wheel may contain code for multiple supported compute capabilities by using
a space-separated architecture list, for example `"12.0 12.1"`. Such a build
is larger, takes longer to compile, and still requires correctness and
performance testing on every included GPU. The optimized path in this
repository has currently been validated only on `sm_120`, so the first release
will use a dedicated `sm_120` binary rather than an unvalidated multi-GPU
wheel.
