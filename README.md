# Installing the Optimized Qwen3.8 Build

This directory contains the optimized vLLM source and production launcher for
`RadixArk/Qwen3.8-27B-NVFP4` on one NVIDIA RTX 5090.

The selected version is R107 on the `champion-r107` branch.

## 1. Select the branch

```bash
cd /root/qwen3_optimze_inference
git switch champion-r107
```

## 2. Prepare the model

By default, the server loads the model from:

```text
/root/models/RadixArk/Qwen3.8-27B-NVFP4
```

If the model is stored elsewhere, you do not need to edit the source. Pass its
location through `MODEL_DIR` when starting the server, as shown in section 4.

## 3. Install from source

The machine must have:

- An NVIDIA driver and CUDA toolkit with `nvcc` available.
- Git and uv.
- Python 3.12. If it is unavailable, run `uv python install 3.12`.

Install the project with:

```bash
cd /root/qwen3_optimze_inference/src

export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8
export NVCC_THREADS=2
export VLLM_VERSION_OVERRIDE=0.27.1

uv sync --frozen
```

`uv sync --frozen` creates `src/.venv`, installs the exact dependencies from
`uv.lock`, and builds vLLM from the optimized source in `src/upstream`.

Do not run an additional `pip install vllm` command.

### Why is `TORCH_CUDA_ARCH_LIST` set to `12.0`?

This value is correct for the machine used to optimize and validate the build:

```text
GPU: NVIDIA GeForce RTX 5090
Compute capability: 12.0
CUDA architecture: sm_120
```

Therefore, `TORCH_CUDA_ARCH_LIST=12.0` instructs the compiler to generate code
for the RTX 5090. When installing on a different GPU, use that GPU's compute
capability instead of copying `12.0` unchanged.

Check the current GPU with:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

## 4. Start the server

If the model is in the default location:

```bash
cd /root/qwen3_optimze_inference/src
./start.sh
```

If the model is stored elsewhere:

```bash
cd /root/qwen3_optimze_inference/src
MODEL_DIR=/path/to/model ./start.sh
```

The server uses:

- Address: `http://127.0.0.1:8000`
- API model name: `qwen38-nvfp4`
- Maximum context length: `262144` tokens
- Maximum concurrency: `4`
- MTP: `3` speculative tokens

## 5. Test the server

In another terminal, run:

```bash
curl -f http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
```

Send a test request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen38-nvfp4",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## Optimized version

```text
Branch: champion-r107
R107 commit: f4a6d9b0181749dae5475b6cc2c5d1ce1b751f14
Fixed-grid score: 84.89430291791149
MMLU-Pro: 57/64
Tool quality: 97/100
```

The optimized source on this branch is based on R107 commit
`f4a6d9b0181749dae5475b6cc2c5d1ce1b751f14`.
