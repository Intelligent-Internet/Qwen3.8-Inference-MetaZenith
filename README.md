# Shared source-built vLLM workspace

This workspace serves `RadixArk/Qwen3.8-27B-NVFP4` from one editable local
source build of vLLM shared by all optimization runs. R000 is the initial
baseline. The upstream source is
<https://github.com/vllm-project/vllm.git> tag `v0.27.1`, commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`.

The only source change relative to that tag is the TurboQuant MTP K+1 verify
routing correction from upstream PR 40914. It is required to prevent
context-blind graph capture and output corruption with MTP over 4-bit KV.
The server configuration otherwise matches the frozen vLLM-serving reference.

The uv environment is `src/.venv`, and the editable runtime source is
`src/upstream`. Build once and launch from this directory:

```bash
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8 NVCC_THREADS=2
export VLLM_VERSION_OVERRIDE=0.27.1
uv sync --frozen
./start.sh
```

Keep using this same environment for later runs. Python, launcher, and
configuration edits are immediately visible through the editable install and
must not invoke a native rebuild. Rebuild only after changing C++, CUDA, or
other compiled sources; retain the shared uv, compiler, and incremental build
caches.

The unified server listens on port 8000 with a 262,144-token maximum context,
maximum concurrency four, and MTP-3. It is used unchanged for correctness,
tool quality, and throughput evaluation.

Verify source provenance after synchronization:

```bash
uv run python -c 'import vllm; print(vllm.__file__)'
uv run python -c 'import vllm._C_stable_libtorch as c; print(c.__file__)'
```

The Python module must resolve under `src/upstream`, the extension must resolve
to the locally built binary in that tree, and the interpreter must resolve
under `src/.venv`.
The vLLM Apache-2.0 license is retained as `LICENSE` and in `upstream/LICENSE`.
