# Blackwell (sm_120) serving landmines — field log

The point of this file: getting vLLM to run *well* on a 5090 is bleeding-edge
and under-documented, and a clean log of what broke and how it was fixed is
half the value of this repo to a reader evaluating production capability. Fill
each entry with the **symptom**, the **root cause**, and the **fix that
worked** — the fix is worthless without the diagnosis that led to it.

Format per entry:

```
### <short symptom>
- Environment: <driver / CUDA / vLLM / kernel / torch versions>
- Symptom: <exact error text or observed behavior>
- Root cause: <what was actually wrong>
- Fix: <the change that resolved it>
- Cost: <time lost / how you found it>
```

---

## Verified on this box (2026-08-18)

Stack under test, and the state every verdict below refers to:

```
GPU        NVIDIA GeForce RTX 5090 (sm_120), 32607 MiB
driver     580.159.03          CUDA runtime 13.0
kernel     Linux 6.17.0-35-generic
torch      2.13.0+cu130        vLLM 0.27.1
flashinfer 0.6.16.post3        transformers 5.15.0
```

### REFUTED on this stack: vanilla vLLM install rejects sm_120
- Verdict: **does not reproduce.** `pip install vllm==0.27.1` into a clean venv
  produced a working sm_120 stack with no special index, no source build, and
  no intervention.
- Evidence: `torch.cuda.get_arch_list()` includes `sm_120`; an fp16 4096x4096
  matmul and a `torch.float8_e4m3fn` tensor both execute on device; vLLM
  reports `DeviceCapability(major=12, minor=0)`.
- Why it changed: stock torch wheels now ship Blackwell kernels. The original
  entry was real in the early Blackwell window; on torch >= 2.13 it is gone.

### NOT REPRODUCED: flash-attn symbol errors on kernel 6.14
- Verdict: **no errors on kernel 6.17.** vLLM selected `FLASH_ATTN` out of
  `[FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION]` and served normally.
  The FlashInfer bypass was not needed.
- Caveat: this box runs 6.17, not the 6.14 the original report named. This is
  evidence the problem is absent on the current kernel, NOT that it was never
  real on 6.14.

### CONFIRMED WORKING: AWQ-4bit serving path
- vLLM auto-detects `quantization=auto_awq` from the checkpoint config and
  selects `MarlinLinearKernel for AutoAWQMarlinLinearMethod`. No
  `--quantization` flag required.
- Qwen3-8B-AWQ at `--gpu-memory-utilization 0.85 --max-model-len 8192`:
  7.19 GiB weights+non-torch, 1.16 GiB peak activation, 0.47 GiB CUDA graphs,
  leaving **18.28 GiB KV cache = 133,104 tokens**.
- Startup 120 s cold (66 s of it CUDA-graph capture), ~40 s with a warm compile
  cache. Warmup discard is not optional; see METHODOLOGY control 7.

### Gotcha found here: attention backend is auto-selected
- Not a crash -- a silent threat to the comparison. vLLM picks the backend at
  startup, and `harness/common.py::backend_env()` treats it as a CONTROLLED
  variable. Nothing guarantees all three arms pick the same one.
- Fix: export `VLLM_ATTENTION_BACKEND=FLASH_ATTN` for every arm. Now pinned in
  the launch comment of each `configs/*.yaml` and in `env/Dockerfile`.

### Gotcha found here: `prompt_logprobs` does not identify the realized token
- Not Blackwell-specific, but it silently corrupted a headline anchor. With
  `prompt_logprobs: 1` vLLM returns the top-1 token AND the realized token when
  they differ, so taking `max(logprob)` scores the model's own greedy path
  instead of the corpus. Measured error: perplexity **1.66 vs a true 5.37**.
- Fix: request `return_token_ids: true` and look the realized token up by id.
  See `harness/run_quality_eval.py::_selected_logprob`.

### NEW LANDMINE: FP8 checkpoint crashes at load via DeepGEMM on sm_120
- Environment: driver 580.159.03, CUDA 13.0, kernel 6.17.0-35-generic,
  torch 2.13.0+cu130, vLLM 0.27.1, Qwen3-8B-FP8 (e4m3, block 128x128).
- Symptom: server dies ~40 s into startup, during weight post-processing:
  ```
  RuntimeError: Assertion error
  (/workspace/.deps/deepgemm-src/csrc/apis/layout.hpp:60):
  Unknown SF transformation
  ```
  Raised from `deepgemm_post_process_fp8_weight_block` ->
  `transform_sf_into_required_layout`.
- Root cause: vLLM defaults `VLLM_USE_DEEP_GEMM=True`, so block-scaled FP8
  weights are routed through DeepGEMM. DeepGEMM's scale-factor layout
  transform does not handle this layout on sm_120 (consumer Blackwell); it
  targets the datacenter parts. Nothing about the checkpoint is wrong.
- Fix: `VLLM_USE_DEEP_GEMM=0`. vLLM then selects
  `CutlassFp8BlockScaledMMKernel for Fp8LinearMethod` -- the native FP8
  tensor-core path -- and the server starts normally.
- Cost: ~5 min. The traceback names DeepGEMM directly, which is the tell; the
  misleading part is that it looks like a bad checkpoint rather than a kernel
  selection default.
- CONTROL NOTE: set `VLLM_USE_DEEP_GEMM=0` for **every** arm, not just FP8, so
  the environment is identical across the comparison. It only affects FP8 GEMM
  selection, so it is a no-op for FP16 and AWQ.

### RESOLVED: FP8 native path confirmed (not emulated)
- `Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod` -- CUTLASS FP8
  block-scaled GEMM on FP8 tensor cores. Not emulated, not a Marlin fallback.
  The WSL2/dxgkrnl emulation artifact does not apply on native Linux, as
  suspected.
- Qwen3-8B-FP8 at `--gpu-memory-utilization 0.85 --max-model-len 8192`:
  10.29 GiB weights+non-torch, 1.24 GiB peak activation, 0.44 GiB CUDA graphs,
  leaving **15.1 GiB KV cache = 109,968 tokens**.
- Output is coherent across several prompts; no garbage characters observed.
- Informal single-request throughput ~140 tok/s (N=3, unlocked clocks, NOT a
  measured result -- recorded only to show the path is not pathologically slow,
  which is how the emulated fallback presents).

### STILL UNVERIFIED on this box
Do not treat these as cleared -- they were not exercised:
- **Garbage-character output.** Not observed in AWQ or FP8 smoke tests, but the
  sample is far too small to call it refuted.
- **Tailscale / CUDA init at boot.** Not tested.
- **P2P deadlocks / `NCCL_DMABUF_ENABLE`.** Single-GPU study; not applicable.

---

## Known-in-the-wild landmines to verify on your box

These are documented by others on 5090/sm_120. Confirm or refute each on your
hardware and record the result — a confirmed-or-refuted list on *your* stack is
more useful than repeating hearsay.

### Vanilla vLLM install rejects sm_120
- Symptom: `CUDA capability sm_120 is not compatible with the current PyTorch
  installation ... no kernel image is available for execution on the device`.
- Root cause: stock torch wheels don't ship Blackwell kernels.
- Fix (to verify): torch cu128 build + vLLM ≥ 0.17.0. Record exact pins in
  `env/requirements.lock`.

### Tailscale interfering with CUDA init at boot
- Symptom: intermittent CUDA-graph capture crash when a service races for the
  GPU at boot.
- Root cause: Tailscale intercepts the network at boot and interferes with CUDA
  initialization (documented on 5090 + WSL2 setups).
- Note: verify whether this reproduces on a native-Linux node or is
  WSL2-specific.

### flash-attn symbol errors on kernel 6.14
- Symptom: flash-attn import/symbol errors under Linux kernel 6.14 on Blackwell.
- Fix (to verify): FlashInfer backend as a Flash-Attn bypass
  (`--attention-backend flashinfer`).

### FP8 falling back to emulated (slow) path
- Symptom: FP8 markedly *slower* than AWQ despite native FP8 tensor cores.
- Root cause: on WSL2, native FP8 isn't exposed through dxgkrnl → emulated path.
- Note: this is a WSL2 artifact. On **native Linux** the native FP8 path should
  be available. Confirm in server logs which path is active — this matters
  because the whole FP8-native comparison depends on it being the real path.

### Garbage-character output on Blackwell
- Symptom: corrupted / garbage tokens in output.
- Fix (documented): AWQ 4-bit quantization resolves it in reported cases.

### P2P deadlocks / memory fragmentation (multi-GPU)
- Relevant if you ever span the 5090 + another card. `NCCL_DMABUF_ENABLE=1`
  (kernel 6.14 native memory handling) is a documented workaround.
