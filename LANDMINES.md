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
