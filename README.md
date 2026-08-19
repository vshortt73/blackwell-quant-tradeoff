# blackwell-quant-tradeoff

**Per-capability quantization degradation on NVIDIA Blackwell (RTX 5090, sm_120).**

As quantization drops FP16 → FP8-native → AWQ-4bit, throughput rises. This study
measures whether model quality degrades *uniformly* across capabilities or
*differentially* — using [APEX](https://github.com/vshortt73/apex)'s three
independent dimensions (factual recall, instruction following, salience) as the
quality instrument, anchored by task accuracy and perplexity.

The finding a serving team actually needs is not "quality dropped X%." It's
*which* capability drops, and at *what* precision. That per-capability profile is
what nothing in the vLLM-benchmark literature currently reports.

> **Status:** methodology and harness complete; results pending runs. No numbers
> in this repo are fabricated — `results/raw/` fills only when you run the
> harness. Headline figure regenerates from raw via `results/analysis.py`.

## Headline

_(populated after runs — throughput gain vs per-dimension retention; where the
three dimension curves separate is the result.)_

![headline](plots/headline_tradeoff.png)

## Why this design is defensible

- **Pre-registered** methodology (`METHODOLOGY.md`) — written before the runs.
- **Locked clocks + fixed power** (`env/gpu_setup.sh`) so no number is thermally
  throttled mid-sweep.
- **Same base weights** across schemes; **only** the quant scheme varies.
- **Greedy decoding** for all quality — deltas attributable to the scheme, not
  RNG.
- **N≥5 repeats, mean ± CI**; **environment fingerprint** stamped into every
  result JSON; dirty git tree warns loudly.
- **Percentile latency** (p50/p90/p99), throughput split into request vs
  **output-token**, single vs concurrent regimes never conflated.

## Repro (five commands)

```bash
# 1. environment (fill env/requirements.lock with your working cu128 pins first)
pip install -r env/requirements.lock

# 2. lock the GPU into a fixed state
sudo ./env/gpu_setup.sh 0 2400 500        # <gpu> <sm_clock_mhz> <power_w>

# 3. start vLLM for a scheme (see the launch comment in each configs/*.yaml)
vllm serve <model> --quantization fp8 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --port 8000 --served-model-name model-under-test

# 4. run the harness against that scheme (repeat per scheme)
python harness/run_serving_bench.py configs/fp8_native.yaml
python harness/run_quality_eval.py  configs/fp8_native.yaml
python harness/run_apex_eval.py     configs/fp8_native.yaml   # after wiring APEX

# 5. build the headline figure + tables
python results/analysis.py
```

## Layout

```
env/          pinned Dockerfile, requirements.lock, gpu_setup.sh (clock lock)
configs/      one YAML per scheme; controlled vars identical across all
harness/      serving bench, quality anchors, APEX adapter, shared fingerprinting
results/raw/  immutable per-run JSON (committed); analysis.py builds plots/tables
writeup/      the differentiated study skeleton
METHODOLOGY.md  pre-registration
LANDMINES.md    sm_120 serving field log
```

## What you must wire before running

1. **APEX adapter** — `harness/run_apex_eval.py::run_apex()` raises
   `NotImplementedError` by design so it cannot emit fabricated quality data.
   Point it at your installed APEX (import or subprocess) against the vLLM
   OpenAI endpoint; return one `ApexDimensionResult` per dimension.
2. **Data** — a controlled `prompt_file` (fixed, declared input-length
   distribution), a `tasks.jsonl` you own, and a held-out `corpus_file`.
3. **Version pins** — fill `env/requirements.lock` from your working stack after
   the first clean sm_120 install.
4. **Perplexity token selection** — `run_quality_eval.py::_selected_logprob()`
   pulls prompt logprobs defensively; confirm the field layout against your
   exact vLLM version.

## Sequence (per the plan)

1. Baseline the harness on the **4080 Super (sm_89)** — fully supported, removes
   "is it my config or the hardware?" ambiguity.
2. Port to the **5090**, log every sm_120 landmine in `LANDMINES.md`.
3. Run the three-scheme tradeoff study; write it up.
4. *(optional)* TensorRT-LLM cross-check as a second, smaller artifact.
