# Methodology (pre-registered)

This document is written **before** the runs and is not edited to fit the
results. It states what is measured, what is held constant, and what would
count as each possible outcome. Backfilling a methodology after seeing the
numbers is the most common way these studies lose credibility; pre-registration
is the cheapest way to earn it back.

## Question

On NVIDIA Blackwell (RTX 5090, sm_120), as quantization precision drops
BF16 → FP8-native → AWQ-4bit, throughput rises. **Does model quality degrade
uniformly across capabilities, or differentially?** Specifically, do the three
APEX dimensions — factual recall, instruction following, and salience — cross
from an exploitable, structured curve into noise at the *same* precision, or at
*different* precisions?

Hypothesis (H1): degradation is **non-uniform**. At least one dimension retains
an exploitable curve at a precision where at least one other has collapsed.

Null (H0): all three dimensions degrade together; no separation in the
precision at which `curve_exists` flips to false.

## Design

- **Independent variable:** quantization scheme (BF16, FP8-native, AWQ-4bit).
  The full-precision arm is **bfloat16** — the checkpoint's native dtype.
  Forcing float16 on a bf16-trained model would degrade the baseline itself.
- **Held constant across every run:** model family (one — Qwen), GPU, locked SM
  clock, fixed power limit, `--gpu-memory-utilization`, `--max-model-len`,
  KV-cache space, input-length distribution, decoding (greedy for quality),
  and prompt set.
- **Dependent variables:**
  - *Throughput:* request throughput (req/s) and **output-token** throughput
    (tok/s), across a concurrency sweep {1, 4, 16, 64}. Single-request and
    concurrent regimes reported separately — never conflated.
  - *Latency:* TTFT, TPOT/ITL, end-to-end, each at **p50/p90/p99** (means are
    forbidden — tail latency is what production buys).
  - *Quality (anchors):* task accuracy on a controlled held-out set;
    held-out-corpus perplexity. Both greedy-decoded.
  - *Quality (mechanistic):* APEX per-dimension `curve_exists` and
    `curve_strength` for {factual_recall, instruction_following, salience},
    plus the refusal / failure / over-application breakdown.

## Controls (the part that makes numbers defensible)

1. **Locked clocks + fixed power limit** (`env/gpu_setup.sh`) so thermal
   throttling cannot inject variance mid-sweep.
2. **Same base weights** for every scheme — FP8 and AWQ derived from the BF16
   checkpoint, not different checkpoints.
3. **Only the quant scheme varies** within a comparison; every other server
   flag is identical and recorded.
4. **Greedy decoding (T=0)** for all quality measurement, so a delta is
   attributable to the scheme, not to sampling RNG.
5. **N ≥ 5 repeats** per config for *timing* metrics (throughput, TTFT,
   TPOT, e2e); report mean ± confidence interval, not a single pass. Variance
   is itself a reported quantity.

   **Quality metrics need a different uncertainty model.** Perplexity and task
   accuracy are deterministic here — greedy decoding, fixed corpus, fixed task
   set — so repeating a run reproduces the same number to within kernel
   reduction-order jitter. Reporting a CI over repeats would be tight and
   meaningless. The real question is *corpus* sampling: would a different
   held-out set have given a different answer? That is estimated by
   bootstrapping over **passages**, not over runs (tokens within a passage are
   strongly correlated, so resampling tokens would understate the interval).

   Perplexity is further compared **paired**: every scheme scores the identical
   token sequence, so passage difficulty — the dominant variance term — cancels
   exactly in the per-passage difference.

   This is not a marginal refinement. Measured on 198 passages / 82,282 tokens
   of held-out prose, AWQ-4bit vs the bf16 baseline: the paired interval is
   `[+0.0534, +0.0636]` nats/token and resolves a real 6.0% perplexity
   degradation, while the unpaired interval `[-0.0667, +0.1833]` spans zero and
   would have reported **no detectable difference**. Same data, same runs — 24x
   tighter. See `harness/paired.py`.
6. **Environment fingerprint** (driver, CUDA, vLLM/torch/flashinfer versions,
   locked clock, observed temp + power, git commit) stamped into **every**
   result JSON. A dirty git tree triggers a loud warning.
7. **Warmup discarded** — CUDA-graph capture and cold KV cache distort the
   first iterations badly on Blackwell.

## Scope discipline (declared limits)

- **One model family** for the core result. Cross-family generalization
  (Gemma, Llama) is declared **future work**, not attempted here. A clean
  single-family result beats a muddy multi-family one.
- **vLLM only** for the headline. A TensorRT-LLM cross-check is optional future
  work in a separate artifact.
- FP8 must be confirmed as the **native** Blackwell path in server logs, not the
  emulated fallback (a WSL2/dxgkrnl artifact that does not apply on native
  Linux).

## Measured knowledge cutoff (Qwen3-8B)

The task-accuracy anchor needs items the baseline can plausibly get wrong, and
on this model the only reliable source of difficulty is facts postdating its
training. That boundary therefore has to be **measured, not assumed**.

**There is no official cutoff from the Qwen team.** The model card states none,
and the QwenLM discussion asking for one
(github.com/QwenLM/Qwen3/discussions/1093) was never answered by a maintainer.
The "January 2025" figure in circulation comes from third-party aggregator
sites, not Alibaba. Self-report is worthless: asked three ways, the model
answered "July 2024", "2024", and "2023".

Measured directly against BF16 (greedy, `/v1/completions`):

| date | probe | result |
|---|---|---|
| 2024-10 | 2024 Physics Nobel for neural networks | **knows** — names Hinton |
| 2024-11 | 2024 US election winner; Republican VP nominee | **knows** — Trump, JD Vance |
| 2024-12-08 | Assad removed from power in Syria | **fails** — *"was not removed"*, says 2020 |
| 2024-12-03 | South Korea martial law | year guessable, details fabricated (invents cyberattacks) |
| 2025-01 | 47th US President | **fails** — hallucinates a film character |
| 2025-01 | DeepSeek R1 | **fails** — answers "SenseTime" |
| 2025-01 | NVIDIA CES 2025 architecture | **fails** — answers "Ada Lovelace" |
| 2025-04 | Qwen3's own release year | **fails** — answers 2024 |

**Effective cutoff: approximately November 2024.** December 2024 is already a
grey zone where the model produces plausible years with confabulated detail —
the characteristic signature of the boundary.

Working rules for authoring items:
- Facts dated **2025 or later are reliably outside** the model's knowledge.
- December 2024 is unusable: right-for-the-wrong-reason answers are
  indistinguishable from knowledge.
- Coverage is **not uniform by domain**. It failed CES 2025 GPU facts while
  handling contemporaneous general news, so hardware/software specifics thin
  out earlier than general events.
- A post-cutoff item measures *absence of knowledge*, which is stable under
  quantization in a way genuine recall is not. Such items make the baseline
  look harder without adding sensitivity — see `harness/check_tasks.py` on why
  chasing a low baseline with post-cutoff trivia is a trap.

## What each outcome means

- **Curves separate** (H1 supported): the headline finding. Report the precise
  precision at which each dimension crosses to `curve_exists == false`. This is
  the decision-relevant statement for a serving team ("AWQ-4bit keeps recall but
  breaks instruction-following").
- **Curves move together** (H0 not rejected): still a publishable negative —
  "on Blackwell/Qwen, quantization degrades these three capabilities in
  lockstep down to X-bit" — provided the controls above hold.
- **No curve exists even at BF16** for a dimension: a property of the
  model/probe set, not the quantization; note it and exclude that dimension
  from the degradation claim.
