# Per-capability quantization degradation on Blackwell: a throughput–retention study

**Status:** first complete three-arm sweep. The serving and perplexity results are
final. The per-capability (APEX) result is **inconclusive and reported as such**;
Section 4.3 states why, and Section 6 states what would change it.

## Abstract

As quantization drops BF16 → FP8-native → AWQ-4bit on an RTX 5090 (sm_120),
throughput rises — but not monotonically, and not in the same order at every
concurrency. Measured under locked clocks with all other server flags held
identical, AWQ-4bit is **2.2× BF16 at concurrency 1** yet **FP8-native overtakes
it at concurrency 64** (3,176 vs 2,639 output tok/s), because the regime shifts
from weight-bound to compute-bound and native FP8 tensor cores beat
dequantize-then-compute. On quality, paired per-passage perplexity over 82,282
held-out tokens finds FP8 **indistinguishable from BF16** (ratio 1.0011,
95% CI [0.9997, 1.0024]) while AWQ-4bit costs **6.0%** (ratio 1.0601, CI
[1.0548, 1.0657]). The decision-relevant statement for a serving team is
therefore: *FP8-native buys 44% more throughput at high concurrency for no
measurable quality cost.* The intended headline — whether the three APEX
capability dimensions degrade non-uniformly — could not be answered on this
model: only one of three dimensions exhibits a positional curve at BF16, and
bootstrapped curve-strength intervals overlap across all three schemes.

## 1. Motivation

A serving team choosing a quantization scheme needs more than "quality dropped
X%". It needs to know *which* capability drops and at *what* precision, because
an agent workload that depends on instruction-following has a different
tolerance than a RAG workload that depends on recall.

Published vLLM benchmarks report throughput, and sometimes a single aggregate
quality number. None report a per-capability degradation profile. This study set
out to produce one, using [APEX](https://github.com/vshortt73/apex)'s three
independent dimensions — factual recall, instruction following, salience — as
the quality instrument, anchored by task accuracy and held-out perplexity.

It delivers the throughput–quality tradeoff with tight intervals. It does not
deliver the per-capability profile, for reasons that are themselves a result
(Section 4.3).

## 2. Background

**Blackwell sm_120 serving.** The RTX 5090 is consumer Blackwell. As of
torch 2.13.0+cu130 and vLLM 0.27.1, a stock `pip install vllm` produces a
working sm_120 stack with no special index or source build — the early-window
claim that vanilla installs reject sm_120 does not reproduce. Native FP8 is
available on Linux via `CutlassFp8BlockScaledMMKernel`; the emulated fallback is
a WSL2/dxgkrnl artifact and does not apply. Both were verified rather than
assumed, and both are recorded in `LANDMINES.md` alongside two failures that do
still bite (Section 3).

**APEX** measures position-influence curves: a probe is embedded at a controlled
position within a filled context, and a downstream query measures whether the
probe influenced the answer. It reports per-position scores; it does not report
whether a curve *exists*. That verdict is derived here (Section 3).

**Positional salience.** The U-shaped "lost in the middle" effect (Liu et al.,
2024) is the canonical positional phenomenon this instrument is built to detect,
and it is what the one measurable dimension in this study exhibits.

## 3. Method

Full pre-registration in `METHODOLOGY.md`; only the parts that materially shaped
the result are restated here.

**Experimental matrix.**

|                       |                                                                   |
|---                    |---                                                                |
| Independent variable  | quantization scheme: BF16 / FP8-native / AWQ-4bit                 |
| Model                 | Qwen3-8B, three official checkpoints from the same base weights   |
| Controlled            | GPU, locked SM clock (set 2400 MHz, observed 2377-2392),<br>power limit (500 W), `gpu_memory_utilization` 0.85,<br>`max_model_len` 12288,<br>attention backend,<br>DeepGEMM off, `ignore_eos`,<br>greedy decoding,<br>prompt set,<br>probe set, evaluator                                              |
| Dependent             | output-token and request throughput; TTFT/TPOT/E2E at p50/p90/p99;<br>energy;<br>task accuracy;<br>held-out perplexity;<br>per-dimension curve existence and strength                        |

The full-precision arm is **bfloat16**, the checkpoint's native dtype. Forcing
float16 on a bf16-trained model would degrade the reference point every other
arm is measured against.

**Model size is hardware-bound.** A three-arm sweep on one 32 GiB card caps the
model at ~10B: the BF16 arm alone needs 15.3 GiB. This constraint is load-bearing
for the negative result in Section 4.3.

**Two configuration errors would have silently corrupted the result**, and are
recorded in `LANDMINES.md` because neither announces itself:

1. *DeepGEMM on sm_120.* vLLM defaults `VLLM_USE_DEEP_GEMM=True`, and its
   scale-factor layout transform aborts loading block-scaled FP8 weights
   ("Unknown SF transformation"). `VLLM_USE_DEEP_GEMM=0` is required, and is set
   for **every** arm so the environment stays identical.
2. *Qwen3 reasoning tokens.* Qwen3 emits `<think>` blocks by default and APEX's
   scorers strip only whitespace, so programmatic probes score the reasoning text
   rather than the answer — measured at 0.332 against 0.983 for exact-match.
   Both `--reasoning-parser qwen3` and `no_think: true` are required; neither
   suffices alone. Dangerously, a quantized model may emit a *different amount*
   of reasoning, so the artefact would read as damage to instruction-following.

**Curve existence is derived, not reported by APEX.** A dimension has an
exploitable curve if position explains variance in score beyond chance, *after
removing probe difficulty*:

- `curve_strength` = within-probe eta-squared (each probe's mean subtracted,
  then SS_between_positions / SS_total on the residuals)
- `curve_exists` = seeded permutation test on that statistic, p < 0.05

Eta-squared rather than a correlation because positional effects are frequently
**non-monotonic** — a U-shape scores ~0 on Pearson. Permutation rather than
ANOVA's F because scores are bounded in [0,1] and far from normal. Within-probe
rather than pooled because, measured across 11,189 rows of prior APEX history,
probe identity explains **0.630** of score variance against position's **0.031**
— pooling drowns the signal in difficulty differences. Re-analysed within-probe,
significant cells in that history rise from 1/35 to 7/34.

Repetition explains **0.000** of variance, confirming greedy decoding is exactly
deterministic. Repeats therefore carry no information about uncertainty for
quality metrics; intervals come from bootstrapping over passages (perplexity) or
probes (curve strength).

**Probe-set calibration.** Probe difficulty is relative to the model under test.
`harness/check_probes.py` measured all 60 seed probes against BF16 and found only
**28 usable**: 14 factual probes at ceiling, 14 salience probes at floor. The
sweep runs the usable subset; the excluded 32 cannot register a result and would
roughly double runtime.

**Evaluator.** Rubric-scored probes (all salience, 15/20 application) are graded
by Qwen2.5-14B-Instruct-Q4_K_M on a separate host — never the model under test,
which would make the judge degrade in lockstep with its subject. It was
fingerprinted before and after the sweep; both reads were `98f010b1ff545412`, so
the judge did not move.

## 4. Results

Environment: RTX 5090, driver 580.159.03, CUDA 13.0, torch 2.13.0+cu130,
vLLM 0.27.1, flashinfer 0.6.16.post3, sm_120, clocks locked at 2400 MHz (observed 2377-2392) / 500 W.
All results stamped with this fingerprint and a clean git commit.

### 4.1 Throughput and latency

Output-token throughput (tok/s), 200 requests per level, 512-in/256-out,
`ignore_eos` forcing exactly 256 decoded tokens per request in every arm:

| scheme      |     c=1   |     c=4   |    c=16     |    c=64     |
|-------------|-----------|-----------|-------------|-------------|
| BF16        | 90.8      | 323.1     | 1032.6      | 2199.9      |
| FP8-native  | 125.9     |  474.0    | 1497.8      | **3175.5**  |
| AWQ-4bit    | **202.1** | **709.2** | **1821.6**  | 2638.8      |    

**The ordering inverts.** AWQ leads by 2.2× at concurrency 1 and still leads at
16, but FP8 overtakes it at 64 by 20.3%. At low concurrency the workload is
weight-bound and 4-bit weights win on memory traffic; at high concurrency it
turns compute-bound, and native FP8 tensor cores beat dequantize-then-compute.

Latency and energy at the extremes:

| scheme      | TTFT p50 (c=1)  | TTFT p99 (c=64) | TPOT p50 (c=64) | tok/J (c=64) |
|-------------|-----------------|-----------------|-----------------|--------------|
| BF16        | 50.5 ms         | 2420 ms         | 23.76 ms        | 5.52         |
| FP8-native  | **34.2 ms**     | **1501 ms**     | **16.85 ms**    | **8.16**     |
| AWQ-4bit    | 49.7 ms         | 2490 ms         | 21.00 ms        | 7.12         |

FP8 takes best time-to-first-token at every level, best tail latency under load
by a wide margin (1.5 s vs ~2.4–2.5 s at p99), and best energy efficiency.

### 4.2 External quality anchors

Held-out perplexity over 198 private, unpublished passages (82,282 scored
tokens), compared **paired per passage** — every arm scores the identical token
sequence, so passage difficulty cancels:

| comparison  | perplexity      | ratio  | 95% CI           | verdict                       |
|-------------|-----------------|--------|------------------|-------------------------------|
| FP8 vs BF16 | 36.451 → 36.490 | 1.0011 | [0.9997, 1.0024] | not distinguishable from zero |
| AWQ vs BF16 | 36.451 → 38.642 | 1.0601 | [1.0548, 1.0657] | **significant**               |

Pairing is what makes this resolvable. On the same data the unpaired interval is
[−0.0667, +0.1833] nats/token — it spans zero and would report *no detectable
difference* for a real 6% degradation.

The corpus scores 38.6 perplexity against 11.5 for formulaic text, which is
direct evidence it is genuinely unseen rather than recalled.

**Task accuracy could not resolve either comparison.** BF16 and FP8 answered all
50 items identically (0 discordant pairs); AWQ produced 1 regression (p = 1,
McNemar exact). At n=50 with a 0.92 baseline only ~3 items sit near the decision
boundary, so flips are vanishingly rare. This was predicted by the task
validator before the run, not discovered after it.

### 4.3 APEX per-dimension retention

This is the section the study was built for, and it does not deliver a finding.

Curve verdicts, within-probe permutation test:

| dimension             | BF16                  | FP8-native          | AWQ-4bit            |
|-----------------------|-----------------------|---------------------|---------------------|
| factual_recall        | 0.112 (p=0.130)       | 0.067 (p=0.595)     | 0.141 (p=0.034)     |
| instruction_following | **0.085 (p=0.0002)**  | **0.063 (p=0.010)** | **0.069 (p=0.004)** |
| salience              | 0.097 (p=0.231)       | 0.048 (p=0.834)     | 0.061 (p=0.682)     |

**Only instruction_following has a curve at BF16.** Per the pre-registered
outcome, a dimension with no curve at full precision is a property of the model
and probe set, not of quantization, and is excluded from the degradation claim
rather than reported as a collapse. That removes factual_recall and salience.

The curve that does exist is a clean **lost-in-the-middle U**, reproduced in all
three arms — mean score by position, instruction_following:

| position  | 0.02  | 0.20  | 0.50  | 0.80  | 0.98  |
|-----------|-------|-------|-------|-------|-------|
| BF16      | 0.860 | 0.725 | 0.710 | 0.796 | 0.815 |
| FP8       | 0.807 | 0.783 | 0.774 | 0.792 | 0.813 |
| AWQ       | 0.807 | 0.697 | 0.727 | 0.732 | 0.779 |

Curve strength appears to fall from 0.085 to 0.063 under FP8. **It is not
distinguishable from noise.** Bootstrapping over probes — the independent unit —
gives heavily overlapping intervals:

| dimension             | BF16           | FP8            | AWQ            |
|-----------------------|----------------|----------------|----------------|
| factual_recall        | [0.080, 0.396] | [0.054, 0.302] | [0.080, 0.445] |
| instruction_following | [0.048, 0.195] | [0.041, 0.152] | [0.045, 0.171] |
| salience              | [0.080, 0.308] | [0.035, 0.295] | [0.061, 0.301] |

The `factual_recall` cell that clears threshold under AWQ (p=0.034) should not be
read as a finding: nine tests were run without correction, ~0.45 false positives
are expected at α=0.05, and it does not survive Bonferroni. Reading it as
"quantization created a curve" would be exactly the kind of noise-mining the
pre-registration exists to prevent.

**With one measurable dimension and intervals this wide, H1 cannot be tested.**
Not rejected — untested. The instrument works; the measurement is underpowered.

## 5. Discussion

**For a serving team, the recommendation is unambiguous.** FP8-native is the
default choice on Blackwell: 44% more throughput than BF16 at concurrency 64,
best TTFT and tail latency at every level, best energy efficiency, and no
measurable quality cost against an 82k-token held-out corpus. AWQ-4bit is the
right choice only for low-concurrency, latency-insensitive, memory-constrained
deployments where its 2.2× single-stream advantage matters more than 6%
perplexity.

**The regime inversion is the transferable methodological point.** A benchmark
publishing one throughput number would name AWQ or FP8 the winner purely
according to which concurrency it happened to sample. Reporting single-stream
and concurrent regimes separately is not fastidiousness; it is the difference
between a correct and an incorrect recommendation.

**Why the per-capability claim failed is itself informative.** It was not a
tooling failure — the detector found a textbook U as soon as it was given a
dimension in range. It was a *calibration* failure, in two directions at once:
factual probes are too easy for an 8B (14/20 at ceiling) and salience probes too
subtle to move one (14/20 at floor). Probe sets have a difficulty, that difficulty
is relative to the model, and a probe set validated on ~30B models does not
transfer down. The models that showed salience curves in prior APEX history were
Qwen3-32B and qwen3-30b-a3b, both scoring mid-range where there is room to move.

That produces a hard constraint: curve detection wants mid-range baseline scores,
which on this probe set means ~30B, while a three-arm precision sweep on one
32 GiB card caps the model at ~10B. **Those two requirements cannot both be
satisfied on this hardware.** Resolving it requires either recalibrated probes
for the 8B class, or a larger card.

## 6. Limitations

- **One model, one size, one family.** Qwen3-8B only. Cross-family
  generalization is not attempted.
- **The per-capability result is underpowered, not negative.** Six usable probes
  in two dimensions produce intervals too wide to compare across arms.
- **Two dimensions had no curve at baseline** and are excluded by
  pre-registration, leaving no basis to test non-uniformity.
- **Task accuracy is underpowered at n=50**, with 0–1 discordant pairs.
- **Single context length** (8,192) for the reported sweep. Prior history shows
  positional effects strengthening at 16k+, which was not sampled here.
- **The evaluator is an LLM** and enters the measurement chain. It was pinned,
  greedy, hosted separately, and fingerprinted identical before and after — a
  stable bias cancels in the deltas — but it is not a neutral instrument.
- **vLLM only**, one serving stack.
- **Publishing the task set eventually contaminates it**, as happened to GSM8K
  and MMLU. Reusers should re-author rather than inherit.

## 7. Future work

In descending order of expected payoff:

1. **Author replacements for the 32 degenerate probes** — harder factual probes,
   stronger-signal salience probes, using the 28 survivors as templates. This
   directly narrows the intervals that made Section 4.3 inconclusive.
2. **Scale the task set to 300–1000 items**, so McNemar has discordant pairs to
   work with.
3. **Extend to 16k–32k context**, where prior history shows positional effects
   are strongest.
4. **Repeat on a ~30B model** where all three dimensions sit in range, accepting
   a two-arm (FP8 vs AWQ) comparison since a 30B BF16 arm does not fit 32 GiB.
5. Cross-family (Gemma, Llama); TensorRT-LLM cross-check; speculative decoding
   interaction.

## Reproducibility

Everything needed to reproduce is committed: `env/requirements.lock` (203 pins
from the validated stack), `env/Dockerfile`, `env/gpu_setup.sh` (clock lock),
`configs/*.yaml` (controlled variables, one file per scheme),
`env/run_sweep.sh` (the exact unattended sweep that produced these numbers), and
`results/raw/*.json` (every result with a full environment fingerprint and the
originating git commit). `data/prompts_512in.txt` and `data/tasks.jsonl` are
committed; the held-out corpus is not, because publishing it would destroy the
property that makes it a valid perplexity anchor.

Regenerate every figure and table from raw with `python results/analysis.py`.

### References

- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., &
  Liang, P. (2024). Lost in the middle: How language models use long contexts.
  *Transactions of the Association for Computational Linguistics, 12*, 157–173.
- Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2024). AWQ:
  Activation-aware weight quantization for LLM compression and acceleration.
  *Proceedings of Machine Learning and Systems, 6*.
- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E.,
  Zhang, H., & Stoica, I. (2023). Efficient memory management for large language
  model serving with PagedAttention. *SOSP '23*.
- Shortt, V. N. (n.d.). *APEX: Attention Profiling and Empirical Cross-model
  Optimization* [Computer software]. https://github.com/vshortt73/apex
