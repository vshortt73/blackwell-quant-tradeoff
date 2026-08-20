# blackwell-quant-tradeoff

**Per-capability quantization degradation on NVIDIA Blackwell (RTX 5090, sm_120).**

As quantization drops BF16 → FP8-native → AWQ-4bit, throughput rises. This study
measures whether model quality degrades *uniformly* across capabilities or
*differentially* — using [APEX](https://github.com/vshortt73/apex)'s three
independent dimensions (factual recall, instruction following, salience) as the
quality instrument, anchored by task accuracy and perplexity.

The finding a serving team actually needs is not "quality dropped X%." It's
*which* capability drops, and at *what* precision. That per-capability profile is
what nothing in the vLLM-benchmark literature currently reports.

> **Status:** first complete three-arm sweep done; see
> [`writeup/tradeoff_study.md`](writeup/tradeoff_study.md). Serving and
> perplexity results are final. The per-capability (APEX) result is
> **inconclusive** — only one of three dimensions has a positional curve at
> BF16, and curve-strength intervals overlap across schemes. Every number here
> comes from `results/raw/`, each stamped with a full environment fingerprint
> and its originating commit.
>
> **Model under test:** Qwen3-8B (dense 8B), three official checkpoints —
> BF16 / FP8 (e4m3) / AWQ-4bit. Sized so all three arms fit one 32 GiB card with
> KV headroom; see `configs/`.

## Headline

**FP8-native buys 44% more throughput at concurrency 64 for no measurable
quality cost.** AWQ-4bit is 2.2x faster than BF16 single-stream but FP8
overtakes it under load, and AWQ costs 6.0% perplexity.

| output tok/s | c=1 | c=4 | c=16 | c=64 | perplexity vs BF16 |
|---|---|---|---|---|---|
| BF16 | 90.8 | 323.1 | 1032.6 | 2199.9 | — |
| FP8-native | 125.9 | 474.0 | 1497.8 | **3175.5** | 1.0011 [0.9997, 1.0024] n.s. |
| AWQ-4bit | **202.1** | **709.2** | **1821.6** | 2638.8 | 1.0601 [1.0548, 1.0657] **sig.** |

The ordering **inverts** with concurrency: 4-bit wins while weight-bound, native
FP8 wins once compute-bound. A benchmark reporting one throughput number would
name a different winner depending on which regime it sampled.

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

## Repro

```bash
# 1. environment -- requirements.lock is a full pip freeze of the VALIDATED
#    sm_120 stack (torch 2.13.0+cu130 / vLLM 0.27.1). Use a venv: vLLM pins an
#    exact torch and will otherwise replace whatever you already have.
python3 -m venv .venv && .venv/bin/pip install -r env/requirements.lock

# 2. lock the GPU into a fixed state
sudo ./env/gpu_setup.sh 0 2400 500        # <gpu> <sm_clock_mhz> <power_w>

# 3. start vLLM for a scheme. Quantization is auto-detected from the
#    checkpoint; the attention backend is NOT, so pin it (controlled variable).
#    Exact per-scheme command is in the launch comment of each configs/*.yaml.
VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_DEEP_GEMM=0 \
  .venv/bin/vllm serve /path/to/Qwen3-8B-FP8 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 \
    --port 8000 --served-model-name model-under-test

# 4. run the harness against that scheme (repeat per scheme).
#    env/run_sweep.sh does steps 3-6 for all three arms unattended; it is the
#    exact script that produced the committed results.
.venv/bin/python harness/run_serving_bench.py configs/fp8_native.yaml
.venv/bin/python harness/run_quality_eval.py  configs/fp8_native.yaml

# 5. APEX runs as a SEPARATE PASS, natively, with its own config. It supports
#    several models in one run, so one invocation can cover all three arms.
#    Name each model entry to match apex.model_id in the scheme configs, and
#    configure an EVALUATOR model -- without one, salience is unscored.
apex run <your-apex-config>.yaml

# 6. import APEX results into this study, once per scheme
.venv/bin/python harness/run_apex_eval.py configs/fp8_native.yaml

# 7. build the headline figure + tables
.venv/bin/python results/analysis.py
```

> **Why two passes.** The serving benchmark and APEX cannot share a run. APEX
> parallelises purely for speed, whereas in the serving benchmark *concurrency
> is the independent variable*; and correctness scoring is timing-insensitive
> while throughput measurement is timing-only and needs an exclusive GPU.
> Running them together would silently corrupt the timing half. This repo
> therefore never invokes APEX — it only reads what APEX produced
> (`harness/run_apex_eval.py`, which accepts a results database or an
> `apex export` JSON file).

## Layout

```
env/          pinned Dockerfile, requirements.lock, gpu_setup.sh (clock lock)
configs/      one YAML per scheme; controlled vars identical across all
harness/      serving bench, quality anchors, APEX importer, grader, fingerprinting
results/raw/  immutable per-run JSON (committed); analysis.py builds plots/tables
results/apex/ raw APEX databases + sweep logs (gitignored; regenerable)
writeup/      the study writeup
METHODOLOGY.md  pre-registration
LANDMINES.md    sm_120 serving field log
```

## Prerequisites for a run

All of these are satisfied in the committed sweep; they are documented because
getting any of them wrong silently corrupts the result rather than failing loudly.

1. **An APEX run (pass 1) configured for all three dimensions.** The importer
   is built (`harness/run_apex_eval.py`); pass 1 needs two things:

   - **An evaluator model.** Salience probes and 15 of 20 application probes
     are rubric-scored; without an evaluator they run but store NULL scores, so
     salience is lost entirely. It must **not** be the model under test — a
     quantized judge degrades in lockstep with its subject. Fingerprint it
     before and after the sweep:

     ```bash
     python harness/check_evaluator.py --base-url http://node2:8080
     ```

     A changed fingerprint means the judge moved mid-sweep, and the
     rubric-scored dimensions stop being comparable across arms.

   - **Reasoning suppression**, via `--reasoning-parser qwen3` on the vLLM
     launch *and* `no_think: true` on the APEX model entry. Neither works
     alone. Getting this wrong is not a small error: programmatic scores
     measured 0.332 against 0.983 for exact-match, because APEX's scorers see
     `<think>` text instead of the answer. See `LANDMINES.md`.

   - **A calibrated probe set.** Probe difficulty is relative to the model
     under test, and a dimension made mostly of probes that every model aces
     (or that move no model) cannot produce a curve however good the detector
     is. Run APEX over the whole probe set at two positions first, then:

     ```bash
     python harness/check_probes.py results/apex/probecal.db \
         --model qwen3-8b-bf16 --emit-select
     ```

     On Qwen3-8B this found **28 of 60 seed probes usable** — 14 factual
     probes at ceiling, 14 salience probes at floor. It emits a
     `probes.select` list so the sweep spends its time only on probes that can
     move. Skipping this step is how you spend hours characterising probes
     that cannot register a result.
2. **Data** — `data/tasks.jsonl` ships with the repo: 50 general-knowledge
   items authored for this study, stratified across 10 domains. One JSON object
   per line:

   ```jsonl
   {"prompt": "The capital of France is", "answer": "Paris"}
   {"prompt": "The largest US city is", "answer": "New York City", "aliases": ["NYC"]}
   ```

   Optional per item: `"aliases": [...]`, `"whole_response": true`.

   Three constraints shape the content:
   - **Stems, not questions.** These hit `/v1/completions` with no chat
     template, so `"What is the capital of France?"` invites the model to
     continue with *more questions*. `"The capital of France is"` does not.
   - **The answer must land in the first sentence, within 32 tokens**
     (`quality.max_tokens`), because the grader scopes to the first segment.
   - **Calibrate difficulty.** A set the baseline aces cannot detect
     degradation — every arm scores 1.0. Aim for a baseline of **0.70–0.85**,
     where the model sits at the edge of its competence and small numerical
     perturbations actually flip answers. Avoid public benchmark items
     (GSM8K/MMLU/TriviaQA); they are in the training data.

   Validate before spending GPU time:

   ```bash
   python harness/check_tasks.py data/tasks.jsonl
   python harness/check_tasks.py data/tasks.jsonl --probe --config configs/bf16_baseline.yaml
   ```

   `--probe` reports the difficulty calibration and warns on ceiling/floor.

   > **Privacy mechanism:** committed results carry only a hash, verdict and
   > deciding rule per item — never task text. Full prompts and responses go to
   > `results/audit/`, which is gitignored. *This* task set is general knowledge
   > and safe to publish, but the mechanism means you can swap in private
   > domain items without leaking them.

   > **Contamination, declared:** publishing a task set eventually poisons it —
   > future models may train on this file, exactly as happened to GSM8K and
   > MMLU. That is the accepted cost of letting a reader rerun the identical
   > items. Anyone reusing this harness years from now should re-author rather
   > than inherit these 50.

   The other two inputs are generated:
   - `prompt_file` — `harness/make_prompts.py` (deterministic, committed).
   - `corpus_file` — `harness/build_corpus.py`, which turns private
     unpublished prose into one-passage-per-line text. It strips markdown
     structure (format predictability is not language modelling), excludes
     model-written and third-party documents by filename, drops passages
     matching credential/PII patterns, deduplicates, and reports statistics
     only — never file contents. Output stays gitignored.

     ```bash
     python harness/build_corpus.py --root /path/to/private/docs \
         --out data/heldout_corpus.txt
     ```

     "Held-out" is the whole ballgame: public text is in the model's training
     data, and memorised text does not merely read low, it may degrade
     differently under quantization and so corrupt the delta itself. As a
     sanity check, genuinely unseen prose scored **38.6** perplexity here
     against **11.5** for formulaic text — the model is predicting, not
     recalling.

Done (kept here as a record of what was verified rather than assumed):

3. ~~**Version pins**~~ — `env/requirements.lock` now holds 203 exact pins
   captured by `pip freeze` from the validated venv.
4. ~~**Perplexity token selection**~~ — fixed. `prompt_logprobs` returns the
   top-1 token *and* the realized token when they differ, so the original
   `max(logprob)` scored the model's greedy path rather than the corpus
   (perplexity 1.66 vs a true 5.37). Now resolved by id via
   `return_token_ids`. See `LANDMINES.md`.
5. ~~**Task grading**~~ — exact match scored 0.0 for every scheme (the model
   answers correctly, then elaborates), making the anchor blind to degradation.
   Replaced by `harness/grader.py`: deterministic goal-satisfaction grading —
   first-segment scoping, word-boundary matching, numeric tolerance, aliases,
   negation guard. Deterministic by design so grader variance cannot
   contaminate the BF16→FP8→AWQ delta. Tests: `harness/test_grader.py`.

## Status and next steps

**Done.** The three-scheme sweep ran on the 5090 under locked clocks; serving,
latency, energy and paired perplexity results are final and written up in
[`writeup/tradeoff_study.md`](writeup/tradeoff_study.md). Every sm_120 landmine
encountered is recorded in [`LANDMINES.md`](LANDMINES.md), including two that
would have silently corrupted results (DeepGEMM aborting FP8 weight loading, and
Qwen3 reasoning tokens reaching APEX's scorers).

**Inconclusive.** The per-capability claim — the reason this study exists — was
not answered. Only `instruction_following` has a positional curve at BF16, and
bootstrapped curve-strength intervals overlap across all three schemes. H1 is
untested, not rejected. Section 4.3 of the writeup gives the numbers; the cause
is probe-set calibration, diagnosed by `harness/check_probes.py`.

**Next, in payoff order.**

1. Author replacements for the 32 degenerate probes (14 factual at ceiling, 14
   salience at floor) using the 28 survivors as templates. This is what narrows
   the intervals that made the headline inconclusive.
2. Scale `data/tasks.jsonl` to 300–1000 items so McNemar has discordant pairs.
3. Extend to 16k–32k context, where prior APEX history shows positional effects
   are strongest.
4. Repeat on a ~30B model where all three dimensions sit in range — necessarily
   a two-arm (FP8 vs AWQ) comparison, since a 30B BF16 arm does not fit 32 GiB.
5. *(optional)* Cross-family (Gemma, Llama); TensorRT-LLM cross-check.

The 4080 Super (sm_89) baseline in the original plan was skipped: the sm_120
stack worked without it, so it would have cost time without removing ambiguity.
