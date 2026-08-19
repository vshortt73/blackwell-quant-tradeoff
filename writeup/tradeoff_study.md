# Per-capability quantization degradation on Blackwell: a throughput–retention study

> Skeleton. Section intents are noted in blockquotes. No numbers are filled —
> they come from `results/analysis.py` after the runs. Delete the blockquotes
> before publishing.

## Abstract
> 150–200 words. State the question (uniform vs differential degradation across
> factual recall / instruction following / salience), the method (vLLM on
> sm_120, three schemes, controlled), and the one-sentence finding. Lead with
> the finding, not the setup.

## 1. Motivation
> Why per-capability matters: a serving team deciding whether to ship a
> quantized model needs to know *which* capability degrades and at *what*
> precision, not an aggregate "quality dropped 4%". Position against the gap:
> vLLM benchmarks report throughput and maybe one aggregate quality number;
> none report a per-capability degradation profile.

## 2. Background
> - Blackwell sm_120 serving state (vLLM ≥ 0.17.0, native FP8 on Linux).
> - APEX in one paragraph: three-dimension position-influence characterization,
>   curve-existence detector, refusal/failure/over-application distinction.
>   Cite your published APEX work (github.com/vshortt73/apex) and the prior
>   Q6-floor finding this study extends onto the serving stack.
> - Positional-salience literature anchor (Liu et al., 2024, lost-in-the-middle)
>   to ground APEX's within-prompt positional measurement in accepted work.

## 3. Method
> Point to METHODOLOGY.md rather than restating it. Include the experimental
> matrix table (IV / controlled / DV) and the exact server launch command per
> scheme. State the input-length distribution explicitly.

## 4. Results

### 4.1 Throughput and latency
> Concurrency-sweep table: req/s and output-tok/s per scheme per level; TTFT /
> TPOT / E2E at p50/p90/p99. Report mean ± CI over N≥5.

### 4.2 External quality anchors
> Task accuracy and perplexity per scheme. These are the grounded signals.

### 4.3 APEX per-dimension retention (the contribution)
> The headline figure: throughput gain (x) vs curve strength (y), one series per
> dimension. Table of per-dimension `curve_exists` per scheme. **Call out
> precisely where the dimensions separate** — the precision at which each
> crosses to noise. This is the paragraph the whole artifact exists for.

## 5. Discussion
> Interpret separation (or lack of it). If differential: which capability is
> most quantization-fragile on Qwen/Blackwell, and the practical implication
> (e.g., "instruction-heavy agent workloads should not run below X-bit even
> though recall-heavy RAG tolerates it"). Tie back to the APEX
> refusal/failure/over-application breakdown for mechanism.

## 6. Limitations
> One model family; one GPU; vLLM only; probe-set dependence of APEX; greedy
> decoding only. State them plainly — declared limits read as rigor, not
> weakness.

## 7. Future work
> Cross-family (Gemma/Llama), TensorRT-LLM cross-check, SGLang, larger
> concurrency, speculative decoding interaction.

## Reproducibility
> Link the pinned env, the config YAMLs, the fingerprinted raw results, and the
> exact commit. Anyone should be able to rerun from `env/` + `configs/`.

---

### References
> APA. At minimum: your APEX repo; Liu et al. (2024) lost-in-the-middle; the
> vLLM Blackwell support references. Fill from your actual citations.
