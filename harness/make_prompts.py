"""
Deterministic prompt-set generator for the serving benchmark.

`prompt_file` is consumed ONLY by run_serving_bench.py. Its job is to hold the
input-length distribution FIXED and DECLARED across every quantization scheme --
prompt *content* barely moves throughput, but prompt *length* moves everything
(prefill cost, KV footprint, batching). APEX does not read this file; it uses
its own probe set.

Why synthetic rather than scraped text:
  - exact token-length control against the real tokenizer
  - no licensing question, so the prompt set can be committed and a reader can
    rerun the identical inputs
  - no benchmark-contamination concern
  - regenerable byte-identical from a seed

Determinism: all randomness comes from random.Random(seed). Same seed + same
tokenizer -> byte-identical file, on any machine.

Usage:
    python harness/make_prompts.py --config configs/fp16_baseline.yaml
    python harness/make_prompts.py --out data/prompts_512in.txt \\
        --tokenizer /path/to/Qwen3-8B --n 200 --target 512 --var 64 --seed 20260819
"""

from __future__ import annotations

import argparse
import random
import sys
import statistics
from pathlib import Path

# Neutral technical-register vocabulary. Combinatorially large enough that 200
# prompts of ~512 tokens do not visibly repeat, while staying content-neutral:
# we are measuring serving throughput, not comprehension.
SYSTEMS = [
    "the ingestion pipeline", "the scheduling layer", "the storage backend",
    "the replication controller", "the metrics collector", "the request router",
    "the cache tier", "the batch executor", "the index builder",
    "the validation stage", "the compaction worker", "the telemetry agent",
    "the configuration service", "the checkpoint manager", "the queue consumer",
]
PROPERTIES = [
    "throughput", "tail latency", "memory residency", "queue depth",
    "retry behaviour", "cache hit rate", "fragmentation", "startup cost",
    "backpressure response", "failure isolation", "batch occupancy",
    "scheduling fairness", "compaction cadence", "write amplification",
]
CONDITIONS = [
    "under sustained load", "during a rolling restart", "when the queue saturates",
    "after a cold start", "at peak concurrency", "when replicas diverge",
    "once the cache warms", "under partial network loss", "during compaction",
    "when the batch window closes", "at steady state", "after a failover",
]
ACTIONS = [
    "is measured across successive intervals", "degrades predictably",
    "is bounded by the slowest participant", "recovers without operator action",
    "is recorded in the run manifest", "remains within the declared envelope",
    "is sampled at a fixed cadence", "converges after a brief transient",
    "is attributed to the responsible component", "is reported per partition",
]
CONNECTORS = [
    "In practice,", "By design,", "Under these conditions,", "As a result,",
    "For this reason,", "In the observed configuration,", "Consequently,",
    "Notably,", "In steady operation,", "Across repeated trials,",
]
TEMPLATES = [
    "{c} {s} reports that {p} {a} {k}.",
    "{s} exposes {p} {k}, and the value {a}.",
    "{c} {p} of {s} {a} {k}.",
    "When {s} is observed {k}, its {p} {a}.",
    "{c} the operator can confirm that {p} of {s} {a} {k}.",
]


def _sentence(rng: random.Random) -> str:
    t = rng.choice(TEMPLATES)
    out = t.format(
        c=rng.choice(CONNECTORS), s=rng.choice(SYSTEMS),
        p=rng.choice(PROPERTIES), a=rng.choice(ACTIONS), k=rng.choice(CONDITIONS),
    )
    return out[:1].upper() + out[1:]


def build_prompt(rng: random.Random, tok, n_tokens: int) -> str:
    """Emit a prompt of EXACTLY n_tokens tokens under `tok`."""
    parts: list[str] = []
    # Overshoot, then trim at the token level so the count is exact rather than
    # approximate -- an approximate input length is an uncontrolled variable.
    while len(tok(" ".join(parts), add_special_tokens=False)["input_ids"]) < n_tokens + 40:
        parts.append(_sentence(rng))
    ids = tok(" ".join(parts), add_special_tokens=False)["input_ids"][:n_tokens]
    text = tok.decode(ids)
    # BPE round-trip can shift the count by a token; correct deterministically.
    for _ in range(8):
        got = len(tok(text, add_special_tokens=False)["input_ids"])
        if got == n_tokens:
            break
        if got > n_tokens:
            ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
            text = tok.decode(ids)
        else:
            text = text + " " + _sentence(rng)
            ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
            text = tok.decode(ids)
    return text.replace("\n", " ").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="read tokenizer/target/n from a configs/*.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokenizer", default=None, help="path to the FP16 checkpoint")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--target", type=int, default=512)
    ap.add_argument("--var", type=int, default=64, help="uniform +/- this many tokens")
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    out, tokenizer, n = args.out, args.tokenizer, args.n
    if args.config:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from common import load_config

        cfg = load_config(args.config)
        out = out or cfg["prompt_file"]
        tokenizer = tokenizer or cfg["checkpoint"]
        n = n or cfg["n_requests"]
    if not (out and tokenizer and n):
        raise SystemExit("need --config, or all of --out/--tokenizer/--n")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    rng = random.Random(args.seed)

    lengths = [rng.randint(args.target - args.var, args.target + args.var) for _ in range(n)]
    prompts = [build_prompt(rng, tok, L) for L in lengths]

    # Verify against the tokenizer rather than trusting the construction.
    actual = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    exact = sum(1 for a, w in zip(actual, lengths) if a == w)
    assert not any("\n" in p for p in prompts), "prompt contains newline; file is one-per-line"

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(prompts) + "\n")

    print(f"wrote {len(prompts)} prompts -> {out}")
    print(f"  seed={args.seed} tokenizer={tokenizer}")
    print(f"  requested {args.target} +/- {args.var}  ->  actual min={min(actual)} "
          f"max={max(actual)} mean={statistics.mean(actual):.1f} median={statistics.median(actual)}")
    print(f"  exact-length matches: {exact}/{len(prompts)}")
    if exact != len(prompts):
        print("  [WARN] some prompts missed their target length; distribution is still "
              "declared+fixed, but check the tokenizer round-trip.")


if __name__ == "__main__":
    main()
