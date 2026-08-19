"""Tests for paired perplexity analysis. Run: python harness/test_paired.py"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired import AlignmentError, paired_perplexity  # noqa: E402


def make(rows, corpus_sha="abc123"):
    return {
        "corpus_sha": corpus_sha,
        "per_passage": [
            {"i": i, "sha": f"h{i:04d}", "n_tokens": t, "sum_nll": s}
            for i, (t, s) in enumerate(rows)
        ],
    }


def synth(n=200, seed=1, shift=0.0, noise=0.0):
    """Passages of widely varying difficulty (that is the real world), with an
    optional consistent per-token shift applied to the variant."""
    rng = random.Random(seed)
    base, var = [], []
    for _ in range(n):
        t = rng.randint(200, 600)
        difficulty = rng.uniform(1.5, 6.0)          # DOMINANT variance term
        base.append((t, difficulty * t))
        d = difficulty + shift + (rng.gauss(0, noise) if noise else 0.0)
        var.append((t, d * t))
    return make(base), make(var)


def main() -> int:
    fails = []

    def check(name, cond):
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # --- alignment guards -------------------------------------------------- #
    a, b = synth(20)
    try:
        paired_perplexity(make([(10, 5.0)]), make([(10, 5.0), (10, 5.0)]))
        check("passage-count mismatch raises", False)
    except AlignmentError:
        check("passage-count mismatch raises", True)

    bad = make([(10, 5.0)]); bad["per_passage"][0]["sha"] = "different"
    try:
        paired_perplexity(make([(10, 5.0)]), bad)
        check("passage-hash mismatch raises", False)
    except AlignmentError:
        check("passage-hash mismatch raises", True)

    try:
        paired_perplexity(make([(10, 5.0)]), make([(11, 5.0)]))
        check("token-count mismatch raises", False)
    except AlignmentError:
        check("token-count mismatch raises", True)

    try:
        paired_perplexity(make([(10, 5.0)]), make([(10, 5.0)], corpus_sha="other"))
        check("different corpus raises", False)
    except AlignmentError:
        check("different corpus raises", True)

    try:
        paired_perplexity({"mean_nll": 1.0}, make([(10, 5.0)]))
        check("missing per_passage raises", False)
    except AlignmentError:
        check("missing per_passage raises", True)

    # --- null case --------------------------------------------------------- #
    a, _ = synth(200, seed=2)
    r = paired_perplexity(a, a, n_boot=2000)
    check("identical inputs -> zero delta", abs(r.delta_mean_nll) < 1e-12)
    check("identical inputs -> ratio 1.0", abs(r.perplexity_ratio - 1.0) < 1e-12)
    check("identical inputs -> not significant", not r.significant)

    # --- real effect detected ---------------------------------------------- #
    a, b = synth(200, seed=3, shift=0.05, noise=0.02)
    r = paired_perplexity(a, b, n_boot=2000)
    check("known +0.05 nat shift recovered", abs(r.delta_mean_nll - 0.05) < 0.01)
    check("known shift is significant", r.significant)
    check("ratio > 1 for a worse variant", r.perplexity_ratio > 1.0)

    # --- determinism ------------------------------------------------------- #
    r1 = paired_perplexity(a, b, n_boot=2000)
    r2 = paired_perplexity(a, b, n_boot=2000)
    check("bootstrap is deterministic", r1.as_dict() == r2.as_dict())

    # --- the point of the whole module ------------------------------------- #
    # A small shift buried under large passage-difficulty variance: paired
    # should resolve it, unpaired should not.
    a, b = synth(200, seed=4, shift=0.02, noise=0.01)
    r = paired_perplexity(a, b, n_boot=4000)
    paired_w = r.delta_ci95[1] - r.delta_ci95[0]

    rng = random.Random(99)
    n = len(a["per_passage"])
    unpaired = []
    for _ in range(4000):
        ia = [rng.randrange(n) for _ in range(n)]
        ib = [rng.randrange(n) for _ in range(n)]   # INDEPENDENT resamples
        f = lambda rows, idx: sum(rows[i]["sum_nll"] for i in idx) / sum(
            rows[i]["n_tokens"] for i in idx)
        unpaired.append(f(b["per_passage"], ib) - f(a["per_passage"], ia))
    unpaired.sort()
    unpaired_w = unpaired[int(0.975 * 4000)] - unpaired[int(0.025 * 4000)]

    print(f"\n  paired   95% CI width: {paired_w:.5f} nats/token")
    print(f"  unpaired 95% CI width: {unpaired_w:.5f} nats/token")
    print(f"  paired is {unpaired_w / paired_w:.0f}x tighter")
    check("paired CI is tighter than unpaired", paired_w < unpaired_w)
    check("paired resolves the effect", r.significant)

    print(f"\n{'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
