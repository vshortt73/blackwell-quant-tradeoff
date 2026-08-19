"""
Paired perplexity comparison across quantization schemes.

Why paired. Every arm scores the IDENTICAL token sequence from the same
held-out corpus. Passage difficulty is by far the largest source of variance in
per-token NLL, and in a paired difference it cancels exactly. Comparing two
aggregate perplexity scalars discards that structure and, worse, leaves no
uncertainty estimate at all -- you get two numbers and no way to say whether
their difference is real.

Why not "N>=5 repeats, mean +/- CI" for this metric. Perplexity here is
deterministic: greedy, no sampling, fixed corpus. Repeating the measurement
reproduces the same number to within kernel reduction-order jitter, which would
yield an impressively tight CI that means nothing. The real question is corpus
sampling -- would a different held-out set have given a different answer? -- and
that is estimated by resampling PASSAGES, not by repeating runs.

Bootstrap over passages, not tokens: tokens within a passage are strongly
correlated, so treating them as independent would understate the interval by a
large factor. Resampling whole passages respects that correlation.

Measured on real data (Qwen3-8B, 198 passages / 82,282 tokens of held-out
private prose, AWQ-4bit vs the bf16 baseline):

    paired    delta +0.05833 nats/token  [+0.05338, +0.06359]  SIGNIFICANT
    unpaired  delta                      [-0.06674, +0.18328]  spans zero

Same data, same measurement. The unpaired comparison FAILS TO DETECT a real
6.0% perplexity degradation; the paired one resolves it with room to spare
(24x tighter interval). That is the entire argument for retaining per-passage
numbers instead of collapsing to a scalar.

The bootstrap is seeded, so the same inputs give the same interval every time.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Any, Sequence

DEFAULT_BOOT = 10_000
DEFAULT_SEED = 20260819


class AlignmentError(ValueError):
    """Raised when two runs did not score the same passages. Pairing them
    anyway would silently compare different text."""


@dataclass
class PairedResult:
    n_passages: int
    n_tokens: int
    baseline_perplexity: float
    variant_perplexity: float
    perplexity_ratio: float          # variant / baseline; >1 means worse
    ratio_ci95: tuple[float, float]
    delta_mean_nll: float            # variant - baseline, in nats/token
    delta_ci95: tuple[float, float]
    significant: bool                # does the 95% CI exclude zero?
    n_boot: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_alignment(a: Sequence[dict], b: Sequence[dict]) -> None:
    if len(a) != len(b):
        raise AlignmentError(f"passage count differs: {len(a)} vs {len(b)}")
    for x, y in zip(a, b):
        if x["sha"] != y["sha"]:
            raise AlignmentError(
                f"passage {x['i']} differs between runs ({x['sha']} vs {y['sha']}): "
                "the two arms did not score the same corpus"
            )
        if x["n_tokens"] != y["n_tokens"]:
            raise AlignmentError(
                f"passage {x['i']} tokenized to {x['n_tokens']} vs {y['n_tokens']} "
                "tokens -- the arms disagree on tokenization; pairing is invalid"
            )


def _weighted_mean_nll(rows: Sequence[dict], idx: Sequence[int]) -> float:
    num = 0.0
    den = 0
    for i in idx:
        num += rows[i]["sum_nll"]
        den += rows[i]["n_tokens"]
    return num / den if den else float("nan")


def paired_perplexity(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    n_boot: int = DEFAULT_BOOT,
    seed: int = DEFAULT_SEED,
) -> PairedResult:
    """Compare two perplexity metric dicts produced by run_quality_eval.

    Both must carry `per_passage` (added when per-passage retention landed);
    older results without it cannot be paired.
    """
    for name, d in (("baseline", baseline), ("variant", variant)):
        if "per_passage" not in d:
            raise AlignmentError(
                f"{name} result has no per_passage data -- it predates paired "
                "analysis and must be re-run"
            )
    if baseline.get("corpus_sha") != variant.get("corpus_sha"):
        raise AlignmentError(
            f"different corpora: {baseline.get('corpus_sha')} vs "
            f"{variant.get('corpus_sha')}"
        )

    a, b = baseline["per_passage"], variant["per_passage"]
    _check_alignment(a, b)

    n = len(a)
    all_idx = list(range(n))
    base_nll = _weighted_mean_nll(a, all_idx)
    var_nll = _weighted_mean_nll(b, all_idx)
    delta = var_nll - base_nll

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        # ONE resample of passage indices, applied to BOTH arms -- that is what
        # makes it paired. Resampling each arm independently would reintroduce
        # exactly the passage-difficulty variance we are trying to cancel.
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(_weighted_mean_nll(b, idx) - _weighted_mean_nll(a, idx))
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[min(int(0.975 * n_boot), n_boot - 1)]

    return PairedResult(
        n_passages=n,
        n_tokens=sum(r["n_tokens"] for r in a),
        baseline_perplexity=math.exp(base_nll),
        variant_perplexity=math.exp(var_nll),
        perplexity_ratio=math.exp(delta),
        ratio_ci95=(math.exp(lo), math.exp(hi)),
        delta_mean_nll=delta,
        delta_ci95=(lo, hi),
        significant=not (lo <= 0.0 <= hi),
        n_boot=n_boot,
        seed=seed,
    )


def format_paired(r: PairedResult, baseline_label: str, variant_label: str) -> str:
    verdict = "SIGNIFICANT" if r.significant else "not distinguishable from zero"
    return (
        f"{variant_label} vs {baseline_label}  "
        f"({r.n_passages} passages, {r.n_tokens:,} tokens, paired bootstrap "
        f"n={r.n_boot})\n"
        f"  perplexity   {r.baseline_perplexity:.3f} -> {r.variant_perplexity:.3f}"
        f"   ratio {r.perplexity_ratio:.4f} "
        f"[{r.ratio_ci95[0]:.4f}, {r.ratio_ci95[1]:.4f}]\n"
        f"  delta NLL    {r.delta_mean_nll:+.5f} nats/token "
        f"[{r.delta_ci95[0]:+.5f}, {r.delta_ci95[1]:+.5f}]   {verdict}"
    )


# --------------------------------------------------------------------------- #
# Paired task accuracy (McNemar)
# --------------------------------------------------------------------------- #
@dataclass
class McNemarResult:
    n_items: int
    baseline_correct: int
    variant_correct: int
    baseline_accuracy: float
    variant_accuracy: float
    delta_accuracy: float      # variant - baseline
    b: int                     # baseline right, variant WRONG (regressions)
    c: int                     # baseline wrong, variant RIGHT (recoveries)
    n_discordant: int
    p_value: float
    significant: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test against p=0.5.

    Exact rather than the chi-square approximation because the quantity that
    matters here is the DISCORDANT count, which is small even when the item
    count is large -- chi-square is unreliable in exactly that regime."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_accuracy(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    alpha: float = 0.05,
) -> McNemarResult:
    """McNemar's test on two task_accuracy results.

    Both arms answer the IDENTICAL items, so comparing two accuracy proportions
    throws away the pairing. Items both arms get right (or both wrong) carry no
    information about a difference between them; only the flips do. McNemar
    tests exactly those discordant pairs, which is far more sensitive than
    comparing proportions -- and it separates regressions from recoveries,
    which a single accuracy delta hides completely.
    """
    for name, d in (("baseline", baseline), ("variant", variant)):
        if "per_item" not in d:
            raise AlignmentError(f"{name} result has no per_item data")
    if baseline.get("task_sha") != variant.get("task_sha"):
        raise AlignmentError(
            f"different task files: {baseline.get('task_sha')} vs {variant.get('task_sha')}"
        )
    if baseline.get("few_shot_sha") != variant.get("few_shot_sha"):
        raise AlignmentError(
            "different few-shot prefixes -- the arms were not prompted "
            "identically, so the comparison is invalid"
        )
    a, b_ = baseline["per_item"], variant["per_item"]
    if len(a) != len(b_):
        raise AlignmentError(f"item count differs: {len(a)} vs {len(b_)}")
    for x, y in zip(a, b_):
        if x["sha"] != y["sha"]:
            raise AlignmentError(
                f"item {x['i']} differs between runs -- the arms answered different tasks"
            )

    n = len(a)
    b_cnt = sum(1 for x, y in zip(a, b_) if x["correct"] and not y["correct"])
    c_cnt = sum(1 for x, y in zip(a, b_) if not x["correct"] and y["correct"])
    base_ok = sum(1 for x in a if x["correct"])
    var_ok = sum(1 for y in b_ if y["correct"])
    p = _binom_two_sided(min(b_cnt, c_cnt), b_cnt + c_cnt)

    return McNemarResult(
        n_items=n,
        baseline_correct=base_ok,
        variant_correct=var_ok,
        baseline_accuracy=base_ok / n if n else float("nan"),
        variant_accuracy=var_ok / n if n else float("nan"),
        delta_accuracy=(var_ok - base_ok) / n if n else float("nan"),
        b=b_cnt,
        c=c_cnt,
        n_discordant=b_cnt + c_cnt,
        p_value=p,
        significant=p < alpha,
    )


def format_mcnemar(r: McNemarResult, baseline_label: str, variant_label: str) -> str:
    verdict = "SIGNIFICANT" if r.significant else "not significant"
    warn = ""
    if r.n_discordant < 10:
        warn = ("\n  [WARN] only %d discordant pairs -- too few to conclude much. "
                "A larger or harder task set would help." % r.n_discordant)
    return (
        f"{variant_label} vs {baseline_label}  ({r.n_items} items, McNemar exact)\n"
        f"  accuracy     {r.baseline_accuracy:.3f} -> {r.variant_accuracy:.3f}"
        f"   delta {r.delta_accuracy:+.3f}\n"
        f"  regressions  {r.b} (was right, now wrong)\n"
        f"  recoveries   {r.c} (was wrong, now right)\n"
        f"  p = {r.p_value:.4g}   {verdict}" + warn
    )
