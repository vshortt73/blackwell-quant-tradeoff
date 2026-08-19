"""
Curve-existence detection over APEX position-influence data.

WHY THIS EXISTS: APEX does not report `curve_exists` or `curve_strength`.
It reports raw `probe_results` rows -- a score for each (dimension, position,
context_length, repetition), plus a refusal flag. The study's central claim
("does an exploitable, structured curve survive quantization, and does it
collapse at the same precision for every dimension?") requires turning those
raw scores into a per-dimension verdict. That derivation is a methodological
choice, so it is defined here explicitly rather than buried in a query.

THE DEFINITION
A dimension has an exploitable curve at a given context length if POSITION
EXPLAINS VARIANCE IN SCORE BEYOND CHANCE.

  curve_strength = eta-squared = SS_between_positions / SS_total
                   (0 = position irrelevant, 1 = position explains everything)

  curve_exists   = permutation test on eta-squared, p < alpha

Why eta-squared rather than a correlation: positional effects are frequently
NON-MONOTONIC -- the lost-in-the-middle U-shape is the canonical example. A
Spearman or Pearson correlation against position would score a perfect U at
nearly zero and call a real, exploitable curve "noise". Eta-squared is
shape-agnostic: it asks only whether position matters, not which direction.

Why a permutation test rather than ANOVA's F distribution: APEX scores are
bounded in [0,1], frequently at ceiling, and far from normal. Shuffling
position labels builds the null distribution from the observed scores
themselves, assuming nothing about their shape.

Determinism: the permutation is seeded, so the same rows always give the same
verdict. A study that cannot reproduce its own significance calls is not
reproducible.

POWER CAVEAT: if scores sit at ceiling (all 1.0) there is no variance to
partition, eta-squared is undefined, and no test can find a curve. That is
reported as curve_exists=False with reason="no_variance" -- which is NOT the
same finding as "position does not matter", and must not be read as one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict, field
from typing import Any, Sequence

DEFAULT_PERM = 10_000
DEFAULT_SEED = 20260819
DEFAULT_ALPHA = 0.05


@dataclass
class CurveResult:
    curve_exists: bool
    curve_strength: float          # eta-squared in [0,1]
    p_value: float
    reason: str                    # why the verdict came out this way
    n_positions: int
    n_observations: int
    mean_score: float
    position_means: dict[str, float] = field(default_factory=dict)
    n_perm: int = DEFAULT_PERM
    seed: int = DEFAULT_SEED

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _eta_squared(groups: list[list[float]]) -> float | None:
    """Fraction of total variance attributable to group membership.

    Returns None when total variance is zero -- every observation identical, so
    the question "does position matter" has no answer rather than the answer
    "no"."""
    vals = [v for g in groups for v in g]
    n = len(vals)
    if n < 2:
        return None
    grand = sum(vals) / n
    ss_total = sum((v - grand) ** 2 for v in vals)
    if ss_total <= 0:
        return None
    ss_between = sum(len(g) * ((sum(g) / len(g)) - grand) ** 2 for g in groups if g)
    return ss_between / ss_total


def detect_curve(
    observations: Sequence[tuple[float, float]],
    n_perm: int = DEFAULT_PERM,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> CurveResult:
    """observations: (position_percent, score) pairs for ONE dimension at ONE
    context length. Repetitions appear as repeated positions."""
    by_pos: dict[float, list[float]] = {}
    for pos, score in observations:
        by_pos.setdefault(float(pos), []).append(float(score))

    n_obs = sum(len(v) for v in by_pos.values())
    positions = sorted(by_pos)
    pos_means = {f"{p:.3f}": sum(by_pos[p]) / len(by_pos[p]) for p in positions}
    mean_score = (sum(v for g in by_pos.values() for v in g) / n_obs) if n_obs else float("nan")

    if len(positions) < 2:
        return CurveResult(False, 0.0, 1.0, "insufficient_positions",
                           len(positions), n_obs, mean_score, pos_means, n_perm, seed)
    if n_obs < 2 * len(positions):
        return CurveResult(False, 0.0, 1.0, "insufficient_observations",
                           len(positions), n_obs, mean_score, pos_means, n_perm, seed)

    groups = [by_pos[p] for p in positions]
    observed = _eta_squared(groups)
    if observed is None:
        # Ceiling or floor: no variance to explain. Explicitly NOT "no curve".
        return CurveResult(False, 0.0, 1.0, "no_variance",
                           len(positions), n_obs, mean_score, pos_means, n_perm, seed)

    flat = [v for g in groups for v in g]
    sizes = [len(g) for g in groups]
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(flat)
        i = 0
        perm_groups = []
        for s in sizes:
            perm_groups.append(flat[i:i + s]); i += s
        e = _eta_squared(perm_groups)
        if e is not None and e >= observed:
            ge += 1
    # +1 correction: a permutation test can never legitimately report p = 0.
    p = (ge + 1) / (n_perm + 1)
    exists = p < alpha
    return CurveResult(exists, observed, p, "permutation_test",
                       len(positions), n_obs, mean_score, pos_means, n_perm, seed)
