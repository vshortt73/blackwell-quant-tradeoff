"""
Import APEX results and derive per-dimension curve verdicts.

TWO-PASS DESIGN. This does NOT run APEX. APEX is run natively, as its own tool,
with its own config -- it already supports several models in one run and records
`quantization` per row, so one `apex run` can cover every arm of the sweep.
This module only reads what APEX produced and turns it into the study's
per-dimension contract.

Why two passes rather than one pipeline: the two measurements have conflicting
discipline. APEX parallelises (`workers: 8`) purely to get through thousands of
requests; in the serving benchmark, concurrency IS the independent variable. And
correctness scoring is timing-insensitive while throughput measurement is
timing-only. Running them together would silently corrupt the timing half, so
they must be separate passes on an exclusive GPU regardless of how the code is
arranged. Keeping them separate here also means no generated configs, no
subprocess, and no chance of this harness redirecting APEX's database.

    Pass 1 (APEX, natively):     apex run <your-apex-config>.yaml
    Pass 2 (this repo, per arm): python harness/run_apex_eval.py configs/awq_4bit.yaml \\
                                     --apex-source /opt/apex/results.db

APEX characterizes position-influence curves across THREE dimensions:

    1. factual_recall        -- can the model retrieve a fact placed in context
    2. instruction_following -- does it apply an instruction placed in context
                                (APEX calls this dimension "application")
    3. salience              -- emotional / attention salience curve

THE CENTRAL HYPOTHESIS lives here: quantization is expected to degrade the three
dimensions NON-uniformly. A scheme may preserve factual_recall while the
instruction_following curve collapses into noise a full precision-step earlier.
Reporting per-dimension curve_exists / curve_strength across BF16 -> FP8 ->
AWQ-4bit is the differentiated contribution.

WHAT APEX DOES NOT REPORT: `curve_exists` and `curve_strength` appear nowhere in
its source. It reports raw per-position scores. Those verdicts are derived by
harness/apex_curve.py (permutation test on eta-squared); see that module for the
definition and why a correlation would be the wrong test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from apex_curve import detect_curve
from common import RunResult, declared_controls, load_config

APEX_DIMENSIONS = ("factual_recall", "instruction_following", "salience")

# APEX names the middle dimension "application"; this study calls the same thing
# "instruction_following". A mapping, not a rename: identical probe set.
APEX_TO_STUDY = {
    "factual_recall": "factual_recall",
    "application": "instruction_following",
    "salience": "salience",
}
STUDY_TO_APEX = {v: k for k, v in APEX_TO_STUDY.items()}


@dataclass
class ApexDimensionResult:
    dimension: str
    curve_exists: bool
    curve_strength: float
    # optional richer breakdown APEX already produces:
    refusal_rate: float | None = None
    failure_rate: float | None = None
    over_application_rate: float | None = None
    extra: dict[str, Any] | None = None


def load_apex_rows(source: str | Path) -> list[dict[str, Any]]:
    """Read APEX results from a SQLite database or an `apex export` JSON file.

    Both are supported because APEX may be configured against SQLite or
    PostgreSQL; `apex export <dsn> -o rows.json` is the universal escape hatch
    and needs no database driver here.
    """
    p = Path(source)
    if not p.exists():
        raise SystemExit(f"APEX source not found: {p}")
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        rows = data if isinstance(data, list) else data.get("results", [])
        if not rows:
            raise SystemExit(f"{p} contains no rows")
        return rows

    import sqlite3

    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute("SELECT * FROM probe_results")
    except sqlite3.OperationalError as e:
        raise SystemExit(
            f"{p} does not look like an APEX results database ({e}). Expected a "
            "probe_results table, or pass an `apex export` JSON file instead."
        )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    if not rows:
        raise SystemExit(f"{p} has a probe_results table but no rows")
    return rows


def build_dimension_results(
    rows: list[dict[str, Any]],
    model_id: str,
    n_perm: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260819,
) -> list[ApexDimensionResult]:
    """Derive one ApexDimensionResult per study dimension.

    Raises rather than emitting plausible zeros: a missing or entirely unscored
    dimension is a setup problem to fix, not a finding to report.
    """
    mine = [r for r in rows if str(r.get("model_id")) == str(model_id)]
    if not mine:
        seen = sorted({str(r.get("model_id")) for r in rows})
        raise SystemExit(
            f"No APEX rows for model_id '{model_id}'. Present in this source: {seen}. "
            "Set apex.model_id in the scheme config to the name APEX ran under."
        )

    results: list[ApexDimensionResult] = []
    for study_dim in APEX_DIMENSIONS:
        apex_dim = STUDY_TO_APEX[study_dim]
        dim_rows = [r for r in mine if r.get("dimension") == apex_dim]
        if not dim_rows:
            raise SystemExit(
                f"Dimension '{apex_dim}' has no rows for model '{model_id}'. "
                f"The study requires all of {APEX_DIMENSIONS}; check probes.select "
                "in the APEX config."
            )
        scored = [r for r in dim_rows if r.get("score") is not None]
        if not scored:
            raise SystemExit(
                f"Dimension '{apex_dim}': {len(dim_rows)} rows present but every "
                "score is NULL. These probes are rubric-scored and need an "
                "EVALUATOR model configured in the APEX run. Without one, "
                "salience is entirely unscored and ~75% of application is lost "
                "(only exact-match and programmatic probes survive)."
            )

        # probe_id is REQUIRED: the detector removes probe difficulty before
        # testing position, because probe identity explains ~20x more variance
        # than position does. See harness/apex_curve.py.
        by_ctx: dict[int, list[tuple[str, float, float]]] = {}
        for r in scored:
            by_ctx.setdefault(int(r["context_length"]), []).append(
                (str(r.get("probe_id", "?")),
                 float(r["target_position_percent"]),
                 float(r["score"]))
            )
        per_ctx = {
            str(c): detect_curve(o, n_perm=n_perm, seed=seed, alpha=alpha).as_dict()
            for c, o in sorted(by_ctx.items())
        }
        # Headline verdict uses the LONGEST context: this measures a positional
        # effect, and position has least room to matter in the shortest context.
        longest = max(by_ctx)
        head = detect_curve(by_ctx[longest], n_perm=n_perm, seed=seed, alpha=alpha)

        refused = [r for r in dim_rows if r.get("refused") is not None]
        refusal_rate = (
            sum(1 for r in refused if r["refused"]) / len(refused) if refused else None
        )

        results.append(
            ApexDimensionResult(
                dimension=study_dim,
                curve_exists=head.curve_exists,
                curve_strength=head.curve_strength,
                refusal_rate=refusal_rate,
                failure_rate=None,           # APEX records only a refusal flag; it
                over_application_rate=None,  # does not separate these two.
                extra={
                    "apex_dimension": apex_dim,
                    "apex_model_id": model_id,
                    "headline_context_length": longest,
                    "n_scored": len(scored),
                    "n_unscored": len(dim_rows) - len(scored),
                    "per_context_length": per_ctx,
                    "n_probes": head.n_probes,
                    "curve_detector": {
                        "method": "permutation test on within-probe eta-squared",
                        "n_perm": n_perm,
                        "alpha": alpha,
                        "seed": seed,
                        "p_value": head.p_value,
                        "reason": head.reason,
                    },
                },
            )
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Import a native APEX run and derive per-dimension curves."
    )
    ap.add_argument("config", help="path to a configs/*.yaml file")
    ap.add_argument("--apex-source", help="APEX results .db or `apex export` .json "
                                         "(overrides apex.source in the config)")
    ap.add_argument("--apex-model", help="APEX model_id for this arm "
                                         "(overrides apex.model_id)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    a = cfg.get("apex", {})

    source = args.apex_source or a.get("source")
    model_id = args.apex_model or a.get("model_id")
    if not source:
        raise SystemExit("need --apex-source or apex.source in the config")
    if not model_id:
        raise SystemExit(
            "need --apex-model or apex.model_id: one APEX database may hold every "
            "arm, so the scheme must say which model_id is its own"
        )

    rows = load_apex_rows(source)
    dims = build_dimension_results(
        rows, model_id,
        n_perm=a.get("n_perm", 10_000),
        alpha=a.get("alpha", 0.05),
        seed=a.get("curve_seed", 20260819),
    )

    got = {d.dimension for d in dims}
    missing = set(APEX_DIMENSIONS) - got
    if missing:
        raise SystemExit(f"APEX did not yield all dimensions; missing: {missing}")

    res = RunResult(
        run_kind="apex",
        config_name=cfg["config_name"],
        quant_scheme=cfg["quant_scheme"],
        model=cfg["model"],
        metrics={"dimensions": [asdict(d) for d in dims],
                 "apex_source": str(source), "apex_model_id": model_id},
        notes=cfg.get("notes", ""),
        declared_controls=declared_controls(cfg),
    )
    res.write()
    print("[apex] wrote result.")
    for d in dims:
        p = d.extra["curve_detector"]["p_value"]
        print(f"  {d.dimension:<22} curve_exists={d.curve_exists!s:<6} "
              f"strength={d.curve_strength:.4f} p={p:.4f} "
              f"({d.extra['curve_detector']['reason']})")


if __name__ == "__main__":
    main()
