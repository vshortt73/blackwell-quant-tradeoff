"""
APEX three-dimension quality evaluation, per quantization scheme.

APEX characterizes position-influence curves across THREE independent
dimensions:

    1. factual_recall        -- can the model retrieve a fact placed in context
    2. instruction_following -- does it apply an instruction placed in context
    3. salience              -- emotional / attention salience curve

For each dimension APEX reports, per model:
    - curve_exists : bool    (the curve-existence detector -- is there an
                              exploitable, structured curve, or is it noise?)
    - curve_strength : float (a scalar summarizing curve shape / exploitability)
    - and distinguishes refusal vs failure vs over-application underneath.

THE central hypothesis of this study lives here: quantization is expected to
degrade the three dimensions NON-uniformly. A scheme may preserve
factual_recall while the instruction_following curve collapses into noise a
full precision-step earlier -- or vice versa. Reporting per-dimension
curve_exists / curve_strength across BF16 -> FP8 -> AWQ-4bit is the
differentiated contribution; no vLLM benchmark publishes a per-capability
degradation profile.

------------------------------------------------------------------------------
WIRING TO YOUR REAL APEX (vshortt73/apex)
------------------------------------------------------------------------------
This module does not reimplement APEX. It defines the adapter boundary. Point
`run_apex` at your installed APEX and return the contract below. Two options:

  (A) import path -- if APEX is importable, call its characterization entry
      point directly and map its output onto ApexDimensionResult.
  (B) subprocess  -- shell out to your APEX CLI against the running vLLM
      OpenAI endpoint (APEX already supports OpenAI-compatible + SGLang
      backends per your adapter layer) and parse its JSON/DB output.

The rest of the pipeline only depends on the contract, not on which path you
choose, so you can swap (A)/(B) without touching anything downstream.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

from apex_curve import detect_curve
from common import RunResult, declared_controls, load_config

APEX_DIMENSIONS = ("factual_recall", "instruction_following", "salience")


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


# APEX names the middle dimension "application"; this study calls the same
# thing "instruction_following". Mapping, not renaming: same probe set.
APEX_TO_STUDY = {
    "factual_recall": "factual_recall",
    "application": "instruction_following",
    "salience": "salience",
}


def _write_apex_config(
    base_url: str, served_model_name: str, cfg: dict[str, Any],
    db_path: Path, out_yaml: Path, quant_scheme: str,
) -> None:
    """Generate an APEX run config pointing at the live vLLM endpoint.

    APEX's `openai` backend takes a base_url, and vLLM serves an
    OpenAI-compatible API, so no shim is needed. api_key is a placeholder --
    vLLM does not check it, but the OpenAI client refuses to start without one.
    """
    doc = {
        "run": {
            "seed": cfg.get("seed", 42),
            "temperature": 0.0,          # greedy: same control as every other quality measure
            "repetitions": cfg.get("repetitions", cfg.get("repeats", 5)),
            "filler_type": cfg.get("filler_type", "neutral"),
        },
        "data": {
            "directory": str(Path(cfg["apex_home"]) / "data"),
            "output_db": str(db_path),
        },
        "probes": {"select": cfg.get("probe_select", cfg.get("probe_set", "all"))},
        "positions": cfg.get("positions", [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                                           0.60, 0.70, 0.80, 0.90, 0.95, 0.98]),
        "context_lengths": cfg.get("context_lengths", [4096, 8192]),
        "models": [{
            "name": served_model_name,
            "backend": "openai",
            "model_name": served_model_name,
            "base_url": base_url.rstrip("/") + "/v1",
            "api_key": "vllm-local",
            "tokenizer": cfg.get("tokenizer", "approximate"),
            "max_context_window": cfg.get("max_context_window", 8192),
            "architecture": cfg.get("architecture", "transformer-dense"),
            "parameters": cfg.get("parameters", "8B"),
            # Recorded by APEX in every row, so its DB is self-describing.
            "quantization": quant_scheme,
        }],
    }
    if cfg.get("workers"):
        doc["run"]["workers"] = cfg["workers"]

    # Rubric-scored probes (ALL of salience, 15 of 20 application) need an LLM
    # evaluator. Without one APEX runs them and stores NULL scores.
    #
    # The evaluator is part of the measurement chain, so it is a CONTROLLED
    # variable: identical across every arm, greedy, and pinned. It must NEVER
    # be the model under test -- a quantized model grading its own output makes
    # the measurement circular, and the grader would degrade in lockstep with
    # the thing being graded.
    ev = cfg.get("evaluator")
    if ev:
        if ev.get("base_url", "").rstrip("/") == base_url.rstrip("/") + "/v1":
            raise SystemExit(
                "apex.evaluator points at the same endpoint as the model under "
                "test. That is circular: the judge would be quantized in "
                "lockstep with the subject. Use a separate model."
            )
        doc["evaluator_models"] = [{
            "name": ev["name"],
            "backend": ev["backend"],
            "model_name": ev["model_name"],
            **({"base_url": ev["base_url"]} if ev.get("base_url") else {}),
            **({"api_key": ev["api_key"]} if ev.get("api_key") else {}),
            "max_context_window": ev.get("max_context_window", 32768),
        }]
    out_yaml.write_text(yaml.safe_dump(doc, sort_keys=False))


def _read_apex_db(db_path: Path) -> dict[str, dict[int, list[tuple[float, float]]]]:
    """dimension -> context_length -> [(position_percent, score), ...]"""
    import sqlite3

    if not db_path.exists():
        raise SystemExit(f"APEX produced no database at {db_path}")
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        "SELECT dimension, context_length, target_position_percent, score, refused "
        "FROM probe_results WHERE score IS NOT NULL"
    ).fetchall()
    con.close()
    out: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for dim, ctx, pos, score, _refused in rows:
        out.setdefault(dim, {}).setdefault(int(ctx), []).append((float(pos), float(score)))
    return out


def _unscored_count(db_path: Path, dimension: str) -> int:
    """Rows APEX created but could not score. Non-zero almost always means a
    rubric-scored probe ran with no evaluator model configured."""
    import sqlite3

    con = sqlite3.connect(str(db_path))
    n = con.execute(
        "SELECT COUNT(*) FROM probe_results WHERE dimension = ? AND score IS NULL",
        (dimension,),
    ).fetchone()[0]
    con.close()
    return int(n)


def _refusal_rate(db_path: Path, dimension: str) -> float | None:
    import sqlite3

    con = sqlite3.connect(str(db_path))
    r = con.execute(
        "SELECT AVG(CASE WHEN refused THEN 1.0 ELSE 0.0 END) FROM probe_results "
        "WHERE dimension = ?", (dimension,)
    ).fetchone()
    con.close()
    return None if r is None or r[0] is None else float(r[0])


def run_apex(
    base_url: str,
    served_model_name: str,
    apex_config: dict[str, Any],
    quant_scheme: str = "unknown",
    results_dir: Path | None = None,
) -> list[ApexDimensionResult]:
    """Run APEX against the live vLLM endpoint and derive per-dimension curves.

    Nothing here invents data. If APEX fails, produces no rows, or omits a
    dimension, this raises rather than emitting a plausible-looking zero.

    curve_exists / curve_strength are NOT reported by APEX -- it reports raw
    per-position scores. They are derived here by harness/apex_curve.py, whose
    definition (permutation test on eta-squared) is documented in that module
    and in METHODOLOGY.md.
    """
    apex_home = apex_config.get("apex_home")
    apex_bin = apex_config.get("apex_bin")
    if not apex_home or not apex_bin:
        raise SystemExit(
            "apex.apex_home and apex.apex_bin must be set in the config "
            "(e.g. /opt/apex and /opt/apex/.venv/bin/apex)"
        )
    if not Path(apex_bin).exists():
        raise SystemExit(f"APEX CLI not found at {apex_bin}")

    results_dir = results_dir or (Path(__file__).resolve().parent.parent / "results" / "apex")
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = results_dir / f"apex__{quant_scheme.replace('/', '_')}.db"
    yaml_path = results_dir / f"apex__{quant_scheme.replace('/', '_')}.yaml"
    if db_path.exists():
        # APEX appends; a stale DB would silently blend two schemes' results.
        raise SystemExit(
            f"{db_path} already exists. APEX appends to its database, so reusing "
            "one would mix schemes. Move or delete it before re-running."
        )

    _write_apex_config(base_url, served_model_name, apex_config, db_path,
                       yaml_path, quant_scheme)

    cmd = [str(apex_bin), "run", str(yaml_path)]
    if apex_config.get("workers"):
        cmd += ["--workers", str(apex_config["workers"])]

    # APEX resolves its database as: APEX_DATABASE_URL env > database_url >
    # output_db. If that env var is set (it is, on this machine, pointing at a
    # shared PostgreSQL instance) it SILENTLY overrides the per-scheme SQLite
    # path -- so all three arms would write into one database and blend
    # together, and APEX's resume logic would then report "all probes already
    # completed" and skip the run entirely. Strip it so each scheme gets an
    # isolated, self-contained result file.
    env = {k: v for k, v in os.environ.items() if k != "APEX_DATABASE_URL"}
    print(f"[apex] {' '.join(cmd)}")
    print(f"[apex] APEX_DATABASE_URL stripped; writing to {db_path}")
    proc = subprocess.run(cmd, cwd=str(apex_home), capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-25:])
        raise SystemExit(f"APEX run failed (exit {proc.returncode}):\n{tail}")

    # Match the SKIP message exactly. APEX's normal progress line reads
    # "N probes to run for X (M already completed)", which also contains
    # "already completed" -- a looser check false-positives on a good run.
    if "All probes already completed" in (proc.stdout + proc.stderr):
        raise SystemExit(
            "APEX reported probes already completed and skipped the run. That "
            "means it resolved to a database that already holds results for "
            f"this model name ({served_model_name}). Check APEX_DATABASE_URL "
            f"and remove {db_path} before re-running."
        )

    data = _read_apex_db(db_path)
    if not data:
        raise SystemExit(f"APEX wrote {db_path} but it contains no scored rows")

    n_perm = apex_config.get("n_perm", 10_000)
    alpha = apex_config.get("alpha", 0.05)
    seed = apex_config.get("curve_seed", 20260819)

    results: list[ApexDimensionResult] = []
    for apex_dim, study_dim in APEX_TO_STUDY.items():
        if apex_dim not in data:
            unscored = _unscored_count(db_path, apex_dim)
            if unscored:
                raise SystemExit(
                    f"Dimension '{apex_dim}': APEX produced {unscored} rows but "
                    "every score is NULL. Those probes are rubric-scored and "
                    "need an EVALUATOR model, which is not configured. Set "
                    "apex.evaluator in the config. Without it, salience is "
                    "entirely unscored and ~75% of application/"
                    "instruction_following is lost -- only exact-match and "
                    "programmatic probes survive."
                )
            raise SystemExit(
                f"APEX returned no rows at all for dimension '{apex_dim}'. Check "
                f"probes.select; the study requires all of {APEX_DIMENSIONS}."
            )
        per_ctx = {}
        for ctx, obs in sorted(data[apex_dim].items()):
            per_ctx[str(ctx)] = detect_curve(obs, n_perm=n_perm, seed=seed,
                                             alpha=alpha).as_dict()
        # Headline verdict uses the LONGEST context: positional effects are
        # what this measures, and they are weakest where there is least context
        # for position to matter in.
        longest = max(data[apex_dim])
        head = detect_curve(data[apex_dim][longest], n_perm=n_perm, seed=seed, alpha=alpha)
        results.append(ApexDimensionResult(
            dimension=study_dim,
            curve_exists=head.curve_exists,
            curve_strength=head.curve_strength,
            refusal_rate=_refusal_rate(db_path, apex_dim),
            failure_rate=None,          # APEX does not distinguish these two from
            over_application_rate=None, # refusal; see METHODOLOGY.
            extra={
                "apex_dimension": apex_dim,
                "headline_context_length": longest,
                "per_context_length": per_ctx,
                "curve_detector": {
                    "method": "permutation test on eta-squared",
                    "n_perm": n_perm, "alpha": alpha, "seed": seed,
                    "p_value": head.p_value, "reason": head.reason,
                },
                "db": str(db_path),
            },
        ))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to a configs/*.yaml file")
    args = ap.parse_args()
    cfg = load_config(args.config)

    dims = run_apex(
        base_url=cfg["base_url"],
        served_model_name=cfg["served_model_name"],
        apex_config=cfg.get("apex", {}),
        quant_scheme=cfg["quant_scheme"],
    )

    got = {d.dimension for d in dims}
    missing = set(APEX_DIMENSIONS) - got
    if missing:
        raise SystemExit(f"APEX did not return all dimensions; missing: {missing}")

    RunResult(
        run_kind="apex",
        config_name=cfg["config_name"],
        quant_scheme=cfg["quant_scheme"],
        model=cfg["model"],
        metrics={"dimensions": [asdict(d) for d in dims]},
        notes=cfg.get("notes", ""),
        declared_controls=declared_controls(cfg),
    ).write()
    print("[apex] wrote result.")


if __name__ == "__main__":
    main()
