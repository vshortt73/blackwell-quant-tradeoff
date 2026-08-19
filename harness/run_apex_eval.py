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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

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


def run_apex(
    base_url: str,
    served_model_name: str,
    apex_config: dict[str, Any],
) -> list[ApexDimensionResult]:
    """
    ADAPTER BOUNDARY -- implement one of (A)/(B) above.

    Must return exactly one ApexDimensionResult per dimension in
    APEX_DIMENSIONS. Below is a NotImplemented guard so a run cannot silently
    emit fabricated quality numbers. Nothing here invents data.
    """
    raise NotImplementedError(
        "Wire run_apex() to vshortt73/apex. Point it at the vLLM OpenAI endpoint "
        f"{base_url} (served model '{served_model_name}') and return one "
        "ApexDimensionResult per dimension. See module docstring for the two "
        "wiring options."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to a configs/*.yaml file")
    args = ap.parse_args()
    cfg = load_config(args.config)

    dims = run_apex(
        base_url=cfg["base_url"],
        served_model_name=cfg["served_model_name"],
        apex_config=cfg.get("apex", {}),
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
