"""
Probe-set difficulty calibration for APEX.

WHY THIS EXISTS: APEX's probe set has a difficulty, and that difficulty is
relative to the model under test. A probe every model answers perfectly, or one
that elicits no influence from any model, contributes no variance -- and a
dimension made mostly of such probes cannot show a positional curve no matter
how good the detector is. APEX has no tool for this. `BaselineRunner` measures
bare-vs-anchored reference scores, but nothing flags a probe set as mis-scaled
for the model you are about to spend hours characterizing.

Measured on Qwen3-8B (BF16, ctx 8192), 28 of 60 seed probes were usable:

    application     4 ceiling,  0 floor, 16 usable   mean 0.848
    factual_recall 14 ceiling,  0 floor,  6 usable   mean 0.912
    salience        0 ceiling, 14 floor,  6 usable   mean 0.275

Two dimensions were dominated by degenerate probes, in OPPOSITE directions:
factual probes too easy, salience probes too subtle to move an 8B. Sampling
just two probes per dimension (which is what a truncated run gives you) landed
on F-001/F-002 and S-001/S-002 -- all four degenerate -- and produced the
entirely wrong conclusion that those dimensions were dead. Hence this tool:
judge a probe set from ALL of it, before committing to a sweep.

HOW TO USE. Run APEX over the whole probe set at two positions -- one near an
edge and one in the middle -- with one repetition. That is cheap (60 probes x 2
positions = 120 requests) and enough to classify:

    positions: [0.02, 0.50]
    context_lengths: [8192]
    repetitions: 1

Then point this at the resulting database:

    python harness/check_probes.py results/apex/probecal.db --model qwen3-8b-bf16

A probe is CEILING if it scores at/near maximum at both positions (no room to
degrade), FLOOR if it scores at/near minimum at both (nothing to lose), and
USABLE otherwise. The edge-vs-middle delta is reported for usable probes as a
first look at positional sensitivity -- two positions cannot describe a curve,
but a probe with zero delta is unlikely to produce one.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_apex_eval import APEX_TO_STUDY, load_apex_rows  # noqa: E402

DEFAULT_CEILING = 0.95
DEFAULT_FLOOR = 0.30
# Below this, a "usable" probe still looks unlikely to yield a curve.
WEAK_DELTA = 0.05


def classify(rows, model_id, edge, middle, ceiling, floor):
    """dimension -> {probe_id: (edge_score, middle_score, verdict)}"""
    out: dict[str, dict[str, tuple]] = {}
    for r in rows:
        if model_id and str(r.get("model_id")) != str(model_id):
            continue
        if r.get("score") is None:
            continue
        pos = float(r["target_position_percent"])
        slot = "edge" if abs(pos - edge) < 1e-6 else "middle" if abs(pos - middle) < 1e-6 else None
        if slot is None:
            continue
        d = out.setdefault(str(r["dimension"]), {}).setdefault(str(r["probe_id"]), {})
        d.setdefault(slot, []).append(float(r["score"]))

    verdicts: dict[str, dict[str, tuple]] = {}
    for dim, probes in out.items():
        for pid, slots in probes.items():
            if "edge" not in slots or "middle" not in slots:
                continue
            a = statistics.mean(slots["edge"])
            b = statistics.mean(slots["middle"])
            if a >= ceiling and b >= ceiling:
                v = "ceiling"
            elif a <= floor and b <= floor:
                v = "floor"
            else:
                v = "usable"
            verdicts.setdefault(dim, {})[pid] = (a, b, v)
    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("source", help="APEX results .db or `apex export` .json")
    ap.add_argument("--model", help="model_id to analyse (required if the source holds several)")
    ap.add_argument("--edge", type=float, default=0.02)
    ap.add_argument("--middle", type=float, default=0.50)
    ap.add_argument("--ceiling", type=float, default=DEFAULT_CEILING)
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    ap.add_argument("--emit-select", action="store_true",
                    help="print a probes.select list of the usable probes")
    args = ap.parse_args()

    rows = load_apex_rows(args.source)
    models = sorted({str(r.get("model_id")) for r in rows})
    if not args.model and len(models) > 1:
        raise SystemExit(f"source holds several models {models}; pass --model")
    model_id = args.model or models[0]

    verdicts = classify(rows, model_id, args.edge, args.middle, args.ceiling, args.floor)
    if not verdicts:
        raise SystemExit(
            f"no probes had scores at BOTH position {args.edge} and {args.middle} for "
            f"model '{model_id}'. Run APEX over the whole probe set at those two "
            "positions first (see this module's docstring)."
        )

    keep_all: list[str] = []
    problems = 0
    print(f"=== probe calibration: {model_id} ===")
    print(f"    edge={args.edge}  middle={args.middle}  "
          f"ceiling>={args.ceiling}  floor<={args.floor}\n")
    for study_dim in APEX_TO_STUDY.values():
        apex_dim = [k for k, v in APEX_TO_STUDY.items() if v == study_dim][0]
        pr = verdicts.get(apex_dim)
        if not pr:
            print(f"  {apex_dim}: no data\n")
            continue
        usable = {p: v for p, v in pr.items() if v[2] == "usable"}
        ceil = sorted(p for p, v in pr.items() if v[2] == "ceiling")
        flr = sorted(p for p, v in pr.items() if v[2] == "floor")
        keep_all += sorted(usable)
        mean = statistics.mean([v[0] for v in pr.values()] + [v[1] for v in pr.values()])
        print(f"  {apex_dim}  ({len(pr)} probes, mean {mean:.3f})")
        print(f"    usable  {len(usable):>2}   {', '.join(sorted(usable)) or '-'}")
        if ceil:
            print(f"    ceiling {len(ceil):>2}   {', '.join(ceil)}")
            print("             -> too easy for this model; make them harder")
        if flr:
            print(f"    floor   {len(flr):>2}   {', '.join(flr)}")
            print("             -> elicit no influence; strengthen the signal")
        if usable:
            deltas = [abs(v[0] - v[1]) for v in usable.values()]
            med = statistics.median(deltas)
            print(f"    usable |edge-middle|: median={med:.3f} max={max(deltas):.3f}")
            if med < WEAK_DELTA:
                print("             [WARN] usable probes show almost no positional "
                      "sensitivity; this dimension may still fail to produce a curve")
                problems += 1
        if len(usable) < 8:
            print(f"             [WARN] only {len(usable)} usable probes -- thin for "
                  "curve detection; consider authoring replacements")
            problems += 1
        print()

    total = sum(len(v) for v in verdicts.values())
    print(f"  USABLE TOTAL: {len(keep_all)}/{total}")
    if args.emit_select:
        print("\nprobes:\n  select: [" + ", ".join(f'"{p}"' for p in keep_all) + "]")
    else:
        print("  (re-run with --emit-select for a pasteable probes.select list)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
