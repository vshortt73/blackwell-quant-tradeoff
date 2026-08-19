"""
Turn results/raw/*.json into the headline figure + summary tables.

Headline: throughput gain (x) vs per-dimension retention (y), one series per
APEX dimension, points ordered BF16 -> FP8 -> AWQ-4bit. Where the three curves
separate is the finding.

Run after you have serving + apex (+ quality) results for each scheme:
    python results/analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from common import load_all  # noqa: E402
from paired import AlignmentError, format_paired, paired_perplexity  # noqa: E402

PLOTS = Path(__file__).resolve().parent.parent / "plots"
# Declared ordering of schemes along the throughput axis (edit to taste).
SCHEME_ORDER = ["BF16", "FP8-native", "AWQ-4bit"]


def index_results(rows: list[dict]) -> dict:
    """scheme -> {serving: [...], apex: [...], quality: [...]}"""
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r["quant_scheme"]][r["run_kind"]].append(r)
    return out


def peak_output_tps(serving_rows: list[dict]) -> float | None:
    best = None
    for r in serving_rows:
        for lvl in r["metrics"]["by_concurrency"]:
            v = lvl.get("output_token_throughput_tps")
            if v is not None and (best is None or v > best):
                best = v
    return best


def dimension_strengths(apex_rows: list[dict]) -> dict[str, float]:
    """dimension -> curve_strength (mean if multiple runs)."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in apex_rows:
        for d in r["metrics"]["dimensions"]:
            acc[d["dimension"]].append(d["curve_strength"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def dimension_curve_exists(apex_rows: list[dict]) -> dict[str, bool]:
    """dimension -> curve_exists (True only if it holds in ALL runs)."""
    acc: dict[str, list[bool]] = defaultdict(list)
    for r in apex_rows:
        for d in r["metrics"]["dimensions"]:
            acc[d["dimension"]].append(bool(d["curve_exists"]))
    return {k: all(v) for k, v in acc.items()}


def build_table(idx: dict) -> list[dict]:
    table = []
    for scheme in SCHEME_ORDER:
        if scheme not in idx:
            continue
        s = idx[scheme]
        row = {
            "scheme": scheme,
            "peak_output_tps": peak_output_tps(s.get("serving", [])),
            "apex_strength": dimension_strengths(s.get("apex", [])),
            "apex_curve_exists": dimension_curve_exists(s.get("apex", [])),
        }
        # attach external anchors if present
        for qr in s.get("quality", []):
            row.setdefault("task_accuracy", qr["metrics"].get("task_accuracy", {}).get("accuracy"))
            row.setdefault("perplexity", qr["metrics"].get("perplexity", {}).get("perplexity"))
        table.append(row)
    return table


def print_table(table: list[dict]) -> None:
    if not table:
        print("No results yet. Run the harness for each scheme first.")
        return
    print("\n=== Per-scheme summary ===")
    for row in table:
        print(f"\n[{row['scheme']}]")
        print(f"  peak output tok/s : {row['peak_output_tps']}")
        print(f"  task accuracy     : {row.get('task_accuracy')}")
        print(f"  perplexity        : {row.get('perplexity')}")
        print("  APEX per-dimension (strength | curve_exists):")
        for dim in ("factual_recall", "instruction_following", "salience"):
            st = row["apex_strength"].get(dim)
            ce = row["apex_curve_exists"].get(dim)
            print(f"    {dim:22s}: {st} | {ce}")


def print_paired_perplexity(idx: dict) -> None:
    """Perplexity deltas vs the BF16 baseline, paired per passage.

    Two aggregate perplexity scalars cannot tell you whether a difference is
    real. Every arm scores the identical corpus, so pairing cancels passage
    difficulty -- the dominant variance term -- and yields an actual confidence
    interval on the delta."""
    base_rows = idx.get(SCHEME_ORDER[0], {}).get("quality", [])
    if not base_rows:
        return
    baseline = base_rows[-1]["metrics"].get("perplexity", {})
    if "per_passage" not in baseline:
        print("\n=== Paired perplexity ===")
        print("  baseline result predates per-passage retention; re-run the "
              "quality eval to enable paired analysis.")
        return

    print("\n=== Paired perplexity vs " + SCHEME_ORDER[0] + " ===")
    for scheme in SCHEME_ORDER[1:]:
        rows = idx.get(scheme, {}).get("quality", [])
        if not rows:
            continue
        variant = rows[-1]["metrics"].get("perplexity", {})
        try:
            r = paired_perplexity(baseline, variant)
        except AlignmentError as e:
            print(f"\n  {scheme}: CANNOT PAIR -- {e}")
            continue
        print("\n  " + format_paired(r, SCHEME_ORDER[0], scheme).replace("\n", "\n  "))


def plot_headline(table: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping figure. (pip install matplotlib)")
        return
    if not table:
        return
    PLOTS.mkdir(parents=True, exist_ok=True)

    # x = normalized throughput gain vs BF16 baseline
    base = table[0]["peak_output_tps"] or 1.0
    x = [(row["peak_output_tps"] or 0) / base for row in table]
    labels = [row["scheme"] for row in table]

    fig, ax = plt.subplots(figsize=(8, 5))
    for dim in ("factual_recall", "instruction_following", "salience"):
        y = [row["apex_strength"].get(dim) for row in table]
        ax.plot(x, y, marker="o", label=dim)
    for xi, lab in zip(x, labels):
        ax.annotate(lab, (xi, ax.get_ylim()[0]), fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Output-token throughput  (× BF16 baseline)")
    ax.set_ylabel("APEX curve strength  (retention)")
    ax.set_title("Blackwell quantization: throughput vs per-dimension retention")
    ax.legend()
    fig.tight_layout()
    out = PLOTS / "headline_tradeoff.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


def main() -> None:
    rows = load_all()
    idx = index_results(rows)
    table = build_table(idx)
    print_table(table)
    print_paired_perplexity(idx)
    plot_headline(table)


if __name__ == "__main__":
    main()
