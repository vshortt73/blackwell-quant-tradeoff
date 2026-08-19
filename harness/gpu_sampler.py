"""
Sample GPU telemetry DURING a measured interval, not after it.

Why this exists: the environment fingerprint is captured once, when the
RunResult is constructed -- i.e. after the sweep has finished and the card is
already winding down. A single post-hoc power reading is a tail-end snapshot,
not load power. (Observed: fingerprint recorded 369W while the card was
actually pulling 455W mid-sweep.)

For the watts-vs-concurrency panel -- and for any claim about where a card
crosses from bandwidth-starved into saturated -- power must be sampled
continuously across each concurrency level and reported as a distribution.

Usage:
    with GpuSampler(interval=0.25) as s:
        ...run the measured workload...
    stats = s.stats()   # -> {"power_w": {...}, "clocks_sm_mhz": {...}, ...}

Notes:
 - Polls `nvidia-smi` in a daemon thread; if nvidia-smi is unavailable the
   sampler degrades to `available: False` rather than raising, so a missing
   tool can never abort a benchmark run.
 - Sampling is deliberately cheap (one subprocess per interval). At 0.25s the
   overhead is negligible next to inference, but do NOT drop it far below that
   -- the subprocess cost starts competing with the thing being measured.
 - Multi-GPU: records index 0 by default (these runs are pinned to one GPU via
   CUDA_VISIBLE_DEVICES). Set gpu_index if that ever changes.
"""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from typing import Any

# Fields sampled per tick. Keep this list VALID -- nvidia-smi fails the whole
# query if any single field is bad (see common.nvidia_smi for that lesson).
_FIELDS = [
    "power.draw",
    "clocks.sm",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
]


def _poll(gpu_index: int) -> dict[str, float] | None:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                f"--query-gpu={','.join(_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
    except Exception:
        return None
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    vals = [v.strip() for v in line.split(",")]
    if len(vals) != len(_FIELDS):
        return None
    out: dict[str, float] = {}
    for k, v in zip(_FIELDS, vals):
        try:
            out[k] = float(v)
        except ValueError:
            continue  # e.g. "[N/A]"
    return out or None


def _summarize(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    xs_sorted = sorted(xs)

    def _p(p: float) -> float:
        k = (len(xs_sorted) - 1) * p
        lo = int(k)
        hi = min(lo + 1, len(xs_sorted) - 1)
        return xs_sorted[lo] + (xs_sorted[hi] - xs_sorted[lo]) * (k - lo)

    return {
        "n": len(xs),
        "min": xs_sorted[0],
        "mean": statistics.fmean(xs),
        "p50": _p(0.50),
        "p90": _p(0.90),
        "max": xs_sorted[-1],
        "stdev": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
    }


class GpuSampler:
    """Context manager: samples telemetry in the background for the duration
    of the `with` block."""

    def __init__(self, interval: float = 0.25, gpu_index: int = 0,
                 settle_seconds: float = 0.0) -> None:
        self.interval = interval
        self.gpu_index = gpu_index
        self.settle_seconds = settle_seconds
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = True

    def _run(self) -> None:
        # Optional settle: skip the first N seconds so ramp-up transients
        # don't drag the mean down. Keep 0 unless you have a reason.
        if self.settle_seconds:
            self._stop.wait(self.settle_seconds)
        while not self._stop.is_set():
            s = _poll(self.gpu_index)
            if s is None:
                self._available = False
                return
            self._samples.append(s)
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def stats(self) -> dict[str, Any]:
        if not self._available or not self._samples:
            return {"available": False, "n_samples": len(self._samples)}
        by_field = {f: [s[f] for s in self._samples if f in s] for f in _FIELDS}
        return {
            "available": True,
            "n_samples": len(self._samples),
            "sample_interval_s": self.interval,
            "power_w": _summarize(by_field["power.draw"]),
            "clocks_sm_mhz": _summarize(by_field["clocks.sm"]),
            "temperature_c": _summarize(by_field["temperature.gpu"]),
            "utilization_pct": _summarize(by_field["utilization.gpu"]),
            "memory_used_mib": _summarize(by_field["memory.used"]),
        }
