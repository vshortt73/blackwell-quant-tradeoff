"""
Shared plumbing for every harness run.

Design rule: no result is ever written without a full environment fingerprint
stamped into it. A number that cannot be traced to the exact hardware/software
state that produced it is not a result, it is an anecdote.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_RAW = Path(__file__).resolve().parent.parent / "results" / "raw"
# Full per-item text for local debugging. GITIGNORED: results/raw/ is committed
# and task sets may be built from private material, so text lives only here.
RESULTS_AUDIT = Path(__file__).resolve().parent.parent / "results" / "audit"


# --------------------------------------------------------------------------- #
# Environment fingerprint
# --------------------------------------------------------------------------- #
def _safe(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def nvidia_smi() -> dict[str, Any]:
    """Query the fields that actually move under load. Locked-clock and
    power-limit values are the ones that decide whether a throughput number
    is honest or thermally throttled mid-run.

    NOTE on field validity: nvidia-smi fails the ENTIRE query if *any* single
    field is invalid. `clocks.applications.sm` is NOT valid (applications
    clocks exist only for .graphics/.memory, not .sm) -- including it silently
    wiped all power/clock/temp data. We now (a) use only valid fields and
    (b) fall back to a minimal field set if the full query fails, so one bad
    field on some future driver/card can never again zero out the fingerprint.
    """
    full_fields = [
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "clocks.max.sm",
        "clocks.sm",              # current SM clock (valid)
        "clocks.applications.gr", # applications graphics clock (valid)
        "power.limit",
        "power.draw",
        "temperature.gpu",
        "pstate",
        "utilization.gpu",
    ]
    minimal_fields = ["name", "driver_version", "power.draw", "clocks.sm", "temperature.gpu"]

    def _query(fields: list[str]) -> list[dict[str, str]] | None:
        raw = _safe(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
        )
        if raw is None:
            return None
        return [dict(zip(fields, [v.strip() for v in line.split(",")])) for line in raw.splitlines()]

    gpus = _query(full_fields)
    used = "full"
    if gpus is None:
        gpus = _query(minimal_fields)
        used = "minimal"
    if gpus is None:
        # Last resort: prove nvidia-smi exists at all, and say so loudly.
        present = _safe(["nvidia-smi", "-L"])
        return {"available": False, "nvidia_smi_L": present}
    return {"available": True, "field_set": used, "gpus": gpus}


def torch_info() -> dict[str, Any]:
    try:
        import torch

        cap = None
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)  # (12, 0) on Blackwell sm_120
        return {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "compute_capability": f"sm_{cap[0]}{cap[1]}" if cap else None,
        }
    except Exception as e:
        return {"error": repr(e)}


def git_commit() -> str | None:
    return _safe(["git", "rev-parse", "HEAD"])


def git_dirty() -> bool | None:
    status = _safe(["git", "status", "--porcelain"])
    if status is None:
        return None
    return len(status) > 0


def backend_env() -> dict[str, str | None]:
    """Capture the vLLM backend selection as a CONTROLLED variable. The
    sampler/attention backend must be identical across the 4080 baseline and
    the 5090 runs; relying on installed-package names to infer it is not enough
    (vLLM vendors attention; disabling FlashInfer leaves no pip fingerprint).
    Record the env that actually drove the choice.

    LIMITATION: this reads THIS process's environment. The vLLM server is a
    separate process, so vars exported only on its launch line do not appear
    here. `declared_controls()` is the authoritative record of what the run was
    configured with; this function only captures what the harness itself saw."""
    keys = [
        "VLLM_USE_FLASHINFER_SAMPLER",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_USE_V1",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
    ]
    return {k: os.environ.get(k) for k in keys}


# Keys whose values must be identical across every scheme in a comparison.
# Recorded from the config because the server that actually applies them runs
# in a DIFFERENT process: backend_env() reads this harness's os.environ, which
# does not see `VLLM_ATTENTION_BACKEND=... vllm serve ...`. Without this the
# controls would look unset in every result.
CONTROL_KEYS = (
    "attention_backend",
    "use_deep_gemm",
    "gpu_memory_utilization",
    "max_model_len",
    "max_tokens",
    "ignore_eos",
    "throughput_temperature",
    "input_profile",
    "checkpoint",
)


def write_audit(records: list[dict[str, Any]], run_kind: str, config_name: str,
                run_id: str, audit_dir: Path = RESULTS_AUDIT) -> Path | None:
    """Write full per-item detail alongside a result, keyed by the same run_id.

    Never committed. Its only job is to let you inspect WHY an item was graded
    the way it was without publishing the task text."""
    if not records:
        return None
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"{run_kind}__{config_name}__{run_id}.json"
    path.write_text(json.dumps(records, indent=2))
    return path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a scheme config MERGED over configs/_shared.yaml.

    _shared.yaml declares itself the single source of truth for the controlled
    variables, but it is only true if something actually loads it. Without this
    merge, keys that live solely in _shared.yaml (attention_backend,
    use_deep_gemm, gpu_memory_utilization, max_model_len) never reach the
    result record, and the controls would silently go unrecorded.

    Scheme keys win over shared keys, so a scheme can still override -- but the
    thing that is supposed to be identical across arms now has exactly one
    place it is written down.
    """
    path = Path(path)
    import yaml

    cfg: dict[str, Any] = {}
    shared = path.parent / "_shared.yaml"
    if shared.exists() and shared.resolve() != path.resolve():
        cfg = yaml.safe_load(shared.read_text()) or {}
    scheme = yaml.safe_load(path.read_text()) or {}
    return _deep_merge(cfg, scheme)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge, `over` winning on conflicts.

    A shallow update() is wrong here: a scheme file that sets `quality:` at all
    would replace the ENTIRE shared quality block, silently dropping controlled
    keys the scheme never mentioned (this really happened -- it dropped the
    few-shot prefix from every run). Nested control blocks must merge key by
    key, not wholesale."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def declared_controls(cfg: dict[str, Any]) -> dict[str, Any]:
    """The controlled variables this run DECLARES, lifted from its config."""
    return {k: cfg[k] for k in CONTROL_KEYS if k in cfg}


def environment_fingerprint() -> dict[str, Any]:
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "nvidia_smi": nvidia_smi(),
        "torch": torch_info(),
        "backend_env": backend_env(),
        "versions": {
            # flashinfer ships as 'flashinfer-python'; check both spellings.
            pkg: _pkg_version(pkg)
            for pkg in (
                "vllm",
                "transformers",
                "flashinfer",
                "flashinfer-python",
                "flash-attn",
                "torch",
                "xformers",
            )
        },
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }


# --------------------------------------------------------------------------- #
# Result envelope
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    """One immutable result record. Written once, never edited by hand."""

    run_kind: str  # "serving" | "quality" | "apex"
    config_name: str  # e.g. "awq_4bit"
    quant_scheme: str  # e.g. "AWQ-4bit"
    model: str
    metrics: dict[str, Any]
    notes: str = ""
    # Controls the config declared. The server applies these in another
    # process, so they cannot be recovered from this process's environment.
    declared_controls: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fingerprint: dict[str, Any] = field(default_factory=environment_fingerprint)

    def write(self, results_dir: Path = RESULTS_RAW) -> Path:
        results_dir.mkdir(parents=True, exist_ok=True)
        if self.fingerprint.get("git_dirty"):
            # A dirty tree means the code that produced this number is not
            # reproducible from any commit. Warn loudly; do not silently accept.
            print(
                f"[WARN] git tree is dirty for run {self.run_id}; "
                "result is not reproducible from a commit."
            )
        fname = f"{self.run_kind}__{self.config_name}__{self.run_id}.json"
        path = results_dir / fname
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def load_all(results_dir: Path = RESULTS_RAW) -> list[dict[str, Any]]:
    out = []
    for p in sorted(results_dir.glob("*.json")):
        out.append(json.loads(p.read_text()))
    return out
