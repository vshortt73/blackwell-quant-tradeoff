#!/usr/bin/env bash
# Lock the GPU into a fixed, reproducible state BEFORE any benchmark run.
# On a single 5090 doing sustained inference, thermal throttling mid-sweep is
# the fastest way to produce dishonest throughput numbers. This is not optional.
#
# Usage:  sudo ./env/gpu_setup.sh <GPU_INDEX> <LOCKED_SM_CLOCK_MHZ> <POWER_LIMIT_W>
# Example: sudo ./env/gpu_setup.sh 0 2400 500
#
# Reset afterwards with:  sudo nvidia-smi -i <GPU_INDEX> -rgc && sudo nvidia-smi -i <GPU_INDEX> -pl <default>

set -euo pipefail

GPU="${1:?GPU index required (e.g. 0)}"
SM_CLOCK="${2:?locked SM clock in MHz required (e.g. 2400)}"
POWER_LIMIT="${3:?power limit in W required (e.g. 500)}"

echo "== Pre-run GPU lock for GPU ${GPU} =="

# Enable persistence so the driver state doesn't reset between runs.
nvidia-smi -i "${GPU}" -pm 1

# Fixed power limit.
nvidia-smi -i "${GPU}" -pl "${POWER_LIMIT}"

# Lock SM clocks to a fixed value (min=max) to eliminate boost variance.
nvidia-smi -i "${GPU}" -lgc "${SM_CLOCK}","${SM_CLOCK}"

echo "== Locked state =="
nvidia-smi -i "${GPU}" --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,power.limit,temperature.gpu,pstate \
  --format=csv,noheader

# Assertions: report the compute capability if we can. This runs under sudo, so
# the python on PATH is usually the SYSTEM one, which does not have torch --
# the project venv does. Try the venv first, and NEVER let this informational
# check abort the run: the clock lock above has already been applied, and
# failing here would leave the GPU locked while the script reports failure.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="${SCRIPT_DIR}/../.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="$(command -v python3 || true)"
CAP="$("$PYBIN" - <<'PY' 2>/dev/null || true
try:
    import torch
    print("sm_%d%d" % torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "none")
except Exception:
    print("unknown (torch not importable by this interpreter)")
PY
)"
CAP="${CAP:-unknown}"
echo "Detected compute capability: ${CAP}"
if [ "${CAP}" != "sm_120" ]; then
  echo "[NOTE] Expected sm_120 (Blackwell 5090). Got ${CAP}. If you're baselining"
  echo "       on the 4080 Super (sm_89) that's fine -- just record it."
fi

echo "== GPU locked. Let temps settle for ~60s before the first measured run. =="
