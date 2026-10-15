#!/bin/bash
# One-time setup on an AutoDL (or any CUDA 12) GPU machine, then self-tests on the real GPU and a short benchmark.
#   bash scripts/autodl_setup.sh            (logs: results/autodl_selftest/*.log; exit code != 0 if a check fails)
set -eo pipefail
cd "$(dirname "$0")/.."
LOG=results/autodl_selftest
mkdir -p "$LOG"
python -m pip install -q --upgrade pip
python -m pip install -q numpy scipy matplotlib imageio imageio-ffmpeg pillow numba
# numba's CUDA target: on numba >= 0.61 it lives in the numba-cuda package; the cu12 extra brings the NVVM/runtime libs.
python -m pip install -q "numba-cuda[cu12]" || python -m pip install -q numba-cuda || true
# Chinese titles in the movies (optional; falls back to the default font)
(command -v apt-get >/dev/null && apt-get install -y -q fonts-wqy-zenhei >/dev/null 2>&1) || true
python - <<'PY'
from numba import cuda
print("CUDA available:", cuda.is_available())
if not cuda.is_available():
    raise SystemExit("no CUDA device visible to numba (check the driver / numba-cuda[cu12])")
d = cuda.get_current_device()
print("device:", d.name.decode() if isinstance(d.name, bytes) else d.name, "cc", d.compute_capability,
      "SMs", d.MULTIPROCESSOR_COUNT)
PY
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4}
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0
echo "[1/4] CPU vs CUDA solver (heart, N=256, 1 s), batched mode == single runs, FP32 smoke"
python -m latte_imex.test_cuda --N 256 --T 1.0 2>&1 | tee "$LOG/test_cuda.log"
echo "[2/4] symmetric viscosity / off-cup switches, CPU and CPU-vs-CUDA"
python -m latte_imex.test_visc cuda 2>&1 | tee "$LOG/test_visc.log"
echo "[3/4] GPU forward pitcher vs CPU ForwardPitcher (S1 poses, 3 patterns x 1500 steps, 3 laws)"
python -m latte_imex.test_cuda_pitcher 2>&1 | tee "$LOG/test_cuda_pitcher.log"
echo "[4/4] benchmark: heart 1 s at N=256, B = 1, 8, 32 members"
python -m latte_imex.bench_cuda --quick --N 256 --T 1.0 --B 1,8,32 --out "$LOG/bench" 2>&1 | tee "$LOG/bench.log"
echo "self-tests passed; logs in $LOG"
