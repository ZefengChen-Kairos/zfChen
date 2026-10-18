#!/bin/bash
# Full pipeline on the AutoDL GPU (runs in the background; re-running resumes finished stages):
#   bash scripts/autodl_fit.sh                 -> prints the log path and returns
#   tail -f results/fit.log                    -> progress
#   CAL=tulip,heart (default) | tulip,heart,leaf ...   calibration patterns (the others are predictions)
# 1. fit   : V0.5 closure fitted to the real pours driven by the measured hand actions (Kona 700 mL pitcher CAD,
#            165 g poured), Sobol 512 + 3 refinement rounds at N=128 on the GPU, best 4 + S1 on all patterns at N=256
# 2. render: every pattern with the fitted closure at N=256 (CPU, one process per pattern) + movies
#            real | simulation | hand action, and summary.png
# Send back: results/fit_results.tar.gz
cd "$(dirname "$0")/.."
LOG=results/fit.log
mkdir -p results
if [ "${_LATTE_BG:-}" != "1" ]; then
  _LATTE_BG=1 nohup bash "$0" "$@" >> "$LOG" 2>&1 &
  echo "started in the background (pid $!); log: $(pwd)/$LOG"
  echo "watch: tail -f $(pwd)/$LOG"
  exit 0
fi
set -eo pipefail
export PYTHONUNBUFFERED=1
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4}
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0
export PYTHONWARNINGS=ignore
python -c "import cv2, matplotlib.pyplot, imageio, imageio_ffmpeg" 2>/dev/null || \
  python -m pip install -q "numpy<2" "opencv-python-headless==4.10.0.84" imageio imageio-ffmpeg
python - <<'PY'
from numba import cuda
assert cuda.is_available(), "no CUDA device visible to numba: run bash scripts/autodl_setup.sh first"
print("GPU:", cuda.get_current_device().name)
PY
OUT=${OUT:-results/fit}
CAL=${CAL:-tulip,heart}
echo "===== $(date '+%F %T') fit: calibration $CAL ====="
python -m latte_imex.rig.fit_pipeline fit --out "$OUT" --cal "$CAL" --backend cuda "$@"
echo "===== $(date '+%F %T') render ====="
python -m latte_imex.rig.fit_pipeline render --out "$OUT" --cal "$CAL" --jobs ${JOBS:-5}
tar -czf results/fit_results.tar.gz "$OUT/summary.json" "$OUT/best_closure.json" "$OUT/records.jsonl" "$OUT/config.json" \
    "$OUT/sheet.png" "$OUT/render" $(ls "$OUT"/fields/C_*_N256.npy 2>/dev/null) \
    raw_data/rig/results/*/sim_$(basename "$OUT")/info.json raw_data/rig/results/*/sim_$(basename "$OUT")/c.npz "$LOG"
ls -lh results/fit_results.tar.gz
echo "===== $(date '+%F %T') DONE ====="
