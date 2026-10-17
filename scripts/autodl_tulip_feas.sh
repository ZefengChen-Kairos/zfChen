#!/bin/bash
# Tulip feasibility check on the AutoDL GPU (runs in the background; re-running resumes):
#   bash scripts/autodl_tulip_feas.sh          -> prints the log path and returns
#   tail -f results/tulip_feas.log             -> progress
# Result: results/tulip_feas/{summary.json, sheet.png, scatter.png} and results/tulip_feas_results.tar.gz (send this back)
cd "$(dirname "$0")/.."
LOG=results/tulip_feas.log
mkdir -p results
if [ "${_LATTE_BG:-}" != "1" ]; then
  _LATTE_BG=1 nohup bash "$0" "$@" >> "$LOG" 2>&1 &
  echo "started in the background (pid $!); log: $(pwd)/$LOG"
  echo "watch: tail -f $(pwd)/$LOG"
  exit 0
fi
set -eo pipefail
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4}
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0
# opencv >= 4.11 wheels pull numpy 2, which breaks the image's matplotlib (built for numpy 1): pin both
python -c "import cv2, matplotlib.pyplot" 2>/dev/null || python -m pip install -q "numpy<2" "opencv-python-headless==4.10.0.84"
python - <<'PY'
from numba import cuda
assert cuda.is_available(), "no CUDA device visible to numba: run bash scripts/autodl_setup.sh first"
print("GPU:", cuda.get_current_device().name)
PY
echo "===== $(date '+%F %T') tulip feasibility: Sobol 256 + refine 64 at N=128, confirm 12 at N=256 ====="
python -m latte_imex.rig.tulip_feas run --out results/tulip_feas --backend cuda "$@"
tar -czf results/tulip_feas_results.tar.gz results/tulip_feas/summary.json results/tulip_feas/records.jsonl \
    results/tulip_feas/config.json results/tulip_feas/sheet.png results/tulip_feas/scatter.png \
    $(ls results/tulip_feas/fields/C_*_N256.npy 2>/dev/null) "$LOG"
ls -lh results/tulip_feas_results.tar.gz
echo "===== $(date '+%F %T') DONE ====="
