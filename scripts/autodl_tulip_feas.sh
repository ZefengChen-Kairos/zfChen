#!/bin/bash
# Tulip feasibility check on the AutoDL GPU (runs in the background; re-running resumes):
#   bash scripts/autodl_tulip_feas.sh          -> prints the log path and returns
#   tail -f results/tulip_feas.log             -> progress
# Result: results/tulip_feas_kona/{summary.json, sheet.png, scatter.png} and results/tulip_feas_kona_results.tar.gz
# (send this back).  Flow: catalogue pitcher, 165 g poured (FEAS_FLOW=kona, the default of tulip_feas.Pour).
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
echo "===== $(date '+%F %T') tulip feasibility: catalogue-pitcher flow; Sobol 256 + refine 64 at N=128, confirm 12 at N=256 ====="
OUT=${OUT:-results/tulip_feas_kona}
python -m latte_imex.rig.tulip_feas run --out "$OUT" --backend cuda "$@"
tar -czf "${OUT}_results.tar.gz" "$OUT/summary.json" "$OUT/records.jsonl" "$OUT/config.json" "$OUT/sheet.png" \
    "$OUT/scatter.png" $(ls "$OUT"/fields/C_*_N256.npy 2>/dev/null) "$LOG"
ls -lh "${OUT}_results.tar.gz"
echo "===== $(date '+%F %T') DONE ====="
