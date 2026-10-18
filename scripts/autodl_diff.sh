#!/bin/bash
# Differentiable solver (NVIDIA Warp) on the AutoDL GPU: self-tests, then the gradient fit of the V0.5 closure,
# then the movies with the result (runs in the background; log results/diff.log).
#   bash scripts/autodl_diff.sh                 -> prints the log path and returns
#   CAL=tulip,heart N=128 ITERS=60 INIT=results/fit/best_closure.json (default if it exists, else the S1 closure)
# Send back: results/diff_results.tar.gz
cd "$(dirname "$0")/.."
LOG=results/diff.log
mkdir -p results
if [ "${_LATTE_BG:-}" != "1" ]; then
  _LATTE_BG=1 nohup bash "$0" "$@" >> "$LOG" 2>&1 &
  echo "started in the background (pid $!); log: $(pwd)/$LOG"
  echo "watch: tail -f $(pwd)/$LOG"
  exit 0
fi
set -eo pipefail
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
export PYTHONWARNINGS=ignore
python -c "import warp" 2>/dev/null || python -m pip install -q warp-lang
python -c "import cv2, matplotlib.pyplot, imageio" 2>/dev/null || \
  python -m pip install -q "numpy<2" "opencv-python-headless==4.10.0.84" imageio imageio-ffmpeg
python - <<'PY'
import warp as wp
wp.init()
print("warp", wp.__version__, "devices", wp.get_devices())
assert wp.is_cuda_available(), "warp sees no CUDA device"
PY
OUT=${OUT:-results/diff_fit}
CAL=${CAL:-tulip,heart}
INIT=${INIT:-}
[ -z "$INIT" ] && [ -f results/fit/best_closure.json ] && INIT=results/fit/best_closure.json
echo "===== $(date '+%F %T') self-test 1/2: forward parity with the numpy solver (GPU, N=128, 300 frames) ====="
python -m latte_imex.diff.test_parity --N 128 --frames 300 --device cuda:0
echo "===== $(date '+%F %T') self-test 2/2: adjoint vs finite differences (GPU, N=64, first 2 s) ====="
python -m latte_imex.diff.test_grad --N 64 --T 2.0 --device cuda:0 --sched 1
echo "===== $(date '+%F %T') gradient fit: $CAL, N=${N:-128}, ${ITERS:-60} iterations, start ${INIT:-S1} ====="
python -m latte_imex.diff.fit --out "$OUT" --cal "$CAL" --N ${N:-128} --iters ${ITERS:-60} ${INIT:+--init "$INIT"} --device cuda:0
echo "===== $(date '+%F %T') render with the fitted closure (N=256) ====="
python -m latte_imex.rig.fit_pipeline render --out "$OUT" --jobs ${JOBS:-5}
tar -czf results/diff_results.tar.gz "$OUT/records.jsonl" "$OUT/best_closure.json" "$OUT/progress.png" "$OUT/sheet.png" \
    "$OUT/fields" "$OUT/render" "$LOG"
ls -lh results/diff_results.tar.gz
echo "===== $(date '+%F %T') DONE ====="
