#!/bin/bash
# Differentiable solver (NVIDIA Warp) on the AutoDL GPU: self-tests, then the gradient fit of the V0.5 closure,
# then the movies with the result (runs in the background; log results/diff.log).
#   bash scripts/autodl_diff.sh                 -> prints the log path and returns
#   FITS="A:tulip AB:tulip,leaf" (default, V1 plan 5.1)  N=128  ITERS=60  INIT=<closure json> (default: S1)
#   FIT_ARGS="--upwell --circ" (extra diff.fit flags, e.g. the area-conserving film)
# Send back: results/diff_results.tar.gz
# If the CUDA-graph capture fails on this driver: LATTE_NO_GRAPHS=1 bash scripts/autodl_diff.sh
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
export PYTHONUNBUFFERED=1
# the container sees all host cores: uncapped numba/BLAS thread pools oversubscribe the few CPUs it actually has
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} \
       OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
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
# V1 plan 5.1: calibration A = tulip (embossed tulip), A+B = tulip + leaf; frozen-parameter tests on the others.
# Start from the S1 closure (INIT=... to override; not from a fit that used a test pattern).
INIT=${INIT:-}
FITS=${FITS:-"A:tulip AB:tulip,leaf"}
echo "===== $(date '+%F %T') self-test 1/2: forward parity with the numpy solver (GPU, N=128, 300 frames, CUDA graphs + device-side PCG) ====="
python -m latte_imex.diff.test_parity --N 128 --frames 300 --device cuda:0
echo "===== $(date '+%F %T') self-test 2/2: adjoint vs finite differences (GPU, N=64, first 2 s) ====="
python -m latte_imex.diff.test_grad --N 64 --T 2.0 --device cuda:0 --sched 1
echo "===== $(date '+%F %T') benchmark: whole tulip pour, N=${N:-128} ====="
python -m latte_imex.diff.bench --pattern tulip --N ${N:-128} --device cuda:0
TARS=""
for f in $FITS; do
  tag=${f%%:*}; cal=${f#*:}; OUT=results/diff_fit_$tag
  echo "===== $(date '+%F %T') gradient fit $tag: calibration $cal, N=${N:-128}, ${ITERS:-60} iterations, start ${INIT:-S1} ====="
  python -m latte_imex.diff.fit --out "$OUT" --cal "$cal" --N ${N:-128} --iters ${ITERS:-60} ${INIT:+--init "$INIT"} ${FIT_ARGS:-} --device cuda:0
  echo "===== $(date '+%F %T') render $tag with the fitted closure (N=256) ====="
  python -m latte_imex.rig.fit_pipeline render --out "$OUT" --jobs ${JOBS:-5}
  TARS="$TARS $OUT/records.jsonl $OUT/best_closure.json $OUT/test.json $OUT/progress.png $OUT/sheet.png $OUT/video_strip.png $OUT/fields $OUT/render"
done
tar -czf results/diff_results.tar.gz $TARS "$LOG"
ls -lh results/diff_results.tar.gz
echo "===== $(date '+%F %T') DONE ====="
