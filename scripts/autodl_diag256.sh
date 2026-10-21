#!/bin/bash
# update20: three-way diagnosis at N (default 256) -- is the unopened tulip the surface model's, its parameters', or the
# pour model's fault?  Four gradient fits on tulip + leaf from one start closure (latte_imex/diff/diag.py):
#   both_wide  closure free (wide box) + inlet free in a box 2.5-3x the measurement error   -> can the model at all
#   both       closure free (wide box) + inlet free within the measurement error
#   params     closure free (wide box), inlet as measured            (+ frozen test on heart / push_heart / swan)
#   inlet      closure fixed, inlet free within the measurement error -> the pitcher taken out
# then results_diag/summary.md (decision rule of README_diag256.md) and two archives to send back.
#   bash scripts/autodl_diag256.sh          -> starts in the background, prints the log path and returns
#   N=256 DEVICE=cuda:0 BUDGET_H=8 ITERS=80 PAR=auto (2 runs at once with >= 45 GB of GPU memory, else 1)
#   INIT=closures/bp256_trained.json  MODES="both_wide both params inlet"  NOWAIT=1 (do not wait for a busy GPU)
#   FG=1 (foreground)  SMOKE=1 (CPU, N 32, first second, 2 iterations, every mode)  FORCE=1
# Re-running resumes: finished modes are skipped, an interrupted fit continues from its state.npz.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.." || exit 1
ROOT=$(pwd)
if [ "${SMOKE:-0}" = "1" ]; then
  OUT=${OUT:-results_diag_smoke}; DEVICE=${DEVICE:-cpu}; N=${N:-32}; EXTRA="--T 1"; ITERS=${ITERS:-2}; FG=1; PAR=1
  BUDGET_H=${BUDGET_H:-2}; NOWAIT=1
fi
OUT=${OUT:-results_diag}; LOG=$OUT/diag.log; LOCK=$OUT/.diag.lock
mkdir -p "$OUT"
HAVE_FLOCK=0; command -v flock >/dev/null && HAVE_FLOCK=1
if [ "${FG:-0}" != "1" ] && [ "${_LATTE_BG:-}" != "1" ]; then
  if [ $HAVE_FLOCK = 1 ]; then exec 9>"$LOCK"; flock -n 9 || { echo "ABORT: a run is still working in $ROOT/$OUT"; exit 1; }; exec 9>&-; fi
  _LATTE_BG=1 nohup bash "$SELF" "$@" >> "$LOG" 2>&1 &
  echo "started in the background (pid $!); log: $ROOT/$LOG"; echo "watch: tail -f $ROOT/$LOG"; exit 0
fi
[ "${FG:-0}" = "1" ] && exec > >(tee -a "$LOG") 2>&1
if [ $HAVE_FLOCK = 1 ]; then exec 9>"$LOCK"; flock -n 9 || { echo "ABORT: a run is still working in $ROOT/$OUT"; exit 1; }; fi
set -o pipefail
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PY=${PY:-python}; command -v "$PY" >/dev/null || PY=python3
DEVICE=${DEVICE:-cuda:0}; N=${N:-256}; BUDGET_H=${BUDGET_H:-8}; ITERS=${ITERS:-80}; EXTRA=${EXTRA:-}
INIT=${INIT:-closures/bp256_trained.json}
[ -f "$INIT" ] || INIT=latte_imex/data/closures/v05n_AB_cmaes_N48.json
MODES=${MODES:-both_wide both params inlet}
RENDER_H=${RENDER_H:-0.3}          # reserved per run for the final forward runs, scores, stills and movies
die() { echo "ABORT: $*"; exit 1; }
T0=$(date +%s)
echo "===== $(date '+%F %T') update20 (diagnosis) in $ROOT: N $N, device $DEVICE, modes $MODES, start $INIT, budget ${BUDGET_H} h ====="
if [ -f MANIFEST.sha256 ]; then
  sha256sum -c --quiet MANIFEST.sha256 || { [ "${FORCE:-0}" = "1" ] || die "files differ from MANIFEST.sha256 (FORCE=1 to go on)"; }
fi
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
$PY -c "import warp" 2>/dev/null || $PY -m pip install -q "warp-lang==1.18.0"
$PY -c "import scipy" 2>/dev/null || $PY -m pip install -q scipy
$PY -c "import cv2" 2>/dev/null || $PY -m pip install -q "opencv-python-headless==4.10.0.84"
$PY -c "import imageio, imageio_ffmpeg, PIL, matplotlib" 2>/dev/null || $PY -m pip install -q imageio imageio-ffmpeg pillow matplotlib
$PY - "$DEVICE" <<'PY' || die "environment check failed"
import os, sys
import warp as wp
wp.init()
import latte_imex
print("latte_imex from", os.path.dirname(latte_imex.__file__), "| warp", wp.__version__, "| CUDA devices", wp.get_cuda_device_count())
if os.path.realpath(os.path.dirname(latte_imex.__file__)) != os.path.realpath(os.path.join(os.getcwd(), "latte_imex")):
    sys.exit("latte_imex is imported from another checkout")
if sys.argv[1].startswith("cuda") and not wp.is_cuda_available():
    sys.exit("warp sees no CUDA device")
PY
# a GPU busy with another job (e.g. update19 still rendering) slows both and may run out of memory: wait for it
if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null && [ "${NOWAIT:-0}" != "1" ]; then
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; do
    echo "$(date '+%T') the GPU is busy ($(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')); waiting (NOWAIT=1 to skip)"
    sleep 120
  done
  T0=$(date +%s)                   # the budget starts when the GPU is ours
fi
if [ "${PAR:-auto}" = "auto" ]; then
  PAR=1
  if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null; then
    MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${DEVICE#cuda:}" 2>/dev/null | head -1)
    [ -n "$MEM" ] && [ "$MEM" -ge 45000 ] && PAR=2
    echo "GPU memory ${MEM:-?} MiB -> $PAR run(s) at a time"
  fi
fi
SAMPLER=""
if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 >> "$OUT/gpu_usage.csv" 2>/dev/null 9>&- &
  SAMPLER=$!; trap '[ -n "$SAMPLER" ] && kill $SAMPLER 2>/dev/null' EXIT
fi
# --- self-test: adjoint vs finite differences through the inlet correction (and two closure numbers)
GN=64; GT=1.5; [ "${SMOKE:-0}" = "1" ] && { GN=32; GT=0.5; }
if [ ! -f "$OUT/selftest_ok" ]; then
  $PY -m latte_imex.diff.diag gradcheck --init "$INIT" --N $GN --T $GT --device "$DEVICE" 2>&1 | grep -v '^Module ' | tee "$OUT/selftest_gradcheck.txt"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then [ "${FORCE:-0}" = "1" ] || die "gradient self-test failed; send $ROOT/$OUT/selftest_gradcheck.txt"; fi
  touch "$OUT/selftest_ok"
fi
# --- the fits: time slot per run from what is left of the budget
set -- $MODES
left_runs=$#
slot() {   # hours for the next group of $1 runs running side by side, out of the remaining budget
  $PY - "$BUDGET_H" "$T0" "$left_runs" "$PAR" "$RENDER_H" <<'PY'
import sys, time
B, t0, n, par, rh = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
rem = B - (time.time() - t0) / 3600.0
groups = -(-n // par)
print(f"{max(0.25, (rem - groups * rh) / groups):.3f}")
PY
}
run_mode() {   # $1 mode, $2 hours
  $PY -m latte_imex.diff.diag run --mode "$1" --init "$INIT" --out "$OUT/$1" --N "$N" --device "$DEVICE" \
      --iters "$ITERS" --time-h "$2" $EXTRA 2>&1 | grep -v '^Module ' >> "$OUT/$1.log"
  return ${PIPESTATUS[0]}
}
while [ $# -gt 0 ]; do
  H=$(slot)
  if [ "$PAR" -ge 2 ] && [ $# -ge 2 ]; then
    m1=$1; m2=$2; shift 2
    echo "$(date '+%T') $m1 + $m2 side by side, ${H} h each (logs $OUT/$m1.log, $OUT/$m2.log)"
    run_mode "$m1" "$H" & p1=$!
    sleep 90                            # stagger the start (CUDA module compilation, memory peak)
    run_mode "$m2" "$H" & p2=$!
    wait $p1; r1=$?; wait $p2; r2=$?
    [ $r1 -eq 0 ] || echo "WARNING: $m1 failed (rc $r1), see $OUT/$m1.log"
    [ $r2 -eq 0 ] || echo "WARNING: $m2 failed (rc $r2), see $OUT/$m2.log"
    left_runs=$((left_runs - 2))
  else
    m1=$1; shift
    echo "$(date '+%T') $m1, ${H} h (log $OUT/$m1.log)"
    run_mode "$m1" "$H"; r1=$?
    [ $r1 -eq 0 ] || echo "WARNING: $m1 failed (rc $r1), see $OUT/$m1.log"
    left_runs=$((left_runs - 1))
  fi
  tail -n 4 "$OUT/$m1.log" 2>/dev/null; [ -n "${m2:-}" ] && tail -n 4 "$OUT/$m2.log" 2>/dev/null; m2=""
done
$PY -m latte_imex.diff.diag summary --out "$OUT" 2>&1 | grep -v '^Module '
# two archives: everything but the videos (small, send first), and the videos
tar -czf "results_diag256.tar.gz" --exclude='*.mp4' --exclude='state.npz' "$OUT" 2>/dev/null
find "$OUT" -name '*.mp4' -print0 | tar -czf "results_diag256_videos.tar.gz" --null -T - 2>/dev/null
echo "===== $(date '+%F %T') done in $(( ($(date +%s) - T0) / 60 )) min; send back: $ROOT/results_diag256.tar.gz (and $ROOT/results_diag256_videos.tar.gz) ====="
