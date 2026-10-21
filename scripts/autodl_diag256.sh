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
#   MAXWAIT_H=3 (longest wait for a busy GPU)
#   FG=1 (foreground)  SMOKE=1 (CPU, N 32, first second, 2 iterations, every mode)  FORCE=1
# Re-running resumes: finished modes are skipped, an interrupted fit continues from its state.npz.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.." || exit 1
ROOT=$(pwd)
if [ "${SMOKE:-0}" = "1" ]; then
  OUT=${OUT:-results_diag_smoke}; DEVICE=${DEVICE:-cpu}; N=${N:-32}; EXTRA="--T 1"; ITERS=${ITERS:-2}; FG=1; PAR=1
  BUDGET_H=${BUDGET_H:-2}; NOWAIT=1; ARCH=${ARCH:-results_diag_smoke}
fi
OUT=${OUT:-results_diag}; LOG=$OUT/diag.log; LOCK=$OUT/.diag.lock; ARCH=${ARCH:-results_diag256}
mkdir -p "$OUT"
HAVE_FLOCK=0; command -v flock >/dev/null && HAVE_FLOCK=1
if [ "${FG:-0}" != "1" ] && [ "${_LATTE_BG:-}" != "1" ]; then
  if [ $HAVE_FLOCK = 1 ]; then exec 9>"$LOCK"; flock -n 9 || { echo "ABORT: a run is still working in $ROOT/$OUT"; exit 1; }; exec 9>&-; fi
  L0=$(( $(wc -c < "$LOG" 2>/dev/null || echo 0) + 1 ))
  _LATTE_BG=1 nohup bash "$SELF" "$@" >> "$LOG" 2>&1 &
  CH=$!
  echo "started in the background (pid $CH); log: $ROOT/$LOG"; echo "watch: tail -f $ROOT/$LOG"
  # stay a few minutes to report the checks that can stop the run early (files, environment, gradient self-test)
  echo "waiting for the start-up checks (at most 25 min; Ctrl-C is safe, the run continues) ..."
  for i in $(seq 1 150); do
    if [ -f "$OUT/selftest_ok" ]; then echo "self-test passed; the fits are running. Check back in a few hours."; exit 0; fi
    if ! kill -0 $CH 2>/dev/null; then echo "the run stopped early:"; tail -c +$L0 "$LOG" | tail -n 15; exit 1; fi
    if tail -c +$L0 "$LOG" 2>/dev/null | grep -q "the GPU is busy"; then echo "it is waiting for the GPU (another job is on it); it starts by itself when the GPU is free."; exit 0; fi
    sleep 10
  done
  echo "start-up checks still running after 25 min; see $ROOT/$LOG"; exit 0
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
  GI=${DEVICE#cuda:}; [[ "$GI" =~ ^[0-9]+$ ]] || GI=0
  W0=$(date +%s)
  while [ -n "$(nvidia-smi -i "$GI" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; do
    if [ $(( $(date +%s) - W0 )) -gt $(python3 -c "print(int(${MAXWAIT_H:-3} * 3600))") ]; then
      echo "WARNING: the GPU is still busy after ${MAXWAIT_H:-3} h; starting anyway"; break
    fi
    echo "$(date '+%T') the GPU is busy ($(nvidia-smi -i "$GI" --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')); waiting (NOWAIT=1 to skip)"
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
slot() {   # hours of fitting for each run of the next group, out of what is left of the budget
  $PY - "$BUDGET_H" "$T0" "$left_runs" "$PAR" "$RENDER_H" "$params_left" <<'PY'
import sys, time
B, t0, n, par, rh, pl = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
rem = B - (time.time() - t0) / 3600.0
groups = -(-n // par)
reserve = groups * rh + 0.3 * pl          # params also renders the three held-out pours
print(f"{max(0.25, (rem - reserve) / groups):.3f}")
PY
}
run_once() {   # $1 mode, $2 hours of fitting; hard stop at fitting + render + 45 min (a hang must not stall the rest)
  local hard
  hard=$($PY -c "print(int((float('$2') + ${RENDER_H} + (0.3 if '$1' == 'params' else 0.0) + 0.75) * 3600))")
  timeout -s TERM -k 120 "$hard" $PY -m latte_imex.diff.diag run --mode "$1" --init "$INIT" --out "$OUT/$1" --N "$N" \
      --device "$DEVICE" --iters "$ITERS" --time-h "$2" $EXTRA 2>&1 | grep -v '^Module ' >> "$OUT/$1.log"
  return ${PIPESTATUS[0]}
}
run_mode() {   # one retry if no done.json (it resumes from state.npz, or goes straight to the render)
  run_once "$1" "$2"; local rc=$?
  if [ ! -f "$OUT/$1/done.json" ]; then
    echo "$(date '+%T') $1: no done.json (rc $rc); one retry (resumes)"
    run_once "$1" 0.25; rc=$?
  fi
  return $rc
}
params_left=0; for m in $MODES; do [ "$m" = params ] && params_left=1; done
while [ $# -gt 0 ]; do
  H=$(slot)
  if [ "$PAR" -ge 2 ] && [ $# -ge 2 ]; then
    m1=$1; m2=$2; shift 2
    echo "$(date '+%T') $m1 + $m2 side by side, ${H} h each (logs $OUT/$m1.log, $OUT/$m2.log)"
    run_mode "$m1" "$H" & p1=$!
    sleep 90                            # stagger the start (CUDA module compilation, memory peak)
    run_mode "$m2" "$H" & p2=$!
    wait $p1; r1=$?; wait $p2; r2=$?
    [ -f "$OUT/$m1/done.json" ] || echo "WARNING: $m1 did not finish (rc $r1), see $OUT/$m1.log"
    [ -f "$OUT/$m2/done.json" ] || echo "WARNING: $m2 did not finish (rc $r2), see $OUT/$m2.log"
    left_runs=$((left_runs - 2))
  else
    m1=$1; shift
    echo "$(date '+%T') $m1, ${H} h (log $OUT/$m1.log)"
    run_mode "$m1" "$H"; r1=$?
    [ -f "$OUT/$m1/done.json" ] || echo "WARNING: $m1 did not finish (rc $r1), see $OUT/$m1.log"
    left_runs=$((left_runs - 1))
  fi
  for mm in $m1 ${m2:-}; do [ "$mm" = params ] && params_left=0; tail -n 4 "$OUT/$mm.log" 2>/dev/null; done; m2=""
done
$PY -m latte_imex.diff.diag summary --out "$OUT" 2>&1 | grep -v '^Module '
# two archives: everything but the videos (small, send first), and the videos
tar -czf "$ARCH.tar.gz" --exclude='*.mp4' --exclude='state.npz' --exclude='*.tmp' "$OUT" 2>/dev/null
find "$OUT" -name '*.mp4' -print0 | tar -czf "${ARCH}_videos.tar.gz" --null -T - 2>/dev/null
echo "===== $(date '+%F %T') done in $(( ($(date +%s) - T0) / 60 )) min; send back: $ROOT/$ARCH.tar.gz (and $ROOT/${ARCH}_videos.tar.gz) ====="
