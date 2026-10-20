#!/bin/bash
# update18: backpropagation training (exact adjoint of the Warp solver, Adam; latte_imex/diff/fit.py) of the new-baseline
# closure (V0.5 recalibrated by CMA-ES at N = 48 on the gap-preserving per-frame targets: v05n_AB_cmaes_N48.json) on
# tulip + leaf, a 256 -> 512 resolution ladder within BUDGET_H hours on one GPU, then the trained closure on all five
# pours: objective J at the N = 48 equivalent (diff/render512.py) and 512 x 512 movies.
#   bash scripts/autodl_bp512.sh                 -> starts in the background, prints the log path and returns
#   BUDGET_H=8  ITERS256=80  ITERS512=40  LR256=0.03  LR512=0.015  DEVICE=cuda:0  INIT=<closure json>
#   FG=1 (foreground)   SMOKE=1 (whole path on the CPU in minutes: N 32 / 48, first second of the pours)   FORCE=1
# Stages (a finished stage is skipped when the command is run again; a lock refuses a second start):
#   0 environment, parity self-test (Warp vs numpy, N 64, 200 frames), gradient smoke (fit.py, N 64, 1 s, 1 iteration)
#   1 N = 256: fit.py, both pours in parallel worker processes, until ITERS256 or its time slot ends
#   2 N = 512: fit.py from the stage-1 best, one pour at a time (the per-frame checkpoints of tulip are ~16 GB),
#     until ITERS512 or its time slot ends (fit.py writes best_closure.json at every new best, so a stop keeps it)
#   3 render512: J at the N = 48 equivalent + movies (<p>_sim512.mp4: one pixel per cell; <p>_compare.mp4: real |
#     target | trained N = 512 | baseline N = 48) for the five pours
#   4 if time is left: J of the untrained baseline at N = 512 (separates the resolution effect from the training effect)
#   pack results_bp512.tar.gz (send this back)
# The time slots are set from the start of this command so that the whole run ends within BUDGET_H.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.." || exit 1
ROOT=$(pwd)
if [ "${SMOKE:-0}" = "1" ]; then
  OUT=${OUT:-results_smoke}; DEVICE=${DEVICE:-cpu}; N1=${N1:-32}; N2=${N2:-48}; ITERS256=${ITERS256:-1}; ITERS512=${ITERS512:-1}
  TSIM="--T 1"; POURS=${POURS:-tulip}; FG=1; PAR_FRAMES=${PAR_FRAMES:-20}
fi
OUT=${OUT:-results}
LOG=$OUT/bp512.log
LOCK=$OUT/.bp512.lock
mkdir -p "$OUT"
HAVE_FLOCK=0; command -v flock >/dev/null && HAVE_FLOCK=1
if [ "${FG:-0}" != "1" ] && [ "${_LATTE_BG:-}" != "1" ]; then
  if [ $HAVE_FLOCK = 1 ]; then
    exec 9>"$LOCK"
    flock -n 9 || { echo "ABORT: a bp512 run is still working in $ROOT/$OUT (see $ROOT/$LOG); not started"; exit 1; }
    exec 9>&-
  fi
  _LATTE_BG=1 nohup bash "$SELF" "$@" >> "$LOG" 2>&1 &
  echo "started in the background (pid $!); log: $ROOT/$LOG"
  echo "watch: tail -f $ROOT/$LOG"
  exit 0
fi
[ "${FG:-0}" = "1" ] && exec > >(tee -a "$LOG") 2>&1
if [ $HAVE_FLOCK = 1 ]; then
  exec 9>"$LOCK"
  flock -n 9 || { echo "ABORT: a bp512 run is still working in $ROOT/$OUT; this one stops"; exit 1; }
fi
set -o pipefail
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"          # this package's latte_imex, never another checkout
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} \
       OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
PY=${PY:-python}; command -v "$PY" >/dev/null || PY=python3
DEVICE=${DEVICE:-cuda:0}; N1=${N1:-256}; N2=${N2:-512}
INIT=${INIT:-latte_imex/data/closures/v05n_AB_cmaes_N48.json}
BUDGET_H=${BUDGET_H:-8}; ITERS256=${ITERS256:-80}; ITERS512=${ITERS512:-40}; LR256=${LR256:-0.03}; LR512=${LR512:-0.015}
POURS=${POURS:-tulip,leaf,heart,push_heart,swan}; TSIM=${TSIM:-}
T0=$(date +%s); [ -f "$OUT/.t0" ] && [ "${KEEP_T0:-0}" = "1" ] && T0=$(cat "$OUT/.t0"); echo $T0 > "$OUT/.t0"
END=$(awk "BEGIN{printf \"%d\", $T0 + $BUDGET_H * 3600}")
left() { echo $((END - $(date +%s))); }
hm() { printf "%dh%02dm" $(($1 / 3600)) $((($1 % 3600) / 60)); }
die() { echo "ABORT: $*"; exit 1; }
echo "===== $(date '+%F %T') bp512 in $ROOT: N $N1 -> $N2, iters $ITERS256 / $ITERS512, lr $LR256 / $LR512, device $DEVICE," \
     "init $INIT, budget ${BUDGET_H} h (ends $(date -d @$END '+%F %T')) ====="

# ---- 0. package integrity, environment, self-tests
if [ -f MANIFEST.sha256 ]; then
  sha256sum -c --quiet MANIFEST.sha256 || { [ "${FORCE:-0}" = "1" ] || die "files differ from MANIFEST.sha256 (FORCE=1 to go on)"; }
fi
[ -f /etc/network_turbo ] && source /etc/network_turbo || true          # AutoDL academic proxy (pip)
$PY -c "import warp" 2>/dev/null || $PY -m pip install -q "warp-lang==1.18.0"
$PY -c "import scipy" 2>/dev/null || $PY -m pip install -q scipy
$PY -c "import matplotlib" 2>/dev/null || $PY -m pip install -q matplotlib
$PY -c "import cv2" 2>/dev/null || $PY -m pip install -q "opencv-python-headless==4.10.0.84"
$PY -c "import imageio, imageio_ffmpeg, PIL" 2>/dev/null || $PY -m pip install -q imageio imageio-ffmpeg pillow
$PY -c "import numba" 2>/dev/null || $PY -m pip install -q numba "numpy==$($PY -c 'import numpy; print(numpy.__version__)')" \
  || echo "WARNING: numba could not be installed next to this numpy"
$PY -c "import numba" 2>/dev/null || { echo "WARNING: no numba: parity self-test shortened to 30 frames"; PAR_FRAMES=${PAR_FRAMES:-30}; }
PAR_FRAMES=${PAR_FRAMES:-200}
{ date '+%F %T'; uname -a; command -v nvidia-smi >/dev/null && nvidia-smi; $PY - "$DEVICE" <<'PY'
import os, sys
import numpy, scipy, cv2, imageio
import warp as wp
wp.init()
import latte_imex
print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__, "cv2", cv2.__version__,
      "warp", wp.__version__)
print("latte_imex from", os.path.dirname(latte_imex.__file__))
print("CUDA devices seen by warp:", wp.get_cuda_device_count(),
      "| driver", wp.get_cuda_driver_version() if wp.is_cuda_available() else None,
      "| toolkit", wp.get_cuda_toolkit_version() if wp.is_cuda_available() else None)
if os.path.realpath(os.path.dirname(latte_imex.__file__)) != os.path.realpath(os.path.join(os.getcwd(), "latte_imex")):
    sys.exit("latte_imex is imported from another checkout")
if sys.argv[1].startswith("cuda") and not wp.is_cuda_available():
    sys.exit("warp sees no CUDA device (driver older than this warp build's CUDA toolkit? see the lines above)")
PY
} > "$OUT/env.txt" 2>&1
rc=$?; grep -E '^(python|latte_imex|CUDA devices)|NVIDIA-SMI|RTX|MiB' "$OUT/env.txt"
[ $rc -eq 0 ] || die "environment check failed (see $OUT/env.txt)"
if [ ! -f "$OUT/selftest.ok" ]; then
  echo "===== $(date '+%F %T') self-test 1: Warp vs numpy solver (heart, N=64, $PAR_FRAMES frames, $DEVICE) ====="
  $PY -m latte_imex.diff.test_parity --N 64 --frames "$PAR_FRAMES" --device "$DEVICE" 2>&1 | grep -v '^Module ' \
    | tee "$OUT/selftest_parity.txt"
  if ! grep -q '^PASS' "$OUT/selftest_parity.txt"; then
    [ "${FORCE:-0}" = "1" ] && echo "WARNING: parity self-test did not pass; FORCE=1, going on" \
      || die "parity self-test did not pass ($OUT/selftest_parity.txt); send that file back (FORCE=1 to run anyway)"
  fi
  echo "===== $(date '+%F %T') self-test 2: one backpropagation step (fit.py, tulip+leaf, N=64, first second) ====="
  $PY -m latte_imex.diff.fit --out "$OUT/selftest_fit" --cal tulip,leaf --N 64 --iters 1 --T 1 --init "$INIT" --e0 world \
      --device "$DEVICE" --workers 0 --no-test 2>&1 | grep -v '^Module ' | tee "$OUT/selftest_fit.txt"
  [ -f "$OUT/selftest_fit/best_closure.json" ] || die "the backpropagation smoke failed ($OUT/selftest_fit.txt)"
  touch "$OUT/selftest.ok"
fi

SAMPLER=""
if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 >> "$OUT/gpu_usage.csv" 2>/dev/null 9>&- &
  SAMPLER=$!
  trap '[ -n "$SAMPLER" ] && kill $SAMPLER 2>/dev/null' EXIT
fi

# time reserved after training: the movies + J of the five pours at N2 (~10 min a pour on a 4090 at 512, estimate)
RES_MOV=${RES_MOV:-$((75 * 60))}; [ "${SMOKE:-0}" = "1" ] && RES_MOV=60
RES_512=${RES_512:-$((150 * 60))}; [ "${SMOKE:-0}" = "1" ] && RES_512=60      # at least this long for stage 2
fit_stage() {  # name N iters lr init workers slot_seconds
  local name=$1 N=$2 it=$3 lr=$4 init=$5 wk=$6 slot=$7 d="$OUT/$1"
  [ -f "$d/DONE" ] && { echo "===== $name done before: skipped"; return 0; }
  [ "$slot" -gt 300 ] || [ "${SMOKE:-0}" = "1" ] || { echo "===== $name: only $(hm $slot) left, skipped"; return 0; }
  echo "===== $(date '+%F %T') $name: N=$N, up to $it iterations, lr $lr, time slot $(hm $slot), from $init ====="
  mkdir -p "$d"
  # a stopped stage restarts from its own best (fit.py keeps best_closure.json at every new best)
  [ -f "$d/best_closure.json" ] && cp "$d/best_closure.json" "$d/resume_from.json" && init="$d/resume_from.json"
  timeout -s TERM -k 60 "$slot" $PY -m latte_imex.diff.fit --out "$d" --cal tulip,leaf --N "$N" --iters "$it" --lr "$lr" \
      --init "$init" --e0 world --device "$DEVICE" --workers "$wk" --no-test $TSIM 2>&1 | grep -v '^Module ' | tee -a "$d/log.txt"
  rc=${PIPESTATUS[0]}
  [ $rc -eq 124 ] && echo "===== $name: time slot used up (that is planned); best so far kept"
  [ -f "$d/best_closure.json" ] || die "$name produced no closure (rc $rc; $d/log.txt). If the log says out of memory at N=$N: rerun the same command with N2=384 (or N2=256 ITERS512=0)"
  touch "$d/DONE"
}
# ---- 1. N1 (both pours in parallel worker processes)
fit_stage train_N$N1 "$N1" "$ITERS256" "$LR256" "$INIT" -1 $(( $(left) - RES_512 - RES_MOV ))
# ---- 2. N2 from the N1 best (one pour at a time: memory)
fit_stage train_N$N2 "$N2" "$ITERS512" "$LR512" "$OUT/train_N$N1/best_closure.json" 0 $(( $(left) - RES_MOV ))
BEST="$OUT/train_N$N2/best_closure.json"; [ -f "$BEST" ] || BEST="$OUT/train_N$N1/best_closure.json"
# ---- 3. J at the N = 48 equivalent + movies of the trained closure
if [ ! -f "$OUT/eval_trained/DONE" ]; then
  echo "===== $(date '+%F %T') evaluation + movies: $BEST at N=$N2, five pours ($(hm $(left)) left) ====="
  timeout -s TERM -k 60 $(( $(left) - 120 )) $PY -m latte_imex.diff.render512 --closure "$BEST" --N "$N2" --out "$OUT/eval_trained" \
      --pours "$POURS" --device "$DEVICE" --baseline "$INIT" --label "反传训练 N=$N2" --blabel "新基线 N=48（CMA）" $TSIM \
      2>&1 | grep -v '^Module ' | tee -a "$OUT/eval_trained.log"
  [ ${PIPESTATUS[0]} -eq 0 ] && touch "$OUT/eval_trained/DONE"
fi
# ---- 4. the untrained baseline at N2 (resolution effect), only if time is left
if [ ! -f "$OUT/eval_start/DONE" ] && [ "$(left)" -gt 2400 ]; then
  echo "===== $(date '+%F %T') J of the untrained baseline at N=$N2 ($(hm $(left)) left) ====="
  timeout -s TERM -k 60 $(( $(left) - 120 )) $PY -m latte_imex.diff.render512 --closure "$INIT" --N "$N2" --out "$OUT/eval_start" \
      --pours "$POURS" --device "$DEVICE" --no-movie $TSIM 2>&1 | grep -v '^Module ' | tee -a "$OUT/eval_start.log"
  [ ${PIPESTATUS[0]} -eq 0 ] && touch "$OUT/eval_start/DONE"
fi
# ---- summary + pack
$PY - "$OUT" "$N1" "$N2" <<'PY' | tee "$OUT/summary.txt"
import json, os, sys
out, n1, n2 = sys.argv[1:4]
base = dict(tulip=0.718, leaf=0.731, heart=0.657, push_heart=0.446, swan=0.566)    # v05n at N = 48 (cmafit, fit17)
def ev(d):
    fn = os.path.join(out, d, "eval.json")
    return json.load(open(fn)) if os.path.exists(fn) else {}
tr, st = ev("eval_trained"), ev("eval_start")
print("J = 0.25 IoU_n + 0.75 structure_n at the N = 48 equivalent (1 = as close as the video to itself 0.25 s later)")
print(f"{'pour':11s} {'baseline N48':>12s} {'baseline N' + n2:>14s} {'trained N' + n2:>13s}")
for p in base:
    f = lambda r: f"{r[p]['J']:.3f}" if p in r else "-"
    print(f"{p:11s} {base[p]:12.3f} {f(st):>14s} {f(tr):>13s}")
for n in (n1, n2):
    fn = os.path.join(out, f"train_N{n}", "records.jsonl")
    if os.path.exists(fn):
        R = [json.loads(l) for l in open(fn) if l.strip()]
        if R:
            b = max(R, key=lambda r: r["viou"])
            print(f"train N{n}: {len(R)} iterations, {sum(r['wall_s'] for r in R) / 3600:.2f} h, mean {R[-1]['wall_s']:.0f} s/it; "
                  f"best (video IoU) it {b['it']}: vIoU {b['viou']:.3f} (start {R[0]['viou']:.3f})")
PY
tar -czf "$OUT/../results_bp512.tar.gz" --exclude='*.npy' -C "$(dirname "$OUT")" "$(basename "$OUT")" 2>/dev/null \
  || tar -czf results_bp512.tar.gz "$OUT"
echo "===== $(date '+%F %T') done ($(hm $(( $(date +%s) - T0 ))) in total); send back: $(cd "$(dirname "$OUT")" && pwd)/results_bp512.tar.gz ====="
