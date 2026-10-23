#!/bin/bash
# update21 (mi27, SPEC27 rev 2 section 6; README_diag21.md): the new pour model -- the measured inlet MI -- and a re-fit.
# Does the tulip open, for the physics model S1kq and for the baseline v05n?  Gradient fits on tulip + leaf (pose v2,
# N 256, latte_imex/diff/diag.py), in the user's order:
#   1 s1kq/params       S1kq start mcd_S1kq_mi_N96, the five F entries free (pset S1), inlet MI-R as measured
#                       (+ frozen test on heart / push_heart / swan under MI-L, + bridges (i) and (ii))
#   2 v05n/params       v05n start bp256_mi, 9 numbers free (wide box), inlet MI-R as measured (+ held-out, bridges)
#   3 v05n_ctrl/params  bp256_trained, same box, pose v2 + the record inlet R1 (separates pose from inlet) (+ held-out R1)
#   4 s1kq/inlet        S1kq fixed, inlet free in the 'meas' box                    } oracle runs: the inlet corrections
#   5 v05n/inlet  6 v05n/both  7 v05n/both_wide ('meas_wide')                    } are diagnostic only (PREREG27_2 D11)
# Start-up (before the fits; reported by the foreground part): file / closure checks, the two gradient self-tests
# (record with LATTE_INLET / LATTE_FOOT unset; MI with the S1 keys and the 'meas' box), SMOKE (every run once at N 32,
# one iteration, on the device), one N 256 S1kq and one N 256 v05n iteration under the measured inlet (their ratio sets
# the S1kq time slots).  Then results_diag21/<model>/<mode>/,
# results_diag21/<model>/summary.md, results_diag21/summary.md (Chinese) and two archives to send back.
#   bash scripts/autodl_diag21.sh          -> starts in the background, reports the start-up checks, returns
#   N=256 DEVICE=cuda:0 BUDGET_H=15 ITERS=80 MAXWAIT_H=3 (longest wait for a busy GPU) NOWAIT=1 (do not wait)
#   RUNS="s1kq/params v05n/params ..." (subset / order)  MODELS="s1kq v05n v05n_ctrl" (filter)
#   FG=1 (foreground)  FORCE=1 (go on after a failed check / SMOKE)  SKIP_SMOKE=1  SKIP_TIMING=1
#   LOCAL_SMOKE=1      CPU dress rehearsal of the whole path for testing the package: N 32, T 1 s (3 s for the params runs
#                      and their bridges: swan's scoring needs it), 2 iterations, no GPU wait, foreground, into
#                      results_diag21_localsmoke/ (+ .tar.gz); SMOKE runs only "v05n/both s1kq/inlet" there
# Re-running resumes: finished runs and checks are skipped, an interrupted fit continues from its state.npz.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.." || exit 1
ROOT=$(pwd)
if [ "${LOCAL_SMOKE:-0}" = "1" ]; then
  OUT=${OUT:-results_diag21_localsmoke}; ARCH=${ARCH:-results_diag21_localsmoke}; DEVICE=${DEVICE:-cpu}; N=${N:-32}
  ITERS=${ITERS:-2}; T_BOX=${T_BOX:-1}; T_PAR=${T_PAR:-3}; FG=1; NOWAIT=1; BUDGET_H=${BUDGET_H:-6}
  SMOKE_RUNS=${SMOKE_RUNS:-v05n/both s1kq/inlet}; SMOKE_PAR=${SMOKE_PAR:-2}; TIMING_K=${TIMING_K:-1}
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2}
fi
OUT=${OUT:-results_diag21}; ARCH=${ARCH:-results_diag21}; LOG=$OUT/diag21.log; LOCK=$OUT/.diag21.lock
mkdir -p "$OUT"
HAVE_FLOCK=0; command -v flock >/dev/null && HAVE_FLOCK=1
fsize() { if [ -f "$1" ]; then wc -c < "$1"; else echo 0; fi; }
if [ "${FG:-0}" != "1" ] && [ "${_LATTE_BG:-}" != "1" ]; then
  if [ $HAVE_FLOCK = 1 ]; then exec 9>"$LOCK"; flock -n 9 || { echo "ABORT: a run is still working in $ROOT/$OUT"; exit 1; }; exec 9>&-; fi
  L0=$(( $(fsize "$LOG") + 1 ))
  S0=$(( $(fsize "$OUT/startup.txt") + 1 ))
  TOKEN="$(date +%s)-$$"
  _LATTE_BG=1 _LATTE_TOKEN="$TOKEN" nohup bash "$SELF" "$@" >> "$LOG" 2>&1 &
  CH=$!
  echo "started in the background (pid $CH); log: $ROOT/$LOG"; echo "watch: tail -f $ROOT/$LOG"
  # stay to report the start-up checks that can stop the run early (files, gradient self-tests, SMOKE, S1kq timing)
  echo "waiting for the start-up checks (about 45 min, at most 90; Ctrl-C is safe, the run continues) ..."
  for i in $(seq 1 540); do
    tail -c +$S0 "$OUT/startup.txt" 2>/dev/null; S0=$(( $(fsize "$OUT/startup.txt") + 1 ))
    if [ "$(cat "$OUT/startup_done" 2>/dev/null)" = "$TOKEN" ]; then
      echo "start-up checks done; the fits are running. Check back in a few hours ($OUT/summary.md)."; exit 0; fi
    if ! kill -0 $CH 2>/dev/null; then echo "the run stopped early:"; tail -c +$L0 "$LOG" | grep -v '^Module ' | tail -n 15; exit 1; fi
    if tail -c +$L0 "$LOG" 2>/dev/null | grep -q "the GPU is busy"; then
      echo "it is waiting for the GPU (another job is on it); it starts by itself when the GPU is free."; exit 0; fi
    sleep 10
  done
  echo "start-up checks still running after 90 min; see $ROOT/$LOG"; exit 0
fi
[ "${FG:-0}" = "1" ] && exec > >(tee -a "$LOG") 2>&1
if [ $HAVE_FLOCK = 1 ]; then exec 9>"$LOCK"; flock -n 9 || { echo "ABORT: a run is still working in $ROOT/$OUT"; exit 1; }; fi
set -o pipefail
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export LATTE_POSE=v2                                # pose v2 (the default track; set explicitly for every run)
PY=${PY:-python}; command -v "$PY" >/dev/null || PY=python3
DEVICE=${DEVICE:-cuda:0}; N=${N:-256}; BUDGET_H=${BUDGET_H:-15}; ITERS=${ITERS:-80}
RENDER_H=${RENDER_H:-0.3}      # per run: the final forward runs, scores, stills and movies
HELDOUT_H=${HELDOUT_H:-0.3}    # params runs: the three held-out pours
BRIDGE_H=${BRIDGE_H:-0.2}      # MI params runs: bridges (i) tulip + leaf under MI-L, (ii) held-out pours under R1
FIT_H=${FIT_H:-1.6}; PFIT_H=${PFIT_H:-1.35}   # planned hours of fitting: box modes / params modes (v05n at ~90 s / it)
GC_DEVICE=${GC_DEVICE:-cpu}    # the gradient self-tests run the tested configuration of mi27 T7 (cpu, N 32, T 1, eps 1e-7)
SMOKE_N=${SMOKE_N:-32}; SMOKE_PAR=${SMOKE_PAR:-2}; TIMING_K=${TIMING_K:-2}
T_ARGS_BOX=""; T_ARGS_PAR=""; [ -n "${T_BOX:-}" ] && T_ARGS_BOX="--T $T_BOX"; [ -n "${T_PAR:-}" ] && T_ARGS_PAR="--T $T_PAR"
ORDER="s1kq/params v05n/params v05n_ctrl/params s1kq/inlet v05n/inlet v05n/both v05n/both_wide"
MODELS=${MODELS:-s1kq v05n v05n_ctrl}
RUNS_IN=${RUNS:-$ORDER}; RUNS=""
for r in $RUNS_IN; do case " $MODELS " in *" ${r%/*} "*) RUNS="$RUNS $r";; esac; done
RUNS=${RUNS# }
SMOKE_RUNS=${SMOKE_RUNS:-$RUNS}
die() { echo "ABORT: $*"; echo "$(date '+%T') ABORT: $*" >> "$OUT/startup.txt"; exit 1; }
say() { echo "$*"; echo "$(date '+%T') $*" >> "$OUT/startup.txt"; }

# ---------------------------------------------------------------------- the three models (SPEC27 6.1)
model_cfg() {   # sets INIT_M, PSET_M, MI_M for model $1
  case $1 in
    s1kq)      INIT_M=latte_imex/data/closures/mcd_S1kq_mi_N96.json; PSET_M=S1;   MI_M=1 ;;  # inlet 2 = MI-replay
    v05n)      INIT_M=latte_imex/data/closures/bp256_mi.json;        PSET_M=v05n; MI_M=1 ;;  # bp256_trained + inlet 2
    v05n_ctrl) INIT_M=closures/bp256_trained.json;                   PSET_M=v05n; MI_M=0 ;;  # record inlet R1
    *) die "unknown model $1" ;;
  esac
}
model_env() {   # the inlet environment of model $1 (in a subshell): MI -> LATTE_FOOT=mi (the inlet comes from the
  model_cfg "$1"  # closure key 'inlet' 2); record -> LATTE_INLET and LATTE_FOOT unset (R1: record, E0 world, v1 feet)
  unset LATTE_INLET
  if [ "$MI_M" = 1 ]; then export LATTE_FOOT=mi; else unset LATTE_FOOT; fi
}
diag_run() {    # model mode out N iters fit-hours hard-seconds [extra args]; the diag log goes to stdout
  local m=$1 md=$2 o=$3 n=$4 it=$5 h=$6 hard=$7; shift 7
  model_cfg "$m"
  local bf=""; [ "$md" != params ] && [ "$MI_M" = 1 ] && bf="--box-family meas"
  ( model_env "$m"
    exec timeout -s TERM -k 120 "$hard" $PY -m latte_imex.diff.diag run --mode "$md" --init "$INIT_M" --pset "$PSET_M" \
      --out "$o" --N "$n" --device "$DEVICE" --iters "$it" --time-h "$h" $bf "$@" ) 2>&1 | grep -v '^Module '
  return ${PIPESTATUS[0]}
}

T0=$(date +%s)
echo "===== $(date '+%F %T') update21 (diag21, measured inlet) in $ROOT: N $N, device $DEVICE, budget ${BUDGET_H} h, runs: $RUNS ====="
if [ -f MANIFEST.sha256 ]; then
  sha256sum -c --quiet MANIFEST.sha256 || { [ "${FORCE:-0}" = "1" ] || die "files differ from MANIFEST.sha256 (FORCE=1 to go on)"; }
  say "files: MANIFEST.sha256 ok"
fi
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
$PY -c "import warp" 2>/dev/null || $PY -m pip install -q "warp-lang==1.18.0"
$PY -c "import scipy" 2>/dev/null || $PY -m pip install -q scipy
$PY -c "import cv2" 2>/dev/null || $PY -m pip install -q "opencv-python-headless==4.10.0.84"
$PY -c "import imageio, imageio_ffmpeg, PIL, matplotlib" 2>/dev/null || $PY -m pip install -q imageio imageio-ffmpeg pillow matplotlib
$PY scripts/diag21_tools.py check --device "$DEVICE" 2>&1 | grep -v '^Module ' | tee "$OUT/startup_check.txt"
if [ ${PIPESTATUS[0]} -ne 0 ]; then [ "${FORCE:-0}" = "1" ] || die "start-up check failed; send $ROOT/$OUT/startup_check.txt"; fi
say "environment and frozen inputs: ok (pose v2, measured inlet, start closures)"

# ---------------------------------------------------------------------- gradient self-tests (CPU; before any GPU wait)
GC_ALL_OK=1
gc() {          # name [extra args]: diag gradcheck N 32, T 1, eps 1e-7 with LATTE_INLET / LATTE_FOOT unset (mi27 T7)
  local name=$1; shift
  ( unset LATTE_INLET LATTE_FOOT
    exec $PY -m latte_imex.diff.diag gradcheck --N 32 --T 1.0 --eps 1e-7 --device "$GC_DEVICE" "$@" ) 2>&1 \
    | grep -v '^Module ' > "$OUT/selftest_gradcheck_$name.txt"
  local rc=${PIPESTATUS[0]}
  cat "$OUT/selftest_gradcheck_$name.txt" | grep -v '^Warp\|^   \|^ *"cpu"\|Kernel cache\|^  *$'
  local res; res=$(grep -E "^gradcheck (PASS|FAIL)" "$OUT/selftest_gradcheck_$name.txt" | tail -1)
  if [ $rc -ne 0 ]; then
    GC_ALL_OK=0
    say "gradient self-test ($name): FAILED (rc $rc) ${res}"
    [ "${FORCE:-0}" = "1" ] || die "gradient self-test ($name) failed; send $ROOT/$OUT/selftest_gradcheck_$name.txt"
  else
    say "gradient self-test ($name): ${res:-passed}"
  fi
}
if [ ! -f "$OUT/selftest_ok" ]; then
  say "gradient self-tests (CPU, a few minutes each) ..."
  gc record                                                                             # record inlet, box 'err'
  gc mi --inlet replay --box meas --init latte_imex/data/closures/mcd_S1kq_N96.json --keys tau_s,p_dep,kappa_c,B_dep
  [ "$GC_ALL_OK" = 1 ] && touch "$OUT/selftest_ok"     # (with FORCE=1 a failed test is not recorded as passed)
else
  say "gradient self-tests: passed before (skipped)"
fi

# ---------------------------------------------------------------------- a GPU busy with another job: wait for it
if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null && [ "${NOWAIT:-0}" != "1" ]; then
  GI=${DEVICE#cuda:}; [[ "$GI" =~ ^[0-9]+$ ]] || GI=0
  W0=$(date +%s)
  while [ -n "$(nvidia-smi -i "$GI" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; do
    if [ $(( $(date +%s) - W0 )) -gt $($PY -c "print(int(${MAXWAIT_H:-3} * 3600))") ]; then
      echo "WARNING: the GPU is still busy after ${MAXWAIT_H:-3} h; starting anyway"; break
    fi
    echo "$(date '+%T') the GPU is busy ($(nvidia-smi -i "$GI" --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')); waiting (NOWAIT=1 to skip)"
    sleep 120
  done
  T0=$(date +%s)                   # the budget starts when the GPU is ours
fi
SAMPLER=""
if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 >> "$OUT/gpu_usage.csv" 2>/dev/null 9>&- &
  SAMPLER=$!; trap '[ -n "$SAMPLER" ] && kill $SAMPLER 2>/dev/null' EXIT
fi

# ---------------------------------------------------------------------- SMOKE: every run once at N 32, one iteration
# (on the device; T 1 s, --no-heldout, --no-movie; the first MI params run keeps its held-out pours at T 3 s, because
# under the measured inlet a held-out failure is fatal and swan's scoring needs ~3 s).  A run whose SMOKE fails is
# skipped later (FORCE=1 runs it anyway); delete results_diag21/_smoke to repeat the SMOKE.
SMK=$OUT/_smoke
HO_SMOKE=""; for r in $SMOKE_RUNS; do case $r in s1kq/params|v05n/params) HO_SMOKE=$r; break;; esac; done
smoke_one() {
  local r=$1 m=${1%/*} md=${1#*/}; local o=$SMK/$m/$md t1 rc
  [ -f "$o/done.json" ] && return 0
  local ex="--T 1 --no-heldout"; [ "$r" = "$HO_SMOKE" ] && ex="--T 3"
  t1=$(date +%s)
  mkdir -p "$SMK/$m"
  diag_run "$m" "$md" "$o" "$SMOKE_N" 1 0.25 2700 --no-movie $ex >> "$SMK/$m.$md.log"; rc=$?
  if [ -f "$o/done.json" ]; then echo "$r ok ($(( ($(date +%s) - t1) / 60 )) min)" >> "$SMK/status.txt"
  else echo "$r FAILED (rc $rc)" >> "$SMK/status.txt"; echo "$r" >> "$SMK/failed.txt"; fi
}
if [ "${SKIP_SMOKE:-0}" != "1" ] && [ ! -f "$SMK/done" ]; then
  mkdir -p "$SMK"; rm -f "$SMK/status.txt" "$SMK/failed.txt"
  say "SMOKE: $(echo $SMOKE_RUNS | wc -w) run(s) at N $SMOKE_N, one iteration each ($SMOKE_PAR at a time) ..."
  set -- $SMOKE_RUNS
  while [ $# -gt 0 ]; do
    if [ "$SMOKE_PAR" -ge 2 ] && [ $# -ge 2 ]; then
      smoke_one "$1" & p1=$!
      [[ "$DEVICE" == cuda* ]] && sleep 90          # stagger: CUDA module compilation of the first one
      smoke_one "$2" & p2=$!
      wait $p1; wait $p2; shift 2
    else
      smoke_one "$1"; shift
    fi
  done
  while read -r l; do say "SMOKE $l"; done < "$SMK/status.txt"
  if [ -s "$SMK/failed.txt" ]; then
    say "WARNING: SMOKE failed for: $(tr '\n' ' ' < "$SMK/failed.txt")-- these runs are skipped (FORCE=1 runs them); logs $SMK/*.log"
  fi
  touch "$SMK/done"
elif [ -f "$SMK/done" ]; then
  say "SMOKE: done before (skipped)$( [ -s "$SMK/failed.txt" ] && echo "; failed: $(tr '\n' ' ' < "$SMK/failed.txt")")"
fi

# ------------------------------------------------- one N-grid S1kq and one v05n iteration (MI-R) -> S1kq time slots
TIM=$OUT/_timing; TIMV=$OUT/_timing_v05n
case " $RUNS " in *" s1kq/"*) NEED_T=1;; *) NEED_T=0;; esac
timing_ok() { $PY -c "import json,sys; sys.exit(0 if json.load(open('$1/timing.json')).get('ok') else 1)" 2>/dev/null; }
time_model() {  # model dir label: TIMING_K params iterations at N on the device, stopped by diag21_tools timing
  local m=$1 d=$2 lab=$3
  rm -rf "$d"; mkdir -p "$d"
  say "$lab timing: $TIMING_K iteration(s) at N $N on $DEVICE (the S1kq / v05n ratio sets the S1kq time slots) ..."
  model_cfg "$m"
  ( model_env "$m"
    exec timeout -s TERM -k 120 3600 $PY scripts/diag21_tools.py timing --out "$d" --k "$TIMING_K" --max-min 45 -- \
      $PY -m latte_imex.diff.diag run --mode params --pset "$PSET_M" --init "$INIT_M" --out "$d" --N "$N" --device "$DEVICE" \
      --iters $((TIMING_K + 1)) --time-h 2 --no-heldout --no-movie $T_ARGS_PAR ) 2>&1 | grep -v '^Module ' | tee -a "$d/timing_driver.log"
  [ ${PIPESTATUS[0]} -eq 0 ] && say "$lab $(tail -n 1 "$d/timing_driver.log")" \
    || say "WARNING: the $lab timing failed (see $d/timing.log)"
}
if [ "$NEED_T" = 1 ] && [ "${SKIP_TIMING:-0}" != "1" ]; then
  timing_ok "$TIM" || time_model s1kq "$TIM" S1kq
  timing_ok "$TIMV" || time_model v05n "$TIMV" "v05n (MI-R)"
  timing_ok "$TIM" || say "S1kq keeps the v05n slots (no S1kq timing)"
  timing_ok "$TIMV" || say "the S1kq slots are scaled against diag256's 90 s per v05n iteration (no v05n timing)"
fi
$PY scripts/diag21_tools.py plan --out "$OUT" --runs "$RUNS" --timing "$TIM/timing.json" --timing-v "$TIMV/timing.json" \
  --N "$N" --fit-h "$FIT_H" \
  --pfit-h "$PFIT_H" --render-h "$RENDER_H" --heldout-h "$HELDOUT_H" --bridge-h "$BRIDGE_H" 2>&1 | tee -a "$OUT/startup.txt"
echo "${_LATTE_TOKEN:-fg}" > "$OUT/startup_done"
say "start-up done after $(( ($(date +%s) - T0) / 60 )) min; the fits start (budget ${BUDGET_H} h from the start-up)"

# ---------------------------------------------------------------------- the fits
run_once() {    # run fit-hours [iters]; hard stop at fitting + render (+ held-out) + 45 min (a hang must not stall the rest)
  local r=$1 h=$2 m=${1%/*} md=${1#*/} hard ex its=${3:-$ITERS}
  ex=$T_ARGS_BOX; [ "$md" = params ] && ex=$T_ARGS_PAR
  hard=$($PY -c "print(int((float('$h') + $RENDER_H + ($HELDOUT_H if '$md' == 'params' else 0.0) + 0.75) * 3600))")
  mkdir -p "$OUT/$m"
  diag_run "$m" "$md" "$OUT/$m/$md" "$N" "$its" "$h" "$hard" $ex >> "$OUT/$m/$md.log"
}
run_mode() {    # one retry if no done.json; it resumes from state.npz with what is left of the planned slot, or, when the
                # first attempt had finished fitting (it reached the render), goes straight to the render (no more fitting)
  local t1 l1 rc h2 its2 st
  t1=$(date +%s); l1=$(( $(fsize "$OUT/$1.log") + 1 ))
  run_once "$1" "$2"; rc=$?
  if [ ! -f "$OUT/$1/done.json" ]; then
    st="$OUT/$1/state.npz"; its2=""
    if tail -c +$l1 "$OUT/$1.log" 2>/dev/null | grep -q "render: best iteration" && [ -f "$st" ]; then
      its2=$($PY -c "import numpy as np; print(int(np.load('$st')['it']))" 2>/dev/null)
    fi
    if [ -n "$its2" ]; then
      h2=0.25
      echo "$(date '+%T') $1: no done.json (rc $rc); one retry: fitting had ended at iteration $its2, render only"
    else
      h2=$($PY -c "print(round(max(0.25, float('$2') - ($(date +%s) - $t1) / 3600.0), 3))")
      echo "$(date '+%T') $1: no done.json (rc $rc); one retry (resumes from state.npz) with the rest of the slot, ${h2} h"
    fi
    run_once "$1" "$h2" $its2; rc=$?
  fi
  return $rc
}
bridge_run() {  # model out closure env(mi|record) cal [extra]: --iters 0 renders the closure as it is (J, separation)
  local m=$1 o=$2 cl=$3 ev=$4 cal=$5; shift 5
  model_cfg "$m"
  local k
  for k in 1 2; do
    [ -f "$o/done.json" ] && return 0
    ( export LATTE_POSE=v2; unset LATTE_INLET
      if [ "$ev" = mi ]; then export LATTE_FOOT=mi; else unset LATTE_FOOT; fi
      exec timeout -s TERM -k 120 3600 $PY -m latte_imex.diff.diag run --mode params --init "$cl" --pset "$PSET_M" \
        --out "$o" --N "$N" --device "$DEVICE" --iters 0 --time-h 0.25 --no-heldout --cal "$cal" "$@" ) 2>&1 \
      | grep -v '^Module ' >> "$o.log"
  done
  [ -f "$o/done.json" ] || echo "WARNING: bridge $o did not finish, see $o.log"
}
bridges() {     # SPEC27 6.1: (i) tulip + leaf under MI-L, (ii) the held-out pours under R1, with the params best closure
  local m=$1 src=$OUT/$1/params/best_closure.json p
  [ -f "$OUT/$m/params/done.json" ] && [ -f "$src" ] || return 0
  mkdir -p "$OUT/$m/bridge_closures"
  $PY scripts/diag21_tools.py bridge-closure "$src" "$OUT/$m/bridge_closures/mil.json" --inlet 1
  $PY scripts/diag21_tools.py bridge-closure "$src" "$OUT/$m/bridge_closures/r1.json" --inlet 0
  echo "$(date '+%T') $m bridges: (i) tulip + leaf under MI-L, (ii) heart / push_heart / swan under R1 (logs $OUT/$m/bridge_*.log)"
  # (same T as the params run whose tulip / leaf J it is compared with)
  bridge_run "$m" "$OUT/$m/bridge_mil" "$OUT/$m/bridge_closures/mil.json" mi tulip,leaf $T_ARGS_PAR
  for p in heart push_heart swan; do        # one pour per run: three N 256 pours at once may not fit a 24 GB card
    bridge_run "$m" "$OUT/$m/bridge_r1_$p" "$OUT/$m/bridge_closures/r1.json" record "$p" $T_ARGS_PAR
  done
}
archive() {     # the small archive (no videos, no state.npz); refreshed after every run
  tar -czf "$ARCH.tar.gz.part" --exclude='*.mp4' --exclude='state.npz' --exclude='*.tmp' --exclude='.diag21.lock' "$OUT" 2>/dev/null \
    && mv -f "$ARCH.tar.gz.part" "$ARCH.tar.gz"
}
skipped() { [ "${FORCE:-0}" != "1" ] && [ -f "$SMK/failed.txt" ] && grep -qx "$1" "$SMK/failed.txt"; }
for r in $RUNS; do
  m=${r%/*}
  if [ -f "$OUT/$r/done.json" ]; then
    echo "$(date '+%T') $r: done before, skipped"
  elif skipped "$r"; then
    echo "WARNING: $r skipped (its SMOKE failed: $SMK/${m}.${r#*/}.log; FORCE=1 runs it)"
  else
    left=""; seen=0
    for q in $RUNS; do [ "$q" = "$r" ] && seen=1; [ $seen = 1 ] && [ ! -f "$OUT/$q/done.json" ] && ! skipped "$q" && left="$left $q"; done
    H=$($PY scripts/diag21_tools.py slot --plan "$OUT/plan.json" --budget-h "$BUDGET_H" --t0 "$T0" $left)
    echo "$(date '+%T') $r, ${H} h of fitting (log $OUT/$r.log)"
    run_mode "$r" "$H"; rc=$?
    [ -f "$OUT/$r/done.json" ] || echo "WARNING: $r did not finish (rc $rc), see $OUT/$r.log"
    grep -v '^Module ' "$OUT/$r.log" 2>/dev/null | tail -n 4
  fi
  case $r in s1kq/params|v05n/params) bridges "$m";; esac
  $PY scripts/diag21_tools.py summary --out "$OUT" > /dev/null 2>&1
  archive
done
$PY scripts/diag21_tools.py summary --out "$OUT" 2>&1 | grep -v '^Module \|^Warp\|^   '
archive
find "$OUT" -name '*.mp4' -not -path "$SMK/*" -print0 | tar -czf "${ARCH}_videos.tar.gz" --null -T - 2>/dev/null
echo "===== $(date '+%F %T') done in $(( ($(date +%s) - T0) / 60 )) min; send back: $ROOT/$ARCH.tar.gz (and $ROOT/${ARCH}_videos.tar.gz) ====="
