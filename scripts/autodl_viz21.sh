#!/bin/bash
# viz21 (viz29): the three-view videos of the update21 fits, rendered on AutoDL after the fits.
# For every FINISHED params run (results_diag21/<model>/params/{done.json,best_closure.json}; models s1kq v05n v05n_ctrl)
# it renders TWO closures, start first (viz/jobs21.py plans the jobs):
#   start  the run's START closure as it is (起点参数，未重新拟合: s1kq mcd_S1kq_mi_N96.json, v05n bp256_mi.json,
#          v05n_ctrl bp256_trained.json under the record inlet R1 + pose v2), found from the run's done.json 'init' /
#          run.log / the INIT table of scripts/autodl_diag21.sh / the repository, checked against best_closure.json
#   best   the fitted best_closure.json (拟合后参数)
# + any closures in VIZ_CLOSURES="name=path[,inlet] ..." (inlet: record | law | replay | 0 | 1 | 2; default: the
#   closure's own key 'inlet' / a done.json beside it / record) -> results_diag21/viz_extra/<name>/
# and every pour: tulip, leaf (calibration); heart, push_heart, swan for a measured-inlet closure (held out: MI-L,
# e0 off, report only), and with VIZ_HELD_RECORD=1 also for a record closure (record inlet + its own e0, as the
# control fit and the R1 bridges evaluated them).  Per (closure, pour):
#   1. the closure re-run forward at N 256 with the field saved at 30 fps (viz/forward_fields.py = diag's own forward
#      re-run; the inlet is resolved as the fit resolved it), and
#   2. the views rendered with the Warp ray tracer (viz/render3.py, on the GPU):
#        results_diag21/<model>/params/viz/<start|best>/<pour>_pour.mp4    倒奶视角 (pitcher translucent: milk, stream, Q)
#                                                       <pour>_guest.mp4   客人视角 (the customer's view of the cup)
#                                                       <pour>_top.mp4     俯视二维 (the pattern, landing marker)
#                                                       <pour>_combo.mp4   2 x 2 with the real rectified top view
#                                                       <pour>_sheet.jpg   6 times x the views
#      + <pour>_viz.json, <pour>.log, <pour>.done; the dense fields (for re-rendering) go to
#      results_diag21_vizfields/<model>/<start|best>/<pour>_fields_N256.npz, outside results_diag21 (never archived)
# The fits must not be running (they use up to ~19 GB of the 24 GB card): while results_diag21/.diag21.lock is held,
# this waits (VIZ_WAIT_H, default 36 h); with the fits finished run it with NOWAIT=1.  Re-running resumes: a
# (model, start|best, pour) whose .done names the same closure (sha256), inlet and settings (N, VIZ_FPS, VIZ_T,
# VIZ_SIZE, VIZ_SS, VIZ_CUP and the add-on's cup model, viz/scene.py CUP_MODEL_VERSION) is skipped; a changed setting
# re-renders (VIZ_FORCE=1: everything).  Cached fields are reused only for the same closure, inlet, N, fps and VIZ_T
# (the cup model is not part of them: a new cup model re-renders from the cached fields, no forward run).  Videos of the first viz21 layout
# (viz/<pour>_*, best closure) are moved to viz/best/.  It never fails its caller: every problem is a log line in
# results_diag21/viz.log and the exit status is 0.
#
#   NOWAIT=1 bash scripts/autodl_viz21.sh    -> starts in the background (log: results_diag21/viz.log), returns
#   FG=1 (foreground)  OUT=results_diag21  DEVICE=cuda:0  N=256  VIZ_FPS=30  VIZ_SIZE=1280x720  VIZ_SS=2 (supersampling)
#   VIZ_MODELS="s1kq v05n v05n_ctrl" ("none": only VIZ_CLOSURES)  VIZ_WHICH=both|start|best
#   VIZ_POURS="tulip leaf heart push_heart swan"  VIZ_CLOSURES="name=path[,inlet] ..."  VIZ_PAR=2 (jobs at a time)
#   VIZ_MAX_H=4 (time cap: no new job after it; a running job is stopped at it)
#   VIZ_JOB_MIN=45 (hard stop of one job, forward run + render together)  VIZ_FORCE=1 (ignore the .done markers)
#   VIZ_HELD_RECORD=1 (held-out pours also for record closures: v05n_ctrl, VIZ_CLOSURES bridge_closures/r1.json)
#   VIZ_WAIT_H=36 (longest wait for the fits)  NOWAIT=1 (do not wait for the fits / the GPU)  VIZ_CUP=dark|white
#   VIZ_IN_DIAG=1 (called from inside autodl_diag21.sh, which holds the lock: do not wait for it)
#   LOCAL_SMOKE=1  CPU dress rehearsal: N 32, 1 s of each pour, 320 x 180 at 10 fps, tulip + heart, foreground,
#                  into results_diag21_localsmoke/ (the finished fits are read from results_diag21: small files copied)
#   PKG_ROOT=<tree holding latte_imex/> (default: this script's parent / own directory)  VIZ_DIR=<viz add-on>
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
HERE=$(dirname "$SELF")
ROOT=${PKG_ROOT:-}
if [ -z "$ROOT" ]; then
  for d in "$HERE/.." "$HERE" "$HERE/../.."; do [ -d "$d/latte_imex" ] && { ROOT=$(cd "$d" && pwd); break; }; done
fi
VIZ=${VIZ_DIR:-}
if [ -z "$VIZ" ]; then
  for d in "$HERE" "$ROOT/viz" "$HERE/../viz"; do [ -f "$d/render3.py" ] && { VIZ=$(cd "$d" && pwd); break; }; done
fi
if [ -z "$ROOT" ] || [ -z "$VIZ" ] || [ ! -f "$VIZ/jobs21.py" ]; then
  echo "viz21: cannot find the package tree (latte_imex/) or the viz add-on (viz/render3.py, viz/jobs21.py); nothing done"
  exit 0
fi
if [ "${LOCAL_SMOKE:-0}" = "1" ]; then
  OUT=${OUT:-results_diag21_localsmoke}; DEVICE=${DEVICE:-cpu}; N=${N:-32}; VIZ_T=${VIZ_T:-1.0}
  VIZ_SIZE=${VIZ_SIZE:-320x180}; VIZ_FPS=${VIZ_FPS:-10}; VIZ_SS=${VIZ_SS:-1}; VIZ_PAR=${VIZ_PAR:-1}; FG=${FG:-1}
  VIZ_POURS=${VIZ_POURS:-tulip heart}; NOWAIT=${NOWAIT:-1}; VIZ_MAX_H=${VIZ_MAX_H:-1}
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2}
fi
OUT=${OUT:-results_diag21}
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT" 2>/dev/null || { echo "viz21: cannot create $OUT; nothing done"; exit 0; }
LOG=$OUT/viz.log
FIELDS=${VIZ_FIELDS:-${OUT}_vizfields}  # the dense N 256 fields: outside results_diag21 (never in its archives)
if [ "${FG:-0}" != "1" ] && [ "${_VIZ_BG:-}" != "1" ]; then
  _VIZ_BG=1 nohup bash "$SELF" "$@" >> "$LOG" 2>&1 &
  echo "viz21 started in the background (pid $!); log: $LOG"
  if [ "${NOWAIT:-0}" = "1" ] || [ "${VIZ_IN_DIAG:-0}" = "1" ]; then
    echo "  it renders the start and the fitted closures of the finished fits into"
  else
    echo "  it waits until the update21 fits are finished (NOWAIT=1: do not wait), then renders into"
  fi
  echo "  $OUT/<model>/params/viz/<start|best>/ (watch: tail -f $LOG)"
  exit 0
fi
[ "${FG:-0}" = "1" ] && [ "${_VIZ_BG:-}" != "1" ] && exec > >(tee -a "$LOG") 2>&1
log() { echo "$(date '+%F %T') viz21: $*"; }
HAVE_FLOCK=0; command -v flock >/dev/null && HAVE_FLOCK=1
if [ $HAVE_FLOCK = 1 ]; then
  exec 8>"$OUT/.viz21.lock"
  flock -n 8 || { log "another viz21 run is working on $OUT; nothing done"; exit 0; }
fi
trap 'log "stopped by a signal"; exit 0' TERM INT
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" VIZ_ASSETS="${VIZ_ASSETS:-$VIZ/assets}" LATTE_POSE=v2
# numba's cache=True (latte_imex/kernels.py) would otherwise create latte_imex/__pycache__ inside the package tree
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$FIELDS/.numba_cache}"
PY=${PY:-python}; command -v "$PY" >/dev/null || PY=python3
DEVICE=${DEVICE:-cuda:0}; N=${N:-256}; VIZ_FPS=${VIZ_FPS:-30}; VIZ_SIZE=${VIZ_SIZE:-1280x720}; VIZ_SS=${VIZ_SS:-2}
VIZ_MODELS=${VIZ_MODELS:-s1kq v05n v05n_ctrl}; VIZ_POURS=${VIZ_POURS:-tulip leaf heart push_heart swan}
VIZ_WHICH=${VIZ_WHICH:-both}; VIZ_CLOSURES=${VIZ_CLOSURES:-}
VIZ_PAR=${VIZ_PAR:-2}; VIZ_MAX_H=${VIZ_MAX_H:-4}; VIZ_JOB_MIN=${VIZ_JOB_MIN:-45}; VIZ_WAIT_H=${VIZ_WAIT_H:-36}
VIZ_CUP=${VIZ_CUP:-dark}; VIZ_FORCE=${VIZ_FORCE:-0}; VIZ_HELD_RECORD=${VIZ_HELD_RECORD:-0}
case "$VIZ_WHICH" in start|best|both) ;; *) log "VIZ_WHICH=$VIZ_WHICH is not start / best / both; using both"; VIZ_WHICH=both ;; esac
T_ARGS=""; [ -n "${VIZ_T:-}" ] && T_ARGS="--T $VIZ_T"
# Warp's start-up banner and module-load lines are dropped from the logs; everything else (tracebacks with their code
# lines, warnings) is kept
wpfilter() {
  awk '/^Warp [0-9][0-9.]* initialized:/ {b = 1; next}
       b && /^   / {next}
       {b = 0}
       /^Module [^ ]+ [0-9a-f]+ load on device/ {next}
       {print; fflush()}'
}
log "===== start: tree $ROOT, add-on $VIZ, results $OUT, N $N, $VIZ_SIZE (ss $VIZ_SS) at $VIZ_FPS fps${VIZ_T:+, first $VIZ_T s}, device $DEVICE, closures: $VIZ_WHICH${VIZ_CLOSURES:+ + $VIZ_CLOSURES}$([ "$VIZ_HELD_RECORD" = 1 ] && echo ', held-out pours also for record closures')$([ "$VIZ_FORCE" = 1 ] && echo ', VIZ_FORCE')"

# ------------------------------------------------------------------ wait for the fits (they hold .diag21.lock)
W0=$(date +%s)
waited() { echo $(( ($(date +%s) - W0) )); }
maxwait=$($PY -c "print(int(float('$VIZ_WAIT_H') * 3600))" 2>/dev/null || echo 129600)
if [ "${VIZ_IN_DIAG:-0}" != "1" ] && [ "${NOWAIT:-0}" != "1" ] && [ $HAVE_FLOCK = 1 ]; then
  said=0
  while [ -f "$OUT/.diag21.lock" ] && ! flock -n "$OUT/.diag21.lock" true 2>/dev/null; do
    if [ "$(waited)" -gt "$maxwait" ]; then
      log "the update21 fits are still running after ${VIZ_WAIT_H} h; nothing rendered (run this again later)"; exit 0
    fi
    if [ $(( $(waited) / 1800 )) -ge $said ]; then
      log "the update21 fits are running ($OUT/.diag21.lock is held); waiting for them to finish ..."; said=$(( said + 1 ))
    fi
    sleep 60
  done
  [ "$(waited)" -gt 5 ] && log "the fits are finished (waited $(( $(waited) / 60 )) min)"
fi
if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null && [ "${NOWAIT:-0}" != "1" ]; then
  GI=${DEVICE#cuda:}; [[ "$GI" =~ ^[0-9]+$ ]] || GI=0
  while true; do
    free=$(nvidia-smi -i "$GI" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
    [ -z "$free" ] || [ "$free" -ge "${VIZ_MIN_FREE_MB:-8000}" ] && break
    if [ "$(waited)" -gt "$maxwait" ]; then log "the GPU stayed busy (${free} MB free); nothing rendered"; exit 0; fi
    log "only ${free} MB free on the GPU (another job); waiting"; sleep 120
  done
fi

# ------------------------------------------------------------------ environment
[ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1 || true
$PY -c "import warp" 2>/dev/null || $PY -m pip install -q "warp-lang==1.18.0" >/dev/null 2>&1
$PY -c "import scipy" 2>/dev/null || $PY -m pip install -q scipy >/dev/null 2>&1
$PY -c "import imageio, imageio_ffmpeg, PIL" 2>/dev/null || $PY -m pip install -q imageio imageio-ffmpeg pillow >/dev/null 2>&1
if ! $PY -c "import warp, numpy, scipy, imageio, imageio_ffmpeg, PIL" 2>/dev/null; then
  log "FAILED: python packages missing (warp-lang, scipy, imageio, imageio-ffmpeg, pillow); nothing rendered"; exit 0
fi
FONT=""; for f in "$ROOT/fonts/wqy-zenhei.ttc" "$VIZ/assets/wqy-zenhei.ttc"; do [ -f "$f" ] && { FONT=$f; break; }; done
[ -z "$FONT" ] && log "WARNING: fonts/wqy-zenhei.ttc not found: the captions fall back to English"
[ -f "$VIZ_ASSETS/pitcher_kona700.stl" ] || [ -f "$ROOT/latte_imex/pitcher_geom/pitcher_kona700.stl" ] || {
  log "FAILED: pitcher_kona700.stl is missing (viz/assets/); nothing rendered"; exit 0; }
# the device must exist (an AutoDL instance started without a GPU, 无卡模式, has no cuda:0): the forward re-runs need
# it, so nothing is rendered without it
if [ "$DEVICE" != "cpu" ]; then
  dchk=$(timeout -k 30 600 $PY -c "import warp as wp; wp.config.quiet = True; wp.init(); d = wp.get_device('$DEVICE'); print('ok', d, getattr(d, 'name', ''))" 2>&1 | tail -n 1)
  if [[ "$dchk" != ok* ]]; then
    log "FAILED: device $DEVICE is not available (${dchk:-no answer}); AutoDL 无卡模式? Restart the instance with its GPU and run this again (DEVICE=cpu renders on the CPU, very slowly); nothing rendered"
    exit 0
  fi
  log "device $DEVICE: ${dchk#ok }"
fi
# the ray tracer on the render device: a sphere / plane scene against the CPU (a numerical mismatch falls back to a CPU
# render at 640 x 360 without supersampling; the forward re-runs stay on $DEVICE)
RDEV=$DEVICE; RSIZE=$VIZ_SIZE; RSS=$VIZ_SS; KEEP=()
if [[ "$DEVICE" == cuda* ]]; then
  chk=$(timeout -k 30 900 $PY "$VIZ/test_viz.py" --device-check "$DEVICE" 2>&1 | wpfilter | tail -n 1)
  if [[ "$chk" == ok* ]]; then log "ray tracer on $DEVICE: $chk"
  else
    log "WARNING: the ray tracer check on $DEVICE failed (${chk:-no answer}); rendering on the CPU at 640x360 instead (forward runs on $DEVICE; finished GPU renders are kept)"
    RDEV=cpu; RSIZE=640x360; RSS=1; KEEP=(--keep-render)
  fi
fi

# LOCAL_SMOKE renders into its own OUT: it reads the finished fits of results_diag21 through copies of their small files
# (done.json, best_closure.json, best.json, run.log, eval_cal.json; nothing is written into results_diag21)
if [ "${LOCAL_SMOKE:-0}" = "1" ]; then
  SRC_RUNS=${VIZ_SMOKE_RUNS:-$ROOT/results_diag21}
  for m in $VIZ_MODELS; do
    s=$SRC_RUNS/$m/params; d=$OUT/$m/params
    [ "$m" = none ] || [ "$(cd "$s" 2>/dev/null && pwd)" = "$(cd "$d" 2>/dev/null && pwd)" ] && continue
    if [ -f "$s/done.json" ] && [ -f "$s/best_closure.json" ] && [ ! -f "$d/done.json" ]; then
      mkdir -p "$d"
      for f in done.json best_closure.json best.json run.log eval_cal.json; do [ -f "$s/$f" ] && cp -p "$s/$f" "$d/"; done
      log "LOCAL_SMOKE: the finished fit $m/params read from $s (small files copied into $d)"
    fi
  done
fi

# ------------------------------------------------------------------ jobs (viz/jobs21.py: closures, pours, resume)
T0=$(date +%s)
CAP=$($PY -c "print(int(float('$VIZ_MAX_H') * 3600))" 2>/dev/null || echo 14400)
JOBCAP=$(( VIZ_JOB_MIN * 60 ))
JARGS=(--out "$OUT" --root "$ROOT" --fields "$FIELDS" --models "$VIZ_MODELS")
[ -f "$ROOT/scripts/autodl_diag21.sh" ] && JARGS+=(--launcher "$ROOT/scripts/autodl_diag21.sh")
# the first viz21 layout (viz/<pour>_*, fitted closure only) -> viz/best/ (files of the same closure only)
$PY "$VIZ/jobs21.py" migrate "${JARGS[@]}" 2>&1 | sed -n 's/^I\t//p' | while IFS= read -r l; do log "$l"; done
job() {         # model which pour closure inlet vizdir fields sha desc: forward fields + render, logged into
                # <vizdir>/<pour>.log; one line into viz.log
  local m=$1 w=$2 p=$3 cl=$4 inl=$5 vd=$6 ff=$7 sha=$8 desc=$9 t1 left lim rc why
  t1=$(date +%s)
  mkdir -p "$vd" "$(dirname "$ff")"
  left=$(( CAP - (t1 - T0) )); lim=$JOBCAP; [ $left -lt $lim ] && lim=$left
  [ $lim -le 60 ] && { log "$m $w $p: no time left (VIZ_MAX_H ${VIZ_MAX_H} h / VIZ_JOB_MIN ${VIZ_JOB_MIN} min); not started"; return 0; }
  rm -f "$vd/$p.done"            # (stale: other closure / settings, or VIZ_FORCE); written again only when all is done
  if [ -f "$ff" ]; then
    why=$($PY "$VIZ/jobs21.py" fields-ok "$ff" "$sha" "$inl" --N "$N" --fps "$VIZ_FPS" --T "${VIZ_T:--}" 2>&1) \
      || { rm -f "$ff"; echo "cached fields removed (${why:-check failed}); recomputing" >> "$vd/$p.log"; }
  fi
  if [ ! -f "$ff" ]; then
    echo "$(date '+%F %T') forward: $cl (inlet $inl), pour $p, N $N" >> "$vd/$p.log"
    timeout -s TERM -k 60 "$lim" $PY "$VIZ/forward_fields.py" --closure "$cl" --inlet "$inl" --pour "$p" --N "$N" \
      --fps "$VIZ_FPS" --out "$ff" --pkg "$ROOT" --device "$DEVICE" $T_ARGS 2>&1 | wpfilter >> "$vd/$p.log"
    rc=${PIPESTATUS[0]}
    if [ ! -f "$ff" ]; then log "$m $w $p: FAILED in the forward run (rc $rc; see $vd/$p.log)"; return 0; fi
  fi
  # VIZ_JOB_MIN bounds the whole job: the render gets what the forward run left of it (and of VIZ_MAX_H)
  left=$(( CAP - ($(date +%s) - T0) )); lim=$(( JOBCAP - ($(date +%s) - t1) )); [ $left -lt $lim ] && lim=$left
  [ $lim -le 30 ] && { log "$m $w $p: fields done, no time left to render (VIZ_JOB_MIN ${VIZ_JOB_MIN} min / VIZ_MAX_H ${VIZ_MAX_H} h)"; return 0; }
  timeout -s TERM -k 60 "$lim" $PY "$VIZ/render3.py" --pour "$p" --fields "$ff" --out "$vd" --fps "$VIZ_FPS" \
    --size "$RSIZE" --ss "$RSS" --device "$RDEV" --model "$m" --model-desc "$desc" --which "$w" --root "$ROOT" \
    --cup "$VIZ_CUP" ${FONT:+--font "$FONT"} 2>&1 | wpfilter >> "$vd/$p.log"
  rc=${PIPESTATUS[0]}
  local v miss=""
  for v in pour guest top combo; do [ -s "$vd/${p}_$v.mp4" ] || miss="$miss $v"; done
  if [ -z "$miss" ] && [ $rc -eq 0 ]; then
    $PY "$VIZ/jobs21.py" mark "$vd" "$p" "model=$m" "which=$w" "closure=$cl" "closure_sha256=$sha" "inlet=$inl" \
      "N=$N" "fps=$VIZ_FPS" "T=${VIZ_T:-}" "size=$RSIZE" "ss=$RSS" "cup=$VIZ_CUP" "device=$RDEV" "t1=$t1"
    log "$m $w $p: ok in $(( ($(date +%s) - t1) / 60 )) min -> $vd/${p}_{pour,guest,top,combo}.mp4, ${p}_sheet.jpg"
  else
    log "$m $w $p: FAILED in the render (rc $rc; missing:${miss:- none}; see $vd/$p.log)"
  fi
  return 0
}
JOBS=()
PARGS=(--N "$N" --fps "$VIZ_FPS" --size "$RSIZE" --ss "$RSS" --cup "$VIZ_CUP" "${KEEP[@]}")
[ -n "${VIZ_T:-}" ] && PARGS+=(--T "$VIZ_T")
[ "$VIZ_FORCE" = "1" ] && PARGS+=(--force)
[ "$VIZ_HELD_RECORD" = "1" ] && PARGS+=(--held-record)
PLAN=$($PY "$VIZ/jobs21.py" plan "${JARGS[@]}" "${PARGS[@]}" --which "$VIZ_WHICH" --pours "$VIZ_POURS" \
       --extras "$VIZ_CLOSURES" 2>&1) || log "WARNING: the job planner failed: $(echo "$PLAN" | tail -n 3 | tr '\n' ' ')"
while IFS= read -r l; do
  case $l in
    I$'\t'*) log "${l#I$'\t'}" ;;
    J$'\t'*) JOBS+=("${l#J$'\t'}") ;;
  esac
done <<< "$PLAN"
desc_job() { local a b c r; IFS=$'\t' read -r a b c r <<< "$1"; echo "$a:$b:$c"; }
JL=""; for jp in "${JOBS[@]}"; do JL="$JL $(desc_job "$jp")"; done
log "${#JOBS[@]} job(s):${JL:- none} ($VIZ_PAR at a time, cap ${VIZ_MAX_H} h)"
k=0; PIDS=()
running() { local n=0 q; for q in "${PIDS[@]}"; do kill -0 "$q" 2>/dev/null && n=$(( n + 1 )); done; echo $n; }
for jp in "${JOBS[@]}"; do
  IFS=$'\t' read -r jm jw jpour jcl jinl jvd jff jsha jdesc <<< "$jp"
  if [ $(( $(date +%s) - T0 )) -ge "$CAP" ]; then log "time cap VIZ_MAX_H=${VIZ_MAX_H} h reached; not started: $jm $jw $jpour ..."; break; fi
  while [ "$(running)" -ge "$VIZ_PAR" ]; do sleep 5; done
  log "$jm $jw $jpour: forward N $N + render (closure $jcl; log $jvd/$jpour.log)"
  job "$jm" "$jw" "$jpour" "$jcl" "$jinl" "$jvd" "$jff" "$jsha" "$jdesc" &
  PIDS+=($!)
  k=$(( k + 1 ))
  [ $k -eq 1 ] && [ "$VIZ_PAR" -ge 2 ] && sleep 90      # stagger: the first one compiles the Warp kernels
done
# wait for the jobs only (a bare `wait` would also wait for the log's tee process substitution and never return)
for q in "${PIDS[@]}"; do wait "$q" 2>/dev/null; done

# ------------------------------------------------------------------ summary + archive (videos, sheets, logs; no npz)
$PY "$VIZ/jobs21.py" summary "$OUT" 2>/dev/null
ARCHV="$(dirname "$OUT")/$(basename "$OUT")_viz.tar.gz"
( cd "$(dirname "$OUT")" && find "$(basename "$OUT")" \( -path '*/params/viz/*' -o -path '*/viz_extra/*' \) \
    \( -name '*.mp4' -o -name '*_sheet.jpg' -o -name '*_viz.json' -o -name '*.log' -o -name '*.done' \) \
    -not -name '*.part.mp4' -print0 \
    | tar -czf "$ARCHV.part" --null -T - "$(basename "$OUT")/viz.log" "$(basename "$OUT")/viz_summary.md" 2>/dev/null ) \
  && mv -f "$ARCHV.part" "$ARCHV" && log "archive: $ARCHV ($(du -h "$ARCHV" | cut -f1))"
log "===== done in $(( ($(date +%s) - T0) / 60 )) min; videos in $OUT/<model>/params/viz/<start|best>/, overview $OUT/viz_summary.md"
exit 0
