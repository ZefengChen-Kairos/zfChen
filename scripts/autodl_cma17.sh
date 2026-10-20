#!/bin/bash
# update17: CMA-ES recalibration of the closure on the gap-preserving video targets (raw_data/rig/video_obs, fuse_occ
# frames + seg_topo whiteness, V2_SURFACE_MODEL.md 10.5) on the GPU (NVIDIA Warp, forward runs only) -- the GPU
# counterpart of the local fit17 batch at N = 48, and its N = 96 version.  Calibration tulip + leaf, objective
# 0.75 structure + 0.25 IoU (each / the video's own 0.25 s lag reference; score.set_grid(N) keeps the score constants
# at a fixed physical size on every grid), E0 world, frozen-closure test on heart, push_heart, swan.  Model letters as
# diff/cmafit.py: v05 none, e entrainment sink (kappa_e), j traction towards the landing point's velocity V_j (vj = 1,
# kappa_tc), k kinematic deposition (mu_s = 1, tau_s started at TAU_S) + landing crater (e_F), y whole-layer yield
# (tau_b, tau_f).  Every run starts from INIT (v05g_AB_cmaes_N48.json, also at N = 96: there is no N = 96 closure).
#   bash scripts/autodl_cma17.sh            -> starts in the background, prints the log path and returns
#   MODELS="v05 ej ejk k y"  SEEDS="11 12"  NS="48 96"         (defaults; a subset e.g. NS=96 MODELS="v05 ej" SEEDS=11)
#   GENS=16  POP=12  SIGMA=0.12  W_IOU=0.25  E0=world  TAU_S=0.08  DEVICE=cuda:0  INIT=<closure json>  OUT=runs
#   FG=1     run in the foreground          FORCE=1  go on even if a self-test or the manifest check fails
#   SMOKE=1  smoke of the whole path (foreground; N 48, model ejk, seed 11, 1 generation x 2, OUT runs_smoke,
#            results_smoke.tar.gz); DEVICE cpu unless given (parity 30 frames on the CPU, the full 200 on a GPU):
#            SMOKE=1 DEVICE=cuda:0 runs both self-tests of the real run plus one short job on the GPU
# Self-tests before the jobs (abort unless FORCE=1): 1. diff/test_parity, Warp vs numpy solver (heart, N = 64, first
# PAR_FRAMES frames, e j k y on); 2. scripts/gpu_cpu_check.py, one whole-pour evaluation of the objective's leaf term on
# DEVICE vs the CPU (the fit's own path, deposition and yield phases included).
# Jobs run one at a time on DEVICE (one process each, --workers 0, as update15/16).  Order: per N, v05 for every seed,
# then the other models seed by seed.  OUT/<N>/<model>_s<seed>/: log.txt, gens.jsonl, best.json, best_closure.json,
# test.json; a job whose test.json exists is skipped and an unfinished one restarts from scratch, so after an
# interruption the same command resumes (a lock on OUT refuses a second start while a run is still working there).
# At the end: the table OUT/summary.txt (+ .json) and results_update17.tar.gz (send this back).
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.." || exit 1
ROOT=$(pwd)
if [ "${SMOKE:-0}" = "1" ]; then
  OUT=${OUT:-runs_smoke}; DEVICE=${DEVICE:-cpu}; NS=${NS:-48}; MODELS=${MODELS:-ejk}; SEEDS=${SEEDS:-11}
  GENS=${GENS:-1}; POP=${POP:-2}; FG=1
  [ "$DEVICE" = cpu ] && PAR_FRAMES=${PAR_FRAMES:-30}          # the numpy reference with yield on is slow on a CPU
fi
OUT=${OUT:-runs}
LOG=$OUT/update17.log
LOCK=$OUT/.update17.lock
mkdir -p "$OUT"
HAVE_FLOCK=0; command -v flock >/dev/null && HAVE_FLOCK=1
if [ "${FG:-0}" != "1" ] && [ "${_LATTE_BG:-}" != "1" ]; then
  if [ $HAVE_FLOCK = 1 ]; then
    exec 9>"$LOCK"
    flock -n 9 || { echo "ABORT: an update17 run is still working in $ROOT/$OUT (see $ROOT/$LOG); not started"; exit 1; }
    exec 9>&-
  fi
  _LATTE_BG=1 nohup bash "$SELF" "$@" >> "$LOG" 2>&1 &
  echo "started in the background (pid $!); log: $ROOT/$LOG"
  echo "watch: tail -f $ROOT/$LOG"
  exit 0
fi
[ "${FG:-0}" = "1" ] && exec > >(tee -a "$LOG") 2>&1
if [ $HAVE_FLOCK = 1 ]; then                       # one run per OUT: a second one would delete the unfinished job dirs
  exec 9>"$LOCK"
  flock -n 9 || { echo "ABORT: an update17 run is still working in $ROOT/$OUT; this one stops"; exit 1; }
fi
set -o pipefail
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"          # this package's latte_imex, never another checkout
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} \
       OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
PY=${PY:-python}; command -v "$PY" >/dev/null || PY=python3
MODELS=${MODELS:-"v05 ej ejk k y"}
SEEDS=${SEEDS:-"11 12"}
NS=${NS:-${N:-"48 96"}}
GENS=${GENS:-16}; POP=${POP:-12}; SIGMA=${SIGMA:-0.12}; W_IOU=${W_IOU:-0.25}; E0=${E0:-world}; TAU_S=${TAU_S:-0.08}
DEVICE=${DEVICE:-cuda:0}; INIT=${INIT:-latte_imex/data/closures/v05g_AB_cmaes_N48.json}
die() { echo "ABORT: $*"; exit 1; }
echo "===== $(date '+%F %T') update17 in $ROOT: MODELS '$MODELS' SEEDS '$SEEDS' NS '$NS' $GENS x $POP sigma $SIGMA" \
     "w_iou $W_IOU e0 $E0 tau_s(k) $TAU_S device $DEVICE init $INIT ====="

# ---- package integrity and environment
if [ -f MANIFEST.sha256 ]; then
  sha256sum -c --quiet MANIFEST.sha256 || { [ "${FORCE:-0}" = "1" ] || die "files differ from MANIFEST.sha256 (FORCE=1 to go on)"; }
fi
[ -f /etc/network_turbo ] && source /etc/network_turbo || true          # AutoDL academic proxy (pip)
# the versions of the local fit17 batch (warp 1.18.0 also ran update13 on the AutoDL 4090: toolkit 13.4, driver 13.2)
$PY -c "import warp" 2>/dev/null || $PY -m pip install -q "warp-lang==1.18.0"
$PY -c "import cma" 2>/dev/null || $PY -m pip install -q "cma==4.5.0"
$PY -c "import scipy" 2>/dev/null || $PY -m pip install -q scipy
$PY -c "import matplotlib" 2>/dev/null || $PY -m pip install -q matplotlib
$PY -c "import cv2" 2>/dev/null || $PY -m pip install -q "opencv-python-headless==4.10.0.84"
# numba: the numpy reference solver of the parity self-test (latte_imex/kernels.py) runs its stencil loops as plain
# Python without it (hours for 200 frames); a numba that fits the installed numpy, numpy itself left as it is
$PY -c "import numba" 2>/dev/null || $PY -m pip install -q numba "numpy==$($PY -c 'import numpy; print(numpy.__version__)')" \
  || echo "WARNING: numba could not be installed next to this numpy"
if ! $PY -c "import numba" 2>/dev/null; then
  echo "WARNING: no numba: parity self-test shortened to ${PAR_FRAMES:-30} frames (the fits themselves do not use it)"
  PAR_FRAMES=${PAR_FRAMES:-30}
fi
PAR_FRAMES=${PAR_FRAMES:-200}
{ date '+%F %T'; uname -a; command -v nvidia-smi >/dev/null && nvidia-smi; $PY - "$DEVICE" <<'PY'
import os, sys
import numpy, scipy, cv2, cma
import warp as wp
wp.init()
import latte_imex
try:
    import numba
    nb = numba.__version__
except Exception:
    nb = None
print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__, "cv2", cv2.__version__,
      "cma", cma.__version__, "warp", wp.__version__, "numba", nb)
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
rc=$?; cat "$OUT/env.txt" | grep -E '^(python|latte_imex|CUDA devices)|NVIDIA-SMI|RTX|MiB'
[ $rc -eq 0 ] || die "environment check failed (see $OUT/env.txt)"

# ---- self-test 1: forward parity of the Warp solver with the numpy solver, every mechanism of these models on
echo "===== $(date '+%F %T') self-test 1: Warp vs numpy solver (heart, N=64, $PAR_FRAMES frames, $DEVICE, e j k y on) ====="
$PY -m latte_imex.diff.test_parity --N 64 --frames "$PAR_FRAMES" --device "$DEVICE" \
    --set kappa_e=0.8,vj=1,kappa_tc=0.3,e_F=0.3,mu_s=1,tau_s=0.08,tau_b=0.02,tau_f=0.02 2>&1 \
    | grep -v '^Module ' | tee "$OUT/selftest_parity.txt"
if ! grep -q '^PASS' "$OUT/selftest_parity.txt"; then
  [ "${FORCE:-0}" = "1" ] && echo "WARNING: parity self-test did not pass; FORCE=1, going on" \
    || die "parity self-test did not pass ($OUT/selftest_parity.txt); send that file back (FORCE=1 to run anyway)"
fi
# ---- self-test 2: the fit's own path (whole leaf pour, N 48, e0 world, scoring) on DEVICE vs the CPU, e j k y on
echo "===== $(date '+%F %T') self-test 2: one objective evaluation on $DEVICE vs the CPU (leaf, N=48, whole pour, e j k y on) ====="
$PY scripts/gpu_cpu_check.py "$DEVICE" "$INIT" 2>&1 | grep -v '^Module ' | tee "$OUT/selftest_gpu_cpu.txt"
if ! grep -q '^PASS' "$OUT/selftest_gpu_cpu.txt"; then
  [ "${FORCE:-0}" = "1" ] && echo "WARNING: self-test 2 did not pass; FORCE=1, going on" \
    || die "self-test 2 did not pass ($OUT/selftest_gpu_cpu.txt); send that file back (FORCE=1 to run anyway)"
fi

# ---- GPU load sampler (one line a minute; tells whether two jobs at a time would fit next time)
SAMPLER=""
if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 \
    >> "$OUT/gpu_usage.csv" 2>/dev/null 9>&- &
  SAMPLER=$!
  trap '[ -n "$SAMPLER" ] && kill $SAMPLER 2>/dev/null' EXIT
fi

# ---- the jobs
JOBS=""
for N in $NS; do
  case " $MODELS " in *" v05 "*) for s in $SEEDS; do JOBS="$JOBS $N:v05:$s"; done;; esac
  for s in $SEEDS; do for m in $MODELS; do [ "$m" = v05 ] || JOBS="$JOBS $N:$m:$s"; done; done
done
NJ=$(echo $JOBS | wc -w); i=0
for job in $JOBS; do
  i=$((i + 1)); IFS=: read -r N m s <<< "$job"
  d=$OUT/$N/${m}_s$s
  if [ -f "$d/test.json" ]; then echo "[$i/$NJ] $d done (test.json present), skipped"; continue; fi
  rm -rf "$d"; mkdir -p "$d"
  SET=""; case $m in *k*) SET="--set tau_s=$TAU_S";; esac
  echo "===== $(date '+%F %T') [$i/$NJ] CMA-ES $m seed $s N $N -> $d ====="
  t0=$(date +%s)
  $PY -m latte_imex.diff.cmafit --out "$d" --cal tulip,leaf --model "$m" --e0 "$E0" --N "$N" --gens "$GENS" --pop "$POP" \
      --sigma "$SIGMA" --w-iou "$W_IOU" --init "$INIT" --device "$DEVICE" --workers 0 --seed "$s" $SET 2>&1 \
      | tee "$d/log.txt" | grep --line-buffered -E '^(CMA-ES|gen |frozen|  test|  held|Traceback)|Error'
  rc=${PIPESTATUS[0]}
  echo "[$i/$NJ] $m seed $s N $N: exit $rc after $(( ($(date +%s) - t0) / 60 )) min"
  if [ "$rc" -ne 0 ] || [ ! -f "$d/test.json" ]; then echo "FAILED: last lines of $d/log.txt"; tail -n 15 "$d/log.txt"; fi
done

# ---- summary: one row per run
echo "===== $(date '+%F %T') summary (objectives: 1 = as close as the video is to itself 0.25 s later) ====="
$PY - "$OUT" "$W_IOU" <<'PY' | tee "$OUT/summary.txt"
import glob, json, os, sys
out, w = sys.argv[1], float(sys.argv[2])
CAL, TEST = ("tulip", "leaf"), ("heart", "push_heart", "swan")
obj = lambda rs: sum(w * r.get("viou_n", 0) + (1 - w) * r.get("struct_n", 0) for r in rs) / max(len(rs), 1)
vs = lambda r: f"{r['viou']:.3f}/{r['struct']:.3f}" if r else "-"
rows = []
for d in sorted(glob.glob(os.path.join(out, "*", "*_s*"))):
    N = int(os.path.basename(os.path.dirname(d))); m, s = os.path.basename(d).rsplit("_s", 1)
    B = json.load(open(f"{d}/best.json")) if os.path.exists(f"{d}/best.json") else None
    T = json.load(open(f"{d}/test.json")) if os.path.exists(f"{d}/test.json") else None
    ng = sum(1 for _ in open(f"{d}/gens.jsonl")) if os.path.exists(f"{d}/gens.jsonl") else 0
    rows.append(dict(model=m, seed=int(s), N=N, gens=ng, done=T is not None, cal=B["score"] if B else None,
                     best_gen=B["gen"] if B else None, cal_per=B["per"] if B else {}, test=T["test"] if T else {},
                     test_obj=obj(list(T["test"].values())) if T else None, params=T["params"] if T else None))
rows.sort(key=lambda r: (r["N"], r["model"] != "v05", r["model"], r["seed"]))
print("cal = calibration objective (tulip+leaf); test = held-out objective (heart, push_heart, swan); per pour: video IoU/structure")
print(f"{'model':6s} {'seed':>4s} {'N':>3s} {'gens':>4s} {'cal':>6s} {'tulip':>11s} {'leaf':>11s} {'test':>6s} "
      + " ".join(f"{p:>11s}" for p in TEST))
for r in rows:
    f = lambda x: f"{x:6.3f}" if x is not None else "     -"
    print(f"{r['model']:6s} {r['seed']:4d} {r['N']:3d} {r['gens']:4d} {f(r['cal'])} "
          + " ".join(f"{vs(r['cal_per'].get(p)):>11s}" for p in CAL) + f" {f(r['test_obj'])} "
          + " ".join(f"{vs(r['test'].get(p)):>11s}" for p in TEST) + ("" if r["done"] else "  (unfinished)"))
print("--- per model and N: best-seed calibration, seed spread |s1 - s2|; calibration-only comparison with v05 at the same N")
by = {}
for r in rows:
    if r["cal"] is not None and r["done"]:
        by.setdefault((r["N"], r["model"]), []).append(r)
for (N, m), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] != "v05", kv[0][1])):
    best = max(rs, key=lambda r: r["cal"]); spread = max(r["cal"] for r in rs) - min(r["cal"] for r in rs)
    line = (f"N {N:3d} {m:6s} best seed {best['seed']}: cal {best['cal']:.3f} test {best['test_obj']:.3f}  "
            f"seed spread cal {spread:.3f} ({len(rs)} seeds)")
    v = by.get((N, "v05"))
    if m != "v05" and v and len(v) > 1 and len(rs) > 1:
        vb = max(r["cal"] for r in v); vsp = vb - min(r["cal"] for r in v)
        line += f"  | cal - v05 {best['cal'] - vb:+.3f} vs max spread {max(spread, vsp):.3f}: " + \
                ("above" if best["cal"] - vb > max(spread, vsp) else "not above")
    print(line)
json.dump(rows, open(os.path.join(out, "summary.json"), "w"), indent=1)
PY
TAR=results_update17.tar.gz; [ "${SMOKE:-0}" = "1" ] && TAR=results_smoke.tar.gz
tar -czf "$TAR" "$OUT"
ls -lh "$TAR"
echo "===== $(date '+%F %T') DONE: send back $ROOT/$TAR ====="
