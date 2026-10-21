#!/bin/bash
# update19: the physics-derived model (mcd22 Stage 1 / Stage 2 closures: no jet momentum in the deposit, foam surfacing
# tau_s ahead of the landing point, physical foam viscosity, sharper milk transport; Stage 2 + the plunging column's
# surface sink) and the untrained new baseline, run at N (default 256) on the five pours, next to the backpropagation
# result of update18 at the same N (closures/bp256_trained.json).  No training: every parameter is fixed.
#   bash scripts/autodl_phys256.sh          -> starts in the background, prints the log path and returns
#   N=256  DEVICE=cuda:0  FG=1 (foreground)  SMOKE=1 (CPU, N 48, first second, tulip only)  FORCE=1
# Outputs results_phys/<run>/{eval.json, <p>_compare.mp4, <p>_simN.mp4, <p>_c128.npz}, results_phys/summary.txt,
# then results_phys256.tar.gz (send this back).  Re-running resumes (finished pours are skipped).
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.." || exit 1
ROOT=$(pwd)
if [ "${SMOKE:-0}" = "1" ]; then OUT=${OUT:-results_phys_smoke}; DEVICE=${DEVICE:-cpu}; N=${N:-48}; EXTRA="--T 1"; POURS=${POURS:-tulip}; FG=1; fi
OUT=${OUT:-results_phys}; LOG=$OUT/phys.log; LOCK=$OUT/.phys.lock
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
DEVICE=${DEVICE:-cuda:0}; N=${N:-256}; POURS=${POURS:-tulip,leaf,heart,push_heart,swan}; EXTRA=${EXTRA:-}
die() { echo "ABORT: $*"; exit 1; }
echo "===== $(date '+%F %T') update19 in $ROOT: N $N, device $DEVICE, pours $POURS ====="
if [ -f MANIFEST.sha256 ]; then
  sha256sum -c --quiet MANIFEST.sha256 || { [ "${FORCE:-0}" = "1" ] || die "files differ from MANIFEST.sha256 (FORCE=1 to go on)"; }
fi
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
$PY -c "import warp" 2>/dev/null || $PY -m pip install -q "warp-lang==1.18.0"
$PY -c "import scipy" 2>/dev/null || $PY -m pip install -q scipy
$PY -c "import cv2" 2>/dev/null || $PY -m pip install -q "opencv-python-headless==4.10.0.84"
$PY -c "import imageio, imageio_ffmpeg, PIL" 2>/dev/null || $PY -m pip install -q imageio imageio-ffmpeg pillow
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
SAMPLER=""
if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 >> "$OUT/gpu_usage.csv" 2>/dev/null 9>&- &
  SAMPLER=$!; trap '[ -n "$SAMPLER" ] && kill $SAMPLER 2>/dev/null' EXIT
fi
C=latte_imex/data/closures
$PY scripts/phys256.py --out "$OUT" --N "$N" --device "$DEVICE" --pours "$POURS" $EXTRA \
    --ref closures/bp256_trained.json:"反传训练 N=$N（update18）" \
    --run $C/mcd_S1_N96.json:"物理版 第一阶段 N=$N" \
    --run $C/mcd_S2_N96.json:"物理版 第二阶段（+奶流吸走表面）N=$N" \
    --run $C/v05n_AB_cmaes_N48.json:"新基线参数 N=$N（未训练）" 2>&1 | grep -v '^Module '
rc=${PIPESTATUS[0]}
[ $rc -eq 0 ] || die "phys256.py failed (rc $rc); send $ROOT/$LOG"
tar -czf "results_phys256.tar.gz" "$OUT" 2>/dev/null
echo "===== $(date '+%F %T') done; send back: $ROOT/results_phys256.tar.gz ====="
