#!/bin/bash
# CMA-ES calibration of the closure on the GPU (NVIDIA Warp, forward runs only) with the structure score
# (latte_imex/diff/score.py) and the robust cup-frame E0 replay; calibration tulip + leaf, frozen-closure test on heart,
# push_heart, swan; then the movies of the best model (runs in the background; log results/cma.log).
#   bash scripts/autodl_cma.sh                  -> prints the log path and returns
#   MODELS="v05 e j k y ej ejk ejky"         (default; model letters as diff/cmafit.py: e entrainment sink, j traction
#                                                towards the landing point's velocity, k kinematic deposition + crater,
#                                                y yield of the floating layer; ':L' = surface-current length circ_L)
#   SEEDS="3 7"  (every model with each CMA seed: the run-to-run spread of the held-out objective is ~0.04)
#   N=48  GENS=16  POP=12  W_IOU=0.25  E0=world  RENDER=1 (render the model with the best mean held-out objective)
# Send back: results/cma_results.tar.gz
cd "$(dirname "$0")/.."
LOG=results/cma.log
mkdir -p results
if [ "${_LATTE_BG:-}" != "1" ]; then
  _LATTE_BG=1 nohup bash "$0" "$@" >> "$LOG" 2>&1 &
  echo "started in the background (pid $!); log: $(pwd)/$LOG"
  echo "watch: tail -f $(pwd)/$LOG"
  exit 0
fi
set -eo pipefail
export PYTHONUNBUFFERED=1
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} \
       OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
export PYTHONWARNINGS=ignore
python -c "import warp" 2>/dev/null || python -m pip install -q warp-lang
python -c "import cma" 2>/dev/null || python -m pip install -q cma
python -c "import cv2, matplotlib.pyplot, imageio" 2>/dev/null || \
  python -m pip install -q "numpy<2" "opencv-python-headless==4.10.0.84" imageio imageio-ffmpeg
MODELS=${MODELS:-"v05 e j k y ej ejk ejky"}
SEEDS=${SEEDS:-"3 7"}
N=${N:-48}; GENS=${GENS:-16}; POP=${POP:-12}; W_IOU=${W_IOU:-0.25}; E0=${E0:-world}
echo "===== $(date '+%F %T') self-test: forward parity with the numpy solver (GPU, N=64, 200 frames, all mechanisms on) ====="
python -m latte_imex.diff.test_parity --N 64 --frames 200 --device cuda:0 \
    --set kappa_e=0.8,vj=1,kappa_tc=0.3,e_F=0.3,mu_s=1,tau_s=0.08,tau_b=0.02,tau_f=0.02
TARS=""
for m in $MODELS; do
  model=${m%%:*}; L=0.4; [ "$m" != "$model" ] && L=${m#*:}
  for seed in $SEEDS; do
    OUT=results/cma_${model}_L${L}_N${N}_s${seed}
    echo "===== $(date '+%F %T') CMA-ES $model (circ_L $L, seed $seed): tulip+leaf, N=$N, $GENS x $POP, e0 $E0 ====="
    python -m latte_imex.diff.cmafit --out "$OUT" --cal tulip,leaf --model "$model" --circ-L "$L" --N "$N" --gens "$GENS" \
        --pop "$POP" --w-iou "$W_IOU" --e0 "$E0" --device cuda:0 --seed "$seed"
    TARS="$TARS $OUT"
  done
done
echo "===== $(date '+%F %T') summary (held-out objective; 1 = the video's own 0.25 s self-consistency) ====="
python - $TARS <<'PY' | tee results/cma_summary.txt
import json, sys
rows, by = [], {}
for d in sys.argv[1:]:
    T = json.load(open(f"{d}/test.json")); B = json.load(open(f"{d}/best.json"))
    w = 0.25
    ho = sum(w * r["viou_n"] + (1 - w) * r["struct_n"] for r in T["test"].values()) / len(T["test"])
    rows.append((ho, d, B["score"], {p: (round(r["viou"], 3), round(r["struct"], 3)) for p, r in T["test"].items()}))
    by.setdefault(d.rsplit("_s", 1)[0], []).append((ho, B["score"], d))
for ho, d, cal, per in sorted(rows, reverse=True):
    print(f"{d:44s} calibration {cal:.3f}  held-out {ho:.3f}  (IoU, structure) {per}")
print("--- mean over seeds (held-out / calibration)")
best = None
for m, v in sorted(by.items(), key=lambda kv: -sum(x[0] for x in kv[1]) / len(kv[1])):
    mh, mc = sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)
    print(f"{m:40s} held-out {mh:.3f} (seeds {' '.join(f'{x[0]:.3f}' for x in v)})  calibration {mc:.3f}")
    best = best or max(v)[2]
print("BEST", best)
PY
if [ "${RENDER:-1}" = "1" ]; then
  BEST=$(grep '^BEST' results/cma_summary.txt | awk '{print $2}')
  echo "===== $(date '+%F %T') render $BEST (N=256) ====="
  python -m latte_imex.rig.fit_pipeline render --out "$BEST" --jobs ${JOBS:-5}
fi
tar -czf results/cma_results.tar.gz $TARS results/cma_summary.txt "$LOG"
ls -lh results/cma_results.tar.gz
echo "===== $(date '+%F %T') DONE ====="
