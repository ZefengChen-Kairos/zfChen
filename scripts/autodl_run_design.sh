#!/bin/bash
# Closure calibration -> action re-design on the forward pitcher -> robustness check -> final renders, on one GPU.
#   bash scripts/autodl_run_design.sh                       (resumable: re-run the same command after an interruption)
# Steps (STEPS=selftest,calib,predict,design,robust,final by default; any subset, in this order):
#   selftest GPU vs CPU on a tiny design / calibration generation, solver reset, resume, pitcher fit (~2-4 min)
#   calib    closure calibration on tulip + leaf (v05_calibrate setup, CMA-ES, 3 inits in lockstep, batched)  -> $OUT/calib
#   predict  frozen closure on all six clips (calibration fit vs test prediction), final frames              -> $OUT/calib/predict
#   design   per-pattern action re-design with the forward pitcher in the loop (CMA-ES, screen N_LO, confirm N_HI)
#            with the closure from CLOSURE (calib = $OUT/calib/best.json if it exists, s1 = the S1 programs' own)  -> $OUT/design
#   robust   prereg hand-error grid (P1-P6, R1-R3) on the re-designed poses                                  -> $OUT/design/robust_check
#   final    finals / movies at NS of the re-designed actions + the S1 poses, comparison figure              -> $OUT/design/final*
# Env overrides (defaults in brackets):
#   OUT [results/autodl_design]   PATTERNS [all six]   STEPS [selftest,calib,predict,design,robust,final]
#   CAL_GENS [20] CAL_POP [12] CAL_N [96] CAL_INITS [0,1,2] PROCS [4]           (calibration; prereg values)
#   GENS [20] POP [12] N_LO [128] N_HI [256] TOPK [2] CONFIRM_EVERY [2] ROB [R1,R2,R3] VARS [tilt,z,shift,fill]
#   CHUNK [0 = whole generation in one GPU batch; set e.g. 96 if the GPU runs out of memory]
#   CLOSURE [calib]  PITCHER_LAW [unset; a gpu_calib fit_pitcher pitcher_law.json]  NS [256,512]  VIDEO [1]
#   T [unset; truncate every run at T s -- smoke tests only]   BACKENDS ["" = GPU; "cpu" = CPU reference, slow]
set -eo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/autodl_design}
PATTERNS=${PATTERNS:-heart,push_heart,layered_heart,tulip,leaf,swan}
STEPS=${STEPS:-selftest,calib,predict,design,robust,final}
CAL_GENS=${CAL_GENS:-20}; CAL_POP=${CAL_POP:-12}; CAL_N=${CAL_N:-96}; CAL_INITS=${CAL_INITS:-0,1,2}; PROCS=${PROCS:-4}
GENS=${GENS:-20}; POP=${POP:-12}; N_LO=${N_LO:-128}; N_HI=${N_HI:-256}; TOPK=${TOPK:-2}; CONFIRM_EVERY=${CONFIRM_EVERY:-2}
ROB=${ROB-R1,R2,R3}; VARS=${VARS:-tilt,z,shift,fill}; CHUNK=${CHUNK:-0}; CLOSURE=${CLOSURE:-calib}; NS=${NS:-256,512}
VIDEO=${VIDEO:-1}
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4}
mkdir -p "$OUT"
BK=""
if [ "${BACKENDS:-}" = "cpu" ]; then BK="--solver-backend cpu --pitcher-backend cpu"; fi
TT=""
if [ -n "${T:-}" ]; then TT="--T $T"; fi
has() { [[ ",$STEPS," == *",$1,"* ]]; }

if has selftest; then
  python -m latte_imex.test_gpu_design 2>&1 | tee -a "$OUT/log_selftest.txt"
fi
if has calib; then
  python -m latte_imex.gpu_calib run --out "$OUT/calib" --N "$CAL_N" --popsize "$CAL_POP" --gens "$CAL_GENS" \
    --inits "$CAL_INITS" --procs "$PROCS" --chunk "$CHUNK" $TT $BK 2>&1 | tee -a "$OUT/log_calib.txt"
fi
if has predict; then
  python -m latte_imex.gpu_calib predict --out "$OUT/calib" --N 128 --N-fig 256 --procs "$PROCS" --chunk "$CHUNK" $TT \
    $BK 2>&1 | tee -a "$OUT/log_predict.txt"
fi
CL=s1
if [ "$CLOSURE" = "calib" ]; then
  if [ -f "$OUT/calib/best.json" ]; then CL="$OUT/calib/best.json"; else echo "no $OUT/calib/best.json: designing with the S1 closure"; fi
elif [ "$CLOSURE" != "s1" ]; then CL="$CLOSURE"; fi
PL=""
if [ -n "${PITCHER_LAW:-}" ]; then PL="--pitcher-law $PITCHER_LAW"; fi
if has design; then
  python -m latte_imex.gpu_design run --out "$OUT/design" --patterns "$PATTERNS" --gens "$GENS" --popsize "$POP" \
    --N-lo "$N_LO" --N-hi "$N_HI" --topk "$TOPK" --confirm-every "$CONFIRM_EVERY" --rob "$ROB" --vars "$VARS" \
    --chunk "$CHUNK" --closure "$CL" $PL $TT $BK 2>&1 | tee -a "$OUT/log_design.txt"
fi
if has robust; then
  python -m latte_imex.gpu_design robust --out "$OUT/design" --patterns "$PATTERNS" --N "$N_HI" --chunk "$CHUNK" $TT $BK \
    2>&1 | tee -a "$OUT/log_robust.txt"
fi
if has final; then
  V=""; if [ "$VIDEO" = "1" ]; then V="--video"; fi
  python -m latte_imex.gpu_design final --out "$OUT/design" --patterns "$PATTERNS" --N "$NS" $V $TT $BK \
    2>&1 | tee -a "$OUT/log_final.txt"
fi
echo "outputs: $OUT/calib/{calib.json,best.json,predict/} $OUT/design/{design.json,data/,robust_check/,final/,fig_design_compare.png}"
