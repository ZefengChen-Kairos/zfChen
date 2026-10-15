#!/bin/bash
# Aligned model (symmetric viscosity, off-cup accounting) on the forward pitcher driven by the S1 ideal-hand poses:
#   (a) six patterns at N = 256 and 512: finals, movies, metrics
#   (b) hand-error robustness grid (P1-P6, R1-R3 of the prereg) at N = 256, batched on the GPU
#   (c) figures (six-pattern montage, robustness grid) and results.json
#   bash scripts/autodl_run_v05_aligned.sh                 OUT=... NS=256,512 NR=256 CHUNK=0 to override
set -eo pipefail
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0
cd "$(dirname "$0")/.."
OUT=${OUT:-results/autodl_v05}
NS=${NS:-256,512}
NR=${NR:-256}
CHUNK=${CHUNK:-0}
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4}
mkdir -p "$OUT"
python -m latte_imex.autodl_v05 aligned --N "$NS" --video --out "$OUT" 2>&1 | tee "$OUT/log_aligned.txt"
python -m latte_imex.autodl_v05 robust --N "$NR" --chunk "$CHUNK" --out "$OUT" 2>&1 | tee "$OUT/log_robust.txt"
python -m latte_imex.autodl_v05 figures --out "$OUT" 2>&1 | tee "$OUT/log_figures.txt"
echo "outputs: $OUT/fig_aligned_six.png $OUT/fig_robust_grid.png $OUT/results.json $OUT/aligned/N*/<pattern>/movie.mp4"
