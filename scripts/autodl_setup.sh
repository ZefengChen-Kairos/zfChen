#!/bin/bash
# One-time setup on an AutoDL (or any CUDA 12) machine for the CUDA back-end.
set -e
cd "$(dirname "$0")/.."
python -m pip install -q --upgrade pip
python -m pip install -q numpy scipy matplotlib imageio imageio-ffmpeg numba
# numba's CUDA target: on numba >= 0.61 it lives in the numba-cuda package; the cu12 extra brings the NVVM/runtime libs.
python -m pip install -q "numba-cuda[cu12]" || python -m pip install -q numba-cuda || true
python - <<'PY'
from numba import cuda
print("CUDA available:", cuda.is_available())
if cuda.is_available():
    d = cuda.get_current_device(); print("device:", d.name.decode() if isinstance(d.name, bytes) else d.name, "cc", d.compute_capability)
PY
echo "smoke test (N=256, 1 s of the heart, CPU vs CUDA):"
python -m latte_imex.test_cuda --N 256 --T 1.0
