"""IMEX + Rusanov surface-layer latte-art simulator (V1 model)."""
import os as _os
# The solver is matrix-free NumPy; BLAS threads only thrash on the small PCG dot products.
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")
