"""Compare the CUDA back-end with the CPU solver on the same inputs.

    python -m latte_imex.test_cuda                       # on a GPU machine (N=256 heart, 1 s)
    NUMBA_ENABLE_CUDASIM=1 python -m latte_imex.test_cuda --N 32 --T 0.15   # no GPU: simulator, small and slow
"""
import argparse, os, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--pattern", default="heart")
    a = ap.parse_args()
    from .run import run
    S = os.environ.get("LATTE_TEST_DIR", "runs/test_cuda")
    t0 = time.time()
    sol_c, met_c = run(a.pattern, N=a.N, out=f"{S}/cpu", control="v05", video=False, quiet=True, T_max=a.T, backend="cpu")
    t1 = time.time()
    sol_g, met_g = run(a.pattern, N=a.N, out=f"{S}/cuda", control="v05", video=False, quiet=True, T_max=a.T, backend="cuda")
    t2 = time.time()
    A = sol_c.host_state(); B = sol_g.host_state()
    print(f"N={a.N} T={a.T}: cpu {t1-t0:.1f}s ({met_c['stats']['steps']} steps, cg {met_c['stats']['cg_mass']})  "
          f"cuda {t2-t1:.1f}s ({met_g['stats']['steps']} steps, cg {met_g['stats']['cg_mass']}, retries {met_g['stats']['retries']})")
    ok = True
    for k in ("l", "qx", "qy", "m"):
        d = float(np.abs(A[k] - B[k]).max()); sc = max(float(np.abs(A[k]).max()), 1e-12)
        print(f"  {k}: max|cpu-cuda| = {d:.3e}  (relative {d/sc:.2e})")
        ok &= d / sc < 1e-9
    Lc, Lg = met_c["ledger"], met_g["ledger"]
    print(f"  ledger cpu: dep {Lc['deposited']:.6f} lerr {Lc['layer_error']:.1e} merr {Lc['milk_error']:.1e}")
    print(f"  ledger gpu: dep {Lg['deposited']:.6f} lerr {Lg['layer_error']:.1e} merr {Lg['milk_error']:.1e}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
