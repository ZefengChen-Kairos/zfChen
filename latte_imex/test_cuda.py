"""Compare the CUDA back-end with the CPU solver on the same inputs, and the batched mode with single runs.

    python -m latte_imex.test_cuda                       # on a GPU machine (N=256 heart, 1 s; batch check at N=64)
    NUMBA_ENABLE_CUDASIM=1 python -m latte_imex.test_cuda --N 20 --T 0.03 --batchN 16 --frames 2   # simulator, slow

Checks (exit code 1 on failure):
  1. run.run(pattern, control 'v05') CPU vs CUDA: max |cpu - cuda| / max|cpu| < 1e-9 for l, m and < 1e-8 for q (test_visc's tolerance; the
     viscous PCG stops at visc_tol = 1e-8: q differs at that level, as before the batched rewrite), ledgers.
  2. BatchCudaSolver with B = 3 members (different inlet paths, one member idle for a frame) == three B = 1 runs
     (relative 1e-12; a member does not depend on the others).
  3. FP32 storage (experimental): runs, finite, within 1e-3 of FP64 (only reported, not part of PASS).
"""
import argparse, math, os, sys, time
import numpy as np


def _inlet(k, phase=0.0):
    from .solver import Inlet
    ang = 0.4 * k + phase
    return Inlet(active=True, x_hit=(0.42 * math.cos(ang), 0.42 * math.sin(ang)), S_eff=0.08, U_perp=4.0, d_jet=0.05,
                 r1=0.09, r2=0.07, phi=ang, u_in=(0.3, -0.2), scan_speed=0.5)


def test_batch(N=64, frames=6, verbose=True):
    from .solver import Params, Numerics
    from .cuda_solver import CudaSolver, BatchCudaSolver
    P = Params(cp=1.0, nu=2e-3, closure="chi", return_law="v05", kappa_r=0.1)
    num = Numerics(N=N)
    plan = [[_inlet(k) for k in range(frames)],
            [_inlet(k, 1.3) for k in range(frames)],
            [None if k == 1 else _inlet(k, 2.6) for k in range(frames)]]
    B = BatchCudaSolver([P, P, P], num)
    for k in range(frames):
        B.advance_frames([plan[b][k] for b in range(3)])
    worst = 0.0
    for b in range(3):
        s = CudaSolver(P, num)
        for inl in plan[b]:
            if inl is not None:
                s.advance_frame(inl)
        A, C = s.host_state(), B.host_state(b)
        rel = max(float(np.abs(A[k] - C[k]).max() / max(np.abs(A[k]).max(), 1e-12)) for k in A)
        same = (s.stats["cg_mass"] == B.stats[b]["cg_mass"] and s.stats["cg_visc"] == B.stats[b]["cg_visc"]
                and abs(s.deposited - B.deposited[b]) < 1e-15 and abs(s.spilled - B.spilled[b]) < 1e-15 and s.t == B.t[b])
        worst = max(worst, rel)
        if verbose:
            print(f"  batch member {b}: max rel diff vs single run {rel:.1e}, counters/ledger equal {same}", flush=True)
        if not same:
            worst = max(worst, 1.0)
    lp = B.stats[0]["launches"] / max(B.stats[0]["steps"], 1)
    if verbose:
        print(f"  batch: {B.stats[0]['launches']} launches over {B.stats[0]['frames']} frames "
              f"(member 0: {B.stats[0]['steps']} sub-steps, {lp:.1f} launches per sub-step incl. frame kernels; "
              f"cg {B.stats[0]['cg_mass']}+{B.stats[0]['cg_visc']}), host syncs {B.stats[0]['host_syncs']}", flush=True)
    return worst < 1e-12, worst


def test_fp32(N=64, frames=6, verbose=True):
    from .solver import Params, Numerics
    from .cuda_solver import CudaSolver
    P = Params(cp=1.0, nu=2e-3, closure="chi", return_law="v05", kappa_r=0.1)
    a, b = CudaSolver(P, Numerics(N=N)), CudaSolver(P, Numerics(N=N), dtype=np.float32)
    for k in range(frames):
        a.advance_frame(_inlet(k)); b.advance_frame(_inlet(k))
    A, C = a.host_state(), b.host_state()
    rel = max(float(np.abs(A[k] - C[k]).max() / max(np.abs(A[k]).max(), 1e-12)) for k in A)
    if verbose:
        print(f"  fp32 storage: max rel diff vs fp64 {rel:.1e} (cg {b.stats['cg_mass']} vs {a.stats['cg_mass']})", flush=True)
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--pattern", default="heart")
    ap.add_argument("--batchN", type=int, default=64)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--skip-run", action="store_true")
    a = ap.parse_args()
    ok = True
    if not a.skip_run:
        from .run import run
        S = os.environ.get("LATTE_TEST_DIR", "runs/test_cuda")
        t0 = time.time()
        sol_c, met_c = run(a.pattern, N=a.N, out=f"{S}/cpu", control="v05", video=False, quiet=True, T_max=a.T, backend="cpu")
        t1 = time.time()
        sol_g, met_g = run(a.pattern, N=a.N, out=f"{S}/cuda", control="v05", video=False, quiet=True, T_max=a.T, backend="cuda")
        t2 = time.time()
        A = sol_c.host_state(); B = sol_g.host_state()
        st = met_g["stats"]
        print(f"N={a.N} T={a.T}: cpu {t1-t0:.1f}s ({met_c['stats']['steps']} steps, cg {met_c['stats']['cg_mass']}+"
              f"{met_c['stats']['cg_visc']})  cuda {t2-t1:.1f}s ({st['steps']} steps, cg {st['cg_mass']}+{st['cg_visc']}, "
              f"retries {st['retries']}, {st['launches'] / max(st['steps'], 1):.1f} launches and "
              f"{st['host_syncs'] / max(st['steps'], 1):.2f} host syncs per sub-step)")
        for k in ("l", "qx", "qy", "m"):
            d = float(np.abs(A[k] - B[k]).max()); sc = max(float(np.abs(A[k]).max()), 1e-12)
            lim = 1e-8 if k in ("qx", "qy") else 1e-9
            print(f"  {k}: max|cpu-cuda| = {d:.3e}  (relative {d/sc:.2e}, limit {lim:.0e})")
            ok &= d / sc < lim
        Lc, Lg = met_c["ledger"], met_g["ledger"]
        print(f"  ledger cpu: dep {Lc['deposited']:.6f} spill {Lc['spilled']:.6f} lerr {Lc['layer_error']:.1e} merr {Lc['milk_error']:.1e}")
        print(f"  ledger gpu: dep {Lg['deposited']:.6f} spill {Lg['spilled']:.6f} lerr {Lg['layer_error']:.1e} merr {Lg['milk_error']:.1e}")
        ok &= abs(Lc["deposited"] - Lg["deposited"]) < 1e-9 * max(Lc["deposited"], 1e-9) + 1e-14
        ok &= abs(Lc["spilled"] - Lg["spilled"]) < 1e-9 * max(Lc["spilled"], 1e-9) + 1e-14
    good, worst = test_batch(a.batchN, a.frames)
    print(f"  batch == single runs: {good} (worst {worst:.1e})")
    ok &= good
    test_fp32(a.batchN, a.frames)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
