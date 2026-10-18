"""Forward parity: the Warp solver against solver.Solver on the first frames of a real pour (same arrival record).
    python -m latte_imex.diff.test_parity [--pattern heart --N 64 --frames 150]"""
import argparse
import time

import numpy as np

from ..solver import Numerics, Params, Solver
from . import schedule as SC
from . import wsolver as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="heart")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    script, pour = SC.pour_script(a.pattern, a.N)
    S = SC.build(script, a.N)
    n = min(a.frames, S["n"])
    P = Params(**script.params)
    print(f"{a.pattern}: {S['n']} frames, testing {n}; return law {S['return_law']}; params "
          + " ".join(f"{k} {getattr(P, k):.4g}" for k in W.PARAMS))
    # reference
    ref = Solver(P, Numerics(N=a.N))
    t0 = time.time()
    nref = []
    for k in range(n):
        _, ns = ref.advance_frame(S["inlets"][k], frame_dt=S["fdt"][k])
        nref.append(ns)
    t_ref = time.time() - t0
    # warp
    import warp as wp
    wp.init()
    g = W.Geo(a.N, device=a.device)
    sol = W.WSolver(g, v05=S["return_law"] == "v05")
    Pnp = np.array([getattr(P, k) for k in W.PARAMS], float)
    Pw = wp.array(Pnp, dtype=W.F, device=g.dev)
    Sw = {k: wp.array(S[k], dtype=W.F, device=g.dev) for k in W.SCHED}
    st = sol.initial_state()
    t0 = time.time()
    nw = []
    for k in range(n):
        st, ns = sol.frame(st, Pw, Pnp, Sw, k, float(S["fdt"][k]), int(S["q"][k]), bool(S["active"][k]))
        nw.append(ns)
    t_w = time.time() - t0
    out = sol.to_numpy(st)
    msk = g.mask_np
    ok = True
    for key, rv in (("l", ref.l), ("qx", ref.qx), ("qy", ref.qy), ("m", ref.m)):
        d = np.abs(out[key] - rv)[msk].max()
        sc = np.abs(rv[msk]).max() + 1e-300
        print(f"  {key}: max |warp - numpy| {d:.3e} (relative {d / sc:.2e})")
        ok &= d / sc < 1e-6
    print(f"  substeps equal: {nw == nref} (warp {sum(nw)}, numpy {sum(nref)}); cg mass {sol.stats['cg_mass']} vs "
          f"{ref.stats['cg_mass']}, visc {sol.stats['cg_visc']} vs {ref.stats['cg_visc']}")
    print(f"  wall: numpy {t_ref:.1f} s, warp ({g.dev}) {t_w:.1f} s")
    print("PASS" if ok and nw == nref else "FAIL")


if __name__ == "__main__":
    main()
