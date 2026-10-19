"""Forward parity: both paths of the Warp solver (recorded `frame`, fast graph path `fast_forward`) against
solver.Solver on the first frames of a real pour (same arrival record).
    python -m latte_imex.diff.test_parity [--pattern heart --N 64 --frames 150 --device cuda:0 --check-every 1]
                                          [--upwell --circ kappa_M,eps_f,kappa_w --set tau_s=0.05]"""
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
    ap.add_argument("--check-every", dest="check_every", type=int, default=1,
                    help="PCG convergence check interval (1: the same iteration counts as the numpy solver)")
    ap.add_argument("--upwell", action="store_true", help="volume-conserving surface source")
    ap.add_argument("--circ", default="", help="kappa_M,eps_f,kappa_w: surface current (circulation.py)")
    ap.add_argument("--set", default="", help="closure values, e.g. tau_s=0.05")
    a = ap.parse_args()
    script, pour = SC.pour_script(a.pattern, a.N)
    S = SC.build(script, a.N)
    n = min(a.frames, S["n"])
    if a.upwell:
        script.params["upwell"] = True
    if a.circ:
        script.params.update(zip(("kappa_M", "eps_f", "kappa_w"), map(float, a.circ.split(","))))
    script.params.update({k: float(v) for k, v in (x.split("=") for x in a.set.split(",") if x)})
    P = Params(**script.params)
    print(f"{a.pattern}: {S['n']} frames, testing {n}; return law {S['return_law']}; params "
          + " ".join(f"{k} {getattr(P, k):.4g}" for k in W.PARAMS))
    ref = Solver(P, Numerics(N=a.N))
    t0 = time.time()
    nref = []
    for k in range(n):
        _, ns = ref.advance_frame(S["inlets"][k], frame_dt=S["fdt"][k])
        nref.append(ns)
    t_ref = time.time() - t0
    print(f"numpy reference: {n} frames in {t_ref:.1f} s; compiling/running the Warp solver", flush=True)
    import warp as wp
    wp.init()
    g = W.Geo(a.N, device=a.device)
    Pnp = np.array([getattr(P, k) for k in W.PARAMS], float)
    Pw = wp.array(Pnp, dtype=W.F, device=g.dev)
    Sw = {k: wp.array(S[k], dtype=W.F, device=g.dev) for k in W.SCHED}
    tol = 1e-12 if a.check_every == 1 else 1e-6
    ok = True
    for path in ("frame", "fast"):
        sol = W.WSolver(g, v05=S["return_law"] == "v05", check_every=a.check_every, up=a.upwell)
        if a.circ:
            from .run import circ_fields
            sol.set_circ(*circ_fields(S, g, dtype=np.float64, shape=dict(L=P.circ_L, r_f=P.circ_rf)))
        nw = []
        t0 = time.time()
        if path == "frame":
            st = sol.initial_state()
            for k in range(n):
                st, ns = sol.frame(st, Pw, Pnp, Sw, k, float(S["fdt"][k]), int(S["q"][k]), bool(S["active"][k]))
                nw.append(ns)
        else:
            sol.fast_forward(Pw, Pnp, Sw, S, keep=False, frames=n)              # warm-up: graph capture
            sol.reset_cg_iterations()
            t0 = time.time()
            st, _, nw = sol.fast_forward(Pw, Pnp, Sw, S, keep=False, frames=n)
        wp.synchronize()
        t_w = time.time() - t0
        out = sol.to_numpy(st)
        msk = g.mask_np
        print(f"[{path}] wall {t_w:.2f} s (numpy {t_ref:.2f} s), device {g.dev}, graphs {sol.runner.use} "
              f"{sol.runner.stats}")
        for key, rv in (("l", ref.l), ("qx", ref.qx), ("qy", ref.qy), ("m", ref.m)):
            d = np.abs(out[key] - rv)[msk].max()
            sc = np.abs(rv[msk]).max() + 1e-300
            print(f"  {key}: max |warp - numpy| {d:.3e} (relative {d / sc:.2e})")
            ok &= d / sc < tol
        it = sol.cg_iterations()
        print(f"  substeps equal: {nw == nref} (warp {sum(nw)}, numpy {sum(nref)}); cg mass {it['mass']} vs "
              f"{ref.stats['cg_mass']}, visc {it['visc']} vs {ref.stats['cg_visc']}; PCG modes {sol.opM.mode}/"
              f"{sol.opV.mode}, fused sub-step graph {sol.fused}")
        ok &= nw == nref
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
