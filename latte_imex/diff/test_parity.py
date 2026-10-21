"""Forward parity: both paths of the Warp solver (recorded `frame`, fast graph path `fast_forward`) against
solver.Solver on the first frames of a real pour (same arrival record).
    python -m latte_imex.diff.test_parity [--pattern heart --N 64 --frames 150 --device cuda:0 --check-every 1]
                                          [--upwell --circ kappa_M,eps_f,kappa_w --set tau_s=0.05]
                                          [--milk vl|mp5fct|thincfct|mp5fct_nb] [--sink v05|column] [--visc-tr]
                                          [--save out.npz]
(--set takes any solver.Params value, e.g. xi_b=0.9,lam_b=10 (two-phase open layer) or circ_r=5 with --circ.)
--milk: face value of c in the milk transport (wsolver.MILK_SCHEMES; both solvers).  Besides the Warp-numpy agreement it
reports the numpy step C's clip (mass the clip to [0, l] removed / added, largest excursion: 0 = the scheme itself kept
0 <= c <= 1 and conserved the milk) and the range of c in the final states.
--sink: form of the entrainment sink's area flux (wsolver.SINK_FORMS; both solvers; acts with kappa_e > 0, e.g.
--sink column --set kappa_e=0.032).  The numpy reference recovers Q = S_eff / kappa_Q, so kappa_Q is not --set here
(the record's Q divides by the script's kQ).  --visc-tr: the floating sheet's trace stress in the viscous operator
(wsolver.k_sym_Td, kernels.sym_visc_apply_tr)."""
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
    ap.add_argument("--milk", default="vl", choices=W.MILK_SCHEMES,
                    help="face value of c in step C (wsolver.MILK_SCHEMES)")
    ap.add_argument("--sink", default="v05", choices=W.SINK_FORMS,
                    help="form of the entrainment sink's area flux ej (wsolver.SINK_FORMS)")
    ap.add_argument("--visc-tr", dest="visc_tr", action="store_true", help="trace stress of the floating sheet")
    ap.add_argument("--save", default="", help="npz with the final states of numpy / frame / fast (bit-for-bit checks)")
    a = ap.parse_args()
    script, pour = SC.pour_script(a.pattern, a.N)
    S = SC.build(script, a.N)
    n = min(a.frames, S["n"])
    if a.upwell:
        script.params["upwell"] = True
    if a.circ:
        script.params.update(zip(("kappa_M", "eps_f", "kappa_w"), map(float, a.circ.split(","))))
    script.params.update({k: float(v) for k, v in (x.split("=") for x in a.set.split(",") if x)})
    script.params["milk"] = W.milk_code(a.milk)
    script.params["sink"] = W.sink_code(a.sink)
    script.params["visc_tr"] = int(a.visc_tr)
    if "kappa_Q" in a.set:
        ap.error("--set kappa_Q: the arrival record (S_eff, Q) was built with the script's kQ")
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
    msk = g.mask_np
    rng = lambda st: (float((st["m"] / st["l"])[msk].min()), float((st["m"] / st["l"])[msk].max()))
    mtot = float(ref.m[msk].sum() * ref.g.area)
    lo, hi = rng(dict(m=ref.m, l=ref.l))
    print(f"milk transport {a.milk}: numpy clip removed / added {ref.stats['clip_mass']:.3e} of the milk "
          f"({ref.stats['clip_mass'] / max(mtot, 1e-300):.1e} of the total {mtot:.6g}), largest excursion "
          f"{ref.stats.get('clip_max', 0.0):.1e}; final c in [{lo:.3g}, {hi:.17g}]")
    ok &= lo >= -1e-12 and hi <= 1.0 + 1e-12                     # (round-off)
    saved = dict(np_l=ref.l, np_qx=ref.qx, np_qy=ref.qy, np_m=ref.m, np_nsub=np.array(nref))
    for path in ("frame", "fast"):
        sol = W.WSolver(g, v05=S["return_law"] == "v05", check_every=a.check_every, up=a.upwell, milk=a.milk,
                        sink=a.sink, visc_tr=int(a.visc_tr))
        if a.circ:
            from .run import circ_fields
            sol.set_circ(*circ_fields(S, g, dtype=np.float64, shape=dict(L=P.circ_L, r_f=P.circ_rf, r=P.circ_r)))
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
        saved.update({f"{path}_{k}": v for k, v in out.items()})
        saved[f"{path}_nsub"] = np.array(nw)
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
        lo, hi = rng(out)
        print(f"  final c in [{lo:.3g}, {hi:.17g}]; total milk warp {float(out['m'][msk].sum() * g.h * g.h):.15g} numpy "
              f"{mtot:.15g}")
        ok &= nw == nref and lo >= -1e-12 and hi <= 1.0 + 1e-12
    if a.save:
        np.savez(a.save, **saved)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
