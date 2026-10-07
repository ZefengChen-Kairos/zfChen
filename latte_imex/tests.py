"""Basic checks: static stripe, conservation, cp sweep, timing."""
import sys, time, json
import numpy as np
sys.path.insert(0, '.')
from latte_imex.solver import Solver, Params, Numerics, Inlet


def static_stripe(N=96):
    sol = Solver(Params(), Numerics(N=N))
    g = sol.g
    stripe = (np.abs(g.x) < 0.05) & g.mask
    sol.m[stripe] = sol.l[stripe]
    c0 = sol.concentration()
    for _ in range(50):
        sol.advance_frame(Inlet(active=False))
    c1 = sol.concentration()
    d = np.nanmax(np.abs(c1 - c0))
    print(f"static stripe: max |c change| after {sol.t:.3f}s = {d:.2e}, max|u| = {sol.stats['max_u']:.2e}")
    return d


def pour_test(N=128, cp=0.3, T=1.0, S=0.06, verbose=True):
    sol = Solver(Params(cp=cp), Numerics(N=N))
    t0 = time.time()
    while sol.t < T - 1e-9:
        inl = Inlet(active=sol.t < 0.6 * T, x_hit=(0.0, -0.15), S_eff=S, u_in=(0.0, 0.0),
                    U_perp=5.0, d_jet=0.07, r1=0.045, r2=0.045)
        sol.advance_frame(inl)
    el = time.time() - t0
    L = sol.ledger(); ke, pe = sol.energy()
    c = sol.concentration()
    white = float(np.nansum(c > 0.4) * sol.g.area)
    if verbose:
        print(f"N={N} cp={cp}: {el:.1f}s, steps={sol.stats['steps']}, max_sub={sol.stats['max_substeps']}, "
              f"cg(mass/visc/mix)={sol.stats['cg_mass']}/{sol.stats['cg_visc']}/{sol.stats['cg_mix']}, "
              f"layer_err={L['layer_error']:.2e} milk_err={L['milk_error']:.2e} clip={sol.stats['clip_mass']:.2e} "
              f"maxl={sol.l.max():.3f} maxu={sol.stats['max_u']:.3f} white_area={white:.3f} KE={ke:.2e} PE={pe:.3f}")
    return sol


if __name__ == "__main__":
    static_stripe()
    pour_test(N=128, cp=0.3, T=1.0)
    pour_test(N=128, cp=1.0, T=1.0)
    pour_test(N=128, cp=3.0, T=1.0)
    pour_test(N=256, cp=0.3, T=0.5)
