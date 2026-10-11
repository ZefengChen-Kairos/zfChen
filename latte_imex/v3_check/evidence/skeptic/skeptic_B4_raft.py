"""B4 check: in closure v2, can a foam raft translate through crema? Does crema move with it (same u), pile ahead, or flow around?
No inlet; raft given an initial velocity; drag beta pulls towards U_s = 0 (no coffee flow)."""
import sys, time, numpy as np
sys.path.insert(0, '/home/user/zfChen')
from latte_imex.solver import Solver, Params, Numerics, Inlet

def run(tau_y, beta=0.5, N=64, T=0.25, u0=0.4):
    P = Params(closure="v2", beta=beta, tau_y=tau_y, nu=1e-3, g_red=75.0, H_f=0.003, H=0.05, D=1e-7,
               l_skin=0.03, yield_eps=0.02, nu_max=0.5, m_o=0.2)
    sol = Solver(P, Numerics(N=N, cfl=0.5))
    g = sol.g
    raft = (np.hypot(g.x + 0.10, g.y) < 0.08) & g.mask
    sol.l = np.where(raft, 4.0, 1.0); sol.m = np.where(raft, 4.0, 0.0)
    sol.qx = np.where(raft, sol.l * u0, 0.0)
    print(f"cp={sol.P.cp:.3f} tau_y={tau_y} beta={beta}")
    def rep(tag):
        mask = g.mask; l, m = sol.l, sol.m
        u = sol.qx / np.where(mask, l, 1)
        cx = (m * g.x)[mask].sum() / m[mask].sum()
        crema = l - m
        ahead = ((g.x > cx + 0.08) & (g.x < cx + 0.16) & (np.abs(g.y) < 0.04)) & mask   # strip ahead of the raft
        side = ((np.abs(g.x - cx) < 0.08) & (np.abs(g.y) > 0.09) & (np.abs(g.y) < 0.16)) & mask
        j = N // 2
        xs = g.x[j]; row_c = (m / np.where(mask, l, 1))[j]; row_l = l[j]; row_u = u[j]
        print(f"[{tag}] t={sol.t:.3f} raft centroid x={cx:+.4f}  u_raft(mean over m)={(u*m)[mask].sum()/m[mask].sum():+.4f}")
        print("   x      c      l      u   (row y=0, from -0.3 to +0.2)")
        for i in range(N):
            if -0.3 < xs[i] < 0.2 and i % 2 == 0:
                print(f" {xs[i]:+.3f} {row_c[i]:6.3f} {row_l[i]:6.3f} {row_u[i]:+.4f}")
        print(f"   crema volume: total={crema[mask].sum()*g.area:.5f}  ahead-strip mean l={l[ahead].mean():.3f} mean u={u[ahead].mean():+.4f}  side-strip mean l={l[side].mean():.3f}")
        # interface sharpness: number of cells along y=0 with 0.1<c<0.9
        print(f"   cells with 0.1<c<0.9 on y=0 row: {int(((row_c>0.1)&(row_c<0.9)).sum())}")
    rep("init")
    t0 = time.time(); nfr = int(round(T * 120))
    for k in range(nfr):
        sol.advance_frame(Inlet(active=False))
    rep("final"); print(f"   {nfr} frames in {time.time()-t0:.1f}s, stats {sol.stats}")

for ty in (0.3, 0.0):
    run(ty)
