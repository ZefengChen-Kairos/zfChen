"""B4 check 2: equal-thickness raft (l=1 everywhere, m=1 in a disc) given an initial velocity u0 in closure v2.
Question: does the crema ahead of the raft move at the raft speed (single velocity), pile up (l>1), or flow around?"""
import sys, time, numpy as np
sys.path.insert(0, '/home/user/zfChen')
from latte_imex.solver import Solver, Params, Numerics, Inlet

def run(tau_y, beta=0.5, N=64, T=0.3, u0=0.4):
    P = Params(closure="v2", beta=beta, tau_y=tau_y, nu=1e-3, g_red=75.0, H_f=0.003, H=0.05, D=1e-7,
               l_skin=0.03, yield_eps=0.02, nu_max=0.5, m_o=0.2)
    sol = Solver(P, Numerics(N=N, cfl=0.5))
    g = sol.g; mask = g.mask
    raft = (np.hypot(g.x + 0.10, g.y) < 0.08) & mask
    sol.m = np.where(raft, 1.0, 0.0)
    sol.qx = np.where(raft, u0, 0.0)
    print(f"\n=== cp={sol.P.cp:.3f} tau_y={tau_y} beta={beta} u0={u0}")
    j = N // 2
    def rep(tag):
        l, m = sol.l, sol.m; u = sol.qx / np.where(mask, l, 1); v = sol.qy / np.where(mask, l, 1)
        c = m / np.where(mask, l, 1)
        cx = (m * g.x)[mask].sum() / m[mask].sum()
        xs = g.x[j]
        print(f"[{tag}] t={sol.t:.3f} raft centroid x={cx:+.4f} (disp {cx+0.0996:+.4f})  <u>_foam={(u*m)[mask].sum()/m[mask].sum():+.4f}  "
              f"<u>_crema={(u*(l-m))[mask].sum()/(l-m)[mask].sum():+.4f}  l range in cup [{l[mask].min():.3f},{l[mask].max():.3f}]")
        print("   x      c      l      u      v(y=+0.11)")
        for i in range(N):
            if -0.3 < xs[i] < 0.25 and i % 2 == 0:
                print(f" {xs[i]:+.3f} {c[j,i]:6.3f} {l[j,i]:6.3f} {u[j,i]:+.4f}   {v[j+7,i]:+.4f}")
        # velocity of foam cells vs crema cells immediately across the front (row y=0)
        row_c = c[j]; fr = np.where((row_c[:-1] > 0.5) & (row_c[1:] < 0.5))[0]
        if len(fr):
            i = fr[-1]
            print(f"   front at x={xs[i]:+.3f}: u(foam side, c={row_c[i]:.2f})={u[j,i]:+.4f}  u(crema side, c={row_c[i+1]:.2f})={u[j,i+1]:+.4f}  l ahead (i+1..i+4)={np.round(l[j,i+1:i+5],3)}")
        print(f"   cells 0.1<c<0.9 on row: {int(((row_c>0.1)&(row_c<0.9)).sum())}   max|v| in cup={np.abs(v[mask]).max():.4f}")
    rep("init")
    t0 = time.time(); nfr = int(round(T * 120))
    for k in range(nfr):
        sol.advance_frame(Inlet(active=False))
        if k + 1 in (6, 18):
            rep(f"frame {k+1}")
    rep("final"); print(f"   {nfr} frames in {time.time()-t0:.1f}s, substeps max {sol.stats['max_substeps']}")

for ty in (0.3, 0.0):
    run(ty)
