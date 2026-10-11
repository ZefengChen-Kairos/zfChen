"""B2 check 2: is the far-field film compression blocked by film pressure? (read-only)"""
import sys, math, json, time
import numpy as np
sys.path.insert(0, "/home/user/zfChen")
from latte_imex.solver import Solver, Params, Numerics, Inlet
D_L = 0.08; N = 96; h = 1.0 / N
V2F = json.load(open("/home/user/zfChen/results/joint_sweep/codexmap/v2_final_fit_N96.json"))["phys"]
Q = 23.3e-6; U = 9.27; d = 0.0783; r1 = 3.86e-3 / D_L; r2 = 2.54e-3 / D_L; uin = 1.88
inl = Inlet(active=True, x_hit=(0.0, 0.1), S_eff=Q / D_L**3, u_in=(0.0, -uin), U_perp=U, d_jet=d, r1=r1, r2=r2, phi=0.0, scan_speed=0.1)
def run(pd, T, label):
    sol = Solver(Params(**pd), Numerics(N=N)); g = sol.g; m = g.mask
    r = np.hypot(g.x, g.y - 0.1); far = (r > 0.25) & m
    Uv = None
    t0 = time.time(); t = 0.0; print(f"=== {label}: cp={sol.P.cp:.3f} beta={sol.P.beta:.2f} ===")
    k = 0
    while t < T - 1e-9:
        sol.advance_frame(inl); t += 1.0 / 120.0; k += 1
        if Uv is None: Uv = np.hypot(*sol.Ub)
        if k % 24 == 0 or t >= T - 1e-9:
            u = np.hypot(sol.qx, sol.qy) / np.where(m, sol.l, 1.0)
            ratio = np.median((u / np.maximum(Uv, 1e-9))[far])
            print(f"    t={t:.2f}s <l>_far={sol.l[far].mean():.4f} [{sol.l[far].min():.4f},{sol.l[far].max():.4f}]  median |u|/|U_v| far={ratio:.3f}  l(footprint cell)={sol.l[m][np.argmax(sol._K[m])]:.3f}  total l={sol.l[m].sum()*g.area:.4f}")
    print(f"    ({time.time()-t0:.1f}s wall)")
base = dict(V2F, phi_foam=0.0, kappa_c=0.0, ent_coef=0.0)
run(base, 1.0, "U_v drag only, v2F cp")
run(dict(base, g_red=V2F["g_red"] / 100), 1.0, "U_v drag only, cp/10")
run(dict(base, g_red=V2F["g_red"] / 10000), 0.6, "U_v drag only, cp/100")
