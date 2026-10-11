"""B2 check: U_v magnitude / uniform sink / far-field film compression in v2 (read-only)."""
import sys, math, json, time
import numpy as np
sys.path.insert(0, "/home/user/zfChen")
from latte_imex.solver import Solver, Params, Numerics, Inlet
D_L = 0.08; N = 96; h = 1.0 / N
V2F = json.load(open("/home/user/zfChen/results/joint_sweep/codexmap/v2_final_fit_N96.json"))["phys"]
Q = 23.3e-6; U = 9.27; d = 0.0783; r1 = 3.86e-3 / D_L; r2 = 2.54e-3 / D_L; uin = 1.88
def inlet(x_hit=(0.0, 0.1)):
    return Inlet(active=True, x_hit=x_hit, S_eff=Q / D_L**3, u_in=(0.0, -uin), U_perp=U, d_jet=d, r1=r1, r2=r2, phi=0.0, scan_speed=0.1)

def fields(pd, x_hit=(0.0, 0.1)):
    sol = Solver(Params(**pd), Numerics(N=N)); P = sol.P
    inl = inlet(x_hit)
    sol.inlet_fields(inl); sol._bulk_flow()
    g = sol.g; m = g.mask
    A = m.sum() * g.area
    print(f"  S_eff={inl.S_eff:.4f}  S_v={sol._S:.4f}/s  S_E={sol._SE:.4f}/s  A_cup={A:.4f} D^2  S_v/A={sol._S/A:.4f}/s  R_c={sol._Rc*80:.2f} mm")
    r = np.hypot(g.x - x_hit[0], g.y - x_hit[1])
    spd = np.hypot(*sol.Ub)
    for rr in (0.05, 0.1, 0.15, 0.2, 0.3):
        sel = (abs(r - rr) < h) & m
        an = sol._S / (2 * math.pi * rr) * (1 - rr**2 / 0.49**2)   # centred-jet analytic 2-D potential
        print(f"    r={rr:.2f} D: median|U_s|={np.median(spd[sel]):.4f}  max={spd[sel].max():.4f}  min={spd[sel].min():.4f} D/s   (centred analytic S_v/(2 pi r)(1-r^2/R^2) = {an:.4f})")
    dv = sol.divUs
    far = (r > 0.2) & m
    print(f"    divUs far (r>0.2): median {np.median(dv[far]):.4f}  min {dv[far].min():.4f} max {dv[far].max():.4f} /s;  divUs min overall {dv.min():.1f}, max {dv.max():.1f}")
    return sol

print("=== v2F, cell ON (U_s = U_v + U_c) ===")
fields(V2F)
print("=== v2F, ent_coef=0 (U_s = U_v only) ===")
fields(dict(V2F, ent_coef=0.0))
print("=== v2F, ent_coef=0, jet centred ===")
fields(dict(V2F, ent_coef=0.0), (0.0, 0.0))

# ---- short run: does the far-field film actually thicken at ~S_v/A ? -------------------
def run(pd, T=0.4, label=""):
    sol = Solver(Params(**pd), Numerics(N=N)); g = sol.g; m = g.mask
    inl = inlet()
    r = np.hypot(g.x, g.y - 0.1)
    far = (r > 0.25) & m; mid = (abs(r - 0.15) < h) & m
    l0 = sol.l[far].mean()
    t0 = time.time(); t = 0.0; rows = []
    while t < T - 1e-9:
        sol.advance_frame(inl); t += 1.0 / 120.0
        if abs(t * 10 - round(t * 10)) < 1e-6 or t >= T - 1e-9:
            rows.append((t, sol.l[far].mean(), sol.l[far].min(), sol.l[far].max(), sol.l[mid].mean(), sol.l[m].sum() * g.area, float(np.hypot(sol.qx, sol.qy)[far].max())))
    print(f"=== run {label}: {time.time()-t0:.1f}s wall; far = r>0.25 D from hit ({far.sum()} cells) ===")
    for t, lm, lmin, lmax, lmid, tot, qmax in rows:
        print(f"    t={t:.2f}s  <l>_far={lm:.4f} [{lmin:.4f},{lmax:.4f}]  <l>_r=0.15={lmid:.4f}  total l={tot:.4f}  max|q|_far={qmax:.4f}   growth rate so far ={(math.log(lm/l0)/t if t>0 else 0):.4f}/s")

import os
if os.environ.get("B2_STATIC_ONLY"): sys.exit(0)
base = dict(V2F)
run(dict(base, phi_foam=0.0, kappa_c=0.0, ent_coef=0.0), label="no foam, no traction, no cell: U_v drag only")
run(dict(base, phi_foam=0.0, kappa_c=0.0), label="no foam, no traction, cell ON")
run(base, label="full v2F")
