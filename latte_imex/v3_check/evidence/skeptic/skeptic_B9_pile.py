"""Pile test for claim B9: a foam disc (c=1) of step height dl on a film l=1, no inlet, v2F physics.
Does the Bingham film hold it?  Variants: tau_y, per-frame vs per-substep stress, nu_max."""
import sys, json, math, time
sys.path.insert(0, '/home/user/zfChen')
import numpy as np
from latte_imex.solver import Params, Numerics, Solver, Inlet
from latte_imex.joint_cma import BASE_V2
h = json.load(open('/home/user/zfChen/runs/action_opt/joint_cma_v2/history.json'))
phys = dict(BASE_V2, **h['best']['best']['phys'])
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
dl = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
FDT = float(sys.argv[4]) if len(sys.argv) > 4 else 1/120

def run(over, per_frame, dl, R0=0.1):
    P = Params(**dict(phys, **over))
    S = Solver(P, Numerics(N=N, ssa_per_frame=per_frame))
    g = S.g
    w = 2 * g.h
    prof = 0.5 * (1 - np.tanh((g.r - R0) / w))
    S.l = np.where(g.mask, 1.0 + dl * prof, 1.0)
    S.m = np.where(g.mask, S.l * prof, 0.0)   # pure foam (c~1) in the pile, crema outside
    S.deposited = float(S.m[g.mask].sum() * g.area)
    inl = Inlet(active=False)
    out = []
    t0 = time.time()
    nf = int(round(T / FDT))
    for k in range(nf + 1):
        if k in (0, nf // 5, nf // 2, nf) or abs(k*FDT-0.1)<FDT/2 or abs(k*FDT-0.25)<FDT/2:
            lm = S.l[g.mask]; mm = S.m[g.mask]
            area_half = float(((S.l - 1.0) > 0.5 * dl)[g.mask].sum() * g.area)
            r_eq = math.sqrt(area_half / math.pi)
            umax = float(np.max(np.hypot(S.qx, S.qy) / np.where(g.mask, S.l, 1)))
            out.append((round(S.t, 3), round(float(lm.max()) - 1, 3), round(r_eq / R0, 3), round(float(mm.max()), 3), round(umax, 3)))
        if k < nf:
            S.advance_frame(inl, FDT)
    print(f"cp={S.P.cp:.2f} tau_y={P.tau_y:.3f} nu_max={P.nu_max} eps={P.yield_eps} per_frame={per_frame} dl={dl}  "
          f"[(t, max(l)-1, r_half/R0, max m, umax)] = {out}   fdt={FDT:.4g} subst_max={S.stats['max_substeps']} cg_visc={S.stats['cg_visc']} ({time.time()-t0:.0f}s)", flush=True)

variants = [
    (dict(tau_y=0.0), False),
    (dict(), False),
    (dict(nu_max=50.0), False),
    (dict(tau_y=10.0), False),
    (dict(tau_y=10.0, nu_max=500.0), False),
]
for over, pf in variants:
    run(over, pf, dl)
