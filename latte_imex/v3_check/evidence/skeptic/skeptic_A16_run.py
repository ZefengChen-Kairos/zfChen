"""A16 check: (1) in-frame velocity growth vs frame-start u_max (CFL lag); (2) time-step convergence of the heart script
(old chi closure, DEFAULT_PARAMS) under cfl refinement and under frame_dt refinement (nested clocks).
usage: python skeptic_A16_run.py <pattern> <N> <cfl> <frame_den> <out.npz>"""
import sys, time, json
import numpy as np
from latte_imex.solver import Solver, Params, Numerics
from latte_imex.actions import SCRIPTS
from latte_imex.run import DEFAULT_PARAMS

pattern, N, cfl, fden, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
num = Numerics(N=N, cfl=cfl, frame_dt=1.0 / fden)
sol = Solver(Params(**DEFAULT_PARAMS), num)
script = SCRIPTS[pattern]()
g = sol.g
rec = dict(t=[], umax0=[], u_end=[], u_sub=[], nsub=[], fdt=[], retries=[], active=[], ru=[])
sub_u = []
orig = sol.substep


def wrapped(dt, s, Lam, vsx, vsy):
    ok = orig(dt, s, Lam, vsx, vsy)
    lsafe = np.where(g.mask, sol.l, 1.0)
    sub_u.append(float(np.max(np.hypot(sol.qx, sol.qy) / lsafe)))
    return ok


sol.substep = wrapped
t0 = time.time()
while sol.t < script.T - 1e-9:
    probe, _ = script.sample(sol.t); fdt = num.frame_dt
    if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
        fdt = min(fdt, max(num.scan_safety * min(g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
    inl, _ = script.sample(sol.t + 0.5 * fdt)
    # replicate the frame-start bound of advance_frame (chi closure: no Ge / Ub)
    s, Lam, vsx, vsy, chi = sol.inlet_fields(inl)
    lsafe = np.where(g.mask, sol.l, 1.0)
    umax0 = float(np.max(np.hypot(sol.qx, sol.qy) / lsafe))
    if inl.active and inl.S_eff > 0:
        umax0 = max(umax0, float(np.max(np.hypot(vsx, vsy) * (s + Lam > 0))))
    r0 = sol.stats["retries"]
    sub_u.clear()
    chi, nsub = sol.advance_frame(inl, frame_dt=fdt)
    lsafe = np.where(g.mask, sol.l, 1.0)
    u_end = float(np.max(np.hypot(sol.qx, sol.qy) / lsafe))
    rec["t"].append(sol.t); rec["umax0"].append(umax0); rec["u_end"].append(u_end)
    rec["u_sub"].append(max(sub_u)); rec["nsub"].append(nsub); rec["fdt"].append(fdt)
    rec["retries"].append(sol.stats["retries"] - r0); rec["active"].append(bool(inl.active and inl.S_eff > 0))
    rec["ru"].append(max(sub_u) / max(umax0, 1e-9))
el = time.time() - t0
c = sol.concentration()
np.savez_compressed(out, c=c, l=sol.l, qx=sol.qx, qy=sol.qy, m=sol.m, mask=g.mask,
                    **{k: np.array(v) for k, v in rec.items()})
ru = np.array(rec["ru"]); act = np.array(rec["active"])
summary = dict(pattern=pattern, N=N, cfl=cfl, frame_den=fden, elapsed=el, frames=len(ru), steps=sol.stats["steps"],
               retries=sol.stats["retries"], max_substeps=sol.stats["max_substeps"], max_u=sol.stats["max_u"],
               ratio_max=float(ru.max()), ratio_p99=float(np.percentile(ru, 99)), ratio_p90=float(np.percentile(ru, 90)),
               frames_ratio_gt_1=int((ru > 1.0).sum()), frames_ratio_gt_1p41=int((ru > 1.414).sum()),
               frames_ratio_gt_2=int((ru > 2.0).sum()), frames_active=int(act.sum()),
               fdt_min=float(np.min(rec["fdt"])), fdt_mean=float(np.mean(rec["fdt"])),
               ledger=sol.ledger())
print(json.dumps(summary), flush=True)
json.dump(summary, open(out.replace(".npz", ".json"), "w"))
