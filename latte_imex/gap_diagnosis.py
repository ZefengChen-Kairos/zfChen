"""Diagnostic: which inlet feature makes the petal gaps?  Candidate #9 (fixed rerun) tulip at N=256 with
(a) footprint scaled, (b) Codex-style 85% flow pulsation at 2f phase-locked to the wiggle extremes, (c) both.
Unphysical on purpose; nothing in the repo model is changed."""
import json, math, sys, os, time, numpy as np
sys.path.insert(0, "/home/user/zfChen")
import latte_imex.joint_sweep as js
from latte_imex.pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript
from latte_imex.solver import Solver, Params, Numerics, Inlet
from latte_imex.optimize import whiteness_sim, loss_fields, target_at
variant = sys.argv[1]; N = int(sys.argv[2]); out = sys.argv[3]
fscale = 0.3 if variant in ("foot", "both") else 1.0
pulse = 0.85 if variant in ("pulse", "both") else 0.0
r = json.load(open("runs/joint_sweep_r2fix/results.json"))[9]
a = r["actions"]["tulip"]; physics = r["physics"]; f = a["wig_f"]

class Diag(Coupling):
    def inlet(self, rec, scan_speed=0.0):
        inl = super().inlet(rec, scan_speed)
        if not inl.active: return inl
        lab = rec.get("label", "")
        S = inl.S_eff; r1, r2, d = inl.r1, inl.r2, inl.d_jet
        if pulse and (lab.startswith("base") or lab.startswith("petal")):
            S *= 1.0 + pulse * (-math.cos(4 * math.pi * f * rec["t"]))
        if fscale != 1.0:
            r1, r2, d = r1 * fscale, r2 * fscale, d * fscale
        return Inlet(active=True, x_hit=inl.x_hit, S_eff=S, u_in=inl.u_in, U_perp=inl.U_perp, d_jet=d,
                     r1=max(r1, 0.5 / N), r2=max(r2, 0.5 / N), phi=inl.phi, scan_speed=inl.scan_speed)

geom = GridPitcherGeometry.load(nr=6)
cp = Diag(c_S=physics["kQ"] / js.D_L ** 3, c_U=1.0 / js.D_L, c_u=1.0 / js.D_L, footprint="physical")
script = BaristaScript(moves=js.tulip_moves(a), pitcher=Pitcher(geom=geom, V0=250e-6), coupling=cp)
pd = {k: v for k, v in physics.items() if k != "kQ"}; pd.update(D=1e-7, kappa_Q=125.0, return_law="v05")
sol = Solver(Params(**pd), Numerics(N=N)); t0 = time.time()
while sol.t < script.T - 1e-9:
    probe, _ = script.sample(sol.t); fdt = sol.num.frame_dt if hasattr(sol, "num") else 1 / 120
    num = sol.num if hasattr(sol, "num") else Numerics(N=N)
    if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
        fdt = min(fdt, max(num.scan_safety * min(sol.g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
    inl, _ = script.sample(sol.t + 0.5 * fdt); sol.advance_frame(inl, frame_dt=fdt)
c = sol.concentration(); mask = sol.g.mask
s = whiteness_sim(c, mask); L, parts = loss_fields(s, target_at("tulip", N), mask, N / 48)
os.makedirs(out, exist_ok=True)
np.save(os.path.join(out, f"{variant}_N{N}.npy"), c.astype(np.float32))
json.dump(dict(variant=variant, N=N, loss=L, **parts, elapsed=time.time() - t0, retries=sol.stats["retries"]), open(os.path.join(out, f"{variant}_N{N}.json"), "w"))
print(variant, N, "loss", round(L, 3), {k: round(v, 3) for k, v in parts.items()}, f"{time.time()-t0:.0f}s", flush=True)
