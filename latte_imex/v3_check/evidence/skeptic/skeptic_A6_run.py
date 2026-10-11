import sys, time, numpy as np
sys.path.insert(0, "/home/user/zfChen")
from latte_imex import action_opt as A
from latte_imex.pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript
from latte_imex.solver import Solver, Params, Numerics
from latte_imex.optimize import visible, loss_fields, target_at
D_L = A.D_L
over = A.MODELS["old"]; print("MODELS[old] =", over)
ph, kQ = A.physics9(); print("kQ =", kQ, " => '1' = D_L/kQ =", 1e3*D_L/kQ, "mm")
geom = GridPitcherGeometry.load(nr=6)
cp = Coupling(c_S=kQ / D_L ** 3, c_U=1.0 / D_L, c_u=1.0 / D_L, footprint="physical")
N = 64
script = BaristaScript(moves=A.scaled_moves("push_heart", 1, 1, 1), pitcher=Pitcher(geom=geom, V0=250e-6), coupling=cp)
pd = dict(ph, D=1e-7, kappa_Q=125.0, return_law="v05"); pd.update(over)
num = Numerics(N=N); sol = Solver(Params(**pd), num); t0 = time.time()
while sol.t < script.T - 1e-9:
    probe, _ = script.sample(sol.t); fdt = num.frame_dt
    if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
        fdt = min(fdt, max(num.scan_safety * min(sol.g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
    inl, _ = script.sample(sol.t + 0.5 * fdt); sol.advance_frame(inl, frame_dt=fdt)
print("elapsed", round(time.time()-t0,1), "s; ledger", {k: round(v,4) for k,v in sol.ledger().items()})
mask = sol.g.mask; c = sol.concentration(); l = sol.l; m = sol.m
white = mask & (c > 0.3); sat = mask & (c >= 0.6)
print("V_end mL", 1e6*script.pitcher.state.V, " poured mL", 250 - 1e6*script.pitcher.state.V)
print("white area frac", white.sum()/mask.sum(), " c>=0.6 frac", sat.sum()/mask.sum())
for name, sel in [("white c>0.3", white), ("sat c>=0.6", sat), ("brown c<0.1", mask & (c < 0.1))]:
    print(f"{name:12s} l: med {np.median(l[sel]):.2f} max {l[sel].max():.2f} | m: med {np.median(m[sel]):.2f} max {m[sel].max():.2f} | c: med {np.median(c[sel]):.2f} max {c[sel].max():.2f}")
print("l range in cup", l[mask].min(), l[mask].max(), " total excess layer (sum(l-1)*area)", ((l-1)[mask].sum()*sol.g.area))
# physical reading with '1' = D_L/kQ
h1 = D_L/kQ
print(f"physical reading: blob l_max {l[sat].max():.2f} -> {1e3*h1*l[sat].max():.0f} mm; m_max {m[sat].max():.2f} -> {1e3*h1*m[sat].max():.0f} mm of milk")
# fraction of milk that must leave a saturated cell at fixed l to reach c<0.3
cs = c[sat]; print("fraction of m to remove at fixed l for c<0.3: median", np.median(1-0.3/cs), "max", (1-0.3/cs).max())
