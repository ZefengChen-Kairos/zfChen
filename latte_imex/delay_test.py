"""V0.5-translated action through the physical chain with the #9 closure plus overrides (delay closure test)."""
import json, sys, os, time, numpy as np

from latte_imex.pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript, v05_moves, v05_moves_low
from latte_imex.solver import Solver, Params, Numerics
name, N, out, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
def _val(v):
    try:
        return float(v)
    except ValueError:
        return v
over = {k: _val(v) for k, v in (a.split("=") for a in sys.argv[5:])}
D_L = 0.08
r = json.load(open("runs/joint_sweep_r2fix/results.json"))[9]; ph = dict(r["physics"]); kQ = ph.pop("kQ")
geom = GridPitcherGeometry.load(nr=6)
cp = Coupling(c_S=kQ / D_L ** 2 / D_L, c_U=1.0 / D_L, c_u=1.0 / D_L, footprint="physical")
moves = v05_moves_low(name) if os.environ.get("LOW") else v05_moves(name)
script = BaristaScript(moves=moves, pitcher=Pitcher(geom=geom, V0=250e-6), coupling=cp)
pd = dict(ph, D=1e-7, kappa_Q=125.0, return_law="v05"); pd.update(over)
num = Numerics(N=N); sol = Solver(Params(**pd), num); t0 = time.time()
while sol.t < script.T - 1e-9:
    probe, _ = script.sample(sol.t); fdt = num.frame_dt
    if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
        fdt = min(fdt, max(num.scan_safety * min(sol.g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
    inl, _ = script.sample(sol.t + 0.5 * fdt); sol.advance_frame(inl, frame_dt=fdt)
c = sol.concentration(); os.makedirs(out, exist_ok=True)
np.save(f"{out}/{name}_{tag}_N{N}.npy", c.astype(np.float32))
L = sol.ledger()
json.dump(dict(name=name, tag=tag, N=N, over=over, ledger=L, elapsed=time.time() - t0), open(f"{out}/{name}_{tag}_N{N}.json", "w"))
print(name, tag, N, f"{time.time()-t0:.0f}s reservoir left {L['reservoir']:.4f} deposited {L['deposited']:.3f} retries {sol.stats['retries']}", flush=True)
