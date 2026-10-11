"""Does S_eff actually reach 0 (-> Ub None) during the push_heart repositions, or only at the lifts?"""
import sys; sys.path.insert(0, "/home/user/zfChen")
from latte_imex import stage_actions as sa, action_opt as ao
from latte_imex.pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript
geom = GridPitcherGeometry.load(nr=6)
p = dict(sa.SPEC["push_heart"]["x0"]); n = 3
cp = Coupling(c_S=1.0 / ao.D_L ** 3, c_U=1.0 / ao.D_L, c_u=1.0 / ao.D_L, footprint="physical")
script = BaristaScript(moves=sa.moves_of(p), pitcher=Pitcher(geom=geom, V0=250e-6), coupling=cp)
for m in sa.moves_of(p):
    print(f"  move {m.name:12s} dur {m.dur:.2f} Q {m.Q0*1e6:.1f}->{m.Q1*1e6:.1f} mL/s z {m.z0:.3f}")
t = 0.0; dt = 1/120; prev = None; zero_spans = []; last = None
rows = []
while t < script.T - 1e-9:
    inl, _ = script.sample(t)
    S = inl.S_eff if inl.active else 0.0
    rows.append((t, S))
    on = S > 0
    if prev is not None and on != prev:
        print(f"  t={t:.3f}  S_eff {'>0' if on else '=0'}")
    prev = on; t += dt
print("total T", script.T, "frames with S_eff=0:", sum(1 for _, S in rows if S <= 0), "of", len(rows))
