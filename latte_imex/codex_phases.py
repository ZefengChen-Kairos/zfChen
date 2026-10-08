"""Per-phase inlet numbers of the pitcher chain (round-2 #7 tulip action) vs Codex's effective inlet.

    python -m latte_imex.codex_phases

Prints, per barista phase, the arriving Q, U_perp, d, u_h of the physical chain and the closure constants
(B_dep, kQ, kappa_r, kappa_t) that would reproduce Codex's dimensionless operating point (chi, kappa_Q Q,
kappa_r V_R, kappa_t |v_par|) of the corresponding V0.5 phase.  See PITCHER_MODEL.md section 13."""
import json, math, numpy as np
from .joint_sweep import tulip_moves, heart_moves_a, D_L
from .pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript
r = json.load(open('results/joint_sweep/round2_results.json'))[7]
geom = GridPitcherGeometry.load(nr=6)
# Codex per-phase invariants (from selected tulip/heart): U^2/(B d), kappa_r*V_R, kappa_t*|v_par|, F=kQ*Q
codex = {"base ramp": dict(g=0.176, chi=0.97, F=0.020, vr=0.24, vt=1.0),
         "base hold": dict(g=0.83, chi=0.59, F=0.045, vr=0.47, vt=1.0),
         "petal 2": dict(g=0.83, chi=0.59, F=0.045, vr=0.47, vt=1.0),
         "petal 3": dict(g=0.002, chi=1.0, F=0.028, vr=0.07, vt=0.17),
         "cut": dict(g=5.5, chi=0.03, F=0.018, vr=0.39, vt=1.26)}
for name, mk in (("tulip", tulip_moves), ("heart", heart_moves_a)):
    a = r["actions"][name]
    cp = Coupling(c_S=1.0 / D_L ** 3, c_U=1.0 / D_L, c_u=1.0 / D_L, footprint="physical")
    sc = BaristaScript(moves=mk(a), pitcher=Pitcher(geom=geom, V0=250e-6), coupling=cp)
    t = 0.0; rows = {}
    while t < sc.T:
        inl, ph = sc.sample(t); t += 1 / 120
        if inl.active:
            rows.setdefault(ph, []).append((inl.S_eff, inl.U_perp, inl.d_jet, math.hypot(*inl.u_in), inl.r1, inl.r2))
    print("==", name)
    Bs = []
    for ph, v in rows.items():
        v = np.array(v); S, U, d, vt, r1, r2 = v.mean(0)
        Smin, Smax = np.percentile(v[:, 0], [5, 95])
        g = U * U / d
        c = codex.get(ph)
        line = f" {ph:10s} Q {S*D_L**3*1e6:5.1f} mL/s (p5-95 {Smin*D_L**3*1e6:4.1f}-{Smax*D_L**3*1e6:4.1f}) U {U:5.2f} D/s d {d*80:4.2f} mm r {r1*80:.2f}x{r2*80:.2f} mm u_h {vt:4.2f} D/s  U^2/d {g:7.0f}"
        if c:
            B = g / c["g"]; kQ = c["F"] / S; kr = c["vr"] / (U * c["chi"] ** 0.25); kt = c["vt"] / max(vt, 1e-9)
            line += f" | Codex-match: B_dep {B:6.0f} kQ {kQ:4.2f} kappa_r {kr:5.3f} kappa_t {kt:4.2f}"
            if ph != "petal 3": Bs.append(B)
        print(line)
    if Bs: print(" geometric-mean B_dep over base/petal/cut:", round(float(np.exp(np.mean(np.log(Bs))))))
