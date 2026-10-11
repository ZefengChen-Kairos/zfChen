"""B1 ablation, cleaner metrics. Old white disc (l = L0, m = L0-1) centred (0, 0.10), radius 0.165 (back edge y = -0.065);
jet (low pour, v2F) at B = back edge - gap, pointing +y; phi_foam = 0 so no new white can appear (isolates the flow).
Track: lowest y of the contiguous W>0.5 run through the block centre on the axis column; film v_y = qy/l at probes
3.3 / 5 / 8 mm ahead of the jet (toward the block); with and without the entrainment cell (ent -> 0)."""
import sys, time, json
import numpy as np
sys.path.insert(0, "/home/user/zfChen")
sys.path.insert(0, "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review")
from skeptic_B1_check import V2F, N, inlet, Solver, Params, Numerics

T = 0.3; C0 = (0.0, 0.10); R0 = 0.165


def run(pd, gap, L0, label):
    hit = (0.0, C0[1] - R0 - gap)
    sol = Solver(Params(**pd), Numerics(N=N)); g = sol.g; m = g.mask
    disc = (np.hypot(g.x - C0[0], g.y - C0[1]) < R0) & m
    sol.l = np.where(disc, L0, 1.0); sol.m = np.where(disc, L0 - 1.0, 0.0)
    col = np.argmin(np.abs(g.x[0, :])); cy = g.y[:, col]; jc = np.argmin(np.abs(cy - C0[1]))
    probes = [np.argmin(np.abs(cy - (hit[1] + d / 80))) for d in (3.3, 5.0, 8.0)]

    def edge():
        W = sol.concentration()[:, col] > 0.5
        j = jc
        while j > 0 and W[j - 1]:
            j -= 1
        return cy[j]

    e0 = edge(); inl = inlet("low", hit=hit); rec = []
    while sol.t < T - 1e-9:
        sol.advance_frame(inl)
        if sol.stats["frames"] in (6, 12, 24, 36):
            v = sol.qy[:, col] / sol.l[:, col]
            rec.append((sol.t, edge(), [v[p] for p in probes], [sol.l[p, col] for p in probes]))
    print(f"== {label}: gap {gap*80:.1f} mm, L0 {L0}, edge0 y={e0:+.4f} ==")
    for t, e, vv, ll in rec:
        print(f"   t={t:.2f} edge moved {(e-e0)*80:+.2f} mm (+ = away from jet); v_y at 3.3/5/8 mm ahead of jet: "
              + " ".join(f"{x:+.3f}" for x in vv) + " D/s; l there: " + " ".join(f"{x:.2f}" for x in ll))


for gap in [float(a) for a in sys.argv[1].split(",")]:
    for L0 in [float(a) for a in sys.argv[2].split(",")]:
        for lab, over in [("full", {}), ("no cell", dict(ent_coef=1e-9))]:
            phi = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
            run(dict(V2F, phi_foam=phi, **over), gap, L0, lab + f" phi={phi}")
