"""Skeptic check of claim B1 (v2 surface converges on the jet and sucks the old block inward). Read-only.
Part A: the v2F surface flow U_s from _bulk_flow at the measured low pour / high cut (closure_numbers.py cases).
Part B: short ablation runs (N=96, v2F): old white disc, jet lands behind it; with / without the entrainment cell.
"""
import sys, math, json, time
import numpy as np
sys.path.insert(0, "/home/user/zfChen")
from latte_imex.solver import Solver, Params, Numerics, Inlet

D_L = 0.08; N = 96; h = 1.0 / N
CASES = {
    "low": dict(Q=23.3e-6, U=9.27, d=0.0783, r1=3.86e-3 / D_L, r2=2.54e-3 / D_L, uin=1.88),
    "cut": dict(Q=3.3e-6, U=15.66, d=0.0235, r1=0.94e-3 / D_L, r2=0.93e-3 / D_L, uin=1.61),
}
V2F = json.load(open("/home/user/zfChen/results/joint_sweep/codexmap/v2_final_fit_N96.json"))["phys"]
B = (0.0, -0.125)


def inlet(case, hit=B, active=True, dirn=+1.0):
    c = CASES[case]
    return Inlet(active=active, x_hit=hit, S_eff=c["Q"] / D_L ** 3, u_in=(0.0, dirn * c["uin"]), U_perp=c["U"],
                 d_jet=c["d"], r1=c["r1"], r2=c["r2"], phi=0.0, scan_speed=0.0)


def part_A(pd, label):
    print(f"\n===== A: {label} =====")
    sol = Solver(Params(**pd), Numerics(N=N)); P = sol.P; g = sol.g; m = g.mask
    for case in CASES:
        inl = inlet(case, hit=(0.0, 0.0))
        s, Lam, vsx, vsy, chi = sol.inlet_fields(inl); sol._bulk_flow()
        K, _ = sol.kernel(inl)
        Fr = math.sqrt(inl.U_perp ** 2 / (P.g_red * inl.d_jet))
        r = np.hypot(g.x, g.y); ux, uy = sol.Ub
        ur = np.where(r > 0, (ux * g.x + uy * g.y) / np.maximum(r, 1e-12), 0.0)
        dv = sol.divUs
        r1f, r2f = max(inl.r1, 1.5 * h), max(inl.r2, 1.5 * h)
        print(f"-- {case}: Fr={Fr:.2f}, 2*ent*Fr={2*P.ent_coef*Fr:.2f}, S_v={sol._S:.4f}/s, S_E={sol._SE:.4f}/s (S_E/S_v={sol._SE/sol._S:.2f}), "
              f"R_c={sol._Rc*80:.2f} mm, footprint semi-axes {r1f*80:.2f}x{r2f*80:.2f} mm, K_peak={K.max():.0f}/D^2, chi={chi:.2f}")
        ic = np.unravel_index(np.argmax(K), K.shape)
        print(f"   div U_s at K peak = {dv[ic]:.0f}/s ; min {dv.min():.0f}, max {dv.max():.0f}/s ;  far field (r>0.2): mean {dv[m&(r>0.2)].mean():.3f}/s")
        # radial profile along +x axis (row through hit point)
        row = ic[0]
        xs = g.x[row]; prof = ur[row]; sel = m[row] & (xs > 0)
        print("   radial U_s(r) along +x [mm -> D/s]  (negative = toward the jet):")
        out = []
        for rr in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30):
            i = np.argmin(np.abs(xs * 80 - rr))
            out.append(f"{rr}mm:{prof[i]:+.3f}")
        print("   " + "  ".join(out))
        # radius where ur changes sign (inward -> outward) along +x
        xx = xs[sel]; pp = prof[sel]
        neg = xx[pp < 0]
        print(f"   inward region along +x: r <= {neg.max()*80 if neg.size else 0:.2f} mm ; |U_s| max {np.hypot(ux,uy).max():.2f} D/s at r={r[np.unravel_index(np.argmax(np.hypot(ux,uy)), r.shape)]*80:.2f} mm")
        # v2 step-B target velocity in the footprint (steady-state proxy: s = foam_src surfacing rate)
        s_f = sol.foam_src; Lmj = sol._Lam_jet; jx, jy = sol._vjet
        s_skin = P.l_skin * np.maximum(dv, 0)
        s_l = s_f + s_skin
        vsx_t = (P.beta * ux + Lmj * jx + s_l * ux) / (P.beta + Lmj + s_l)
        vsy_t = (P.beta * uy + Lmj * jy + s_l * uy) / (P.beta + Lmj + s_l)
        vr_t = (vsx_t * g.x + vsy_t * g.y) / np.maximum(r, 1e-12)
        inK = K > 0.05 * K.max()
        print(f"   step-B target v* inside K: |v*| median {np.median(np.hypot(vsx_t,vsy_t)[inK]):.2f} D/s, along-jet comp median {np.median(vsy_t[inK]):+.2f} (v_jet={jy:+.2f}), "
              f"radial comp median {np.median(vr_t[inK]):+.3f}; weights at K peak: beta {P.beta:.1f}, Lam {Lmj[ic]:.0f}, s_l {s_l[ic]:.0f}")
        ring = (r > max(r1f, r2f)) & (r < max(sol._Rc, 2 * h)) & m
        print(f"   ring cells {int(ring.sum())}: div median {np.median(dv[ring]) if ring.any() else float('nan'):+.1f}/s, ur median {np.median(ur[ring]) if ring.any() else float('nan'):+.3f} D/s")
        # how big is the convergent region relative to the physics-prior near field 2d
        print(f"   2 d_jet = {2*inl.d_jet*80:.1f} mm (physics_prior near-field outward zone); R_c/d = {sol._Rc/inl.d_jet:.2f}")


def run_block(pd, label, T=0.5, L0=3.0, C0=(0.0, 0.10), R0=0.165, hit=B, case="low"):
    sol = Solver(Params(**pd), Numerics(N=N)); g = sol.g; m = g.mask
    disc = (np.hypot(g.x - C0[0], g.y - C0[1]) < R0) & m
    sol.l = np.where(disc, L0, 1.0); sol.m = np.where(disc, L0 - 1.0, 0.0)   # white block: foam L0-1 on crema 1
    col = np.argmin(np.abs(g.x[0, :])); cy = g.y[:, col]
    edge0 = cy[(sol.m[:, col] > 0.5) & (cy > hit[1])].min()

    def near_edge():
        W = sol.concentration()[:, col]; sel = (cy > hit[1] + 0.03) & (W > 0.5)   # exclude the fresh footprint region
        return cy[sel].min() if sel.any() else float("nan")

    def far_centroid():
        W = sol.concentration(); far = m & (g.y > hit[1] + 0.06) & (W > 0.5)
        return (float((g.y * far).sum() / max(far.sum(), 1)), float(far.sum() * g.area))

    t0 = time.time(); rec = []
    inl = inlet(case, hit=hit)
    while sol.t < T - 1e-9:
        sol.advance_frame(inl)
        if any(abs(sol.t - tt) < 1e-6 for tt in (0.05, 0.1, 0.2, 0.3, 0.5)):
            v_edge = float(sol.qy[np.argmin(np.abs(cy - near_edge())), col] / sol.l[np.argmin(np.abs(cy - near_edge())), col])
            yc, A = far_centroid()
            rec.append((sol.t, near_edge(), v_edge, yc, A))
    print(f"\n== B: {label} (block edge0 y={edge0:+.4f}, hit y={hit[1]:+.3f}, gap0={(edge0-hit[1])*80:.1f} mm, {time.time()-t0:.0f}s) ==")
    for t, e, v, yc, A in rec:
        print(f"   t={t:.2f}s near-edge y={e:+.4f} (moved {(e-edge0)*80:+.2f} mm, + = away from jet), v_y at edge {v:+.3f} D/s, far-block centroid y={yc:+.4f}, white area {A:.4f}")


if __name__ == "__main__":
    part_A(V2F, "v2F")
    if len(sys.argv) > 1 and sys.argv[1] == "B":
        gap = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06     # jet distance behind the block's back edge [D]
        hit = (0.0, 0.10 - 0.165 - gap)
        for lab, over in [("v2F full", {}), ("v2F no cell (ent=0)", dict(ent_coef=1e-9)),
                          ("v2F full, no foam (phi=0)", dict(phi_foam=0.0)), ("v2F no cell, no foam", dict(ent_coef=1e-9, phi_foam=0.0))]:
            run_block(dict(V2F, **over), lab + f" gap {gap:.3f} D", hit=hit)
