"""Skeptic check of claim A3: is there a force in the old ("chi") closure that pushes an existing white block
0.1-0.3 D away from the impact as a whole?  Synthetic set-up, #9 physics, N = 96, read-only on the repo.

Old block: disc radius 0.165 D (0.33 D wide) centred at (0, +0.10), l = L0, c = 1.  Back edge at y = -0.065.
Jet at B = (0, -0.125) (0.06 D behind the back edge), pointing +y, low-pour inlet numbers measured with the NX
pitcher (closure_numbers.py), stationary for T_on s, then off until T_end.
"""
import sys, math, json
import numpy as np
sys.path.insert(0, "/home/user/zfChen")
from latte_imex.solver import Solver, Params, Numerics, Inlet

D_L = 0.08
N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
L0 = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
T_on, T_end = 1.0, 1.5
LOW = dict(Q=23.3e-6, U=9.27, d=0.0783, r1=3.86e-3 / D_L, r2=2.54e-3 / D_L, uin=1.88)

phys9 = json.load(open("/home/user/zfChen/runs/joint_sweep_r2fix/results.json"))[9]["physics"]
kQ = phys9.pop("kQ")
OLD = dict(phys9, D=1e-7, kappa_Q=125.0, return_law="v05")

B = (0.0, -0.125)
C0 = (0.0, 0.10); R0 = 0.165


def inlet(active=True):
    c = LOW
    return Inlet(active=active, x_hit=B, S_eff=kQ * c["Q"] / D_L ** 3, u_in=(0.0, c["uin"]), U_perp=c["U"],
                 d_jet=c["d"], r1=c["r1"], r2=c["r2"], phi=0.0, scan_speed=0.0)


def metrics(sol, tag):
    g = sol.g
    c = np.where(g.mask, sol.m / np.where(g.mask, sol.l, 1.0), 0.0)
    white = c > 0.5
    # old-block proxy: white cells with y > B_y + 0.03 (excludes the fresh footprint region only when it is still at B)
    col = np.argmin(np.abs(g.x[0, :]))              # column nearest x = 0
    cy = g.y[:, col]; cc = c[:, col]
    ys_white = cy[cc > 0.5]
    # brown band on the axis between B and the old block: cells with c < 0.3 and y > B_y
    sel = (cy > B[1]) & (cy < 0.4)
    band = cy[sel][(cc[sel] < 0.3)]
    # contiguous brown run just ahead of the white nearest to B
    yb_lo = yb_hi = float("nan")
    if band.size:
        # first brown run that lies between two white regions (new white at B and old block) or ahead of B
        runs = np.split(band, np.where(np.diff(band) > 1.5 * g.h)[0] + 1)
        for r in runs:
            if r.min() > B[1] and r.max() < 0.35 and (cc[(cy > r.max() + g.h) & (cy < 0.35)] > 0.5).any():
                yb_lo, yb_hi = float(r.min()), float(r.max()); break
    # old block geometry (meaningful when no new white): centroid, y-extent, x-span
    far = (g.y > B[1] + 0.06) & white
    if far.any():
        A = far.sum() * g.area
        xc = float((g.x * far).sum() / far.sum()); yc = float((g.y * far).sum() / far.sum())
        yext = float(g.y[far].max() - g.y[far].min())
        xspan = float(g.x[far].max() - g.x[far].min())
    else:
        A = xc = yc = yext = xspan = float("nan")
    umax = float(np.max(np.hypot(sol.qx, sol.qy) / np.where(g.mask, sol.l, 1.0)))
    # mean velocity of the old block (white cells beyond 0.06 D of B), along +y
    if far.any():
        uy_blk = float((sol.qy[far] / sol.l[far]).mean())
    else:
        uy_blk = float("nan")
    print(f"  {tag:>5s} t={sol.t:5.2f}  white y-range on axis [{ys_white.min() if ys_white.size else float('nan'):+.3f},"
          f"{ys_white.max() if ys_white.size else float('nan'):+.3f}]  far-edge {ys_white.max() if ys_white.size else float('nan'):+.3f}"
          f"  brown band y[{yb_lo:+.3f},{yb_hi:+.3f}] w={(yb_hi - yb_lo + g.h) * 80 if yb_lo == yb_lo else float('nan'):4.1f} mm"
          f"  block: yc={yc:+.3f} (d={yc - C0[1]:+.3f}) y-ext {yext:.3f} x-span {xspan:.3f} area {A:.4f}"
          f"  uy_blk {uy_blk:+.3f} umax {umax:.2f} lmax {sol.l[g.mask].max():.2f}")
    return dict(t=sol.t, yc=yc, yext=yext, xspan=xspan, A=A, far=ys_white.max() if ys_white.size else None,
                band=(yb_lo, yb_hi))


def run(label, over, T_on=T_on, T_end=T_end, N=N):
    pd = dict(OLD); pd.update(over)
    sol = Solver(Params(**pd), Numerics(N=N))
    g = sol.g
    blk = (np.hypot(g.x - C0[0], g.y - C0[1]) <= R0) & g.mask
    sol.l = np.where(blk, L0, 1.0); sol.m = np.where(blk, L0, 0.0)
    print(f"== {label}: {over}")
    m0 = metrics(sol, "init")
    fd = sol.num.frame_dt
    out = []
    checkpoints = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    k = 0
    while sol.t < T_end - 1e-9:
        sol.advance_frame(inlet(active=sol.t < T_on - 1e-9), fd)
        if k < len(checkpoints) and sol.t >= checkpoints[k] - 1e-9:
            out.append(metrics(sol, "")); k += 1
    led = sol.ledger()
    print(f"  ledger: deposited {led['deposited']:.4f} reservoir {led['reservoir']:.4f} layer {led['layer']:.3f}  retries {sol.stats['retries']} steps {sol.stats['steps']} frames {sol.stats['frames']} max_sub {sol.stats['max_substeps']} cg_mass {sol.stats['cg_mass']} cg_visc {sol.stats['cg_visc']} cg_mix {sol.stats['cg_mix']}")
    return m0, out


if __name__ == "__main__":
    which = sys.argv[3] if len(sys.argv) > 3 else "all"
    cases = {
        "C  control, no jet": dict(_T_on=0.0),
        "V0 baseline #9 (pile pressure + s v* + Lam)": dict(tau_d=0.0),
        "V1 momentum only (tau_d=inf, w=0, alpha=0)": dict(tau_d=1e9, sub_pressure=0.0, sub_advect=0.0),
        "V2 momentum + mound pressure w=0.5 (tau_d=inf)": dict(tau_d=1e9, sub_pressure=0.5, sub_advect=0.0),
        "V3 momentum + mound pressure w=1.0 (tau_d=inf)": dict(tau_d=1e9, sub_pressure=1.0, sub_advect=0.0),
        "V4 baseline, kappa_r=0.5": dict(tau_d=0.0, kappa_r=0.5),
        "V5 baseline, kappa_t=0 (isotropic pile pressure only)": dict(tau_d=0.0, kappa_t=0.0),
        "V6 baseline, kappa_t=0, kappa_c=0 (pure pile)": dict(tau_d=0.0, kappa_t=0.0, kappa_c=0.0),
        "V7 patch defaults tau_d=0.2 w=0.5 alpha=0.3": dict(tau_d=0.2, sub_pressure=0.5, sub_advect=0.3),
        "V8 s v* only (tau_d=inf, w=0, kappa_c=0)": dict(tau_d=1e9, sub_pressure=0.0, sub_advect=0.0, kappa_c=0.0),
        "V9 Lam only (tau_d=inf, w=0, kappa_c=0.665, S-momentum off via kappa_t=0? no) -- skipped": None,
        "B0 baseline, 0.3 s burst then off": dict(tau_d=0.0, _T_on=0.3),
        "B1 momentum only, 0.3 s burst then off": dict(tau_d=1e9, sub_pressure=0.0, sub_advect=0.0, _T_on=0.3),
        "B5 kappa_t=0 (pile only), 0.3 s burst": dict(tau_d=0.0, kappa_t=0.0, _T_on=0.3),
    }
    for lab, over in cases.items():
        if over is None:
            continue
        if which != "all" and lab.split()[0] not in which.split(","):
            continue
        over = dict(over); ton = over.pop("_T_on", T_on)
        try:
            run(lab, over, T_on=ton)
        except Exception as e:
            print(f"  !! {lab}: {type(e).__name__}: {e}")
