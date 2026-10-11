"""Skeptic B6: submerged foam in twolayer/v2 -- where/when does it surface, first-white time, any pressure/volume role.
Read-only. N=96, stationary low-pour jet at the origin, v2F and cell_fit physics."""
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
phys9 = json.load(open("/home/user/zfChen/runs/joint_sweep_r2fix/results.json"))[9]["physics"]; kQ = phys9.pop("kQ")
OLD = dict(phys9, D=1e-7, kappa_Q=125.0, return_law="v05")
CELL = dict(OLD, **json.load(open("/home/user/zfChen/runs/action_opt/cell_fit_N96.json"))["best"]["over"])

def inlet(case, kq):
    c = CASES[case]
    return Inlet(active=True, x_hit=(0.0, 0.0), S_eff=kq * c["Q"] / D_L ** 3, u_in=(0.0, c["uin"]), U_perp=c["U"],
                 d_jet=c["d"], r1=c["r1"], r2=c["r2"], phi=0.0, scan_speed=0.0)

def run(pd, kq, label, case="low", T=0.6, wthr=0.5, ablate_ud=False):
    sol = Solver(Params(**pd), Numerics(N=N)); P = sol.P; g = sol.g; m = g.mask
    inl = inlet(case, kq)
    K, _ = sol.kernel(inl)
    r = np.hypot(g.x, g.y)
    foot = K > 0
    t0 = time.time()
    first = {}; rows = []
    Rc = None
    cum_ring = 0.0
    if ablate_ud:
        orig = sol._bulk_flow
        def patched():
            orig()
            if sol.Ud is not None:
                sol.Ud = (np.zeros_like(sol.Ud[0]), np.zeros_like(sol.Ud[1]))
        sol._bulk_flow = patched
    for k in range(int(round(T * 120))):
        # intercept the surfacing field: replicate substep bookkeeping by diffing deposited
        dep0 = sol.deposited
        f0 = sol.f.copy()
        chi, nsub = sol.advance_frame(inl)
        if Rc is None:
            Rc = max(sol._Rc, 2 * h)
            ring = (r >= 0) & ~foot & (r < Rc) & m
            out = (r >= Rc) & m
        W = sol.concentration()
        Wm = np.nanmax(W)
        for thr in (0.1, 0.3, 0.5, 0.8):
            if thr not in first and Wm > thr:
                first[thr] = sol.t
        # surfacing this frame = deposited increment; spatial split via m increase is confounded by transport, so
        # estimate the surfacing *location* from the submerged field f (which surfaces locally at rate f/tau_f)
        f = sol.f; ft = sol.ft
        tau = np.where(f > 1e-14, ft / np.maximum(f, 1e-300), np.nan)
        sf = np.where(f > 1e-14, f / np.maximum(tau, 1e-3), 0.0)   # local surfacing rate [H_f/s]
        tot = sf[m].sum() + 1e-300
        cum_ring += sf[ring].sum() * g.area / 120.0
        tau_w = (ft[m].sum() / max(f[m].sum(), 1e-300))
        if k % 6 == 5 or k < 3:
            rows.append((sol.t, Wm, sf[foot].sum() / tot, sf[ring].sum() / tot, sf[out].sum() / tot,
                         float(np.nanmin(tau)) if np.isfinite(tau).any() else float('nan'), float(np.nanmax(tau)) if np.isfinite(tau).any() else float('nan'),
                         f[m].sum() * g.area, sol.deposited, (r[m] * sf[m]).sum() / tot * 80,
                         float(np.nanmax(W[ring])) if ring.any() else float('nan'), float(np.nanmedian(W[ring])) if ring.any() else float('nan'),
                         float(np.nanmax(W[foot])), nsub, tau_w, cum_ring, sol.m[ring].sum() * g.area, sol.m[foot].sum() * g.area, (sol.m[m].sum()*g.area)))
    print(f"\n===== {label} / {case}: chi={chi:.2f}, tau_new={sol.tau_new*1e3:.0f} ms, R_c={Rc*80:.2f} mm, footprint cells {int(foot.sum())}, ring cells {int(ring.sum())}, wall {time.time()-t0:.0f}s")
    print(f"  tau_cur={sol.tau_cur}, msub max={sol.msub.max():.2e} (delay/sub_pressure path active? {sol.tau_cur is not None})")
    print(f"  first W>thr: " + ", ".join(f"{thr}: {first.get(thr, float('nan')):.3f}s" for thr in (0.1, 0.3, 0.5, 0.8)))
    print("  t      Wmax  surf-frac foot/ring/out  tau_f min/max/fwmean[ms]  f_sub  deposited  <r>_surf[mm]  W_ring max/med  W_foot max  nsub  cum_surf_ring  m_ring  m_foot  m_tot")
    for rw in rows:
        print(f"  {rw[0]:.3f}  {rw[1]:.2f}   {rw[2]:.2f} / {rw[3]:.2f} / {rw[4]:.2f}     {rw[5]*1e3:.0f}/{rw[6]*1e3:.0f}/{rw[14]*1e3:.0f}     {rw[7]:.4f}  {rw[8]:.4f}    {rw[9]:.2f}        {rw[10]:.2f}/{rw[11]:.2f}        {rw[12]:.2f}   {rw[13]}   {rw[15]:.5f}   {rw[16]:.5f}  {rw[17]:.5f}  {rw[18]:.4f}")
    return sol

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "v2"
    case = sys.argv[2] if len(sys.argv) > 2 else "low"
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    abl = len(sys.argv) > 4 and sys.argv[4] == "ud0"
    if which == "v2":
        run(V2F, 1.0, "v2F" + (" [Ud=0 ablation: foam surfaces where poured]" if abl else ""), case, T, ablate_ud=abl)
    else:
        run(CELL, kQ, "twolayer cell_fit", case, T)
