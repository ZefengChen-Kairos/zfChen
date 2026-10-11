"""B3 check (read-only): does v2's film pressure fill a crema gap between two foam blocks within milliseconds, and is the
crema "squeezed out" (crema leaves the gap) or is foam pushed in?  No pour: two foam blocks on the initial crema layer,
4 mm crema gap between them, relax for 0.3 s.  Variants: v2s (the physics of the E-18 video-script run), v2F, tau_y = 0,
tau_y at physical values, per-substep SSA, no nu_max cap, and the old closure (#9 physics) for comparison."""
import sys, math, json, time
import numpy as np
sys.path.insert(0, "/home/user/zfChen")
from latte_imex.solver import Solver, Params, Numerics, Inlet
from latte_imex.action_opt import physics9

N = 96
V2F = json.load(open("/home/user/zfChen/results/joint_sweep/codexmap/v2_final_fit_N96.json"))["phys"]
V2S = json.load(open("/home/user/zfChen/runs/action_opt/joint_cma_v2s/history.json"))["best"]["best"]["phys"]
from latte_imex.joint_cma import BASE_V2
V2S = dict(BASE_V2, **V2S)
PH9, _ = physics9()
OLD = dict(PH9, D=1e-7, kappa_Q=125.0, return_law="v05")

W_GAP = 0.05        # 4 mm gap (5 cells)
XB, Y1 = 0.12, 0.20 # blocks |x|<0.12, W_GAP/2<|y|<0.20

def setup(pd, mb, num=None):
    sol = Solver(Params(**pd), num or Numerics(N=N)); g = sol.g
    blk = (np.abs(g.x) < XB) & (np.abs(g.y) > W_GAP / 2) & (np.abs(g.y) < Y1) & g.mask
    sol.l = np.where(blk, 1.0 + mb, 1.0); sol.m = np.where(blk, mb, 0.0)
    return sol

def run(label, pd, mb=1.0, T=0.3, num=None, times=(0.0083, 0.025, 0.05, 0.1, 0.2, 0.3)):
    sol = setup(pd, mb, num); g = sol.g
    strip = (np.abs(g.x) < XB) & (np.abs(g.y) <= W_GAP / 2) & g.mask     # the gap region
    centre = (np.abs(g.x) < 0.02) & (np.abs(g.y) <= W_GAP / 2) & g.mask  # middle of the gap
    blkc = (np.abs(g.x) < 0.02) & (np.abs(g.y) > 0.08) & (np.abs(g.y) < 0.12) & g.mask  # block interior
    inl = Inlet(active=False)
    P = sol.P
    crema0 = float((sol.l - sol.m)[strip].sum() * g.area); foam0 = float(sol.m[strip].sum() * g.area)
    print(f"\n=== {label}: closure={P.closure} c_p={P.cp:.3f} beta={P.beta:.2f} nu={P.nu:.2e} tau_y={getattr(P,'tau_y',0):.3f} "
          f"nu_max={getattr(P,'nu_max',0)} ssa_per_frame={sol.num.ssa_per_frame}  blocks l={1+mb}, m={mb}; gap l=1, m=0 ===")
    print(f"   t[ms]   l_gap  m_gap  crema_gap  W_gap | l_blk  m_blk | strip crema/crema0  strip foam [1e-3 Hf D^2] | nsub  wall")
    t = 0.0; t0 = time.time(); k = 0
    while t < T - 1e-9:
        _, nsub = sol.advance_frame(inl); t += 1.0 / 120.0
        if k < len(times) and t >= times[k] - 1e-9:
            k += 1
            W = 1 - math.exp(-float(sol.m[centre].mean()) / P.m_o) if P.closure == "v2" else float((sol.m / sol.l)[centre].mean()) / 0.6
            cr = float((sol.l - sol.m)[strip].sum() * g.area); fo = float(sol.m[strip].sum() * g.area)
            print(f"   {t*1000:6.1f}  {sol.l[centre].mean():.3f}  {sol.m[centre].mean():.3f}   {(sol.l-sol.m)[centre].mean():.3f}    {W:.2f} | "
                  f"{sol.l[blkc].mean():.3f}  {sol.m[blkc].mean():.3f} |   {cr/crema0:.3f}             {fo*1e3:6.2f}                 | {nsub:3d}  {time.time()-t0:5.1f}s")
    return sol

if __name__ == "__main__":
    which = sys.argv[1:] or ["v2s", "v2F", "v2F_ty0", "v2F_ty3", "v2F_ty10", "v2F_ty10_numax50", "v2F_sub", "old"]
    for w in which:
        if w == "v2s": run("v2s as in E-18 run", V2S)
        if w == "v2F": run("v2F", V2F)
        if w == "v2F_ty0": run("v2F tau_y=0", dict(V2F, tau_y=0.0))
        if w == "v2F_ty3": run("v2F tau_y=3 (fit upper bound)", dict(V2F, tau_y=3.0))
        if w == "v2F_ty10": run("v2F tau_y=10 (20 Pa)", dict(V2F, tau_y=10.0))
        if w == "v2F_ty10_numax50": run("v2F tau_y=10 nu_max=50", dict(V2F, tau_y=10.0, nu_max=50.0))
        if w == "v2F_sub": run("v2F SSA per substep", V2F, num=Numerics(N=N, ssa_per_frame=False))
        if w == "v2F_cp1": run("v2F with g_red/6.4 (c_p=0.97)", dict(V2F, g_red=V2F["g_red"] / 6.4))
        if w == "v2F_fdt": run("v2F frame_dt=1/1200 (SSA every 0.83 ms, no nsub=1 first frame)", V2F, num=Numerics(N=N, frame_dt=1/1200))
        if w == "v2F_fdt_ty0": run("v2F frame_dt=1/1200 tau_y=0", dict(V2F, tau_y=0.0), num=Numerics(N=N, frame_dt=1/1200))
        if w == "v2F_fdt_ty10_numax50": run("v2F frame_dt=1/1200 tau_y=10 nu_max=50", dict(V2F, tau_y=10.0, nu_max=50.0), num=Numerics(N=N, frame_dt=1/1200))
        if w == "v2F_fdt_ty3_numax50": run("v2F frame_dt=1/1200 tau_y=3 nu_max=50", dict(V2F, tau_y=3.0, nu_max=50.0), num=Numerics(N=N, frame_dt=1/1200))
        if w == "v2s_fdt": run("v2s frame_dt=1/1200", V2S, num=Numerics(N=N, frame_dt=1/1200))
        if w == "old": run("old closure #9", OLD)
