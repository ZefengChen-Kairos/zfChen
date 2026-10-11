"""Short old-closure (#9) push_heart run at N=96: |u| at c-interfaces, local Courant numbers, interface widths."""
import sys, time, json, numpy as np
sys.path.insert(0, "/home/user/zfChen")
import os; os.chdir("/home/user/zfChen")
from latte_imex import action_opt as ao
from latte_imex.pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript
from latte_imex.solver import Solver, Params, Numerics
N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
WALL = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
geom = GridPitcherGeometry.load(nr=6)
ph, kQ = ao.physics9()
cp = Coupling(c_S=kQ / ao.D_L ** 3, c_U=1.0 / ao.D_L, c_u=1.0 / ao.D_L, footprint="physical")
script = BaristaScript(moves=ao.scaled_moves("push_heart", 1, 1, 1), pitcher=Pitcher(geom=geom, V0=250e-6), coupling=cp)
pd = dict(ph, D=1e-7, kappa_Q=125.0, return_law="v05")
num = Numerics(N=N); sol = Solver(Params(**pd), num)
print("T_script =", script.T, " nu =", pd["nu"], " h(mm) =", 80 / N)
t0 = time.time(); nxt = 0.5; h = sol.g.h
while sol.t < script.T - 1e-9 and time.time() - t0 < WALL:
    probe, _ = script.sample(sol.t); fdt = num.frame_dt
    if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
        fdt = min(fdt, max(num.scan_safety * min(h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
    inl, _ = script.sample(sol.t + 0.5 * fdt); chi, nsub = sol.advance_frame(inl, frame_dt=fdt)
    if sol.t >= nxt:
        nxt += 0.5
        m = sol.g.mask; l = np.where(m, sol.l, 1.0)
        u = np.hypot(sol.qx, sol.qy) / l; c = np.where(m, sol.m / l, 0.0)
        gx, gy = np.gradient(c, h); gc = np.hypot(gx, gy)
        iface = m & (gc * h > 0.05)              # cells where c changes by >0.05 per cell
        trans = m & (c > 0.1) & (c < 0.9)
        white = m & (c > 0.5)
        umax = u[m].max(); dt = fdt / nsub
        ui = u[iface] if iface.any() else np.array([0.0])
        q = np.percentile(ui, [10, 50, 90])
        print(f"t={sol.t:5.2f}s wall={time.time()-t0:5.1f}s nsub={nsub:2d} dt={dt*1e3:.2f}ms umax={umax:.2f} D/s | "
              f"|u| at interfaces p10/50/90 = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f} D/s -> nu_num=|u|h = {q[0]*h:.1e}/{q[1]*h:.1e}/{q[2]*h:.1e} "
              f"(nu/{pd['nu']:.1e}: {q[0]*h/pd['nu']:.1f}/{q[1]*h/pd['nu']:.1f}/{q[2]*h/pd['nu']:.1f}x) | local Courant p50/p90 = {np.median(ui)*dt/h:.3f}/{q[2]*dt/h:.3f} | "
              f"white cells {int(white.sum())}, transitional(0.1<c<0.9) {int(trans.sum())}, interface cells {int(iface.sum())}; trans/iface = {trans.sum()/max(iface.sum(),1):.2f}")
print("stats", {k: v for k, v in sol.stats.items() if k in ("max_u", "max_substeps", "retries", "frames", "steps")})
np.save(f"/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review/skeptic_A14_c_N{N}.npy", np.where(sol.g.mask, sol.m / np.where(sol.g.mask, sol.l, 1), 0).astype(np.float32))
