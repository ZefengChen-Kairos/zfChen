"""Evaluate the inlet closures of latte_imex/solver.py at the fitted constants (read-only)."""
import sys, math, json
import numpy as np
sys.path.insert(0, "/home/user/zfChen")
from latte_imex.solver import Solver, Params, Numerics, Inlet

D_L = 0.08
N = 96
h = 1.0 / N

# measured with the NX pitcher on push_heart (see pitcher run): Q [mL/s], U_perp [D/s], d_jet [D], r1, r2 [D], |u_in| [D/s]
CASES = {
    "low pour 23 mL/s @2cm": dict(Q=23.3e-6, U=9.27, d=0.0783, r1=3.86e-3 / D_L, r2=2.54e-3 / D_L, uin=1.88),
    "high cut 3.3 mL/s @7-9cm": dict(Q=3.3e-6, U=15.66, d=0.0235, r1=0.94e-3 / D_L, r2=0.93e-3 / D_L, uin=1.61),
}

phys9 = json.load(open("/home/user/zfChen/runs/joint_sweep_r2fix/results.json"))[9]["physics"]
kQ = phys9.pop("kQ")
OLD = dict(phys9, D=1e-7, kappa_Q=125.0, return_law="v05")
CELL = json.load(open("/home/user/zfChen/runs/action_opt/cell_fit_N96.json"))["best"]["over"]
V2F = json.load(open("/home/user/zfChen/results/joint_sweep/codexmap/v2_final_fit_N96.json"))["phys"]
V2S = json.load(open("/home/user/zfChen/runs/action_opt/joint_cma_v2s/history.json"))["best"]["best"]["phys"]
BASE_V2 = dict(closure="v2", l_skin=0.03, yield_eps=0.02, nu_max=0.5, tau_y_crema=0.0, D=1e-7, D_L=0.08)
V2S = dict(BASE_V2, **V2S)

def inlet_for(case, kq):
    c = CASES[case]
    S = kq * c["Q"] / D_L ** 3
    return Inlet(active=True, x_hit=(0.0, 0.1), S_eff=S, u_in=(0.0, -c["uin"]), U_perp=c["U"], d_jet=c["d"],
                 r1=c["r1"], r2=c["r2"], phi=0.0, scan_speed=0.1)

def report(label, pd, kq):
    print(f"\n=== {label} ===")
    sol = Solver(Params(**pd), Numerics(N=N))
    P = sol.P
    print(f"  cp used by solver = {P.cp:.3f} D/s ; beta {P.beta:.2f} ; nu {P.nu:.2e}")
    for case in CASES:
        c = CASES[case]
        inl = inlet_for(case, kq)
        s, Lam, vsx, vsy, chi = sol.inlet_fields(inl)
        g = sol.g; m = g.mask
        K, _ = sol.kernel(inl)
        r1f, r2f = max(inl.r1, 1.5 * h), max(inl.r2, 1.5 * h)
        print(f"  -- {case}: S_eff={inl.S_eff:.4f}/s  kernel radii floored {inl.r1*80:.2f}->{r1f*80:.2f} mm, {inl.r2*80:.2f}->{r2f*80:.2f} mm, K_peak={K.max():.0f}")
        print(f"     chi={chi:.3f}")
        if P.closure == "chi":
            zc = inl.U_perp ** 2 / (P.B_dep * inl.d_jet)
            VR = min(inl.U_perp * chi ** (1 / (2 * P.p_dep)), inl.U_perp, math.sqrt(P.B_dep * inl.d_jet))
            print(f"     zc=U^2/(B d)={zc:.3f}; s_peak={s.max():.1f} /s (layer units/s), Lam_peak={Lam.max():.1f} /s, "
                  f"VR={VR:.2f} -> radial target {P.kappa_r*VR:.3f} D/s, tangential target {P.kappa_t*c['uin']:.2f} D/s")
            # pile-up rate -> pressure: time to add Delta l = 1 at the kernel peak
            print(f"     layer added at K peak per 0.1 s dwell: {s.max()*0.1:.2f} (units of 'thickness 1' = D_L/kQ = {80/kq:.1f} mm)")
            print(f"     drag-limited spreading diffusivity cp^2 l^2/beta (l=1): {P.cp**2/P.beta:.3f} D^2/s ; cp*dt damping: see numerics")
        else:
            sol._bulk_flow()
            Fr2 = inl.U_perp ** 2 / (P.g_red * inl.d_jet)
            print(f"     Fr^2={Fr2:.2f} (Fr={math.sqrt(Fr2):.2f}), tau_new=2U/g'={sol.tau_new*1e3:.0f} ms, S(=S_v)={sol._S:.4f}/s, S_E={sol._SE:.4f}/s, "
                  f"R_c={sol._Rc*80:.2f} mm (grid floor 2h={2*h*80:.2f} mm), foam_src_peak={sol.foam_src.max():.1f}/s, Lam_jet_peak={sol._Lam_jet.max():.1f}/s, v_jet={np.hypot(*sol._vjet):.2f} D/s")
            Ub, Ud = sol.Ub, sol.Ud
            spd_s = np.hypot(*Ub); spd_d = np.hypot(*Ud)
            dv = sol.divUs
            x0, y0 = inl.x_hit
            r = np.hypot(g.x - x0, g.y - y0)
            ring = (r > max(r1f, r2f)) & (r < max(sol._Rc, 2 * h)) & m
            print(f"     |U_s| max={spd_s.max():.2f} D/s, |U_d| max={spd_d.max():.2f} D/s; |U_s| at r=0.1 D: {np.median(spd_s[(abs(r-0.1)<h)&m]):.3f}, at r=0.2: {np.median(spd_s[(abs(r-0.2)<h)&m]):.3f} D/s")
            print(f"     div U_s: min={dv.min():.0f}/s (at jet), max={dv.max():.1f}/s; in upwelling ring (footprint<r<R_c, {int(ring.sum())} cells): median {np.median(dv[ring]) if ring.any() else float('nan'):.1f}/s")
            if P.closure == "v2":
                sv = inl.S_eff * P.D_L / P.H; sf = inl.S_eff * P.D_L / P.H_f
                sk = P.l_skin * np.maximum(dv, 0)
                print(f"     S_v={sv:.4f}/s, S_f={sf:.4f}/s (H/H_f={P.H/P.H_f:.1f}); s_skin in ring median {np.median(sk[ring]) if ring.any() else float('nan'):.2f} H_f/s -> equilibrium l in ring = l_skin = {P.l_skin}")
                print(f"     film e-fold thinning time in ring = 1/divUs = {1e3/max(np.median(dv[ring]),1e-9) if ring.any() else float('nan'):.0f} ms")
                print(f"     drag-limited spreading diffusivity cp^2 l^2/beta: l=1: {P.cp**2/P.beta:.2f}, l=2: {4*P.cp**2/P.beta:.2f} D^2/s; inertial gravity speed sqrt(cp^2 l) l=2: {math.sqrt(2*P.cp**2):.2f} D/s")
                print(f"     yield: pressure beats yield stress when cp^2*Delta_l > tau_y -> Delta_l > {P.tau_y/P.cp**2:.3f} H_f = {P.tau_y/P.cp**2*P.H_f*1e3:.2f} mm ; tau_y/yield_eps={P.tau_y/P.yield_eps:.0f} vs nu_max={P.nu_max}")
                print(f"     SSA per-frame smoothing length sqrt(nu_max*frame_dt)={math.sqrt(P.nu_max/120)*80:.1f} mm; beta/l: l=0.03: {P.beta/0.03:.0f}/s, l=1: {P.beta:.1f}/s, l=3: {P.beta/3:.2f}/s")
                print(f"     opacity: m_o={P.m_o:.3f} H_f = {P.m_o*P.H_f*1e3:.2f} mm; W=0.5 at m={math.log(2)*P.m_o:.3f} H_f")
            else:
                print(f"     twolayer: 'thickness 1' = D_L/kQ = {80/kq:.1f} mm; film source s_f peak (after rise) ~ foam_src peak; drag -beta(q - l U_s) acts on momentum, rate beta={P.beta:.2f}/s independent of l")
                print(f"     drag-limited spreading diffusivity cp^2 l^2/beta (l=1): {P.cp**2/P.beta:.3f} D^2/s")

report("OLD chi closure, physics #9", OLD, kQ)
report("TWOLAYER + cell, cell_fit_N96 best", dict(OLD, **CELL), kQ)
report("V2 final (joint_cma_v2 gen14 + polish)", V2F, 1.0)
report("V2s (struct objective, joint_cma_v2s gen13)", V2S, 1.0)

# numerics
print("\n=== numerics at N=96 ===")
for umax in (0.5, 2.0):
    dt = 0.5 * h / (2 * umax)
    print(f"  umax={umax} D/s: dt_cfl={dt*1e3:.2f} ms, nsub per 8.33 ms frame = {math.ceil((1/120)/dt)}; Rusanov nu_num=|u|h={umax*h:.4f} D^2/s (physical nu 1e-3..4e-3); mass-flux nu_num=|u|h/2={umax*h/2:.4f}")
    for cp in (0.97, 2.45):
        for lam_mm in (4.0, 10.0):
            k = 2 * math.pi / (lam_mm / 80)
            amp = 1 / math.sqrt(1 + (cp * k * dt) ** 2)
            print(f"     implicit-Euler pressure wave damping cp={cp}, wavelength {lam_mm} mm: per step {amp:.3f}, per 0.1 s {amp**(0.1/dt):.2e}")
print(f"  first-order upwind (f, ft, msub) nu_num=|U_d|h/2; at |U_d|=1 D/s: {1*h/2:.4f} D^2/s -> spreads {math.sqrt(2*h/2*0.3)*80:.1f} mm in a 0.3 s rise time")
print(f"  cell size {h*80:.2f} mm; van Leer interface ~2-3 cells = {2.5*h*80:.1f} mm")
