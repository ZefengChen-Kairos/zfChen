"""closed-form numbers of variant Q used in test_Qquasist.md (SI; D = 0.08 m)."""
import math
D = 0.08; g = 9.81; v0 = 0.35; Gamma = 0.0395; de = 1e-3; phi_j = 0.25; rho_l = 1005.0
def jet(Q, v): return math.sqrt(Q / (math.pi * v))
cases = {'low (Q 15 mL/s, z_s 1.5 cm)': (15e-6, math.sqrt(v0**2 + 2*g*0.015)), 'thin (Q 3 mL/s, v 1.2 m/s)': (3e-6, 1.2)}
for name, (Q, v) in cases.items():
    r0 = jet(Q, v); re = r0 + de; S = Q / Gamma
    print(f'{name}: v = {v:.3f} m/s, r0 = {r0*1e3:.2f} mm, A_fp = Q/v = {Q/v*1e6:.2f} mm^2, r_e = {re*1e3:.2f} mm, S = wQ/Gamma = {S*1e4:.3f} cm^2/s (w=1)')
    for aj in (5, 8, 12):
        ut = v * math.sin(math.radians(aj))
        for Vj, lab in ((0.0, 'held'), (0.1 * D / 1.5, 'push 0.1D/1.5s'), (0.4 * D, 'cut 0.4D/s')):
            if name.startswith('low') and Vj == 0.4 * D: continue
            if name.startswith('thin') and Vj == 0.1 * D / 1.5: continue
            U = ut + Vj; m = U * Q / v
            s = f'   alpha_j {aj:2d} deg, {lab:15s}: u_tilt {ut*100:5.2f} cm/s, |U_J| {U*100:5.2f} cm/s, m = {m:.3e} m^3/s'
            s += f', u_in = U/2 = {U/2*100:4.2f} cm/s, u_dip(r_e) = {m/(2*math.pi*re**2)*100:4.2f} cm/s, u_dip(1.2 cm) = {m/(2*math.pi*0.012**2)*1e3:4.2f} mm/s'
            if name.startswith('low'):
                s += f', Pi_C = m/(S r_e) = {m/(S*re):.2f}'
                if Vj > 0: s += f', d* = sqrt(m/(2 pi V_j)) = {math.sqrt(m/(2*math.pi*Vj))/D:.3f} D'
                s += f', pre-white push m t/(2 pi d^2) (d = 0.15 D, t = 1.4 s) = {m*1.4/(2*math.pi*(0.15*D)**2)/D:.3f} D'
            else:
                if Vj > 0: s += f', Pi_L = |U_J|/(2V) = {U/(2*Vj):.2f}'
                md = ut * Q / v
                s += f', dwell notch (3 m_dwell t/2pi)^(1/3), t = 0.7 s: {(3*md*0.7/(2*math.pi))**(1/3)/D:.3f} D'
            print(s)
# wiggle
Q, v = cases['low (Q 15 mL/s, z_s 1.5 cm)']; a = 4e-3; f = 3.0
Vw = 2 * math.pi * a * f; mw = Vw * Q / v
print(f'wiggle: lateral speed amplitude 2 pi a f = {Vw*100:.2f} cm/s, oscillating dipole amplitude m_w = {mw:.3e} m^3/s;'
      f' lateral sloshing amplitude at r_e: m_w/(2 pi r_e^2 2 pi f) = {mw/(2*math.pi*(jet(Q,v)+de)**2)/(2*math.pi*f)*1e3:.2f} mm')
# quasi-static Stokes numbers (rho_f h_eff = 0.33-1.4 kg/m^2, zeta = 0.33-1 Pa s/m)
for T_, lab in ((1.0, 'push ~1 s'), (1/(2*math.pi*3), 'wiggle 3 Hz'), (2*0.89e-3/(0.4*D), 'cut footprint transit')):
    lo = 0.33 / (1.0 * T_); hi = 1.4 / (0.33 * T_)
    print(f'St = rho_f h_eff/(zeta T) for {lab} (T = {T_:.3f} s): {lo:.2g} - {hi:.2g}')
# eta = zeta/(rho_j v)
for name, (Q, v) in cases.items():
    rj = (1 - phi_j) * rho_l
    print(f'eta = zeta/(rho_j v) {name}: {0.33/(rj*v):.2e} - {1.0/(rj*v):.2e}')
