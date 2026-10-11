"""Revision numbers for LATTE_MODEL_V3.md v3.1-spec (section 10 review log).
Every new or changed number in v3.1 is printed here, using only formulas written in the document.
Run:  python3 -I v3_numbers3.py > v3_numbers3.txt
"""
import math
import numpy as np

g = 9.81; D = 0.08; R = 0.04; Acup = math.pi * R**2
rho = 1005.0; mu = 1.0e-3; nu = mu / rho; sigma = 0.045
phi_j = 0.25; phi_f = 0.70; phi_J = 0.64; nRZ = 4.65
z_e = 0.30e-3; tau_y = 2.0; k_ax = math.sqrt(3.0); k_pl = 2.0
rho_f = (1 - phi_f) * rho; gp = g * (1 - rho_f / rho)


def P(*a):
    print(*a)


def vst(Rb, mu_=mu):
    return 2 * rho * g * Rb**2 / (9 * mu_)


def vb(Rb, phi, mu_=mu):
    return vst(Rb, mu_) * (1 - phi)**nRZ


def kynch_rate(Rb, phi):
    """gas column accumulation rate of the jammed layer, db/dt = phi_J phi v_b/(phi_J - phi)"""
    return phi_J * phi * vb(Rb, phi) / (phi_J - phi)


b_star = phi_J * z_e
P("=== 0. basics")
P(f"rho_f={rho_f:.0f} g'={gp:.2f}  b_*=phi_J h_*={b_star*1e3:.3f} mm  phi_f h_*={phi_f*z_e*1e3:.3f} mm")

P("\n=== 1. (F1) bulk gas: canvas nucleation time without loss, t = b_*/(db/dt)")
for Rb in (45e-6, 20e-6, 10e-6):
    row = []
    for pB in (0.002, 0.005, 0.01, 0.02, 0.05):
        row.append(f"phi_B={pB}: {b_star/kynch_rate(Rb, pB):.1f} s")
    P(f"R={Rb*1e6:.0f} um: " + "; ".join(row))
T_draw = 20.0
P(f"V3 bound (no loss, 20 s window): mean bulk Kynch rate < b_*/20 s = {b_star/T_draw*1e3:.4f} mm/s")
for Rb in (45e-6, 30e-6, 20e-6, 10e-6):
    # largest phi_B with kynch_rate < b_*/T
    lo, hi = 0.0, 0.3
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if kynch_rate(Rb, mid) * T_draw < b_star:
            lo = mid
        else:
            hi = mid
    P(f"  R={Rb*1e6:.0f} um: phi_B < {lo:.4f}")
P("well-mixed bulk gas from the jet (fraction not captured, 1-F_white) in 200 mL:")
for Fw in (0.1, 0.25):
    rate = (1 - Fw) * phi_j * 15e-6 / 200e-6
    P(f"  F_white={Fw}: dphi_B/dt={rate:.4f} /s; after 5 s phi_B={5*rate:.3f}; steady exit flux if all leaves through cup surface = {(1-Fw)*phi_j*15e-6/Acup*1e3:.2f} mm/s")

P("\n=== 2. (F1/F2) flux-threshold nucleation with bare-canvas loss k_s: db/dt = r_K - k_s b")
for phiN in (0.10, 0.06):
    rK = kynch_rate(45e-6, phiN)
    P(f"phi_N={phiN}: Kynch rate r_K={rK*1e3:.3f} mm/s (source j=phi v_b={phiN*vb(45e-6,phiN)*1e3:.3f}; ratio {rK/(phiN*vb(45e-6,phiN)):.2f})")
    for ks in (0.0, 0.25, 0.5, 1.0, 1.5):
        x = ks * b_star / rK
        if ks == 0:
            tk = b_star / rK
        elif x < 1:
            tk = -math.log(1 - x) / ks
        else:
            tk = float('inf')
        P(f"   k_s={ks:4.2f}/s: k_s b_*={ks*b_star*1e3:.3f} mm/s, t_K={tk:.2f} s")
P("k_s bounds: lower from bulk (j_B < k_s b_*), upper from first white (r_K(r_e) > k_s b_*):")
for Rb, pB in ((45e-6, 0.01), (45e-6, 0.03), (20e-6, 0.05), (10e-6, 0.05)):
    P(f"  bulk R={Rb*1e6:.0f} um phi_B={pB}: k_s > {kynch_rate(Rb,pB)/b_star:.2f} /s")
for phiN in (0.10, 0.06):
    P(f"  near field phi_N={phiN}: k_s < {kynch_rate(45e-6,phiN)/b_star:.2f} /s")

P("\n=== 3. (F2) flux-area budget: supply under rafts <= max_phi phi v_b(phi) x active area")
for Rb in (35e-6, 45e-6, 60e-6, 80e-6):
    phis = np.linspace(0.001, 0.6, 6000)
    fl = phis * vb(Rb, phis)
    i = np.argmax(fl)
    P(f"R={Rb*1e6:.0f} um: max flux {fl[i]*1e3:.3f} mm/s at phi={phis[i]:.3f}")
jmax45 = max(p * vb(45e-6, p) for p in np.linspace(0.001, 0.6, 6000))
for dA in (1.6e-4, 3.8e-4):
    for phi_r, h in ((0.66, 1.0e-3), (0.68, 2.0e-3), (0.70, 3.35e-3)):
        need = phi_r * h * dA
        A_act = need / jmax45
        A_act2 = need / (0.10 * vb(45e-6, 0.10))
        P(f"dA/dt={dA*1e4:.1f} cm2/s, raft phi h={phi_r*h*1e3:.2f} mm: gas {need*1e6:.2f} mL/s -> active area >= {A_act*1e4:.1f} cm2 (r={math.sqrt(A_act/math.pi)*100:.2f} cm) at max flux; {A_act2*1e4:.1f} cm2 at phi_N=0.1")
P("identity (same flux sets delay and supply, no loss): A_act = (dA/dt) t_K phi h / b_*")
for phi_r, h in ((0.66, 1.0e-3), (0.70, 3.35e-3)):
    for tK in (0.6, 0.9):
        P(f"   phi h={phi_r*h*1e3:.2f} mm t_K={tK}: A_act={3.8e-4*tK*phi_r*h/b_star*1e4:.1f} cm2")

P("\n=== 4. (F3) rigid old block (moves with undisturbed canvas velocity at its centre) vs material points")


def vel(x, xi, Sa):
    d = x - xi; r2 = np.dot(d, d)
    xin = xi * R**2 / np.dot(xi, xi); di = x - xin; ri2 = np.dot(di, di)
    return Sa * ((d / r2 + di / ri2) / (2 * np.pi) - x / (2 * np.pi * R**2))


xi = np.array([0.3 * D, 0.0]); a_blk = 0.165 * D
for S_raft, S_eff in ((2.6e-4, 2.6e-4), (3.8e-4, 3.8e-4), (2.6e-4, 1.6e-4), (3.8e-4, 2.8e-4)):
    for s0 in (0.12 * D, 0.15 * D, 0.20 * D):
        c = xi - np.array([s0 + a_blk, 0.0]); pm = xi - np.array([s0, 0.0]); c0 = c.copy()
        t = 0.0; dt = 0.002; A0 = 0.5e-4; out = []; closed = None
        while t < 3.0001:
            Rn = math.sqrt((A0 + S_raft * t) / math.pi)
            gr = (np.linalg.norm(c - xi) - a_blk - Rn) / D
            gm = (np.linalg.norm(pm - xi) - Rn) / D
            if gr <= 0 and closed is None:
                closed = t
            if min(abs(t - k) for k in (0, 1, 2, 3)) < 1e-6:
                out.append(f"t={t:.0f}: {max(gr,0):.3f}/{gm:.3f}")
            k1 = vel(c, xi, S_eff); k2 = vel(c + 0.5 * dt * k1, xi, S_eff); c = c + dt * k2
            k1 = vel(pm, xi, S_eff); k2 = vel(pm + 0.5 * dt * k1, xi, S_eff); pm = pm + dt * k2
            t += dt
        P(f"S_raft={S_raft*1e4:.1f} S_eff={S_eff*1e4:.1f} s0={s0/D:.2f}D gap rigid/material: " + " | ".join(out)
          + (f"  closes at {closed:.2f} s" if closed else "") + f"  block centre moved {np.linalg.norm(c-c0)/D:.3f} D in 3 s")

P("\n=== 5. (F4/M1/M2) canvas pressure during and after the pour")
dAcup = [2 * math.pi * R * math.tan(math.radians(b)) * 2.2e-3 for b in (5, 10)]
P(f"cone wall area source at dH/dt=2.2 mm/s: {dAcup[0]*1e4:.2f}-{dAcup[1]*1e4:.2f} cm2/s")
for S in (2.6e-4, 3.8e-4):
    for Fc in (0.7e-4, 1.5e-4):
        for w in dAcup:
            edot = (S - Fc - w) / 35e-4
            row = []
            for E, td in ((0.03, 1.0), (0.03, 3.0), (0.005, 3.0), (0.03, 10.0)):
                row.append(f"E={E*1e3:.0f},tau_d={td:.0f}: {E*edot*td*1e3:.1f}")
            P(f"S={S*1e4:.1f} F_c={Fc*1e4:.1f} wall={w*1e4:.2f}: |e_dot|={edot:.3f}/s; pi_ss=E|e|tau_d [mN/m] " + "; ".join(row))
P("whole-pattern canvas strain (tau_d -> inf): eps_tot = (A_white - F_c t_low - dA_cone)/A_canvas_final")
for fw in (0.23, 0.42):
    for tlow, Fc in ((5, 0.7e-4), (10, 1.5e-4)):
        for gain in (0.066, 0.136):
            Aw = fw * Acup * (1 + gain)
            Acan = Acup * (1 + gain) - Aw
            eps = (Aw - Fc * tlow - gain * Acup) / Acan
            P(f"  f_w={fw} t_low={tlow}s F_c={Fc*1e4:.1f} cone+{gain:.3f}: eps_tot={eps:+.2f} -> E eps (E=30)={max(eps,0)*30:.1f} mN/m, (E=10) {max(eps,0)*10:.1f}")


def hpi(Pi, k=k_ax, ty=tau_y, rf=rho_f, gpp=gp):
    a = 0.5 * rf * gpp; b = -k * ty
    return (-b + math.sqrt(b * b + 4 * a * Pi)) / (2 * a)


hm = hpi(0.0)
P(f"h_m(axi)={hm*1e3:.2f} mm")
for Pi in (0.5e-3, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 10e-3):
    h = hpi(Pi)
    gain = h / hm - 1
    # a 0.03 D stripe between two lines: each line loses half the gain of each neighbour
    dw = 0.03 * D * gain
    P(f"pi_bar={Pi*1e3:.1f} mN/m: h_pi={h*1e3:.2f} mm, post-stop area gain {gain:.0%}, line between two 0.03 D stripes narrows by {dw*1e3:.2f} mm = {dw/D:.4f} D")


def hcrush(Pi, k, ty):
    a = 0.5 * rho_f * gp; b = k * ty
    return (-b + math.sqrt(b * b + 4 * a * Pi)) / (2 * a)


for Pi in (1e-3, 2e-3, 5e-3, 10e-3, 15e-3):
    P(f"crush: pi_bar={Pi*1e3:.0f} mN/m crushes rafts thinner than {hcrush(Pi,2.0,2.0)*1e3:.2f} mm (k=2, 2 Pa) / {hcrush(Pi,k_ax,0.5)*1e3:.2f} mm (k=sqrt3, fresh 0.5 Pa)")

P("\n=== 6. (F4b) expansion screening length l_a = sqrt(E tau_a / zeta), zeta=0.57")
for E in (0.005, 0.03):
    P(f"E={E*1e3:.0f}: " + ", ".join(f"tau_a={ta:g}: {math.sqrt(E*ta/0.57)*1e3:.1f} mm" for ta in (1e-4, 1e-3, 3e-2, 1.0)))
P("self-consistent t_Pi = (L^2 mu /(E sqrt(pi nu)))^(2/3), L=4 cm:")
for E in (0.06, 0.03, 0.01, 0.005, 0.002, 0.001):
    tP = (0.04**2 * mu / (E * math.sqrt(math.pi * nu)))**(2 / 3)
    P(f"  E={E*1e3:.0f} mN/m: t_Pi={tP:.3f} s, zeta(t_Pi)={mu/math.sqrt(math.pi*nu*tP):.2f}")
P(f"reservoir drag pressure drop across r_N: zeta U_N r_N = {0.57*0.03*0.02*1e3:.2f}-{0.57*0.12*0.02*1e3:.2f} mN/m; strain {0.57*0.06*0.02/0.03:.3f} (U_N=6 cm/s, E=30)")
P(f"inner-zone drag that holds the stagnation ring: zeta F_c/(2 pi) = {0.57*1e-4/(2*math.pi)*1e3:.4f} mN/m")

P("\n=== 7. (M3) fresh-raft yield stress and stopping thickness")
for phi, ty in ((0.66, 0.25), (0.66, 0.5), (0.68, 1.0), (0.70, 2.0)):
    rf = (1 - phi) * rho; gpp = g * (1 - rf / rho)
    P(f"phi={phi} tau_y={ty} Pa: rho_f g'={rf*gpp:.0f} N/m3, h_m(axi)={2*k_ax*ty/(rf*gpp)*1e3:.2f} mm, gas column phi h={phi*2*k_ax*ty/(rf*gpp)*1e3:.2f} mm")
kD = (45e-6)**2 * (0.35)**3 / (180 * 0.65**2)   # Carman-Kozeny estimate
vD = kD * rho * g / mu
P(f"Darcy drainage estimate: k_D={kD:.2e} m2, v_D={vD*1e3:.3f} mm/s; compaction 0.64->0.70 of 1-3 mm: {0.06*1e-3/vD:.1f}-{0.06*3e-3/vD:.1f} s")

P("\n=== 8. (M4) h_* proportional to R_32")
for Rb in (35e-6, 45e-6, 60e-6):
    P(f"R={Rb*1e6:.0f}: z_e=0.7*3.5*2R={0.7*3.5*2*Rb*1e3:.2f} mm (c_z={0.7*3.5*2:.1f})")
phiN = 0.1
tk = lambda Rb, hs: hs * (phi_J - phiN) / (phiN * vb(Rb, phiN))
P(f"t_K ratio R 35/60: fixed h_* {tk(35e-6,3e-4)/tk(60e-6,3e-4):.2f}; h_* = c_z R {tk(35e-6,6.67*35e-6)/tk(60e-6,6.67*60e-6):.2f}")
P(f"t_K at 45 um with h_* = 6.67 R: {tk(45e-6,6.67*45e-6):.2f} s")
P(f"mu_l x2 (fixed R): t_K x2; t_w = t_r + t_K: {0.4+0.6:.1f} -> {0.4+1.2:.1f} s")

P("\n=== 9. (M7) inner-zone geometry")
for name, Q, z in (("low", 15e-6, 0.015), ("low20", 20e-6, 0.02), ("cut", 3e-6, 0.07)):
    v = math.sqrt(0.35**2 + 2 * g * z); d = math.sqrt(4 * Q / (math.pi * v)); r0 = d / 2
    Fr0 = v / math.sqrt(phi_j * g * r0); zm = 2.46 * Fr0 * r0; zz = min(zm, 0.045)
    for dc in (0.2e-3, 1.0e-3):
        rc = r0 + dc
        P(f"{name}: r0={r0*1e3:.2f} mm, r_c=r0+{dc*1e3:.1f}={rc*1e3:.2f} mm; sheath b_s=0.1-0.2 min(z_m,H)={0.1*zz*1e3:.1f}-{0.2*zz*1e3:.1f} mm; "
          f"r_e in ({rc*1e3:.1f}, {(r0+0.2*zz)*1e3:.1f}) mm; hole diameter {2*rc/D:.3f}-{2*(r0+0.2*zz)/D:.3f} D")
for re_, rc_, Fc in ((3.5e-3, 3.0e-3, 1.0e-4), (5.4e-3, 3.0e-3, 1.0e-4), (5.4e-3, 3.0e-3, 0.7e-4)):
    P(f"inner residence pi(r_e^2-r_c^2)/F_c: r_e={re_*1e3:.1f} r_c={rc_*1e3:.1f} F_c={Fc*1e4:.1f}: {math.pi*(re_**2-rc_**2)/Fc:.2f} s")

P("\n=== 10. (M8) P_s prior form P_s=min(1, exp(-lam (z_m/H-1)))")
for lam in (0.5, 1.4):
    P(f"lam={lam}: " + ", ".join(f"z_m/H={x}: {min(1,math.exp(-lam*(x-1))):.2f}" for x in (0.96, 1.18, 1.37, 1.57)))
for Ps, S in ((0.46, 10), (0.9, 5)):
    P(f"thin stream phi_N = P_s phi_j/(1+S) = {Ps*phi_j/(1+S):.3f} (P_s={Ps}, S={S})")

P("\n=== 11. (M10) delay-spread smearing of wiggle modulation, f=2.75 Hz")
for st in (0.03, 0.06, 0.1, 0.2):
    P(f"sigma_t={st}: {math.exp(-(2*math.pi*2.75*st)**2/2):.3f}")
for ta, lab in ((1.0, '1 s'), (2.0, '2 s')):
    for phih in (0.66e-3, 2.35e-3):
        widen = 0.27e-3 / phih * 0.03 * D * ta
        P(f"arc incorporation: residence {lab}, raft gas column {phih*1e3:.2f} mm: each line narrows by {widen*1e3:.2f} mm")

P("\n=== 12. (M11) stripe spacing, one arc per half-cycle")
for U in (0.08, 0.12, 0.16):
    P(f"U_rel={U} D/s: half-cycle {U/(2*2.75):.3f} D, cycle {U/2.75:.3f} D; retreat (U+0.16)/(2f)={(U+0.16)/(2*2.75):.3f} D")

P("\n=== 13. (M12) cut: prompt line, stem, notch")
for name, p0, d in (("Q3 z7", 564, 1.8e-3), ("Q5 z9", 712, 2.2e-3)):
    rfp = d / 2; psub = (rho - rho_f) * g * 3.35e-3
    rsub = rfp * math.sqrt(math.log(p0 / psub))
    for Pi in (400, 200):
        rd = rfp * math.sqrt(math.log(p0 / Pi)) if p0 > Pi else 0
        fd = (rd / rsub)**2
        P(f"{name} Pi_coh={Pi}: r_sub={rsub*1e3:.2f} r_d={rd*1e3:.2f} mm; prompt line 2r_d={2*rd/D:.3f} D; stem 2r_sub(1-f)={2*rsub*(1-fd)*1e3:.2f} mm={2*rsub*(1-fd)/D:.3f} D; notch r_sub={rsub/D:.3f} D")
P(f"sink convergence behind the jet, F=0.4 cm2/s at 1 cm over 0.3 s: {0.4e-4/(2*math.pi*0.01)*0.3*1e3:.2f} mm")

P("\n=== 14. (M13) translational mobility and vortex-dipole bound")
for F in (0.01e-3, 0.1e-3, 0.5e-3):
    L = 0.01; A = math.pi * (L / 2)**2
    P(f"net edge line force {F*1e3:.2f} mN/m on 1 cm raft -> U~F L/(2 zeta A) = {F*L/(2*0.57*A)*100:.2f} cm/s")
for eta in (0.03, 0.1, 1.0):
    for r in (0.012, 0.024):
        I = eta * 1.1e-3 * 1.3
        U = I / (4 * math.pi * rho * r**3)
        P(f"eta_h={eta}: impulse {I*1e3:.3f} mN s, dipole U at r={r*100:.1f} cm ~ {U*100:.2f} cm/s -> 1.3 s displacement {U*1.3/D:.3f} D")
for U in (0.03, 0.12):
    for frac in (0.3, 1.0):
        df = rho_f / rho * 3.35e-3
        P(f"form drag U_N={U*100:.0f} cm/s x{frac}: {0.5*rho*(frac*U)**2*df*1e3:.3f} mN/m")

P("\n=== 15. (m12) consistency")
for Fw in (0.083, 0.2, 0.3):
    P(f"F_white={Fw}: dA/dt={Fw*phi_j*15e-6/(phi_f*3.35e-3)*1e4:.2f} cm2/s (h=3.35 mm)")
for Q, S in ((3e-6, 5), (3e-6, 10)):
    Qr = Q * (1 + S); vbb = vb(45e-6, 0.015)
    Hz = vbb * math.pi * 0.0225**2 / Qr
    P(f"thin stream Q=3 S={S}: Hz={Hz:.2f}, f_loc={Hz/(1+Hz):.2f}")
for Fw, Q, Vp in ((0.10, 2e-6, 0.016), (0.15, 3e-6, 0.008)):
    w = Fw * phi_j * Q / (phi_f * 3.35e-3 * Vp)
    P(f"neck F_white={Fw} Q={Q*1e6:.0f} V_p={Vp*100:.1f} cm/s: w={w/D:.3f} D")

P("\n=== 16. (senior #6) cup-diameter sensitivity, D in [7, 10] cm, reference 8 cm")
for Dn in (0.07, 0.10):
    P(f"D={Dn*100:.0f} cm: absolute-length features in D units x{0.08/Dn:.2f}; area-flux features (line widths, push, dA/dt in D^2) x{(0.08/Dn)**2:.2f}; "
      f"video 3.8 cm2/s at D=8 means {3.8*(Dn/0.08)**2:.1f} cm2/s")
P(f"hole 2r_e 5.4-10.8 mm: D=7 {5.4e-3/0.07:.3f}-{10.8e-3/0.07:.3f} D; D=10 {5.4e-3/0.10:.3f}-{10.8e-3/0.10:.3f} D")

P("\n=== 17. revised t_K table: h_* = c_z R_32 (c_z = 6.67, anchored to h_* = 0.30 mm at 45 um), Kynch source, loss k_s")
c_z = 0.30e-3 / 45e-6
for Rb in (35e-6, 45e-6, 60e-6):
    hs = c_z * Rb
    for Ps in (1.0, 0.6):
        phiN = Ps * phi_j / (1 + 1.5)
        rK = phi_J * phiN * vb(Rb, phiN) / (phi_J - phiN)
        bs = phi_J * hs
        row = []
        for ks in (0.0, 0.5):
            x = ks * bs / rK
            tk = bs / rK if ks == 0 else (-math.log(1 - x) / ks if x < 1 else float('inf'))
            row.append(f"k_s={ks}: t_K={tk:.2f} s, t_w={0.39+tk:.2f}-{0.79+tk:.2f} s")
        P(f"R={Rb*1e6:.0f} um h_*={hs*1e3:.3f} mm P_s={Ps} phi_N={phiN:.2f} v_b={vb(Rb,phiN)*1e3:.2f} mm/s: " + "; ".join(row))

P("\n=== 18. growth with fresh (thin) rafts: dA/dt = F_white phi_j Q/(phi h)")
for Fw in (0.083, 0.125, 0.2):
    row = []
    for phi_r, h in ((0.66, 0.78e-3), (0.68, 1.61e-3), (0.70, 3.35e-3)):
        row.append(f"phi h={phi_r*h*1e3:.2f} mm: {Fw*phi_j*15e-6/(phi_r*h)*1e4:.1f}")
    P(f"F_white={Fw}, Q=15: dA/dt [cm2/s] " + "; ".join(row))
P("F_white needed for 3.8 cm2/s (Q=15): " + "; ".join(
    f"phi h={phi_r*h*1e3:.2f} mm: {3.8e-4*phi_r*h/(phi_j*15e-6):.3f}" for phi_r, h in ((0.66, 0.78e-3), (0.68, 1.61e-3), (0.70, 3.35e-3))))
P(f"pale first white: R=h/(h+z_e) for h=0.39/0.78/1.61 mm: {0.39/(0.39+0.3):.2f}/{0.78/(0.78+0.3):.2f}/{1.61/(1.61+0.3):.2f}")

P("\n=== 19. rigid gap after the push_heart stage-2 feed (white 7.2 s -> lift 9.8 s, 2.6 s)")
for S in (2.6e-4, 3.8e-4):
    for s0 in (0.15 * D, 0.20 * D):
        c = xi - np.array([s0 + a_blk, 0.0]); t = 0.0; dt = 0.002; A0 = 0.5e-4
        while t < 2.6 - 1e-9:
            k1 = vel(c, xi, S); k2 = vel(c + 0.5 * dt * k1, xi, S); c = c + dt * k2; t += dt
        Rn = math.sqrt((A0 + S * 2.6) / math.pi)
        P(f"S={S*1e4:.1f} s0={s0/D:.2f} D: rigid gap at 2.6 s = {max(0,(np.linalg.norm(c-xi)-a_blk-Rn)/D):.3f} D")

P("\n=== 20. post-stop reservoir residue: gas left in reservoir, white added if a fraction f_cov lies under rafts")
psi_tot = 1.1e-6
for fc in (0.3, 0.7):
    for phih in (0.52e-3, 2.34e-3):
        P(f"f_cov={fc}, raft gas column {phih*1e3:.2f} mm: +{psi_tot*fc/phih*1e4:.1f} cm2")

P("\n=== 21. high pour / thin stream t_K with the v3.1 P_s range (phi_N 0.007-0.04), 45 um, h_* = 0.30 mm")
for phiN in (0.007, 0.01, 0.02, 0.03, 0.037, 0.04):
    rK = kynch_rate(45e-6, phiN)
    row = []
    for ks in (0.0, 0.5):
        x = ks * b_star / rK
        tk = b_star / rK if ks == 0 else (-math.log(1 - x) / ks if x < 1 else float('inf'))
        row.append(f"k_s={ks}: t_K={tk:.2f} s")
    P(f"phi_N={phiN}: r_K={rK*1e3:.3f} mm/s; " + "; ".join(row))
