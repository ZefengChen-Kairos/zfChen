"""Reference numbers for LATTE_MODEL_V3.md.  Every number quoted in the document is printed here.
Run:  python3 -I v3_numbers.py > v3_numbers.txt
Inputs are the central values / ranges of the v3 parameter table (section 5)."""
import math
import numpy as np

g = 9.81
D = 0.08; R = 0.04; A = math.pi * R**2
rho = 1005.0          # canvas liquid (coffee + liquid milk, 60-65 C)
mu = 1.0e-3           # canvas liquid viscosity (central); range 0.6-1.5 mPa s
nu = mu / rho
sigma = 0.045
v0 = 0.35             # spout exit speed
phi_j = 0.25          # gas fraction of the poured mixture
phi_f = 0.70          # gas fraction of the raft foam
phi_J = 0.64          # 3-D random close packing (Kynch jammed layer)
H0 = 0.045            # liquid depth at start of drawing
z_e = 0.30e-3         # diffuse-reflectance extrapolation length of the foam (h_* = z_e)

def P(*a):
    print(*a)

P("=== 0. geometry")
P(f"A = {A*1e4:.2f} cm2")

P("\n=== 1. jets")
def jet(Q, z):
    v = math.sqrt(v0**2 + 2*g*z)
    d = math.sqrt(4*Q/(math.pi*v))
    rho_j = (1-phi_j)*rho
    Re = v*d/nu
    We = rho_j*v**2*d/sigma
    pst = 0.5*rho_j*v**2
    M = rho_j*Q*v
    gp = phi_j*g
    r0 = d/2
    Fr0 = v/math.sqrt(gp*r0)
    zm = 2.46*Fr0*r0
    Tf = v/gp
    return dict(v=v, d=d, Re=Re, We=We, pst=pst, M=M, Fr0=Fr0, zm=zm, Tf=Tf)
rows = []
for Q, z in ((10e-6, .01), (15e-6, .015), (20e-6, .02), (2e-6, .05), (3e-6, .07), (5e-6, .09)):
    J = jet(Q, z)
    rows.append((Q, z, J))
    P(f"Q={Q*1e6:4.0f} mL/s z={z*100:3.1f} cm: v={J['v']:.2f} m/s d={J['d']*1e3:.1f} mm Re={J['Re']:.0f} We={J['We']:.0f} "
      f"p_st={J['pst']:.0f} Pa M={J['M']*1e3:.1f} mN Fr0={J['Fr0']:.1f} z_m={J['zm']*100:.1f} cm z_m/H={J['zm']/H0:.2f} T_f={J['Tf']:.2f} s")
Jlow = jet(15e-6, 0.015); Jcut = jet(3e-6, 0.07)

P("\n=== 2. bubbles and Kynch delay")
def vst(Rb, mu_=mu):
    return 2*rho*g*Rb**2/(9*mu_)
def vb(Rb, phi, n=4.65, mu_=mu):
    return vst(Rb, mu_)*(1-phi)**n
for Rb in (35e-6, 45e-6, 50e-6, 60e-6):
    P(f"R32={Rb*1e6:.0f} um: v_St={vst(Rb)*1e3:.2f} mm/s (mu=1.0); {vst(Rb,0.65e-3)*1e3:.2f} (mu=0.65); {vst(Rb,1.5e-3)*1e3:.2f} (mu=1.5)")
S_dil = 1.5
def tw(Rb, Ps=1.0, S=S_dil, ct=(1.5, 3.0), J=Jlow, mu_=mu):
    phiN = Ps*phi_j/(1+S)
    vbb = vb(Rb, phiN, mu_=mu_)
    jcr = phiN*vbb
    shock = jcr/(phi_J-phiN)
    tK = z_e/shock
    tr = (ct[0]*J['Tf'], ct[1]*J['Tf'])
    return phiN, vbb, jcr, shock, tK, tr
for Rb in (35e-6, 45e-6, 60e-6):
    for Ps in (1.0, 0.6):
        phiN, vbb, jcr, shock, tK, tr = tw(Rb, Ps)
        P(f"R32={Rb*1e6:.0f} um P_s={Ps}: phi_N={phiN:.3f} v_b={vbb*1e3:.2f} mm/s j={jcr*1e3:.3f} mm/s Kynch shock {shock*1e3:.2f} mm/s "
          f"t_K={tK:.2f} s t_r={tr[0]:.2f}-{tr[1]:.2f} s -> t_w={tr[0]+tK:.2f}-{tr[1]+tK:.2f} s")
# cut stream
phiN, vbb, jcr, shock, tK, tr = tw(45e-6, 0.3, S=4.0, J=Jcut)
P(f"cut stream (P_s 0.3, S 4): phi_N={phiN:.3f} t_K={tK:.1f} s, t_r={tr[0]:.2f}-{tr[1]:.2f} s")

P("\n=== 3. raft (membrane / SSA regime)")
rho_f = (1-phi_f)*rho
gpf = g*(1-rho_f/rho)
P(f"rho_f={rho_f:.0f} kg/m3, g'={gpf:.2f} m/s2, rho_f g' = {rho_f*gpf:.0f} N/m3; emergent fraction {1-rho_f/rho:.2f}")
for ty in (1.0, 2.0, 3.0, 4.0):
    hp = 4*ty/(rho_f*gpf); ha = 2*math.sqrt(3)*ty/(rho_f*gpf)
    sc = 4*ty**2/(2*rho_f*gpf)
    P(f"tau_y={ty} Pa: h_m plane={hp*1e3:.2f} mm axisym={ha*1e3:.2f} mm; margin s_c={sc*1e3:.1f} mN/m")
ty = 2.0; hm = 2*math.sqrt(3)*ty/(rho_f*gpf); hm_p = 4*ty/(rho_f*gpf)
P(f"central h_m (axisym) = {hm*1e3:.2f} mm")
for h in (3e-3, 4e-3):
    Pc = 2*ty*h + 0.5*rho_f*gpf*h**2
    P(f"compressive capacity of a raft h={h*1e3:.0f} mm, tau_y=2: 2 tau_y h + N = {Pc*1e3:.0f} mN/m")
P(f"free gravity current sqrt(g' h) at h=3.4 mm: {math.sqrt(gpf*3.4e-3):.3f} m/s = {math.sqrt(gpf*3.4e-3)/D:.1f} D/s")
# lubrication formula for comparison
P(f"physics_prior lubrication h_stop sqrt(2 tau R/(drho g)) R=2cm tau=2: {math.sqrt(2*2*0.02/(700*g))*1e3:.1f} mm")
# basal drag / yield ratio (drag Bingham)
for t in (0.3, 1.0, 3.0):
    dnu = math.sqrt(math.pi*nu*t)
    P(f"Rayleigh layer t={t}s: delta={dnu*1e3:.2f} mm zeta=mu/delta={mu/dnu:.2f} Pa s/m")
zeta = mu/math.sqrt(math.pi*nu*1.0)
Bn_d = zeta*0.005*0.03/(2*ty*hm)
P(f"drag Bingham number zeta dU L/(2 tau h) (dU 5 mm/s, L 3 cm) = {Bn_d:.3f}")
# footprint line force vs yield
for name, J in (("low", Jlow), ("cut", Jcut)):
    P(f"{name}: stagnation p={J['pst']:.0f} Pa; footprint line force p*h={J['pst']*hm*1e3:.0f} mN/m vs yield 2 tau h={2*ty*hm*1e3:.0f} mN/m; "
      f"submergence resistance drho g h={(rho-rho_f)*g*hm:.0f} Pa")

P("\n=== 4. canvas film")
for E in (0.005, 0.01, 0.03, 0.06):
    for L in (0.04,):
        tPi = L**2*zeta/E
        P(f"E={E*1e3:.0f} mN/m: D_Pi=E/zeta={E/zeta:.3f} m2/s, t_Pi=L^2 zeta/E={tPi:.3f} s (L=4cm, zeta={zeta:.2f})")
# strain needed for film to stop a 5 mm/s underflow over L=4 cm
eps = zeta*0.005*0.04/0.03
P(f"isobaric strain perturbation to resist 5 mm/s underflow over 4 cm with E=30 mN/m: {eps*100:.2f} %")

P("\n=== 5. near-field area fluxes")
Fc_lam = 8*math.pi*nu
for name, J in (("low", Jlow), ("cut", Jcut)):
    Fturb = 2*math.pi*0.08*(J['d']/2)*J['v']
    P(f"{name}: laminar capture flux 8 pi nu = {Fc_lam*1e4:.2f} cm2/s; turbulent bound 2 pi alpha r0 v = {Fturb*1e4:.1f} cm2/s")
P(f"PE mobile-surface source Q/H: low {15e-6/H0*1e4:.2f} cm2/s, cut {3e-6/H0*1e4:.2f} cm2/s")
for Vp in (0.3*D, 0.45*D, 0.6*D):
    for Fc in (Fc_lam, 0.35e-4, 0.5e-4):
        P(f"trail width F_c/V_p, V_p={Vp/D:.2f} D/s, F_c={Fc*1e4:.2f} cm2/s: {Fc/Vp*1e3:.2f} mm = {Fc/Vp/D:.4f} D")
# dispersal slot
for Pi_coh in (200, 300, 400):
    We = Jcut['pst']/Pi_coh
    rfp = Jcut['d']/2
    if We > 1:
        wd = 2*rfp*math.sqrt(math.log(We))*1.0
    else:
        wd = 0
    P(f"cut: Pi_coh={Pi_coh} Pa We_coh={We:.2f}; dispersed slot 2 r_fp sqrt(ln We) (r_fp=d/2) = {wd*1e3:.2f} mm; low pour We_coh={Jlow['pst']/Pi_coh:.2f}")

P("\n=== 6. delay-free pre-white displacement (film regime, area sources only)")
def vel(x, xi, Sa):
    d = x - xi; r2 = np.sum(d*d, axis=-1)[..., None]
    xin = xi*R**2/np.dot(xi, xi)
    di = x - xin; ri2 = np.sum(di*di, axis=-1)[..., None]
    gg = (d/r2 + di/ri2)/(2*np.pi) - x/(2*np.pi*R**2)
    return Sa*gg
def advect(p0, xi, Sa, T, n=400):
    p = np.array(p0, float); dt = T/n
    for k in range(n):
        k1 = vel(p, xi, Sa); k2 = vel(p+0.5*dt*k1, xi, Sa); p = p+dt*k2
    return p
xi = np.array([0.3*D, 0.0])
def disp_table(Sa, T, label):
    out = []
    for s in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55):
        p0 = xi - np.array([s*D, 0.0])
        p = advect(p0[None, :], xi, Sa, T)[0]
        out.append(f"{s:.2f}D:{np.linalg.norm(p-p0)/D:.3f}")
    P(f"{label}: area source {Sa*1e4:.2f} cm2/s for {T} s -> " + "  ".join(out))
disp_table(15e-6/H0, 1.3, "PE mobile limit Q/H (Q15,H4.5)")
disp_table(0.1e-4, 1.3, "film regime stationary jet F_net<=0.1")
disp_table(0.3e-4, 1.3, "film regime F_net=0.3")
disp_table(1.0e-4, 1.4, "film regime moving jet F_up=1.0")
for dA in (2.6e-4, 3.8e-4):
    disp_table(dA, 1.0, f"after white: raft growth {dA*1e4:.1f} cm2/s, per second")
    disp_table(dA, 2.0, f"after white: raft growth {dA*1e4:.1f} cm2/s, 2 s stage")

P("\n=== 7. white growth (supply limited)")
for Q in (10e-6, 15e-6, 20e-6):
    for F in (0.10, 0.2, 0.3):
        dA = F*phi_j*Q/(phi_f*hm)
        P(f"Q={Q*1e6:.0f} F_white={F}: dA/dt={dA*1e4:.2f} cm2/s")
need = 3.8e-4*phi_f*hm/(phi_j*15e-6)
P(f"F_white required for 3.8 cm2/s at Q=15: {need:.2f}")
# Hazen
for Rb in (45e-6, 60e-6):
    phiN = phi_j/(1+S_dil)
    Vh = vb(Rb, phiN)
    for rN in (0.02, 0.025):
        Qr = 15e-6*(1+S_dil)
        Hz = Vh*math.pi*rN**2/Qr
        P(f"Hazen R32={Rb*1e6:.0f} r_N={rN*100:.1f}cm: V_h={Vh*1e3:.2f} mm/s Hz={Hz:.3f} f_loc(CSTR)={Hz/(1+Hz):.3f}")
# heart disc check: 0.3 D -> 0.65 D diameter 2.3 -> 6.7 s
a1 = math.pi*(0.15*D)**2; a2 = math.pi*(0.325*D)**2
P(f"heart: area {a1*1e4:.1f} -> {a2*1e4:.1f} cm2 over 4.4 s: {(a2-a1)/4.4*1e4:.2f} cm2/s")
for Rr in (0.09*D, 0.15*D, 0.25*D):
    P(f"edge speed at R={Rr/D:.2f} D with 3.8 cm2/s: {3.8e-4/(2*math.pi*Rr)/D:.3f} D/s")

P("\n=== 8. stripes")
a = 0.008; f = 2.75
Vlat = 4*a*f
P(f"lateral spout speed 4af = {Vlat*100:.1f} cm/s")
for Fc in (Fc_lam, 0.5e-4, 1.0e-4, 1.5e-4):
    P(f"brown line F_c/(4af) F_c={Fc*1e4:.2f}: {Fc/Vlat*1e3:.2f} mm = {Fc/Vlat/D:.4f} D")
for dA in (2.6e-4, 3.8e-4):
    for Ls in (0.03, 0.04):
        U = dA/Ls
        P(f"dA/dt={dA*1e4:.1f} L_s={Ls*100:.0f}cm: U_rel={U*100:.2f} cm/s={U/D:.3f} D/s; pitch/half-cycle {U/(2*f)/D:.4f} D; per period {U/f/D:.4f} D; "
          f"with retreat 0.16 D/s: half {(U+0.16*D)/(2*f)/D:.4f} D")
P(f"video layered stack growth 0.35 D/2.9 s = {0.35/2.9:.3f} D/s")
for dA in (3.8e-4,):
    Vhalf = dA/(2*f)  # area per half-cycle
    for Ls in (0.03, 0.04):
        P(f"arc width per half cycle (all area into arcs) L_s={Ls*100:.0f}: {Vhalf/Ls/D:.4f} D")
# stripe survival
for h in (2e-3, 3e-3):
    P(f"stripe self-support needs tau_y >= rho_f g' h/4 = {rho_f*gpf*h/4:.2f} Pa (h={h*1e3:.0f} mm)")

P("\n=== 9. neck / head")
for F in (0.1, 0.15, 0.25):
    for Q, Vp in ((2e-6, 0.008), (3e-6, 0.008), (3e-6, 0.016)):
        w = F*phi_j*Q/(phi_f*hm*Vp)
        P(f"neck F={F} Q={Q*1e6:.0f} V_p={Vp*100:.1f}cm/s: width {w*1e3:.1f} mm = {w/D:.3f} D")
for F in (0.14, 0.25):
    for t in (0.5, 1.0):
        Ah = F*phi_j*15e-6*t/(phi_f*hm)
        P(f"head F={F} t={t}s: area {Ah*1e4:.2f} cm2 diameter {2*math.sqrt(Ah/math.pi)/D:.2f} D")

P("\n=== 10. level, taper, tilt")
for Q in (10e-6, 15e-6, 20e-6):
    P(f"Q={Q*1e6:.0f}: level rise rate (1-phi_j)Q/A = {(1-phi_j)*Q/A*1e3:.2f} mm/s")
for V in (50e-6, 100e-6):
    dH = (1-phi_j)*V/A
    for beta in (5, 10):
        dR = dH*math.tan(math.radians(beta))
        P(f"pour {V*1e6:.0f} mL: dH={dH*100:.2f} cm; taper {beta} deg: dR={dR*1e3:.2f} mm, area +{((R+dR)**2/R**2-1)*100:.1f} %")
k11 = 1.841/R
for th in (15, 20, 25):
    t = math.tan(math.radians(th))
    xi_c = 0.72*R*t/math.tanh(k11*H0)
    P(f"tilt {th} deg: mobile-limit drift 0.72 R tan coth(kH) = {xi_c/D:.3f} D; PB coefficient 0.86 -> {0.86*R*t/math.tanh(k11*H0)/D:.3f} D; "
      f"film-limit affine shortening 1-cos = {(1-math.cos(math.radians(th)))*100:.1f} %; low-side extra depth R sin = {R*math.sin(math.radians(th))*100:.2f} cm")
P(f"sloshing period 2pi/sqrt(g k tanh kH) = {2*math.pi/math.sqrt(g*k11*math.tanh(k11*H0)):.2f} s; kH={k11*H0:.2f}")

P("\n=== 11. canvas strain budget over a pattern")
for wf in (0.23, 0.42):
    for taper in (0.0, 0.09):
        eps = wf - taper
        P(f"white fraction {wf}, taper area gain {taper}: canvas areal strain ~{eps:.2f}; E*eps (E=30) = {30*eps:.1f} mN/m")

P("\n=== 12. gaps")
for w0, strain, stretch in ((0.15, 0.2, 1.5), (0.12, 0.3, 1.3), (0.15, 0.1, 1.0)):
    P(f"gap w0={w0} D, canvas strain {strain}, arc stretch x{stretch}: w = {w0*(1-strain)/stretch:.3f} D")

P("\n=== 13. resolution / time step")
for N in (256, 384, 512):
    dx = D/N
    P(f"N={N}: dx={dx*1e3:.3f} mm; 0.01 D line = {0.01*D/dx:.1f} cells; cut d=1.8mm = {1.8e-3/dx:.1f} cells; "
      f"dt<=dx/|x_dot|: wiggle {dx/Vlat*1e3:.2f} ms, cut 0.6D/s {dx/(0.6*D)*1e3:.2f} ms")
UN = 15e-6*(1+S_dil)/(2*math.pi*0.005*0.01)
P(f"U_N at r=5 mm (Q_r 37.5 mL/s, g_N 1 cm) = {UN*100:.1f} cm/s; at 1 cm {UN/2*100:.1f}; at 2 cm {UN/4*100:.1f} cm/s")

P("\n=== 14. post-stop growth and crescent area loss")
psi_tot = 0.5*phi_j*15e-6*0.6   # gas in transit ~ (t_r..t_K) window of supply
P(f"gas in near-field reservoir at stop ~ phi_j Q * 0.6 s * 0.5 = {psi_tot*1e6:.2f} mL; if 0.3 floats locally: area {(0.3*psi_tot/(phi_f*hm))*1e4:.2f} cm2")
Ablk = math.pi*(0.165*D)**2
for w, L in ((0.006, 0.012), (0.01, 0.012)):
    P(f"block area {Ablk*1e4:.2f} cm2; plowed channel {w*1e3:.0f} mm x {L*1e3:.0f} mm = {w*L*1e4:.2f} cm2 = {w*L/Ablk*100:.0f} %")

P("\n=== 15. Marangoni closure of fresh slots")
for ratio in (0.2, 0.5):
    for w0 in (1.8e-3, 3.6e-3):
        P(f"slot w0={w0*1e3:.1f} mm closure Gamma_fresh/Gamma={ratio}: {w0*ratio*1e3:.2f} mm = {w0*ratio/D:.4f} D")
