import math, numpy as np
g=9.81; D=0.08; R=0.04; rho=1005.; mu=1e-3; nu=mu/rho; sigma=0.045
phi_f=0.70; rho_f=(1-phi_f)*rho; gp=g*(1-rho_f/rho); phiJ=0.64; nRZ=4.65
R32=45e-6; vSt=2*rho*g*R32**2/(9*mu)
vb=lambda p: vSt*(1-p)**nRZ
hstar=0.30e-3; bstar=phiJ*hstar
tau=2.0; k=math.sqrt(3)
hm=2*k*tau/(rho_f*gp)
print(f"vSt={vSt*1e3:.2f} mm/s  h_m(axi)={hm*1e3:.2f} mm  b*={bstar*1e3:.3f} mm  raft at nucleation={bstar/phi_f*1e3:.3f} mm (< h*={hstar*1e3})")

print("\n== 1. Bulk-gas resurfacing: time for canvas Kynch layer to reach b* (uniform phi_B)")
for pB in (0.005,0.01,0.02,0.03,0.05):
    rate=phiJ*pB*vb(pB)/(phiJ-pB)   # Kynch accumulation of gas per area
    print(f"phi_B={pB:.3f}: j_B={pB*vb(pB)*1e3:.3f} mm/s  db/dt={rate*1e3:.3f} mm/s  t(b*)={bstar/rate:.1f} s ; clear-zone reaches top after H/v_b={0.045/vb(pB):.0f} s")
# bulk gas accumulation during drawing
for Fw in (0.1,0.25):
    for Q in (10e-6,15e-6):
        Vcup=200e-6
        dphidt=(1-Fw)*0.25*Q/Vcup
        print(f"F_white={Fw} Q={Q*1e6:.0f} mL/s: bulk phi_B rises {dphidt:.4f}/s -> after 3 s {3*dphidt:.3f}, after 10 s {10*dphidt:.3f}")
# with linearly rising phi_B (well mixed), time to accumulate b*
for Fw in (0.1,0.25):
    Q=15e-6; a=(1-Fw)*0.25*Q/200e-6
    t=0; b=0; dt=1e-3
    while b<bstar and t<60:
        pB=a*t; b+=phiJ*pB*vb(pB)/(phiJ-pB)*dt; t+=dt
    print(f"well-mixed bulk, F_white={Fw}, Q=15: whole canvas reaches b* after {t:.1f} s of pouring (phi_B then {a*t:.3f})")

print("\n== 2. optical threshold scaling: l*~3.5 d_b, z_e~0.7 l*")
for Rb in (35e-6,45e-6,60e-6):
    ze=0.7*3.5*2*Rb
    print(f"R32={Rb*1e6:.0f} um: z_e~{ze*1e3:.2f} mm")
print("t_K ratio for R ratio 1.7: R^-2 ->", 1.7**2, " with h*~z_e~R -> ", 1.7)

print("\n== 3. supply with phi_N ~ r_e/r (needed for slow nucleation front)")
re=3e-3
for rN in (0.02,0.025):
    unif=math.pi*rN**2
    dec=2*math.pi*re*(rN-re)
    print(f"r_N={rN*100:.1f} cm: captured flux ratio (phi_N~1/r vs uniform) = {dec/unif:.2f}")
Fw_needed=3.8e-4*phi_f*3.35e-3/(0.25*15e-6)
print(f"F_white needed for 3.8 cm2/s: {Fw_needed:.2f}")
for fl in (0.083,0.125):
    print(f"Hazen f_loc={fl}: x0.21 -> {fl*0.21:.3f}; dA/dt={fl*0.21*0.25*15e-6/(phi_f*3.35e-3)*1e4:.2f} cm2/s")

print("\n== 4. Gap/brown line inside r_N: Kynch nucleation of the gap canvas")
for pN in (0.10,0.06):
    print(f"uniform phi_N={pN}: t_K={hstar*(phiJ-pN)/(pN*vb(pN)):.2f} s")
# Lagrangian parcel moving outward with u=S/(2 pi r), phi_N = 0.1 re/r
for S in (2.6e-4,3.8e-4):
  for r0 in (4e-3,8e-3,12e-3):
    r=r0; b=0; t=0; dt=1e-3
    while b<bstar and t<10:
        pN=min(0.1,0.1*re/r); b+=phiJ*pN*vb(pN)/(phiJ-pN)*dt; r=math.sqrt(r*r+S*dt/math.pi); t+=dt
    print(f"S={S*1e4:.1f} cm2/s parcel from r0={r0*1e3:.0f} mm (phi_N~1/r): reaches b* at t={t:.2f} s, r={r*1e3:.1f} mm")

print("\n== 5. rigid old block vs material points (Neumann Green fn)")
def vel(x, xi, Sa):
    d=x-xi; r2=np.dot(d,d)
    xin=xi*R**2/np.dot(xi,xi); di=x-xin; ri2=np.dot(di,di)
    return Sa*((d/r2+di/ri2)/(2*np.pi)-x/(2*np.pi*R**2))
xi=np.array([0.3*D,0.0]); a_blk=0.165*D
for Sa in (2.6e-4,3.8e-4):
  for s0 in (0.12*D,0.15*D,0.20*D):
    c=xi-np.array([s0+a_blk,0.0]); pm=xi-np.array([s0,0.0]); t=0; dt=0.002; A0=0.5e-4; out=[]
    while t<2.0001:
        Rn=math.sqrt((A0+Sa*t)/math.pi)
        if min(abs(t-0),abs(t-1),abs(t-2))<1e-6:
            gr=(np.linalg.norm(c-xi)-a_blk-Rn)/D; gm=(np.linalg.norm(pm-xi)-Rn)/D
            out.append(f"t={t:.0f}: rigid {gr:.3f} / material {gm:.3f}")
        # mean-value theorem: rigid disc moves with undisturbed velocity at its centre
        k1=vel(c,xi,Sa); k2=vel(c+0.5*dt*k1,xi,Sa); c=c+dt*k2
        k1=vel(pm,xi,Sa); k2=vel(pm+0.5*dt*k1,xi,Sa); pm=pm+dt*k2
        t+=dt
    print(f"S={Sa*1e4:.1f} s0={s0/D:.2f}D gap: "+" | ".join(out))

print("\n== 6. Canvas screening length for expansion, l_a = sqrt(E tau_a / zeta)")
zeta=0.57
for E in (0.005,0.01,0.03,0.06):
    print("E=%.0f mN/m: "%(E*1e3)+", ".join(f"tau_a={ta:g}s -> {math.sqrt(E*ta/zeta)*1e3:.1f} mm" for ta in (1e-4,1e-3,3e-3,3e-2)))

print("\n== 7. raft over-thickness when pi relaxes, and crushing of young rafts")
def hpi(P):
    a=0.5*rho_f*gp; b=-k*tau; return (-b+math.sqrt(b*b+4*a*P))/(2*a)
for P in (0.003,0.005,0.006,0.010,0.015):
    h=hpi(P); print(f"pi={P*1e3:.0f} mN/m: h_pi={h*1e3:.2f} mm, area gain when pi->0: {h/hm-1:.0%}")
def Py(h,kk=2.0): return kk*tau*h+0.5*rho_f*gp*h*h
for P in (0.005,0.010,0.015):
    a=0.5*rho_f*gp; b=2*tau; hc=(-b+math.sqrt(b*b+4*a*P))/(2*a)
    print(f"pi_bar={P*1e3:.0f} mN/m crushes rafts thinner than {hc*1e3:.2f} mm (k=2); area shrink of a fresh 0.27 mm raft: x{0.274e-3/hc:.2f}")
print("canvas strain from final white fraction (cup area +10..20% from taper):")
for fw in (0.23,0.42):
    for gain in (0.10,0.20):
        Afin=1+gain; canv=Afin*(1-fw)
        print(f"  f_w={fw} area gain {gain}: film strain {1/canv-1:+.2f} -> pi={max(0,1/canv-1)*30:.1f} mN/m (E=30)")

print("\n== 8. Fountain return sheath vs r_e")
for name,z_m,H,r0 in (("low",0.053,0.045,2.72e-3),("cut",0.057,0.045,0.9e-3)):
    zz=min(z_m,H)
    print(f"{name}: sheath 0.1-0.2 min(z_m,H) = {0.1*zz*1e3:.1f}-{0.2*zz*1e3:.1f} mm; r_e=(0.5-1)d={r0*1e3:.1f}-{2*r0*1e3:.1f} mm; r_c=(0.3-0.5)d={0.6*r0*1e3:.1f}-{r0*1e3:.1f} mm vs r0={r0*1e3:.1f}")

print("\n== 9. Form drag on raft draft from reservoir outflow")
df=rho_f/rho*3.35e-3
for U in (0.03,0.06,0.12):
    for frac in (0.3,1.0):
        F=0.5*rho*(frac*U)**2*df
        print(f"U_N={U*100:.0f} cm/s (x{frac} at draft depth): form-drag line force {F*1e3:.2f} mN/m; vs yield 2 tau h={2*tau*3.35e-3*1e3:.1f}")
# translation speed of a 1 cm raft disc in the isobaric canvas under net force F*L
for F in (0.5e-3,2e-3):
    L=0.01; A=math.pi*(L/2)**2
    print(f"edge force {F*1e3} mN/m on half-perimeter of 1 cm disc -> net {F*L*1e6:.1f} uN -> U~F/(2 zeta A)={F*L/(2*zeta*A)*100:.1f} cm/s")

print("\n== 10. stripe smearing by delay spread")
f=2.75
for st in (0.03,0.06,0.1,0.2):
    print(f"sigma_t={st}s: modulation factor exp(-(2 pi f sigma)^2/2) = {math.exp(-(2*math.pi*f*st)**2/2):.3f}")
print("N10 check wiggle: 4af =", 4*0.008*2.75, "m/s vs U_N 0.03-0.12")

print("\n== 11. hole/notch: E8 v_c and consistency")
for H in (0.03,0.045):
    Q=15e-6; vc=(H*math.sqrt(0.25*g)/(2.46*(Q/math.pi)**0.25))**(4/3)
    zc=(vc**2-0.35**2)/(2*g)
    print(f"H={H*100} cm: v_c={vc:.2f} m/s -> drop height {zc*100:.2f} cm (but low pours at 1-2 cm are already z_m/H>1 and still whiten)")

print("\n== 12. stem width")
for rsub in (1.6e-3,2.0e-3):
    print(f"2 r_sub={2*rsub*1e3:.1f} mm = {2*rsub/D:.3f} D ; sink convergence over t_cl: F/(2 pi L)*t = {0.4e-4/(2*math.pi*0.01)*0.3*1e3:.2f} mm")

print("\n== 13. neck: reservoir width vs line")
print("thin stream r_N=0.5 min(z_m,H) ~ 2.2 cm -> incorporation spread over chord ~4 cm of body edge")
