import math
rho_l=1005; mu=1e-3; g=9.81; nRZ=4.65; phiJ=0.64; hs=0.30e-3; phif=0.70
def vSt(R): return 2*rho_l*g*R**2/(9*mu)
def vb(phi,R=45e-6): return vSt(R)*(1-phi)**nRZ
def tK(phi,R=45e-6,h=hs): return h*(phiJ-phi)/(phi*vb(phi,R))
print("== (1) bulk dispersed gas: time for canvas Kynch layer to reach h_* (own formula K3 + nucleation) ==")
for R in (45e-6,20e-6,10e-6):
    for phiB in (0.002,0.005,0.01,0.02,0.05):
        print(f"R={R*1e6:.0f}um phi_B={phiB:.3f}: v_b={vb(phiB,R)*1e3:.2f} mm/s  t_nuc={tK(phiB,R):.1f} s")
# gas routed to bulk during a pour
A=math.pi*0.04**2; H=0.045; Vl=A*H
for Fw in (0.1,0.2,0.3):
    for T in (5,10):
        Q=15e-6; phij=0.25
        G=(1-Fw)*phij*Q*T
        print(f"F_white={Fw} pour {T}s: G_B={G*1e6:.1f} mL, phi_B(mixed)={G/Vl:.3f}, gas column G/A={G/A*1e3:.2f} mm vs b_*={phiJ*hs*1e3:.3f} mm")
print("== (2) post-stop creep when canvas pressure relaxes: raft built at h_pi spreads to h_m ==")
rhof=(1-phif)*rho_l; gp=g*(1-rhof/rho_l)
for k in (math.sqrt(3),2.0):
    tau=2.0
    hm=2*k*tau/(rhof*gp)
    for dpi in (0.005,0.010,0.015):
        hpi=(k*tau+math.sqrt(k*k*tau*tau+2*rhof*gp*dpi))/(rhof*gp)
        print(f"k={k:.3f} pi-S0={dpi*1e3:.0f} mN/m: h_pi={hpi*1e3:.2f} mm, h_m={hm*1e3:.2f} mm, area gain after relaxation = {hpi/hm-1:.0%}")
print("canvas strain needed: white fraction f_w=0.23..0.42 -> canvas area loss f_w/(1)~; eps_c=dPi_c/E=",0.010/0.030)
print("== (3) E13 scaling if h_* = n bubble layers (prop. to R32) ==")
for R in (35e-6,60e-6):
    h=hs*R/45e-6
    print(f"R={R*1e6:.0f}: t_K(h_* fixed)={tK(0.1,R):.2f}s, t_K(h_* ~ R)={tK(0.1,R,h):.2f}s")
print("ratio 35/60 fixed h_*:", tK(0.1,35e-6)/tK(0.1,60e-6), " h_*~R:", tK(0.1,35e-6,hs*35/45)/tK(0.1,60e-6,hs*60/45))
print("== (4) C3 sensitivity: raft drift if zeta_R != zeta_C ==")
for UN in (0.03,0.06,0.12):
    for mis in (0.1,0.3):
        print(f"U_N={UN*100:.0f} cm/s mismatch {mis:.0%}: drift ~ {mis*UN*100:.1f} cm/s = {mis*UN/0.08:.2f} D/s")
