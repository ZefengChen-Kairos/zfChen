import math, numpy as np
g=9.81; D=0.08; R=0.04; rho=1005.; mu=1e-3; nu=mu/rho; sigma=0.045
phi_f=0.70; rho_f=(1-phi_f)*rho; gpf=g*(1-rho_f/rho)
print("Bo_b R=45um:", rho*g*(45e-6)**2/sigma)
print("St_b v_b/v:", 2.72e-3/0.65)
for etaf in (1.0,10.0):
    print("Lambda_m (L=2cm,delta=1mm,h=3.35mm) eta_f=",etaf, mu*0.02**2/(1e-3*etaf*3.35e-3))
print("Fr_f U/sqrt(g'h) U=1cm/s:", 0.01/math.sqrt(gpf*3.35e-3))
print("Fr_cup U^2/(gH) U=1cm/s:", 0.01**2/(g*0.045))
print("bulk circulation decay H^2/(pi^2 nu):", 0.045**2/(math.pi**2*nu))
# raft thickness under canvas overpressure: 1/2 rho_f g' h^2 - k tau h = P
for k,name in ((math.sqrt(3),'axi'),(2.0,'plane')):
    for tau in (2.0,):
        for Pbar in (0.0,0.005,0.010,0.015):
            a=0.5*rho_f*gpf; b=-k*tau; c=-Pbar
            h=(-b+math.sqrt(b*b-4*a*c))/(2*a)
            print(f"{name} tau={tau} Pbar={Pbar*1e3:.0f} mN/m: h={h*1e3:.2f} mm")
# footprint radii (Gaussian stagnation profile p0 exp(-r^2/rfp^2))
def radii(p0, d, Pi_coh, h=3.35e-3):
    rfp=d/2
    psub=(rho-rho_f)*g*h
    rsub=rfp*math.sqrt(max(0,math.log(p0/psub)))
    rd=rfp*math.sqrt(max(0,math.log(p0/Pi_coh))) if p0>Pi_coh else 0.0
    return rsub, rd
for name,p0,d in (("low Q15 z1.5",157,5.4e-3),("low Q20 z2",194,6.0e-3),("cut Q3 z7",564,1.8e-3),("cut Q5 z9",712,2.2e-3)):
    for Pi in (200,300,400):
        rs,rd=radii(p0,d,Pi)
        fd=(rd/rs)**2 if rs>0 else 0
        print(f"{name} Pi_coh={Pi}: r_sub={rs*1e3:.2f} mm r_disp={rd*1e3:.2f} mm dispersed fraction={fd:.2f}")
# cluster rise
for a in (1e-3,2e-3,3e-3):
    Ua=math.sqrt(gpf*a)
    print(f"cluster a={a*1e3:.0f} mm U_a={Ua:.3f} m/s; from 1 cm {0.01/Ua:.2f} s, from 4.5 cm {0.045/Ua:.2f} s")
for Vp in (0.3*D,0.6*D):
    for tcl in (0.2,0.4):
        print(f"stem V_p t_cl: V_p={Vp/D:.1f}D/s t_cl={tcl}: {Vp*tcl/D:.3f} D")
# notch from stationary pre-cut dwell: area F*t
for F in (0.25e-4,0.5e-4):
    for t in (0.7,1.2):
        A=F*t; r=math.sqrt(A/math.pi)
        print(f"notch: F={F*1e4:.2f} cm2/s dwell {t}s -> area {A*1e4:.2f} cm2 radius {r*1e3:.1f} mm = {r/D:.3f} D")
# gap closing between growing new raft (area source at B) and old block near edge
def vel(x, xi, Sa):
    d=x-xi; r2=np.sum(d*d,axis=-1)[...,None]
    xin=xi*R**2/np.dot(xi,xi); di=x-xin; ri2=np.sum(di*di,axis=-1)[...,None]
    return Sa*((d/r2+di/ri2)/(2*np.pi)-x/(2*np.pi*R**2))
xi=np.array([0.3*D,0.0])
for Sa in (2.6e-4,3.8e-4):
  for s0 in (0.12*D,0.15*D,0.20*D):
    p=xi-np.array([s0,0.0]); t=0; dt=0.002; A0=0.5e-4  # raft starts with 0.5 cm2 at first white
    out=[]
    while t<2.0001:
        Rn=math.sqrt((A0+Sa*t)/math.pi)
        if abs(t-0.0)<1e-9 or abs(t-1.0)<1e-6 or abs(t-2.0)<1e-6:
            out.append(f"t={t:.0f}s gap={(np.linalg.norm(p-xi)-Rn)/D:.3f}D (block moved {(np.linalg.norm(p-xi)-s0)/D:.3f}D)")
        k1=vel(p[None,:],xi,Sa)[0]; k2=vel((p+0.5*dt*k1)[None,:],xi,Sa)[0]; p=p+dt*k2; t+=dt
    print(f"Sa={Sa*1e4:.1f} cm2/s s0={s0/D:.2f}D: "+"; ".join(out))
# far edge displacement after 2 s for a block 0.33 D wide whose near edge is at s0=0.15 D
for Sa in (3.8e-4,):
    for s in (0.15,0.48):
        p0=xi-np.array([s*D,0.0]); p=p0.copy(); dt=0.002
        for i in range(1000):
            k1=vel(p[None,:],xi,Sa)[0]; k2=vel((p+0.5*dt*k1)[None,:],xi,Sa)[0]; p=p+dt*k2
        print(f"Sa={Sa*1e4:.1f}, 2 s: point at {s} D from B moves {np.linalg.norm(p-p0)/D:.3f} D")
# adsorption time estimate
for c in (5.,15.,30.):
    for Dp in (5e-11,5e-12):
        print(f"protein c={c} kg/m3 D={Dp}: (Gamma/c)^2/D = {(2e-6/c)**2/Dp:.2e} s")
# canvas strain from one stage of raft growth
for dA in (5e-4, 8e-4):
    print(f"one stage adds {dA*1e4:.0f} cm2 raft into ~35 cm2 canvas: strain {dA/35e-4:.2f}")
