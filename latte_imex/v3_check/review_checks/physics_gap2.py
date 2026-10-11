import math, numpy as np
D=0.08; R=0.04
def vel(x, xi, Sa):
    d=x-xi; r2=np.dot(d,d)
    xin=xi*R**2/np.dot(xi,xi); di=x-xin; ri2=np.dot(di,di)
    return Sa*((d/r2+di/ri2)/(2*np.pi)-x/(2*np.pi*R**2))
xi=np.array([0.3*D,0.0]); a_blk=0.165*D
for Sa in (2.6e-4,3.8e-4):
  for s0 in (0.15*D,0.20*D):
    c=xi-np.array([s0+a_blk,0.0]); pm=xi-np.array([s0,0.0]); pf=xi-np.array([s0+2*a_blk,0.0]); t=0; dt=0.002; A0=0.5e-4; out=[]
    while t<4.0001:
        Rn=math.sqrt((A0+Sa*t)/math.pi)
        if min(abs(t-k) for k in (0,1,2,3,4))<1e-6:
            gr=(np.linalg.norm(c-xi)-a_blk-Rn)/D; gm=(np.linalg.norm(pm-xi)-Rn)/D
            out.append(f"t={t:.0f}: rigid {gr:.3f} / material {gm:.3f} (far-edge-near-edge material {np.linalg.norm(pf-pm)/D:.3f})")
        for p in ('c','pm','pf'):
            v=locals()[p]
            k1=vel(v,xi,Sa); k2=vel(v+0.5*dt*k1,xi,Sa); v=v+dt*k2
            if p=='c': c=v
            elif p=='pm': pm=v
            else: pf=v
        t+=dt
    print(f"S={Sa*1e4:.1f} s0={s0/D:.2f}D: "+" | ".join(out))
