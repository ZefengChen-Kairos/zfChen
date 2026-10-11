"""B8 check: coffee flow memory / vorticity / film decay after the pour stops (v2 closure, N=48)."""
import sys, math, numpy as np
sys.path.insert(0, "/home/user/zfChen")
from latte_imex.solver import Solver, Params, Numerics, Inlet
from latte_imex.joint_cma import BASE_V2, X0_V2

P = Params(**dict(BASE_V2, **X0_V2))
num = Numerics(N=48)
sol = Solver(P, num)
g = sol.g
# low pour with horizontal arrival velocity, off-centre
Q = 15e-6; S_eff = Q / P.D_L**3
inl = Inlet(active=True, x_hit=(-0.1, 0.0), S_eff=S_eff, u_in=(0.6, 0.0), U_perp=0.68/P.D_L*1.0, d_jet=0.066, r1=0.04, r2=0.04)
print("S_eff", S_eff, "U_perp[D/s]", inl.U_perp)
def curl(U):
    ux, uy = U
    h = g.h
    duy_dx = np.zeros_like(ux); dux_dy = np.zeros_like(ux)
    duy_dx[:, 1:-1] = (uy[:, 2:] - uy[:, :-2]) / (2*h)
    dux_dy[1:-1, :] = (ux[2:, :] - ux[:-2, :]) / (2*h)
    inner = g.mask.copy(); inner[:, [0, -1]] = False; inner[[0, -1], :] = False
    w = duy_dx - dux_dy
    return w, inner
# pour 0.4 s
while sol.t < 0.4 - 1e-9:
    sol.advance_frame(inl)
w, inner = curl(sol.Ub)
spd = np.hypot(*sol.Ub)
print(f"during pour: Ub is None? {sol.Ub is None}; |Ub|max {spd.max():.3f} D/s, mean {spd[g.mask].mean():.4f}; "
      f"|curl Ub| max {np.abs(w[inner]).max():.3e}, |grad| scale ~ {spd.max()/g.h:.1f}  (ratio {np.abs(w[inner]).max()/(spd.max()/g.h):.1e})")
lsafe = np.where(g.mask, sol.l, 1.0)
u_film = np.hypot(sol.qx, sol.qy) / lsafe
print(f"film |u| max {u_film.max():.3f} D/s, mean {u_film[g.mask].mean():.4f}")
# stop the pour: one frame
off = Inlet(active=False)
hist = []
t0 = sol.t
for k in range(60):
    sol.advance_frame(off)
    lsafe = np.where(g.mask, sol.l, 1.0)
    u_film = np.hypot(sol.qx, sol.qy) / lsafe
    hist.append((sol.t - t0, float(u_film[g.mask].max()), float(u_film[g.mask].mean()), float(sol.l[g.mask].max())))
    if k == 0:
        print(f"first frame after stop: Ub None? {sol.Ub is None}, Ud None? {sol.Ud is None}, divUs None? {sol.divUs is None}, Ds None? {sol.Ds is None}, foam_src None? {sol.foam_src is None}, f_sub remaining {float(sol.f[g.mask].sum()*g.area):.3e}")
for t, umax, umean, lmax in hist[::6]:
    print(f"  t+{t:.3f}s  film |u| max {umax:.4f} mean {umean:.5f}  lmax {lmax:.2f}")
# decay rate fit between frames 1 and 13
ts = np.array([h[0] for h in hist]); um = np.array([h[2] for h in hist])
rate = -np.polyfit(ts[:12], np.log(um[:12]), 1)[0]
print(f"fitted decay rate of mean film speed over first 0.1 s: {rate:.2f} /s  (beta={P.beta}, beta/l for l=1: {P.beta}, for lmax: {P.beta/hist[0][3]:.2f})")
