"""PX prototype (PX.md section 3 + 6): jet-attached supply + area-source kinematics, Eulerian VOF version.

State: one transported field c (raft = white area fraction) on a 128^2 Cartesian grid that contains the
unit-diameter disc (D = 8 cm, staircase mask, Neumann walls).  Passive diagnostics: an "old raft" colour o
(carried by the raft fluxes) and Lagrangian markers.

Equations (PX.md 3.3-3.6, cylinder, no tilt => dA_cup/dt = 0):
  u = grad(phi), lap(phi) = sigma_c + sigma_f + sigma_e - edot (1-c),  d_n phi = 0,
  edot = int(sigma_c+sigma_f+sigma_e) / int(1-c)
  d_t c + div(c u) = sigma_f + c sigma_c - Rdot_core
  sigma_c = -F K_c                                  (sink, all phases, current jet)
  sigma_f = [G w Q/Gamma + (1-f_d) F cbar_s] K_f      (supply + low re-float, current jet)
  sigma_e = F_tr(t - t_r) K_e                          (fresh surface re-emitted on the delayed ring)
  F_tr    = F (1 - cbar_s) + f_d F cbar_s
  Rdot_core: raft -> canvas inside the capsule swept by |x - x_j| < r_d  (area conserving)
Numerics: 5-point Neumann Poisson factorised once (SuperLU), MAC face velocities, Weymouth-Yue split
PLIC VOF (Youngs normals, exact strip areas), conservative source update, explicit ledgers.
Units SI.
"""
import math, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage

D = 0.08
R = D / 2
GACC = 9.81

FIXED = dict(c_t=2.0, delta_c=0.5e-3, phi_j=0.25, v0=0.35, rho_l=1005.0, p_sub=10.0,
             Q_star=8e-6, De_ratio=0.6, geom='px', De_abs=None, emit_canvas=False, kappa=0.0, sleeve=1.5,
             # v3-min-2 options (defaults = v3-min-1 behaviour; V2 dict below = new defaults)
             De_rule='ratio', alpha_j=8.0, H=0.045, obst=None, Ke_col=False, sink_disp=True, w_rule='sub',
             bounded=False)
V2 = dict(geom='pxx', emit_canvas=True, De_rule='formula', alpha_j=8.0, obst='re_low', Ke_col=True, Ke_mode='annulus',
          sink_disp=False, w_rule='sub', bounded=True)
NOMINAL = dict(Gamma=0.0395, tau_w=1.4, delta_e=1.0e-3, Pi_coh=300.0, alpha_F=0.009)
PNAMES = ['Gamma', 'tau_w', 'delta_e', 'Pi_coh', 'alpha_F']


def v_from_zs(zs, v0=FIXED['v0']):
    return math.sqrt(v0 ** 2 + 2 * GACC * zs)


def jet_derived(Q, v, P):
    """PX.md 3.3.  Q [m^3/s], v impact speed [m/s]."""
    r0 = math.sqrt(Q / (math.pi * v))
    p0 = 0.5 * (1 - P['phi_j']) * P['rho_l'] * v ** 2
    Tf = v / (P['phi_j'] * GACC)
    tr = P['c_t'] * Tf
    F = P['alpha_F'] * 2 * math.pi * r0 * v
    rc = r0 + P['delta_c']
    re = r0 + P['delta_e']
    zm = 2.46 * v ** 0.75 * (Q / math.pi) ** 0.25 / math.sqrt(P['phi_j'] * GACC)
    if P.get('De_abs') is not None:
        De = P['De_abs']
    elif P.get('De_rule') == 'formula':
        De = min(0.5 * min(zm, P['H']) * math.tan(math.radians(P['alpha_j'])), 0.95 * (re + rc))
    else:
        De = P['De_ratio'] * re
    lnp = math.log(p0 / P['Pi_coh'])
    rd = r0 * math.sqrt(max(0.0, lnp))
    rsub = r0 * math.sqrt(max(0.0, math.log(p0 / P['p_sub'])))
    fd = min(1.0, max(0.0, lnp / math.log(p0 / P['p_sub'])))           # pressed-in existing foam (v3.1 3.9)
    fdJ = fd if P.get('w_rule', 'sub') == 'sub' else min(1.0, max(0.0, lnp))   # jet's own supply (closure C-w)
    ob = P.get('obst')
    if ob == 're_low':      # v3-min-2 L11': exclusion zone = inner region r < r_e, only while L11 holds (p0 <= Pi_coh)
        rp = max(re, rc) if lnp <= 0 else 0.0
    else:
        rp = {None: 0.0, 'rc': rc, 'r0': r0, 'rsub': max(rsub, r0), 're': max(re, rc)}[ob]
    return dict(Q=Q, v=v, r0=r0, p0=p0, Tf=Tf, tr=tr, F=F, rc=rc, re=re, De=De, rd=rd, fd=fd, w=1 - fdJ, fdJ=fdJ,
                rsub=rsub, rp=rp, zm=zm)


# ----------------------------------------------------------------------------------------------- PLIC
def plic_alpha(c, m1, m2):
    """line constant in the reflected unit cell, m1+m2 = 1, raft = {m1 x + m2 y <= alpha}."""
    mn = np.minimum(m1, m2)
    mx = np.maximum(m1, m2)
    c1 = mn / (2 * mx)
    a1 = np.sqrt(np.maximum(2 * mn * mx * c, 0))
    a3 = 1 - np.sqrt(np.maximum(2 * mn * mx * (1 - c), 0))
    a2 = c * mx + 0.5 * mn
    return np.where(c <= c1, a1, np.where(c >= 1 - c1, a3, a2))


def area_frac(al, a, b):
    """fraction of the unit square with a*xi + b*eta <= al (a, b >= 0)."""
    s = a + b
    mn = np.minimum(a, b)
    mx = np.maximum(a, b)
    small = mn < 1e-6 * np.maximum(mx, 1e-300)
    with np.errstate(divide='ignore', invalid='ignore'):
        Rf = lambda x: np.maximum(x, 0.0)
        gen = (Rf(al) ** 2 - Rf(al - a) ** 2 - Rf(al - b) ** 2 + Rf(al - s) ** 2) / (2 * a * b)
        one_d = (al - 0.5 * mn) / mx
    out = np.where(small, one_d, gen)
    out = np.where(mx <= 0, (al >= 0).astype(float), out)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)


class Grid:
    def __init__(self, N=128):
        self.N = N
        self.h = D / N
        x = (np.arange(N) + 0.5) * self.h - R
        self.x1 = x
        self.X, self.Y = np.meshgrid(x, x, indexing='ij')
        self.inside = (self.X ** 2 + self.Y ** 2) < R ** 2
        self.ncell = int(self.inside.sum())
        self.cellA = self.h ** 2
        self.A_cup = self.ncell * self.cellA
        idx = -np.ones((N, N), int)
        idx[self.inside] = np.arange(self.ncell)
        self.idx = idx
        t0 = time.perf_counter()
        I, J = np.nonzero(self.inside)
        rows, cols, vals = [], [], []
        diag = np.zeros(self.ncell)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            I2, J2 = I + di, J + dj
            ok = (I2 >= 0) & (I2 < N) & (J2 >= 0) & (J2 < N)
            ok[ok] &= self.inside[I2[ok], J2[ok]]
            a = idx[I[ok], J[ok]]
            b = idx[I2[ok], J2[ok]]
            rows.append(a); cols.append(b); vals.append(np.ones(len(a)))
            np.add.at(diag, a, -1.0)
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(self.ncell, self.ncell)) + sp.diags(diag)
        A = A.tolil()
        self.pin = self.ncell // 2
        A[self.pin, :] = 0
        A[self.pin, self.pin] = 1.0
        self.lu = spla.splu(A.tocsc())
        self.t_factor = time.perf_counter() - t0
        # face masks
        self.mx = np.zeros((N + 1, N), bool)
        self.mx[1:-1, :] = self.inside[1:, :] & self.inside[:-1, :]
        self.my = np.zeros((N, N + 1), bool)
        self.my[:, 1:-1] = self.inside[:, 1:] & self.inside[:, :-1]
        # nearest-inside extension for normals
        _, inds = ndimage.distance_transform_edt(~self.inside, return_indices=True)
        self.ext_i, self.ext_j = inds[0], inds[1]

    def solve(self, rhs):
        b = rhs[self.inside] * self.h ** 2
        imb = b.sum()
        b[self.pin] = 0.0
        p = self.lu.solve(b)
        phi = np.zeros((self.N, self.N))
        phi[self.inside] = p
        return phi, imb

    def face_vel(self, phi):
        N, h = self.N, self.h
        ux = np.zeros((N + 1, N))
        uy = np.zeros((N, N + 1))
        ux[1:-1, :] = (phi[1:, :] - phi[:-1, :]) / h
        uy[:, 1:-1] = (phi[:, 1:] - phi[:, :-1]) / h
        ux[~self.mx] = 0.0
        uy[~self.my] = 0.0
        return ux, uy

    def div(self, ux, uy):
        return ((ux[1:, :] - ux[:-1, :]) + (uy[:, 1:] - uy[:, :-1])) / self.h

    def interp(self, ux, uy, x, y):
        N, h = self.N, self.h
        fx = (x + R) / h
        fy = (y + R) / h - 0.5
        i0 = np.clip(np.floor(fx).astype(int), 0, N - 1)
        j0 = np.clip(np.floor(fy).astype(int), 0, N - 2)
        tx = np.clip(fx - i0, 0, 1); ty = np.clip(fy - j0, 0, 1)
        vx = (ux[i0, j0] * (1 - tx) * (1 - ty) + ux[i0 + 1, j0] * tx * (1 - ty)
              + ux[i0, j0 + 1] * (1 - tx) * ty + ux[i0 + 1, j0 + 1] * tx * ty)
        fx = (x + R) / h - 0.5
        fy = (y + R) / h
        i0 = np.clip(np.floor(fx).astype(int), 0, N - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, N - 1)
        tx = np.clip(fx - i0, 0, 1); ty = np.clip(fy - j0, 0, 1)
        vy = (uy[i0, j0] * (1 - tx) * (1 - ty) + uy[i0 + 1, j0] * tx * (1 - ty)
              + uy[i0, j0 + 1] * (1 - tx) * ty + uy[i0 + 1, j0 + 1] * tx * ty)
        return vx, vy

    def normals(self, c):
        ce = c[self.ext_i, self.ext_j]
        cp = np.pad(ce, 1, mode='edge')
        gx = (cp[2:, 2:] + 2 * cp[2:, 1:-1] + cp[2:, :-2]) - (cp[:-2, 2:] + 2 * cp[:-2, 1:-1] + cp[:-2, :-2])
        gy = (cp[2:, 2:] + 2 * cp[1:-1, 2:] + cp[:-2, 2:]) - (cp[2:, :-2] + 2 * cp[1:-1, :-2] + cp[:-2, :-2])
        nx, ny = -gx, -gy
        s = np.abs(nx) + np.abs(ny)
        zero = s < 1e-14
        nx = np.where(zero, 1.0, nx); s = np.where(zero, 1.0, s)
        return nx, ny, s


def _flux_axis0(c, nx, ny, s_norm, sx):
    """raft area (cell units) crossing faces normal to axis 0; sx Courant on faces (N+1, M)."""
    m1 = np.abs(nx) / s_norm
    m2 = np.abs(ny) / s_norm
    cc = np.clip(c, 0, 1)
    al = plic_alpha(cc, m1, m2)
    fl = nx < 0
    s = sx[1:-1, :]
    pos = s > 0
    def pick(A):
        return np.where(pos, A[:-1, :], A[1:, :])
    cd, m1d, m2d, ald, fld = pick(cc), pick(m1), pick(m2), pick(al), pick(fl)
    sa = np.abs(s)
    # reflected lower bound of the strip
    a_lo = np.where(pos, np.where(fld, 0.0, 1 - sa), np.where(fld, 1 - sa, 0.0))
    fr = area_frac(ald - m1d * a_lo, m1d * sa, m2d)
    eps = 1e-12
    fr = np.where(cd <= eps, 0.0, np.where(cd >= 1 - eps, 1.0, fr))
    F = np.zeros_like(sx)
    F[1:-1, :] = np.sign(s) * sa * fr
    return F


def flux_x(c, nx, ny, s, sx):
    return _flux_axis0(c, nx, ny, s, sx)


def flux_y(c, nx, ny, s, sy):
    return _flux_axis0(c.T, ny.T, nx.T, s.T, sy.T).T


def bounded_redistribute(c, inside, nband=3):
    """Conservative, provably bounded correction: overshoot (c-1)+ and undershoot (-c)+ are removed and their
    signed net is put back into cells that have room (or raft), first in a band of nband cells around the
    offending cells, else cup-wide; each receiving cell gets at most its own room, so 0 <= c <= 1 exactly
    (up to round-off) and sum(c) is unchanged.  Returns (c, moved_cells, residual_cells)."""
    over = np.maximum(c - 1.0, 0.0); under = np.maximum(-c, 0.0)
    tot_o, tot_u = float(over.sum()), float(under.sum())
    if tot_o == 0.0 and tot_u == 0.0:
        return c, 0.0, 0.0
    cc = np.clip(c, 0.0, 1.0) * inside
    net = tot_o - tot_u
    bad = (over > 0) | (under > 0)
    band = ndimage.binary_dilation(bad, iterations=nband) & inside
    if net > 0:
        cap = (1.0 - cc) * inside
    else:
        cap = cc.copy()
    w = cap * band
    if w.sum() < abs(net) * (1 + 1e-9):
        w = cap
    if abs(net) > 0 and w.sum() > 0:
        cc = cc + np.sign(net) * abs(net) * w / w.sum()
    resid = float(c.sum() - cc.sum())
    return cc, tot_o + tot_u, resid


# --------------------------------------------------------------------------------------------- model
class PXSim:
    def __init__(self, N=128, params=None, fixed=None, dt_max=0.02, cfl=0.5, src_cfl=0.2, nsub=4,
                 grid=None, clip=True):
        self.g = grid if grid is not None else Grid(N)
        self.P = dict(FIXED)
        if fixed: self.P.update(fixed)
        self.P.update(NOMINAL)
        if params: self.P.update(params)
        self.dt_max, self.cfl, self.src_cfl = dt_max, cfl, src_cfl
        self.nsub = nsub
        o = (np.arange(nsub) + 0.5) / nsub - 0.5
        self.sub_dx, self.sub_dy = [a.ravel() for a in np.meshgrid(o, o, indexing='ij')]
        N = self.g.N
        self.c = np.zeros((N, N))
        self.o = np.zeros((N, N))
        self.t = 0.0
        self.packets = []      # [t_start_release, dt_window, area, cx, cy, re]
        self.order = 0
        self.clip = clip
        self.L = dict(V_in=0.0, A_r0=0.0, disp_sink=0.0, core=0.0, clip_c=0.0, clip_o=0.0,
                      Ftr_in=0.0, emitted=0.0, sink_all=0.0, refloat=0.0, supply_area=0.0)
        self.stats = dict(max_c=0.0, min_c=0.0, max_courant=0.0, max_srcdt=0.0, max_divres=0.0,
                          steps=0, t_poisson=0.0, t_vof=0.0, t_src=0.0, t_wall=0.0, max_imb=0.0)
        self.markers = None
        self.ux = np.zeros((N + 1, N)); self.uy = np.zeros((N, N + 1))
        self.last_jet = None

    # ---------------------------------------------------------------- set-up helpers
    def disc_fraction(self, cx, cy, rad, ns=8):
        g = self.g
        o = (np.arange(ns) + 0.5) / ns - 0.5
        frac = np.zeros((g.N, g.N))
        for a in o:
            for b in o:
                frac += ((g.X + a * g.h - cx) ** 2 + (g.Y + b * g.h - cy) ** 2 < rad ** 2)
        return frac / ns ** 2 * g.inside

    def add_disc(self, cx, cy, rad, old=True):
        f = self.disc_fraction(cx, cy, rad)
        self.c = np.maximum(self.c, f)
        if old:
            self.o = np.maximum(self.o, f)
        self.L['A_r0'] = self.c.sum() * self.g.cellA

    # ---------------------------------------------------------------- kernels
    def _window(self, cx, cy, rmax):
        g = self.g
        i0 = max(0, int(math.floor((cx - rmax + R) / g.h)) - 1)
        i1 = min(g.N, int(math.floor((cx + rmax + R) / g.h)) + 2)
        j0 = max(0, int(math.floor((cy - rmax + R) / g.h)) - 1)
        j1 = min(g.N, int(math.floor((cy + rmax + R) / g.h)) + 2)
        return i0, i1, j0, j1

    def kernel(self, cx, cy, kind, a, b):
        """returns (i0,i1,j0,j1,K) normalised so that sum K h^2 = 1 over inside cells.
        kind 'ann': indicator a<r<b; kind 'ring': quartic bump centred a, half-width b."""
        g = self.g
        rmax = b if kind == 'ann' else a + b
        i0, i1, j0, j1 = self._window(cx, cy, rmax)
        if i1 <= i0 or j1 <= j0:
            return None
        Xw = g.X[i0:i1, j0:j1][..., None] + self.sub_dx * g.h
        Yw = g.Y[i0:i1, j0:j1][..., None] + self.sub_dy * g.h
        r = np.hypot(Xw - cx, Yw - cy)
        if kind == 'ann':
            val = ((r > a) & (r < b)).astype(float)
        else:
            val = np.clip(1 - ((r - a) / b) ** 2, 0, None) ** 2
        K = val.mean(axis=2) * g.inside[i0:i1, j0:j1]
        s = K.sum() * g.cellA
        if s <= 0:   # whole support outside / unresolved: nearest inside cell
            ii = int(np.clip((cx + R) / g.h, 0, g.N - 1)); jj = int(np.clip((cy + R) / g.h, 0, g.N - 1))
            ii, jj = g.ext_i[ii, jj], g.ext_j[ii, jj]
            K = np.zeros((1, 1)); K[0, 0] = 1.0 / g.cellA
            return ii, ii + 1, jj, jj + 1, K
        return i0, i1, j0, j1, K / s


    def kernel_min_supply(self, xj, yj, xf, yf, re, sleeve):
        """supply region = quartic ring [re,1.5re] about the jet foot x_j, restricted to the return-sleeve
        disc |x-x_f| <= sleeve*re (x_f = x_j + De e).  Normalised to 1 inside the cup."""
        g = self.g
        rmax = 1.5 * re
        i0, i1, j0, j1 = self._window(xj, yj, rmax)
        Xw = g.X[i0:i1, j0:j1][..., None] + self.sub_dx * g.h
        Yw = g.Y[i0:i1, j0:j1][..., None] + self.sub_dy * g.h
        r = np.hypot(Xw - xj, Yw - yj)
        val = np.clip(1 - ((r - 1.25 * re) / (0.25 * re)) ** 2, 0, None) ** 2
        val = val * (np.hypot(Xw - xf, Yw - yf) <= sleeve * re)
        K = val.mean(axis=2) * g.inside[i0:i1, j0:j1]
        s_ = K.sum() * g.cellA
        if s_ <= 0:
            return self.kernel(xj, yj, 'ring', 1.25 * re, 0.25 * re)
        return i0, i1, j0, j1, K / s_


    def kernel_ring_excl(self, xf, yf, a, b, xj, yj, rex):
        """quartic ring (centre a, half-width b) about x_f, with the jet-foot disc |x-x_j|<rex removed; renormalised."""
        g = self.g
        i0, i1, j0, j1 = self._window(xf, yf, a + b)
        Xw = g.X[i0:i1, j0:j1][..., None] + self.sub_dx * g.h
        Yw = g.Y[i0:i1, j0:j1][..., None] + self.sub_dy * g.h
        r = np.hypot(Xw - xf, Yw - yf)
        val = np.clip(1 - ((r - a) / b) ** 2, 0, None) ** 2
        val = val * (np.hypot(Xw - xj, Yw - yj) >= rex)
        K = val.mean(axis=2) * g.inside[i0:i1, j0:j1]
        s_ = K.sum() * g.cellA
        if s_ <= 0:
            return self.kernel(xf, yf, 'ring', a, b)
        return i0, i1, j0, j1, K / s_

    @staticmethod
    def _add(field, kw, coef):
        i0, i1, j0, j1, K = kw
        field[i0:i1, j0:j1] += coef * K

    @staticmethod
    def _int(field, kw, cellA):
        i0, i1, j0, j1, K = kw
        return float((field[i0:i1, j0:j1] * K).sum() * cellA)

    # ---------------------------------------------------------------- core dispersal
    def core_remove(self, segs, rd):
        """raft -> canvas inside the union of capsules (segments, radius rd); PLIC sub-sampling 8x8."""
        if rd <= 0 or not segs:
            return 0.0
        g = self.g
        pts = np.array([s[0] for s in segs] + [segs[-1][1]])
        xmin, ymin = pts.min(0) - rd; xmax, ymax = pts.max(0) + rd
        i0 = max(0, int((xmin + R) / g.h) - 1); i1 = min(g.N, int((xmax + R) / g.h) + 2)
        j0 = max(0, int((ymin + R) / g.h) - 1); j1 = min(g.N, int((ymax + R) / g.h) + 2)
        ns = 8
        o = (np.arange(ns) + 0.5) / ns
        ox, oy = [a.ravel() for a in np.meshgrid(o, o, indexing='ij')]
        cw = np.clip(self.c[i0:i1, j0:j1], 0, 1)
        if cw.max() <= 1e-12:
            return 0.0
        nx, ny, s = self.g.normals(np.clip(self.c, 0, 1))
        nx, ny, s = nx[i0:i1, j0:j1], ny[i0:i1, j0:j1], s[i0:i1, j0:j1]
        m1 = np.abs(nx) / s; m2 = np.abs(ny) / s
        al = plic_alpha(cw, m1, m2)
        # sub-sample local coords (unit cell, original orientation)
        xl = ox[None, None, :]; yl = oy[None, None, :]
        xt = np.where((nx < 0)[..., None], 1 - xl, xl)
        yt = np.where((ny < 0)[..., None], 1 - yl, yl)
        raft = (m1[..., None] * xt + m2[..., None] * yt) <= al[..., None]
        raft |= (cw >= 1 - 1e-12)[..., None]
        raft &= (cw > 1e-12)[..., None]
        Xs = (g.X[i0:i1, j0:j1] - 0.5 * g.h)[..., None] + xl * g.h
        Ys = (g.Y[i0:i1, j0:j1] - 0.5 * g.h)[..., None] + yl * g.h
        incap = np.zeros(Xs.shape, bool)
        for (p, q) in segs:
            px, py = p; qx, qy = q
            dx, dy = qx - px, qy - py
            L2 = dx * dx + dy * dy
            if L2 > 0:
                tt = np.clip(((Xs - px) * dx + (Ys - py) * dy) / L2, 0, 1)
            else:
                tt = 0.0
            incap |= (Xs - px - tt * dx) ** 2 + (Ys - py - tt * dy) ** 2 < rd * rd
        nr = raft.sum(2)
        hit = (raft & incap).sum(2)
        frac_raft = np.where(nr > 0, hit / np.maximum(nr, 1), incap.mean(2))
        rem = cw * frac_raft * g.inside[i0:i1, j0:j1]
        cold = self.c[i0:i1, j0:j1]
        ratio_o = np.where(cold > 1e-12, self.o[i0:i1, j0:j1] / np.maximum(cold, 1e-300), 0.0)
        self.c[i0:i1, j0:j1] = cold - rem
        self.o[i0:i1, j0:j1] -= rem * ratio_o
        return float(rem.sum() * g.cellA)

    # ---------------------------------------------------------------- one step
    def step(self, inp, dt_try, t_bp, tk_fn):
        g, P = self.g, self.P
        t0w = time.perf_counter()
        t0 = self.t
        dt = min(dt_try, self.dt_max, t_bp - t0) if t_bp > t0 + 1e-12 else min(dt_try, self.dt_max)
        # ---- v3-min-2: footprint press-in (v3.1 M5, t_cl -> 0): raft inside |x - x_j| < r_p is pressed in
        #      (relabelled canvas, area conserving) and re-floated on K_f in this same step.
        pressed = 0.0
        st0 = inp(t0)
        if P.get('obst') and st0['Q'] > 0:
            J0 = jet_derived(st0['Q'], st0['v'], P)
            rp = J0['rp']
            if rp > 0:
                i0, i1, j0, j1 = self._window(st0['x'], st0['y'], rp)
                Xw = g.X[i0:i1, j0:j1][..., None] + self.sub_dx * g.h
                Yw = g.Y[i0:i1, j0:j1][..., None] + self.sub_dy * g.h
                fp = (np.hypot(Xw - st0['x'], Yw - st0['y']) < rp).mean(axis=2) * g.inside[i0:i1, j0:j1]
                cw = self.c[i0:i1, j0:j1]
                cn = np.minimum(cw, 1.0 - fp)
                rem = (cw - cn)
                if rem.sum() > 0:
                    ow = self.o[i0:i1, j0:j1]
                    ro = np.where(cw > 1e-12, ow / np.maximum(cw, 1e-300), 0.0)
                    self.o[i0:i1, j0:j1] = ow - rem * ro
                    self.pressed_o = float((rem * ro).sum() * g.cellA)
                    self.c[i0:i1, j0:j1] = cn
                    pressed = float(rem.sum() * g.cellA)
        self.L['pressed'] = self.L.get('pressed', 0.0) + pressed
        pressed_o = getattr(self, 'pressed_o', 0.0) if pressed > 0 else 0.0
        self.pressed_o = 0.0
        c0 = self.c.copy()
        o0 = self.o.copy()
        for attempt in range(8):
            ts = time.perf_counter()
            # sub-positions along the path
            probe = [inp(t0 + dt * k / 8) for k in range(9)]
            Lpath = sum(math.hypot(probe[k + 1]['x'] - probe[k]['x'], probe[k + 1]['y'] - probe[k]['y'])
                        for k in range(8))
            M = max(1, int(math.ceil(Lpath / (0.5 * g.h))))
            sig_c = np.zeros((g.N, g.N)); sig_f = np.zeros((g.N, g.N)); sig_e = np.zeros((g.N, g.N))
            sig_fo = np.zeros((g.N, g.N))
            new_packets = []
            rec = dict(S=0.0, refl=0.0, disp=0.0, sink=0.0, Ftr=0.0, cbar=0.0, M=M)
            core_pts = []
            rd_max = 0.0
            tk = tk_fn(t0)
            for k in range(M):
                ta, tb = t0 + dt * k / M, t0 + dt * (k + 1) / M
                st = inp(0.5 * (ta + tb))
                Q = st['Q']
                if Q <= 0:
                    continue
                J = jet_derived(Q, st['v'], P)
                xj, yj, ex, ey = st['x'], st['y'], st['ex'], st['ey']
                t_open = tk + P['tau_w']
                gfrac = min(1.0, max(0.0, (tb - t_open) / (tb - ta)))
                Kc = self.kernel(xj, yj, 'ann', J['r0'], J['rc'])
                xf, yf = xj + J['De'] * ex, yj + J['De'] * ey
                if P['geom'] == 'min':
                    Kf = self.kernel_min_supply(xj, yj, xf, yf, J['re'], P['sleeve'])
                elif P['geom'] in ('pxx', 'pxo'):
                    Kf = self.kernel_ring_excl(xf, yf, 1.25 * J['re'], 0.25 * J['re'], xj, yj, max(J['rc'], J['rp']))
                else:
                    Kf = self.kernel(xf, yf, 'ring', 1.25 * J['re'], 0.25 * J['re'])
                cbar = self._int(c0, Kc, g.cellA)
                obar = self._int(o0, Kc, g.cellA)
                F = J['F']
                S = gfrac * J['w'] * Q / P['Gamma']
                fds = J['fd'] if P.get('sink_disp', True) else 0.0
                self._add(sig_c, Kc, -F / M)
                self._add(sig_f, Kf, (S + (1 - fds) * F * cbar + pressed / dt) / M)
                self._add(sig_fo, Kf, ((1 - fds) * F * obar + pressed_o / dt) / M)
                Ftr = F * (1 - cbar) + fds * F * cbar + P['kappa'] * F
                if P['geom'] == 'min':
                    new_packets.append([ta + J['tr'], tb - ta, Ftr * (tb - ta), xj, yj, J['re'], J['rc']])
                elif P['geom'] == 'pxo':
                    new_packets.append([ta + J['tr'], tb - ta, Ftr * (tb - ta), xf, yf, J['re'], 'out'])
                elif P.get('Ke_mode') == 'annulus':
                    new_packets.append([ta + J['tr'], tb - ta, Ftr * (tb - ta), xj, yj, J['re'], ('ann', J['r0'])])
                else:
                    new_packets.append([ta + J['tr'], tb - ta, Ftr * (tb - ta), xf, yf, J['re'], None])
                rec['S'] += S * (tb - ta); rec['refl'] += (1 - fds) * F * cbar * (tb - ta)
                rec['disp'] += fds * F * cbar * (tb - ta); rec['sink'] += F * (tb - ta)
                rec['Ftr'] += Ftr * (tb - ta); rec['cbar'] = cbar
                rec['V_in'] = rec.get('V_in', 0.0) + gfrac * J['w'] * Q * (tb - ta)
                self.last_jet = J
                if J['rd'] > 0:
                    rd_max = max(rd_max, J['rd'])
            # core path: positions at the sub-interval boundaries (only while pouring)
            if rd_max > 0:
                pts = [(inp(t0 + dt * k / M)['x'], inp(t0 + dt * k / M)['y']) for k in range(M + 1)]
                core_pts = [(pts[k], pts[k + 1]) for k in range(M)]
            # delayed emission from packets overlapping [t0, t0+dt]
            emitted = 0.0
            lost = 0.0
            stn = inp(t0 + 0.5 * dt)
            Jn = jet_derived(stn['Q'], stn['v'], P) if (P.get('Ke_col') and stn['Q'] > 0) else None
            for pk in self.packets:
                a0, w, area, xc, yc, re, rcx = pk
                ov = min(t0 + dt, a0 + w) - max(t0, a0)
                if ov <= 0:
                    continue
                amt = area * ov / w
                if isinstance(rcx, tuple) and rcx[0] == 'ann':
                    a_, b_ = 0.5 * (rcx[1] + re), max(0.5 * (re - rcx[1]), 0.5 * g.h)
                    if Jn is not None:
                        Ke = self.kernel_ring_excl(xc, yc, a_, b_, stn['x'], stn['y'], Jn['r0'])
                    else:
                        Ke = self.kernel(xc, yc, 'ring', a_, b_)
                elif rcx == 'out':
                    Ke = self.kernel(xc, yc, 'ring', 1.75 * re, 0.25 * re)
                elif rcx is not None:
                    Ke = self.kernel(xc, yc, 'ring', 0.5 * (re + rcx), max(0.5 * (re - rcx), 0.25 * g.h))
                elif Jn is not None:
                    Ke = self.kernel_ring_excl(xc, yc, 0.75 * re, 0.25 * re, stn['x'], stn['y'], Jn['r0'])
                else:
                    Ke = self.kernel(xc, yc, 'ring', 0.75 * re, 0.25 * re)
                if P['emit_canvas']:
                    i0_, i1_, j0_, j1_, K_ = Ke
                    cw_ = np.clip(c0[i0_:i1_, j0_:j1_], 0, 1)
                    frac_c = float((K_ * (1 - cw_)).sum() * g.cellA)
                    sig_e[i0_:i1_, j0_:j1_] += (amt / dt) * K_ * (1 - cw_)
                    lost += amt * (1 - frac_c)
                    emitted += amt * frac_c
                else:
                    self._add(sig_e, Ke, amt / dt)
                    emitted += amt
            sig = sig_c + sig_f + sig_e
            canv = (1 - c0) * g.inside
            Acan = canv.sum() * g.cellA
            edot = sig.sum() * g.cellA / Acan
            rhs = (sig - edot * canv) * g.inside
            self.stats['t_src'] += time.perf_counter() - ts
            tp = time.perf_counter()
            phi, imb = g.solve(rhs)
            ux, uy = g.face_vel(phi)
            self.stats['t_poisson'] += time.perf_counter() - tp
            umax = max(np.abs(ux).max(), np.abs(uy).max())
            smax = np.abs(sig).max() + abs(edot)
            cour = umax * dt / g.h
            sdt = smax * dt
            if (cour <= self.cfl * 1.0000001 and sdt <= self.src_cfl * 1.0000001) or attempt == 7:
                break
            dt = 0.95 * min(self.cfl * g.h / max(umax, 1e-30), self.src_cfl / max(smax, 1e-30), dt)
        # ---- accept the step
        self.stats['max_courant'] = max(self.stats['max_courant'], cour)
        self.stats['max_srcdt'] = max(self.stats['max_srcdt'], sdt)
        self.stats['max_imb'] = max(self.stats['max_imb'], abs(imb))
        dres = np.abs(g.div(ux, uy) - rhs)[g.inside].max() * dt
        self.stats['max_divres'] = max(self.stats['max_divres'], dres)
        self.packets.extend(new_packets)
        self.packets = [p for p in self.packets if p[0] + p[1] > t0 + dt + 1e-12]
        self.L['V_in'] += rec.get('V_in', 0.0)
        self.L['supply_area'] += rec['S']
        self.L['refloat'] += rec['refl']
        self.L['disp_sink'] += rec['disp']
        self.L['sink_all'] += rec['sink']
        self.L['Ftr_in'] += rec['Ftr']
        self.L['emitted'] += emitted
        self.L['lost_under_raft'] = self.L.get('lost_under_raft', 0.0) + lost
        # ---- VOF transport (Weymouth-Yue split, alternating order) + conservative sources
        tv = time.perf_counter()
        sx = ux * dt / g.h
        sy = uy * dt / g.h
        cc = (c0 > 0.5).astype(float)
        ratio_o = np.where(c0 > 1e-12, o0 / np.maximum(c0, 1e-300), 0.0)
        if self.order == 0:
            nx, ny, s = g.normals(c0)
            Fx = flux_x(c0, nx, ny, s, sx)
            c1 = c0 - (Fx[1:, :] - Fx[:-1, :]) + cc * (sx[1:, :] - sx[:-1, :])
            c1 = np.clip(c1, 0, 1) * g.inside
            nx, ny, s = g.normals(c1)
            Fy = flux_y(c1, nx, ny, s, sy)
        else:
            nx, ny, s = g.normals(c0)
            Fy = flux_y(c0, nx, ny, s, sy)
            c1 = c0 - (Fy[:, 1:] - Fy[:, :-1]) + cc * (sy[:, 1:] - sy[:, :-1])
            c1 = np.clip(c1, 0, 1) * g.inside
            nx, ny, s = g.normals(c1)
            Fx = flux_x(c1, nx, ny, s, sx)
        self.order ^= 1
        divF = (Fx[1:, :] - Fx[:-1, :]) + (Fy[:, 1:] - Fy[:, :-1])
        c_new = c0 - divF + dt * (sig_f + c0 * sig_c)
        # old-raft colour carried by the raft fluxes (donor ratio from the start of the step)
        def donor_ratio_x(F):
            r = np.zeros_like(F)
            r[1:-1, :] = np.where(F[1:-1, :] > 0, ratio_o[:-1, :], ratio_o[1:, :])
            return r
        def donor_ratio_y(F):
            r = np.zeros_like(F)
            r[:, 1:-1] = np.where(F[:, 1:-1] > 0, ratio_o[:, :-1], ratio_o[:, 1:])
            return r
        Fox = Fx * donor_ratio_x(Fx)
        Foy = Fy * donor_ratio_y(Fy)
        o_new = o0 - ((Fox[1:, :] - Fox[:-1, :]) + (Foy[:, 1:] - Foy[:, :-1])) + dt * (sig_fo + o0 * sig_c)
        c_new *= g.inside
        o_new *= g.inside
        self.stats['max_c'] = max(self.stats['max_c'], float(c_new.max()))
        self.stats['min_c'] = min(self.stats['min_c'], float(c_new.min()))
        if P.get('bounded'):
            c_new, moved, resid = bounded_redistribute(c_new, g.inside)
            self.L['redist'] = self.L.get('redist', 0.0) + moved * g.cellA
            cl = np.clip(c_new, 0, 1)
            self.L['clip_c'] += float((cl - c_new).sum() * g.cellA)   # residual net clip (round-off only)
            c_new = cl
        elif self.clip:
            cl = np.clip(c_new, 0, 1)
            self.L['clip_c'] += float((cl - c_new).sum() * g.cellA)   # area ADDED by clipping
            c_new = cl
        ol = np.clip(o_new, 0, c_new)
        self.L['clip_o'] += float(np.abs(ol - o_new).sum() * g.cellA)
        self.c, self.o = c_new, ol
        self.stats['t_vof'] += time.perf_counter() - tv
        # ---- dispersal core (area-conserving phase change)
        if core_pts:
            self.L['core'] += self.core_remove(core_pts, rd_max)
        # ---- markers (RK2 midpoint, frozen face field)
        if self.markers is not None:
            mx_, my_ = self.markers[:, 0], self.markers[:, 1]
            vx, vy = g.interp(ux, uy, mx_, my_)
            vx, vy = g.interp(ux, uy, mx_ + 0.5 * dt * vx, my_ + 0.5 * dt * vy)
            self.markers[:, 0] += dt * vx
            self.markers[:, 1] += dt * vy
        self.ux, self.uy = ux, uy
        self.edot = edot
        self.t = t0 + dt
        self.stats['steps'] += 1
        self.stats['t_wall'] += time.perf_counter() - t0w
        # next trial dt
        return 0.9 * min(self.cfl * g.h / max(umax, 1e-30), self.src_cfl / max(smax, 1e-30), self.dt_max)

    # ---------------------------------------------------------------- ledgers
    def ledgers(self):
        g, L = self.g, self.L
        A_r = self.c.sum() * g.cellA
        G = self.P['Gamma']
        # foam (white-capable milk-foam volume): V_in = Gamma*(A_r - A_r0) + Gamma*(dispersed) - Gamma*clip_added
        lhs = L['V_in']
        rhs = G * (A_r - L['A_r0']) + G * (L['disp_sink'] + L['core']) - G * L['clip_c']
        intransit = sum(p[2] for p in self.packets if True) - 0.0
        return dict(A_r=A_r, foam_err=(lhs - rhs), foam_rel=(lhs - rhs) / max(abs(lhs), 1e-30),
                    surf_err=L['Ftr_in'] - L['emitted'] - L.get('lost_under_raft', 0.0) - self.intransit(),
                    clip_c=L['clip_c'], clip_o=L['clip_o'])

    def intransit(self):
        t = self.t
        tot = 0.0
        for a0, w, area, *_r in self.packets:
            rem = (a0 + w - max(t, a0)) / w
            tot += area * min(1.0, max(0.0, rem))
        return tot

    # ---------------------------------------------------------------- driver
    def run(self, inp, t_end, tk_fn, breakpoints=(), callback=None, cb_times=()):
        bps = sorted(set([b for b in list(breakpoints) + list(cb_times) + [t_end] if b > self.t + 1e-12]))
        cbs = sorted(cb_times)
        ci = 0
        dt_try = self.dt_max
        while self.t < t_end - 1e-10:
            nb = next((b for b in bps if b > self.t + 1e-10), t_end)
            dt_try = self.step(inp, dt_try, nb, tk_fn)
            while ci < len(cbs) and self.t >= cbs[ci] - 1e-9:
                if callback: callback(self, cbs[ci])
                ci += 1
        return self


# ------------------------------------------------------------------------------------ measurements
def white_area(sim, thr=None):
    g = sim.g
    if thr is None:
        return float(sim.c.sum() * g.cellA)
    return float((sim.c >= thr).sum() * g.cellA)


def sample_bilinear(field, g, x, y):
    fx = (x + R) / g.h - 0.5
    fy = (y + R) / g.h - 0.5
    i0 = np.clip(np.floor(fx).astype(int), 0, g.N - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, g.N - 2)
    tx = np.clip(fx - i0, 0, 1); ty = np.clip(fy - j0, 0, 1)
    return (field[i0, j0] * (1 - tx) * (1 - ty) + field[i0 + 1, j0] * tx * (1 - ty)
            + field[i0, j0 + 1] * (1 - tx) * ty + field[i0 + 1, j0 + 1] * tx * ty)


def interface_width_cells(c, g):
    """mixed-cell count (0.05<c<0.95) per interface length, in cells (1 = perfectly sharp, grid aligned)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mixed = ((c > 0.05) & (c < 0.95) & g.inside).sum()
    fig = plt.figure()
    cs = plt.contour(g.x1, g.x1, c.T, levels=[0.5])
    L = 0.0
    for seg in cs.allsegs[0]:
        if len(seg) > 1:
            L += np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1])).sum()
    plt.close(fig)
    return mixed * g.h / max(L, 1e-30), L
