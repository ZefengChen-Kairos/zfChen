"""viz29 scene: everything that moves in the three-view videos of one pour, per render frame, from the simulation chain
(pure numpy; nothing hand-animated).  Conventions: viz29/scout/CONVENTIONS.md.

Frame F (the cup frame of latte_imex.rig.real_program): metres, origin = registered disc centre ON the liquid surface,
x = board x, y = -board y, z = up (gravity -z).  The liquid surface is z = 0 at every instant.  One clock: t [s] of
real_program.cup_poses (t = 0 at the first sample with both a pitcher pose and a disc; = the solver clock = the
video_obs clock).

Inputs (one npz written by viz/forward_fields.py; or a development fields file + a separate inlet capture):
  C (n, N, N) uint8 (or float in [0, 1]) white fraction m / l of the solver at times t (n,); mask (N, N)
  S_* the solver schedule (x, y landing [solver units], Q, ux, uy, U, d, r1, r2, phi, vjx, vjy, active, fdt, q)
  P_* the measured pose series on the 1 ms grid (t, tip (n, 3), yaw, tilt, roll [rad, CAD ZYX], axis_tilt_deg, on_obs)
  E_* the pour model's emission history (ForwardPitcher.run, after MI.replay when the inlet is 'replay')
  R_* the arrival record at the surface (ForwardPitcher.arrivals: t bin ends, Q, x_hit, V_cup, V_P, ...)
  G_* the pitcher grid geometry (pitcher_kona700.json, mm) and LAW (json: exit 'wall' | 'lip', capillary, rho, sigma)
  meta (json): L_unit_m (= R_disc / 0.49), R_disc_m, pour, inlet, closure, ...

Scene(fields, inlet=None, stl=None).frame(t) -> dict (all in F, metres, SI):
  pitcher   R (3, 3), t (3,): X_F = R @ v + t for the STL vertices v of pitcher_kona700.stl in METRES (STL is mm);
            R_body, t_body for the body frame of the grid geometry (spout at phi = 0, z from the inside bottom);
            tip (3,), yaw, tilt, roll, axis_tilt_deg
  milk      V_P [m^3], z_f (free-surface height, F), cap (k, 3) polygon of the free surface inside the pitcher,
            section (j, 3) polygon of the milk in the pitcher's spout plane (half-section cutaway), mesh via milk_mesh(t)
  flow      Q_lip (emission at the lip) [m^3/s], Q_cup (arrivals at the surface), V_out, V_cup, V0, H (lip head),
            U_perp, u_h (arrival speeds), on (stream emitting)
  stream    list of segments, each dict(P (k, 3) centre line from the lip down to the surface, r (k,) radius,
            speed (k,), Q (k,)); foot (3,) where the solver lands the stream (snapped), foot_model, snap (2,)
  cup       R_disc, L, rim / profile of the white latte-cup STAND-IN (not measured beyond the liquid disc), rise
  field     C (N, N) float32 in [0, 1] (linear blend of the two saved frames around t), w, k0, k1
Field array layout (wsolver.Geo): C[i, j] at solver x_j = (j + 0.5) / N - 0.5, y_i = (i + 0.5) / N - 0.5, i.e.
X_F = x L, Y_F = y L; rows = +y_F.  A top-down image (x right, y_F up, = the rectified real top view, row = +board y)
is C[::-1] (render512.native).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

G = 9.81
CUP_R = 0.49                    # Numerics.cup_radius [solver units] <-> R_disc
STL_BOTTOM_M = 0.8e-3           # make_kona700_cad.BOTTOM: grid-json body z = STL z - 0.8 mm (inside bottom)
TIP_KONA = np.array([0.0, 0.0502, 0.0879])          # full_pour.TIP_KONA (tag frame), used by tulip_feas.Pour('kona')
M_CAD = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # real_program.M_CAD (tag <- CAD columns)
S_FLIP = np.diag([1.0, -1.0, -1.0])                  # board -> F (about the disc centre)
CREMA = np.array([150, 96, 62], float)               # render512 latte palette
FOAM = np.array([246, 243, 236], float)
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------------------------------------- basics
def rotation(yaw, tilt, roll):
    """pitcher.rotation: R = Rz(yaw) Ry(tilt) Rx(roll); tilt > 0 brings the spout (body +X) down."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    ct, st = math.cos(tilt), math.sin(tilt)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[ct, 0, st], [0, 1.0, 0], [-st, 0, ct]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def load_stl(fn, scale=1e-3):
    """Binary (or ASCII) STL -> (V (n, 3) float64 [m with scale 1e-3], F (m, 3) int32), duplicate vertices merged."""
    with open(fn, "rb") as f:
        b = f.read()
    n = int(np.frombuffer(b[80:84], "<u4")[0]) if len(b) >= 84 else 0
    if len(b) == 84 + 50 * n:
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        T = np.frombuffer(b[84:84 + 50 * n], dt)["v"].astype(np.float64)
    else:                                                  # ASCII
        vs = [list(map(float, l.split()[1:4])) for l in b.decode("ascii", "ignore").splitlines()
              if l.strip().startswith("vertex")]
        T = np.array(vs, float).reshape(-1, 3, 3)
    P = T.reshape(-1, 3)
    key = np.round(P / 1e-4).astype(np.int64)              # 0.1 um (STL units) weld tolerance
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return P[first] * scale, inv.reshape(-1, 3).astype(np.int32)


def _interp_periodic(phis, tab, phi):
    phi = (np.asarray(phi, float) - phis[0]) % (2 * math.pi) + phis[0]
    xp = np.concatenate([phis, [phis[0] + 2 * math.pi]])
    return np.interp(phi, xp, np.concatenate([tab, [tab[0]]]))


class PitcherGeom:
    """The pitcher cavity of the pour model (GridPitcherGeometry convention, metres): body frame z up from the inside
    bottom, spout at phi = 0 (body +X); r_in (z, phi) inner wall, z_rim / r_rim (phi) the rim; tip = the notch apex
    (argmax r_rim), the anchor placed at the measured spout tip."""

    def __init__(self, G_):
        s = 1e-3 if str(G_.get("units", "mm")) == "mm" else 1.0
        self.zs = np.asarray(G_["zs"], float) * s
        self.phis = np.asarray(G_["phis"], float) - float(G_.get("spout_phi_residual", 0.0))
        self.r_in = np.asarray(G_["r_in"], float) * s
        self.r_out = np.asarray(G_["r_out"], float) * s
        self.z_rim_tab = np.asarray(G_["z_rim"], float) * s
        self.r_rim_tab = np.asarray(G_["r_rim"], float) * s
        k = int(np.argmax(self.r_rim_tab))
        self.tip = np.array([self.r_rim_tab[k] * math.cos(self.phis[k]), self.r_rim_tab[k] * math.sin(self.phis[k]),
                             self.z_rim_tab[k]])
        self._mesh = None

    @classmethod
    def load(cls, fn):
        with open(fn) as f:
            return cls(json.load(f))

    def z_rim(self, phi):
        return _interp_periodic(self.phis, self.z_rim_tab, phi)

    def r_wall(self, z, phi):
        """Inner wall radius at body height z and azimuth phi (bilinear; periodic in phi; clamped in z)."""
        z = np.clip(np.asarray(z, float), self.zs[0], self.zs[-1])
        phi = np.asarray(phi, float)
        fz = np.interp(z, self.zs, np.arange(len(self.zs)))
        i0 = np.clip(np.floor(fz).astype(int), 0, len(self.zs) - 2)
        a = fz - i0
        nph = len(self.phis)
        dph = 2 * math.pi / nph
        fp = ((phi - self.phis[0]) % (2 * math.pi)) / dph
        j0 = np.floor(fp).astype(int) % nph
        b = fp - np.floor(fp)
        j1 = (j0 + 1) % nph
        R = self.r_in
        return ((1 - a) * ((1 - b) * R[i0, j0] + b * R[i0, j1]) + a * ((1 - b) * R[i0 + 1, j0] + b * R[i0 + 1, j1]))

    def inside(self, Pb):
        """Body-frame points (..., 3) inside the cavity (0 <= z <= z_rim(phi), r <= r_in(z, phi))."""
        Pb = np.asarray(Pb, float)
        phi = np.arctan2(Pb[..., 1], Pb[..., 0])
        r = np.hypot(Pb[..., 0], Pb[..., 1])
        z = Pb[..., 2]
        return (z >= 0) & (z <= self.z_rim(phi)) & (r <= self.r_wall(z, phi))

    def cavity_mesh(self, inset=0.0):
        """Closed-bottom, open-top triangle mesh of the cavity (body frame): rings at z = 0, the grid heights below the
        lowest rim point, and the rim (z_rim, r_rim) per azimuth; outward normals (out of the liquid into the wall).
        Returns V (n, 3), F (m, 3), rim ring vertex indices (nphi,)."""
        if self._mesh is not None and self._mesh[0] == inset:
            return self._mesh[1]
        ph = self.phis
        nph = len(ph)
        c, s = np.cos(ph), np.sin(ph)
        zmax = self.z_rim_tab.min() - 1e-4
        d = float(inset)
        rings = [np.stack([(self.r_in[0] - d) * c, (self.r_in[0] - d) * s, np.full(nph, d)], 1)]
        for iz, z in enumerate(self.zs):
            if d < z < zmax:
                rings.append(np.stack([(self.r_in[iz] - d) * c, (self.r_in[iz] - d) * s, np.full(nph, z)], 1))
        rings.append(np.stack([(self.r_rim_tab - d) * c, (self.r_rim_tab - d) * s, self.z_rim_tab], 1))
        V = np.concatenate(rings + [np.array([[0.0, 0.0, d]])])
        nr = len(rings)
        ctr = len(V) - 1
        F = []
        for a in range(nr - 1):
            i0, i1 = a * nph, (a + 1) * nph
            for j in range(nph):
                j1 = (j + 1) % nph
                # outward normal (away from the axis): (ring a, j) -> (a, j1) -> (a+1, j1) is counter-clockwise from inside
                F.append((i0 + j, i1 + j1, i0 + j1))
                F.append((i0 + j, i1 + j, i1 + j1))
        for j in range(nph):                                # bottom fan, normal -z
            F.append((ctr, j, (j + 1) % nph))
        F = np.array(F, np.int64)
        # orientation check: the bottom fan must point down; walls outward
        n0 = np.cross(V[F[-1, 1]] - V[F[-1, 0]], V[F[-1, 2]] - V[F[-1, 0]])
        if n0[2] > 0:
            F[-nph:] = F[-nph:, ::-1]
        w = F[0]
        nw = np.cross(V[w[1]] - V[w[0]], V[w[2]] - V[w[0]])
        if np.dot(nw[:2], V[w[0], :2]) < 0:
            F[:-nph] = F[:-nph, ::-1]
        self._mesh = (inset, (V, F, np.arange((nr - 1) * nph, nr * nph)))
        return self._mesh[1]


# ----------------------------------------------------------------------------------------------------------- clipping
def mesh_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def clip_below(V, F, zc, rim=None):
    """Clip mesh (V, F) to z <= zc.  Returns (Vc, Fc, loops): the kept part (with the new edge points) and the closed
    boundary loops of the cut (each (k, 3), counter-clockwise seen from above).  An open cut chain (the plane leaving
    the cavity through its open top, i.e. over the lip) is closed along the submerged rim ring vertices `rim`."""
    s = V[:, 2] - zc
    ins = s <= 0
    pts = {}

    def ep(a, b):                                          # point on edge (a, b) where s = 0, keyed by the edge
        k = (a, b) if a < b else (b, a)
        if k not in pts:
            f = s[a] / (s[a] - s[b])
            pts[k] = V[a] + f * (V[b] - V[a])
        return k
    keep, newtri, segs = [], [], []
    for t_ in F:
        m = ins[t_]
        n = int(m.sum())
        if n == 3:
            keep.append(t_)
        elif n == 0:
            continue
        else:
            # rotate so the odd vertex is first
            if n == 1:
                r = int(np.argmax(m))
            else:
                r = int(np.argmin(m))
            a, b, c = t_[r], t_[(r + 1) % 3], t_[(r + 2) % 3]
            kab, kac = ep(a, b), ep(a, c)
            if n == 1:                                     # a inside
                newtri.append(("v", a, "e", kab, "e", kac))
            else:                                          # a outside, b, c inside
                newtri.append(("e", kab, "v", b, "v", c))
                newtri.append(("e", kab, "v", c, "e", kac))
            segs.append((kab, kac))
    # assemble
    ekeys = list(pts)
    eidx = {k: len(V) + i for i, k in enumerate(ekeys)}
    Vc = np.concatenate([V, np.array([pts[k] for k in ekeys]).reshape(-1, 3)])
    Ft = [tuple(t_) for t_ in keep]
    for tr in newtri:
        Ft.append(tuple(int(tr[i + 1]) if tr[i] == "v" else eidx[tr[i + 1]] for i in (0, 2, 4)))
    Fc = np.array(Ft, np.int64).reshape(-1, 3)
    # chain the cut segments into loops
    adj = {}
    for i, (a, b) in enumerate(segs):
        adj.setdefault(a, []).append(i)
        adj.setdefault(b, []).append(i)
    used = np.zeros(len(segs), bool)
    chains = []
    ends = [k for k, v in adj.items() if len(v) == 1]
    starts = ends + [k for k in adj if len(adj[k]) != 1]
    for k0 in starts:
        if all(used[i] for i in adj[k0]):
            continue
        chain = [k0]
        k = k0
        while True:
            nxt = [i for i in adj[k] if not used[i]]
            if not nxt:
                break
            i = nxt[0]
            used[i] = True
            a, b = segs[i]
            k = b if a == k else a
            chain.append(k)
            if k == k0:
                break
        chains.append(chain)
    rim = None if rim is None else np.asarray(rim)
    open_chains = [c for c in chains if c[0] != c[-1]]
    closed = [c[:-1] for c in chains if c[0] == c[-1] and len(c) > 3]
    out = []
    for c in closed:
        out.append((_ccw(np.array([pts[k] for k in c])), None))
    nr = 0 if rim is None else len(rim)
    pos = {} if rim is None else {int(v): i for i, v in enumerate(rim)}
    for c in open_chains:
        # the plane leaves the cavity over the lip: the chain (at z = zc) is closed by the chord between its ends, and
        # the submerged rim vertices between its end edges (walked from the end edge to the start edge) form the lip
        # sheet hanging from that chord
        P = np.array([pts[k] for k in c])
        lip = []
        a, b = c[-1]
        if a in pos and b in pos:
            ia, ib = pos[a], pos[b]
            cur = ia if ins[a] else ib
            step = 1 if (cur - (ib if cur == ia else ia)) % nr == 1 else -1
            for _ in range(nr):
                lip.append(V[rim[cur]])
                nx = (cur + step) % nr
                e = (min(rim[cur], rim[nx]), max(rim[cur], rim[nx]))
                if e == c[0] or not ins[rim[nx]]:
                    break
                cur = nx
        if len(P) >= 3:
            out.append((_ccw(P), _ccw(np.vstack([P[-1:], np.array(lip).reshape(-1, 3), P[:1]])) if lip else None))
    return Vc, Fc, out


def _ccw(L):
    """Polygon loop oriented counter-clockwise seen from above (+z)."""
    nz = np.sum(L[:, 0] * np.roll(L[:, 1], -1) - np.roll(L[:, 0], -1) * L[:, 1])
    return L if nz >= 0 else L[::-1]


def fan(loop, centre=None):
    """Triangle fan of a (star-shaped) polygon loop from its centroid: (V (k + 1, 3), F (k, 3)), normal +z for a
    counter-clockwise loop."""
    c = loop.mean(0) if centre is None else centre
    k = len(loop)
    V = np.concatenate([loop, c[None]])
    F = np.array([(k, i, (i + 1) % k) for i in range(k)], np.int64)
    return V, F


def clip_poly_below(P, zc):
    """Sutherland-Hodgman: polygon P (k, 3) clipped to z <= zc."""
    out = []
    k = len(P)
    for i in range(k):
        a, b = P[i], P[(i + 1) % k]
        ia, ib = a[2] <= zc, b[2] <= zc
        if ia:
            out.append(a)
        if ia != ib:
            f = (zc - a[2]) / (b[2] - a[2])
            out.append(a + f * (b - a))
    return np.array(out).reshape(-1, 3)


# ----------------------------------------------------------------------------------------------------------- the jet
def jet_paths(origin, v_lip, e, U0, Q, A_b, W_b, P_b, rho, sigma, capillary, g=G, dt=2.5e-4, t_max=0.6, rec=4):
    """latte_imex.pitcher.jet_flight (P4 of the wall exit) with the same arithmetic, recording every parcel's position
    and speed every `rec` steps (tau = 0, rec dt, 2 rec dt, ...).  Returns tau (K,), P (n, K, 3) float32, S (n, K)
    float32, t_f (n,), x_hit (n, 2) -- t_f / x_hit equal jet_flight's."""
    n = len(U0)
    rho, sig = float(rho), float(sigma)
    if capillary == "none":
        sig = 0.0
    rQ = rho * Q
    a0 = np.sqrt(Q / (math.pi * np.maximum(U0, 1e-6)))
    T0 = sig * (W_b + P_b) if capillary == "sheet" else math.pi * sig * a0
    T0 = np.minimum(T0, rQ * U0)
    M0 = (rQ * U0 - T0)[:, None] * e
    hbar = Q / np.maximum(U0, 1e-6) / np.maximum(W_b, 1e-6)
    uTC = np.sqrt(2 * sig / (rho * hbar)) if sig > 0 else np.full(n, np.inf)
    tcap = np.where(np.isfinite(uTC), 0.5 * W_b / np.maximum(uTC, 1e-9), 0.0)
    Tex0 = T0 - math.pi * sig * a0

    def vrel(t):
        M = M0.copy(); M[:, 2] -= rQ * g * t
        Mm = np.linalg.norm(M, axis=1)
        Mh = np.where(Mm[:, None] > 1e-15, M / np.maximum(Mm, 1e-30)[:, None], np.array([0.0, 0.0, -1.0]))
        Tex = Tex0 * np.exp(-t / np.maximum(tcap, 1e-9))
        sp = (Mm + T0) / rQ
        for _ in range(4):
            T = math.pi * sig * np.sqrt(Q / (math.pi * np.maximum(sp, 1e-6))) + Tex
            sp = (Mm + T) / rQ
        return sp[:, None] * Mh
    p = origin.copy()
    done = p[:, 2] <= 0
    tf = np.where(done, 0.0, np.nan)
    xh = np.where(done[:, None], p[:, :2], np.nan)
    v_prev = v_lip + vrel(0.0)
    Ps, Ss, taus = [p.astype(np.float32)], [np.linalg.norm(v_prev, axis=1).astype(np.float32)], [0.0]
    t, it = 0.0, 0
    while t < t_max and not done.all():
        v_mid = v_lip + vrel(t + 0.5 * dt)
        v_new = v_lip + vrel(t + dt)
        p_new = p + v_mid * dt
        cross = (~done) & (p_new[:, 2] <= 0)
        if cross.any():
            f = p[cross, 2] / np.maximum(p[cross, 2] - p_new[cross, 2], 1e-12)
            tf[cross] = t + f * dt
            xh[cross] = p[cross, :2] + f[:, None] * (p_new[cross, :2] - p[cross, :2])
            done |= cross
        p, v_prev, t, it = p_new, v_new, t + dt, it + 1
        if it % rec == 0:
            Ps.append(p.astype(np.float32)); Ss.append(np.linalg.norm(v_new, axis=1).astype(np.float32)); taus.append(t)
    if it % rec != 0:                                      # the last state (beyond every landing)
        Ps.append(p.astype(np.float32)); Ss.append(np.linalg.norm(v_prev, axis=1).astype(np.float32)); taus.append(t)
    return np.array(taus), np.stack(Ps, 1), np.stack(Ss, 1), tf, xh


# ----------------------------------------------------------------------------------------------------------- scene
def _load(src):
    if src is None:
        return {}
    if isinstance(src, dict):
        return dict(src)
    Z = np.load(src, allow_pickle=False)
    return {k: Z[k] for k in Z.files}


def _json(v):
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    return json.loads(str(v))


def latte_rgb(c):
    """render512.latte: white fraction -> RGB (0-255 float)."""
    c = np.clip(np.asarray(c, float), 0.0, 1.0)[..., None]
    return CREMA * (1 - c) + FOAM * c


class Scene:
    """Per-frame geometry of one pour (module docstring)."""

    def __init__(self, fields, inlet=None, stl=None, geom=None, fps=30.0, q_min=1e-7):
        Z = _load(fields)
        Zi = _load(inlet)
        for k, v in Zi.items():                            # the inlet capture may come separately (dev fields)
            if k.startswith(("P_", "E_", "R_", "G_", "S_")) or k in ("LAW", "meta_inlet"):
                Z.setdefault(k, v)
        self.Z = Z
        self.meta = _json(Z.get("meta"))
        for k, v in _json(Zi.get("meta")).items():         # the inlet capture's meta fills what the fields lack
            self.meta.setdefault(k, v)
        self.law = _json(Z.get("LAW"))
        self.fps = float(fps)
        # ---- clock / fields
        C = Z["C"]
        self.C = C
        self.C_scale = 1.0 / 255.0 if C.dtype == np.uint8 else 1.0
        if "t" in Z:
            self.tC = np.asarray(Z["t"], float)
        else:                                              # development fields (probe run_mi.py): state after frame k
            self.tC = np.cumsum(np.asarray(Z["S_fdt"], float))[np.asarray(Z["k"], int)]
        self.mask = np.asarray(Z["mask"], bool)
        self.N = self.mask.shape[0]
        self.L = float(self.meta.get("L_unit_m") or Z.get("L_unit_m", np.nan))
        if not np.isfinite(self.L):
            self.L = float(Z["P_L"]) if "P_L" in Z else float("nan")
        self.R_disc = float(self.meta.get("R_disc_m", self.L * CUP_R))
        # ---- schedule
        self.S = {k[2:]: np.asarray(v) for k, v in Z.items() if k.startswith("S_")}
        if self.S:
            self.S_t1 = np.cumsum(self.S["fdt"])
            self.S_t0 = self.S_t1 - self.S["fdt"]
        # ---- pose + pour model
        self.P = {k[2:]: np.asarray(v) for k, v in Z.items() if k.startswith("P_")}
        self.E = {k[2:]: np.asarray(v) for k, v in Z.items() if k.startswith("E_")}
        self.R = {k[2:]: np.asarray(v) for k, v in Z.items() if k.startswith("R_")}
        if self.P:
            self._ang = {k: np.unwrap(self.P[k].astype(float)) for k in ("yaw", "tilt", "roll")}
        gj = {k[2:]: Z[k] for k in Z if k.startswith("G_")}
        if gj:
            gj["units"] = str(gj.get("units", "mm"))
            self.geom = PitcherGeom(gj)
        else:
            self.geom = PitcherGeom.load(geom or self._find("pitcher_kona700.json"))
        self.stl_path = stl or self._find("pitcher_kona700.stl", required=False)
        self.q_min = float(q_min)
        self._paths = None

    def use_schedule(self, src):
        """Take the solver schedule (S_*) from another npz / dict (e.g. a fresh inlet capture instead of the old
        schedule stored with development fields)."""
        Z = _load(src)
        self.S = {k[2:]: np.asarray(v) for k, v in Z.items() if k.startswith("S_")}
        self.S_t1 = np.cumsum(self.S["fdt"])
        self.S_t0 = self.S_t1 - self.S["fdt"]

    @staticmethod
    def _find(name, required=True):
        for d in (os.environ.get("VIZ_ASSETS", ""), os.path.join(HERE, "assets"), HERE,
                  os.path.join(os.getcwd(), "latte_imex", "pitcher_geom"),
                  os.path.join(HERE, "..", "..", "mi27", "wt", "latte_imex", "pitcher_geom")):
            if d and os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
        if required:
            raise FileNotFoundError(name)
        return None

    # ------------------------------------------------------------------ clock
    def t_end(self):
        return float(self.tC[-1])

    def clock(self, t0=0.0, t1=None):
        t1 = self.t_end() if t1 is None else t1
        return np.arange(max(t0, 0.0), t1 + 1e-9, 1.0 / self.fps)

    # ------------------------------------------------------------------ pose
    def pose(self, t):
        tt = self.P["t"]
        tip = np.array([np.interp(t, tt, self.P["tip"][:, k]) for k in range(3)])
        yaw, tilt, roll = (float(np.interp(t, tt, self._ang[k])) for k in ("yaw", "tilt", "roll"))
        R = rotation(yaw, tilt, roll)
        t_body = tip - R @ self.geom.tip                 # X_F = R @ X_body + t_body
        t_stl = t_body - R @ np.array([0.0, 0.0, STL_BOTTOM_M])   # X_F = R @ v_stl[m] + t_stl
        ax = float(np.interp(t, tt, self.P["axis_tilt_deg"])) if "axis_tilt_deg" in self.P else float("nan")
        return dict(R=R, t=t_stl, R_body=R, t_body=t_body, tip=tip, yaw=yaw, tilt=tilt, roll=roll, axis_tilt_deg=ax)

    def stl(self):
        """(V [m, STL frame], F) of the CAD pitcher; place with frame(t)['pitcher'] R, t."""
        if self.stl_path is None:
            raise FileNotFoundError("pitcher_kona700.stl (VIZ_ASSETS or viz/assets)")
        if not hasattr(self, "_stl"):
            self._stl = load_stl(self.stl_path)
        return self._stl

    # ------------------------------------------------------------------ milk in the pitcher
    def _e(self, key, t, default=np.nan):
        if key not in self.E:
            return default
        return float(np.interp(t, self.E["t"], self.E[key]))

    def V_P(self, t):
        """Milk left in the pitcher at t.  The ledger's V_P[k] is the state at the END of emission step k
        (V_P = V0 - cumsum(Q_o) dt to 1e-12), so it is interpolated on the step ends E_t + dt/2 -- the clock of
        flow()['V_out'] -- and V_out + V_P = V0 holds at every t."""
        V0 = float(self.E.get("V0", np.nan)) if "V0" in self.E else float(self.meta.get("V0_mL", np.nan)) * 1e-6
        dt = float(self.E.get("dt", 1e-3)) if np.ndim(self.E.get("dt", 1e-3)) == 0 else 1e-3
        return float(np.interp(t, self.E["t"] + 0.5 * dt, self.E["V_P"], left=V0))

    def milk_mesh(self, t, pose=None, z_f=None, inset=0.0):
        """Closed milk body in F: the cavity clipped at the free surface + the planar free-surface cap at z_f (+ the
        lip sheet down to the submerged rim while pouring).  Returns V, F (outward normals), cap loop (k, 3) at z = z_f,
        or (None, None, None) when the pitcher is dry.  inset [m]: walls / bottom moved inwards (e.g. 2e-4 so that the
        milk does not z-fight the STL's inner wall, which is the same surface; costs ~1 % of the volume)."""
        pose = pose or self.pose(t)
        z_f = self._e("z_f", t) if z_f is None else z_f
        V, F, rim = self.geom.cavity_mesh(inset)
        Vf = V @ pose["R_body"].T + pose["t_body"]
        Vc, Fc, loops = clip_below(Vf, F, z_f, rim)
        if not len(Fc):
            return None, None, None
        Vs, Fs = [Vc], [Fc]
        off = len(Vc)
        for cap_, lip in loops:
            for poly, ctr in ((cap_, None), (lip, None if lip is None else 0.5 * (lip[0] + lip[-1]))):
                if poly is None or len(poly) < 3:
                    continue
                fv, ff = fan(poly, ctr)
                Vs.append(fv); Fs.append(ff + off); off += len(fv)
        Vm = np.concatenate(Vs)
        Fm = np.concatenate(Fs)
        # drop unused vertices
        used = np.unique(Fm)
        remap = -np.ones(len(Vm), np.int64); remap[used] = np.arange(len(used))
        cap = max((l_[0] for l_ in loops), key=len) if loops else None
        return Vm[used], remap[Fm], cap

    def milk_section(self, t, pose=None, z_f=None):
        """Milk in the pitcher's spout plane (body XZ plane through the axis, F): polygon (j, 3) of the cavity profile
        (spout side phi = 0 and back side phi = pi) below the free surface -- for a half-section cutaway."""
        pose = pose or self.pose(t)
        z_f = self._e("z_f", t) if z_f is None else z_f
        g = self.geom
        zt = np.r_[0.0, g.zs[g.zs < g.z_rim_tab.min()]]
        rf = g.r_wall(zt, 0.0); rb = g.r_wall(zt, math.pi)
        front = np.c_[rf, np.zeros_like(zt), zt]
        back = np.c_[-rb, np.zeros_like(zt), zt]
        top_f = np.array([[g.r_rim_tab[np.argmin(np.abs(g.phis))], 0.0, float(g.z_rim(0.0))]])
        kb = np.argmin(np.abs(np.abs(g.phis) - math.pi))
        top_b = np.array([[-g.r_rim_tab[kb], 0.0, float(g.z_rim(math.pi))]])
        poly = np.concatenate([front, top_f, top_b, back[::-1]])   # bottom edge closes back(0) -> front(0)
        Pf = poly @ pose["R_body"].T + pose["t_body"]
        return clip_poly_below(Pf, z_f)

    # ------------------------------------------------------------------ stream
    def _prepare_paths(self):
        E = self.E
        Q = E["Q_o"]
        idx = np.nonzero(Q > 0)[0]
        self._idx = idx
        wall = str(self.law.get("exit", "wall" if "kM" in E else "lip")) == "wall" and "v_lip" in E
        if wall:
            ok = np.isfinite(E["v_lip"][idx, 0])
            if not ok.all():                                # wall_miss parcels (never seen): ballistic below
                pass
            tau, Pp, Sp, tf, xh = jet_paths(E["origin"][idx], np.nan_to_num(E["v_lip"][idx]), E["d_o"][idx], E["U"][idx],
                                            Q[idx], np.nan_to_num(E["A_b"][idx]), np.nan_to_num(E["W_b"][idx], nan=1e-3),
                                            np.nan_to_num(E["P_b"][idx], nan=1e-3), self.law.get("rho", 800.0),
                                            self.law.get("sigma", 0.045), self.law.get("capillary", "round"),
                                            g=float(self.law.get("g", G)))
            self._paths = ("wall", tau, Pp, Sp)
            self._tf_law = tf
            self._xh_law = xh
        else:
            o, v = E["origin"][idx], E["v_o"][idx]
            g = float(self.law.get("g", G))
            vz, z0 = v[:, 2], o[:, 2]
            tf = np.where(z0 > 0, (vz + np.sqrt(vz ** 2 + 2 * g * np.maximum(z0, 0.0))) / g, 0.0)
            self._paths = ("ballistic", o, v, g)
            self._tf_law = tf
            self._xh_law = o[:, :2] + v[:, :2] * tf[:, None]
        # replayed flight time (MI-replay changes t_f / t_arr, not the law path): the parcel runs its law path in t_f
        self._tf = E["t_f"][idx] if "t_f" in E else self._tf_law
        self._te = E["t"][idx]
        self._ta = self._te + self._tf

    def _pos(self, j, tau):
        """Positions / speeds of parcels j (indices into self._idx) after law flight time tau."""
        if self._paths[0] == "wall":
            _, taus, Pp, Sp = self._paths
            f = np.clip(tau / (taus[1] - taus[0]), 0, len(taus) - 1 - 1e-9)
            i0 = np.floor(f).astype(int)
            a = (f - i0)[:, None]
            P = (1 - a) * Pp[j, i0] + a * Pp[j, i0 + 1]
            S = (1 - a[:, 0]) * Sp[j, i0] + a[:, 0] * Sp[j, i0 + 1]
            return P.astype(float), S.astype(float)
        _, o, v, g = self._paths
        P = o[j] + v[j] * tau[:, None]
        P[:, 2] -= 0.5 * g * tau ** 2
        V = v[j].copy(); V[:, 2] -= g * tau
        return P, np.linalg.norm(V, axis=1)

    def solver_foot(self, t):
        """Landing point the solver used at t (schedule frame containing t), F metres, or None when inactive."""
        if not self.S:
            return None
        f = int(np.clip(np.searchsorted(self.S_t1, t, side="right"), 0, len(self.S_t1) - 1))
        if not bool(self.S["active"][f]):
            return None
        return np.array([self.S["x"][f] * self.L, self.S["y"][f] * self.L, 0.0])

    def stream(self, t, snap=True):
        """Parcel locus at t (module docstring).  Returns dict(segments, foot, foot_model, snap)."""
        if self._paths is None:
            self._prepare_paths()
        te, tf, ta = self._te, self._tf, self._ta
        fl = np.nonzero((te <= t) & (ta >= t) & (tf > 0))[0]
        out = dict(segments=[], foot=None, foot_model=None, snap=np.zeros(2))
        if not len(fl):
            return out
        tau = (t - te[fl]) / tf[fl] * self._tf_law[fl]
        P, S = self._pos(fl, tau)
        Qp = self.E["Q_o"][self._idx[fl]]
        # segments: runs of consecutive emission steps
        brk = np.nonzero(np.diff(self._idx[fl]) != 1)[0] + 1
        groups = np.split(np.arange(len(fl)), brk)
        segs = []
        for gi in groups:
            gi = gi[::-1]                                  # youngest (at the lip) first
            Pg, Sg, Qg = P[gi], S[gi], Qp[gi]
            # foot: where the stream meets the surface now (interpolated between the oldest flying and the parcel
            # that landed just before it)
            j_old = fl[gi[-1]]
            foot = None
            if j_old > 0 and self._idx[j_old - 1] == self._idx[j_old] - 1 and ta[j_old - 1] < t:
                a = (ta[j_old] - t) / max(ta[j_old] - ta[j_old - 1], 1e-12)
                xh = self._xh_law
                foot = np.r_[xh[j_old] + a * (xh[j_old - 1] - xh[j_old]), 0.0]
                Pg = np.vstack([Pg, foot]); Sg = np.r_[Sg, Sg[-1]]; Qg = np.r_[Qg, Qg[-1]]
            segs.append(dict(P=Pg, speed=Sg, Q=Qg, landed=foot is not None))
        # the solver's landing: snap the landed segment's foot onto it, blended linearly in height to 0 at the top
        landed = [s for s in segs if s["landed"]]
        fs = self.solver_foot(t) if snap else None
        if landed:
            sl = max(landed, key=lambda s_: float(s_["Q"][-1]))   # the segment reaching the surface now (largest flow)
            out["foot_model"] = sl["P"][-1].copy()
            if fs is not None:
                d = fs[:2] - sl["P"][-1, :2]
                out["snap"] = d
                for s in segs:
                    ztop = max(float(s["P"][:, 2].max()), 1e-6)
                    w = np.clip(1.0 - s["P"][:, 2] / ztop, 0.0, 1.0)
                    s["P"] = s["P"].copy(); s["P"][:, :2] += w[:, None] * d[None, :]
            out["foot"] = sl["P"][-1].copy()
        for s in segs:
            s["r"] = np.sqrt(np.maximum(s["Q"], 0.0) / (math.pi * np.maximum(s["speed"], 1e-6)))
        out["segments"] = [s for s in segs if len(s["P"]) >= 2 and float(np.median(s["Q"])) >= self.q_min]
        return out

    # ------------------------------------------------------------------ flow numbers (HUD)
    def flow(self, t):
        E, R = self.E, self.R
        dt = float(E.get("dt", 1e-3)) if np.ndim(E.get("dt", 1e-3)) == 0 else 1e-3
        V0 = float(E["V0"]) if "V0" in E else float(self.meta.get("V0_mL", np.nan)) * 1e-6
        cumQ = np.cumsum(E["Q_o"]) * dt
        out = dict(V0=V0, V_P=self.V_P(t), Q_lip=self._e("Q_o", t, 0.0) if t >= E["t"][0] - 0.5 * dt else 0.0,
                   V_out=float(np.interp(t, E["t"] + 0.5 * dt, cumQ, left=0.0)), H=self._e("H", t, 0.0),
                   z_f=self._e("z_f", t))
        out["on"] = out["Q_lip"] > 0
        if R:
            tb = R["t"]
            out["V_cup"] = float(np.interp(t, tb, R["V_cup"], left=0.0))
            j = int(np.clip(np.searchsorted(tb, t), 0, len(tb) - 1))
            out["Q_cup"] = float(R["Q"][j])
            out["U_perp"] = float(R["U_perp"][j]) if R["Q"][j] > 0 else 0.0
            out["u_h"] = float(np.hypot(*R["u_h"][j])) if R["Q"][j] > 0 else 0.0
        return out

    # ------------------------------------------------------------------ cup (stand-in)
    def cup(self, t):
        """The cup beyond the liquid disc is NOT measured (disc.json: centre, surface plane, radius only).  A plain
        white latte-cup STAND-IN: inner radius R_disc at the liquid surface, cup_register's flare dR/dz 0.0146 near the
        top, rim ASSUMED 3 mm above the surface at the end of the pattern phase; the surface rises in the rigid cup by
        V_cup / (pi R_disc^2) (pour model arrivals), so the cup sits lower in F by rise(t_end) - rise(t) ... expressed
        as rim_z(t) = rim height above the surface at t."""
        R = self.R
        Vc = float(np.interp(t, R["t"], R["V_cup"], left=0.0)) if R else 0.0
        Ve = float(R["V_cup"][-1]) if R else 0.0
        area = math.pi * self.R_disc ** 2
        rise_left = (Ve - Vc) / area
        rim_above = 0.003 + rise_left
        return dict(R_disc=self.R_disc, L=self.L, centre=np.zeros(3), surface_z=0.0, standin=True, flare=0.0146,
                    rim_z=rim_above, rim_r_in=self.R_disc + 0.0146 * rim_above, wall=0.004, depth=0.075,
                    bottom_r_in=0.026, rise_left=rise_left)

    # ------------------------------------------------------------------ field
    def field(self, t):
        tc = self.tC
        if t <= tc[0]:
            k0 = k1 = 0; w = 0.0
        elif t >= tc[-1]:
            k0 = k1 = len(tc) - 1; w = 0.0
        else:
            k1 = int(np.searchsorted(tc, t)); k0 = k1 - 1
            w = float((t - tc[k0]) / max(tc[k1] - tc[k0], 1e-12))
        C0 = self.C[k0].astype(np.float32) * self.C_scale
        C = C0 if w == 0.0 else (1 - w) * C0 + w * self.C[k1].astype(np.float32) * self.C_scale
        return dict(C=C, k0=k0, k1=k1, w=w, t0=float(tc[k0]), t1=float(tc[k1]))

    def filled(self, C):
        """C with the cells outside the cup mask set from their kept neighbours (3 rings), for bilinear sampling up to
        the disc edge (render512.to_H uses the nearest kept cell)."""
        A = np.where(self.mask, C, 0.0).astype(np.float64)
        W = self.mask.astype(np.float64)
        for _ in range(3):
            pa = np.pad(A * W, 1); pw = np.pad(W, 1)
            s = pa[:-2, 1:-1] + pa[2:, 1:-1] + pa[1:-1, :-2] + pa[1:-1, 2:]
            n = pw[:-2, 1:-1] + pw[2:, 1:-1] + pw[1:-1, :-2] + pw[1:-1, 2:]
            new = (W == 0) & (n > 0)
            A = np.where(new, s / np.maximum(n, 1e-9), A)
            W = np.where(new, 1.0, W)
        return A

    def sample(self, C, X, Y, filled=True):
        """Bilinear sample of a solver field at F points (metres)."""
        A = self.filled(C) if filled else C
        N = self.N
        fj = np.clip((np.asarray(X) / self.L + 0.5) * N - 0.5, 0, N - 1 - 1e-9)
        fi = np.clip((np.asarray(Y) / self.L + 0.5) * N - 0.5, 0, N - 1 - 1e-9)
        j0, i0 = np.floor(fj).astype(int), np.floor(fi).astype(int)
        a, b = fj - j0, fi - i0
        return ((1 - b) * ((1 - a) * A[i0, j0] + a * A[i0, j0 + 1]) + b * ((1 - a) * A[i0 + 1, j0] + a * A[i0 + 1, j0 + 1]))

    def top_image(self, C, H=512, ext=1.05):
        """Top-down image of the field in disc coordinates (render512.to_H: pixel centres linspace(-ext, ext, H) disc
        radii, col = +x, row = +board y = -y_F), i.e. the rectified real top view's orientation."""
        s = np.linspace(-ext, ext, H)
        U, V = np.meshgrid(s, s)
        return self.sample(C, U * self.R_disc, -V * self.R_disc)

    # ------------------------------------------------------------------ one frame
    def frame(self, t, milk=True, stream=True, field=True):
        pose = self.pose(t)
        out = dict(t=float(t), pitcher=pose, flow=self.flow(t), cup=self.cup(t))
        if milk:
            z_f = out["flow"]["z_f"]
            Vm, Fm, cap = self.milk_mesh(t, pose, z_f)
            out["milk"] = dict(V_P=out["flow"]["V_P"], z_f=z_f, cap=cap, section=self.milk_section(t, pose, z_f),
                               mesh_volume=(mesh_volume(Vm, Fm) if Vm is not None else 0.0))
            out["_milk_mesh"] = (Vm, Fm)
        if stream:
            out["stream"] = self.stream(t)
        if field:
            out["field"] = self.field(t)
        return out


# ----------------------------------------------------------------------------------------------------------- checks
def check(fields, inlet=None, times=(1.0, 2.5, 4.0, 8.0), stl=None):
    """Self-test: STL anchor, milk volume vs the ledger, jet replica vs the pour model, snap offsets, field blend."""
    sc = Scene(fields, inlet, stl=stl)
    rep = {}
    # 1. the CAD tip vertex lands on the measured tip
    if sc.stl_path:
        V, F = sc.stl()
        g = sc.geom
        tip_stl = g.tip + np.array([0.0, 0.0, STL_BOTTOM_M])
        i = int(np.argmin(np.linalg.norm(V - tip_stl, axis=1)))
        errs = []
        for t in times:
            p = sc.pose(t)
            errs.append(float(np.linalg.norm(p["R"] @ tip_stl + p["t"] - p["tip"])))
        rep["stl_tip_anchor_err_m"] = max(errs)
        rep["stl_nearest_vertex_to_tip_mm"] = float(np.linalg.norm(V[i] - tip_stl) * 1e3)
    # 2. milk volume vs the ledger
    vols = []
    for t in times:
        f = sc.frame(t, stream=False, field=False)
        vols.append((t, f["milk"]["V_P"] * 1e6, f["milk"]["mesh_volume"] * 1e6,
                     None if f["milk"]["cap"] is None else len(f["milk"]["cap"])))
    rep["milk_mL_ledger_vs_mesh_capk"] = vols
    # 3. the jet replica lands where the pour model's parcels land (law)
    sc._prepare_paths()
    E = sc.E
    xh_model = E["x_hit"][sc._idx]
    if "law_t_f" in E:                                     # replay keeps the law's x_hit and records law_t_f
        rep["jet_tf_vs_law_max_s"] = float(np.nanmax(np.abs(sc._tf_law - E["law_t_f"][sc._idx])))
    else:
        rep["jet_tf_vs_model_max_s"] = float(np.nanmax(np.abs(sc._tf_law - E["t_f"][sc._idx])))
    rep["jet_xhit_vs_model_max_mm"] = float(np.nanmax(np.abs(sc._xh_law - xh_model)) * 1e3)
    # 4. snap offsets on a 30 fps clock
    sn, nseg = [], []
    for t in sc.clock():
        s = sc.stream(t)
        if s["foot"] is not None and sc.solver_foot(t) is not None:
            sn.append(np.hypot(*s["snap"]) * 1e3)
        nseg.append(len(s["segments"]))
    sn = np.array(sn)
    rep["snap_mm_p50_p90_max"] = [float(np.percentile(sn, 50)), float(np.percentile(sn, 90)), float(sn.max())] if len(sn) else None
    rep["frames_with_stream"] = int(np.sum(np.array(nseg) > 0))
    rep["frames"] = len(nseg)
    # 5. field blend reproduces the saved frames
    k = min(3, len(sc.tC) - 1)
    f = sc.field(float(sc.tC[k]))
    rep["field_blend_exact"] = bool(np.allclose(f["C"], sc.C[k].astype(np.float32) * sc.C_scale))
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("fields")
    ap.add_argument("--inlet", default=None)
    ap.add_argument("--stl", default=None)
    a = ap.parse_args()
    print(json.dumps(check(a.fields, a.inlet, stl=a.stl), indent=1, default=float))
