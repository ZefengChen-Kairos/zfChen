"""viz29 raytrace: a small Warp ray tracer for the three-view latte videos (device 'cuda' on AutoDL, 'cpu' locally).

Scene, all in the cup frame F of viz/scene.py (metres, z up, the liquid surface is z = 0):
  pitcher  static wp.Mesh of the CAD STL in its own frame (metres).  Per frame only (R, t) change and the RAYS are moved
           into the STL frame (x_stl = R^T (x_F - t)) instead of rebuilding the BVH.  Optional translucency: a per-surface
           opacity a0 that rises to 1 at grazing incidence, so the milk inside shows and the silhouette stays readable.
  cup      static wp.Mesh (lathe + handle) of the stand-in mug in its own frame B (scene.cup_profile); per frame the
           rigid pose (R, t) of scene.cup(t) -- the measured cup axis and rim height -- moves the RAYS into B, as for the
           pitcher.  A STAND-IN mug placed on the measured rim (only its shape is not measured).
  dyn      per-frame wp.Mesh: the milk body in the pitcher (scene.milk_mesh, inset) + the stream tube; material per face.
  latte    analytic plane z = 0 inside the cup's inner wall (r_B <= R_disc; a tilted cup's section is an ellipse just
           wider than the disc: the field is clamped radially to the disc there): the white fraction C (filled, bilinear) through crema -> foam (render512
           palette, mixed in sRGB as render512 does), with a gentle bump normal from grad C.
  table    analytic plane under the cup foot (moves with the cup), fading into the background.
Light: one directional key light with shadow rays (any-hit against pitcher / cup / dyn), a hemisphere fill, Blinn-Phong
specular; ss x ss supersampling per pixel; shading in linear light, gamma 2.2 at the end; output uint8 (H, W, 3)."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAT_PITCHER, MAT_CUP, MAT_MILK, MAT_STREAM, MAT_LATTE, MAT_TABLE = 0, 1, 2, 3, 4, 5
CREMA = (150, 96, 62)            # render512 latte palette (sRGB)
FOAM = (246, 243, 236)
CUP_DARK = (30, 27, 29)          # the real cup is a black mug (render512 CUP (28, 24, 26))
CUP_WHITE = (236, 234, 228)
PITCHER_COL = (226, 228, 230)    # the real pitcher is white
MILK_COL = (246, 243, 236)
STREAM_COL = (243, 241, 236)
TABLE_COL = (62, 64, 78)
BG_TOP, BG_BOT = (40, 42, 52), (16, 16, 20)


def _lin(c):
    return wp.vec3(*[(float(x) / 255.0) ** 2.2 for x in c])


def _srgb(c):
    return wp.vec3(*[float(x) / 255.0 for x in c])


@wp.struct
class Cam:
    pos: wp.vec3
    fwd: wp.vec3
    right: wp.vec3
    up: wp.vec3
    tan_y: float          # tan(fov_y / 2) (perspective)
    half_h: float         # half height of the view [m] (orthographic)
    aspect: float
    ortho: int


@wp.struct
class Sc:
    pm: wp.uint64
    pn: wp.array(dtype=wp.vec3)
    pR: wp.mat33
    pt: wp.vec3
    p_on: int
    p_a0: float           # pitcher surface opacity at normal incidence (1 = opaque)
    p_sh: float           # light transmitted through the pitcher for shadows
    p_col: wp.vec3
    cm: wp.uint64
    cn: wp.array(dtype=wp.vec3)
    cR: wp.mat33
    ct: wp.vec3
    c_on: int
    c_col: wp.vec3
    c_ks: float
    dm: wp.uint64
    dn: wp.array(dtype=wp.vec3)
    dmat: wp.array(dtype=wp.int32)
    d_on: int
    milk_col: wp.vec3
    stream_col: wp.vec3
    fld: wp.array2d(dtype=wp.float32)
    N: int
    L: float
    R_disc: float
    disc_on: int
    bump: float
    crema: wp.vec3        # sRGB 0..1 (mixed in sRGB, then linearised)
    foam: wp.vec3
    table_z: float
    table_on: int
    table_col: wp.vec3
    light: wp.vec3        # unit vector towards the key light
    key: float
    amb: float
    bg_top: wp.vec3
    bg_bot: wp.vec3
    ss: int


@wp.struct
class Hit:
    t: float
    mat: int
    n: wp.vec3            # shading normal (F)
    ng: wp.vec3           # geometric normal (F)


@wp.func
def smooth_n(m: wp.uint64, nrm: wp.array(dtype=wp.vec3), f: int, u: float, v: float, ng: wp.vec3):
    i0 = wp.mesh_get_index(m, f * 3 + 0)
    i1 = wp.mesh_get_index(m, f * 3 + 1)
    i2 = wp.mesh_get_index(m, f * 3 + 2)
    n = nrm[i0] * u + nrm[i1] * v + nrm[i2] * (1.0 - u - v)
    ln = wp.length(n)
    out = ng
    if ln > 1.0e-6:
        n = n / ln
        if wp.dot(n, ng) > 0.4:          # crease: keep the face normal where the vertex normals disagree
            out = n
    return out


@wp.func
def trace(sc: Sc, o: wp.vec3, d: wp.vec3, tmax: float):
    h = Hit()
    h.t = tmax
    h.mat = -1
    h.n = wp.vec3(0.0, 0.0, 1.0)
    h.ng = wp.vec3(0.0, 0.0, 1.0)
    if sc.p_on == 1:
        RT = wp.transpose(sc.pR)
        q = wp.mesh_query_ray(sc.pm, RT @ (o - sc.pt), RT @ d, h.t)
        if q.result:
            ng = wp.normalize(q.normal)
            h.t = q.t
            h.mat = 0
            h.ng = sc.pR @ ng
            h.n = sc.pR @ smooth_n(sc.pm, sc.pn, q.face, q.u, q.v, ng)
    if sc.c_on == 1:
        CT = wp.transpose(sc.cR)
        q = wp.mesh_query_ray(sc.cm, CT @ (o - sc.ct), CT @ d, h.t)
        if q.result:
            ng = wp.normalize(q.normal)
            h.t = q.t
            h.mat = 1
            h.ng = sc.cR @ ng
            h.n = sc.cR @ smooth_n(sc.cm, sc.cn, q.face, q.u, q.v, ng)
    if sc.d_on == 1:
        q = wp.mesh_query_ray(sc.dm, o, d, h.t)
        if q.result:
            ng = wp.normalize(q.normal)
            h.t = q.t
            h.mat = sc.dmat[q.face]
            h.ng = ng
            h.n = smooth_n(sc.dm, sc.dn, q.face, q.u, q.v, ng)
    if sc.disc_on == 1 and d[2] < 0.0 and o[2] > 0.0:
        t = -o[2] / d[2]
        if t < h.t:
            p = o + d * t
            rr = sc.R_disc * 1.003
            pc = p
            if sc.c_on == 1:                # inside the (tilted) cup's inner wall
                pc = wp.transpose(sc.cR) @ (p - sc.ct)
            if pc[0] * pc[0] + pc[1] * pc[1] <= rr * rr:
                h.t = t
                h.mat = 4
                h.n = wp.vec3(0.0, 0.0, 1.0)
                h.ng = h.n
    if sc.table_on == 1 and d[2] < 0.0 and o[2] > sc.table_z:
        t = (sc.table_z - o[2]) / d[2]
        if t < h.t:
            h.t = t
            h.mat = 5
            h.n = wp.vec3(0.0, 0.0, 1.0)
            h.ng = h.n
    return h


@wp.func
def field_at(sc: Sc, x: float, y: float):
    nf = float(sc.N)
    fj = wp.clamp((x / sc.L + 0.5) * nf - 0.5, 0.0, nf - 1.001)
    fi = wp.clamp((y / sc.L + 0.5) * nf - 0.5, 0.0, nf - 1.001)
    j0 = int(fj)
    i0 = int(fi)
    a = fj - float(j0)
    b = fi - float(i0)
    j1 = wp.min(j0 + 1, sc.N - 1)
    i1 = wp.min(i0 + 1, sc.N - 1)
    return ((1.0 - b) * ((1.0 - a) * sc.fld[i0, j0] + a * sc.fld[i0, j1])
            + b * ((1.0 - a) * sc.fld[i1, j0] + a * sc.fld[i1, j1]))


@wp.func
def lin3(c: wp.vec3):
    return wp.vec3(wp.pow(wp.max(c[0], 0.0), 2.2), wp.pow(wp.max(c[1], 0.0), 2.2), wp.pow(wp.max(c[2], 0.0), 2.2))


@wp.func
def background(sc: Sc, d: wp.vec3):
    s = wp.clamp(0.5 + 0.8 * d[2], 0.0, 1.0)
    return sc.bg_bot * (1.0 - s) + sc.bg_top * s


@wp.func
def shadow(sc: Sc, p: wp.vec3, L: wp.vec3):
    vis = float(1.0)
    if sc.d_on == 1:
        if wp.mesh_query_ray_anyhit(sc.dm, p, L, 10.0):
            vis = 0.0
    if vis > 0.0 and sc.c_on == 1:
        CT = wp.transpose(sc.cR)
        if wp.mesh_query_ray_anyhit(sc.cm, CT @ (p - sc.ct), CT @ L, 10.0):
            vis = 0.0
    if vis > 0.0 and sc.p_on == 1:
        RT = wp.transpose(sc.pR)
        if wp.mesh_query_ray_anyhit(sc.pm, RT @ (p - sc.pt), RT @ L, 10.0):
            vis = vis * sc.p_sh
    return vis


@wp.func
def shade(sc: Sc, p: wp.vec3, d: wp.vec3, h: Hit):
    n = h.n
    ng = h.ng
    if wp.dot(ng, d) > 0.0:                 # face the viewer
        ng = -ng
        n = -n
    if wp.dot(n, d) > 0.0:
        n = ng
    alb = sc.p_col
    ks = float(0.18)
    shin = float(40.0)
    if h.mat == 1:
        alb = sc.c_col
        ks = sc.c_ks
        shin = 90.0
    elif h.mat == 2:
        alb = sc.milk_col
        ks = 0.12
        shin = 30.0
    elif h.mat == 3:
        alb = sc.stream_col
        ks = 0.35
        shin = 70.0
    elif h.mat == 4:
        px = p[0]
        py = p[1]
        rp = wp.sqrt(px * px + py * py)
        rl = sc.R_disc * 0.995
        if rp > rl:                         # the tilted cup's elliptic section beyond the disc: the disc's edge value
            px = px * rl / rp
            py = py * rl / rp
        c = wp.clamp(field_at(sc, px, py), 0.0, 1.0)
        alb = lin3(sc.crema * (1.0 - c) + sc.foam * c)
        e = 0.5 * sc.L / float(sc.N)
        gx = (field_at(sc, px + e, py) - field_at(sc, px - e, py)) / (2.0 * e)
        gy = (field_at(sc, px, py + e) - field_at(sc, px, py - e)) / (2.0 * e)
        n = wp.normalize(wp.vec3(-sc.bump * gx, -sc.bump * gy, 1.0))
        ks = 0.10
        shin = 60.0
    elif h.mat == 5:
        alb = sc.table_col
        ks = 0.03
        shin = 10.0
    L = sc.light
    ndl = wp.max(wp.dot(n, L), 0.0)
    vis = float(1.0)
    if wp.dot(ng, L) > 0.0:
        vis = shadow(sc, p + ng * 3.0e-5, L)
    else:
        vis = 0.0
    hv = wp.normalize(L - d)
    spec = ks * wp.pow(wp.max(wp.dot(n, hv), 0.0), shin) * vis * sc.key
    amb = sc.amb * (0.6 + 0.4 * n[2])
    return alb * (sc.key * ndl * vis + amb) + wp.vec3(spec, spec, spec)


@wp.func
def radiance(sc: Sc, o0: wp.vec3, d: wp.vec3):
    col = wp.vec3(0.0, 0.0, 0.0)
    T = float(1.0)
    o = o0
    layer = int(0)
    done = int(0)
    while layer < 10 and done == 0:
        h = trace(sc, o, d, 1.0e3)
        if h.mat < 0:
            col = col + background(sc, d) * T
            done = 1
        else:
            p = o + d * h.t
            c = shade(sc, p, d, h)
            a = float(1.0)
            if h.mat == 0 and sc.p_a0 < 1.0:
                cv = wp.abs(wp.dot(h.ng, d))
                a = wp.min(sc.p_a0 + (1.0 - sc.p_a0) * wp.pow(1.0 - cv, 4.0), 0.95)
            if h.mat == 5:                  # the table fades into the background with distance from the cup
                f = wp.exp(-wp.sqrt(p[0] * p[0] + p[1] * p[1]) / 0.22)
                c = c * f + background(sc, d) * (1.0 - f)
            col = col + c * (T * a)
            T = T * (1.0 - a)
            if T < 0.01:
                done = 1
            else:
                o = p + d * 5.0e-5
        layer = layer + 1
    if done == 0:
        col = col + background(sc, d) * T
    return col


@wp.kernel
def render_kernel(cam: Cam, sc: Sc, W: int, H: int, img: wp.array3d(dtype=wp.uint8)):
    i, j = wp.tid()
    acc = wp.vec3(0.0, 0.0, 0.0)
    ss = sc.ss
    for sy in range(ss):
        for sx in range(ss):
            px = ((float(j) + (float(sx) + 0.5) / float(ss)) / float(W)) * 2.0 - 1.0
            py = 1.0 - ((float(i) + (float(sy) + 0.5) / float(ss)) / float(H)) * 2.0
            o = cam.pos
            d = cam.fwd
            if cam.ortho == 1:
                o = cam.pos + cam.right * (px * cam.half_h * cam.aspect) + cam.up * (py * cam.half_h)
            else:
                d = wp.normalize(cam.fwd + cam.right * (px * cam.tan_y * cam.aspect) + cam.up * (py * cam.tan_y))
            acc = acc + radiance(sc, o, d)
    c = acc / float(ss * ss)
    for k in range(3):
        g = wp.pow(wp.clamp(c[k], 0.0, 1.0), 1.0 / 2.2)
        img[i, j, k] = wp.uint8(int(g * 255.0 + 0.5))


# --------------------------------------------------------------------------------------------- geometry (numpy)
def vertex_normals(V, F):
    """Area-weighted vertex normals (winding of F)."""
    V = np.asarray(V, np.float64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(ln, 1e-30)


def lathe(profile, n_phi=96):
    """Surface of revolution about z of a polyline profile [(r, z), ...] (r >= 0; points with r = 0 collapse to the
    axis).  Faces wound so that the normal points to the left of the profile direction (outward for a profile walked
    bottom -> top on the outside, then top -> bottom on the inside)."""
    P = np.asarray(profile, float)
    ph = np.linspace(0.0, 2 * math.pi, n_phi, endpoint=False)
    c, s = np.cos(ph), np.sin(ph)
    V = np.concatenate([np.stack([r * c, r * s, np.full(n_phi, z)], 1) for r, z in P])
    F = []
    for a in range(len(P) - 1):
        i0, i1 = a * n_phi, (a + 1) * n_phi
        for j in range(n_phi):
            j1 = (j + 1) % n_phi
            F.append((i0 + j, i0 + j1, i1 + j1))
            F.append((i0 + j, i1 + j1, i1 + j))
    F = np.array(F, np.int64)
    a = V[F[:, 0]]; b = V[F[:, 1]]; cc = V[F[:, 2]]
    keep = np.linalg.norm(np.cross(b - a, cc - a), axis=1) > 1e-14          # drop the degenerate axis triangles
    return V, F[keep]


def tube(P, r, n_side=12, cap0=True, cap1=True, frame0=None):
    """Tube mesh along the centre line P (k, 3) with radii r (k,): (V, F, Nv) with analytic (radial) vertex normals,
    outward winding, flat end caps."""
    P = np.asarray(P, float)
    r = np.asarray(r, float)
    k = len(P)
    T = np.gradient(P, axis=0) if k > 2 else np.repeat((P[-1] - P[0])[None], k, 0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if frame0 is None else np.asarray(frame0, float)
    if abs(np.dot(a, T[0])) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    Nn = np.cross(T[0], a); Nn /= np.linalg.norm(Nn)
    Ns = [Nn]
    for i in range(1, k):                                   # parallel transport of the normal
        n = Ns[-1] - np.dot(Ns[-1], T[i]) * T[i]
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            n = np.cross(T[i], a); ln = np.linalg.norm(n)
        Ns.append(n / ln)
    Ns = np.array(Ns)
    Bs = np.cross(T, Ns)
    th = np.linspace(0, 2 * math.pi, n_side, endpoint=False)
    dirs = np.cos(th)[None, :, None] * Ns[:, None, :] + np.sin(th)[None, :, None] * Bs[:, None, :]
    V = (P[:, None, :] + r[:, None, None] * dirs).reshape(-1, 3)
    Nv = dirs.reshape(-1, 3)
    F = []
    for i in range(k - 1):
        for j in range(n_side):
            j1 = (j + 1) % n_side
            a0, a1, b0, b1 = i * n_side + j, i * n_side + j1, (i + 1) * n_side + j, (i + 1) * n_side + j1
            F.append((a0, a1, b1)); F.append((a0, b1, b0))
    F = np.array(F, np.int64).reshape(-1, 3)
    # orientation: outward (normal along dirs)
    if len(F):
        f = F[0]
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        if np.dot(n, Nv[f[0]]) < 0:
            F = F[:, ::-1]
    Vs, Fs, Ns_ = [V], [F], [Nv]
    off = len(V)
    for end, sgn, do in ((0, -1.0, cap0), (k - 1, 1.0, cap1)):
        if not do:
            continue
        ring = V[end * n_side:(end + 1) * n_side]
        cv = np.vstack([ring, P[end][None]])
        cf = np.array([(n_side, j, (j + 1) % n_side) for j in range(n_side)], np.int64)
        n = np.cross(cv[cf[0, 1]] - cv[cf[0, 0]], cv[cf[0, 2]] - cv[cf[0, 0]])
        if np.dot(n, sgn * T[end]) < 0:
            cf = cf[:, ::-1]
        Vs.append(cv); Fs.append(cf + off); Ns_.append(np.repeat((sgn * T[end])[None], n_side + 1, 0))
        off += n_side + 1
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ns_)


def stream_mesh(segments, n_side=12, ds=6e-4, r_min=1.5e-4, sink=1.5e-3):
    """The stream tube from scene.stream(t)['segments'] (each P (k, 3) from the lip down, r (k,)): resampled to a centre
    line spacing >= ds, radius >= r_min; a landed segment is continued `sink` below the surface along its last direction
    so that it pierces the latte disc without a gap.  Returns (V, F, Nv, lines) or None; lines = the centre lines
    (k, 3) the tubes were built on, with the radii (k,): [(P, r, landed), ...]."""
    Vs, Fs, Ns, lines = [], [], [], []
    off = 0
    for s in segments:
        P = np.asarray(s["P"], float)
        r = np.maximum(np.asarray(s["r"], float), r_min)
        if len(P) < 2:
            continue
        keep = [0]
        acc = 0.0
        d = np.linalg.norm(np.diff(P, axis=0), axis=1)
        for i in range(1, len(P) - 1):
            acc += d[i - 1]
            if acc >= ds:
                keep.append(i); acc = 0.0
        keep.append(len(P) - 1)
        P, r = P[keep], r[keep]
        if len(P) < 2 or np.linalg.norm(P[-1] - P[0]) < 1e-5:
            continue
        if s.get("landed") and abs(P[-1, 2]) < 1e-6:
            t = P[-1] - P[-2]; t /= max(np.linalg.norm(t), 1e-12)
            P = np.vstack([P, P[-1] + sink * t / max(-t[2], 0.3)])
            r = np.r_[r, r[-1]]
        V, F, Nv = tube(P, r, n_side)
        Vs.append(V); Fs.append(F + off); Ns.append(Nv); off += len(V)
        lines.append((P, r, bool(s.get("landed"))))
    if not Vs:
        return None
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ns), lines


def cup_mesh(R_disc, rise0, wall=4e-3, rim_above=3e-3, n_phi=128, handle=True, handle_phi=0.0):
    """The cup STAND-IN in its own frame B (scene.cup_profile: liquid of the old end level at z_B = 0, rim top
    `rim_above` above it; the straight inner wall reaches rise0 + 4 mm below z_B = 0): lathed wall + a loop handle at
    azimuth handle_phi (scene.cup_handle).  About the size of the registered black mug, but NOT its measured shape.
    Returns (V, F, Nv, z_foot)."""
    import scene as S  # noqa: E402
    prof, z_foot = S.cup_profile(R_disc, rise0, wall, rim_above)
    V, F = lathe(prof[:-1] if np.allclose(prof[-1], prof[0]) else prof, n_phi)
    # winding: the outer wall must face away from the axis
    c = V[F].mean(1)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    i = int(np.argmax(np.hypot(c[:, 0], c[:, 1])))               # a face on the outermost wall
    if np.dot(n[i, :2], c[i, :2]) < 0:
        F = F[:, ::-1]
    Nv = vertex_normals(V, F)
    if handle:
        P, hr = S.cup_handle(R_disc, rise0, wall, rim_above, handle_phi)
        hv, hf, hn = tube(P, np.full(len(P), hr), 14)
        F = np.concatenate([F, hf + len(V)]); V = np.concatenate([V, hv]); Nv = np.concatenate([Nv, hn])
    return V, F, Nv, z_foot


def icosphere(radius=1.0, centre=(0, 0, 0), sub=3):
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1),
         (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7),
         (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(sub):
        cache, F2 = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m)); cache[k] = len(V) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    U = np.array(V)
    return U * radius + np.asarray(centre, float), np.array(F, np.int64), U


# --------------------------------------------------------------------------------------------- cameras
def look_at(pos, target, up=(0.0, 0.0, 1.0), fov_y_deg=30.0, aspect=16 / 9, ortho_half_h=None):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return dict(pos=pos, fwd=f, right=r, up=u, tan_y=math.tan(math.radians(fov_y_deg) / 2), aspect=float(aspect),
                ortho=ortho_half_h is not None, half_h=float(ortho_half_h or 0.0), target=target)


def orbit(target, az_deg, el_deg, dist, **kw):
    a, e = math.radians(az_deg), math.radians(el_deg)
    pos = np.asarray(target, float) + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return look_at(pos, target, **kw)


def fit_camera(points, az_deg, el_deg, fov_y_deg=30.0, aspect=16 / 9, margin=0.06, box=(-1.0, 1.0, -1.0, 1.0), it=4):
    """A perspective camera looking along the fixed direction (azimuth, elevation of the camera seen from the target)
    that frames all `points` (n, 3) inside the NDC box (x0, x1, y0, y1) (x right, y up, [-1, 1]) with a margin
    (fraction of the box half-size), centred in the box."""
    X = np.asarray(points, float)
    c = X.mean(0)
    a, e = math.radians(az_deg), math.radians(el_deg)
    back = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    ty = math.tan(math.radians(fov_y_deg) / 2); tx = ty * aspect
    hx, hy = 0.5 * (box[1] - box[0]) * (1 - margin), 0.5 * (box[3] - box[2]) * (1 - margin)
    bx, by = 0.5 * (box[1] + box[0]), 0.5 * (box[3] + box[2])
    cam = look_at(c + back, c, fov_y_deg=fov_y_deg, aspect=aspect)

    def fit(c):
        D = X - c
        xr, yu, zf = D @ cam["right"], D @ cam["up"], D @ cam["fwd"]
        dist = float(max(np.max(np.abs(xr) / (tx * hx) - zf), np.max(np.abs(yu) / (ty * hy) - zf), 0.05))
        depth = dist + zf
        nx, ny = xr / (depth * tx), yu / (depth * ty)
        return dist, 0.5 * (nx.max() + nx.min()), 0.5 * (ny.max() + ny.min())
    for _ in range(it):                                  # centre the projected points on the camera axis
        dist, mx, my = fit(c)
        c = c + cam["right"] * mx * tx * dist + cam["up"] * my * ty * dist
    dist, mx, my = fit(c)
    c = c - cam["right"] * bx * tx * dist - cam["up"] * by * ty * dist   # then move them to the box centre
    return orbit(c, az_deg, el_deg, dist, fov_y_deg=fov_y_deg, aspect=aspect)


def project(cam, X):
    """Pixel coordinates (u right, v down) in [0, 1] of F points for a camera dict (perspective / ortho)."""
    X = np.atleast_2d(np.asarray(X, float))
    D = X - cam["pos"]
    xr, yu, zf = D @ cam["right"], D @ cam["up"], D @ cam["fwd"]
    if cam["ortho"]:
        nx, ny = xr / (cam["half_h"] * cam["aspect"]), yu / cam["half_h"]
    else:
        nx, ny = xr / (zf * cam["tan_y"] * cam["aspect"]), yu / (zf * cam["tan_y"])
    return np.stack([(nx + 1) / 2, (1 - ny) / 2], 1), zf


# --------------------------------------------------------------------------------------------- renderer
class Renderer:
    """Holds the static BVHs (pitcher in its own frame, cup at the end level) on one device; per frame: set_dyn (milk +
    stream), set_field, render(cam, W, H, ...)."""

    def __init__(self, device="cpu", pitcher=None, cup=None, cup_white=False, ss=2):
        wp.config.quiet = True
        wp.init()
        self.device = device
        self.ss = int(ss)
        self._keep = []
        self.pm = self._mesh(*pitcher) if pitcher is not None else None
        self.cm = self._mesh(*cup) if cup is not None else None
        self.dm = None
        self.cup_white = bool(cup_white)
        self._dummy = self._mesh(np.array([[0, 0, -10.0], [1e-3, 0, -10.0], [0, 1e-3, -10.0]]), np.array([[0, 1, 2]]))
        self.set_field(np.zeros((2, 2), np.float32), 1.0, 0.0)
        self.set_dyn(None)

    def _mesh(self, V, F, Nv=None, mat=None):
        V = np.asarray(V, np.float32)
        F = np.asarray(F, np.int32).reshape(-1, 3)
        if Nv is None:
            Nv = vertex_normals(V, F)
        pts = wp.array(V, dtype=wp.vec3, device=self.device)
        idx = wp.array(F.reshape(-1), dtype=wp.int32, device=self.device)
        m = wp.Mesh(points=pts, indices=idx)
        nrm = wp.array(np.asarray(Nv, np.float32), dtype=wp.vec3, device=self.device)
        mt = wp.array(np.zeros(len(F), np.int32) if mat is None else np.asarray(mat, np.int32), dtype=wp.int32,
                      device=self.device)
        return dict(mesh=m, pts=pts, idx=idx, nrm=nrm, mat=mt, n=len(F))

    def set_dyn(self, parts):
        """parts: list of (V, F, Nv or None, material) for this frame (milk MAT_MILK, stream MAT_STREAM) or None."""
        parts = [p for p in (parts or []) if p is not None and len(p[1])]
        if not parts:
            self.dm = None
            return
        Vs, Fs, Ns, Ms = [], [], [], []
        off = 0
        for V, F, Nv, mat in parts:
            V = np.asarray(V, float); F = np.asarray(F, np.int64)
            Vs.append(V); Fs.append(F + off); Ns.append(vertex_normals(V, F) if Nv is None else Nv)
            Ms.append(np.full(len(F), mat, np.int32)); off += len(V)
        self.dm = self._mesh(np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ns), np.concatenate(Ms))

    def set_field(self, A, L, R_disc):
        A = np.ascontiguousarray(np.asarray(A, np.float32))
        self.fld = wp.array(A, dtype=wp.float32, device=self.device)
        self.N, self.L, self.R_disc = int(A.shape[0]), float(L), float(R_disc)

    def render(self, cam, W, H, pitcher_pose=None, pitcher_alpha=1.0, pitcher_shadow=0.0, cup_dz=0.0, table_z=None,
               disc=True, dyn=True, light=(-0.45, -0.55, 1.0), key=0.78, amb=0.30, bump=2.5e-4, ss=None, cup_pose=None):
        """cup_pose (R, t): X_F = R @ X_B + t for the cup mesh (scene.cup(t)['R'], ['t']); default (I, (0, 0, cup_dz))."""
        c = Cam()
        c.pos, c.fwd, c.right, c.up = (wp.vec3(*map(float, cam[k])) for k in ("pos", "fwd", "right", "up"))
        c.tan_y, c.half_h, c.aspect, c.ortho = float(cam["tan_y"]), float(cam["half_h"]), float(W) / float(H), int(cam["ortho"])
        s = Sc()
        pm = self.pm if (self.pm is not None and pitcher_pose is not None) else None
        s.p_on = int(pm is not None)
        pm = pm or self._dummy
        s.pm, s.pn = pm["mesh"].id, pm["nrm"]
        R, t = (np.eye(3), np.zeros(3)) if pitcher_pose is None else (np.asarray(pitcher_pose[0], float), np.asarray(pitcher_pose[1], float))
        s.pR = wp.mat33(*R.astype(np.float32).reshape(-1).tolist())
        s.pt = wp.vec3(*map(float, t))
        s.p_a0, s.p_sh, s.p_col = float(pitcher_alpha), float(pitcher_shadow), _lin(PITCHER_COL)
        cm = self.cm or self._dummy
        s.cm, s.cn, s.c_on = cm["mesh"].id, cm["nrm"], int(self.cm is not None)
        cR, ct = (np.eye(3), np.array([0.0, 0.0, float(cup_dz)])) if cup_pose is None else (
            np.asarray(cup_pose[0], float), np.asarray(cup_pose[1], float))
        s.cR = wp.mat33(*cR.astype(np.float32).reshape(-1).tolist())
        s.ct = wp.vec3(*map(float, ct))
        s.c_col = _lin(CUP_WHITE if self.cup_white else CUP_DARK)
        s.c_ks = 0.25 if self.cup_white else 0.45
        dm = self.dm if (dyn and self.dm is not None) else None
        s.d_on = int(dm is not None)
        dm = dm or self._dummy
        s.dm, s.dn, s.dmat = dm["mesh"].id, dm["nrm"], dm["mat"]
        s.milk_col, s.stream_col = _lin(MILK_COL), _lin(STREAM_COL)
        s.fld, s.N, s.L, s.R_disc = self.fld, self.N, self.L, self.R_disc
        s.disc_on = int(bool(disc) and self.R_disc > 0)
        s.bump, s.crema, s.foam = float(bump), _srgb(CREMA), _srgb(FOAM)
        s.table_on = int(table_z is not None)
        s.table_z = float(table_z if table_z is not None else -1.0)
        s.table_col = _lin(TABLE_COL)
        Lv = np.asarray(light, float); Lv /= np.linalg.norm(Lv)
        s.light, s.key, s.amb = wp.vec3(*map(float, Lv)), float(key), float(amb)
        s.bg_top, s.bg_bot = _lin(BG_TOP), _lin(BG_BOT)
        s.ss = int(ss or self.ss)
        img = wp.zeros((H, W, 3), dtype=wp.uint8, device=self.device)
        wp.launch(render_kernel, dim=(H, W), inputs=[c, s, int(W), int(H), img], device=self.device)
        return img.numpy()
