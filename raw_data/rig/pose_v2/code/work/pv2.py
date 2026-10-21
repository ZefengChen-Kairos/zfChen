"""Pose track v2 core: tag model, JAX residuals / Jacobians, per-sample bundle adjustment, continuous-time B-spline
trajectory fit (tags + CAD-to-depth + smoothness prior).

Conventions (identical to the current chain, latte_imex/rig/pitcher_ba.py / tip_track.py):
  pitcher (tag) frame: z = jug axis towards the rim, origin on the axis at the height of tag 4's centre, x towards tag 4;
  pose (R, p): x_board = R x_pitcher + p; board z points into the table (up = -z).
Tag model: tag i centred at azimuth phi_i, height z_i, in-plane roll psi_i, edge s*TAG; 'flat' = planar tag in the
tangent plane at distance r from the axis (the jug is a 14-sided prism at the tag height, flats 19.4 mm wide), 'cyl' =
wrapped on the cylinder of radius r (pitcher_ba.tag_corners).  r is FIXED (CAD), tag 4: phi = z = 0.
"""
import os, sys, math, pickle, json
import numpy as np
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.spatial.transform import Rotation as Rot
import scipy.sparse as sp
import scipy.sparse.linalg as spl

sys.path.insert(0, '/home/user/zfChen/latte_imex/rig')
from trimrec import Take

SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
V2 = f'{SCR}/pose_v2'
POURS = ('tulip', 'leaf', 'heart', 'push_heart', 'swan')
TAG = 0.0190
REF = 4
CAMS = (0, 1)

# ------------------------------------------------------------------------------------------------ JAX geometry


def rodrigues(w):
    th2 = jnp.dot(w, w)
    th = jnp.sqrt(th2 + 1e-300)
    small = th2 < 1e-10
    A = jnp.where(small, 1.0 - th2 / 6.0, jnp.sin(th) / th)
    B = jnp.where(small, 0.5 - th2 / 24.0, (1.0 - jnp.cos(th)) / jnp.where(small, 1.0, th2))
    K = jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    return jnp.eye(3) + A * K + B * (K @ K)


def tag_corners(tp, s, r, flat):
    """tp = (phi, z, psi) -> (4,3) corners TL, TR, BR, BL (aruco order) in the pitcher frame."""
    phi, z, psi = tp[0], tp[1], tp[2]
    h = s * TAG / 2
    loc = jnp.array([[-h, h], [h, h], [h, -h], [-h, -h]])
    c, sn = jnp.cos(psi), jnp.sin(psi)
    loc = loc @ jnp.array([[c, -sn], [sn, c]]).T
    if flat:
        n = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0]); ex = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0.0])
        return r * n[None] + loc[:, :1] * ex[None] + (z + loc[:, 1:2]) * jnp.array([0.0, 0.0, 1.0])[None]
    ang = phi + loc[:, 0] / r
    return jnp.stack([r * jnp.cos(ang), r * jnp.sin(ang), z + loc[:, 1]], 1)


def project(Xw, W, Kv):
    """W: (3,4) world->camera, Kv = (fx, fy, cx, cy)."""
    Xc = Xw @ W[:, :3].T + W[:, 3]
    return jnp.stack([Xc[:, 0] / Xc[:, 2] * Kv[0] + Kv[2], Xc[:, 1] / Xc[:, 2] * Kv[1] + Kv[3]], 1)


def _tag_res_pose(pose, tp, s, W, Kv, q, r, flat):
    R = rodrigues(pose[:3])
    X = tag_corners(tp, s, r, flat) @ R.T + pose[3:]
    return (project(X, W, Kv) - q).ravel()


# uniform cubic B-spline basis (value, 1st, 2nd derivative in u)
_M = np.array([[1, 4, 1, 0], [-3, 0, 3, 0], [3, -6, 3, 0], [-1, 3, -3, 1]], float) / 6.0


def bvec(u):
    return jnp.array([1.0, u, u * u, u * u * u]) @ _M


def bvec_np(u):
    u = np.asarray(u, float)
    U = np.stack([np.ones_like(u), u, u * u, u ** 3], -1)
    return U @ _M


def bvec_np_d(u, d):
    u = np.asarray(u, float)
    if d == 1:
        U = np.stack([np.zeros_like(u), np.ones_like(u), 2 * u, 3 * u * u], -1)
    else:
        U = np.stack([np.zeros_like(u), np.zeros_like(u), 2 * np.ones_like(u), 6 * u], -1)
    return U @ _M


def spl_pose(F4, P4, u, Rc):
    """F4, P4: (4,3) local control points of the rotation-vector chart and of the origin; u in [0,1).
    R(t) = Rc exp(phi(t)), p(t) = sum B_j P_j."""
    b = bvec(u)
    phi = b @ F4
    return Rc @ rodrigues(phi), b @ P4


def _tag_res_spl(F4, P4, u, Rc, tp, s, W, Kv, q, r, flat):
    R, p = spl_pose(F4, P4, u, Rc)
    X = tag_corners(tp, s, r, flat) @ R.T + p
    return (project(X, W, Kv) - q).ravel()


def _dep_res_spl(F4, P4, u, Rc, Xw, sp_, n_):
    R, p = spl_pose(F4, P4, u, Rc)
    Xp = R.T @ (Xw - p)
    return jnp.dot(n_, Xp - sp_)


_jit = {}


def tag_pose_fun(r, flat):
    key = ('pose', round(r, 7), flat)
    if key not in _jit:
        f = lambda pose, tp, s, W, Kv, q: _tag_res_pose(pose, tp, s, W, Kv, q, r, flat)
        fj = jax.jacfwd(f, argnums=(0, 1, 2))
        _jit[key] = (jax.jit(jax.vmap(f)), jax.jit(jax.vmap(fj)))
    return _jit[key]


def tag_spl_fun(r, flat):
    key = ('spl', round(r, 7), flat)
    if key not in _jit:
        f = lambda F4, P4, u, Rc, tp, s, W, Kv, q: _tag_res_spl(F4, P4, u, Rc, tp, s, W, Kv, q, r, flat)
        fj = jax.jacfwd(f, argnums=(0, 1, 2, 4, 5))
        _jit[key] = (jax.jit(jax.vmap(f, in_axes=(0, 0, 0, None, 0, None, 0, 0, 0))),
                     jax.jit(jax.vmap(fj, in_axes=(0, 0, 0, None, 0, None, 0, 0, 0))))
    return _jit[key]


def dep_spl_fun():
    key = ('dep',)
    if key not in _jit:
        fj = jax.jacfwd(_dep_res_spl, argnums=(0, 1, 2))
        _jit[key] = (jax.jit(jax.vmap(_dep_res_spl, in_axes=(0, 0, 0, None, 0, 0, 0))),
                     jax.jit(jax.vmap(fj, in_axes=(0, 0, 0, None, 0, 0, 0))))
    return _jit[key]


# ------------------------------------------------------------------------------------------------ data


class PourData:
    def __init__(self, p, corners='edge'):
        self.p = p
        self.tk = Take(f"{SCR}/pours/ghdata/{p}")
        self.W2C = [np.linalg.inv(T)[:3] for T in self.tk.c2w]
        self.Kv = [np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]) for K in self.tk.K]
        rows = pickle.load(open(f"{V2}/data/tags_{p}.pkl", "rb"))
        key = 'q_' + corners
        self.rows = [d for d in rows if d[key] is not None and d['cam'] in CAMS]
        for d in self.rows:
            d['q'] = np.asarray(d[key], float)
        self.ts = {c: dict(zip(self.tk.steps[c], self.tk.ts[c] - self.tk.t0)) for c in range(self.tk.ncam)}

    def frame_time(self, c, k):
        return float(self.ts[c][k])


# ------------------------------------------------------------------------------------------------ layout container


class Layout:
    """tag params dict tag -> [phi, z, psi], scale s, radius r (fixed), flat flag."""

    def __init__(self, r, tags, flat=True, s=1.0, init=None):
        self.r, self.flat, self.s = float(r), bool(flat), float(s)
        self.tags = sorted(tags)
        self.tp = {t: np.zeros(3) for t in self.tags}
        if init:
            for t, v in init.items():
                if t in self.tp:
                    self.tp[t] = np.array(v, float)
        self.tp[REF][:2] = 0.0

    def copy(self):
        L = Layout(self.r, self.tags, self.flat, self.s)
        L.tp = {t: v.copy() for t, v in self.tp.items()}
        return L

    def free_index(self, fix_scale=True):
        """list of (tag, component) free parameters (tag 4: psi only) + 's' if scale free."""
        idx = []
        for t in self.tags:
            for j in range(3):
                if t == REF and j < 2:
                    continue
                idx.append((t, j))
        if not fix_scale:
            idx.append(('s', 0))
        return idx

    def corners(self, t):
        return np.asarray(tag_corners(jnp.asarray(self.tp[t]), self.s, self.r, self.flat))

    def as_dict(self):
        return dict(r=self.r, flat=self.flat, s=self.s, tag_mm=self.s * TAG * 1e3,
                    tp={int(t): [float(v[0]), float(v[1]), float(v[2])] for t, v in self.tp.items()})


def T_pt(L, t):
    """pitcher <- tag transform (tag x tangent, y axis, z outward normal) for initialisation."""
    phi, z, psi = L.tp[t]
    n = np.array([np.cos(phi), np.sin(phi), 0.0]); ez = np.array([0, 0, 1.0]); ex = np.cross(ez, n)
    Rt = np.stack([ex, ez, n], 1)
    c, s = np.cos(psi), np.sin(psi); Rt = Rt @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T = np.eye(4); T[:3, :3] = Rt; T[:3, 3] = L.r * n + [0, 0, z]
    return T


# ------------------------------------------------------------------------------------------------ robust weights


def huber_w(r, k):
    a = np.abs(r)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))


def huber_rho(r, k):
    a = np.abs(r)
    return np.where(a <= k, 0.5 * r * r, k * a - 0.5 * k * k)


def tukey_w(r, c):
    x = r / c
    return np.where(np.abs(x) < 1, (1 - x * x) ** 2, 0.0)


def tukey_rho(r, c):
    x = np.minimum(np.abs(r) / c, 1.0)
    return c * c / 6.0 * (1 - (1 - x * x) ** 3)


# ------------------------------------------------------------------------------------------------ per-sample BA


def build_samples(D, sync=0.012):
    """cam-0 frames with detections, joined with the cam-1 frame within `sync` s (as pitcher_ba.build, cams 0/1 only);
    unjoined cam-1 frames are their own samples.  Returns list of lists of (cam, step) and sample times (mean)."""
    by = {}
    for d in D.rows:
        by.setdefault((d['cam'], d['step']), []).append(d)
    k0 = sorted([k for k in by if k[0] == 0], key=lambda k: k[1])
    t1 = np.array([D.ts[1][k[1]] for k in sorted([k for k in by if k[0] == 1], key=lambda k: k[1])])
    s1 = [k for k in sorted([k for k in by if k[0] == 1], key=lambda k: k[1])]
    used, samples = set(), []
    for k in k0:
        t = D.ts[0][k[1]]; g = [k]
        if len(t1):
            j = int(np.argmin(np.abs(t1 - t)))
            if abs(t1[j] - t) < sync and s1[j] not in used:
                g.append(s1[j])
        samples.append(g); used.update(g)
    for k in s1:
        if k not in used:
            samples.append([k])
    tt = np.array([np.mean([D.ts[c][s] for c, s in g]) for g in samples])
    o = np.argsort(tt)
    samples = [samples[i] for i in o]
    return samples, tt[o], by


def sample_ba(D, L, poses, samples, by, fix_layout=False, fix_scale=True, iters=60, huber=1.0, verbose=False,
              tol=1e-10):
    """LM on corner reprojection (Huber), analytic Jacobians (JAX).  poses: (n,6) rotvec+t.  Returns poses, L, info."""
    f, fj = tag_pose_fun(L.r, L.flat)
    obs = [(si, d) for si, g in enumerate(samples) for k in g for d in by[k] if d['tag'] in L.tp]
    oi = np.array([o[0] for o in obs]); tags = [o[1]['tag'] for o in obs]
    W = np.stack([D.W2C[o[1]['cam']] for o in obs]); Kv = np.stack([D.Kv[o[1]['cam']] for o in obs])
    Q = np.stack([o[1]['q'] for o in obs])
    fidx = [] if fix_layout else L.free_index(fix_scale)
    col = {k: i for i, k in enumerate(fidx)}
    nP = 6 * len(samples); nL = len(fidx)
    poses = np.array(poses, float).copy(); L = L.copy()

    def evaluate(poses, L):
        TP = np.stack([L.tp[t] for t in tags])
        return np.asarray(f(jnp.asarray(poses[oi]), jnp.asarray(TP), L.s, W, Kv, Q))

    def cost(r):
        return float(huber_rho(r, huber).sum())

    lam = 1e-3
    r = evaluate(poses, L); c0 = cost(r)
    hist = [c0]
    for it in range(iters):
        TP = np.stack([L.tp[t] for t in tags])
        Jp, Jt, Js = [np.asarray(a) for a in fj(jnp.asarray(poses[oi]), jnp.asarray(TP), L.s, W, Kv, Q)]
        w = huber_w(r, huber)
        rows_, cols_, vals_ = [], [], []
        no = len(obs)
        base = np.arange(no)[:, None] * 8 + np.arange(8)[None]
        # pose block
        rr = np.repeat(base[:, :, None], 6, 2); cc = np.repeat((6 * oi)[:, None, None] + np.arange(6)[None, None], 8, 1)
        rows_.append(rr.ravel()); cols_.append(cc.ravel()); vals_.append(Jp.ravel())
        for n_, t in enumerate(tags):
            for j in range(3):
                if (t, j) in col:
                    rows_.append(base[n_]); cols_.append(np.full(8, nP + col[(t, j)])); vals_.append(Jt[n_, :, j])
            if ('s', 0) in col:
                rows_.append(base[n_]); cols_.append(np.full(8, nP + col[('s', 0)])); vals_.append(Js[n_])
        J = sp.csr_matrix((np.concatenate(vals_), (np.concatenate(rows_), np.concatenate(cols_))), shape=(no * 8, nP + nL))
        rf = r.ravel(); wf = w.ravel()
        JW = J.multiply(wf[:, None]).tocsr()
        H = (J.T @ JW).tocsc(); g = J.T @ (wf * rf)
        dH = H.diagonal()
        while True:
            A = H + sp.diags(lam * np.maximum(dH, 1e-9))
            dx = -spl.spsolve(A.tocsc(), g)
            P2 = poses.copy(); P2 += dx[:nP].reshape(-1, 6)
            L2 = L.copy()
            for (t, j), i in col.items():
                if t == 's':
                    L2.s += dx[nP + i]
                else:
                    L2.tp[t][j] += dx[nP + i]
            r2 = evaluate(P2, L2); c2 = cost(r2)
            if c2 < hist[-1]:
                poses, L, r = P2, L2, r2; lam = max(lam / 3, 1e-9); break
            lam *= 4
            if lam > 1e8:
                break
        rel = (hist[-1] - c2) / max(hist[-1], 1e-12) if c2 < hist[-1] else 0.0
        hist.append(min(c2, hist[-1]))
        if verbose:
            print(' it', it, 'cost', round(hist[-1], 4), 'lam', lam, 'rms', round(float(np.sqrt(np.mean(r ** 2))), 4), flush=True)
        if lam > 1e8 or (0 <= rel < tol and it > 2):
            break
    info = dict(cost=hist[-1], rms=float(np.sqrt(np.mean(r ** 2))), med=float(np.median(np.abs(r))), iters=len(hist) - 1,
                converged=bool(lam <= 1e8), n_obs=len(obs), resid=r, oi=oi, tags=tags,
                cams=np.array([o[1]['cam'] for o in obs]))
    return poses, L, info


def init_sample_poses(D, L, samples, by):
    """best IPPE candidate per sample (max corner error over the sample's detections)."""
    out = []
    for g in samples:
        cand = [d for k in g for d in by[k] if d['tag'] in L.tp]
        best = None
        for d in cand:
            for Tb in d['T_board_alts']:
                T = Tb @ np.linalg.inv(T_pt(L, d['tag']))
                e = 0
                for dd in cand:
                    X = L.corners(dd['tag']) @ T[:3, :3].T + T[:3, 3]
                    W = D.W2C[dd['cam']]; Xc = X @ W[:, :3].T + W[:, 3]; Kv = D.Kv[dd['cam']]
                    uv = np.stack([Xc[:, 0] / Xc[:, 2] * Kv[0] + Kv[2], Xc[:, 1] / Xc[:, 2] * Kv[1] + Kv[3]], 1)
                    e = max(e, np.abs(uv - dd['q']).max())
                if best is None or e < best[0]:
                    best = (e, T)
        T = best[1]
        out.append(np.r_[Rot.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])
    return np.array(out)


def frame_pose_candidates(D, L, obs, extra=(), max_nfev=50):
    """Fixed-layout single-frame (one camera) pose: multi-start from every IPPE alternative of every tag (+extra);
    returns distinct local minima sorted by cost: list of (cost, x(6), rms)."""
    from scipy.optimize import least_squares
    f, fj = tag_pose_fun(L.r, L.flat)
    W = np.stack([D.W2C[d['cam']] for d in obs]); Kv = np.stack([D.Kv[d['cam']] for d in obs])
    Q = np.stack([d['q'] for d in obs]); TP = np.stack([L.tp[d['tag']] for d in obs])
    n = len(obs)

    def res(x):
        return np.asarray(f(jnp.asarray(np.repeat(x[None], n, 0)), jnp.asarray(TP), L.s, W, Kv, Q)).ravel()

    def jac(x):
        Jp = np.asarray(fj(jnp.asarray(np.repeat(x[None], n, 0)), jnp.asarray(TP), L.s, W, Kv, Q)[0])
        return Jp.reshape(-1, 6)
    starts = []
    for d in obs:
        for Tb in d['T_board_alts']:
            T = Tb @ np.linalg.inv(T_pt(L, d['tag']))
            starts.append(np.r_[Rot.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])
    starts += list(extra)
    sols = []
    for x0 in starts:
        try:
            r = least_squares(res, x0, jac=jac, method='lm', max_nfev=max_nfev, xtol=1e-12, ftol=1e-12)
        except Exception:
            continue
        if not np.all(np.isfinite(r.fun)):
            continue
        sols.append((float(0.5 * np.sum(r.fun ** 2)), r.x, float(np.sqrt(np.mean(r.fun ** 2)))))
    sols.sort(key=lambda s: s[0])
    uniq = []
    for s_ in sols:
        if all(rot_deg(s_[1], u[1]) > 2.0 or np.linalg.norm(s_[1][3:] - u[1][3:]) > 0.004 for u in uniq):
            uniq.append(s_)
    return uniq


def rot_deg(a, b):
    return float(np.degrees((Rot.from_rotvec(a[:3]).inv() * Rot.from_rotvec(b[:3])).magnitude()))


UP = np.array([0, 0, -1.0])


def tilt_az(Rm):
    """axis tilt from vertical [deg] and downhill azimuth in the pitcher frame [deg] (pitcher_model.py)."""
    Rm = np.atleast_3d(Rm) if Rm.ndim == 3 else Rm[None]
    a = Rm[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(a @ UP, -1, 1)))
    g = np.einsum('nji,j->ni', Rm, -UP)
    az = np.degrees(np.arctan2(g[:, 1], g[:, 0]))
    return tilt, az
