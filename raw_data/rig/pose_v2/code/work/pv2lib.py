"""Pose track v2 core library.

Conventions (identical to latte_imex/rig/pitcher_ba.py, tip_track.py, real_program.py):
  pose (R, p): x_board = R x_pitcher + p.  Board z points into the table (up = -z).
  Tag (pitcher) frame: z = jug axis towards the rim, origin on the axis at the height of tag 4's centre, x towards
  tag 4's flat.
Tag model.  The jug is a 14-sided prism at the tag height (CAD pitcher_kona700: circumradius 43.7 mm at CAD z 28.5,
apothem 42.6 mm, flats 19.45 mm wide; tags are 19.0 mm squares on the flats).  'flat': tag i lies in the plane
x.n(phi_i) = a (a = apothem, FIXED from the CAD), centre a n + u t + z ez, tag x along t = ez x n, tag y along ez,
in-plane roll psi.  'cyl': wrapped on the cylinder of radius a (pitcher_ba.tag_corners) for comparison.
Tag 4 defines the frame: phi = u = z = 0, psi free.
"""
import os, sys, math, pickle, json
import numpy as np
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, '/home/user/zfChen/latte_imex/rig')
from trimrec import Take  # noqa: E402

SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
V2 = f'{SCR}/pose_v2'
POURS = ('tulip', 'leaf', 'heart', 'push_heart', 'swan')
TAG = 0.0190
REF = 4
CAMS = (0, 1)
UP = np.array([0, 0, -1.0])

# ------------------------------------------------------------------------------------------------ JAX geometry


def skew(w):
    return jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def expm3(w):
    th2 = jnp.dot(w, w)
    small = th2 < 1e-12
    th = jnp.sqrt(jnp.where(small, 1.0, th2))
    A = jnp.where(small, 1.0 - th2 / 6.0, jnp.sin(th) / th)
    B = jnp.where(small, 0.5 - th2 / 24.0, (1.0 - jnp.cos(th)) / jnp.where(small, 1.0, th2))
    K = skew(w)
    return jnp.eye(3) + A * K + B * (K @ K)


def logm3(R):
    """rotation matrix -> rotation vector (angles < ~170 deg; used for small relative rotations)."""
    c = (jnp.trace(R) - 1.0) / 2.0
    v = jnp.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s2 = jnp.dot(v, v) / 4.0                                   # sin^2(theta)
    small = s2 < 1e-14
    s = jnp.sqrt(jnp.where(small, 1.0, s2))
    th = jnp.arctan2(s, c)
    f = jnp.where(small, 0.5 + (1.0 - c) / 6.0, th / (2.0 * s))   # theta / (2 sin theta)
    return f * v


def tag_corners(tp, s, a, flat=True):
    """tp = (phi, z, psi, u[, tx, ty]) -> (4,3) corners TL, TR, BR, BL (aruco order) in the pitcher frame.
    tx, ty: tilt of the tag plane about its own tangent (x) / axis (y) direction (0 = on the ideal flat)."""
    phi, z, psi, u = tp[0], tp[1], tp[2], tp[3]
    h = s * TAG / 2
    loc = jnp.array([[-h, h], [h, h], [h, -h], [-h, -h]])
    c, sn = jnp.cos(psi), jnp.sin(psi)
    loc = loc @ jnp.array([[c, -sn], [sn, c]]).T
    n = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
    t = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0.0])
    ez = jnp.array([0.0, 0.0, 1.0])
    if flat:
        if tp.shape[0] > 4:
            B = jnp.stack([t, ez, n], 1) @ expm3(jnp.array([tp[4], tp[5], 0.0]))
            loc3 = jnp.concatenate([loc, jnp.zeros((4, 1))], 1) @ B.T
            return (a * n + u * t + z * ez)[None] + loc3
        return (a * n + u * t)[None] + loc[:, :1] * t[None] + (z + loc[:, 1:2]) * ez[None]
    ang = phi + (u + loc[:, 0]) / a
    return jnp.stack([a * jnp.cos(ang), a * jnp.sin(ang), z + loc[:, 1]], 1)


def project(Xw, W, Kv):
    Xc = Xw @ W[:, :3].T + W[:, 3]
    return jnp.stack([Xc[:, 0] / Xc[:, 2] * Kv[0] + Kv[2], Xc[:, 1] / Xc[:, 2] * Kv[1] + Kv[3]], 1)


def _res_pose(pose, tp, s, W, Kv, q, a, flat):
    R = expm3(pose[:3])
    X = tag_corners(tp, s, a, flat) @ R.T + pose[3:]
    return (project(X, W, Kv) - q).ravel()


_JIT = {}


def pose_fun(a, flat):
    key = ('pose', round(float(a), 8), bool(flat))
    if key not in _JIT:
        f = lambda pose, tp, s, W, Kv, q: _res_pose(pose, tp, s, W, Kv, q, a, flat)
        fj = jax.jacfwd(f, argnums=(0, 1, 2))
        _JIT[key] = (jax.jit(jax.vmap(f, in_axes=(0, 0, None, 0, 0, 0))), jax.jit(jax.vmap(fj, in_axes=(0, 0, None, 0, 0, 0))))
    return _JIT[key]


# ------------------------------------------------------------------------------------------------ robust kernels


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


# ------------------------------------------------------------------------------------------------ data


class PourData:
    """Take + tag detections of cameras 0 and 1 (camera 2 is never read here)."""

    def __init__(self, p, corners='edge', tags=None):
        self.p = p
        self.tk = Take(f"{SCR}/pours/ghdata/{p}")
        self.W2C = [np.linalg.inv(T)[:3] for T in self.tk.c2w]
        self.Kv = [np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]) for K in self.tk.K]
        rows = pickle.load(open(f"{V2}/data/tags_{p}.pkl", "rb"))
        key = 'q_' + corners
        self.rows = [d for d in rows if d[key] is not None and d['cam'] in CAMS and (tags is None or d['tag'] in tags)]
        for d in self.rows:
            d['q'] = np.asarray(d[key], float)
        self.ts = {c: dict(zip(self.tk.steps[c].tolist(), (self.tk.ts[c] - self.tk.t0).tolist())) for c in range(self.tk.ncam)}
        for d in self.rows:
            d['t'] = self.ts[d['cam']][d['step']]

    def by_frame(self):
        by = {}
        for d in self.rows:
            by.setdefault((d['cam'], d['step']), []).append(d)
        return by


# ------------------------------------------------------------------------------------------------ layout


class Layout:
    """per-tag params [phi, z, psi, u]; apothem a (fixed), scale s, model flat/cyl."""
    NP = 6

    def __init__(self, a, tags, flat=True, s=1.0, init=None, free_u=False):
        self.a, self.flat, self.s, self.free_u = float(a), bool(flat), float(s), bool(free_u)
        self.tags = sorted(tags)
        self.tp = {t: np.zeros(6) for t in self.tags}
        self.ext1 = np.zeros(6)                 # world-frame correction of camera 1's extrinsics (rotvec, t)
        self.ext_cen = np.zeros(3)              # rotation centre of that correction [world]
        self.pl = np.zeros(6)                   # correction of the CAD placement in the tag frame (rotvec about pl_cen, t)
        self.pl_cen = np.zeros(3)
        if init:
            for t, v in init.items():
                if t in self.tp:
                    v = list(v) + [0.0] * (6 - len(v))
                    self.tp[t] = np.array(v[:6], float)
        self.tp[REF][[0, 1, 3]] = 0.0

    def copy(self):
        L = Layout(self.a, self.tags, self.flat, self.s, free_u=self.free_u)
        L.tp = {t: v.copy() for t, v in self.tp.items()}
        L.ext1 = self.ext1.copy(); L.ext_cen = self.ext_cen.copy()
        L.pl = self.pl.copy(); L.pl_cen = self.pl_cen.copy()
        return L

    def free_index(self, fix_scale=True):
        idx = []
        for t in self.tags:
            for j in range(4):
                if t == REF and j != 2:
                    continue
                if j == 3 and not self.free_u:
                    continue
                idx.append((t, j))
        if not fix_scale:
            idx.append(('s', 0))
        return idx

    def corners(self, t):
        return np.asarray(tag_corners(jnp.asarray(self.tp[t]), self.s, self.a, self.flat))

    def as_dict(self):
        return dict(a=self.a, flat=self.flat, s=self.s, tag_mm=self.s * TAG * 1e3, free_u=self.free_u,
                    tp={int(t): [float(x) for x in v] for t, v in self.tp.items()},
                    ext1=[float(x) for x in self.ext1], ext_cen=[float(x) for x in self.ext_cen],
                    pl=[float(x) for x in self.pl], pl_cen=[float(x) for x in self.pl_cen])

    @staticmethod
    def from_dict(d):
        L = Layout(d['a'], [int(t) for t in d['tp']], d['flat'], d['s'], free_u=d.get('free_u', False))
        L.tp = {int(t): np.r_[np.array(v, float), np.zeros(6 - len(v))] for t, v in d['tp'].items()}
        L.ext1 = np.array(d.get('ext1', np.zeros(6)), float); L.ext_cen = np.array(d.get('ext_cen', np.zeros(3)), float)
        L.pl = np.array(d.get('pl', np.zeros(6)), float); L.pl_cen = np.array(d.get('pl_cen', np.zeros(3)), float)
        return L

    def pretty(self):
        out = {t: [round(math.degrees(v[0]), 2), round(float(v[1]) * 1e3, 2), round(math.degrees(v[2]), 2), round(float(v[3]) * 1e3, 2),
                   round(math.degrees(v[4]), 2), round(math.degrees(v[5]), 2)] for t, v in self.tp.items()}
        if np.any(self.ext1):
            out['ext1'] = [round(math.degrees(x), 3) for x in self.ext1[:3]] + [round(float(x) * 1e3, 2) for x in self.ext1[3:]]
        if np.any(self.pl):
            out['pl'] = [round(math.degrees(x), 3) for x in self.pl[:3]] + [round(float(x) * 1e3, 2) for x in self.pl[3:]]
        return out


def T_pt(L, t):
    """pitcher <- tag transform (tag x tangent, y axis, z outward normal)."""
    phi, z, psi, u = L.tp[t][:4]
    n = np.array([np.cos(phi), np.sin(phi), 0.0]); ez = np.array([0, 0, 1.0]); ex = np.cross(ez, n)
    Rt = np.stack([ex, ez, n], 1)
    c, s = np.cos(psi), np.sin(psi); Rt = Rt @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T = np.eye(4); T[:3, :3] = Rt; T[:3, 3] = L.a * n + u * ex + [0, 0, z]
    return T


# ------------------------------------------------------------------------------------------------ per-sample BA


def build_samples(D, sync=0.012, cams=CAMS):
    """samples = cam-0 frames with detections joined with the cam-1 frame within `sync` s (pitcher_ba.build rule,
    cameras 0/1 only); unjoined cam-1 frames are their own samples.  Returns samples (list of [(cam, step)]), times."""
    by = D.by_frame()
    keys = {c: sorted([k for k in by if k[0] == c], key=lambda k: k[1]) for c in cams}
    samples, used = [], set()
    if 0 in cams:
        t1 = np.array([D.ts[1][k[1]] for k in keys.get(1, [])])
        for k in keys[0]:
            t = D.ts[0][k[1]]; g = [k]
            if 1 in cams and len(t1):
                j = int(np.argmin(np.abs(t1 - t)))
                if abs(t1[j] - t) < sync and keys[1][j] not in used:
                    g.append(keys[1][j])
            samples.append(g); used.update(g)
    for c in cams:
        for k in keys[c]:
            if k not in used:
                samples.append([k]); used.add(k)
    tt = np.array([np.mean([D.ts[c][s] for c, s in g]) for g in samples])
    o = np.argsort(tt)
    return [samples[i] for i in o], tt[o], by


def sample_ba(D, L, poses, samples, by, fix_layout=False, fix_scale=True, iters=200, huber=1.0, verbose=False,
              tol=1e-12, wt=None):
    """Levenberg-Marquardt (IRLS Huber) on corner reprojection with analytic (JAX) Jacobians; Schur complement on the
    6x6 pose blocks.  Converges to |relative cost decrease| < tol or a vanishing step.  Returns poses, L, info."""
    f, fj = pose_fun(L.a, L.flat)
    obs = [(si, d) for si, g in enumerate(samples) for k in g for d in by[k] if d['tag'] in L.tp]
    oi = np.array([o[0] for o in obs]); tags = [o[1]['tag'] for o in obs]
    W = np.stack([D.W2C[o[1]['cam']] for o in obs]); Kv = np.stack([D.Kv[o[1]['cam']] for o in obs])
    Q = np.stack([o[1]['q'] for o in obs])
    ow = np.ones(len(obs)) if wt is None else np.asarray(wt, float)
    fidx = [] if fix_layout else L.free_index(fix_scale)
    col = {k: i for i, k in enumerate(fidx)}
    n = len(samples); nL = len(fidx)
    poses = np.array(poses, float).copy(); L = L.copy()

    def evaluate(poses, L):
        TP = np.stack([L.tp[t] for t in tags])
        return np.asarray(f(jnp.asarray(poses[oi]), jnp.asarray(TP), L.s, W, Kv, Q))

    def cost(r):
        return float((huber_rho(r, huber) * ow[:, None]).sum())

    lam = 1e-4
    r = evaluate(poses, L); hist = [cost(r)]
    tcol = np.array([[col.get((t, j), -1) for j in range(6)] for t in tags])
    scol = col.get(('s', 0), -1)
    it = 0; status = 'maxiter'
    for it in range(iters):
        TP = np.stack([L.tp[t] for t in tags])
        Jp, Jt, Js = [np.asarray(x) for x in fj(jnp.asarray(poses[oi]), jnp.asarray(TP), L.s, W, Kv, Q)]
        w = huber_w(r, huber) * ow[:, None]                          # (no, 8)
        # layout Jacobian per observation (no, 8, nL)
        JL = np.zeros((len(obs), 8, nL))
        for j in range(Jt.shape[2]):
            m = tcol[:, j] >= 0
            if m.any():
                JL[np.where(m)[0], :, tcol[m, j]] = Jt[m, :, j]
        if scol >= 0:
            JL[:, :, scol] = Js
        # normal equations
        Hpp = np.zeros((n, 6, 6)); gp = np.zeros((n, 6))
        np.add.at(Hpp, oi, np.einsum('nki,nk,nkj->nij', Jp, w, Jp))
        np.add.at(gp, oi, np.einsum('nki,nk,nk->ni', Jp, w, r))
        Hpl = np.zeros((n, 6, nL)); np.add.at(Hpl, oi, np.einsum('nki,nk,nkj->nij', Jp, w, JL))
        Hll = np.einsum('nki,nk,nkj->ij', JL, w, JL); gl = np.einsum('nki,nk,nk->i', JL, w, r)
        while True:
            Ap = Hpp + lam * (np.einsum('nii->ni', Hpp)[:, :, None] * np.eye(6)[None] + 1e-12 * np.eye(6)[None])
            Ai = np.linalg.inv(Ap)
            if nL:
                All = Hll + lam * np.diag(np.diag(Hll) + 1e-12)
                S = All - np.einsum('nia,nij,njb->ab', Hpl, Ai, Hpl)
                bs = -gl + np.einsum('nia,nij,nj->a', Hpl, Ai, gp)
                dl = np.linalg.solve(S, bs)
                dp = -np.einsum('nij,nj->ni', Ai, gp + np.einsum('nia,a->ni', Hpl, dl))
            else:
                dl = np.zeros(0)
                dp = -np.einsum('nij,nj->ni', Ai, gp)
            P2 = poses + dp
            L2 = L.copy()
            for (t, j), i in col.items():
                if t == 's':
                    L2.s += dl[i]
                else:
                    L2.tp[t][j] += dl[i]
            r2 = evaluate(P2, L2); c2 = cost(r2)
            if np.isfinite(c2) and c2 <= hist[-1]:
                accepted = True
                poses, L, r = P2, L2, r2; lam = max(lam / 5, 1e-12); break
            lam *= 5
            if lam > 1e10:
                accepted = False; break
        rel = (hist[-1] - c2) / max(hist[-1], 1e-12) if accepted else 0.0
        hist.append(min(c2, hist[-1]) if accepted else hist[-1])
        stepn = float(np.abs(dp[:, 3:]).max() if len(dp) else 0) if accepted else 0.0
        if verbose and (it % 10 == 0):
            print(f' it {it} cost {hist[-1]:.6f} lam {lam:.1e} rel {rel:.2e} max step mm {stepn * 1e3:.4f}', flush=True)
        if not accepted:
            status = 'converged(lambda)'; break
        if it > 3 and rel < tol and stepn < 1e-8:
            status = 'converged'; break
    gnorm = float(np.sqrt((gp ** 2).sum() + (gl ** 2).sum()))
    info = dict(cost=hist[-1], rms=float(np.sqrt(np.mean(r ** 2))), med=float(np.median(np.abs(r))), iters=it + 1,
                status=status, grad_norm_last=gnorm, n_obs=len(obs), resid=r, oi=oi, tags=np.array(tags),
                cams=np.array([o[1]['cam'] for o in obs]), steps=np.array([o[1]['step'] for o in obs]), hist=hist)
    if nL:
        # layout covariance (Gauss-Newton, Huber weights, scaled by the robust residual variance)
        S = Hll - np.einsum('nia,nij,njb->ab', Hpl, np.linalg.inv(Hpp + 1e-12 * np.eye(6)[None]), Hpl)
        dof = max(r.size - 6 * n - nL, 1)
        s2 = float((w * r * r).sum() / dof)
        info['layout_cov'] = np.linalg.pinv(S) * s2
        info['layout_idx'] = fidx
    return poses, L, info


def ippe_starts(L, obs):
    starts = []
    for d in obs:
        for Tb in d['T_board_alts']:
            T = Tb @ np.linalg.inv(T_pt(L, d['tag']))
            starts.append(np.r_[Rot.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])
    return starts


def frame_candidates(D, L, obs, extra=(), iters=30):
    """fixed-layout pose from one sample's detections: Gauss-Newton from every IPPE alternative of every tag (+extra);
    distinct local minima sorted by cost: list of (cost, x6, rms_px)."""
    f, fj = pose_fun(L.a, L.flat)
    W = np.stack([D.W2C[d['cam']] for d in obs]); Kv = np.stack([D.Kv[d['cam']] for d in obs])
    Q = np.stack([d['q'] for d in obs]); TP = jnp.asarray(np.stack([L.tp[d['tag']] for d in obs]))
    m = len(obs)
    sols = []
    for x0 in list(ippe_starts(L, obs)) + list(extra):
        x = np.array(x0, float); lam = 1e-3
        r = np.asarray(f(jnp.asarray(np.repeat(x[None], m, 0)), TP, L.s, W, Kv, Q)).ravel(); c = float(huber_rho(r, 1.0).sum())
        for _ in range(iters):
            J = np.asarray(fj(jnp.asarray(np.repeat(x[None], m, 0)), TP, L.s, W, Kv, Q)[0]).reshape(-1, 6)
            w = huber_w(r, 1.0)
            H = J.T @ (J * w[:, None]); g = J.T @ (w * r)
            ok = False
            while lam < 1e8:
                dx = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
                r2 = np.asarray(f(jnp.asarray(np.repeat((x + dx)[None], m, 0)), TP, L.s, W, Kv, Q)).ravel()
                c2 = float(huber_rho(r2, 1.0).sum())
                if np.isfinite(c2) and c2 <= c:
                    x, r, lam, ok = x + dx, r2, lam / 5, True
                    rel = (c - c2) / max(c, 1e-12); c = c2
                    break
                lam *= 5
            if not ok or rel < 1e-10:
                break
        if np.all(np.isfinite(r)):
            sols.append((c, x, float(np.sqrt(np.mean(r ** 2)))))
    sols.sort(key=lambda s_: s_[0])
    uniq = []
    for s_ in sols:
        if all(rot_deg(s_[1], u_[1]) > 2.0 or np.linalg.norm(s_[1][3:] - u_[1][3:]) > 0.004 for u_ in uniq):
            uniq.append(s_)
    return uniq


def rot_deg(a, b):
    return float(np.degrees((Rot.from_rotvec(a[:3]).inv() * Rot.from_rotvec(b[:3])).magnitude()))


def tilt_az(Rm):
    """axis tilt from vertical [deg] and downhill azimuth in the pitcher frame [deg] (pitcher_model.py)."""
    Rm = np.asarray(Rm)
    if Rm.ndim == 2:
        Rm = Rm[None]
    a = Rm[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(a @ UP, -1, 1)))
    g = np.einsum('nji,j->ni', Rm, -UP)
    az = np.degrees(np.arctan2(g[:, 1], g[:, 0]))
    return tilt, az
