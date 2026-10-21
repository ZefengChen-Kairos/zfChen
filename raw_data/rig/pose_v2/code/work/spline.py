"""Continuous-time pitcher trajectory: cumulative cubic B-spline on SO(3) (Kim et al. 1995 / Sommer et al. 2020)
for the rotation and a uniform cubic B-spline for the origin.

  segment i covers [T0 + i dt, T0 + (i+1) dt) and uses control points i..i+3;  u = (t - T0)/dt - i
  R(t) = R_i prod_{j=1..3} exp(Bc_j(u) log(R_{i+j-1}^T R_{i+j})),   p(t) = sum_j B_j(u) p_{i+j}
Parameters are updated by right perturbations R_k <- R_k exp(d_k[:3]), p_k <- p_k + d_k[3:].
Every observation depends on the pose at its frame time only, so it is chained through
  pose perturbation eps = (eps_r body-frame rotation, eps_p world translation) at t:  d eps / d controls  (6 x 24)
and the time offset of its camera stream through d eps / d u (6).
"""
import numpy as np
import jax, jax.numpy as jnp
from pv2lib import expm3, logm3

_M = np.array([[1, 4, 1, 0], [-3, 0, 3, 0], [3, -6, 3, 0], [-1, 3, -3, 1]], float) / 6.0
_Mj = jnp.asarray(_M)


def basis(u):
    return jnp.array([1.0, u, u * u, u * u * u]) @ _Mj


def basis_d2(u):
    return jnp.array([0.0, 0.0, 2.0, 6.0 * u]) @ _Mj


def cbasis(u):
    b = basis(u)
    return jnp.array([1.0, b[1] + b[2] + b[3], b[2] + b[3], b[3]])


def _pert(Rk, pk, d):
    Rp = jnp.stack([Rk[j] @ expm3(d[j, :3]) for j in range(4)])
    return Rp, pk + d[:, 3:]


def seg_pose(Rk, pk, d, u):
    """Rk (4,3,3), pk (4,3), d (4,6) perturbations, u in [0,1] -> R (3,3), p (3,)."""
    Rp, pp = _pert(Rk, pk, d)
    bc = cbasis(u)
    R = Rp[0]
    for j in range(1, 4):
        R = R @ expm3(bc[j] * logm3(Rp[j - 1].T @ Rp[j]))
    return R, basis(u) @ pp


def _pose_eps(Rk, pk, d, u, R0, p0):
    R, p = seg_pose(Rk, pk, d, u)
    return jnp.concatenate([logm3(R0.T @ R), p - p0])


def _pose_and_jac(Rk, pk, u):
    d0 = jnp.zeros((4, 6))
    R0, p0 = seg_pose(Rk, pk, d0, u)
    G = jax.jacfwd(_pose_eps, argnums=2)(Rk, pk, d0, u, R0, p0).reshape(6, 24)
    gu = jax.jacfwd(_pose_eps, argnums=3)(Rk, pk, d0, u, R0, p0)
    return R0, p0, G, gu


pose_and_jac = jax.jit(jax.vmap(_pose_and_jac))
pose_only = jax.jit(jax.vmap(lambda Rk, pk, u: seg_pose(Rk, pk, jnp.zeros((4, 6)), u)))


def _acc(Rk, pk, d, u, h, dt):
    """body angular acceleration [rad/s^2] (central second difference of the rotation in u) and origin acceleration."""
    Ra, _ = seg_pose(Rk, pk, d, u - h)
    Rb, _ = seg_pose(Rk, pk, d, u)
    Rc, _ = seg_pose(Rk, pk, d, u + h)
    alpha = (logm3(Rb.T @ Rc) - logm3(Ra.T @ Rb)) / (h * dt) ** 2
    _, pp = _pert(Rk, pk, d)
    a = basis_d2(u) @ pp / dt ** 2
    return jnp.concatenate([alpha, a])


def _acc_and_jac(Rk, pk, u, h, dt):
    d0 = jnp.zeros((4, 6))
    r = _acc(Rk, pk, d0, u, h, dt)
    J = jax.jacfwd(_acc, argnums=2)(Rk, pk, d0, u, h, dt).reshape(6, 24)
    return r, J


acc_and_jac = jax.jit(jax.vmap(_acc_and_jac, in_axes=(0, 0, 0, None, None)))
acc_only = jax.jit(jax.vmap(lambda Rk, pk, u, h, dt: _acc(Rk, pk, jnp.zeros((4, 6)), u, h, dt), in_axes=(0, 0, 0, None, None)))


def _vel(Rk, pk, u, h, dt):
    Ra, pa = seg_pose(Rk, pk, jnp.zeros((4, 6)), u - h)
    Rc, pc = seg_pose(Rk, pk, jnp.zeros((4, 6)), u + h)
    return jnp.concatenate([logm3(Ra.T @ Rc) / (2 * h * dt), (pc - pa) / (2 * h * dt)])


vel_only = jax.jit(jax.vmap(_vel, in_axes=(0, 0, 0, None, None)))


class Traj:
    def __init__(self, T0, dt, R, P):
        self.T0, self.dt = float(T0), float(dt)
        self.R = np.asarray(R, float).copy(); self.P = np.asarray(P, float).copy()

    @property
    def K(self):
        return len(self.P)

    @property
    def t_end(self):
        return self.T0 + (self.K - 3) * self.dt

    def ctrl_times(self):
        return self.T0 + (np.arange(self.K) - 1) * self.dt

    @staticmethod
    def make(t_lo, t_hi, dt, pose_fn):
        """knots covering [t_lo, t_hi]; control point k initialised to pose_fn(T0 + (k-1) dt) -> (R, p)."""
        T0 = t_lo - 1e-3
        M = int(np.ceil((t_hi + 1e-3 - T0) / dt))
        K = M + 3
        tc = T0 + (np.arange(K) - 1) * dt
        R, P = pose_fn(tc)
        return Traj(T0, dt, R, P)

    def seg(self, t):
        s = (np.asarray(t, float) - self.T0) / self.dt
        i = np.clip(np.floor(s).astype(int), 0, self.K - 4)
        u = s - i
        return i, u

    def blocks(self, i):
        idx = i[:, None] + np.arange(4)[None]
        return self.R[idx], self.P[idx]

    def eval(self, t):
        i, u = self.seg(t)
        Rk, pk = self.blocks(i)
        R, p = pose_only(jnp.asarray(Rk), jnp.asarray(pk), jnp.asarray(u))
        return np.asarray(R), np.asarray(p)

    def eval_jac(self, t):
        i, u = self.seg(t)
        Rk, pk = self.blocks(i)
        R, p, G, gu = pose_and_jac(jnp.asarray(Rk), jnp.asarray(pk), jnp.asarray(u))
        return i, u, np.asarray(R), np.asarray(p), np.asarray(G), np.asarray(gu)

    def update(self, dx):
        from scipy.spatial.transform import Rotation as Rot
        dx = dx.reshape(self.K, 6)
        self.R = np.einsum('kij,kjl->kil', self.R, Rot.from_rotvec(dx[:, :3]).as_matrix())
        self.P = self.P + dx[:, 3:]

    def copy(self):
        return Traj(self.T0, self.dt, self.R, self.P)

    def quad(self, nq=2):
        """Gauss-Legendre points per segment (u) and segment indices for the prior."""
        g = {1: [0.5], 2: [0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)], 3: [0.5 - 0.5 * np.sqrt(0.6), 0.5, 0.5 + 0.5 * np.sqrt(0.6)]}[nq]
        M = self.K - 3
        i = np.repeat(np.arange(M), nq); u = np.tile(g, M)
        return i, u

    def acc(self, i, u, h=0.2):
        Rk, pk = self.blocks(i)
        return np.asarray(acc_only(jnp.asarray(Rk), jnp.asarray(pk), jnp.asarray(u), h, self.dt))

    def acc_jac(self, i, u, h=0.2):
        Rk, pk = self.blocks(i)
        r, J = acc_and_jac(jnp.asarray(Rk), jnp.asarray(pk), jnp.asarray(u), h, self.dt)
        return np.asarray(r), np.asarray(J)

    def vel(self, t, h=0.05):
        i, u = self.seg(t)
        u = np.clip(u, h, 1 - h)
        Rk, pk = self.blocks(i)
        return np.asarray(vel_only(jnp.asarray(Rk), jnp.asarray(pk), jnp.asarray(u), h, self.dt))

    def acc_at(self, t, h=0.2):
        i, u = self.seg(t)
        u = np.clip(u, h, 1 - h)
        return self.acc(i, u, h)
