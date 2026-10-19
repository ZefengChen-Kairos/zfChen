"""Jet-driven overturning circulation of the cup: the lower layer the surface film is dragged by.

V1 3.3 drags the film towards an effective lower layer at rest in R.  Measured surface velocities of the real pours
(optical flow of the rectified top view) show instead a broad current from the pitcher side to the far side over the
whole cup, without surface return flow: the plunging stream drives an overturning cell (upwelling behind the stream,
surface current across the cup, downwelling along the wall, return at depth).  Its surface velocity is modelled as the
potential flow of that cell,
    V_b = grad(phi),   lap(phi) = a (K_up - W)  in the cup,   d(phi)/dn = 0 at the wall,
K_up: upwelling kernel (radius R_up, centred BACK * R_up behind the landing point against the stream's horizontal
direction), W: the wall cells (downwelling along the wall, each integrating to 1), a = Q: the received flow (overturning
flux per unit kappa_o).  The cell keeps turning for a while after the forcing changes: the field relaxes towards the
current one with time constant TAU, per frame,  V <- e V + (1 - e) Q grad(phi_k),  K <- e K + (1 - e) Q K_up,k,
e = exp(-dt_frame / TAU).  The solver uses
    drag towards kappa_o V           (the film's -beta (q - l kappa_o V) instead of -beta q)
    film source kappa_s K            (upwelled base material: the film diverges where the cell upwells)
so kappa_o, kappa_s are linear multipliers (exact gradients in the Warp solver), TAU, R_UP, BACK are fixed shape
constants.  kappa_o = kappa_s = 0 is V1/V0.5."""
from __future__ import annotations

import math

import numpy as np

TAU, R_UP, BACK = 1.0, 0.10, 1.0


class CircField:
    """Unit circulation fields on a solver grid (cell centres x, y; cup mask)."""

    def __init__(self, x, y, mask, h, tau=TAU, r_up=R_UP, back=BACK):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        self.x, self.y, self.m, self.h = x, y, mask, h
        self.tau, self.r_up, self.back = tau, r_up, back
        N = mask.shape[0]
        idx = -np.ones((N, N), int)
        idx[mask] = np.arange(mask.sum())
        fx = np.zeros((N, N), bool); fx[:, :-1] = mask[:, :-1] & mask[:, 1:]
        fy = np.zeros((N, N), bool); fy[:-1, :] = mask[:-1, :] & mask[1:, :]
        self.fx, self.fy = fx, fy
        r, c, v = [], [], []
        for jj, ii in zip(*np.where(fx)):
            a, b = idx[jj, ii], idx[jj, ii + 1]; r += [a, b, a, b]; c += [a, b, b, a]; v += [-1, -1, 1, 1]
        for jj, ii in zip(*np.where(fy)):
            a, b = idx[jj, ii], idx[jj + 1, ii]; r += [a, b, a, b]; c += [a, b, b, a]; v += [-1, -1, 1, 1]
        n = int(mask.sum())
        lap = sp.csr_matrix((v, (r, c)), shape=(n, n)) / (h * h)
        self.lu = spl.splu((lap - 1e-6 * sp.eye(n)).tocsc())
        pad = np.pad(mask, 1)
        wall = mask & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
        self.wall = wall.astype(float) / (wall.sum() * h * h)
        self.reset()

    def reset(self):
        N = self.m.shape[0]
        self.V = np.zeros((2, N, N))
        self.K = np.zeros((N, N))
        self._last = None

    def unit(self, x_hit, u_in):
        """(grad phi x, grad phi y, K_up) for a unit flux at landing point x_hit, stream direction u_in; None outside."""
        h, m = self.h, self.m
        x, y = float(x_hit[0]), float(x_hit[1])
        un = math.hypot(float(u_in[0]), float(u_in[1]))
        if self.back and un > 1e-9:
            x -= self.back * self.r_up * u_in[0] / un
            y -= self.back * self.r_up * u_in[1] / un
        rr = np.hypot(self.x - x, self.y - y) / self.r_up
        K = np.where(m, np.maximum(0.0, 1.0 - rr * rr) ** 2, 0.0)
        if K.sum() <= 0:
            return None
        K = K / (K.sum() * h * h)
        N = m.shape[0]
        phi = np.zeros((N, N))
        phi[m] = self.lu.solve((K - self.wall)[m])
        gx, gy, nx, ny = (np.zeros((N, N)) for _ in range(4))
        d = (phi[:, 1:] - phi[:, :-1]) / h * self.fx[:, :-1]
        gx[:, :-1] += d; gx[:, 1:] += d; nx[:, :-1] += self.fx[:, :-1]; nx[:, 1:] += self.fx[:, :-1]
        d = (phi[1:, :] - phi[:-1, :]) / h * self.fy[:-1, :]
        gy[:-1, :] += d; gy[1:, :] += d; ny[:-1, :] += self.fy[:-1, :]; ny[1:, :] += self.fy[:-1, :]
        return (np.where(nx > 0, gx / np.maximum(nx, 1), 0.0), np.where(ny > 0, gy / np.maximum(ny, 1), 0.0), K)

    def step(self, frame_dt, active, Q, x_hit=None, u_in=None):
        """Advance the relaxed unit fields by one frame; returns (V (2, N, N), K (N, N)) for this frame."""
        if active and Q > 0 and x_hit is not None:
            u = self.unit(x_hit, u_in)
            if u is not None:
                self._last = u
        e = math.exp(-frame_dt / self.tau)
        a = Q if active else 0.0
        if self._last is not None:
            tx, ty, K = self._last
            self.V[0] = e * self.V[0] + (1 - e) * a * tx
            self.V[1] = e * self.V[1] + (1 - e) * a * ty
            self.K = e * self.K + (1 - e) * a * K
        else:
            self.V *= e
            self.K *= e
        return self.V, self.K
