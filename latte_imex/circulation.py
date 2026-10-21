"""Surface current of the cup driven by the stream's horizontal momentum: the water the film floats on.

V1 3.3 drags the film towards a lower layer at rest.  Measured surface velocities of the real pours (optical flow of the
rectified top view, rig/video_flow.py) show instead a broad current from the pitcher side across the whole cup, peaked
on the centre line, with no surface return flow, that dies within about a second when the flow stops: the plunging
stream pushes the surface water forward, which upwells behind the stream (the pitcher side) and goes down ahead of it.
The film itself (crema and microfoam) is nearly incompressible, so it can follow that current only if its area is
created where the current diverges and removed where it converges: the current's divergence is the film's area budget.

Current (2-D, cell centres, one-way coupled: the film's reaction on the water is ~ rho_f h_f / (rho delta) ~ 0.1):
    dU/dt = kappa_M F - r U + nu (lap U + grad div U),   nu = r L^2,   U.n = 0 at the wall (normal penalty), free slip,
    F = Q Z u_in K_F(x - x_hit):  the horizontal momentum flux the stream brings in (received flow Q Z, horizontal arrival
    velocity u_in, smooth kernel K_F of radius R_F integrating to 1).
U is linear in F; it is represented by the steady response U_ss(F) relaxed with the spin-down rate r,
    U <- e U + (1 - e) U_ss(F),   e = exp(-r dt_frame)
(exact for the slowest mode; the faster small scales are slaved to it).  sigma = div U (face-averaged fluxes, closed
wall faces: its integral over the cup is 0 to round-off); upwelling sigma+ = max(sigma, 0), downwelling sigma- = max(-sigma, 0).

Film coupling (solver.py, diff/wsolver.py), all with the unit current (kappa_M = 1) of this module:
    drag towards kappa_M U                      -beta (q - l kappa_M U)
    vertical exchange after each sub-step, on the brown part b = l - m only (the buoyant microfoam raft, rho_f / rho ~
    0.4, floats over the exchange: the water surfaces beside it and goes down beside it):
        db/dt = -W b + kappa_M sigma+ (1 - c),   W = kappa_M sigma- + kappa_w kappa_Q W_n
        dm/dt = -eps_f W m                       (eps_f: fraction of the foam that does go down; physical value ~ 0)
    the exchanged material moves with the film (u unchanged).
W_n: the received flow's area (abar: the relaxed received flow) leaving where the current goes down (sigma- / its
integral; along the wall when there is no current): kappa_w = 1 removes the area the inlet adds.  kappa_M, eps_f,
kappa_w are the closure parameters (multipliers of these unit fields in the Warp solver: exact gradients); R_C, L_C,
R_F are the default shape constants (calibrated as closure values circ_r, circ_L, circ_rf: solver.Params,
diff/run.DiffPour(circ_shape=dict(r, L, r_f)), diff/cmafit.py --circ-r --circ-L --circ-rf; not differentiated).
kappa_M = kappa_w = 0 is V1/V0.5."""
from __future__ import annotations

import math

import numpy as np

R_C = 1.0                # spin-down rate of the current [1/s] (old optical flow: the mean surface current dies within ~1 s;
                         # the current frames: the broad forward motion stops within ~0.35 s of a stream stop -> circ_r)
L_C = 0.4                # momentum spreading length sqrt(nu / r) [solver units; cup diameter 0.98]
R_F = 0.06               # radius of the momentum kernel around the landing point
PEN = 50.0               # wall normal penalty (x nu / h^2)


class SurfaceCurrent:
    """Unit surface current on a solver grid (cell centres x, y; cup mask; cell size h)."""

    def __init__(self, x, y, mask, h, r=R_C, L=L_C, r_f=R_F):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        self.x, self.y, self.m, self.h = x, y, mask, h
        self.r, self.r_f = r, r_f
        N = mask.shape[0]
        idx = -np.ones((N, N), int)
        idx[mask] = np.arange(mask.sum())
        n = int(mask.sum())
        self.n = n
        fx = np.zeros((N, N), bool); fx[:, :-1] = mask[:, :-1] & mask[:, 1:]
        fy = np.zeros((N, N), bool); fy[:-1, :] = mask[:-1, :] & mask[1:, :]
        ax, bx = idx[:, :-1][fx[:, :-1]], idx[:, 1:][fx[:, :-1]]
        ay, by = idx[:-1, :][fy[:-1, :]], idx[1:, :][fy[:-1, :]]
        # scalar Neumann Laplacian
        a, b = np.r_[ax, ay], np.r_[bx, by]
        lap = sp.csr_matrix((np.r_[-np.ones(2 * len(a)), np.ones(2 * len(a))], (np.r_[a, b, a, b], np.r_[a, b, b, a])),
                            shape=(n, n)) / (h * h)
        # divergence from face-averaged fluxes (wall faces closed): sigma = Dx u + Dy v
        def div(a, b):
            rows = np.r_[a, a, b, b]
            cols = np.r_[a, b, a, b]
            vals = np.r_[np.full(2 * len(a), 0.5 / h), np.full(2 * len(a), -0.5 / h)]
            return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        self.D = sp.hstack([div(ax, bx), div(ay, by)]).tocsr()
        pad = np.pad(mask, 1)
        wall = mask & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
        self.wall = wall.astype(float) / (wall.sum() * h * h)
        rr = np.maximum(np.hypot(x, y), 1e-12)
        nx, ny, wl = (x / rr)[mask], (y / rr)[mask], wall[mask].astype(float)
        Pw = sp.bmat([[sp.diags(wl * nx * nx), sp.diags(wl * nx * ny)], [sp.diags(wl * ny * nx), sp.diags(wl * ny * ny)]])
        nu = r * L * L
        A = (r * sp.eye(2 * n) - nu * sp.block_diag([lap, lap]) + nu * (self.D.T @ self.D) + (PEN * nu / (h * h)) * Pw)
        self.lu = spl.splu(A.tocsc())
        self.reset()

    def reset(self):
        N = self.m.shape[0]
        self.U = np.zeros((2, N, N))
        self.sig = np.zeros((N, N))
        self.abar = 0.0

    def steady(self, x_hit, u_in):
        """Steady current and its divergence for a unit received flow landing at x_hit with horizontal velocity u_in
        (None if the kernel misses the cup or the stream has no horizontal velocity)."""
        if math.hypot(float(u_in[0]), float(u_in[1])) <= 1e-12:
            return None
        h, m = self.h, self.m
        K = np.where(m, np.maximum(0.0, 1.0 - ((self.x - x_hit[0]) ** 2 + (self.y - x_hit[1]) ** 2) / self.r_f ** 2) ** 2, 0.0)
        if K.sum() <= 0:
            return None
        K = K[m] / (K.sum() * h * h)
        sol = self.lu.solve(np.r_[K * float(u_in[0]), K * float(u_in[1])])
        N = m.shape[0]
        U = np.zeros((2, N, N))
        U[0][m], U[1][m] = sol[:self.n], sol[self.n:]
        sig = np.zeros((N, N))
        sig[m] = self.D @ sol
        return U, sig

    def step(self, frame_dt, active, Q, x_hit=None, u_in=None):
        """Advance the unit current by one frame (received flow Q Z, landing point x_hit, horizontal arrival velocity
        u_in); returns (U (2, N, N), sigma+ (N, N), sigma- (N, N), W_n (N, N))."""
        e = math.exp(-self.r * frame_dt)
        a = Q if (active and Q > 0) else 0.0
        st = self.steady(x_hit, u_in) if (a > 0 and x_hit is not None) else None
        self.U *= e
        self.sig *= e
        if st is not None:
            self.U += (1 - e) * a * st[0]
            self.sig += (1 - e) * a * st[1]
        self.abar = e * self.abar + (1 - e) * a
        up = np.where(self.m, np.maximum(self.sig, 0.0), 0.0)
        dn = np.where(self.m, np.maximum(-self.sig, 0.0), 0.0)
        tot = dn.sum() * self.h * self.h
        Wn = self.abar * (dn / tot if tot > 1e-12 * max(self.abar, 1e-30) else self.wall)
        return self.U, up, dn, Wn
