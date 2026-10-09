"""V1 surface-layer latte-art model, IMEX + Rusanov discretisation.

State per cell (cell averages): l (effective layer), qx, qy (layer momentum), m (new milk).
Units: length = cup diameter D_L, time = seconds.

Splitting per sub-step (inlet fields s, Lambda, v* frozen within a control frame):
  A  explicit momentum advection, Rusanov flux, alpha = 2 max(|u_n,L|, |u_n,R|)
  B  implicit mass + pressure + drag/traction -> SPD elliptic equation for l^{n+1}, PCG
  C  milk transported with the SAME face flux G_f, upwind c_f with van Leer limiter
  D  implicit viscosity (per velocity component) and implicit mixing, l frozen at l^{n+1}
"""
from __future__ import annotations
from . import kernels as K

import math
from dataclasses import dataclass, field, asdict

import numpy as np


# --------------------------------------------------------------------------- params
@dataclass
class Params:
    cp: float = 0.30        # effective pressure response  [D_L/s]
    beta: float = 3.0       # momentum damping             [1/s]
    nu: float = 1e-3        # viscosity                    [D_L^2/s]
    D: float = 1e-7         # milk mixing                  [D_L^2/s]
    kappa_Q: float = 125.0  # received volume -> layer     [1/D_L]
    kappa_c: float = 1.0    # traction of the non-deposited part
    kappa_t: float = 0.7    # tangential (material velocity) share of v*
    kappa_r: float = 0.3    # radial return-flow share of v*
    B_dep: float = 240.0    # deposition switch scale      [D_L/s^2]
    p_dep: float = 2.0      # deposition switch exponent
    return_law: str = "v1"  # "v1": chi*kappa_r*U_perp ; "v05": kappa_r*U_perp*chi^(1/(2p)) capped
    # "push first, whiten later": the deposited milk enters a sub-surface reservoir and surfaces after tau_d
    # (its impact momentum acts immediately); tau_d = 0 reproduces the instantaneous closure
    # ---- closure "skin" (bulk + submerged foam + surface foam), replaces chi-splitting, kappa_r, return law and delay
    closure: str = "chi"    # "chi": original single-layer closure ; "skin": see SkinNotes in PITCHER_MODEL.md section 15
    g_red: float = 75.0     # reduced gravity of the milk foam g(1 - rho_f/rho)   [D_L/s^2]  (6 m/s^2 for rho_f/rho = 0.4)
    Fr_c2: float = 23.0     # critical densimetric Froude number^2 of foam survival: chi = 1/(1+(U^2/(g' d Fr_c^2))^p_dep)
    phi_foam: float = 0.5   # foam volume fraction of the poured milk
    m_opaque: float = 0.02  # surface-foam thickness (liquid-depth units) that looks fully white
    tau_d: float = 0.0      # surfacing time constant [s]; 0 = instantaneous closure.  Tested variants that did not
                            # help were removed: a delay growing with U_perp (over-pushed the high cut) and a
                            # sub-surface spreading D_sub (smeared the layers).
    sub_advect: float = 0.3     # fraction of the surface velocity that carries the submerged plume (0 = it stays
                                # where it was poured; 1 = it follows the pushed layer and fills the gap again)
    sub_pressure: float = 0.5   # the submerged milk is a mound under the surface: it enters the pressure as
                                # c_p^2 l grad(l + sub_pressure*m_sub) and pushes the old layer outward before it surfaces

    def as_dict(self):
        return asdict(self)


@dataclass
class Numerics:
    N: int = 256
    cup_radius: float = 0.49
    cfl: float = 0.5
    frame_dt: float = 1.0 / 120.0   # control frame; inlet frozen inside a frame
    max_substeps: int = 64
    cg_tol: float = 1e-10
    cg_maxiter: int = 500
    kernel_quadrature: int = 3
    scan_safety: float = 0.5        # frame_dt <= scan_safety * min(h, r1, r2) / scan_speed
    explicit_mixing_limit: float = 0.05   # D*dt/h^2 below this: explicit mixing update instead of a PCG solve
    explicit_visc_limit: float = 0.2      # nu*dt/h^2 below this: explicit viscosity update (2-D limit 0.25)
    visc_tol: float = 1e-8                # PCG tolerance for the (non-stiff) viscosity solves


@dataclass
class Inlet:
    """Arrival record at one instant, already in the cup frame R."""
    active: bool = False
    x_hit: tuple = (0.0, 0.0)      # hit point (x, y)
    S_eff: float = 0.0             # kappa_Q * Q_received   [1/s * D_L^2]  (effective inlet rate)
    u_in: tuple = (0.0, 0.0)       # horizontal material arrival velocity [D_L/s]
    U_perp: float = 0.0            # normal arrival speed [D_L/s]
    d_jet: float = 0.05            # jet diameter (cross-section equivalent) [D_L]
    r1: float = 0.04               # footprint ellipse semi-axes [D_L]
    r2: float = 0.04
    phi: float = 0.0               # footprint orientation [rad]
    scan_speed: float = 0.0        # bound on the hit-point speed [D_L/s] (limits the control-frame length)


# --------------------------------------------------------------------------- grid
class Grid:
    def __init__(self, N: int, R: float):
        self.N = N
        self.h = 1.0 / N
        c = (np.arange(N) + 0.5) * self.h - 0.5
        self.x = c[None, :].repeat(N, 0)   # [iy, ix]
        self.y = c[:, None].repeat(N, 1)
        self.r = np.hypot(self.x, self.y)
        self.mask = self.r <= R
        self.fx = self.mask[:, :-1] & self.mask[:, 1:]     # x-faces (N, N-1)
        self.fy = self.mask[:-1, :] & self.mask[1:, :]     # y-faces (N-1, N)
        self.area = self.h * self.h
        self.n_cells = int(self.mask.sum())
        # number of active faces per cell (for averaging cell gradients)
        self.zero = np.zeros((N, N)); self.one = np.ones((N, N))
        self.fx_h = self.fx / self.h   # face-gradient factors (inactive faces -> 0)
        self.fy_h = self.fy / self.h
        self.nfx = np.zeros((N, N)); self.nfx[:, :-1] += self.fx; self.nfx[:, 1:] += self.fx
        self.nfy = np.zeros((N, N)); self.nfy[:-1, :] += self.fy; self.nfy[1:, :] += self.fy

    def div(self, Fx, Fy):
        """Divergence of face fluxes (normal flux density per face). Fx (N,N-1), Fy (N-1,N)."""
        return K.div(np.ascontiguousarray(Fx), np.ascontiguousarray(Fy), 1.0 / self.h, np.empty((self.N, self.N)))

    def face_avg(self, a):
        return 0.5 * (a[:, :-1] + a[:, 1:]) * self.fx, 0.5 * (a[:-1, :] + a[1:, :]) * self.fy

    def face_grad(self, a):
        gx = np.subtract(a[:, 1:], a[:, :-1]); gx *= self.fx_h
        gy = np.subtract(a[1:, :], a[:-1, :]); gy *= self.fy_h
        return gx, gy

    def cell_grad(self, a):
        """Cell gradient = average of the two face gradients (inactive faces count as zero)."""
        gx, gy = self.face_grad(a)
        cx = np.zeros((self.N, self.N)); cx[:, :-1] += gx; cx[:, 1:] += gx
        cy = np.zeros((self.N, self.N)); cy[:-1, :] += gy; cy[1:, :] += gy
        return cx / np.maximum(self.nfx, 1), cy / np.maximum(self.nfy, 1)

    def diffusion(self, kx, ky, phi):
        """div( k grad phi ) with face coefficients kx (N,N-1), ky (N-1,N)."""
        return K.apply_diffusion(phi, np.ascontiguousarray(kx), np.ascontiguousarray(ky), self.fx, self.fy,
                                 self.zero, -1.0, 1.0 / (self.h * self.h), np.empty((self.N, self.N)))

    def mass_operator(self, kx, ky, dt, phi):
        """phi - dt * div( k grad phi )  (the implicit layer operator), one fused pass."""
        return K.apply_diffusion(phi, kx, ky, self.fx, self.fy, self.one, dt, 1.0 / (self.h * self.h),
                                 np.empty((self.N, self.N)))


# --------------------------------------------------------------------------- PCG
def pcg(apply_A, b, diag, x0, tol=1e-10, maxiter=500):
    x = x0.copy()
    r = b - apply_A(x)
    p = r / diag
    rz = K.dot(r, p)
    bnorm = math.sqrt(K.dot(b, b)) + 1e-300
    it = 0
    for it in range(1, maxiter + 1):
        Ap = apply_A(p)
        alpha = rz / (K.dot(p, Ap) + 1e-300)
        K.axpy(alpha, p, x)
        K.axpy(-alpha, Ap, r)
        if math.sqrt(K.dot(r, r)) <= tol * bnorm:
            break
        rz_new = K.dot(r, r / diag)
        K.zbp(r, diag, rz_new / rz, p)
        rz = rz_new
    return x, it


# --------------------------------------------------------------------------- limiter
def van_leer(r):
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def limited_face_value(c, G, grid, axis):
    """Upwind face value of c with van Leer limited linear reconstruction.
    axis=1: x-faces (N, N-1), G>0 means flow from left (ix) to right (ix+1).
    axis=0: y-faces (N-1, N)."""
    if axis == 1:
        cm = c
        dcf = (cm[:, 1:] - cm[:, :-1]) * grid.fx          # difference across each x-face
        # difference across the face to the left of cell ix: pad with zero at the boundary
        dleft = np.zeros_like(c); dleft[:, 1:] = dcf          # d_{ix-1/2}
        dright = np.zeros_like(c); dright[:, :-1] = dcf       # d_{ix+1/2}
        cL = cm[:, :-1]; cR = cm[:, 1:]
        # flow L->R: c_f = c_L + 0.5 phi(r_L) * d_f, r_L = d_{L-1/2}/d_f
        dface = dcf
        eps = 1e-14
        rL = dleft[:, :-1] / np.where(np.abs(dface) > eps, dface, eps)
        rR = dright[:, 1:] / np.where(np.abs(dface) > eps, dface, eps)
        rL = np.where(np.abs(dface) > eps, rL, 0.0)
        rR = np.where(np.abs(dface) > eps, rR, 0.0)
        cfL = cL + 0.5 * van_leer(rL) * dface
        cfR = cR - 0.5 * van_leer(rR) * dface
    else:
        cm = c
        dcf = (cm[1:, :] - cm[:-1, :]) * grid.fy
        dleft = np.zeros_like(c); dleft[1:, :] = dcf
        dright = np.zeros_like(c); dright[:-1, :] = dcf
        cL = cm[:-1, :]; cR = cm[1:, :]
        dface = dcf
        eps = 1e-14
        rL = dleft[:-1, :] / np.where(np.abs(dface) > eps, dface, eps)
        rR = dright[1:, :] / np.where(np.abs(dface) > eps, dface, eps)
        rL = np.where(np.abs(dface) > eps, rL, 0.0)
        rR = np.where(np.abs(dface) > eps, rR, 0.0)
        cfL = cL + 0.5 * van_leer(rL) * dface
        cfR = cR - 0.5 * van_leer(rR) * dface
    cf = np.where(G > 0, cfL, cfR)
    lo = np.minimum(cL, cR); hi = np.maximum(cL, cR)
    return np.clip(cf, lo, hi)


# --------------------------------------------------------------------------- solver
class Solver:
    def __init__(self, params: Params, num: Numerics):
        self.P = params
        self.num = num
        self.g = Grid(num.N, num.cup_radius)
        g = self.g
        self.l = np.ones((g.N, g.N))
        self.qx = np.zeros((g.N, g.N))
        self.qy = np.zeros((g.N, g.N))
        self.m = np.zeros((g.N, g.N))
        self.t = 0.0
        self.deposited = 0.0           # cumulative integral of s over domain and time (surfaced milk)
        self.msub = np.zeros((g.N, g.N))   # sub-surface milk waiting to surface (delay closure)
        self.tau_cur = None            # surfacing time constant of the current/last inlet
        self.f = np.zeros((g.N, g.N))      # skin closure: submerged foam (liquid-depth units)
        self.ft = np.zeros((g.N, g.N))     # skin closure: f * (mean remaining rise time) to carry tau with the foam
        self.foam_src = None; self.tau_new = 0.0
        self.stats = dict(steps=0, frames=0, cg_mass=0, cg_visc=0, cg_mix=0,
                          clip_mass=0.0, max_substeps=0, max_u=0.0, retries=0)
        self.l_floor = 1e-3
        self._kernel_cache = {}

    # ----- inlet fields -----------------------------------------------------
    def kernel(self, inlet: Inlet):
        """Normalised compact ellipse kernel K on cells, sum(K)*area = 1 inside the cup.
        Evaluated only on the bounding box of the footprint; sub-cell quadrature adapts to thin footprints."""
        g = self.g
        r1, r2, phi = max(inlet.r1, 1.5 * g.h), max(inlet.r2, 1.5 * g.h), inlet.phi
        x0, y0 = inlet.x_hit
        q = max(self.num.kernel_quadrature, int(math.ceil(4 * g.h / min(r1, r2))))
        q = min(q, 12)
        R = max(r1, r2)
        i0 = max(0, int((x0 - R + 0.5) / g.h) - 1); i1 = min(g.N, int((x0 + R + 0.5) / g.h) + 2)
        j0 = max(0, int((y0 - R + 0.5) / g.h) - 1); j1 = min(g.N, int((y0 + R + 0.5) / g.h) + 2)
        K = np.zeros((g.N, g.N))
        if i1 <= i0 or j1 <= j0:
            raise ValueError("inlet footprint outside the grid")
        xs = g.x[j0:j1, i0:i1]; ys = g.y[j0:j1, i0:i1]
        offs = (np.arange(q) + 0.5) / q - 0.5
        cphi, sphi = math.cos(phi), math.sin(phi)
        Kl = np.zeros_like(xs)
        for ox in offs:
            for oy in offs:
                dx = xs + ox * g.h - x0
                dy = ys + oy * g.h - y0
                a = (cphi * dx + sphi * dy) / r1
                b = (-sphi * dx + cphi * dy) / r2
                Kl += np.maximum(0.0, 1.0 - a * a - b * b) ** 2
        Kl *= g.mask[j0:j1, i0:i1]
        K[j0:j1, i0:i1] = Kl
        Z = K.sum() * g.area
        if Z <= 0:
            raise ValueError("inlet footprint does not intersect the cup")
        return K / Z, Z

    def inlet_fields(self, inlet: Inlet):
        P = self.P
        g = self.g
        if not inlet.active or inlet.S_eff <= 0:
            z = np.zeros((g.N, g.N))
            self.foam_src = None
            return z, z, z, z, 0.0
        K, _ = self.kernel(inlet)
        d = max(inlet.d_jet, 1e-6)
        if P.closure == "skin":
            # the whole poured volume enters the bulk at once (it drives the push through the pressure); the foam part that
            # survives the impact, chi(Fr), is submerged and surfaces after the fountain rise time 2 U_perp / g'
            Fr2 = inlet.U_perp ** 2 / (P.g_red * d)
            chi = 1.0 / (1.0 + (Fr2 / P.Fr_c2) ** P.p_dep)
            s = inlet.S_eff * K
            Lam = P.kappa_c * inlet.S_eff * K
            self.foam_src = chi * P.phi_foam * s
            self.tau_new = max(2.0 * inlet.U_perp / P.g_red, 1e-3)
            vsx = P.kappa_t * inlet.u_in[0] * np.ones_like(K)
            vsy = P.kappa_t * inlet.u_in[1] * np.ones_like(K)
            return s, Lam, vsx, vsy, chi
        if P.tau_d > 0:
            self.tau_cur = P.tau_d
        zc = inlet.U_perp ** 2 / (P.B_dep * d)
        chi = 1.0 / (1.0 + zc ** P.p_dep)
        s = chi * inlet.S_eff * K
        Lam = P.kappa_c * (1.0 - chi) * inlet.S_eff * K
        # target velocity
        dx = g.x - inlet.x_hit[0]
        dy = g.y - inlet.x_hit[1]
        Rdep = math.sqrt(max(inlet.r1 * inlet.r2, 1e-12))
        den = np.sqrt(Rdep * Rdep + dx * dx + dy * dy)
        if P.return_law == "v05":
            VR = inlet.U_perp * chi ** (1.0 / (2.0 * P.p_dep))
            VR = min(VR, inlet.U_perp, math.sqrt(P.B_dep * d))
            rad = P.kappa_r * VR
        else:
            rad = chi * P.kappa_r * inlet.U_perp
        vsx = P.kappa_t * inlet.u_in[0] + rad * dx / den
        vsy = P.kappa_t * inlet.u_in[1] + rad * dy / den
        return s, Lam, vsx, vsy, chi

    # ----- one sub-step --------------------------------------------------------
    def substep(self, dt, s, Lam, vsx, vsy):
        P, g, h = self.P, self.g, self.g.h
        l, qx, qy, m = self.l, self.qx, self.qy, self.m
        mask = g.mask
        lsafe = np.where(mask, l, 1.0)
        ux = np.where(mask, qx / lsafe, 0.0)
        uy = np.where(mask, qy / lsafe, 0.0)
        c = np.where(mask, m / lsafe, 0.0)

        # ---- delay closure: impact momentum now, white later --------------------
        s_mom = s
        msub_new = None
        if P.tau_d > 0 and (self.tau_cur is not None):
            tau = self.tau_cur
            tot = self.msub + dt * s                       # what is below the surface before surfacing
            msub_new = tot / (1.0 + dt / tau)              # implicit decay
            s = (tot - msub_new) / dt                      # surfacing rate = mass source of the layer
            # the submerged plume is carried by the flow (first-order upwind with the surface velocity); without this
            # the reservoir stays where it was poured and surfaces as a straight bar along the path
            if P.sub_advect > 0:
                ufx = P.sub_advect * 0.5 * (ux[:, 1:] + ux[:, :-1]) * g.fx
                ufy = P.sub_advect * 0.5 * (uy[1:, :] + uy[:-1, :]) * g.fy
                Fx = ufx * np.where(ufx > 0, msub_new[:, :-1], msub_new[:, 1:])
                Fy = ufy * np.where(ufy > 0, msub_new[:-1, :], msub_new[1:, :])
                msub_new = np.where(mask, np.maximum(msub_new - dt * g.div(Fx, Fy), 0.0), 0.0)

        # ---- A: explicit Rusanov momentum advection (l frozen) --------------
        N = g.N
        Fx_qx = np.empty((N, N - 1)); Fx_qy = np.empty((N, N - 1)); Fy_qx = np.empty((N - 1, N)); Fy_qy = np.empty((N - 1, N))
        K.rusanov_fluxes(qx, qy, ux, uy, g.fx, g.fy, Fx_qx, Fx_qy, Fy_qx, Fy_qy)
        qx_s = qx - dt * g.div(Fx_qx, Fy_qx)
        qy_s = qy - dt * g.div(Fx_qy, Fy_qy)

        # ---- B: implicit mass + pressure + drag/traction --------------------
        a_x = np.empty((N, N)); a_y = np.empty((N, N)); coef = np.empty((N, N))
        K.stepB_prepare(qx_s, qy_s, s_mom, Lam, vsx, vsy, lsafe, P.beta, P.cp ** 2, dt, a_x, a_y, coef)
        if msub_new is not None and P.sub_pressure > 0:
            # bathymetry-like push of the submerged mound (explicit, known field): a -= dt (c_p^2 l/theta) grad(m_sub)
            gbx, gby = g.cell_grad(msub_new)
            a_x = a_x - dt * coef * P.sub_pressure * gbx
            a_y = a_y - dt * coef * P.sub_pressure * gby
        # explicit part of face flux
        Gex = np.empty((N, N - 1)); Gey = np.empty((N - 1, N))
        K.mass_fluxes(a_x, a_y, l, lsafe, g.fx, g.fy, Gex, Gey, None, None)
        kx = np.empty((N, N - 1)); ky = np.empty((N - 1, N)); diag = np.empty((N, N))
        K.face_coef_and_diag(coef, g.fx, g.fy, dt, 1.0 / (h * h), kx, ky, diag)   # k_f = dt avg(c_p^2 l^n/theta)
        rhs = l + dt * s - dt * g.div(Gex, Gey)

        def A_mass(phi):
            return g.mass_operator(kx, ky, dt, phi)

        l_new, it = pcg(A_mass, rhs, diag, l, self.num.cg_tol, self.num.cg_maxiter)
        self.stats["cg_mass"] += it
        Gx = np.empty((N, N - 1)); Gy = np.empty((N - 1, N))
        K.finish_B(Gex, Gey, kx, ky, l_new, g.fx, g.fy, 1.0 / h, Gx, Gy)   # k_f (l_R - l_L)/h added to Ge
        l_new = l + dt * s - dt * g.div(Gx, Gy)          # conservative update with the final flux
        qx_new = np.empty((N, N)); qy_new = np.empty((N, N))
        K.q_from_gradient(a_x, a_y, coef, l_new, g.fx, g.fy, g.nfx, g.nfy, dt, 1.0 / h, qx_new, qy_new)
        if msub_new is not None:
            # surfacing milk joins the layer with the local surface velocity (no extra momentum of its own)
            qx_new = qx_new + dt * s * ux
            qy_new = qy_new + dt * s * uy

        # ---- C: milk with the same face flux ---------------------------------
        cfx = K.limited_face_x(c, Gx, g.fx, np.empty((N, N - 1)))
        cfy = K.limited_face_y(c, Gy, g.fy, np.empty((N - 1, N)))
        if P.closure == "skin":
            # submerged foam and its rise-time moment move with the same face flux as the bulk (van Leer limited)
            fl = np.where(mask, self.f / lsafe, 0.0); tl = np.where(mask, self.ft / lsafe, 0.0)
            ffx = K.limited_face_x(fl, Gx, g.fx, np.empty((N, N - 1))); ffy = K.limited_face_y(fl, Gy, g.fy, np.empty((N - 1, N)))
            tfx = K.limited_face_x(tl, Gx, g.fx, np.empty((N, N - 1))); tfy = K.limited_face_y(tl, Gy, g.fy, np.empty((N - 1, N)))
            f1 = self.f - dt * g.div(Gx * ffx, Gy * ffy)
            t1 = self.ft - dt * g.div(Gx * tfx, Gy * tfy)
            if self.foam_src is not None:
                f1 = f1 + dt * self.foam_src
                t1 = t1 + dt * self.foam_src * self.tau_new
            f1 = np.where(mask, np.maximum(f1, 0.0), 0.0); t1 = np.where(mask, np.maximum(t1, 0.0), 0.0)
            tau_f = np.where(f1 > 1e-14, t1 / np.maximum(f1, 1e-300), 1.0)
            tau_f = np.maximum(tau_f, 1e-3)
            surf = f1 * (1.0 - np.exp(-dt / tau_f))          # foam reaching the surface this step
            f_new = f1 - surf
            ft_new = tau_f * f_new
            s_m = surf / dt
        else:
            s_m = s
        m_new = m + dt * s_m - dt * g.div(Gx * cfx, Gy * cfy)
        # boundedness guard (should be inactive under the material CFL)
        over = np.maximum(m_new - l_new, 0.0) + np.maximum(-m_new, 0.0)
        self.stats["clip_mass"] += float(over[mask].sum() * g.area)
        m_new = np.clip(m_new, 0.0, l_new)

        # outside the cup keep the reference state
        l_new = np.where(mask, l_new, 1.0)
        qx_new = np.where(mask, qx_new, 0.0)
        qy_new = np.where(mask, qy_new, 0.0)
        m_new = np.where(mask, m_new, 0.0)
        lsn = np.where(mask, l_new, 1.0)

        # ---- D: implicit viscosity and mixing (l frozen at l^{n+1}) ---------
        if P.nu > 0 and P.nu * dt / (h * h) < self.num.explicit_visc_limit:
            # explicit, stable for nu*dt/h^2 < 1/4 (2-D); l frozen at l^{n+1}
            lfx, lfy = g.face_avg(lsn)
            kvx, kvy = P.nu * lfx, P.nu * lfy
            ux_new = qx_new / lsn
            uy_new = qy_new / lsn
            qx_new = np.where(mask, qx_new + dt * g.diffusion(kvx, kvy, ux_new), 0.0)
            qy_new = np.where(mask, qy_new + dt * g.diffusion(kvx, kvy, uy_new), 0.0)
        elif P.nu > 0:
            lfx, lfy = g.face_avg(lsn)
            kvx, kvy = P.nu * lfx, P.nu * lfy
            dvis = lsn + dt / (h * h) * (np.pad(kvx, ((0, 0), (1, 0))) + np.pad(kvx, ((0, 0), (0, 1)))
                                         + np.pad(kvy, ((1, 0), (0, 0))) + np.pad(kvy, ((0, 1), (0, 0))))

            def A_visc(phi):
                return lsn * phi - dt * g.diffusion(kvx, kvy, phi)

            ux_new = qx_new / lsn
            uy_new = qy_new / lsn
            ux_new, it1 = pcg(A_visc, lsn * ux_new, dvis, ux_new, self.num.visc_tol, self.num.cg_maxiter)
            uy_new, it2 = pcg(A_visc, lsn * uy_new, dvis, uy_new, self.num.visc_tol, self.num.cg_maxiter)
            self.stats["cg_visc"] += it1 + it2
            qx_new = np.where(mask, lsn * ux_new, 0.0)
            qy_new = np.where(mask, lsn * uy_new, 0.0)
        if P.D > 0 and P.D * dt / (h * h) < self.num.explicit_mixing_limit:
            lfx, lfy = g.face_avg(lsn)
            c_new = np.where(mask, m_new / lsn, 0.0)
            m_new = m_new + dt * g.diffusion(P.D * lfx, P.D * lfy, c_new)
            m_new = np.where(mask, np.clip(m_new, 0.0, l_new), 0.0)
        elif P.D > 0:
            lfx, lfy = g.face_avg(lsn)
            kmx, kmy = P.D * lfx, P.D * lfy
            dmix = lsn + dt / (h * h) * (np.pad(kmx, ((0, 0), (1, 0))) + np.pad(kmx, ((0, 0), (0, 1)))
                                         + np.pad(kmy, ((1, 0), (0, 0))) + np.pad(kmy, ((0, 1), (0, 0))))

            def A_mix(phi):
                return lsn * phi - dt * g.diffusion(kmx, kmy, phi)

            c_new = m_new / lsn
            c_new, it3 = pcg(A_mix, lsn * c_new, dmix, c_new, self.num.cg_tol, self.num.cg_maxiter)
            self.stats["cg_mix"] += it3
            m_new = np.where(mask, lsn * np.clip(c_new, 0.0, 1.0), 0.0)

        # accept only finite, positive-layer states; otherwise the frame is retried with a smaller step
        if not (np.isfinite(l_new).all() and np.isfinite(qx_new).all() and np.isfinite(qy_new).all()
                and np.isfinite(m_new).all() and l_new[mask].min() > self.l_floor):
            return False
        self.l, self.qx, self.qy, self.m = l_new, qx_new, qy_new, m_new
        if P.closure == "skin":
            self.f, self.ft = f_new, ft_new
        if msub_new is not None:
            self.msub = msub_new
        self.deposited += float(s[mask].sum()) * g.area * dt
        self.stats["steps"] += 1
        return True

    # ----- one control frame ---------------------------------------------------
    def advance_frame(self, inlet: Inlet, frame_dt=None):
        frame_dt = frame_dt or self.num.frame_dt
        if inlet.active and inlet.S_eff > 0 and inlet.scan_speed > 0:
            lim = self.num.scan_safety * min(self.g.h, inlet.r1, inlet.r2) / inlet.scan_speed
            frame_dt = min(frame_dt, max(lim, 1e-4))
        s, Lam, vsx, vsy, chi = self.inlet_fields(inlet)
        g = self.g
        lsafe = np.where(g.mask, self.l, 1.0)
        umax = float(np.max(np.hypot(self.qx, self.qy) / lsafe))
        # include the inlet target speed in the material-speed bound (it is reached quickly)
        if inlet.active and inlet.S_eff > 0:
            umax = max(umax, float(np.max(np.hypot(vsx, vsy) * (s + Lam > 0))))
        self.stats["max_u"] = max(self.stats["max_u"], umax)
        dt_cfl = self.num.cfl * g.h / (2.0 * max(umax, 1e-9))
        nsub = int(math.ceil(frame_dt / dt_cfl))
        nsub = max(1, min(nsub, self.num.max_substeps))
        saved = (self.l.copy(), self.qx.copy(), self.qy.copy(), self.m.copy(), self.deposited, self.msub.copy(),
                 self.f.copy(), self.ft.copy())
        for attempt in range(8):
            dt = frame_dt / nsub
            ok = True
            for _ in range(nsub):
                if not self.substep(dt, s, Lam, vsx, vsy):
                    ok = False
                    break
            if ok:
                break
            # restore the committed state and retry the whole frame with a smaller sub-step
            self.l, self.qx, self.qy, self.m = (a.copy() for a in saved[:4])
            self.deposited = saved[4]
            self.msub = saved[5].copy()
            self.f, self.ft = saved[6].copy(), saved[7].copy()
            self.stats["retries"] += 1
            nsub *= 2
        else:
            raise RuntimeError(f"frame at t={self.t:.4f} failed after retries (nsub={nsub})")
        self.stats["max_substeps"] = max(self.stats["max_substeps"], nsub)
        self.t += frame_dt
        self.stats["frames"] += 1
        return chi, nsub

    # ----- diagnostics ----------------------------------------------------------
    def host_state(self):
        return dict(l=self.l, qx=self.qx, qy=self.qy, m=self.m)

    def ledger(self):
        g = self.g
        L = float(self.l[g.mask].sum() * g.area)
        M = float(self.m[g.mask].sum() * g.area)
        R = float(self.msub[g.mask].sum() * g.area)
        return dict(t=self.t, layer=L, milk=M, brown=L - M, deposited=self.deposited, reservoir=R,
                    layer_error=L - (g.n_cells * g.area + self.deposited),
                    milk_error=M - self.deposited)

    def concentration(self):
        """Visible whiteness in [0, 1]: milk fraction m/l (chi closure) or surface-foam thickness over the opaque
        thickness (skin closure)."""
        g = self.g
        if self.P.closure == "skin":
            return np.where(g.mask, np.clip(self.m / self.P.m_opaque, 0.0, 1.0), np.nan)
        return np.where(g.mask, self.m / np.where(g.mask, self.l, 1.0), np.nan)

    def energy(self):
        g = self.g
        lsafe = np.where(g.mask, self.l, 1.0)
        ke = 0.5 * (self.qx ** 2 + self.qy ** 2) / lsafe
        pe = 0.5 * self.P.cp ** 2 * self.l ** 2
        return float(ke[g.mask].sum() * g.area), float(pe[g.mask].sum() * g.area)
