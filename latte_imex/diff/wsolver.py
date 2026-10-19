"""Differentiable port of solver.Solver to NVIDIA Warp (the path the real-pour runs use: closure 'chi', visc
'symmetric', offcup 'account', tau_d = 0), float64.

Forward = solver.Solver step for step (same splitting A-D, same stencils, PCG tolerances), so the two agree to the
CG tolerance (test_parity.py).  Gradients: Warp's kernel adjoints (wp.Tape) for every explicit stencil; the two
implicit solves (layer mass, symmetric viscosity) get the exact adjoint of a linear solve A(theta) x = b:
    A lambda = dL/dx  (A is SPD, same PCG),   dL/db += lambda,   dL/dtheta -= lambda^T (dA/dtheta) x
where the last term is the adjoint of the operator-apply kernels run on x with output adjoint -lambda (so every
coefficient that A depends on -- face coefficients, l, viscosity weights -- gets its gradient without hand-written
derivatives).  Memory: the state (l, qx, qy, m) is checkpointed at every control frame on the host; the backward pass
replays one frame at a time under a tape, from the last frame to the first.

Differentiable inputs: the closure parameters P (array, layout PARAMS) and the per-frame arrival record (x_hit, Q,
u_in, U_perp, d_jet, r1, r2, phi), so the same machinery gives gradients for the hand actions.  Not differentiated:
the number of sub-steps per frame (CFL, from the forward pass) and the PCG iteration counts (implicit adjoint)."""
from __future__ import annotations

import math

import numpy as np
import warp as wp

wp.config.quiet = True
F = wp.float64
PARAMS = ("cp", "beta", "nu", "D", "kappa_Q", "kappa_c", "kappa_t", "kappa_r", "B_dep", "p_dep", "kappa_M", "eps_f",
          "kappa_w", "tau_s", "mu_s", "xi_X", "tau_f", "alpha_u", "kappa_e", "kappa_tc", "vj", "e_F", "tau_b")
CP, BETA, NU, DM, KQ, KC, KT, KR, BDEP, PDEP, KM, KE, KW, TS, MS, XX, TF, AU, KJ, KTC, VJ, EF, YB = range(23)
YIELD_EPS, NU_MAX = 0.02, 0.5      # solver.Params.yield_eps, nu_max (Bingham foam raft, tau_f > 0)
SCHED = ("x", "y", "Q", "ux", "uy", "U", "d", "r1", "r2", "phi", "vjx", "vjy")
A2 = wp.array2d(dtype=F)
I2 = wp.array2d(dtype=wp.int32)
A1 = wp.array(dtype=F)


# ============================================================================ stencil helpers
@wp.func
def divF(Fx: A2, Fy: A2, j: int, i: int, N: int, inv_h: F):
    v = F(0.0)
    if i < N - 1:
        v += Fx[j, i]
    if i > 0:
        v -= Fx[j, i - 1]
    if j < N - 1:
        v += Fy[j, i]
    if j > 0:
        v -= Fy[j - 1, i]
    return v * inv_h


@wp.func
def vl(r: F):
    return (r + wp.abs(r)) / (F(1.0) + wp.abs(r))


# ============================================================================ inlet fields
@wp.kernel
def k_kernel_raw(xc: A2, yc: A2, mask: I2, S: A1, Sy: A1, Sux: A1, Suy: A1, P: A1, dep: F, Sr1: A1, Sr2: A1, Sphi: A1,
                 kq: wp.array(dtype=wp.int32), h: F, Kraw: A2, sums: A1):
    """Unnormalised footprint kernel (1 - a^2 - b^2)_+^2, q x q sub-cell quadrature; sums = [inside cup, all cells].
    Centred at the landing point (dep = 0) or where the foam surfaces, tau_s u_in ahead of it (dep = 1)."""
    j, i = wp.tid()
    k = kq[0]
    q = kq[1]
    x0 = S[k] + dep * P[TS] * Sux[k]
    y0 = Sy[k] + dep * P[TS] * Suy[k]
    r1 = wp.max(Sr1[k], F(1.5) * h)
    r2 = wp.max(Sr2[k], F(1.5) * h)
    cph = wp.cos(Sphi[k])
    sph = wp.sin(Sphi[k])
    acc = F(0.0)
    for ox in range(q):
        for oy in range(q):
            dx = xc[j, i] + ((F(ox) + F(0.5)) / F(q) - F(0.5)) * h - x0
            dy = yc[j, i] + ((F(oy) + F(0.5)) / F(q) - F(0.5)) * h - y0
            a = (cph * dx + sph * dy) / r1
            b = (-sph * dx + cph * dy) / r2
            w = wp.max(F(0.0), F(1.0) - a * a - b * b)
            acc += w * w
    Kraw[j, i] = acc
    if mask[j, i] != 0:
        wp.atomic_add(sums, 0, acc)
    wp.atomic_add(sums, 1, acc)


@wp.kernel
def k_fields(Kraw: A2, sums: A1, Kd: A2, sumsd: A1, mask: I2, xc: A2, yc: A2, P: A1, Sx: A1, Sy: A1, SQ: A1, Sux: A1,
             Suy: A1, SU: A1, Sd: A1, Sr1: A1, Sr2: A1, Svjx: A1, Svjy: A1, kq: wp.array(dtype=wp.int32), area: F,
             v05: int, up: int, l0: A2, m0: A2, s: A2, Lam: A2, vsx: A2, vsy: A2, sm: A2, ej: A2):
    """Solver.inlet_fields (closure 'chi', offcup 'account'): sm = chi S Z K_d (new milk, surfacing at the deposit kernel
    K_d, tau_s u_in ahead of the landing point; the landing kernel K if K_d misses the cup), s = sm (V0.5) or
    sm + (1 - chi) S Z K (up: the dived part resurfaces at the landing point as base material), + e_F chi S Z K (the
    landing crater's fountain: brown surface, no milk), Lam = kappa_c (1 - chi) S Z K, target v* (with vj: the
    traction part pulls towards V_j + kappa_tc (u_in - V_j), V_j the landing point's own velocity), ej = (1 - chi) S Z K
    (the diving flux, drives the entrainment sink kappa_e in k_exchange)."""
    j, i = wp.tid()
    k = kq[0]
    tot = sums[1]
    ins = sums[0]
    s[j, i] = F(0.0)
    Lam[j, i] = F(0.0)
    vsx[j, i] = F(0.0)
    vsy[j, i] = F(0.0)
    sm[j, i] = F(0.0)
    ej[j, i] = F(0.0)
    if tot <= F(0.0) or ins <= F(0.0):
        return
    Z = ins / tot
    Kn = F(0.0)
    Kdn = F(0.0)
    if mask[j, i] != 0:
        Kn = Kraw[j, i] / (ins * area)
        if sumsd[0] > F(0.0):
            Kdn = Kd[j, i] / (sumsd[0] * area)
        else:
            Kdn = Kn
    d = wp.max(Sd[k], F(1e-6))
    U = SU[k]
    zc = U * U / (P[BDEP] * d)
    chi = F(1.0) / (F(1.0) + wp.pow(zc, P[PDEP]))
    S_rec = P[KQ] * SQ[k] * Z
    smv = chi * S_rec * Kdn
    if P[AU] > F(0.0) and mask[j, i] != 0:          # underplating (Solver.inlet_fields): x (1 - alpha_u c), frame-start c
        smv = smv * (F(1.0) - P[AU] * wp.min(wp.max(m0[j, i] / l0[j, i], F(0.0)), F(1.0)))
    sm[j, i] = smv
    sv = smv + P[EF] * chi * S_rec * Kn
    if up != 0:
        sv = sv + (F(1.0) - chi) * S_rec * Kn
    s[j, i] = sv
    lam = P[KC] * (F(1.0) - chi) * S_rec * Kn
    Lam[j, i] = lam
    ej[j, i] = (F(1.0) - chi) * S_rec * Kn
    dx = xc[j, i] - Sx[k]
    dy = yc[j, i] - Sy[k]
    Rdep = wp.sqrt(wp.max(Sr1[k] * Sr2[k], F(1e-12)))
    den = wp.sqrt(Rdep * Rdep + dx * dx + dy * dy)
    rad = F(0.0)
    if v05 != 0:
        VR = U * wp.pow(chi, F(1.0) / (F(2.0) * P[PDEP]))
        VR = wp.min(wp.min(VR, U), wp.sqrt(P[BDEP] * d))
        rad = P[KR] * VR
    else:
        rad = chi * P[KR] * U
    tx = P[KT] * Sux[k] + rad * dx / den
    ty = P[KT] * Suy[k] + rad * dy / den
    if P[VJ] > F(0.0):
        ws = (F(1.0) - P[MS]) * smv
        if ws + lam > F(0.0):
            lx = Svjx[k] + P[KTC] * (Sux[k] - Svjx[k])
            ly = Svjy[k] + P[KTC] * (Suy[k] - Svjy[k])
            tx = (ws * tx + lam * lx) / (ws + lam)
            ty = (ws * ty + lam * ly) / (ws + lam)
    vsx[j, i] = tx
    vsy[j, i] = ty


@wp.kernel
def k_exchange(l: A2, qx: A2, qy: A2, m: A2, Kb: A2, Db: A2, Wf: A2, ej: A2, P: A1, dta: A1, mask: I2,
               lo: A2, qxo: A2, qyo: A2, mo: A2):
    """Vertical exchange of the film with the surface current (circulation.py), after each sub-step: the brown part
    b = l - m goes down at W = k_X sigma- + kappa_w kappa_Q W_n and is resupplied by the upwelled water k_X sigma+
    where the surface is brown (fraction 1 - c: the water surfaces beside the buoyant foam raft, not under it); the
    foam goes down at eps_f W.  k_X = (1 - xi_X) kappa_M: xi_X the share of the current's divergence taken up by the
    floating layer instead of exchanging it with the bulk (0: V2).  Exchanged material moves with the film.
    Entrainment sink of the diving stream: the brown part also goes down at kappa_e ej / l (ej = (1 - chi) S Z K), i.e.
    the brown area flux kappa_e ej (1 - c); the buoyant foam goes at eps_f of that rate."""
    j, i = wp.tid()
    fl = F(1.0)
    fm = F(1.0)
    if mask[j, i] != 0:
        L = l[j, i]
        c = wp.min(wp.max(m[j, i] / L, F(0.0)), F(1.0))
        kX = (F(1.0) - P[XX]) * P[KM]
        W = kX * Db[j, i] + P[KW] * P[KQ] * Wf[j, i] + P[KJ] * ej[j, i] / L
        fm = wp.exp(-dta[0] * P[KE] * W)
        b = (L - m[j, i]) * wp.exp(-dta[0] * W) + dta[0] * kX * Kb[j, i] * (F(1.0) - c)
        fl = (b + m[j, i] * fm) / L
    lo[j, i] = l[j, i] * fl
    qxo[j, i] = qx[j, i] * fl
    qyo[j, i] = qy[j, i] * fl
    mo[j, i] = m[j, i] * fm


# ============================================================================ sub-step kernels (solver.substep)
@wp.kernel
def k_prim(l: A2, qx: A2, qy: A2, m: A2, mask: I2, ls: A2, ux: A2, uy: A2, c: A2):
    j, i = wp.tid()
    if mask[j, i] != 0:
        L = l[j, i]
        ls[j, i] = L
        ux[j, i] = qx[j, i] / L
        uy[j, i] = qy[j, i] / L
        c[j, i] = m[j, i] / L
    else:
        ls[j, i] = F(1.0)
        ux[j, i] = F(0.0)
        uy[j, i] = F(0.0)
        c[j, i] = F(0.0)


@wp.kernel
def k_rusanov(qx: A2, qy: A2, ux: A2, uy: A2, fx: I2, fy: I2, Fxqx: A2, Fxqy: A2, Fyqx: A2, Fyqy: A2):
    """Step A face fluxes (x faces stored at [j, i] for the face i | i+1, y faces at [j, i] for j | j+1)."""
    j, i = wp.tid()
    Fxqx[j, i] = F(0.0)
    Fxqy[j, i] = F(0.0)
    Fyqx[j, i] = F(0.0)
    Fyqy[j, i] = F(0.0)
    if fx[j, i] != 0:
        uL = ux[j, i]
        uR = ux[j, i + 1]
        a = F(2.0) * wp.max(wp.abs(uL), wp.abs(uR))
        Fxqx[j, i] = F(0.5) * (qx[j, i] * uL + qx[j, i + 1] * uR) - F(0.5) * a * (qx[j, i + 1] - qx[j, i])
        Fxqy[j, i] = F(0.5) * (qy[j, i] * uL + qy[j, i + 1] * uR) - F(0.5) * a * (qy[j, i + 1] - qy[j, i])
    if fy[j, i] != 0:
        vL = uy[j, i]
        vR = uy[j + 1, i]
        a = F(2.0) * wp.max(wp.abs(vL), wp.abs(vR))
        Fyqx[j, i] = F(0.5) * (qx[j, i] * vL + qx[j + 1, i] * vR) - F(0.5) * a * (qx[j + 1, i] - qx[j, i])
        Fyqy[j, i] = F(0.5) * (qy[j, i] * vL + qy[j + 1, i] * vR) - F(0.5) * a * (qy[j + 1, i] - qy[j, i])


@wp.kernel
def k_stepA_B(qx: A2, qy: A2, Fxqx: A2, Fxqy: A2, Fyqx: A2, Fyqy: A2, s: A2, Lam: A2, vsx: A2, vsy: A2, ls: A2,
              P: A1, dta: A1, inv_h: F, N: int, Vbx: A2, Vby: A2, sm: A2, ux: A2, uy: A2, ax: A2, ay: A2, coef: A2):
    """q* = q - dt div(F) (step A), then stepB_prepare: theta = 1 + dt (beta + Lam/l),
    a = (q* + dt ((1-mu_s) sm + Lam) v* + dt (s - (1-mu_s) sm) u + dt beta l kappa_M U)/theta, coef = cp^2 l / theta:
    the new milk sm arrives with the jet target v* except the share mu_s of it that surfaces at rest relative to the
    film (its jet momentum lost under water), the base material s - sm (dived milk resurfacing) with the film's own
    velocity u (V0.5: s = sm, mu_s = 0)."""
    j, i = wp.tid()
    dt = dta[0]
    qxs = qx[j, i] - dt * divF(Fxqx, Fyqx, j, i, N, inv_h)
    qys = qy[j, i] - dt * divF(Fxqy, Fyqy, j, i, N, inv_h)
    L = ls[j, i]
    theta = F(1.0) + dt * (P[BETA] + Lam[j, i] / L)
    smv = (F(1.0) - P[MS]) * sm[j, i]
    f = dt * (smv + Lam[j, i])
    fb = dt * (s[j, i] - smv)
    g = dt * P[BETA] * L * P[KM]                         # drag towards the surface current (circulation.py; 0: at rest)
    ax[j, i] = (qxs + f * vsx[j, i] + fb * ux[j, i] + g * Vbx[j, i]) / theta
    ay[j, i] = (qys + f * vsy[j, i] + fb * uy[j, i] + g * Vby[j, i]) / theta
    coef[j, i] = P[CP] * P[CP] * L / theta


@wp.kernel
def k_massflux(ax: A2, ay: A2, l: A2, ls: A2, fx: I2, fy: I2, Gex: A2, Gey: A2):
    j, i = wp.tid()
    Gex[j, i] = F(0.0)
    Gey[j, i] = F(0.0)
    if fx[j, i] != 0:
        w = wp.max(wp.abs(ax[j, i] / ls[j, i]), wp.abs(ax[j, i + 1] / ls[j, i + 1]))
        Gex[j, i] = F(0.5) * (ax[j, i] + ax[j, i + 1]) - F(0.5) * w * (l[j, i + 1] - l[j, i])
    if fy[j, i] != 0:
        w = wp.max(wp.abs(ay[j, i] / ls[j, i]), wp.abs(ay[j + 1, i] / ls[j + 1, i]))
        Gey[j, i] = F(0.5) * (ay[j, i] + ay[j + 1, i]) - F(0.5) * w * (l[j + 1, i] - l[j, i])


@wp.kernel
def k_facecoef(coef: A2, fx: I2, fy: I2, dta: A1, kx: A2, ky: A2):
    j, i = wp.tid()
    dt = dta[0]
    kx[j, i] = F(0.0)
    ky[j, i] = F(0.0)
    if fx[j, i] != 0:
        kx[j, i] = dt * F(0.5) * (coef[j, i] + coef[j, i + 1])
    if fy[j, i] != 0:
        ky[j, i] = dt * F(0.5) * (coef[j, i] + coef[j + 1, i])


@wp.kernel
def k_massdiag(kx: A2, ky: A2, dta: A1, inv_h2: F, N: int, diag: A2):
    j, i = wp.tid()
    dt = dta[0]
    v = F(0.0)
    if i < N - 1:
        v += kx[j, i]
    if i > 0:
        v += kx[j, i - 1]
    if j < N - 1:
        v += ky[j, i]
    if j > 0:
        v += ky[j - 1, i]
    diag[j, i] = F(1.0) + dt * inv_h2 * v


@wp.kernel
def k_rhs(l: A2, s: A2, Gex: A2, Gey: A2, dta: A1, inv_h: F, N: int, rhs: A2):
    j, i = wp.tid()
    dt = dta[0]
    rhs[j, i] = l[j, i] + dt * s[j, i] - dt * divF(Gex, Gey, j, i, N, inv_h)


@wp.kernel
def k_mass_apply(phi: A2, kx: A2, ky: A2, fx: I2, fy: I2, dta: A1, inv_h2: F, N: int, out: A2):
    """phi - dt div(k grad phi) (kernels.apply_diffusion with a = 1, b = dt)."""
    j, i = wp.tid()
    dt = dta[0]
    p = phi[j, i]
    v = F(0.0)
    if i < N - 1:
        if fx[j, i] != 0:
            v += kx[j, i] * (phi[j, i + 1] - p)
    if i > 0:
        if fx[j, i - 1] != 0:
            v += kx[j, i - 1] * (phi[j, i - 1] - p)
    if j < N - 1:
        if fy[j, i] != 0:
            v += ky[j, i] * (phi[j + 1, i] - p)
    if j > 0:
        if fy[j - 1, i] != 0:
            v += ky[j - 1, i] * (phi[j - 1, i] - p)
    out[j, i] = p - dt * v * inv_h2


@wp.kernel
def k_finishB(Gex: A2, Gey: A2, kx: A2, ky: A2, ln: A2, fx: I2, fy: I2, inv_h: F, Gx: A2, Gy: A2):
    j, i = wp.tid()
    Gx[j, i] = F(0.0)
    Gy[j, i] = F(0.0)
    if fx[j, i] != 0:
        Gx[j, i] = Gex[j, i] - kx[j, i] * (ln[j, i + 1] - ln[j, i]) * inv_h
    if fy[j, i] != 0:
        Gy[j, i] = Gey[j, i] - ky[j, i] * (ln[j + 1, i] - ln[j, i]) * inv_h


@wp.kernel
def k_lcons_q(l: A2, s: A2, Gx: A2, Gy: A2, ax: A2, ay: A2, coef: A2, fx: I2, fy: I2, nfx: A2, nfy: A2, dta: A1,
              inv_h: F, N: int, l2: A2):
    """Conservative layer update with the final flux."""
    j, i = wp.tid()
    dt = dta[0]
    l2[j, i] = l[j, i] + dt * s[j, i] - dt * divF(Gx, Gy, j, i, N, inv_h)


@wp.kernel
def k_qgrad(ax: A2, ay: A2, coef: A2, l2: A2, fx: I2, fy: I2, nfx: A2, nfy: A2, dta: A1, inv_h: F, N: int,
            qxn: A2, qyn: A2):
    """q = a - dt coef cell_grad(l2) (average of the active face gradients)."""
    j, i = wp.tid()
    dt = dta[0]
    gx = F(0.0)
    gy = F(0.0)
    if i < N - 1:
        if fx[j, i] != 0:
            gx += (l2[j, i + 1] - l2[j, i]) * inv_h
    if i > 0:
        if fx[j, i - 1] != 0:
            gx += (l2[j, i] - l2[j, i - 1]) * inv_h
    if j < N - 1:
        if fy[j, i] != 0:
            gy += (l2[j + 1, i] - l2[j, i]) * inv_h
    if j > 0:
        if fy[j - 1, i] != 0:
            gy += (l2[j, i] - l2[j - 1, i]) * inv_h
    if nfx[j, i] > F(0.0):
        gx = gx / nfx[j, i]
    else:
        gx = F(0.0)
    if nfy[j, i] > F(0.0):
        gy = gy / nfy[j, i]
    else:
        gy = F(0.0)
    qxn[j, i] = ax[j, i] - dt * coef[j, i] * gx
    qyn[j, i] = ay[j, i] - dt * coef[j, i] * gy


@wp.func
def limited(cL: F, cR: F, dn: F, G: F, has_dn: int):
    """van Leer limited upwind face value; dn = the neighbouring difference on the upwind side (0 if none)."""
    d = cR - cL
    v = F(0.0)
    if wp.abs(d) <= F(1e-14):
        if G >= F(0.0):
            v = cL
        else:
            v = cR
    elif G >= F(0.0):
        v = cL + F(0.5) * vl(dn / d) * d
    else:
        v = cR - F(0.5) * vl(dn / d) * d
    return wp.min(wp.max(v, wp.min(cL, cR)), wp.max(cL, cR))


@wp.kernel
def k_limit(c: A2, Gx: A2, Gy: A2, fx: I2, fy: I2, N: int, cfx: A2, cfy: A2):
    j, i = wp.tid()
    cfx[j, i] = F(0.0)
    cfy[j, i] = F(0.0)
    if fx[j, i] != 0:
        cL = c[j, i]
        cR = c[j, i + 1]
        dn = F(0.0)
        if Gx[j, i] >= F(0.0):
            if i > 0:
                if fx[j, i - 1] != 0:
                    dn = cL - c[j, i - 1]
        else:
            if i + 1 < N - 1:
                if fx[j, i + 1] != 0:
                    dn = c[j, i + 2] - cR
        cfx[j, i] = limited(cL, cR, dn, Gx[j, i], 0)
    if fy[j, i] != 0:
        cL = c[j, i]
        cR = c[j + 1, i]
        dn = F(0.0)
        if Gy[j, i] >= F(0.0):
            if j > 0:
                if fy[j - 1, i] != 0:
                    dn = cL - c[j - 1, i]
        else:
            if j + 1 < N - 1:
                if fy[j + 1, i] != 0:
                    dn = c[j + 2, i] - cR
        cfy[j, i] = limited(cL, cR, dn, Gy[j, i], 0)


@wp.kernel
def k_milk(m: A2, sm: A2, Gx: A2, Gy: A2, cfx: A2, cfy: A2, l2: A2, qxn: A2, qyn: A2, mask: I2, dta: A1, inv_h: F, N: int,
           l3: A2, qx3: A2, qy3: A2, m3: A2):
    """Step C (milk with the same face flux, clipped to [0, l]) and the outside-cup reference state."""
    j, i = wp.tid()
    dt = dta[0]
    v = F(0.0)
    if i < N - 1:
        v += Gx[j, i] * cfx[j, i]
    if i > 0:
        v -= Gx[j, i - 1] * cfx[j, i - 1]
    if j < N - 1:
        v += Gy[j, i] * cfy[j, i]
    if j > 0:
        v -= Gy[j - 1, i] * cfy[j - 1, i]
    mn = m[j, i] + dt * sm[j, i] - dt * v * inv_h
    L = l2[j, i]
    mn = wp.min(wp.max(mn, F(0.0)), L)
    if mask[j, i] != 0:
        l3[j, i] = L
        qx3[j, i] = qxn[j, i]
        qy3[j, i] = qyn[j, i]
        m3[j, i] = mn
    else:
        l3[j, i] = F(1.0)
        qx3[j, i] = F(0.0)
        qy3[j, i] = F(0.0)
        m3[j, i] = F(0.0)


# ---------------------------------------------------------------------------- symmetric viscosity (kernels.sym_*)
@wp.kernel
def k_visc_nu(qx3: A2, qy3: A2, l3: A2, m3: A2, mask: I2, fx: I2, fy: I2, P: A1, inv_h: F, N: int, nuc: A2):
    """Solver.yield_visc: per-cell viscosity of the Bingham floating layer, nu + (tau_b (1 - c) + tau_f c) / (2 |D|_e + eps)
    capped at nu_max, from the velocity before the step (mean of the available one-sided differences); nu where
    tau_f = tau_b = 0."""
    j, i = wp.tid()
    nuc[j, i] = P[NU]
    if mask[j, i] != 0 and P[TF] + P[YB] > F(0.0):
        L = l3[j, i]
        u = qx3[j, i] / L
        v = qy3[j, i] / L
        ax = F(0.0)
        cx = F(0.0)
        nx = F(0.0)
        by = F(0.0)
        dy = F(0.0)
        ny = F(0.0)
        if i < N - 1:
            if fx[j, i] != 0:
                ax += qx3[j, i + 1] / l3[j, i + 1] - u
                cx += qy3[j, i + 1] / l3[j, i + 1] - v
                nx += F(1.0)
        if i > 0:
            if fx[j, i - 1] != 0:
                ax += u - qx3[j, i - 1] / l3[j, i - 1]
                cx += v - qy3[j, i - 1] / l3[j, i - 1]
                nx += F(1.0)
        if j < N - 1:
            if fy[j, i] != 0:
                by += qx3[j + 1, i] / l3[j + 1, i] - u
                dy += qy3[j + 1, i] / l3[j + 1, i] - v
                ny += F(1.0)
        if j > 0:
            if fy[j - 1, i] != 0:
                by += u - qx3[j - 1, i] / l3[j - 1, i]
                dy += v - qy3[j - 1, i] / l3[j - 1, i]
                ny += F(1.0)
        a = ax / (wp.max(nx, F(1.0)) / inv_h)
        c = cx / (wp.max(nx, F(1.0)) / inv_h)
        b = by / (wp.max(ny, F(1.0)) / inv_h)
        d = dy / (wp.max(ny, F(1.0)) / inv_h)
        De = wp.sqrt(a * a + d * d + a * d + F(0.25) * (b + c) * (b + c) + F(1e-24))
        cl = wp.min(wp.max(m3[j, i] / L, F(0.0)), F(1.0))
        nuc[j, i] = wp.min(P[NU] + (P[YB] * (F(1.0) - cl) + P[TF] * cl) / (F(2.0) * De + F(YIELD_EPS)), F(NU_MAX))


@wp.kernel
def k_visc_coefs(l3: A2, fx: I2, fy: I2, cm: I2, geo: A2, P: A1, nuc: A2, N: int, wx: A2, wy: A2, wc: A2, ww: A2):
    """Solver.sym_visc_coefs: constant nu (tau_f = tau_b = 0) or the per-cell nuc averaged to faces and corners."""
    j, i = wp.tid()
    nu = P[NU]
    wx[j, i] = F(0.0)
    wy[j, i] = F(0.0)
    wc[j, i] = F(0.0)
    if P[TF] + P[YB] > F(0.0):
        if fx[j, i] != 0:
            wx[j, i] = F(2.0) * (F(0.5) * (nuc[j, i] + nuc[j, i + 1])) * (F(0.5) * (l3[j, i] + l3[j, i + 1]))
        if fy[j, i] != 0:
            wy[j, i] = F(2.0) * (F(0.5) * (nuc[j, i] + nuc[j + 1, i])) * (F(0.5) * (l3[j, i] + l3[j + 1, i]))
        if cm[j, i] != 0:
            ncm = F(0.5) * (F(0.5) * (nuc[j, i] + nuc[j + 1, i]) + F(0.5) * (nuc[j, i + 1] + nuc[j + 1, i + 1]))
            wc[j, i] = ncm * (F(0.25) * (l3[j, i] + l3[j + 1, i] + l3[j, i + 1] + l3[j + 1, i + 1]))
        ww[j, i] = F(2.0) * nuc[j, i] * l3[j, i] * geo[j, i]
    else:
        if fx[j, i] != 0:
            wx[j, i] = F(2.0) * nu * F(0.5) * (l3[j, i] + l3[j, i + 1])
        if fy[j, i] != 0:
            wy[j, i] = F(2.0) * nu * F(0.5) * (l3[j, i] + l3[j + 1, i])
        if cm[j, i] != 0:
            wc[j, i] = nu * F(0.25) * (l3[j, i] + l3[j + 1, i] + l3[j, i + 1] + l3[j + 1, i + 1])
        ww[j, i] = F(2.0) * nu * l3[j, i] * geo[j, i]


@wp.kernel
def k_visc_rhs(qx3: A2, qy3: A2, l3: A2, mask: I2, X0x: A2, X0y: A2, bx: A2, by: A2):
    j, i = wp.tid()
    if mask[j, i] != 0:
        L = l3[j, i]
        X0x[j, i] = qx3[j, i] / L
        X0y[j, i] = qy3[j, i] / L
        bx[j, i] = qx3[j, i]
        by[j, i] = qy3[j, i]
    else:
        X0x[j, i] = F(0.0)
        X0y[j, i] = F(0.0)
        bx[j, i] = F(0.0)
        by[j, i] = F(0.0)


@wp.kernel
def k_sym_T(ux: A2, uy: A2, wc: A2, inv_h: F, N: int, T: A2):
    j, i = wp.tid()
    T[j, i] = F(0.0)
    if j < N - 1 and i < N - 1:
        w = wc[j, i]
        if w != F(0.0):
            g = (ux[j + 1, i] + ux[j + 1, i + 1] - ux[j, i] - ux[j, i + 1]
                 + uy[j, i + 1] + uy[j + 1, i + 1] - uy[j, i] - uy[j + 1, i]) * F(0.5) * inv_h
            T[j, i] = w * g


@wp.kernel
def k_sym_apply(ux: A2, uy: A2, l3: A2, wx: A2, wy: A2, T: A2, ww: A2, nbx: A2, nby: A2, mask: I2, dta: A1, inv_h: F,
                N: int, ox: A2, oy: A2):
    j, i = wp.tid()
    dt = dta[0]
    u = ux[j, i]
    v = uy[j, i]
    if mask[j, i] == 0:
        ox[j, i] = u
        oy[j, i] = v
        return
    inv_h2 = inv_h * inv_h
    ku = F(0.0)
    kv = F(0.0)
    if i < N - 1:
        ku -= wx[j, i] * (ux[j, i + 1] - u)
    if i > 0:
        ku -= wx[j, i - 1] * (ux[j, i - 1] - u)
    if j < N - 1:
        kv -= wy[j, i] * (uy[j + 1, i] - v)
    if j > 0:
        kv -= wy[j - 1, i] * (uy[j - 1, i] - v)
    ku *= inv_h2
    kv *= inv_h2
    c = F(0.0)
    d = F(0.0)
    if j < N - 1 and i < N - 1:
        c -= T[j, i]
        d -= T[j, i]
    if j < N - 1 and i > 0:
        c -= T[j, i - 1]
        d += T[j, i - 1]
    if j > 0 and i < N - 1:
        c += T[j - 1, i]
        d -= T[j - 1, i]
    if j > 0 and i > 0:
        c += T[j - 1, i - 1]
        d += T[j - 1, i - 1]
    ku += F(0.5) * inv_h * c
    kv += F(0.5) * inv_h * d
    w = ww[j, i]
    if w != F(0.0):
        un = w * (u * nbx[j, i] + v * nby[j, i])
        ku += un * nbx[j, i]
        kv += un * nby[j, i]
    L = l3[j, i]
    ox[j, i] = L * u + dt * ku
    oy[j, i] = L * v + dt * kv


@wp.kernel
def k_visc_diag(l3: A2, wx: A2, wy: A2, wc: A2, ww: A2, nbx: A2, nby: A2, mask: I2, dta: A1, inv_h: F, N: int,
                dx: A2, dy: A2):
    j, i = wp.tid()
    dt = dta[0]
    if mask[j, i] == 0:
        dx[j, i] = F(1.0)
        dy[j, i] = F(1.0)
        return
    ih2 = inv_h * inv_h
    ex = F(0.0)
    ey = F(0.0)
    c4 = F(0.0)
    if i < N - 1:
        ex += wx[j, i]
    if i > 0:
        ex += wx[j, i - 1]
    if j < N - 1:
        ey += wy[j, i]
    if j > 0:
        ey += wy[j - 1, i]
    if j < N - 1 and i < N - 1:
        c4 += wc[j, i]
    if j < N - 1 and i > 0:
        c4 += wc[j, i - 1]
    if j > 0 and i < N - 1:
        c4 += wc[j - 1, i]
    if j > 0 and i > 0:
        c4 += wc[j - 1, i - 1]
    dx[j, i] = l3[j, i] + dt * (ex * ih2 + F(0.25) * ih2 * c4 + ww[j, i] * nbx[j, i] * nbx[j, i])
    dy[j, i] = l3[j, i] + dt * (ey * ih2 + F(0.25) * ih2 * c4 + ww[j, i] * nby[j, i] * nby[j, i])


@wp.kernel
def k_visc_out(Xx: A2, Xy: A2, l3: A2, mask: I2, qx: A2, qy: A2):
    j, i = wp.tid()
    if mask[j, i] != 0:
        qx[j, i] = l3[j, i] * Xx[j, i]
        qy[j, i] = l3[j, i] * Xy[j, i]
    else:
        qx[j, i] = F(0.0)
        qy[j, i] = F(0.0)


@wp.kernel
def k_mix(m3: A2, l3: A2, fx: I2, fy: I2, mask: I2, P: A1, dta: A1, inv_h2: F, N: int, m4: A2):
    """Explicit mixing m += dt div(D l_f grad c), clipped to [0, l] (the D dt/h^2 < explicit_mixing_limit branch)."""
    j, i = wp.tid()
    dt = dta[0]
    if mask[j, i] == 0:
        m4[j, i] = F(0.0)
        return
    Dm = P[DM]
    c = m3[j, i] / l3[j, i]
    v = F(0.0)
    if i < N - 1:
        if fx[j, i] != 0:
            v += Dm * F(0.5) * (l3[j, i] + l3[j, i + 1]) * (m3[j, i + 1] / l3[j, i + 1] - c)
    if i > 0:
        if fx[j, i - 1] != 0:
            v += Dm * F(0.5) * (l3[j, i] + l3[j, i - 1]) * (m3[j, i - 1] / l3[j, i - 1] - c)
    if j < N - 1:
        if fy[j, i] != 0:
            v += Dm * F(0.5) * (l3[j, i] + l3[j + 1, i]) * (m3[j + 1, i] / l3[j + 1, i] - c)
    if j > 0:
        if fy[j - 1, i] != 0:
            v += Dm * F(0.5) * (l3[j, i] + l3[j - 1, i]) * (m3[j - 1, i] / l3[j - 1, i] - c)
    m4[j, i] = wp.min(wp.max(m3[j, i] + dt * v * inv_h2, F(0.0)), l3[j, i])


# ---------------------------------------------------------------------------- PCG building blocks (not taped)
@wp.kernel
def k_dot(a: A2, b: A2, out: A1):
    j, i = wp.tid()
    wp.atomic_add(out, 0, a[j, i] * b[j, i])


@wp.kernel
def k_cg_init(b: A2, Ax: A2, diag: A2, r: A2, p: A2):
    j, i = wp.tid()
    rr = b[j, i] - Ax[j, i]
    r[j, i] = rr
    p[j, i] = rr / diag[j, i]


@wp.kernel
def k_cg_step(alpha: A1, k_: int, p: A2, Ap: A2, x: A2, r: A2):
    j, i = wp.tid()
    a = alpha[k_]
    x[j, i] = x[j, i] + a * p[j, i]
    r[j, i] = r[j, i] - a * Ap[j, i]


@wp.kernel
def k_cg_dir(r: A2, diag: A2, beta: F, p: A2):
    j, i = wp.tid()
    p[j, i] = r[j, i] / diag[j, i] + beta * p[j, i]


@wp.kernel
def k_copy(a: A2, b: A2):
    j, i = wp.tid()
    b[j, i] = a[j, i]


@wp.kernel
def k_add(a: A2, b: A2):
    j, i = wp.tid()
    b[j, i] = b[j, i] + a[j, i]


@wp.kernel
def k_neg(a: A2, b: A2):
    j, i = wp.tid()
    b[j, i] = -a[j, i]


@wp.kernel
def k_rowdot(a: A2, b: A2, N: int, add: int, rows: A1):
    """rows[j] (+)= sum_i a[j, i] b[j, i]  (first level of a contention-free dot product)."""
    j = wp.tid()
    acc = F(0.0)
    for i in range(N):
        acc += a[j, i] * b[j, i]
    if add != 0:
        rows[j] = rows[j] + acc
    else:
        rows[j] = acc


@wp.kernel
def k_rowsum(rows: A1, N: int, sc: A1, slot: int):
    acc = F(0.0)
    for j in range(N):
        acc += rows[j]
    sc[slot] = acc


@wp.kernel
def k_alpha(sc: A1):
    sc[2] = sc[0] / (sc[1] + F(1e-300))


@wp.kernel
def k_beta(sc: A1):
    if sc[0] != F(0.0):
        sc[4] = sc[3] / sc[0]
    else:
        sc[4] = F(0.0)
    sc[0] = sc[3]


@wp.kernel
def k_zero1(a: A1):
    a[wp.tid()] = F(0.0)


@wp.kernel
def k_rowmax_speed(qx: A2, qy: A2, l: A2, s: A2, Lam: A2, vsx: A2, vsy: A2, mask: I2, N: int, active: int,
                   Vbx: A2, Vby: A2, P: A1, rows: A1):
    """CFL speed per row: max |q| / l_safe, and the inlet target speed where s + Lam > 0 (Solver.advance_frame)."""
    j = wp.tid()
    acc = F(0.0)
    for i in range(N):
        ls = F(1.0)
        if mask[j, i] != 0:
            ls = l[j, i]
        acc = wp.max(acc, wp.sqrt(qx[j, i] * qx[j, i] + qy[j, i] * qy[j, i]) / ls)
        acc = wp.max(acc, wp.abs(P[KM]) * wp.sqrt(Vbx[j, i] * Vbx[j, i] + Vby[j, i] * Vby[j, i]))
        if active != 0:
            if s[j, i] + Lam[j, i] > F(0.0):
                acc = wp.max(acc, wp.sqrt(vsx[j, i] * vsx[j, i] + vsy[j, i] * vsy[j, i]))
    rows[j] = acc


@wp.kernel
def k_rowcheck(l: A2, qx: A2, qy: A2, m: A2, mask: I2, N: int, rows: A1, rows2: A1):
    """Per row: min l over the cup, number of non-finite values (frame acceptance test)."""
    j = wp.tid()
    mn = F(1e300)
    bad = F(0.0)
    for i in range(N):
        if mask[j, i] != 0:
            mn = wp.min(mn, l[j, i])
        v = l[j, i] + qx[j, i] + qy[j, i] + m[j, i]
        if not wp.isfinite(v):
            bad += F(1.0)
    rows[j] = mn
    rows2[j] = bad


@wp.kernel
def k_rowreduce3(rows: A1, rows2: A1, N: int, sc: A1):
    """sc[0] = max(rows) (speed) ; or with rows2: sc[1] = min(rows), sc[2] = sum(rows2)."""
    mx = F(-1e300)
    mn = F(1e300)
    sm = F(0.0)
    for j in range(N):
        mx = wp.max(mx, rows[j])
        mn = wp.min(mn, rows[j])
        sm += rows2[j]
    sc[0] = mx
    sc[1] = mn
    sc[2] = sm


@wp.kernel
def k_cg_dir_dev(r: A2, diag: A2, sc: A1, p: A2):
    j, i = wp.tid()
    p[j, i] = r[j, i] / diag[j, i] + sc[4] * p[j, i]


@wp.kernel
def k_rdiv(r: A2, diag: A2, z: A2):
    j, i = wp.tid()
    z[j, i] = r[j, i] / diag[j, i]


# ============================================================================ geometry
class Geo:
    def __init__(self, N, R=0.49, device=None):
        self.N, self.h = N, 1.0 / N
        self.dev = wp.get_device(device) if device else wp.get_preferred_device()
        h = self.h
        c = (np.arange(N) + 0.5) * h - 0.5
        x = np.repeat(c[None, :], N, 0)
        y = np.repeat(c[:, None], N, 1)
        r = np.hypot(x, y)
        m = r <= R
        fx = np.zeros((N, N), bool); fx[:, :-1] = m[:, :-1] & m[:, 1:]
        fy = np.zeros((N, N), bool); fy[:-1, :] = m[:-1, :] & m[1:, :]
        nfx = np.zeros((N, N)); nfx[:, :-1] += fx[:, :-1]; nfx[:, 1:] += fx[:, :-1]
        nfy = np.zeros((N, N)); nfy[:-1, :] += fy[:-1, :]; nfy[1:, :] += fy[:-1, :]
        cm = np.zeros((N, N), bool); cm[:-1, :-1] = m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]
        pad = np.zeros((N + 2, N + 2), bool); pad[1:-1, 1:-1] = m
        nwx = (~pad[1:-1, :-2]).astype(float) + (~pad[1:-1, 2:])
        nwy = (~pad[:-2, 1:-1]).astype(float) + (~pad[2:, 1:-1])
        rs = np.maximum(r, 1e-300)
        nbx, nby = x / rs, y / rs
        geo = np.where(m, 4.0 / (h * h) * (np.abs(nbx) * nwx + np.abs(nby) * nwy), 0.0)
        self.mask_np = m
        self.x_np, self.y_np = x, y
        I = lambda a: wp.array(a.astype(np.int32), dtype=wp.int32, device=self.dev)
        D = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float64), dtype=F, device=self.dev)
        self.mask, self.fx, self.fy, self.cm = I(m), I(fx), I(fy), I(cm)
        self.nfx, self.nfy, self.geo, self.nbx, self.nby, self.xc, self.yc = D(nfx), D(nfy), D(geo), D(nbx), D(nby), D(x), D(y)


# ============================================================================ fused PCG kernels (device-resident loop)
# Scalars of one solve live in a small device array sc (slots below); the loop runs on the device under wp.capture_while
# (a CUDA conditional graph node when captured): no host synchronisation inside a solve.  Dot products: block-level
# tile reductions (one atomic add per block of TB cells) fused into the kernels that produce their operands.
TB = 128
TBc = wp.constant(TB)
POOL_BLK = 32                                  # arrays per block of the recorded-path pool
S_RZ, S_PAP, S_RZN, S_RR, S_BB, S_BETA, S_IT, S_ITT = range(8)


@wp.func
def cell_of(blk: int, e: int, N: int):
    idx = blk * TBc + e
    j = idx // N
    return j, idx - j * N, idx < N * N


@wp.kernel
def kc_zero(sc: A1):
    sc[S_RZ] = F(0.0)
    sc[S_BB] = F(0.0)
    sc[S_PAP] = F(0.0)
    sc[S_RZN] = F(0.0)
    sc[S_RR] = F(0.0)
    sc[S_IT] = F(0.0)


@wp.kernel
def kc_init(b1: A2, Ap1: A2, d1: A2, r1: A2, p1: A2, b2: A2, Ap2: A2, d2: A2, r2: A2, p2: A2, n: int, N: int, sc: A1):
    """r = b - A x0, p = z = r / d; rz = r . z and bb = b . b (tile sums)."""
    blk, lane = wp.tid()
    rz = F(0.0)
    bb = F(0.0)
    for e in range(lane, TBc, wp.block_dim()):
        j, i, ok = cell_of(blk, e, N)
        if ok:
            rr = b1[j, i] - Ap1[j, i]
            z = rr / d1[j, i]
            r1[j, i] = rr
            p1[j, i] = z
            rz += rr * z
            bb += b1[j, i] * b1[j, i]
            if n > 1:
                rr2 = b2[j, i] - Ap2[j, i]
                z2 = rr2 / d2[j, i]
                r2[j, i] = rr2
                p2[j, i] = z2
                rz += rr2 * z2
                bb += b2[j, i] * b2[j, i]
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(rz)), offset=S_RZ)
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(bb)), offset=S_BB)


@wp.kernel
def kc_prep(sc: A1, cond: wp.array(dtype=wp.int32)):
    sc[S_PAP] = F(0.0)
    sc[S_RZN] = F(0.0)
    sc[S_RR] = F(0.0)
    sc[S_IT] = F(0.0)
    cond[0] = 1


@wp.func
def mass_apply_at(phi: A2, kx: A2, ky: A2, fx: I2, fy: I2, dt: F, inv_h2: F, N: int, j: int, i: int):
    p = phi[j, i]
    v = F(0.0)
    if i < N - 1:
        if fx[j, i] != 0:
            v += kx[j, i] * (phi[j, i + 1] - p)
    if i > 0:
        if fx[j, i - 1] != 0:
            v += kx[j, i - 1] * (phi[j, i - 1] - p)
    if j < N - 1:
        if fy[j, i] != 0:
            v += ky[j, i] * (phi[j + 1, i] - p)
    if j > 0:
        if fy[j - 1, i] != 0:
            v += ky[j - 1, i] * (phi[j - 1, i] - p)
    return p - dt * v * inv_h2


@wp.kernel
def km_apply_dot(p: A2, kx: A2, ky: A2, fx: I2, fy: I2, dta: A1, inv_h2: F, N: int, Ap: A2, sc: A1):
    """Ap = A p (layer mass operator) and pAp (tile sum)."""
    blk, lane = wp.tid()
    dt = dta[0]
    acc = F(0.0)
    for e in range(lane, TBc, wp.block_dim()):
        j, i, ok = cell_of(blk, e, N)
        if ok:
            v = mass_apply_at(p, kx, ky, fx, fy, dt, inv_h2, N, j, i)
            Ap[j, i] = v
            acc += p[j, i] * v
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(acc)), offset=S_PAP)


@wp.func
def sym_apply_at(ux: A2, uy: A2, l3: A2, wx: A2, wy: A2, T: A2, ww: A2, nbx: A2, nby: A2, mask: I2, dt: F, inv_h: F,
                 N: int, j: int, i: int):
    u = ux[j, i]
    v = uy[j, i]
    if mask[j, i] == 0:
        return wp.vec2d(u, v)
    inv_h2 = inv_h * inv_h
    ku = F(0.0)
    kv = F(0.0)
    if i < N - 1:
        ku -= wx[j, i] * (ux[j, i + 1] - u)
    if i > 0:
        ku -= wx[j, i - 1] * (ux[j, i - 1] - u)
    if j < N - 1:
        kv -= wy[j, i] * (uy[j + 1, i] - v)
    if j > 0:
        kv -= wy[j - 1, i] * (uy[j - 1, i] - v)
    ku *= inv_h2
    kv *= inv_h2
    c = F(0.0)
    d = F(0.0)
    if j < N - 1 and i < N - 1:
        c -= T[j, i]
        d -= T[j, i]
    if j < N - 1 and i > 0:
        c -= T[j, i - 1]
        d += T[j, i - 1]
    if j > 0 and i < N - 1:
        c += T[j - 1, i]
        d -= T[j - 1, i]
    if j > 0 and i > 0:
        c += T[j - 1, i - 1]
        d += T[j - 1, i - 1]
    ku += F(0.5) * inv_h * c
    kv += F(0.5) * inv_h * d
    w = ww[j, i]
    if w != F(0.0):
        un = w * (u * nbx[j, i] + v * nby[j, i])
        ku += un * nbx[j, i]
        kv += un * nby[j, i]
    L = l3[j, i]
    return wp.vec2d(L * u + dt * ku, L * v + dt * kv)


@wp.kernel
def kv_apply_dot(px: A2, py: A2, l3: A2, wx: A2, wy: A2, T: A2, ww: A2, nbx: A2, nby: A2, mask: I2, dta: A1, inv_h: F,
                 N: int, Apx: A2, Apy: A2, sc: A1):
    """[Apx, Apy] = (L + dt K) p (symmetric viscosity, T from k_sym_T) and pAp (tile sum)."""
    blk, lane = wp.tid()
    dt = dta[0]
    acc = F(0.0)
    for e in range(lane, TBc, wp.block_dim()):
        j, i, ok = cell_of(blk, e, N)
        if ok:
            o = sym_apply_at(px, py, l3, wx, wy, T, ww, nbx, nby, mask, dt, inv_h, N, j, i)
            Apx[j, i] = o[0]
            Apy[j, i] = o[1]
            acc += px[j, i] * o[0] + py[j, i] * o[1]
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(acc)), offset=S_PAP)


@wp.kernel
def kc_update(sc: A1, x1: A2, r1: A2, p1: A2, Ap1: A2, d1: A2, x2: A2, r2: A2, p2: A2, Ap2: A2, d2: A2, n: int, N: int):
    """alpha = rz / pAp; x += alpha p; r -= alpha Ap; rz_new = r . (r / d), rr = r . r (tile sums)."""
    blk, lane = wp.tid()
    alpha = sc[S_RZ] / (sc[S_PAP] + F(1e-300))
    rzn = F(0.0)
    rr = F(0.0)
    for e in range(lane, TBc, wp.block_dim()):
        j, i, ok = cell_of(blk, e, N)
        if ok:
            x1[j, i] = x1[j, i] + alpha * p1[j, i]
            r = r1[j, i] - alpha * Ap1[j, i]
            r1[j, i] = r
            rzn += r * r / d1[j, i]
            rr += r * r
            if n > 1:
                x2[j, i] = x2[j, i] + alpha * p2[j, i]
                r2_ = r2[j, i] - alpha * Ap2[j, i]
                r2[j, i] = r2_
                rzn += r2_ * r2_ / d2[j, i]
                rr += r2_ * r2_
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(rzn)), offset=S_RZN)
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(rr)), offset=S_RR)


@wp.kernel
def kc_scalar(sc: A1, cond: wp.array(dtype=wp.int32), tol: F, maxiter: int):
    """Convergence (|r| <= tol |b|, as solver.pcg), iteration counters, beta; reset the accumulators."""
    it = sc[S_IT] + F(1.0)
    sc[S_IT] = it
    sc[S_ITT] = sc[S_ITT] + F(1.0)
    if wp.sqrt(wp.max(sc[S_RR], F(0.0))) <= tol * (wp.sqrt(wp.max(sc[S_BB], F(0.0))) + F(1e-300)) or it >= F(maxiter):
        cond[0] = 0
    if sc[S_RZ] != F(0.0):
        sc[S_BETA] = sc[S_RZN] / sc[S_RZ]
    else:
        sc[S_BETA] = F(0.0)
    sc[S_RZ] = sc[S_RZN]
    sc[S_RZN] = F(0.0)
    sc[S_RR] = F(0.0)
    sc[S_PAP] = F(0.0)


@wp.kernel
def kc_dir(sc: A1, r1: A2, d1: A2, p1: A2, r2: A2, d2: A2, p2: A2, n: int):
    j, i = wp.tid()
    beta = sc[S_BETA]
    p1[j, i] = r1[j, i] / d1[j, i] + beta * p1[j, i]
    if n > 1:
        p2[j, i] = r2[j, i] / d2[j, i] + beta * p2[j, i]


# ============================================================================ CUDA graphs
@wp.kernel
def k_set0(a: A2):
    j, i = wp.tid()
    a[j, i] = F(0.0)


@wp.kernel
def k_init_state(l: A2, qx: A2, qy: A2, m: A2):
    j, i = wp.tid()
    l[j, i] = F(1.0)
    qx[j, i] = F(0.0)
    qy[j, i] = F(0.0)
    m[j, i] = F(0.0)


class Runner:
    """Runs a fixed launch sequence: on CUDA captured once per key into a CUDA graph and replayed (one launch instead
    of tens); elsewhere executed directly.  The sequences only launch kernels on persistent buffers; everything that
    changes between replays (dt, frame index, quadrature) is read by the kernels from device arrays."""

    def __init__(self, dev, use_graphs=True):
        self.dev = dev
        self.use = bool(use_graphs and dev.is_cuda)
        self.graphs = {}
        self.stats = dict(captures=0, replays=0)

    def __call__(self, key, fn):
        if not self.use or self.dev.is_capturing:       # (inside an enclosing capture: plain launches, become its nodes)
            fn()
            return
        g = self.graphs.get(key)
        if g is None:
            with wp.ScopedCapture(device=self.dev, force_module_load=True) as cap:
                fn()
            g = self.graphs[key] = cap.graph
            self.stats["captures"] += 1
        wp.capture_launch(g)
        self.stats["replays"] += 1


class CGOp:
    """Jacobi-PCG on persistent buffers (x, b, d and the operator coefficients c[...]).
    mode 'device' (default): fused kernels (operator + dot, update + dots, scalars, direction), block tile
      reductions, the whole loop on the device under wp.capture_while -- no host synchronisation; on CUDA the solve
      is one graph with a conditional node (or part of the sub-step graph).
    mode 'chunk' (fallback when conditional nodes are unavailable): iterations in graph-captured chunks of K,
      convergence read back after each chunk."""

    def __init__(self, sol, name, n, coef_names, apply, apply_dot):
        g = sol.g
        z = lambda: wp.zeros((g.N, g.N), dtype=F, device=g.dev)
        self.sol, self.name, self.n = sol, name, n
        self.x, self.b, self.d = [z() for _ in range(n)], [z() for _ in range(n)], [z() for _ in range(n)]
        self.r, self.p, self.Ap, self.z = [z() for _ in range(n)], [z() for _ in range(n)], [z() for _ in range(n)], [z() for _ in range(n)]
        self.c = {k: z() for k in coef_names}
        self.T = z()
        self.rows = wp.zeros(g.N, dtype=F, device=g.dev)
        self.sc = wp.zeros(8, dtype=F, device=g.dev)       # device mode scalars (S_* slots)
        self.scc = wp.zeros(8, dtype=F, device=g.dev)      # chunk mode: rz, pAp, alpha, rz_new, beta, rr, bb
        self.cond = wp.zeros(1, dtype=wp.int32, device=g.dev)
        self.apply = apply                                  # apply(op, src list, dst list)          (chunk mode)
        self.apply_dot = apply_dot                          # apply_dot(op, src list) -> op.Ap, pAp     (device mode)
        self.nb = (g.N * g.N + TB - 1) // TB
        self.mode = "device"
        self.chunk_iters = 0

    def _L(self, kernel, inputs, dim=None):
        g = self.sol.g
        wp.launch(kernel, dim=dim or (g.N, g.N), inputs=inputs, device=g.dev, record_tape=False)

    def _T(self, kernel, inputs):
        wp.launch_tiled(kernel, dim=[self.nb], inputs=inputs, block_dim=TB, device=self.sol.g.dev, record_tape=False)

    def _two(self, lst):
        return (lst[0], lst[1]) if self.n > 1 else (lst[0], lst[0])

    # ---------------------------------------------------------------- device mode
    def _dev_init(self):
        N = self.sol.g.N
        self._L(kc_zero, [self.sc], dim=1)
        self.apply_dot(self, self.x)
        b1, b2 = self._two(self.b); A1_, A2_ = self._two(self.Ap); d1, d2 = self._two(self.d)
        r1, r2 = self._two(self.r); p1, p2 = self._two(self.p)
        self._T(kc_init, [b1, A1_, d1, r1, p1, b2, A2_, d2, r2, p2, self.n, N, self.sc])
        self._L(kc_prep, [self.sc, self.cond], dim=1)

    def _dev_iter(self, tol, maxiter):
        N = self.sol.g.N
        self.apply_dot(self, self.p)
        x1, x2 = self._two(self.x); r1, r2 = self._two(self.r); p1, p2 = self._two(self.p)
        A1_, A2_ = self._two(self.Ap); d1, d2 = self._two(self.d)
        self._T(kc_update, [self.sc, x1, r1, p1, A1_, d1, x2, r2, p2, A2_, d2, self.n, N])
        self._L(kc_scalar, [self.sc, self.cond, F(tol), int(maxiter)], dim=1)
        self._L(kc_dir, [self.sc, r1, d1, p1, r2, d2, p2, self.n])

    def launch_solve(self, tol, maxiter):
        """Raw launches of one whole solve (to be run directly or inside a bigger graph capture)."""
        self._dev_init()
        wp.capture_while(self.cond, lambda: self._dev_iter(tol, maxiter))

    # ---------------------------------------------------------------- chunk mode (fallback)
    def _dot(self, A, B, slot):
        N = self.sol.g.N
        for c in range(self.n):
            self._L(k_rowdot, [A[c], B[c], N, int(c > 0), self.rows], dim=N)
        self._L(k_rowsum, [self.rows, N, self.scc, slot], dim=1)

    def _init(self):
        self.apply(self, self.x, self.Ap)
        for c in range(self.n):
            self._L(k_cg_init, [self.b[c], self.Ap[c], self.d[c], self.r[c], self.p[c]])
        self._dot(self.r, self.p, 0)
        self._dot(self.b, self.b, 6)

    def _chunk(self, K):
        for _ in range(K):
            self.apply(self, self.p, self.Ap)
            self._dot(self.p, self.Ap, 1)
            self._L(k_alpha, [self.scc], dim=1)
            for c in range(self.n):
                self._L(k_cg_step, [self.scc, 2, self.p[c], self.Ap[c], self.x[c], self.r[c]])
                self._L(k_rdiv, [self.r[c], self.d[c], self.z[c]])
            self._dot(self.r, self.z, 3)
            self._L(k_beta, [self.scc], dim=1)
            for c in range(self.n):
                self._L(k_cg_dir_dev, [self.r[c], self.d[c], self.scc, self.p[c]])
        self._dot(self.r, self.r, 5)

    def _run_chunks(self, tol, maxiter):
        R, K = self.sol.runner, self.sol.check_every
        R(self.name + "_init", self._init)
        bnorm = math.sqrt(max(float(self.scc.numpy()[6]), 0.0)) + 1e-300
        it = 0
        while it < maxiter:
            R(f"{self.name}_chunk{K}", lambda: self._chunk(K))
            it += K
            if math.sqrt(max(float(self.scc.numpy()[5]), 0.0)) <= tol * bnorm:
                break
        self.chunk_iters += it

    # ---------------------------------------------------------------- entry points
    def run(self, tol, maxiter):
        if self.mode == "device":
            try:
                self.sol.runner(self.name + "_solve", lambda: self.launch_solve(tol, maxiter))
                return
            except Exception as e:                       # e.g. no conditional graph nodes on this driver
                if not self.sol.runner.use:
                    raise
                print(f"[wsolver] device-side PCG loop unavailable ({type(e).__name__}: {e}); chunked fallback", flush=True)
                self.sol.runner.graphs.pop(self.name + "_solve", None)
                self.mode = "chunk"
                self.sol.fused = False
        self._run_chunks(tol, maxiter)

    def iterations(self):
        """Total PCG iterations so far (device counter + chunk-mode counter); synchronises."""
        return int(round(float(self.sc.numpy()[S_ITT]))) + self.chunk_iters

    def reset_iterations(self):
        self.sc.zero_()
        self.chunk_iters = 0


def _apply_dot_mass(op, src):
    g, s = op.sol.g, op.sol
    op._T(km_apply_dot, [src[0], op.c["kx"], op.c["ky"], g.fx, g.fy, s.dta, F(1.0 / (g.h * g.h)), g.N, op.Ap[0], op.sc])


def _apply_dot_visc(op, src):
    g, s = op.sol.g, op.sol
    ih = F(1.0 / g.h)
    op._L(k_sym_T, [src[0], src[1], op.c["wc"], ih, g.N, op.T])
    op._T(kv_apply_dot, [src[0], src[1], op.c["l3"], op.c["wx"], op.c["wy"], op.T, op.c["ww"], g.nbx, g.nby, g.mask,
                         s.dta, ih, g.N, op.Ap[0], op.Ap[1], op.sc])


def _apply_mass(op, src, dst):
    g, s = op.sol.g, op.sol
    op._L(k_mass_apply, [src[0], op.c["kx"], op.c["ky"], g.fx, g.fy, s.dta, F(1.0 / (g.h * g.h)), g.N, dst[0]])


def _apply_visc(op, src, dst):
    g, s = op.sol.g, op.sol
    ih = F(1.0 / g.h)
    op._L(k_sym_T, [src[0], src[1], op.c["wc"], ih, g.N, op.T])
    op._L(k_sym_apply, [src[0], src[1], op.c["l3"], op.c["wx"], op.c["wy"], op.T, op.c["ww"], g.nbx, g.nby, g.mask,
                        s.dta, ih, g.N, dst[0], dst[1]])


# ============================================================================ solver
class WSolver:
    """Two paths over the same kernels:
      fast_* : no gradients; every buffer persistent, each piece of a sub-step a CUDA graph (forward passes)
      frame  : recorded under wp.Tape (the per-frame replay of the backward pass, and the CPU tests); arrays allocated
               per launch, or taken from the persistent pool (pool_begin / pool_end) so that the whole replay of a
               frame, forward + backward, can be one CUDA graph (run.DiffPour._replay); its PCG solves run on the same
               CGOp as the fast path.
    rec: allocate gradients and register the adjoints of the PCG solves on `tape` (set by the caller)."""

    def __init__(self, geo, cfl=0.5, max_substeps=64, cg_tol=1e-10, visc_tol=1e-8, cg_maxiter=500,
                 kernel_quadrature=3, explicit_mixing_limit=0.05, l_floor=1e-3, v05=True, check_every=1, graphs=True,
                 up=False):
        self.g = geo
        self.cfl, self.max_substeps = cfl, max_substeps
        self.cg_tol, self.visc_tol, self.cg_maxiter = cg_tol, visc_tol, cg_maxiter
        self.kq_default, self.mix_lim, self.l_floor, self.v05 = kernel_quadrature, explicit_mixing_limit, l_floor, int(v05)
        self.check_every = check_every
        self.up = int(up)                    # volume-conserving surface source (k_fields)
        self.tape = None
        self.rec = False
        self.stats = dict(cg_mass=0, cg_visc=0, cg_adj=0, substeps=0, frames=0)
        dev = geo.dev
        self.dta = wp.zeros(1, dtype=F, device=dev)
        self.kq = wp.zeros(2, dtype=wp.int32, device=dev)
        self.Vbx, self.Vby, self.Kb, self.Db, self.Wf = (wp.zeros((geo.N, geo.N), dtype=F, device=dev) for _ in range(5))
        self.circ, self._ck = None, None     # unit surface-current fields of the frames (set_circ)
        self.ex_on = False                   # vertical exchange step on (surface current or entrainment sink; set_ex)
        self._dt = None
        self._kq = None
        import os
        self.runner = Runner(dev, graphs and os.environ.get("LATTE_NO_GRAPHS", "0") != "1")
        self.opM = CGOp(self, "mass", 1, ("kx", "ky"), _apply_mass, _apply_dot_mass)
        self.opV = CGOp(self, "visc", 2, ("l3", "wx", "wy", "wc", "ww"), _apply_visc, _apply_dot_visc)
        self.fused = True                    # whole sub-step as one graph (needs the device-side PCG loop)
        self.fb = None
        self.pool_on = False                 # recorded path: arrays from the persistent pool (pool_begin)
        self._blocks, self._views, self._blocks1, self._views1 = [], [], [], []
        self.pool_i = self.pool1_i = 0

    # ------------------------------------------------------------------ small helpers
    def new(self):
        """Output array of a recorded launch.  Zero-initialised on purpose: some kernels store an element twice (0, then
        the value under a condition), and the generated adjoints are only exact on a zeroed output.  With the pool on,
        the next pool slot (zeroed with its gradient by pool_begin): no allocation, and fixed addresses, so a whole
        recorded frame (forward + backward) can be captured into a CUDA graph and replayed."""
        if self.pool_on:
            return self._take(self._blocks, self._views, (self.g.N, self.g.N), "pool_i")
        return wp.zeros((self.g.N, self.g.N), dtype=F, device=self.g.dev, requires_grad=self.rec)

    def new1(self):
        """Length-2 accumulator (pooled like new)."""
        if self.pool_on:
            return self._take(self._blocks1, self._views1, (2,), "pool1_i")
        return wp.zeros(2, dtype=F, device=self.g.dev, requires_grad=self.rec)

    def _take(self, blocks, views, shape, ctr):
        i = getattr(self, ctr)
        b, e = divmod(i, POOL_BLK)
        if b == len(blocks):
            if self.g.dev.is_cuda and self.g.dev.is_capturing:
                raise RuntimeError("array pool would grow inside a graph capture")
            blk = wp.zeros((POOL_BLK, *shape), dtype=F, device=self.g.dev, requires_grad=True)
            blocks.append(blk)
            views.append([blk[t] for t in range(POOL_BLK)])
        setattr(self, ctr, i + 1)
        return views[b][e]

    def pool_begin(self, nblk=None):
        """Start a pooled recorded frame: slots from the first one again; zero the first nblk blocks (values and
        gradients; all blocks if None)."""
        self.pool_on, self.pool_i, self.pool1_i = True, 0, 0
        for blk in self._blocks[:nblk] + self._blocks1:
            blk.zero_()
            blk.grad.zero_()

    def pool_end(self):
        """Stop pooling; returns the number of blocks this frame used."""
        self.pool_on = False
        return -(-self.pool_i // POOL_BLK)

    def L(self, kernel, inputs, outputs=(), record=True):
        wp.launch(kernel, dim=(self.g.N, self.g.N), inputs=list(inputs) + list(outputs), device=self.g.dev,
                  record_tape=record)

    def cp(self, a, b):
        wp.launch(k_copy, dim=(self.g.N, self.g.N), inputs=[a, b], device=self.g.dev, record_tape=False)

    def set_dt(self, dt):
        if self._dt != dt:
            self.dta.assign(np.array([dt], np.float64))
            self._dt = dt

    def set_circ(self, U=None, up=None, dn=None, Wn=None):
        """Unit surface-current fields per frame (circulation.SurfaceCurrent: host arrays U (n, 2, N, N), sigma+,
        sigma-, W_n (n, N, N)) or None."""
        self.circ = None if U is None else (U, up, dn, Wn)
        self._ck = None
        for a in (self.Vbx, self.Vby, self.Kb, self.Db, self.Wf):
            a.zero_()

    def set_ex(self, Pnp):
        """The end-of-sub-step exchange runs with the surface current or the entrainment sink (kappa_e > 0)."""
        self.ex_on = bool(self.circ is not None or Pnp[KJ] > 0)

    def set_kq(self, k, q):
        if self.circ is not None and self._ck != k:
            U, up, dn, Wn = self.circ
            for a, b in ((self.Vbx, U[k, 0]), (self.Vby, U[k, 1]), (self.Kb, up[k]), (self.Db, dn[k]), (self.Wf, Wn[k])):
                a.assign(np.ascontiguousarray(b, np.float64))
            self._ck = k
        if self._kq != (k, q):
            self.kq.assign(np.array([k, q], np.int32))
            self._kq = (k, q)

    def state_from_numpy(self, l, qx, qy, m, requires_grad=False):
        mk = lambda a: wp.array(np.ascontiguousarray(a, np.float64), dtype=F, device=self.g.dev, requires_grad=requires_grad)
        return dict(l=mk(l), qx=mk(qx), qy=mk(qy), m=mk(m))

    def clone_state(self, st, requires_grad=False):
        out = {}
        for k, a in st.items():
            b = wp.empty_like(a, requires_grad=requires_grad)
            self.cp(a, b)
            out[k] = b
        return out

    def initial_state(self, requires_grad=False):
        N = self.g.N
        return self.state_from_numpy(np.ones((N, N)), np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N)), requires_grad)

    @staticmethod
    def to_numpy(st):
        return {k: v.numpy().copy() for k, v in st.items()}

    def mix_on(self, Pnp, dt):
        h = self.g.h
        return bool(Pnp[DM] > 0 and Pnp[DM] * dt / (h * h) < self.mix_lim)

    # ------------------------------------------------------------------ implicit solve with exact adjoint
    def solve(self, op, coefs, apply_eager, b, diag, x0, tol, key):
        """x = A^{-1} b, A = op's operator with the coefficient arrays `coefs` (name -> array); on a recording pass
        the adjoint is registered on self.tape (module docstring)."""
        for name, a in coefs.items():
            self.cp(a, op.c[name])
        for c in range(op.n):
            self.cp(b[c], op.b[c])
            self.cp(diag[c], op.d[c])
            self.cp(x0[c], op.x[c])
        op.run(tol, self.cg_maxiter)
        x = []
        for c in range(op.n):
            xa = self.new()
            self.cp(op.x[c], xa)
            x.append(xa)
        if self.rec and self.tape is not None:
            dt_now = self._dt

            def backward():
                self.set_dt(dt_now)
                for name, a in coefs.items():
                    self.cp(a, op.c[name])
                for c in range(op.n):
                    self.cp(x[c].grad, op.b[c])
                    self.cp(diag[c], op.d[c])
                    wp.launch(k_set0, dim=(self.g.N, self.g.N), inputs=[op.x[c]], device=self.g.dev, record_tape=False)
                op.run(tol, self.cg_maxiter)
                lam_neg, xs, outs = [], [], []
                for c in range(op.n):
                    self.L(k_add, [op.x[c], b[c].grad], record=False)
                    ln = self.new()
                    self.L(k_neg, [op.x[c], ln], record=False)
                    lam_neg.append(ln)
                    xc = self.new()
                    self.cp(x[c], xc)
                    xs.append(xc)
                    outs.append(self.new())
                sub = wp.Tape()
                with sub:
                    apply_eager(xs, outs)
                sub.backward(grads={o: g for o, g in zip(outs, lam_neg)})
            self.tape.record_func(backward, arrays=list(x) + list(b) + list(coefs.values()))
        return x

    # ------------------------------------------------------------------ recorded path: inlet fields, sub-step, frame
    def fields(self, P, S, k, q, active, st):
        """Inlet fields of frame k (k_fields); st: the state at the start of the frame (underplating reads its colour)."""
        g = self.g
        out = [self.new() for _ in range(6)]
        self.set_kq(k, q)
        if not active:
            return out
        Kraw, Kd = self.new(), self.new()
        sums, sumsd = self.new1(), self.new1()
        for dep, K_, s_ in ((0.0, Kraw, sums), (1.0, Kd, sumsd)):
            self.L(k_kernel_raw, [g.xc, g.yc, g.mask, S["x"], S["y"], S["ux"], S["uy"], P, F(dep), S["r1"], S["r2"],
                                  S["phi"], self.kq, F(g.h)], [K_, s_])
        self.L(k_fields, [Kraw, sums, Kd, sumsd, g.mask, g.xc, g.yc, P, S["x"], S["y"], S["Q"], S["ux"], S["uy"], S["U"],
                          S["d"], S["r1"], S["r2"], S["vjx"], S["vjy"], self.kq, F(g.h * g.h), self.v05, self.up, st["l"],
                          st["m"]], out)
        return out

    def substep(self, st, fl, P, nu_on, mix_on):
        g = self.g
        N, h = g.N, g.h
        ih, ih2 = F(1.0 / h), F(1.0 / (h * h))
        dt = self.dta
        s, Lam, vsx, vsy, sm, ej = fl
        ls, ux, uy, c = self.new(), self.new(), self.new(), self.new()
        self.L(k_prim, [st["l"], st["qx"], st["qy"], st["m"], g.mask], [ls, ux, uy, c])
        Fs = [self.new() for _ in range(4)]
        self.L(k_rusanov, [st["qx"], st["qy"], ux, uy, g.fx, g.fy], Fs)
        ax, ay, coef = self.new(), self.new(), self.new()
        self.L(k_stepA_B, [st["qx"], st["qy"], *Fs, s, Lam, vsx, vsy, ls, P, dt, ih, N, self.Vbx, self.Vby, sm, ux, uy],
               [ax, ay, coef])
        Gex, Gey = self.new(), self.new()
        self.L(k_massflux, [ax, ay, st["l"], ls, g.fx, g.fy], [Gex, Gey])
        kx, ky, diag = self.new(), self.new(), self.new()
        self.L(k_facecoef, [coef, g.fx, g.fy, dt], [kx, ky])
        self.L(k_massdiag, [kx, ky, dt, ih2, N], [diag])
        rhs = self.new()
        self.L(k_rhs, [st["l"], s, Gex, Gey, dt, ih, N], [rhs])

        def A_mass(xs, outs):
            self.L(k_mass_apply, [xs[0], kx, ky, g.fx, g.fy, dt, ih2, N], [outs[0]])

        (ln,) = self.solve(self.opM, dict(kx=kx, ky=ky), A_mass, [rhs], [diag], [st["l"]], self.cg_tol, "cg_mass")
        Gx, Gy = self.new(), self.new()
        self.L(k_finishB, [Gex, Gey, kx, ky, ln, g.fx, g.fy, ih], [Gx, Gy])
        l2 = self.new()
        self.L(k_lcons_q, [st["l"], s, Gx, Gy, ax, ay, coef, g.fx, g.fy, g.nfx, g.nfy, dt, ih, N], [l2])
        qxn, qyn = self.new(), self.new()
        self.L(k_qgrad, [ax, ay, coef, l2, g.fx, g.fy, g.nfx, g.nfy, dt, ih, N], [qxn, qyn])
        cfx, cfy = self.new(), self.new()
        self.L(k_limit, [c, Gx, Gy, g.fx, g.fy, N], [cfx, cfy])
        l3, qx3, qy3, m3 = self.new(), self.new(), self.new(), self.new()
        self.L(k_milk, [st["m"], sm, Gx, Gy, cfx, cfy, l2, qxn, qyn, g.mask, dt, ih, N], [l3, qx3, qy3, m3])
        qx4, qy4 = qx3, qy3
        if nu_on:
            wx, wy, wc, ww, nuc = self.new(), self.new(), self.new(), self.new(), self.new()
            self.L(k_visc_nu, [qx3, qy3, l3, m3, g.mask, g.fx, g.fy, P, ih, N], [nuc])
            self.L(k_visc_coefs, [l3, g.fx, g.fy, g.cm, g.geo, P, nuc, N], [wx, wy, wc, ww])
            X0x, X0y, bx, by = self.new(), self.new(), self.new(), self.new()
            self.L(k_visc_rhs, [qx3, qy3, l3, g.mask], [X0x, X0y, bx, by])
            dx, dy = self.new(), self.new()
            self.L(k_visc_diag, [l3, wx, wy, wc, ww, g.nbx, g.nby, g.mask, dt, ih, N], [dx, dy])

            def A_visc(xs, outs):
                T = self.new()
                self.L(k_sym_T, [xs[0], xs[1], wc, ih, N], [T])
                self.L(k_sym_apply, [xs[0], xs[1], l3, wx, wy, T, ww, g.nbx, g.nby, g.mask, dt, ih, N], [outs[0], outs[1]])

            Xx, Xy = self.solve(self.opV, dict(l3=l3, wx=wx, wy=wy, wc=wc, ww=ww), A_visc, [bx, by], [dx, dy], [X0x, X0y],
                                self.visc_tol, "cg_visc")
            qx4, qy4 = self.new(), self.new()
            self.L(k_visc_out, [Xx, Xy, l3, g.mask], [qx4, qy4])
        m4 = m3
        if mix_on:
            m4 = self.new()
            self.L(k_mix, [m3, l3, g.fx, g.fy, g.mask, P, dt, ih2, N], [m4])
        self.stats["substeps"] += 1
        if self.ex_on:
            lo, qxo, qyo, mo = self.new(), self.new(), self.new(), self.new()
            self.L(k_exchange, [l3, qx4, qy4, m4, self.Kb, self.Db, self.Wf, ej, P, dt, g.mask], [lo, qxo, qyo, mo])
            return dict(l=lo, qx=qxo, qy=qyo, m=mo)
        return dict(l=l3, qx=qx4, qy=qy4, m=m4)

    def n_substeps_host(self, st, fl, fdt, active):
        g = self.g
        msk = g.mask_np
        ls = np.where(msk, st["l"].numpy(), 1.0)
        umax = float(np.max(np.hypot(st["qx"].numpy(), st["qy"].numpy()) / ls))
        if active:
            s, Lam, vx, vy = (a.numpy() for a in fl[:4])
            umax = max(umax, float(np.max(np.hypot(vx, vy) * (s + Lam > 0))))
        return self._nsub(umax, fdt)

    def _nsub(self, umax, fdt):
        dt_cfl = self.cfl * self.g.h / (2.0 * max(umax, 1e-9))
        return max(1, min(int(math.ceil(fdt / dt_cfl)), self.max_substeps))

    def frame(self, st, P, Pnp, S, k, fdt, q, active, nsub=None):
        """Recorded-path frame; nsub None: CFL count + retries; given: replay with that count."""
        self.set_ex(Pnp)
        fl = self.fields(P, S, k, q, active, st)
        nu_on = Pnp[NU] > 0
        if nsub is None:
            nsub = self.n_substeps_host(st, fl, fdt, active)
            for _ in range(8):
                self.set_dt(fdt / nsub)
                out = st
                for _ in range(nsub):
                    out = self.substep(out, fl, P, nu_on, self.mix_on(Pnp, fdt / nsub))
                l = out["l"].numpy()
                if np.isfinite(l).all() and l[self.g.mask_np].min() > self.l_floor and all(
                        np.isfinite(out[c].numpy()).all() for c in ("qx", "qy", "m")):
                    break
                nsub *= 2
            else:
                raise RuntimeError(f"frame {k} failed after retries")
        else:
            self.set_dt(fdt / nsub)
            out = st
            for _ in range(nsub):
                out = self.substep(out, fl, P, nu_on, self.mix_on(Pnp, fdt / nsub))
        self.stats["frames"] += 1
        return out, nsub

    # ------------------------------------------------------------------ fast path (persistent buffers, graphs)
    def _fast_setup(self, P, S):
        """Persistent copies of the parameters and of the schedule (graphs hold pointers) + work buffers."""
        g = self.g
        if self.fb is None:
            z = lambda: wp.zeros((g.N, g.N), dtype=F, device=g.dev)
            names = ("l qx qy m l0 qx0 qy0 m0 Kraw Kd s Lam vsx vsy sm ej ls ux uy c Fxqx Fxqy Fyqx Fyqy ax ay coef Gex Gey "
                     "Gx Gy l2 qxn qyn cfx cfy qx3 qy3 m3 qx4 qy4 m4 nuc").split()
            self.fb = {k: z() for k in names}
            self.fb["sums"] = wp.zeros(2, dtype=F, device=g.dev)
            self.fb["sumsd"] = wp.zeros(2, dtype=F, device=g.dev)
            self.fb["rows"] = wp.zeros(g.N, dtype=F, device=g.dev)
            self.fb["rows2"] = wp.zeros(g.N, dtype=F, device=g.dev)
            self.fb["red"] = wp.zeros(16, dtype=F, device=g.dev)      # [0:8] frame acceptance, [8:16] CFL speed
            self.fb["red_check"] = self.fb["red"][0:8]
            self.fb["red_speed"] = self.fb["red"][8:16]
            self.fb["P"] = wp.zeros(len(PARAMS), dtype=F, device=g.dev)
            self.fb["S"] = {}
        fb = self.fb
        wp.copy(fb["P"], P)
        n = S["x"].shape[0]
        if not fb["S"] or fb["S"]["x"].shape[0] != n:
            fb["S"] = {k: wp.zeros(n, dtype=F, device=g.dev) for k in SCHED}
            self.runner.graphs.clear()                       # graphs held the old schedule arrays
        for k in SCHED:
            wp.copy(fb["S"][k], S[k])

    def _g_fields(self):
        g, fb = self.g, self.fb
        S, P = fb["S"], fb["P"]
        for dep, K_, s_ in ((0.0, "Kraw", "sums"), (1.0, "Kd", "sumsd")):
            wp.launch(k_zero1, dim=2, inputs=[fb[s_]], device=g.dev, record_tape=False)
            self.L(k_kernel_raw, [g.xc, g.yc, g.mask, S["x"], S["y"], S["ux"], S["uy"], P, F(dep), S["r1"], S["r2"],
                                  S["phi"], self.kq, F(g.h)], [fb[K_], fb[s_]], record=False)
        self.L(k_fields, [fb["Kraw"], fb["sums"], fb["Kd"], fb["sumsd"], g.mask, g.xc, g.yc, P, S["x"], S["y"], S["Q"],
                          S["ux"], S["uy"], S["U"], S["d"], S["r1"], S["r2"], S["vjx"], S["vjy"], self.kq, F(g.h * g.h),
                          self.v05, self.up, fb["l"], fb["m"]], [fb["s"], fb["Lam"], fb["vsx"], fb["vsy"], fb["sm"], fb["ej"]],
               record=False)

    def _g_zero_fields(self):
        for k in ("s", "Lam", "vsx", "vsy", "sm", "ej"):
            self.L(k_set0, [self.fb[k]], record=False)

    def _g_speed(self, active):
        g, fb = self.g, self.fb
        wp.launch(k_rowmax_speed, dim=g.N, inputs=[fb["qx"], fb["qy"], fb["l"], fb["s"], fb["Lam"], fb["vsx"], fb["vsy"],
                                                   g.mask, g.N, int(active), self.Vbx, self.Vby, fb["P"], fb["rows"]],
                  device=g.dev, record_tape=False)
        wp.launch(k_rowreduce3, dim=1, inputs=[fb["rows"], fb["rows2"], g.N, fb["red_speed"]], device=g.dev, record_tape=False)

    def _g_check(self):
        g, fb = self.g, self.fb
        wp.launch(k_rowcheck, dim=g.N, inputs=[fb["l"], fb["qx"], fb["qy"], fb["m"], g.mask, g.N, fb["rows"], fb["rows2"]],
                  device=g.dev, record_tape=False)
        wp.launch(k_rowreduce3, dim=1, inputs=[fb["rows"], fb["rows2"], g.N, fb["red_check"]], device=g.dev, record_tape=False)

    def _g_save(self, back=False):
        fb = self.fb
        for a in ("l", "qx", "qy", "m"):
            if back:
                self.cp(fb[a + "0"], fb[a])
            else:
                self.cp(fb[a], fb[a + "0"])

    def _g_pre(self):
        g, fb, M = self.g, self.fb, self.opM
        N, h = g.N, g.h
        ih, ih2, dt, P = F(1.0 / h), F(1.0 / (h * h)), self.dta, fb["P"]
        R = lambda kern, ins, outs: self.L(kern, ins, outs, record=False)
        R(k_prim, [fb["l"], fb["qx"], fb["qy"], fb["m"], g.mask], [fb["ls"], fb["ux"], fb["uy"], fb["c"]])
        R(k_rusanov, [fb["qx"], fb["qy"], fb["ux"], fb["uy"], g.fx, g.fy], [fb["Fxqx"], fb["Fxqy"], fb["Fyqx"], fb["Fyqy"]])
        R(k_stepA_B, [fb["qx"], fb["qy"], fb["Fxqx"], fb["Fxqy"], fb["Fyqx"], fb["Fyqy"], fb["s"], fb["Lam"], fb["vsx"],
                      fb["vsy"], fb["ls"], P, dt, ih, N, self.Vbx, self.Vby, fb["sm"], fb["ux"], fb["uy"]],
          [fb["ax"], fb["ay"], fb["coef"]])
        R(k_massflux, [fb["ax"], fb["ay"], fb["l"], fb["ls"], g.fx, g.fy], [fb["Gex"], fb["Gey"]])
        R(k_facecoef, [fb["coef"], g.fx, g.fy, dt], [M.c["kx"], M.c["ky"]])
        R(k_massdiag, [M.c["kx"], M.c["ky"], dt, ih2, N], [M.d[0]])
        R(k_rhs, [fb["l"], fb["s"], fb["Gex"], fb["Gey"], dt, ih, N], [M.b[0]])
        self.cp(fb["l"], M.x[0])

    def _g_mid(self, nu_on):
        g, fb, M, V = self.g, self.fb, self.opM, self.opV
        N, h = g.N, g.h
        ih, dt, P = F(1.0 / h), self.dta, fb["P"]
        R = lambda kern, ins, outs: self.L(kern, ins, outs, record=False)
        R(k_finishB, [fb["Gex"], fb["Gey"], M.c["kx"], M.c["ky"], M.x[0], g.fx, g.fy, ih], [fb["Gx"], fb["Gy"]])
        R(k_lcons_q, [fb["l"], fb["s"], fb["Gx"], fb["Gy"], fb["ax"], fb["ay"], fb["coef"], g.fx, g.fy, g.nfx, g.nfy, dt, ih,
                      N], [fb["l2"]])
        R(k_qgrad, [fb["ax"], fb["ay"], fb["coef"], fb["l2"], g.fx, g.fy, g.nfx, g.nfy, dt, ih, N], [fb["qxn"], fb["qyn"]])
        R(k_limit, [fb["c"], fb["Gx"], fb["Gy"], g.fx, g.fy, N], [fb["cfx"], fb["cfy"]])
        R(k_milk, [fb["m"], fb["sm"], fb["Gx"], fb["Gy"], fb["cfx"], fb["cfy"], fb["l2"], fb["qxn"], fb["qyn"], g.mask, dt, ih, N],
          [V.c["l3"], fb["qx3"], fb["qy3"], fb["m3"]])
        if nu_on:
            R(k_visc_nu, [fb["qx3"], fb["qy3"], V.c["l3"], fb["m3"], g.mask, g.fx, g.fy, P, ih, N], [fb["nuc"]])
            R(k_visc_coefs, [V.c["l3"], g.fx, g.fy, g.cm, g.geo, P, fb["nuc"], N], [V.c["wx"], V.c["wy"], V.c["wc"], V.c["ww"]])
            R(k_visc_rhs, [fb["qx3"], fb["qy3"], V.c["l3"], g.mask], [V.x[0], V.x[1], V.b[0], V.b[1]])
            R(k_visc_diag, [V.c["l3"], V.c["wx"], V.c["wy"], V.c["wc"], V.c["ww"], g.nbx, g.nby, g.mask, dt, ih, N],
              [V.d[0], V.d[1]])

    def _g_post(self, nu_on, mix_on):
        g, fb, V = self.g, self.fb, self.opV
        N, h = g.N, g.h
        R = lambda kern, ins, outs: self.L(kern, ins, outs, record=False)
        if nu_on:
            R(k_visc_out, [V.x[0], V.x[1], V.c["l3"], g.mask], [fb["qx"], fb["qy"]])
        else:
            self.cp(fb["qx3"], fb["qx"])
            self.cp(fb["qy3"], fb["qy"])
        if mix_on:
            R(k_mix, [fb["m3"], V.c["l3"], g.fx, g.fy, g.mask, fb["P"], self.dta, F(1.0 / (h * h)), N], [fb["m"]])
        else:
            self.cp(fb["m3"], fb["m"])
        self.cp(V.c["l3"], fb["l"])
        if self.ex_on:
            R(k_exchange, [fb["l"], fb["qx"], fb["qy"], fb["m"], self.Kb, self.Db, self.Wf, fb["ej"], fb["P"], self.dta, g.mask],
              [fb["l"], fb["qx"], fb["qy"], fb["m"]])

    def cg_iterations(self):
        """PCG iterations so far (mass, viscosity; forward and adjoint solves together).  Synchronises."""
        return dict(mass=self.opM.iterations(), visc=self.opV.iterations())

    def reset_cg_iterations(self):
        self.opM.reset_iterations()
        self.opV.reset_iterations()

    def _substep_launches(self, nu_on, mix_on):
        self._g_pre()
        self.opM.launch_solve(self.cg_tol, self.cg_maxiter)
        self._g_mid(nu_on)
        if nu_on:
            self.opV.launch_solve(self.visc_tol, self.cg_maxiter)
        self._g_post(nu_on, mix_on)

    def fast_substep(self, nu_on, mix_on):
        """One sub-step on the persistent buffers: a single graph (both PCG loops as conditional nodes) when the
        device supports it, else pre / PCG / mid / PCG / post pieces."""
        R = self.runner
        if self.fused and self.opM.mode == "device" and self.opV.mode == "device":
            try:
                R(f"sub{int(nu_on)}{int(mix_on)}{int(self.ex_on)}", lambda: self._substep_launches(nu_on, mix_on))
                self.stats["substeps"] += 1
                return
            except Exception as e:
                if not R.use:
                    raise
                print(f"[wsolver] fused sub-step graph unavailable ({type(e).__name__}: {e}); piecewise graphs", flush=True)
                R.graphs.pop(f"sub{int(nu_on)}{int(mix_on)}{int(self.ex_on)}", None)
                self.fused = False
        R("pre", self._g_pre)
        self.opM.run(self.cg_tol, self.cg_maxiter)
        R(f"mid{int(nu_on)}", lambda: self._g_mid(nu_on))
        if nu_on:
            self.opV.run(self.visc_tol, self.cg_maxiter)
        R(f"post{int(nu_on)}{int(mix_on)}{int(self.ex_on)}", lambda: self._g_post(nu_on, mix_on))
        self.stats["substeps"] += 1

    def fast_forward(self, P, Pnp, S, sched, keep=True, nsub=None, frames=None, snap=()):
        """Whole pour on persistent buffers.  P, S: device arrays (copied in); sched: host schedule (fdt, q, active).
        Returns final state (device arrays, copies), checkpoints (device arrays, state at each frame start) and the
        sub-step counts; with snap (frame indices) self.snaps = {k: state after frame k}.
        Host synchronisation: none with nsub given; otherwise one read-back per frame (acceptance of this frame +
        the CFL speed of the next, computed in the same graph)."""
        self.snaps = {}
        snap = set(snap)
        self._fast_setup(P, S)
        self.set_ex(Pnp)
        R, fb = self.runner, self.fb
        R("init", lambda: self.L(k_init_state, [], [fb["l"], fb["qx"], fb["qy"], fb["m"]], record=False))
        nu_on = Pnp[NU] > 0
        ck, ns = [], []
        n = sched["n"] if frames is None else min(frames, sched["n"])
        fdt = lambda k: float(sched["fdt"][k])
        act = lambda k: bool(sched["active"][k])

        def prep(k):                                       # inlet fields of frame k (+ its CFL speed when adaptive)
            self.set_kq(k, int(sched["q"][k]))
            if nsub is None:
                R(f"prep{int(act(k))}", lambda: (self._g_fields() if act(k) else self._g_zero_fields(), self._g_speed(act(k))))
            else:
                R(f"fields{int(act(k))}", lambda: self._g_fields() if act(k) else self._g_zero_fields())

        prep(0)
        speed = float(fb["red_speed"].numpy()[0]) if nsub is None else 0.0
        for k in range(n):
            if keep:
                ck.append(self.clone_state({a: fb[a] for a in ("l", "qx", "qy", "m")}))
            if nsub is not None:
                n_ = nsub[k]
                self.set_dt(fdt(k) / n_)
                for _ in range(n_):
                    self.fast_substep(nu_on, self.mix_on(Pnp, fdt(k) / n_))
                if k + 1 < n:
                    prep(k + 1)
            else:
                n_ = self._nsub(speed, fdt(k))
                R("save", self._g_save)
                for _ in range(8):
                    self.set_dt(fdt(k) / n_)
                    for _ in range(n_):
                        self.fast_substep(nu_on, self.mix_on(Pnp, fdt(k) / n_))
                    if k + 1 < n:                           # acceptance of k and the next frame's fields + speed
                        self.set_kq(k + 1, int(sched["q"][k + 1]))
                        R(f"next{int(act(k + 1))}", lambda: (self._g_check(), self._g_fields() if act(k + 1)
                                                             else self._g_zero_fields(), self._g_speed(act(k + 1))))
                    else:
                        R("check", self._g_check)
                    red = fb["red"].numpy()                  # the one host synchronisation of the frame
                    if red[2] == 0 and red[1] > self.l_floor:
                        speed = float(red[8]) if k + 1 < n else 0.0
                        break
                    R("restore", lambda: self._g_save(back=True))
                    self.set_kq(k, int(sched["q"][k]))
                    R(f"fields{int(act(k))}", lambda: self._g_fields() if act(k) else self._g_zero_fields())
                    n_ *= 2
                else:
                    raise RuntimeError(f"frame {k} failed after retries")
            ns.append(n_)
            if k in snap:
                self.snaps[k] = self.clone_state({a: fb[a] for a in ("l", "qx", "qy", "m")})
            self.stats["frames"] += 1
        return self.clone_state({a: fb[a] for a in ("l", "qx", "qy", "m")}), ck, ns
