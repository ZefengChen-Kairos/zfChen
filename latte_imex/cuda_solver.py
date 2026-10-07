"""CUDA back-end of the IMEX + Rusanov surface-layer solver (numba.cuda, no CuPy dependency).

Same equations, same splitting and the same discrete formulas as solver.py (results agree to rounding);
everything stays on the device, one thread per cell, fused kernels:

  k_momentum      step A: Rusanov momentum fluxes on the 4 faces of a cell -> q*            (1 launch)
  k_prepare       step B: theta, q_hat/theta, c_p^2 l/theta                                 (1 launch)
  k_rhs_diag      step B: explicit layer flux on the 4 faces, rhs, Jacobi diagonal, x0       (1 launch)
  PCG             matrix-free apply (k_apply_A), dots by block reduction + atomics            (~6 launches/iter)
  k_finish        step B/C: final flux, conservative l^{n+1}, q^{n+1}, van Leer milk flux     (1 launch)
  k_viscosity     step D: explicit viscosity (sub-cycled if nu dt/h^2 is large)              (1 launch)
  k_mixing        step D: explicit mixing                                                     (1 launch)
  k_check/k_umax  reductions for the retry guard and the material CFL                        (2 launches)

Install on a CUDA machine:  pip install numba numba-cuda[cu12]   (or conda cudatoolkit for older numba).
Test without a GPU:         NUMBA_ENABLE_CUDASIM=1 python -m latte_imex.test_cuda   (slow, small grids only)
"""
from __future__ import annotations
import math, time
import numpy as np
from numba import cuda, float64, int32

from .solver import Params, Numerics, Inlet, Grid

TPB = (16, 16)     # threads per block (i fastest -> coalesced along rows)
RED = 256          # threads per block for 1-D reductions


# ----------------------------------------------------------------------------- device helpers
@cuda.jit(device=True, inline=True)
def _vl(r):
    return (r + abs(r)) / (1.0 + abs(r))


@cuda.jit(device=True, inline=True)
def _in(mask, j, i, N):
    return j >= 0 and j < N and i >= 0 and i < N and mask[j, i] != 0


@cuda.jit(device=True, inline=True)
def _lsafe(l, mask, j, i):
    return l[j, i] if mask[j, i] != 0 else 1.0


# ----------------------------------------------------------------------------- step A
@cuda.jit
def k_momentum(l, qx, qy, mask, dt, inv_h, qx_s, qy_s):
    """q* = q - dt div(F), Rusanov flux with alpha = 2 max(|u_L|, |u_R|); l frozen."""
    i, j = cuda.grid(2)
    N = l.shape[0]
    if j >= N or i >= N:
        return
    if mask[j, i] == 0:
        qx_s[j, i] = 0.0; qy_s[j, i] = 0.0
        return
    lc = l[j, i]
    uxc = qx[j, i] / lc; uyc = qy[j, i] / lc
    dx = 0.0; dy = 0.0
    # +x face
    if _in(mask, j, i + 1, N):
        ln = l[j, i + 1]; un = qx[j, i + 1] / ln
        a = 2.0 * max(abs(uxc), abs(un))
        dx += 0.5 * (qx[j, i] * uxc + qx[j, i + 1] * un) - 0.5 * a * (qx[j, i + 1] - qx[j, i])
        dy += 0.5 * (qy[j, i] * uxc + qy[j, i + 1] * un) - 0.5 * a * (qy[j, i + 1] - qy[j, i])
    # -x face
    if _in(mask, j, i - 1, N):
        ln = l[j, i - 1]; un = qx[j, i - 1] / ln
        a = 2.0 * max(abs(un), abs(uxc))
        dx -= 0.5 * (qx[j, i - 1] * un + qx[j, i] * uxc) - 0.5 * a * (qx[j, i] - qx[j, i - 1])
        dy -= 0.5 * (qy[j, i - 1] * un + qy[j, i] * uxc) - 0.5 * a * (qy[j, i] - qy[j, i - 1])
    # +y face
    if _in(mask, j + 1, i, N):
        ln = l[j + 1, i]; vn = qy[j + 1, i] / ln
        a = 2.0 * max(abs(uyc), abs(vn))
        dx += 0.5 * (qx[j, i] * uyc + qx[j + 1, i] * vn) - 0.5 * a * (qx[j + 1, i] - qx[j, i])
        dy += 0.5 * (qy[j, i] * uyc + qy[j + 1, i] * vn) - 0.5 * a * (qy[j + 1, i] - qy[j, i])
    # -y face
    if _in(mask, j - 1, i, N):
        ln = l[j - 1, i]; vn = qy[j - 1, i] / ln
        a = 2.0 * max(abs(vn), abs(uyc))
        dx -= 0.5 * (qx[j - 1, i] * vn + qx[j, i] * uyc) - 0.5 * a * (qx[j, i] - qx[j - 1, i])
        dy -= 0.5 * (qy[j - 1, i] * vn + qy[j, i] * uyc) - 0.5 * a * (qy[j, i] - qy[j - 1, i])
    qx_s[j, i] = qx[j, i] - dt * dx * inv_h
    qy_s[j, i] = qy[j, i] - dt * dy * inv_h


# ----------------------------------------------------------------------------- step B
@cuda.jit
def k_prepare(qx_s, qy_s, s, Lam, vsx, vsy, l, mask, beta, cp2, dt, a_x, a_y, coef):
    i, j = cuda.grid(2)
    N = l.shape[0]
    if j >= N or i >= N:
        return
    ls = _lsafe(l, mask, j, i)
    theta = 1.0 + dt * (beta + Lam[j, i] / ls)
    f = dt * (s[j, i] + Lam[j, i])
    a_x[j, i] = (qx_s[j, i] + f * vsx[j, i]) / theta
    a_y[j, i] = (qy_s[j, i] + f * vsy[j, i]) / theta
    coef[j, i] = cp2 * ls / theta


@cuda.jit(device=True, inline=True)
def _ge_x(a_x, l, mask, j, i):
    """Explicit layer flux through the x-face between (j,i) and (j,i+1); caller guarantees both inside."""
    w = max(abs(a_x[j, i] / l[j, i]), abs(a_x[j, i + 1] / l[j, i + 1]))
    return 0.5 * (a_x[j, i] + a_x[j, i + 1]) - 0.5 * w * (l[j, i + 1] - l[j, i])


@cuda.jit(device=True, inline=True)
def _ge_y(a_y, l, mask, j, i):
    w = max(abs(a_y[j, i] / l[j, i]), abs(a_y[j + 1, i] / l[j + 1, i]))
    return 0.5 * (a_y[j, i] + a_y[j + 1, i]) - 0.5 * w * (l[j + 1, i] - l[j, i])


@cuda.jit
def k_rhs_diag(l, a_x, a_y, coef, s, mask, dt, inv_h, rhs, diag, x0):
    """rhs = l + dt s - dt div(Ge);  diag = 1 + dt/h^2 sum k_f;  x0 = l (warm start)."""
    i, j = cuda.grid(2)
    N = l.shape[0]
    if j >= N or i >= N:
        return
    if mask[j, i] == 0:
        rhs[j, i] = 1.0; diag[j, i] = 1.0; x0[j, i] = 1.0
        return
    d = 0.0; ksum = 0.0
    cc = coef[j, i]
    if _in(mask, j, i + 1, N):
        d += _ge_x(a_x, l, mask, j, i); ksum += 0.5 * (cc + coef[j, i + 1])
    if _in(mask, j, i - 1, N):
        d -= _ge_x(a_x, l, mask, j, i - 1); ksum += 0.5 * (cc + coef[j, i - 1])
    if _in(mask, j + 1, i, N):
        d += _ge_y(a_y, l, mask, j, i); ksum += 0.5 * (cc + coef[j + 1, i])
    if _in(mask, j - 1, i, N):
        d -= _ge_y(a_y, l, mask, j - 1, i); ksum += 0.5 * (cc + coef[j - 1, i])
    rhs[j, i] = l[j, i] + dt * s[j, i] - dt * d * inv_h
    diag[j, i] = 1.0 + dt * dt * ksum * inv_h * inv_h
    x0[j, i] = l[j, i]


@cuda.jit
def k_apply_A(phi, coef, mask, dt, inv_h2, out):
    """out = phi - dt div(k grad phi), k_f = dt avg(coef) on faces inside the cup."""
    i, j = cuda.grid(2)
    N = phi.shape[0]
    if j >= N or i >= N:
        return
    if mask[j, i] == 0:
        out[j, i] = phi[j, i]
        return
    p = phi[j, i]; cc = coef[j, i]
    v = 0.0
    if _in(mask, j, i + 1, N):
        v += 0.5 * (cc + coef[j, i + 1]) * (phi[j, i + 1] - p)
    if _in(mask, j, i - 1, N):
        v += 0.5 * (cc + coef[j, i - 1]) * (phi[j, i - 1] - p)
    if _in(mask, j + 1, i, N):
        v += 0.5 * (cc + coef[j + 1, i]) * (phi[j + 1, i] - p)
    if _in(mask, j - 1, i, N):
        v += 0.5 * (cc + coef[j - 1, i]) * (phi[j - 1, i] - p)
    out[j, i] = p - dt * dt * v * inv_h2


@cuda.jit
def k_init_pcg(rhs, Ax, diag, r, p):
    """r = rhs - A x0 ;  p = r / diag."""
    i, j = cuda.grid(2)
    N = rhs.shape[0]
    if j >= N or i >= N:
        return
    rr = rhs[j, i] - Ax[j, i]
    r[j, i] = rr
    p[j, i] = rr / diag[j, i]


@cuda.jit
def k_axpy2(alpha, p, Ap, x, r):
    """x += alpha p ;  r -= alpha Ap   (one launch)."""
    i, j = cuda.grid(2)
    N = x.shape[0]
    if j >= N or i >= N:
        return
    x[j, i] += alpha * p[j, i]
    r[j, i] -= alpha * Ap[j, i]


@cuda.jit
def k_update_p(r, diag, beta, p):
    i, j = cuda.grid(2)
    N = p.shape[0]
    if j >= N or i >= N:
        return
    p[j, i] = r[j, i] / diag[j, i] + beta * p[j, i]


@cuda.jit
def k_dots(p, Ap, r, diag, out):
    """Block reduction of  p.Ap  and  r.r  and  r.(r/diag)  into out[0:3] (atomic adds)."""
    sh = cuda.shared.array((3, RED), float64)
    t = cuda.threadIdx.x
    N = p.shape[0]
    n = N * N
    k = cuda.grid(1)
    s0 = 0.0; s1 = 0.0; s2 = 0.0
    stride = cuda.gridsize(1)
    while k < n:
        jj = k // N; ii = k - jj * N
        s0 += p[jj, ii] * Ap[jj, ii]
        rv = r[jj, ii]
        s1 += rv * rv
        s2 += rv * rv / diag[jj, ii]
        k += stride
    sh[0, t] = s0; sh[1, t] = s1; sh[2, t] = s2
    cuda.syncthreads()
    w = RED // 2
    while w > 0:
        if t < w:
            sh[0, t] += sh[0, t + w]; sh[1, t] += sh[1, t + w]; sh[2, t] += sh[2, t + w]
        cuda.syncthreads()
        w //= 2
    if t == 0:
        cuda.atomic.add(out, 0, sh[0, 0])
        cuda.atomic.add(out, 1, sh[1, 0])
        cuda.atomic.add(out, 2, sh[2, 0])


@cuda.jit
def k_dot1(a, b, out):
    sh = cuda.shared.array(RED, float64)
    t = cuda.threadIdx.x
    N = a.shape[0]
    n = N * N
    k = cuda.grid(1)
    s0 = 0.0
    stride = cuda.gridsize(1)
    while k < n:
        jj = k // N; ii = k - jj * N
        s0 += a[jj, ii] * b[jj, ii]
        k += stride
    sh[t] = s0
    cuda.syncthreads()
    w = RED // 2
    while w > 0:
        if t < w:
            sh[t] += sh[t + w]
        cuda.syncthreads()
        w //= 2
    if t == 0:
        cuda.atomic.add(out, 0, sh[0])


# ----------------------------------------------------------------------------- step B finish + step C
@cuda.jit(device=True, inline=True)
def _cf_x(c, G, mask, j, i, N):
    """van Leer limited upwind c on the x-face (j,i)|(j,i+1); G>0 flows i -> i+1. Both cells inside."""
    cL = c[j, i]; cR = c[j, i + 1]
    d = cR - cL
    if abs(d) <= 1e-14:
        v = cL if G >= 0.0 else cR
    elif G >= 0.0:
        dl = (cL - c[j, i - 1]) if _in(mask, j, i - 1, N) else 0.0
        v = cL + 0.5 * _vl(dl / d) * d
    else:
        dr = (c[j, i + 2] - cR) if _in(mask, j, i + 2, N) else 0.0
        v = cR - 0.5 * _vl(dr / d) * d
    return min(max(v, min(cL, cR)), max(cL, cR))


@cuda.jit(device=True, inline=True)
def _cf_y(c, G, mask, j, i, N):
    cL = c[j, i]; cR = c[j + 1, i]
    d = cR - cL
    if abs(d) <= 1e-14:
        v = cL if G >= 0.0 else cR
    elif G >= 0.0:
        dl = (cL - c[j - 1, i]) if _in(mask, j - 1, i, N) else 0.0
        v = cL + 0.5 * _vl(dl / d) * d
    else:
        dr = (c[j + 2, i] - cR) if _in(mask, j + 2, i, N) else 0.0
        v = cR - 0.5 * _vl(dr / d) * d
    return min(max(v, min(cL, cR)), max(cL, cR))


@cuda.jit
def k_finish(l, x, a_x, a_y, coef, m, c, s, mask, dt, inv_h, l_out, qx_out, qy_out, m_out, acc):
    """Final flux G = Ge - k (x_R - x_L)/h with x = PCG solution; l^{n+1} = l + dt s - dt div G (conservative);
    q^{n+1} = a - dt coef grad_cell(l^{n+1}) (average of active face gradients of x);
    m^{n+1} = m + dt s - dt div(G c_f), c_f limited upwind;  acc[0] += clipped milk."""
    i, j = cuda.grid(2)
    N = l.shape[0]
    if j >= N or i >= N:
        return
    if mask[j, i] == 0:
        l_out[j, i] = 1.0; qx_out[j, i] = 0.0; qy_out[j, i] = 0.0; m_out[j, i] = 0.0
        return
    cc = coef[j, i]
    dG = 0.0; dM = 0.0; gx = 0.0; gy = 0.0; nx = 0; ny = 0
    if _in(mask, j, i + 1, N):
        G = _ge_x(a_x, l, mask, j, i) - dt * 0.5 * (cc + coef[j, i + 1]) * (x[j, i + 1] - x[j, i]) * inv_h
        dG += G; dM += G * _cf_x(c, G, mask, j, i, N)
        gx += (x[j, i + 1] - x[j, i]) * inv_h; nx += 1
    if _in(mask, j, i - 1, N):
        G = _ge_x(a_x, l, mask, j, i - 1) - dt * 0.5 * (cc + coef[j, i - 1]) * (x[j, i] - x[j, i - 1]) * inv_h
        dG -= G; dM -= G * _cf_x(c, G, mask, j, i - 1, N)
        gx += (x[j, i] - x[j, i - 1]) * inv_h; nx += 1
    if _in(mask, j + 1, i, N):
        G = _ge_y(a_y, l, mask, j, i) - dt * 0.5 * (cc + coef[j + 1, i]) * (x[j + 1, i] - x[j, i]) * inv_h
        dG += G; dM += G * _cf_y(c, G, mask, j, i, N)
        gy += (x[j + 1, i] - x[j, i]) * inv_h; ny += 1
    if _in(mask, j - 1, i, N):
        G = _ge_y(a_y, l, mask, j - 1, i) - dt * 0.5 * (cc + coef[j - 1, i]) * (x[j, i] - x[j - 1, i]) * inv_h
        dG -= G; dM -= G * _cf_y(c, G, mask, j - 1, i, N)
        gy += (x[j, i] - x[j - 1, i]) * inv_h; ny += 1
    ln = l[j, i] + dt * s[j, i] - dt * dG * inv_h
    l_out[j, i] = ln
    if nx > 0:
        gx /= nx
    if ny > 0:
        gy /= ny
    qx_out[j, i] = a_x[j, i] - dt * cc * gx
    qy_out[j, i] = a_y[j, i] - dt * cc * gy
    mn = m[j, i] + dt * s[j, i] - dt * dM * inv_h
    over = max(mn - ln, 0.0) + max(-mn, 0.0)
    if over > 0.0:
        cuda.atomic.add(acc, 0, over)
    m_out[j, i] = min(max(mn, 0.0), ln)


@cuda.jit
def k_concentration(m, l, mask, c):
    i, j = cuda.grid(2)
    N = l.shape[0]
    if j >= N or i >= N:
        return
    c[j, i] = m[j, i] / l[j, i] if mask[j, i] != 0 else 0.0


# ----------------------------------------------------------------------------- step D
@cuda.jit
def k_viscosity(l, qx, qy, mask, nu, dt, inv_h2, qx_out, qy_out):
    """q += dt div(nu l_f grad u), u = q/l, explicit; l frozen."""
    i, j = cuda.grid(2)
    N = l.shape[0]
    if j >= N or i >= N:
        return
    if mask[j, i] == 0:
        qx_out[j, i] = 0.0; qy_out[j, i] = 0.0
        return
    lc = l[j, i]; uxc = qx[j, i] / lc; uyc = qy[j, i] / lc
    vx = 0.0; vy = 0.0
    if _in(mask, j, i + 1, N):
        k = nu * 0.5 * (lc + l[j, i + 1]); vx += k * (qx[j, i + 1] / l[j, i + 1] - uxc); vy += k * (qy[j, i + 1] / l[j, i + 1] - uyc)
    if _in(mask, j, i - 1, N):
        k = nu * 0.5 * (lc + l[j, i - 1]); vx += k * (qx[j, i - 1] / l[j, i - 1] - uxc); vy += k * (qy[j, i - 1] / l[j, i - 1] - uyc)
    if _in(mask, j + 1, i, N):
        k = nu * 0.5 * (lc + l[j + 1, i]); vx += k * (qx[j + 1, i] / l[j + 1, i] - uxc); vy += k * (qy[j + 1, i] / l[j + 1, i] - uyc)
    if _in(mask, j - 1, i, N):
        k = nu * 0.5 * (lc + l[j - 1, i]); vx += k * (qx[j - 1, i] / l[j - 1, i] - uxc); vy += k * (qy[j - 1, i] / l[j - 1, i] - uyc)
    qx_out[j, i] = qx[j, i] + dt * vx * inv_h2
    qy_out[j, i] = qy[j, i] + dt * vy * inv_h2


@cuda.jit
def k_mixing(l, m, mask, D, dt, inv_h2, m_out):
    """m += dt div(D l_f grad c), c = m/l, explicit, clipped to [0, l]."""
    i, j = cuda.grid(2)
    N = l.shape[0]
    if j >= N or i >= N:
        return
    if mask[j, i] == 0:
        m_out[j, i] = 0.0
        return
    lc = l[j, i]; cc = m[j, i] / lc
    v = 0.0
    if _in(mask, j, i + 1, N):
        v += D * 0.5 * (lc + l[j, i + 1]) * (m[j, i + 1] / l[j, i + 1] - cc)
    if _in(mask, j, i - 1, N):
        v += D * 0.5 * (lc + l[j, i - 1]) * (m[j, i - 1] / l[j, i - 1] - cc)
    if _in(mask, j + 1, i, N):
        v += D * 0.5 * (lc + l[j + 1, i]) * (m[j + 1, i] / l[j + 1, i] - cc)
    if _in(mask, j - 1, i, N):
        v += D * 0.5 * (lc + l[j - 1, i]) * (m[j - 1, i] / l[j - 1, i] - cc)
    mn = m[j, i] + dt * v * inv_h2
    m_out[j, i] = min(max(mn, 0.0), lc)


# ----------------------------------------------------------------------------- reductions for the frame logic
@cuda.jit
def k_check(l, qx, qy, m, mask, l_floor, flags):
    """flags[0] += 1 for every bad cell (non-finite anywhere, or l <= floor inside the cup)."""
    i, j = cuda.grid(2)
    N = l.shape[0]
    if j >= N or i >= N:
        return
    a = l[j, i]; b = qx[j, i]; c = qy[j, i]; d = m[j, i]
    bad = (math.isnan(a) or math.isinf(a) or math.isnan(b) or math.isinf(b) or math.isnan(c) or math.isinf(c)
           or math.isnan(d) or math.isinf(d))
    if mask[j, i] != 0 and a <= l_floor:
        bad = True
    if bad:
        cuda.atomic.add(flags, 0, 1)


@cuda.jit
def k_umax(l, qx, qy, mask, out):
    i, j = cuda.grid(2)
    N = l.shape[0]
    if j >= N or i >= N:
        return
    if mask[j, i] == 0:
        return
    u = math.sqrt(qx[j, i] * qx[j, i] + qy[j, i] * qy[j, i]) / l[j, i]
    cuda.atomic.max(out, 0, u)


@cuda.jit
def k_vsmax(s, Lam, vsx, vsy, out):
    """max |v*| over cells with s + Lam > 0 (the CPU CFL bound)."""
    i, j = cuda.grid(2)
    N = s.shape[0]
    if j >= N or i >= N:
        return
    if s[j, i] + Lam[j, i] > 0.0:
        cuda.atomic.max(out, 1, math.sqrt(vsx[j, i] * vsx[j, i] + vsy[j, i] * vsy[j, i]))


@cuda.jit
def k_energy(l, qx, qy, mask, cp2, out):
    """out[0] += kinetic 0.5 |q|^2/l, out[1] += pressure 0.5 cp^2 (l^2 - 1)  (sum over cells)."""
    i, j = cuda.grid(2)
    N = l.shape[0]
    if j >= N or i >= N or mask[j, i] == 0:
        return
    lc = l[j, i]
    cuda.atomic.add(out, 0, 0.5 * (qx[j, i] * qx[j, i] + qy[j, i] * qy[j, i]) / lc)
    cuda.atomic.add(out, 1, 0.5 * cp2 * lc * lc)


# ----------------------------------------------------------------------------- inlet
@cuda.jit
def k_inlet_kernel(mask, h, x0, y0, r1, r2, cphi, sphi, q, i0, i1, j0, j1, Kout, Z):
    """Compact ellipse kernel (1 - a^2 - b^2)_+^2 with q x q sub-cell quadrature on the bounding box; Z[0] += sum."""
    i, j = cuda.grid(2)
    N = mask.shape[0]
    if j >= N or i >= N:
        return
    if i < i0 or i >= i1 or j < j0 or j >= j1 or mask[j, i] == 0:
        Kout[j, i] = 0.0
        return
    xc = (i + 0.5) * h - 0.5
    yc = (j + 0.5) * h - 0.5
    acc = 0.0
    for a_ in range(q):
        ox = (a_ + 0.5) / q - 0.5
        for b_ in range(q):
            oy = (b_ + 0.5) / q - 0.5
            dx = xc + ox * h - x0
            dy = yc + oy * h - y0
            a = (cphi * dx + sphi * dy) / r1
            b = (-sphi * dx + cphi * dy) / r2
            v = 1.0 - a * a - b * b
            if v > 0.0:
                acc += v * v
    Kout[j, i] = acc
    if acc > 0.0:
        cuda.atomic.add(Z, 0, acc)


@cuda.jit
def k_inlet_fields(Kraw, invZ, chiS, LamS, h, x0, y0, Rdep, kt_ux, kt_uy, rad, s, Lam, vsx, vsy):
    """s = chi S K, Lam = kappa_c (1-chi) S K, v* = kappa_t u_in + rad * (x - x0)/sqrt(Rdep^2 + |x - x0|^2)."""
    i, j = cuda.grid(2)
    N = Kraw.shape[0]
    if j >= N or i >= N:
        return
    Kn = Kraw[j, i] * invZ
    s[j, i] = chiS * Kn
    Lam[j, i] = LamS * Kn
    dx = (i + 0.5) * h - 0.5 - x0
    dy = (j + 0.5) * h - 0.5 - y0
    den = math.sqrt(Rdep * Rdep + dx * dx + dy * dy)
    vsx[j, i] = kt_ux + rad * dx / den
    vsy[j, i] = kt_uy + rad * dy / den


@cuda.jit
def k_zero(a):
    i, j = cuda.grid(2)
    N = a.shape[0]
    if j < N and i < N:
        a[j, i] = 0.0


# ----------------------------------------------------------------------------- host side
class CudaSolver:
    """Drop-in replacement for solver.Solver (same attributes used by run.py), state resident on the GPU."""

    def __init__(self, params: Params, num: Numerics, dtype=np.float64):
        self.P, self.num = params, num
        self.g = Grid(num.N, num.cup_radius)
        g = self.g; N = g.N
        self.dtype = dtype
        self.t = 0.0
        self.deposited = 0.0
        self.stats = dict(steps=0, frames=0, cg_mass=0, cg_visc=0, cg_mix=0, clip_mass=0.0, max_substeps=0,
                          max_u=0.0, retries=0, launches=0)
        self.l_floor = 1e-3
        self.grid2 = ((N + TPB[0] - 1) // TPB[0], (N + TPB[1] - 1) // TPB[1])
        self.red_blocks = min(1024, (N * N + RED - 1) // RED)
        z = np.zeros((N, N), dtype)
        self.mask = cuda.to_device(g.mask.astype(np.uint8))
        self.l = cuda.to_device(np.ones((N, N), dtype)); self.qx = cuda.to_device(z); self.qy = cuda.to_device(z); self.m = cuda.to_device(z)
        names = ["l_save", "qx_save", "qy_save", "m_save", "qx_s", "qy_s", "qx_t", "qy_t", "m_t", "a_x", "a_y", "coef",
                 "rhs", "diag", "x", "r", "p", "Ap", "l_n", "qx_n", "qy_n", "m_n", "c", "s", "Lam", "vsx", "vsy", "Kraw"]
        for nm in names:
            setattr(self, nm, cuda.to_device(z))
        self.scal = cuda.to_device(np.zeros(4, np.float64))
        self.red = cuda.to_device(np.zeros(3, np.float64))
        self._zero4 = np.zeros(4); self._zero3 = np.zeros(3)
        self.flags = cuda.to_device(np.zeros(1, np.int32)); self._zero_flag = np.zeros(1, np.int32)
        self._chi = 0.0
        self._inlet_sum = 0.0   # sum(s)*area for the ledger (= chi * S_eff)

    # ----- data access -------------------------------------------------------
    def concentration(self):
        k_concentration[self.grid2, TPB](self.m, self.l, self.mask, self.c)
        c = self.c.copy_to_host()
        return np.where(self.g.mask, c, np.nan)

    def host_state(self):
        return dict(l=self.l.copy_to_host(), qx=self.qx.copy_to_host(), qy=self.qy.copy_to_host(), m=self.m.copy_to_host())

    def ledger(self):
        st = self.host_state(); g = self.g
        L = float(st["l"][g.mask].sum() * g.area); M = float(st["m"][g.mask].sum() * g.area)
        return dict(t=self.t, layer=L, milk=M, brown=L - M, deposited=self.deposited,
                    layer_error=L - (g.n_cells * g.area + self.deposited), milk_error=M - self.deposited)

    def energy(self):
        self.scal.copy_to_device(self._zero4)
        k_energy[self.grid2, TPB](self.l, self.qx, self.qy, self.mask, self.P.cp ** 2, self.scal)
        e = self.scal.copy_to_host()
        return float(e[0] * self.g.area), float(e[1] * self.g.area)

    # ----- inlet ----------------------------------------------------------------
    def set_inlet(self, inlet: Inlet):
        P, g = self.P, self.g
        if not inlet.active or inlet.S_eff <= 0:
            for a in (self.s, self.Lam, self.vsx, self.vsy):
                k_zero[self.grid2, TPB](a)
            self._chi = 0.0; self._inlet_sum = 0.0
            return 0.0
        r1, r2, phi = max(inlet.r1, 1.5 * g.h), max(inlet.r2, 1.5 * g.h), inlet.phi
        x0, y0 = inlet.x_hit
        q = min(12, max(self.num.kernel_quadrature, int(math.ceil(4 * g.h / min(r1, r2)))))
        R = max(r1, r2)
        i0 = max(0, int((x0 - R + 0.5) / g.h) - 1); i1 = min(g.N, int((x0 + R + 0.5) / g.h) + 2)
        j0 = max(0, int((y0 - R + 0.5) / g.h) - 1); j1 = min(g.N, int((y0 + R + 0.5) / g.h) + 2)
        if i1 <= i0 or j1 <= j0:
            raise ValueError("inlet footprint outside the grid")
        self.scal.copy_to_device(self._zero4)
        k_inlet_kernel[self.grid2, TPB](self.mask, g.h, x0, y0, r1, r2, math.cos(phi), math.sin(phi), q, i0, i1, j0, j1,
                                        self.Kraw, self.scal)
        Z = float(self.scal.copy_to_host()[0]) * g.area
        if Z <= 0:
            raise ValueError("inlet footprint does not intersect the cup")
        d = max(inlet.d_jet, 1e-6)
        zc = inlet.U_perp ** 2 / (P.B_dep * d)
        chi = 1.0 / (1.0 + zc ** P.p_dep)
        Rdep = math.sqrt(max(inlet.r1 * inlet.r2, 1e-12))
        if P.return_law == "v05":
            VR = inlet.U_perp * chi ** (1.0 / (2.0 * P.p_dep))
            VR = min(VR, inlet.U_perp, math.sqrt(P.B_dep * d))
            rad = P.kappa_r * VR
        else:
            rad = chi * P.kappa_r * inlet.U_perp
        k_inlet_fields[self.grid2, TPB](self.Kraw, 1.0 / Z, chi * inlet.S_eff, P.kappa_c * (1.0 - chi) * inlet.S_eff,
                                        g.h, x0, y0, Rdep, P.kappa_t * inlet.u_in[0], P.kappa_t * inlet.u_in[1], rad,
                                        self.s, self.Lam, self.vsx, self.vsy)
        self._chi = chi; self._inlet_sum = chi * inlet.S_eff
        return chi

    # ----- PCG ------------------------------------------------------------------
    def pcg(self, dt):
        g = self.g; inv_h2 = 1.0 / (g.h * g.h)
        G2, B = self.grid2, TPB
        k_apply_A[G2, B](self.x, self.coef, self.mask, dt, inv_h2, self.Ap)
        k_init_pcg[G2, B](self.rhs, self.Ap, self.diag, self.r, self.p)
        self.scal.copy_to_device(self._zero4); self.red.copy_to_device(self._zero3)
        k_dot1[self.red_blocks, RED](self.rhs, self.rhs, self.scal)
        k_dots[self.red_blocks, RED](self.p, self.Ap, self.r, self.diag, self.red)
        bnorm = math.sqrt(float(self.scal.copy_to_host()[0])) + 1e-300
        rz = float(self.red.copy_to_host()[2])
        tol = self.num.cg_tol
        it = 0
        for it in range(1, self.num.cg_maxiter + 1):
            k_apply_A[G2, B](self.p, self.coef, self.mask, dt, inv_h2, self.Ap)
            self.scal.copy_to_device(self._zero4)
            k_dot1[self.red_blocks, RED](self.p, self.Ap, self.scal)
            pAp = float(self.scal.copy_to_host()[0])
            alpha = rz / (pAp + 1e-300)
            k_axpy2[G2, B](alpha, self.p, self.Ap, self.x, self.r)
            self.red.copy_to_device(self._zero3)
            k_dots[self.red_blocks, RED](self.p, self.Ap, self.r, self.diag, self.red)   # [p.Ap, r.r, r.z]
            v = self.red.copy_to_host()
            if math.sqrt(v[1]) <= tol * bnorm:
                break
            rz_new = float(v[2])
            k_update_p[G2, B](self.r, self.diag, rz_new / rz, self.p)
            rz = rz_new
        return it

    # ----- one sub-step -----------------------------------------------------------
    def substep(self, dt):
        P, g = self.P, self.g
        G2, B = self.grid2, TPB
        h = g.h; inv_h = 1.0 / h; inv_h2 = inv_h * inv_h
        k_momentum[G2, B](self.l, self.qx, self.qy, self.mask, dt, inv_h, self.qx_s, self.qy_s)
        k_prepare[G2, B](self.qx_s, self.qy_s, self.s, self.Lam, self.vsx, self.vsy, self.l, self.mask,
                         P.beta, P.cp ** 2, dt, self.a_x, self.a_y, self.coef)
        k_rhs_diag[G2, B](self.l, self.a_x, self.a_y, self.coef, self.s, self.mask, dt, inv_h, self.rhs, self.diag, self.x)
        it = self.pcg(dt)
        self.stats["cg_mass"] += it
        k_concentration[G2, B](self.m, self.l, self.mask, self.c)
        self.scal.copy_to_device(self._zero4)
        k_finish[G2, B](self.l, self.x, self.a_x, self.a_y, self.coef, self.m, self.c, self.s, self.mask, dt, inv_h,
                        self.l_n, self.qx_n, self.qy_n, self.m_n, self.scal)
        self.stats["clip_mass"] += float(self.scal.copy_to_host()[0]) * g.area
        # step D: explicit viscosity (sub-cycled if needed) and mixing, l frozen at l^{n+1}
        if P.nu > 0:
            r = P.nu * dt * inv_h2
            nsub = max(1, int(math.ceil(r / self.num.explicit_visc_limit)))
            src_x, src_y, dst_x, dst_y = self.qx_n, self.qy_n, self.qx_t, self.qy_t
            for _ in range(nsub):
                k_viscosity[G2, B](self.l_n, src_x, src_y, self.mask, P.nu, dt / nsub, inv_h2, dst_x, dst_y)
                src_x, src_y, dst_x, dst_y = dst_x, dst_y, src_x, src_y
            self.qx_n, self.qy_n, self.qx_t, self.qy_t = src_x, src_y, dst_x, dst_y
        if P.D > 0:
            r = P.D * dt * inv_h2
            nsub = max(1, int(math.ceil(r / 0.2)))
            src, dst = self.m_n, self.m_t
            for _ in range(nsub):
                k_mixing[G2, B](self.l_n, src, self.mask, P.D, dt / nsub, inv_h2, dst)
                src, dst = dst, src
            self.m_n, self.m_t = src, dst
        self.flags.copy_to_device(self._zero_flag)
        k_check[G2, B](self.l_n, self.qx_n, self.qy_n, self.m_n, self.mask, self.l_floor, self.flags)
        if int(self.flags.copy_to_host()[0]) > 0:
            return False
        # accept: swap state buffers
        self.l, self.l_n = self.l_n, self.l
        self.qx, self.qx_n = self.qx_n, self.qx
        self.qy, self.qy_n = self.qy_n, self.qy
        self.m, self.m_n = self.m_n, self.m
        self.deposited += self._inlet_sum * dt
        self.stats["steps"] += 1
        return True

    # ----- one control frame -------------------------------------------------------
    def advance_frame(self, inlet: Inlet, frame_dt=None):
        frame_dt = frame_dt or self.num.frame_dt
        if inlet.active and inlet.S_eff > 0 and inlet.scan_speed > 0:
            lim = self.num.scan_safety * min(self.g.h, inlet.r1, inlet.r2) / inlet.scan_speed
            frame_dt = min(frame_dt, max(lim, 1e-4))
        chi = self.set_inlet(inlet)
        g = self.g
        self.scal.copy_to_device(self._zero4)
        k_umax[self.grid2, TPB](self.l, self.qx, self.qy, self.mask, self.scal)
        if inlet.active and inlet.S_eff > 0:
            k_vsmax[self.grid2, TPB](self.s, self.Lam, self.vsx, self.vsy, self.scal)
        v = self.scal.copy_to_host()
        umax = max(float(v[0]), float(v[1]))
        self.stats["max_u"] = max(self.stats["max_u"], umax)
        dt_cfl = self.num.cfl * g.h / (2.0 * max(umax, 1e-9))
        nsub = max(1, min(int(math.ceil(frame_dt / dt_cfl)), self.num.max_substeps))
        self.l_save.copy_to_device(self.l); self.m_save.copy_to_device(self.m)
        self.qx_save.copy_to_device(self.qx); self.qy_save.copy_to_device(self.qy)
        saved_dep = self.deposited
        for attempt in range(8):
            dt = frame_dt / nsub
            ok = True
            for _ in range(nsub):
                if not self.substep(dt):
                    ok = False
                    break
            if ok:
                break
            self.l.copy_to_device(self.l_save); self.m.copy_to_device(self.m_save)
            self.qx.copy_to_device(self.qx_save); self.qy.copy_to_device(self.qy_save)
            self.deposited = saved_dep
            self.stats["retries"] += 1
            nsub *= 2
        else:
            raise RuntimeError("sub-step failed repeatedly")
        self.t += frame_dt
        self.stats["frames"] += 1
        self.stats["max_substeps"] = max(self.stats["max_substeps"], nsub)
        return chi, nsub

    def _rad_est(self, inlet, chi):
        """Upper bound of the radial return speed used in the CFL estimate (mirrors solver.inlet_fields)."""
        P = self.P
        d = max(inlet.d_jet, 1e-6)
        if P.return_law == "v05":
            VR = inlet.U_perp * chi ** (1.0 / (2.0 * P.p_dep))
            VR = min(VR, inlet.U_perp, math.sqrt(P.B_dep * d))
            return P.kappa_r * VR
        return chi * P.kappa_r * inlet.U_perp
