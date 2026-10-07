"""Numba kernels for the stencil operations (parallel over rows); NumPy fallbacks when numba is missing.

All arrays are [iy, ix] cell-centred (N, N); x-faces (N, N-1); y-faces (N-1, N).
"""
import numpy as np
try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]
    prange = range


@njit(parallel=True, cache=True)
def div(Fx, Fy, inv_h, out):
    N = out.shape[0]
    for j in prange(N):
        for i in range(N):
            v = 0.0
            if i < N - 1:
                v += Fx[j, i]
            if i > 0:
                v -= Fx[j, i - 1]
            if j < N - 1:
                v += Fy[j, i]
            if j > 0:
                v -= Fy[j - 1, i]
            out[j, i] = v * inv_h
    return out


@njit(parallel=True, cache=True)
def apply_diffusion(phi, kx, ky, fx, fy, a, b, inv_h2, out):
    """out = a*phi - b*div(k grad phi), k on faces, inactive faces masked. inv_h2 = 1/h^2."""
    N = phi.shape[0]
    for j in prange(N):
        for i in range(N):
            p = phi[j, i]
            v = 0.0
            if i < N - 1 and fx[j, i]:
                v += kx[j, i] * (phi[j, i + 1] - p)
            if i > 0 and fx[j, i - 1]:
                v += kx[j, i - 1] * (phi[j, i - 1] - p)
            if j < N - 1 and fy[j, i]:
                v += ky[j, i] * (phi[j + 1, i] - p)
            if j > 0 and fy[j - 1, i]:
                v += ky[j - 1, i] * (phi[j - 1, i] - p)
            out[j, i] = a[j, i] * p - b * v * inv_h2
    return out


@njit(parallel=True, cache=True)
def rusanov_fluxes(qx, qy, ux, uy, fx, fy, Fx_qx, Fx_qy, Fy_qx, Fy_qy):
    """Step A face fluxes, alpha = 2 max(|u_L|, |u_R|)."""
    N = qx.shape[0]
    for j in prange(N):
        for i in range(N - 1):
            if fx[j, i]:
                uL = ux[j, i]; uR = ux[j, i + 1]
                a = 2.0 * max(abs(uL), abs(uR))
                Fx_qx[j, i] = 0.5 * (qx[j, i] * uL + qx[j, i + 1] * uR) - 0.5 * a * (qx[j, i + 1] - qx[j, i])
                Fx_qy[j, i] = 0.5 * (qy[j, i] * uL + qy[j, i + 1] * uR) - 0.5 * a * (qy[j, i + 1] - qy[j, i])
            else:
                Fx_qx[j, i] = 0.0; Fx_qy[j, i] = 0.0
    for j in prange(N - 1):
        for i in range(N):
            if fy[j, i]:
                vL = uy[j, i]; vR = uy[j + 1, i]
                a = 2.0 * max(abs(vL), abs(vR))
                Fy_qx[j, i] = 0.5 * (qx[j, i] * vL + qx[j + 1, i] * vR) - 0.5 * a * (qx[j + 1, i] - qx[j, i])
                Fy_qy[j, i] = 0.5 * (qy[j, i] * vL + qy[j + 1, i] * vR) - 0.5 * a * (qy[j + 1, i] - qy[j, i])
            else:
                Fy_qx[j, i] = 0.0; Fy_qy[j, i] = 0.0


@njit(parallel=True, cache=True)
def mass_fluxes(ax, ay, l, lsafe, fx, fy, Gex, Gey, wx_out, wy_out):
    """Explicit part of the layer face flux: avg(a) - 0.5 w (l_R - l_L), w = max |a/l| of the two sides."""
    N = l.shape[0]
    for j in prange(N):
        for i in range(N - 1):
            if fx[j, i]:
                w = max(abs(ax[j, i] / lsafe[j, i]), abs(ax[j, i + 1] / lsafe[j, i + 1]))
                Gex[j, i] = 0.5 * (ax[j, i] + ax[j, i + 1]) - 0.5 * w * (l[j, i + 1] - l[j, i])
            else:
                Gex[j, i] = 0.0
    for j in prange(N - 1):
        for i in range(N):
            if fy[j, i]:
                w = max(abs(ay[j, i] / lsafe[j, i]), abs(ay[j + 1, i] / lsafe[j + 1, i]))
                Gey[j, i] = 0.5 * (ay[j, i] + ay[j + 1, i]) - 0.5 * w * (l[j + 1, i] - l[j, i])
            else:
                Gey[j, i] = 0.0


@njit(cache=True)
def _vl(r):
    return (r + abs(r)) / (1.0 + abs(r))


@njit(parallel=True, cache=True)
def limited_face_x(c, G, fx, out):
    """Upwind van Leer limited face value on x-faces (N, N-1); G>0 means flow ix -> ix+1."""
    N = c.shape[0]
    eps = 1e-14
    for j in prange(N):
        for i in range(N - 1):
            if not fx[j, i]:
                out[j, i] = 0.0
                continue
            cL = c[j, i]; cR = c[j, i + 1]
            d = cR - cL
            if abs(d) <= eps:
                v = cL if G[j, i] >= 0.0 else cR
            elif G[j, i] >= 0.0:
                dl = (cL - c[j, i - 1]) if (i > 0 and fx[j, i - 1]) else 0.0
                v = cL + 0.5 * _vl(dl / d) * d
            else:
                dr = (c[j, i + 2] - cR) if (i + 1 < N - 1 and fx[j, i + 1]) else 0.0
                v = cR - 0.5 * _vl(dr / d) * d
            lo = min(cL, cR); hi = max(cL, cR)
            out[j, i] = min(max(v, lo), hi)
    return out


@njit(parallel=True, cache=True)
def limited_face_y(c, G, fy, out):
    N = c.shape[0]
    eps = 1e-14
    for j in prange(N - 1):
        for i in range(N):
            if not fy[j, i]:
                out[j, i] = 0.0
                continue
            cL = c[j, i]; cR = c[j + 1, i]
            d = cR - cL
            if abs(d) <= eps:
                v = cL if G[j, i] >= 0.0 else cR
            elif G[j, i] >= 0.0:
                dl = (cL - c[j - 1, i]) if (j > 0 and fy[j - 1, i]) else 0.0
                v = cL + 0.5 * _vl(dl / d) * d
            else:
                dr = (c[j + 2, i] - cR) if (j + 1 < N - 1 and fy[j + 1, i]) else 0.0
                v = cR - 0.5 * _vl(dr / d) * d
            lo = min(cL, cR); hi = max(cL, cR)
            out[j, i] = min(max(v, lo), hi)
    return out


@njit(parallel=True, cache=True)
def axpy_dot(x, y, alpha):
    """x += alpha*y  (parallel), returns nothing."""
    N = x.shape[0]
    for j in prange(N):
        for i in range(x.shape[1]):
            x[j, i] += alpha * y[j, i]


@njit(parallel=True, cache=True)
def stepB_prepare(qx_s, qy_s, s, Lam, vsx, vsy, lsafe, beta, cp2, dt, a_x, a_y, coef):
    """theta = 1 + dt (beta + Lam/l); a = (q* + dt (s+Lam) v*)/theta; coef = cp^2 l / theta."""
    N = qx_s.shape[0]
    for j in prange(N):
        for i in range(N):
            ls = lsafe[j, i]
            theta = 1.0 + dt * (beta + Lam[j, i] / ls)
            f = dt * (s[j, i] + Lam[j, i])
            a_x[j, i] = (qx_s[j, i] + f * vsx[j, i]) / theta
            a_y[j, i] = (qy_s[j, i] + f * vsy[j, i]) / theta
            coef[j, i] = cp2 * ls / theta


@njit(parallel=True, cache=True)
def face_coef_and_diag(coef, fx, fy, dt, inv_h2, kx, ky, diag):
    """kx = dt*avg(coef)*fx on faces; diag = 1 + dt/h^2 * sum of the four adjacent k."""
    N = coef.shape[0]
    for j in prange(N):
        for i in range(N - 1):
            kx[j, i] = dt * 0.5 * (coef[j, i] + coef[j, i + 1]) if fx[j, i] else 0.0
    for j in prange(N - 1):
        for i in range(N):
            ky[j, i] = dt * 0.5 * (coef[j, i] + coef[j + 1, i]) if fy[j, i] else 0.0
    for j in prange(N):
        for i in range(N):
            v = 0.0
            if i < N - 1:
                v += kx[j, i]
            if i > 0:
                v += kx[j, i - 1]
            if j < N - 1:
                v += ky[j, i]
            if j > 0:
                v += ky[j - 1, i]
            diag[j, i] = 1.0 + dt * inv_h2 * v


@njit(parallel=True, cache=True)
def finish_B(Gex, Gey, kx, ky, l_new, fx, fy, inv_h, Gx, Gy):
    """Final layer face flux G = Ge - k (l_R - l_L)/h."""
    N = l_new.shape[0]
    for j in prange(N):
        for i in range(N - 1):
            Gx[j, i] = Gex[j, i] - kx[j, i] * (l_new[j, i + 1] - l_new[j, i]) * inv_h if fx[j, i] else 0.0
    for j in prange(N - 1):
        for i in range(N):
            Gy[j, i] = Gey[j, i] - ky[j, i] * (l_new[j + 1, i] - l_new[j, i]) * inv_h if fy[j, i] else 0.0


@njit(parallel=True, cache=True)
def q_from_gradient(a_x, a_y, coef, l_new, fx, fy, nfx, nfy, dt, inv_h, qx_new, qy_new):
    """q^{n+1} = a - dt coef * cell_grad(l^{n+1}); cell gradient = average of the active face gradients."""
    N = l_new.shape[0]
    for j in prange(N):
        for i in range(N):
            gx = 0.0; gy = 0.0
            if i < N - 1 and fx[j, i]:
                gx += (l_new[j, i + 1] - l_new[j, i]) * inv_h
            if i > 0 and fx[j, i - 1]:
                gx += (l_new[j, i] - l_new[j, i - 1]) * inv_h
            if j < N - 1 and fy[j, i]:
                gy += (l_new[j + 1, i] - l_new[j, i]) * inv_h
            if j > 0 and fy[j - 1, i]:
                gy += (l_new[j, i] - l_new[j - 1, i]) * inv_h
            nx = nfx[j, i]; ny = nfy[j, i]
            gx = gx / nx if nx > 0 else 0.0
            gy = gy / ny if ny > 0 else 0.0
            qx_new[j, i] = a_x[j, i] - dt * coef[j, i] * gx
            qy_new[j, i] = a_y[j, i] - dt * coef[j, i] * gy


@njit(parallel=True, cache=True)
def dot(a, b):
    N = a.shape[0]; M = a.shape[1]
    acc = 0.0
    for j in prange(N):
        s = 0.0
        for i in range(M):
            s += a[j, i] * b[j, i]
        acc += s
    return acc


@njit(parallel=True, cache=True)
def axpy(alpha, x, y):
    """y += alpha * x"""
    N = y.shape[0]; M = y.shape[1]
    for j in prange(N):
        for i in range(M):
            y[j, i] += alpha * x[j, i]


@njit(parallel=True, cache=True)
def zbp(r, diag, beta, p):
    """p = r/diag + beta * p"""
    N = p.shape[0]; M = p.shape[1]
    for j in prange(N):
        for i in range(M):
            p[j, i] = r[j, i] / diag[j, i] + beta * p[j, i]
