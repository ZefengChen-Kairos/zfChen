"""CUDA back-end of the IMEX + Rusanov surface-layer solver (numba.cuda, no CuPy dependency), batched.

Same equations, same splitting and the same discrete formulas as solver.py (results agree to rounding); the state
lives on the device as [B, N, N] arrays, B independent simulations (members) advanced in one launch set.
`CudaSolver` is the single-run drop-in for solver.Solver (B = 1); `BatchCudaSolver` runs B members, each with its own
inlet, frame length, sub-step count, retries, ledger and statistics (a member gives the same result as a B = 1 run).

Per sub-step (Params.visc = "symmetric", the default), launches:
  k_mom_prep     step A + step B prepare: Rusanov momentum q*, theta, a = (q* + dt(s+Lam)v*)/theta, c_p^2 l/theta
  k_mass_init    step B: explicit layer flux, rhs, Jacobi diagonal, x0 = l, r0 = rhs - A x0, |rhs|^2, r0.z0
  k_mass_k1/k2   PCG, 2 launches per iteration: k1 = p = z + beta p (computed on the fly at the stencil) + A p + p.Ap;
                 k2 = x += alpha p, r -= alpha Ap, r.r, r.z.  All scalars stay on the device ([B, it, 3] slots);
                 k1 decides convergence on the device (same test and the same iteration count as solver.pcg) and
                 turns the remaining launches into no-ops; the host only polls a [B, 2] status word once the
                 predicted iteration count (last solve's) is reached.
  k_finish       step B/C: final flux, conservative l^{n+1}, q^{n+1}, van Leer milk flux (c = m/l on the fly)
  k_sym_init     step D: viscous weights, X0 = q/l, b = l X0, diagonal, r0 = b - A X0 (corner shear on the fly)
  k_sym_k1/k2    PCG of (L + dt K) u+ = L u- on [u; v], as above (kernels.sym_visc_apply on the fly)
  k_final        q = l X, explicit mixing, finite / positive-layer check (flags on the device)
  = 5 + 2 (it_mass + it_visc) launches, ~2 host polls per PCG solve (instead of ~10 + 6 it launches and 2 host
  reads per iteration).  Per control frame: k_inlet_K (footprint kernel on its bounding box, in-cup / total mass and
  the off-cup fraction Z on the device), k_inlet_fields (s, Lam, v*, material CFL maxima, frame save); 1 upload of
  the inlet scalars, 1 read of (Z, umax), 1 read of the check flags at the end of the frame.  The retry guard is
  checked once per frame: a failed sub-step makes the whole frame retry (as in solver.Solver), so checking at the end
  of the frame gives the same accepted states.

Params.visc = "componentwise": explicit viscosity sub-cycled when nu dt/h^2 > 0.2, mixing explicit (sub-cycled when
D dt/h^2 > 0.2), as before.  Storage precision: float64 (default); dtype=np.float32 stores the fields in FP32
(arithmetic and reductions in FP64, PCG tolerances raised to 1e-6) -- experimental, not the reference.

Install on a CUDA machine:  pip install numba numba-cuda[cu12]
Test without a GPU:         NUMBA_ENABLE_CUDASIM=1 python -m latte_imex.test_cuda --N 20 --T 0.05   (slow, tiny grids)
"""
from __future__ import annotations
import math, os, time
import numpy as np
from numba import cuda, float64

from .solver import Params, Numerics, Inlet, Grid

SIM = os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1"
TPB = (32, 8)      # threads per block (i fastest -> coalesced rows); 256 threads
NT = 256           # threads per block (shared-memory reductions)
# per-member parameter columns (pm)
PM_CP2, PM_BETA, PM_NU, PM_D = 0, 1, 2, 3
# per-frame inlet table columns (ip); the last columns are device accumulators
IP_ACT, IP_X0, IP_Y0, IP_R1, IP_R2, IP_C, IP_S, IP_Q, IP_I0, IP_I1, IP_J0, IP_J1 = range(12)
IP_CHIS, IP_LAMS, IP_RDEP, IP_KTX, IP_KTY, IP_RAD, IP_ACC = 12, 13, 14, 15, 16, 17, 18
IP_ZIN, IP_ZTOT, IP_ZOUT, IP_UMAX, IP_VSMAX = 19, 20, 21, 22, 23
IP_NCOL = 24


# ============================================================================= device helpers
@cuda.jit(device=True, inline=True)
def _vl(r):
    return (r + abs(r)) / (1.0 + abs(r))


@cuda.jit(device=True, inline=True)
def _in(mask, j, i, N):
    return j >= 0 and j < N and i >= 0 and i < N and mask[j, i] != 0


@cuda.jit(device=True, inline=True)
def _tid():
    return cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x


@cuda.jit(device=True)
def _bred(sh, t, k):
    """Tree reduction of the first k rows of sh (k, NT) over the block; result in sh[:, 0]. All threads call."""
    cuda.syncthreads()
    w = NT // 2
    while w > 0:
        if t < w:
            for q in range(k):
                sh[q, t] += sh[q, t + w]
        cuda.syncthreads()
        w //= 2


@cuda.jit(device=True, inline=True)
def _first_thread():
    return (cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.blockIdx.x == 0 and cuda.blockIdx.y == 0)


@cuda.jit(device=True, inline=True)
def _cval(m, l, mask, b, j, i):
    return m[b, j, i] / l[b, j, i] if mask[j, i] != 0 else 0.0


@cuda.jit(device=True, inline=True)
def _ge_x(a_x, l, b, j, i):
    """Explicit layer flux through the x-face (j,i)|(j,i+1); both cells inside."""
    w = max(abs(a_x[b, j, i] / l[b, j, i]), abs(a_x[b, j, i + 1] / l[b, j, i + 1]))
    return 0.5 * (a_x[b, j, i] + a_x[b, j, i + 1]) - 0.5 * w * (l[b, j, i + 1] - l[b, j, i])


@cuda.jit(device=True, inline=True)
def _ge_y(a_y, l, b, j, i):
    w = max(abs(a_y[b, j, i] / l[b, j, i]), abs(a_y[b, j + 1, i] / l[b, j + 1, i]))
    return 0.5 * (a_y[b, j, i] + a_y[b, j + 1, i]) - 0.5 * w * (l[b, j + 1, i] - l[b, j, i])


@cuda.jit(device=True, inline=True)
def _cf_x(m, l, G, mask, b, j, i, N):
    """van Leer limited upwind c = m/l on the x-face (j,i)|(j,i+1); G>0 flows i -> i+1. Both cells inside."""
    cL = m[b, j, i] / l[b, j, i]; cR = m[b, j, i + 1] / l[b, j, i + 1]
    d = cR - cL
    if abs(d) <= 1e-14:
        v = cL if G >= 0.0 else cR
    elif G >= 0.0:
        dl = (cL - _cval(m, l, mask, b, j, i - 1)) if _in(mask, j, i - 1, N) else 0.0
        v = cL + 0.5 * _vl(dl / d) * d
    else:
        dr = (_cval(m, l, mask, b, j, i + 2) - cR) if _in(mask, j, i + 2, N) else 0.0
        v = cR - 0.5 * _vl(dr / d) * d
    return min(max(v, min(cL, cR)), max(cL, cR))


@cuda.jit(device=True, inline=True)
def _cf_y(m, l, G, mask, b, j, i, N):
    cL = m[b, j, i] / l[b, j, i]; cR = m[b, j + 1, i] / l[b, j + 1, i]
    d = cR - cL
    if abs(d) <= 1e-14:
        v = cL if G >= 0.0 else cR
    elif G >= 0.0:
        dl = (cL - _cval(m, l, mask, b, j - 1, i)) if _in(mask, j - 1, i, N) else 0.0
        v = cL + 0.5 * _vl(dl / d) * d
    else:
        dr = (_cval(m, l, mask, b, j + 2, i) - cR) if _in(mask, j + 2, i, N) else 0.0
        v = cR - 0.5 * _vl(dr / d) * d
    return min(max(v, min(cL, cR)), max(cL, cR))


# ============================================================================= frame: inlet
@cuda.jit
def k_inlet_K(mask, h, ip, Kraw):
    """Compact ellipse kernel (1 - a^2 - b^2)_+^2, q x q sub-cell quadrature, on the UNCLIPPED bounding box
    [i0, i1) x [j0, j1) of member b (solver.kernel_fraction): Kraw on the in-grid cup cells; ip[b, ZIN] += in-cup sum,
    ip[b, ZTOT] += total, ip[b, ZOUT] += number of cells with kernel mass outside the cup (or the grid)."""
    sh = cuda.shared.array((3, NT), float64)
    t = _tid()
    b = cuda.blockIdx.z
    N = mask.shape[0]
    ii, jj = cuda.grid(2)
    zin = 0.0; ztot = 0.0; zout = 0.0
    if ip[b, IP_ACT] != 0.0:
        i = int(ip[b, IP_I0]) + ii
        j = int(ip[b, IP_J0]) + jj
        if i < int(ip[b, IP_I1]) and j < int(ip[b, IP_J1]):
            q = int(ip[b, IP_Q])
            x0 = ip[b, IP_X0]; y0 = ip[b, IP_Y0]; r1 = ip[b, IP_R1]; r2 = ip[b, IP_R2]
            cphi = ip[b, IP_C]; sphi = ip[b, IP_S]
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
                    bb = (-sphi * dx + cphi * dy) / r2
                    v = 1.0 - a * a - bb * bb
                    if v > 0.0:
                        acc += v * v
            ztot = acc
            if i >= 0 and i < N and j >= 0 and j < N and mask[j, i] != 0:
                Kraw[b, j, i] = acc
                zin = acc
            elif acc > 0.0:
                zout = 1.0
    sh[0, t] = zin; sh[1, t] = ztot; sh[2, t] = zout
    _bred(sh, t, 3)
    if t == 0:
        if sh[1, 0] != 0.0:
            cuda.atomic.add(ip, (b, IP_ZIN), sh[0, 0])
            cuda.atomic.add(ip, (b, IP_ZTOT), sh[1, 0])
        if sh[2, 0] != 0.0:
            cuda.atomic.add(ip, (b, IP_ZOUT), sh[2, 0])


@cuda.jit(device=True, inline=True)
def _zfrac(ip, b):
    """Off-cup fraction Z of member b (solver.kernel_fraction) from the device accumulators."""
    tot = ip[b, IP_ZTOT]
    if tot <= 0.0:
        return 0.0
    if ip[b, IP_ZOUT] == 0.0:
        return 1.0
    return ip[b, IP_ZIN] / tot


@cuda.jit
def k_inlet_fields(Kraw, mask, h, area, ip, nsub, l, qx, qy, m, s, Lam, vsx, vsy, l_s, qx_s, qy_s, m_s):
    """s = chi S_rec K, Lam = kappa_c (1-chi) S_rec K (K normalised to the cup), v* = kappa_t u_in + rad (x-x0)/
    sqrt(Rdep^2 + |x-x0|^2); S_rec = S Z (offcup 'account') or S; CFL maxima |u| and |v*| (s+Lam>0) into ip; frame save."""
    i, j = cuda.grid(2)
    b = cuda.blockIdx.z
    N = mask.shape[0]
    if j >= N or i >= N or nsub[b] == 0:
        return
    l_s[b, j, i] = l[b, j, i]; qx_s[b, j, i] = qx[b, j, i]; qy_s[b, j, i] = qy[b, j, i]; m_s[b, j, i] = m[b, j, i]
    if mask[j, i] != 0:
        u = math.sqrt(qx[b, j, i] * qx[b, j, i] + qy[b, j, i] * qy[b, j, i]) / l[b, j, i]
        if u > 0.0:
            cuda.atomic.max(ip, (b, IP_UMAX), u)
    if ip[b, IP_ACT] == 0.0:
        s[b, j, i] = 0.0; Lam[b, j, i] = 0.0; vsx[b, j, i] = 0.0; vsy[b, j, i] = 0.0
        return
    Z = _zfrac(ip, b)
    sc = 1.0
    if ip[b, IP_ACC] != 0.0:
        sc = Z
    zin = ip[b, IP_ZIN] * area
    Kn = 0.0
    if ((ip[b, IP_ACC] == 0.0 or Z > 0.0) and zin > 0.0 and mask[j, i] != 0 and i >= int(ip[b, IP_I0])
            and i < int(ip[b, IP_I1]) and j >= int(ip[b, IP_J0]) and j < int(ip[b, IP_J1])):
        Kn = Kraw[b, j, i] / zin
    sv = ip[b, IP_CHIS] * sc * Kn
    lv = ip[b, IP_LAMS] * sc * Kn
    s[b, j, i] = sv
    Lam[b, j, i] = lv
    if ip[b, IP_ACC] != 0.0 and Z <= 0.0:
        vsx[b, j, i] = 0.0; vsy[b, j, i] = 0.0
        return
    x0 = ip[b, IP_X0]; y0 = ip[b, IP_Y0]; Rd = ip[b, IP_RDEP]; rad = ip[b, IP_RAD]
    dx = (i + 0.5) * h - 0.5 - x0
    dy = (j + 0.5) * h - 0.5 - y0
    den = math.sqrt(Rd * Rd + dx * dx + dy * dy)
    vx = ip[b, IP_KTX] + rad * dx / den
    vy = ip[b, IP_KTY] + rad * dy / den
    vsx[b, j, i] = vx; vsy[b, j, i] = vy
    if sv + lv > 0.0:
        cuda.atomic.max(ip, (b, IP_VSMAX), math.sqrt(vx * vx + vy * vy))


@cuda.jit
def k_restore(nsub, l_s, qx_s, qy_s, m_s, l, qx, qy, m):
    i, j = cuda.grid(2)
    b = cuda.blockIdx.z
    N = l.shape[1]
    if j >= N or i >= N or nsub[b] == 0:
        return
    l[b, j, i] = l_s[b, j, i]; qx[b, j, i] = qx_s[b, j, i]; qy[b, j, i] = qy_s[b, j, i]; m[b, j, i] = m_s[b, j, i]


# ============================================================================= step A + B
@cuda.jit
def k_mom_prep(l, qx, qy, mask, s, Lam, vsx, vsy, pm, dtv, nsub, rnd, inv_h, a_x, a_y, coef, redm):
    """Rusanov momentum (alpha = 2 max|u|, l frozen) -> q*, then theta = 1 + dt(beta + Lam/l),
    a = (q* + dt (s+Lam) v*)/theta, coef = c_p^2 l/theta.  Zeroes the PCG slot 0 of the mass solve."""
    i, j = cuda.grid(2)
    b = cuda.blockIdx.z
    N = l.shape[1]
    if rnd >= nsub[b]:
        return
    if _first_thread():
        redm[b, 0, 0] = 0.0; redm[b, 0, 1] = 0.0; redm[b, 0, 2] = 0.0
    if j >= N or i >= N:
        return
    dt = dtv[b]
    if mask[j, i] == 0:
        qsx = 0.0; qsy = 0.0
        ls = 1.0
    else:
        lc = l[b, j, i]
        qxc = qx[b, j, i]; qyc = qy[b, j, i]
        uxc = qxc / lc; uyc = qyc / lc
        dx = 0.0; dy = 0.0
        if _in(mask, j, i + 1, N):
            ln = l[b, j, i + 1]; un = qx[b, j, i + 1] / ln
            a = 2.0 * max(abs(uxc), abs(un))
            dx += 0.5 * (qxc * uxc + qx[b, j, i + 1] * un) - 0.5 * a * (qx[b, j, i + 1] - qxc)
            dy += 0.5 * (qyc * uxc + qy[b, j, i + 1] * un) - 0.5 * a * (qy[b, j, i + 1] - qyc)
        if _in(mask, j, i - 1, N):
            ln = l[b, j, i - 1]; un = qx[b, j, i - 1] / ln
            a = 2.0 * max(abs(un), abs(uxc))
            dx -= 0.5 * (qx[b, j, i - 1] * un + qxc * uxc) - 0.5 * a * (qxc - qx[b, j, i - 1])
            dy -= 0.5 * (qy[b, j, i - 1] * un + qyc * uxc) - 0.5 * a * (qyc - qy[b, j, i - 1])
        if _in(mask, j + 1, i, N):
            ln = l[b, j + 1, i]; vn = qy[b, j + 1, i] / ln
            a = 2.0 * max(abs(uyc), abs(vn))
            dx += 0.5 * (qxc * uyc + qx[b, j + 1, i] * vn) - 0.5 * a * (qx[b, j + 1, i] - qxc)
            dy += 0.5 * (qyc * uyc + qy[b, j + 1, i] * vn) - 0.5 * a * (qy[b, j + 1, i] - qyc)
        if _in(mask, j - 1, i, N):
            ln = l[b, j - 1, i]; vn = qy[b, j - 1, i] / ln
            a = 2.0 * max(abs(vn), abs(uyc))
            dx -= 0.5 * (qx[b, j - 1, i] * vn + qxc * uyc) - 0.5 * a * (qxc - qx[b, j - 1, i])
            dy -= 0.5 * (qy[b, j - 1, i] * vn + qyc * uyc) - 0.5 * a * (qyc - qy[b, j - 1, i])
        qsx = qxc - dt * dx * inv_h
        qsy = qyc - dt * dy * inv_h
        ls = lc
    lam = Lam[b, j, i]
    theta = 1.0 + dt * (pm[b, PM_BETA] + lam / ls)
    f = dt * (s[b, j, i] + lam)
    a_x[b, j, i] = (qsx + f * vsx[b, j, i]) / theta
    a_y[b, j, i] = (qsy + f * vsy[b, j, i]) / theta
    coef[b, j, i] = pm[b, PM_CP2] * ls / theta


@cuda.jit
def k_mass_init(l, a_x, a_y, coef, s, mask, dtv, nsub, rnd, inv_h, rhs, diag, x, r, redm, st):
    """rhs = l + dt s - dt div(Ge); diag = 1 + dt^2/h^2 sum k_f; x0 = l; r0 = rhs - A x0;
    redm[b,0,1] += rhs.rhs, redm[b,0,2] += r0.(r0/diag); zeroes the PCG slots 1.. and the status of member b."""
    sh = cuda.shared.array((2, NT), float64)
    t = _tid()
    b = cuda.blockIdx.z
    N = l.shape[1]
    if rnd >= nsub[b]:
        if _first_thread():
            st[b, 0] = 1; st[b, 1] = 0
        return
    i, j = cuda.grid(2)
    # zero the iteration slots (no other kernel touches them during this launch)
    nb = cuda.gridDim.x * cuda.gridDim.y * NT
    k = (cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.x) * NT + t
    nslot = redm.shape[1] * 3
    while k < nslot:
        sl = k // 3
        if sl > 0:
            redm[b, sl, k - sl * 3] = 0.0
        k += nb
    if _first_thread():
        st[b, 0] = 0; st[b, 1] = 0
    s0 = 0.0; s1 = 0.0
    if j < N and i < N:
        dt = dtv[b]
        if mask[j, i] == 0:
            rhs[b, j, i] = 1.0; diag[b, j, i] = 1.0; x[b, j, i] = 1.0; r[b, j, i] = 0.0
            s0 = 1.0
        else:
            d = 0.0; ksum = 0.0; v = 0.0
            cc = coef[b, j, i]; lc = l[b, j, i]
            if _in(mask, j, i + 1, N):
                kf = 0.5 * (cc + coef[b, j, i + 1])
                d += _ge_x(a_x, l, b, j, i); ksum += kf; v += kf * (l[b, j, i + 1] - lc)
            if _in(mask, j, i - 1, N):
                kf = 0.5 * (cc + coef[b, j, i - 1])
                d -= _ge_x(a_x, l, b, j, i - 1); ksum += kf; v += kf * (l[b, j, i - 1] - lc)
            if _in(mask, j + 1, i, N):
                kf = 0.5 * (cc + coef[b, j + 1, i])
                d += _ge_y(a_y, l, b, j, i); ksum += kf; v += kf * (l[b, j + 1, i] - lc)
            if _in(mask, j - 1, i, N):
                kf = 0.5 * (cc + coef[b, j - 1, i])
                d -= _ge_y(a_y, l, b, j - 1, i); ksum += kf; v += kf * (l[b, j - 1, i] - lc)
            rh = lc + dt * s[b, j, i] - dt * d * inv_h
            dg = 1.0 + dt * dt * ksum * inv_h * inv_h
            rr = rh - (lc - dt * dt * v * inv_h * inv_h)
            rhs[b, j, i] = rh; diag[b, j, i] = dg; x[b, j, i] = lc; r[b, j, i] = rr
            s0 = rh * rh
            s1 = rr * (rr / dg)
    sh[0, t] = s0; sh[1, t] = s1
    _bred(sh, t, 2)
    if t == 0:
        cuda.atomic.add(redm, (b, 0, 1), sh[0, 0])
        cuda.atomic.add(redm, (b, 0, 2), sh[1, 0])


@cuda.jit(device=True, inline=True)
def _pn(r, diag, po, b, j, i, beta, first):
    """New search direction p = r/diag + beta p_old at one cell (solver.pcg / kernels.zbp)."""
    if first:
        return r[b, j, i] / diag[b, j, i]
    return r[b, j, i] / diag[b, j, i] + beta * po[b, j, i]


@cuda.jit(device=True, inline=True)
def _pcg_gate(red, st, b, it, tol):
    """Convergence test of iteration it-1 on the device (solver.pcg: sqrt(r.r) <= tol |b|); returns beta, or -1 when the
    member is done (sets the status)."""
    if st[b, 0] != 0:
        return -1.0
    if it == 1:
        return 0.0
    bnorm = math.sqrt(red[b, 0, 1]) + 1e-300
    if not (math.sqrt(red[b, it - 1, 1]) > tol * bnorm):
        if _first_thread():
            st[b, 0] = 1
        return -1.0
    return red[b, it - 1, 2] / red[b, it - 2, 2]


@cuda.jit
def k_mass_k1(r, diag, po, pn_out, coef, mask, dtv, inv_h2, red, st, it, tol, Ap):
    """PCG iteration it, part 1: p = r/diag + beta p (also at the 4 neighbours, on the fly), Ap = p - dt^2/h^2
    sum k_f (p_nb - p), red[b,it,0] += p.Ap."""
    sh = cuda.shared.array((1, NT), float64)
    t = _tid()
    b = cuda.blockIdx.z
    beta = _pcg_gate(red, st, b, it, tol)
    if beta < 0.0:
        return
    first = it == 1
    N = r.shape[1]
    i, j = cuda.grid(2)
    s0 = 0.0
    if j < N and i < N:
        p = _pn(r, diag, po, b, j, i, beta, first)
        if mask[j, i] == 0:
            out = p
        else:
            cc = coef[b, j, i]
            v = 0.0
            if _in(mask, j, i + 1, N):
                v += 0.5 * (cc + coef[b, j, i + 1]) * (_pn(r, diag, po, b, j, i + 1, beta, first) - p)
            if _in(mask, j, i - 1, N):
                v += 0.5 * (cc + coef[b, j, i - 1]) * (_pn(r, diag, po, b, j, i - 1, beta, first) - p)
            if _in(mask, j + 1, i, N):
                v += 0.5 * (cc + coef[b, j + 1, i]) * (_pn(r, diag, po, b, j + 1, i, beta, first) - p)
            if _in(mask, j - 1, i, N):
                v += 0.5 * (cc + coef[b, j - 1, i]) * (_pn(r, diag, po, b, j - 1, i, beta, first) - p)
            dt = dtv[b]
            out = p - dt * dt * v * inv_h2
        pn_out[b, j, i] = p
        Ap[b, j, i] = out
        s0 = p * out
    sh[0, t] = s0
    _bred(sh, t, 1)
    if t == 0:
        cuda.atomic.add(red, (b, it, 0), sh[0, 0])


@cuda.jit
def k_pcg_k2(p, Ap, x, r, diag, red, st, it):
    """PCG iteration it, part 2: alpha = rz/pAp; x += alpha p; r -= alpha Ap; red[b,it,1] += r.r, red[b,it,2] += r.z."""
    sh = cuda.shared.array((2, NT), float64)
    t = _tid()
    b = cuda.blockIdx.z
    if st[b, 0] != 0:
        return
    alpha = red[b, it - 1, 2] / (red[b, it, 0] + 1e-300)
    i, j = cuda.grid(2)
    N1 = x.shape[1]; N2 = x.shape[2]
    s1 = 0.0; s2 = 0.0
    if j < N1 and i < N2:
        x[b, j, i] += alpha * p[b, j, i]
        rv = r[b, j, i] - alpha * Ap[b, j, i]
        r[b, j, i] = rv
        s1 = rv * rv
        s2 = rv * (rv / diag[b, j, i])
    sh[0, t] = s1; sh[1, t] = s2
    _bred(sh, t, 2)
    if t == 0:
        cuda.atomic.add(red, (b, it, 1), sh[0, 0])
        cuda.atomic.add(red, (b, it, 2), sh[1, 0])
        if cuda.blockIdx.x == 0 and cuda.blockIdx.y == 0:
            st[b, 1] = it


@cuda.jit
def k_pcg2_k2(pu, pv, Apu, Apv, xu, xv, ru, rv, du, dv, red, st, it):
    """k_pcg_k2 on the two velocity components (one vector [u; v])."""
    sh = cuda.shared.array((2, NT), float64)
    t = _tid()
    b = cuda.blockIdx.z
    if st[b, 0] != 0:
        return
    alpha = red[b, it - 1, 2] / (red[b, it, 0] + 1e-300)
    i, j = cuda.grid(2)
    N = xu.shape[1]
    s1 = 0.0; s2 = 0.0
    if j < N and i < N:
        xu[b, j, i] += alpha * pu[b, j, i]
        xv[b, j, i] += alpha * pv[b, j, i]
        a = ru[b, j, i] - alpha * Apu[b, j, i]
        c = rv[b, j, i] - alpha * Apv[b, j, i]
        ru[b, j, i] = a; rv[b, j, i] = c
        s1 = a * a + c * c
        s2 = a * (a / du[b, j, i]) + c * (c / dv[b, j, i])
    sh[0, t] = s1; sh[1, t] = s2
    _bred(sh, t, 2)
    if t == 0:
        cuda.atomic.add(red, (b, it, 1), sh[0, 0])
        cuda.atomic.add(red, (b, it, 2), sh[1, 0])
        if cuda.blockIdx.x == 0 and cuda.blockIdx.y == 0:
            st[b, 1] = it


# ============================================================================= step B finish + C
@cuda.jit
def k_finish(l, x, a_x, a_y, coef, m, s, mask, dtv, nsub, rnd, inv_h, l_out, qx_out, qy_out, m_out, fa, reds):
    """Final flux G = Ge - k (x_R - x_L)/h; l^{n+1} = l + dt s - dt div G (conservative); q^{n+1} = a - dt coef
    grad_cell(x); m^{n+1} = m + dt s - dt div(G c_f), c_f van Leer limited upwind of c = m/l; fa[b,1] += clipped milk.
    Zeroes the PCG slot 0 of the viscous solve."""
    i, j = cuda.grid(2)
    b = cuda.blockIdx.z
    N = l.shape[1]
    if rnd >= nsub[b]:
        return
    if _first_thread():
        reds[b, 0, 0] = 0.0; reds[b, 0, 1] = 0.0; reds[b, 0, 2] = 0.0
    if j >= N or i >= N:
        return
    if mask[j, i] == 0:
        l_out[b, j, i] = 1.0; qx_out[b, j, i] = 0.0; qy_out[b, j, i] = 0.0; m_out[b, j, i] = 0.0
        return
    dt = dtv[b]
    cc = coef[b, j, i]
    xc = x[b, j, i]
    dG = 0.0; dM = 0.0; gx = 0.0; gy = 0.0; nx = 0; ny = 0
    if _in(mask, j, i + 1, N):
        G = _ge_x(a_x, l, b, j, i) - dt * 0.5 * (cc + coef[b, j, i + 1]) * (x[b, j, i + 1] - xc) * inv_h
        dG += G; dM += G * _cf_x(m, l, G, mask, b, j, i, N)
        gx += (x[b, j, i + 1] - xc) * inv_h; nx += 1
    if _in(mask, j, i - 1, N):
        G = _ge_x(a_x, l, b, j, i - 1) - dt * 0.5 * (cc + coef[b, j, i - 1]) * (xc - x[b, j, i - 1]) * inv_h
        dG -= G; dM -= G * _cf_x(m, l, G, mask, b, j, i - 1, N)
        gx += (xc - x[b, j, i - 1]) * inv_h; nx += 1
    if _in(mask, j + 1, i, N):
        G = _ge_y(a_y, l, b, j, i) - dt * 0.5 * (cc + coef[b, j + 1, i]) * (x[b, j + 1, i] - xc) * inv_h
        dG += G; dM += G * _cf_y(m, l, G, mask, b, j, i, N)
        gy += (x[b, j + 1, i] - xc) * inv_h; ny += 1
    if _in(mask, j - 1, i, N):
        G = _ge_y(a_y, l, b, j - 1, i) - dt * 0.5 * (cc + coef[b, j - 1, i]) * (xc - x[b, j - 1, i]) * inv_h
        dG -= G; dM -= G * _cf_y(m, l, G, mask, b, j - 1, i, N)
        gy += (xc - x[b, j - 1, i]) * inv_h; ny += 1
    sv = s[b, j, i]
    ln = l[b, j, i] + dt * sv - dt * dG * inv_h
    l_out[b, j, i] = ln
    if nx > 0:
        gx /= nx
    if ny > 0:
        gy /= ny
    qx_out[b, j, i] = a_x[b, j, i] - dt * cc * gx
    qy_out[b, j, i] = a_y[b, j, i] - dt * cc * gy
    mn = m[b, j, i] + dt * sv - dt * dM * inv_h
    over = max(mn - ln, 0.0) + max(-mn, 0.0)
    if over > 0.0:
        cuda.atomic.add(fa, (b, 1), over)
    m_out[b, j, i] = min(max(mn, 0.0), ln)


# ============================================================================= step D: symmetric viscosity
@cuda.jit(device=True, inline=True)
def _wxf(l, mask, nu2, b, j, i):
    return nu2 * (0.5 * (l[b, j, i] + l[b, j, i + 1])) if (mask[j, i] != 0 and mask[j, i + 1] != 0) else 0.0


@cuda.jit(device=True, inline=True)
def _wyf(l, mask, nu2, b, j, i):
    return nu2 * (0.5 * (l[b, j, i] + l[b, j + 1, i])) if (mask[j, i] != 0 and mask[j + 1, i] != 0) else 0.0


@cuda.jit(device=True, inline=True)
def _wcf(l, cmask, nu, b, j, i):
    return (nu * 0.25) * (l[b, j, i] + l[b, j + 1, i] + l[b, j, i + 1] + l[b, j + 1, i + 1]) if cmask[j, i] != 0 else 0.0


@cuda.jit(device=True, inline=True)
def _u0(q, l, mask, b, j, i):
    return q[b, j, i] / l[b, j, i] if mask[j, i] != 0 else 0.0


@cuda.jit(device=True, inline=True)
def _corner(u00, u01, u10, u11, v00, v01, v10, v11, w, inv_h):
    """T = w (u_y + v_x) on the corner (j,i)|(j+1,i+1); uab = u[j+a, i+b] (kernels.sym_corner_T)."""
    return w * ((u10 + u11 - u00 - u01 + v01 + v11 - v00 - v10) * 0.5 * inv_h)


@cuda.jit
def k_sym_init(l, qx, qy, mask, cmask, geo, nbx, nby, pm, dtv, nsub, rnd, inv_h,
               wx, wy, wc, ww, xu, xv, ru, rv, du, dv, red, st):
    """Weights wx, wy = 2 nu l_f, wc = nu l_corner, ww = 2 nu l geo; X0 = q/l (0 outside), b = l X0, Jacobi diagonal
    of l + dt K (1 outside), r0 = b - (l + dt K) X0; red[b,0,1] += b.b, red[b,0,2] += r0.z0; zeroes slots 1.."""
    sh = cuda.shared.array((2, NT), float64)
    t = _tid()
    b = cuda.blockIdx.z
    N = l.shape[1]
    if rnd >= nsub[b]:
        if _first_thread():
            st[b, 0] = 1; st[b, 1] = 0
        return
    i, j = cuda.grid(2)
    nb = cuda.gridDim.x * cuda.gridDim.y * NT
    k = (cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.x) * NT + t
    nslot = red.shape[1] * 3
    while k < nslot:
        sl = k // 3
        if sl > 0:
            red[b, sl, k - sl * 3] = 0.0
        k += nb
    if _first_thread():
        st[b, 0] = 0; st[b, 1] = 0
    s0 = 0.0; s1 = 0.0
    if j < N and i < N:
        nu = pm[b, PM_NU]; nu2 = 2.0 * nu
        dt = dtv[b]
        ih2 = inv_h * inv_h
        if i < N - 1:
            wx[b, j, i] = _wxf(l, mask, nu2, b, j, i)
        if j < N - 1:
            wy[b, j, i] = _wyf(l, mask, nu2, b, j, i)
        if i < N - 1 and j < N - 1:
            wc[b, j, i] = _wcf(l, cmask, nu, b, j, i)
        lc = l[b, j, i]
        w = (nu2 * lc) * geo[j, i]
        ww[b, j, i] = w
        if mask[j, i] == 0:
            xu[b, j, i] = 0.0; xv[b, j, i] = 0.0; ru[b, j, i] = 0.0; rv[b, j, i] = 0.0
            du[b, j, i] = 1.0; dv[b, j, i] = 1.0
        else:
            u = qx[b, j, i] / lc; v = qy[b, j, i] / lc
            bu = lc * u; bv = lc * v
            ex = 0.0; ey = 0.0; c4 = 0.0
            ku = 0.0; kv = 0.0
            if i < N - 1:
                wf = _wxf(l, mask, nu2, b, j, i); ex += wf; ku -= wf * (_u0(qx, l, mask, b, j, i + 1) - u)
            if i > 0:
                wf = _wxf(l, mask, nu2, b, j, i - 1); ex += wf; ku -= wf * (_u0(qx, l, mask, b, j, i - 1) - u)
            if j < N - 1:
                wf = _wyf(l, mask, nu2, b, j, i); ey += wf; kv -= wf * (_u0(qy, l, mask, b, j + 1, i) - v)
            if j > 0:
                wf = _wyf(l, mask, nu2, b, j - 1, i); ey += wf; kv -= wf * (_u0(qy, l, mask, b, j - 1, i) - v)
            ku *= ih2; kv *= ih2
            c = 0.0; d = 0.0
            for cj in range(2):
                for ci in range(2):
                    jc = j - 1 + cj; ic = i - 1 + ci
                    if jc >= 0 and ic >= 0 and jc < N - 1 and ic < N - 1:
                        wcv = _wcf(l, cmask, nu, b, jc, ic)
                        c4 += wcv
                        T = 0.0
                        if wcv != 0.0:
                            T = _corner(_u0(qx, l, mask, b, jc, ic), _u0(qx, l, mask, b, jc, ic + 1),
                                        _u0(qx, l, mask, b, jc + 1, ic), _u0(qx, l, mask, b, jc + 1, ic + 1),
                                        _u0(qy, l, mask, b, jc, ic), _u0(qy, l, mask, b, jc, ic + 1),
                                        _u0(qy, l, mask, b, jc + 1, ic), _u0(qy, l, mask, b, jc + 1, ic + 1), wcv, inv_h)
                        # corner (j,i): c -= T, d -= T; (j,i-1): c -= T, d += T; (j-1,i): c += T, d -= T; (j-1,i-1): c += T, d += T
                        if cj == 1:
                            c -= T
                        else:
                            c += T
                        if ci == 1:
                            d -= T
                        else:
                            d += T
            ku += 0.5 * inv_h * c; kv += 0.5 * inv_h * d
            nx_ = nbx[j, i]; ny_ = nby[j, i]
            if w != 0.0:
                un = w * (u * nx_ + v * ny_)
                ku += un * nx_; kv += un * ny_
            a = bu - (lc * u + dt * ku)
            e = bv - (lc * v + dt * kv)
            dgu = lc + dt * (ex * ih2 + 0.25 * ih2 * c4 + w * nx_ * nx_)
            dgv = lc + dt * (ey * ih2 + 0.25 * ih2 * c4 + w * ny_ * ny_)
            xu[b, j, i] = u; xv[b, j, i] = v; ru[b, j, i] = a; rv[b, j, i] = e
            du[b, j, i] = dgu; dv[b, j, i] = dgv
            s0 = bu * bu + bv * bv
            s1 = a * (a / dgu) + e * (e / dgv)
    sh[0, t] = s0; sh[1, t] = s1
    _bred(sh, t, 2)
    if t == 0:
        cuda.atomic.add(red, (b, 0, 1), sh[0, 0])
        cuda.atomic.add(red, (b, 0, 2), sh[1, 0])


@cuda.jit
def k_sym_k1(ru, rv, du, dv, pu_o, pv_o, pu, pv, l, mask, wx, wy, wc, ww, nbx, nby, dtv, inv_h, red, st, it, tol, Apu, Apv):
    """PCG iteration it of the viscous solve, part 1: p = r/diag + beta p on the 3x3 neighbourhood (on the fly),
    Ap = l p + dt K p (kernels.sym_visc_apply, corner shear T on the 4 corners), red[b,it,0] += p.Ap."""
    sh = cuda.shared.array((1, NT), float64)
    t = _tid()
    b = cuda.blockIdx.z
    beta = _pcg_gate(red, st, b, it, tol)
    if beta < 0.0:
        return
    first = it == 1
    N = l.shape[1]
    i, j = cuda.grid(2)
    s0 = 0.0
    if j < N and i < N:
        u = _pn(ru, du, pu_o, b, j, i, beta, first)
        v = _pn(rv, dv, pv_o, b, j, i, beta, first)
        pu[b, j, i] = u; pv[b, j, i] = v
        if mask[j, i] == 0:
            ou = u; ov = v
        else:
            ih2 = inv_h * inv_h
            ku = 0.0; kv = 0.0
            if i < N - 1:
                ku -= wx[b, j, i] * (_pn(ru, du, pu_o, b, j, i + 1, beta, first) - u)
            if i > 0:
                ku -= wx[b, j, i - 1] * (_pn(ru, du, pu_o, b, j, i - 1, beta, first) - u)
            if j < N - 1:
                kv -= wy[b, j, i] * (_pn(rv, dv, pv_o, b, j + 1, i, beta, first) - v)
            if j > 0:
                kv -= wy[b, j - 1, i] * (_pn(rv, dv, pv_o, b, j - 1, i, beta, first) - v)
            ku *= ih2; kv *= ih2
            c = 0.0; d = 0.0
            for cj in range(2):
                for ci in range(2):
                    jc = j - 1 + cj; ic = i - 1 + ci
                    if jc >= 0 and ic >= 0 and jc < N - 1 and ic < N - 1:
                        wcv = wc[b, jc, ic]
                        T = 0.0
                        if wcv != 0.0:
                            T = _corner(_pn(ru, du, pu_o, b, jc, ic, beta, first), _pn(ru, du, pu_o, b, jc, ic + 1, beta, first),
                                        _pn(ru, du, pu_o, b, jc + 1, ic, beta, first), _pn(ru, du, pu_o, b, jc + 1, ic + 1, beta, first),
                                        _pn(rv, dv, pv_o, b, jc, ic, beta, first), _pn(rv, dv, pv_o, b, jc, ic + 1, beta, first),
                                        _pn(rv, dv, pv_o, b, jc + 1, ic, beta, first), _pn(rv, dv, pv_o, b, jc + 1, ic + 1, beta, first),
                                        wcv, inv_h)
                        if cj == 1:
                            c -= T
                        else:
                            c += T
                        if ci == 1:
                            d -= T
                        else:
                            d += T
            ku += 0.5 * inv_h * c; kv += 0.5 * inv_h * d
            w = ww[b, j, i]
            if w != 0.0:
                nx_ = nbx[j, i]; ny_ = nby[j, i]
                un = w * (u * nx_ + v * ny_)
                ku += un * nx_; kv += un * ny_
            lc = l[b, j, i]
            dt = dtv[b]
            ou = lc * u + dt * ku
            ov = lc * v + dt * kv
        Apu[b, j, i] = ou; Apv[b, j, i] = ov
        s0 = u * ou + v * ov
    sh[0, t] = s0
    _bred(sh, t, 1)
    if t == 0:
        cuda.atomic.add(red, (b, it, 0), sh[0, 0])


# ============================================================================= step D: componentwise viscosity, mixing, check
@cuda.jit
def k_viscosity(l, qx, qy, mask, pm, dtv, nsub, rnd, nv, sub, inv_h2, qx_out, qy_out):
    """q += dt/nv div(nu l_f grad u), u = q/l, explicit; l frozen; members with sub >= nv[b] copy."""
    i, j = cuda.grid(2)
    b = cuda.blockIdx.z
    N = l.shape[1]
    if j >= N or i >= N or rnd >= nsub[b]:
        return
    if sub >= nv[b]:
        qx_out[b, j, i] = qx[b, j, i]; qy_out[b, j, i] = qy[b, j, i]
        return
    if mask[j, i] == 0:
        qx_out[b, j, i] = 0.0; qy_out[b, j, i] = 0.0
        return
    nu = pm[b, PM_NU]
    dt = dtv[b] / nv[b]
    lc = l[b, j, i]; uxc = qx[b, j, i] / lc; uyc = qy[b, j, i] / lc
    vx = 0.0; vy = 0.0
    if _in(mask, j, i + 1, N):
        k = nu * 0.5 * (lc + l[b, j, i + 1]); vx += k * (qx[b, j, i + 1] / l[b, j, i + 1] - uxc); vy += k * (qy[b, j, i + 1] / l[b, j, i + 1] - uyc)
    if _in(mask, j, i - 1, N):
        k = nu * 0.5 * (lc + l[b, j, i - 1]); vx += k * (qx[b, j, i - 1] / l[b, j, i - 1] - uxc); vy += k * (qy[b, j, i - 1] / l[b, j, i - 1] - uyc)
    if _in(mask, j + 1, i, N):
        k = nu * 0.5 * (lc + l[b, j + 1, i]); vx += k * (qx[b, j + 1, i] / l[b, j + 1, i] - uxc); vy += k * (qy[b, j + 1, i] / l[b, j + 1, i] - uyc)
    if _in(mask, j - 1, i, N):
        k = nu * 0.5 * (lc + l[b, j - 1, i]); vx += k * (qx[b, j - 1, i] / l[b, j - 1, i] - uxc); vy += k * (qy[b, j - 1, i] / l[b, j - 1, i] - uyc)
    qx_out[b, j, i] = qx[b, j, i] + dt * vx * inv_h2
    qy_out[b, j, i] = qy[b, j, i] + dt * vy * inv_h2


@cuda.jit(device=True, inline=True)
def _mix(l, m, mask, D, dt, inv_h2, b, j, i, N):
    lc = l[b, j, i]; cc = m[b, j, i] / lc
    v = 0.0
    if _in(mask, j, i + 1, N):
        v += D * 0.5 * (lc + l[b, j, i + 1]) * (m[b, j, i + 1] / l[b, j, i + 1] - cc)
    if _in(mask, j, i - 1, N):
        v += D * 0.5 * (lc + l[b, j, i - 1]) * (m[b, j, i - 1] / l[b, j, i - 1] - cc)
    if _in(mask, j + 1, i, N):
        v += D * 0.5 * (lc + l[b, j + 1, i]) * (m[b, j + 1, i] / l[b, j + 1, i] - cc)
    if _in(mask, j - 1, i, N):
        v += D * 0.5 * (lc + l[b, j - 1, i]) * (m[b, j - 1, i] / l[b, j - 1, i] - cc)
    mn = m[b, j, i] + dt * v * inv_h2
    return min(max(mn, 0.0), lc)


@cuda.jit
def k_mixing(l, m, mask, pm, dtv, nsub, rnd, nmix, sub, inv_h2, m_out):
    """m += dt/nm div(D l_f grad c), explicit, clipped to [0, l]; members with sub >= nmix[b] copy."""
    i, j = cuda.grid(2)
    b = cuda.blockIdx.z
    N = l.shape[1]
    if j >= N or i >= N or rnd >= nsub[b]:
        return
    if sub >= nmix[b] or mask[j, i] == 0:
        m_out[b, j, i] = m[b, j, i] if sub >= nmix[b] else 0.0
        return
    m_out[b, j, i] = _mix(l, m, mask, pm[b, PM_D], dtv[b] / nmix[b], inv_h2, b, j, i, N)


@cuda.jit
def k_final(l, qx, qy, m, l_n, qsx, qsy, xu, xv, msrc, mask, pm, dtv, nsub, rnd, nmix, sym, mixing, inv_h2, l_floor,
            qx_n, qy_n, m_out, fa):
    """End of the sub-step: q = l X (symmetric viscosity) or the explicit result, the last mixing sub-cycle, the
    finite / positive-layer check (fa[b,0] += bad cells).  Members idle in this round copy the state through."""
    i, j = cuda.grid(2)
    b = cuda.blockIdx.z
    N = l.shape[1]
    if j >= N or i >= N:
        return
    if rnd >= nsub[b]:
        l_n[b, j, i] = l[b, j, i]; qx_n[b, j, i] = qx[b, j, i]; qy_n[b, j, i] = qy[b, j, i]; m_out[b, j, i] = m[b, j, i]
        return
    ln = l_n[b, j, i]
    inside = mask[j, i] != 0
    if sym:
        if inside:
            ax = ln * xu[b, j, i]; ay = ln * xv[b, j, i]
        else:
            ax = 0.0; ay = 0.0
    else:
        ax = qsx[b, j, i]; ay = qsy[b, j, i]
    qx_n[b, j, i] = ax; qy_n[b, j, i] = ay
    if mixing:
        if inside:
            mn = _mix(l_n, msrc, mask, pm[b, PM_D], dtv[b] / nmix[b], inv_h2, b, j, i, N)
        else:
            mn = 0.0
        m_out[b, j, i] = mn
    else:
        mn = msrc[b, j, i]
        m_out[b, j, i] = mn
    bad = (math.isnan(ln) or math.isinf(ln) or math.isnan(ax) or math.isinf(ax) or math.isnan(ay) or math.isinf(ay)
           or math.isnan(mn) or math.isinf(mn))
    if inside and ln <= l_floor:
        bad = True
    if bad:
        cuda.atomic.add(fa, (b, 0), 1.0)


# ============================================================================= diagnostics
@cuda.jit
def k_concentration(m, l, mask, c):
    i, j = cuda.grid(2)
    b = cuda.blockIdx.z
    N = l.shape[1]
    if j >= N or i >= N:
        return
    c[b, j, i] = m[b, j, i] / l[b, j, i] if mask[j, i] != 0 else 0.0


@cuda.jit
def k_energy(l, qx, qy, mask, pm, out):
    """out[b,0] += kinetic 0.5 |q|^2/l, out[b,1] += 0.5 cp^2 l^2 (sum over cup cells)."""
    sh = cuda.shared.array((2, NT), float64)
    t = _tid()
    b = cuda.blockIdx.z
    i, j = cuda.grid(2)
    N = l.shape[1]
    a = 0.0; c = 0.0
    if j < N and i < N and mask[j, i] != 0:
        lc = l[b, j, i]
        a = 0.5 * (qx[b, j, i] * qx[b, j, i] + qy[b, j, i] * qy[b, j, i]) / lc
        c = 0.5 * pm[b, PM_CP2] * lc * lc
    sh[0, t] = a; sh[1, t] = c
    _bred(sh, t, 2)
    if t == 0:
        cuda.atomic.add(out, (b, 0), sh[0, 0])
        cuda.atomic.add(out, (b, 1), sh[1, 0])


# ============================================================================= launcher
class _Launcher:
    """Counts launches; optional direct path into the compiled kernel (skips the per-call argument typing of the
    dispatcher, the dominant host cost of a small launch).  LATTE_CUDA_FASTLAUNCH=0 turns it off; any failure falls
    back to the dispatcher for the rest of the process."""
    fast_ok = os.environ.get("LATTE_CUDA_FASTLAUNCH", "1") == "1" and not SIM

    def __init__(self):
        self.n = 0
        self.cache = {}

    def __call__(self, kern, grid, block, *args):
        self.n += 1
        if _Launcher.fast_ok:
            k = self.cache.get(kern)
            if k is not None:
                try:
                    k.launch(args, tuple(grid) + (1,) * (3 - len(grid)), tuple(block) + (1,) * (3 - len(block)), 0, 0)
                    return
                except Exception:
                    _Launcher.fast_ok = False
            else:
                kern[grid, block](*args)
                try:
                    tys = tuple(kern.typeof_pyval(a) for a in args)
                    self.cache[kern] = kern.overloads[tys]
                except Exception:
                    self.cache[kern] = None
                    _Launcher.fast_ok = False
                return
        kern[grid, block](*args)


# ============================================================================= host side
def _stats0():
    return dict(steps=0, frames=0, cg_mass=0, cg_visc=0, cg_mix=0, clip_mass=0.0, max_substeps=0, max_u=0.0, retries=0,
                kernel_clamp=0, offcup_frames=0, min_Z=1.0, launches=0, host_syncs=0)


class BatchCudaSolver:
    """B independent simulations on one grid (same Numerics, same mask; Params may differ in cp, beta, nu, D and the
    inlet closure constants, the switches visc / offcup / return_law and the presence of nu > 0, D > 0 must agree).
    advance_frames(inlets, frame_dts): every member with an Inlet (None = idle) advances one control frame of its own
    length; each member keeps its own t, ledger, statistics and retry logic (a member reproduces a B = 1 run)."""

    def __init__(self, params, num: Numerics, B=None, dtype=np.float64, poll_every=1):
        if isinstance(params, Params):
            params = [params] * (B or 1)
        self.Ps = list(params)
        B = self.B = len(self.Ps)
        self.P = self.Ps[0]
        self.num = num
        self.g = Grid(num.N, num.cup_radius)
        g = self.g; N = g.N
        for P in self.Ps:
            if P.visc not in ("symmetric", "componentwise"):
                raise ValueError(f"Params.visc must be 'symmetric' or 'componentwise', got {P.visc!r}")
            if P.offcup not in ("account", "renormalise"):
                raise ValueError(f"Params.offcup must be 'account' or 'renormalise', got {P.offcup!r}")
            if P.closure != "chi":
                raise ValueError("the CUDA back-end implements the 'chi' closure only")
        keys = lambda P: (P.visc, P.offcup, P.nu > 0, P.D > 0)
        if len({keys(P) for P in self.Ps}) != 1:
            raise ValueError("batch members must share visc, offcup and the presence of nu > 0, D > 0")
        self.sym = self.P.visc == "symmetric" and self.P.nu > 0
        self.mixing = self.P.D > 0
        self.dtype = np.dtype(dtype)
        self.fp32 = self.dtype == np.float32
        self.cg_tol = max(num.cg_tol, 1e-6) if self.fp32 else num.cg_tol
        self.visc_tol = max(num.visc_tol, 1e-6) if self.fp32 else num.visc_tol
        self.poll_every = max(1, int(poll_every))
        self.t = np.zeros(B); self.deposited = np.zeros(B); self.spilled = np.zeros(B); self.last_Z = np.ones(B)
        self.stats = [_stats0() for _ in range(B)]
        self.l_floor = 1e-3
        self.L = _Launcher()
        self.grid3 = ((N + TPB[0] - 1) // TPB[0], (N + TPB[1] - 1) // TPB[1], B)
        self.area = g.area
        dt_ = self.dtype
        z = np.zeros((B, N, N), dt_)
        self.mask = cuda.to_device(g.mask.astype(np.uint8))
        self.l = cuda.to_device(np.ones((B, N, N), dt_))
        for nm in ("qx", "qy", "m"):
            setattr(self, nm, cuda.to_device(z))
        names = ["l_s", "qx_s", "qy_s", "m_s", "a_x", "a_y", "coef", "rhs", "diag", "x", "r", "p0", "p1", "Ap",
                 "l_n", "qx_n", "qy_n", "m_n", "m_t", "qx_t", "qy_t", "s", "Lam", "vsx", "vsy", "Kraw", "c"]
        if self.sym:
            names += ["wx", "wy", "wc", "ww", "xu", "xv", "ru", "rv", "du", "dv", "pu0", "pv0", "pu1", "pv1", "Apu", "Apv"]
        for nm in names:
            setattr(self, nm, cuda.to_device(z))
        self.redm = cuda.to_device(np.zeros((B, num.cg_maxiter + 2, 3)))
        self.reds = cuda.to_device(np.zeros((B, num.cg_maxiter + 2, 3)))
        self.st = cuda.to_device(np.zeros((B, 2), np.int32))
        self.fa = cuda.to_device(np.zeros((B, 2)))
        self.eacc = cuda.to_device(np.zeros((B, 2)))
        pm = np.array([[P.cp ** 2, P.beta, P.nu, P.D] for P in self.Ps], np.float64)
        self.pm = cuda.to_device(pm)
        self.ip_h = np.zeros((B, IP_NCOL))
        self.ip = cuda.to_device(self.ip_h)
        self.nsub_d = cuda.to_device(np.zeros(B, np.int32))
        self.dt_d = cuda.to_device(np.zeros(B))
        self.nv_d = cuda.to_device(np.ones(B, np.int32))
        self.nm_d = cuda.to_device(np.ones(B, np.int32))
        if self.sym:
            cm, geo, nbx, nby = g.sym_geometry()
            self.cmask = cuda.to_device(cm.astype(np.uint8)); self.geo = cuda.to_device(np.ascontiguousarray(geo))
            self.nbx = cuda.to_device(nbx); self.nby = cuda.to_device(nby)
        self._pred = dict(mass=1, visc=1)
        self._inlet_sum = np.zeros(B); self._spill_rate = np.zeros(B); self._chi = np.zeros(B)
        try:
            self._cbuf = cuda.pinned_array((N, N), dt_)
        except Exception:
            self._cbuf = np.empty((N, N), dt_)

    # ------------------------------------------------------------------ data access
    def concentration(self, b=0):
        self.L(k_concentration, self.grid3, TPB, self.m, self.l, self.mask, self.c)
        self.c[b].copy_to_host(self._cbuf)
        return np.where(self.g.mask, np.asarray(self._cbuf, np.float64), np.nan)

    def concentrations(self, bs):
        """{b: c (N, N) with NaN outside the cup} for several members, one launch (snapshots of a batch)."""
        self.L(k_concentration, self.grid3, TPB, self.m, self.l, self.mask, self.c)
        out = {}
        for b in bs:
            self.c[b].copy_to_host(self._cbuf)
            out[b] = np.where(self.g.mask, np.asarray(self._cbuf, np.float64), np.nan)
        return out

    def host_state(self, b=0):
        return dict(l=self.l[b].copy_to_host().astype(np.float64), qx=self.qx[b].copy_to_host().astype(np.float64),
                    qy=self.qy[b].copy_to_host().astype(np.float64), m=self.m[b].copy_to_host().astype(np.float64))

    def ledger(self, b=0):
        st = self.host_state(b); g = self.g
        L = float(st["l"][g.mask].sum() * g.area); M = float(st["m"][g.mask].sum() * g.area)
        dep = float(self.deposited[b])
        return dict(t=float(self.t[b]), layer=L, milk=M, brown=L - M, deposited=dep, spilled=float(self.spilled[b]),
                    layer_error=L - (g.n_cells * g.area + dep), milk_error=M - dep)

    def energy(self, b=0):
        self.eacc.copy_to_device(np.zeros((self.B, 2)))
        self.L(k_energy, self.grid3, TPB, self.l, self.qx, self.qy, self.mask, self.pm, self.eacc)
        e = self.eacc.copy_to_host()[b]
        return float(e[0] * self.g.area), float(e[1] * self.g.area)

    # ------------------------------------------------------------------ inlet table (host scalars)
    def _inlet_row(self, b, inlet):
        """Scalars of the inlet of member b (solver.Solver.inlet_fields / kernel / kernel_fraction); returns chi."""
        P, g, num = self.Ps[b], self.g, self.num
        row = self.ip_h[b]
        row[:] = 0.0
        if inlet is None or not inlet.active or inlet.S_eff <= 0:
            return 0.0
        if inlet.r1 < 1.5 * g.h or inlet.r2 < 1.5 * g.h:
            self.stats[b]["kernel_clamp"] += 1
        r1, r2, phi = max(inlet.r1, 1.5 * g.h), max(inlet.r2, 1.5 * g.h), inlet.phi
        x0, y0 = inlet.x_hit
        q = min(max(num.kernel_quadrature, int(math.ceil(4 * g.h / min(r1, r2)))), 12)
        R = max(r1, r2)
        i0 = int(math.floor((x0 - R + 0.5) / g.h)) - 1; i1 = int(math.floor((x0 + R + 0.5) / g.h)) + 2
        j0 = int(math.floor((y0 - R + 0.5) / g.h)) - 1; j1 = int(math.floor((y0 + R + 0.5) / g.h)) + 2
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
        row[IP_ACT] = 1.0
        row[IP_X0], row[IP_Y0], row[IP_R1], row[IP_R2] = x0, y0, r1, r2
        row[IP_C], row[IP_S], row[IP_Q] = math.cos(phi), math.sin(phi), q
        row[IP_I0], row[IP_I1], row[IP_J0], row[IP_J1] = i0, i1, j0, j1
        row[IP_CHIS] = chi * inlet.S_eff
        row[IP_LAMS] = P.kappa_c * (1.0 - chi) * inlet.S_eff
        row[IP_RDEP] = Rdep
        row[IP_KTX], row[IP_KTY] = P.kappa_t * inlet.u_in[0], P.kappa_t * inlet.u_in[1]
        row[IP_RAD] = rad
        row[IP_ACC] = 1.0 if P.offcup == "account" else 0.0
        return chi

    # ------------------------------------------------------------------ PCG drivers
    def _pcg(self, kind, k1, k1args, k2, k2args, red, tol, active):
        """Launch PCG iterations (k1, k2 per iteration) until the device reports every member done; poll the [B,2]
        status word only from the predicted iteration count on.  Returns the per-member iteration counts."""
        L, G3 = self.L, self.grid3
        pred = self._pred[kind]
        st = None
        for it in range(1, self.num.cg_maxiter + 1):
            L(k1, G3, TPB, *k1args(it), red, self.st, it, tol, *k1args.tail(it))
            if it > pred and (it - pred - 1) % self.poll_every == 0:
                st = self.st.copy_to_host()
                self._syncs += 1
                if st[active, 0].all():
                    break
            L(k2, G3, TPB, *k2args(it), red, self.st, it)
        else:
            st = self.st.copy_to_host()
            self._syncs += 1
        its = st[:, 1]
        self._pred[kind] = max(1, int(its[active].max()) if active.any() else 1)
        return its

    # ------------------------------------------------------------------ one round of sub-steps
    def _round(self, rnd, nsub_h, dt_h, active):
        L, G3 = self.L, self.grid3
        g = self.g; h = g.h; inv_h = 1.0 / h; inv_h2 = inv_h * inv_h
        rnd = int(rnd)
        L(k_mom_prep, G3, TPB, self.l, self.qx, self.qy, self.mask, self.s, self.Lam, self.vsx, self.vsy, self.pm,
          self.dt_d, self.nsub_d, rnd, inv_h, self.a_x, self.a_y, self.coef, self.redm)
        L(k_mass_init, G3, TPB, self.l, self.a_x, self.a_y, self.coef, self.s, self.mask, self.dt_d, self.nsub_d, rnd,
          inv_h, self.rhs, self.diag, self.x, self.r, self.redm, self.st)
        P_ = (self.p0, self.p1)
        k1a = _Args(lambda it: (self.r, self.diag, P_[it % 2], P_[(it + 1) % 2], self.coef, self.mask, self.dt_d, inv_h2),
                    lambda it: (self.Ap,))
        k2a = _Args(lambda it: (P_[(it + 1) % 2], self.Ap, self.x, self.r, self.diag))
        its = self._pcg("mass", k_mass_k1, k1a, k_pcg_k2, k2a, self.redm, float(self.cg_tol), active)
        for b in np.nonzero(active)[0]:
            self.stats[b]["cg_mass"] += int(its[b])
        L(k_finish, G3, TPB, self.l, self.x, self.a_x, self.a_y, self.coef, self.m, self.s, self.mask, self.dt_d,
          self.nsub_d, rnd, inv_h, self.l_n, self.qx_n, self.qy_n, self.m_n, self.fa, self.reds)
        qsx, qsy = self.qx_n, self.qy_n
        if self.sym:
            L(k_sym_init, G3, TPB, self.l_n, self.qx_n, self.qy_n, self.mask, self.cmask, self.geo, self.nbx, self.nby,
              self.pm, self.dt_d, self.nsub_d, rnd, inv_h, self.wx, self.wy, self.wc, self.ww, self.xu, self.xv,
              self.ru, self.rv, self.du, self.dv, self.reds, self.st)
            PU, PV = (self.pu0, self.pu1), (self.pv0, self.pv1)
            k1a = _Args(lambda it: (self.ru, self.rv, self.du, self.dv, PU[it % 2], PV[it % 2], PU[(it + 1) % 2],
                                    PV[(it + 1) % 2], self.l_n, self.mask, self.wx, self.wy, self.wc, self.ww, self.nbx,
                                    self.nby, self.dt_d, inv_h),
                        lambda it: (self.Apu, self.Apv))
            k2a = _Args(lambda it: (PU[(it + 1) % 2], PV[(it + 1) % 2], self.Apu, self.Apv, self.xu, self.xv, self.ru,
                                    self.rv, self.du, self.dv))
            its = self._pcg("visc", k_sym_k1, k1a, k_pcg2_k2, k2a, self.reds, float(self.visc_tol), active)
            for b in np.nonzero(active)[0]:
                self.stats[b]["cg_visc"] += int(its[b])
        elif self.P.nu > 0:
            nv = self._nv
            src = (self.qx_n, self.qy_n); dst = (self.qx_t, self.qy_t)
            for sub in range(int(nv.max())):
                L(k_viscosity, G3, TPB, self.l_n, src[0], src[1], self.mask, self.pm, self.dt_d, self.nsub_d, rnd,
                  self.nv_d, sub, inv_h2, dst[0], dst[1])
                src, dst = dst, src
            qsx, qsy = src
        msrc = self.m_n
        if self.mixing:
            nm = self._nm
            src, dst = self.m_n, self.m_t
            for sub in range(int(nm.max()) - 1):
                L(k_mixing, G3, TPB, self.l_n, src, self.mask, self.pm, self.dt_d, self.nsub_d, rnd, self.nm_d, sub,
                  inv_h2, dst)
                src, dst = dst, src
            msrc = src
            m_out = dst
        else:
            m_out = self.m_n
        L(k_final, G3, TPB, self.l, self.qx, self.qy, self.m, self.l_n, qsx, qsy,
          self.xu if self.sym else self.x, self.xv if self.sym else self.x, msrc, self.mask, self.pm, self.dt_d,
          self.nsub_d, rnd, self.nm_d, bool(self.sym), bool(self.mixing), inv_h2, float(self.l_floor),
          self.qx_n, self.qy_n, m_out, self.fa)
        # swap: state <- (l_n, qx_n, qy_n, m_out) for every member (idle ones were copied through)
        self.l, self.l_n = self.l_n, self.l
        self.qx, self.qx_n = self.qx_n, self.qx
        self.qy, self.qy_n = self.qy_n, self.qy
        if m_out is self.m_n:
            self.m, self.m_n = self.m_n, self.m
        else:
            self.m, self.m_t = self.m_t, self.m

    # ------------------------------------------------------------------ one control frame for every member
    def advance_frames(self, inlets, frame_dts=None):
        """inlets: list of B (Inlet or None = member idle); frame_dts: list of B (None = num.frame_dt).
        Returns a list of (chi, nsub) (None for idle members)."""
        B, g, num = self.B, self.g, self.num
        L, G3 = self.L, self.grid3
        self._syncs = 0
        n0 = L.n
        fdt = np.zeros(B)
        part = np.zeros(B, bool)
        chis = np.zeros(B)
        for b in range(B):
            inl = inlets[b]
            if inl is None:
                self.ip_h[b, :] = 0.0
                continue
            part[b] = True
            fd = (frame_dts[b] if frame_dts is not None else None) or num.frame_dt
            if inl.active and inl.S_eff > 0 and inl.scan_speed > 0:
                lim = num.scan_safety * min(g.h, inl.r1, inl.r2) / inl.scan_speed
                fd = min(fd, max(lim, 1e-4))
            fdt[b] = fd
            chis[b] = self._inlet_row(b, inl)
        if not part.any():
            return [None] * B
        nsub = part.astype(np.int32)                 # marks the participating members for the frame kernels
        self.nsub_d.copy_to_device(nsub)
        self.ip.copy_to_device(self.ip_h)
        act = self.ip_h[:, IP_ACT] != 0
        if act.any():
            bw = int(max(self.ip_h[b, IP_I1] - self.ip_h[b, IP_I0] for b in np.nonzero(act)[0]))
            bh = int(max(self.ip_h[b, IP_J1] - self.ip_h[b, IP_J0] for b in np.nonzero(act)[0]))
            L(k_inlet_K, ((bw + TPB[0] - 1) // TPB[0], (bh + TPB[1] - 1) // TPB[1], B), TPB, self.mask, g.h, self.ip,
              self.Kraw)
        L(k_inlet_fields, G3, TPB, self.Kraw, self.mask, g.h, g.area, self.ip, self.nsub_d, self.l, self.qx, self.qy,
          self.m, self.s, self.Lam, self.vsx, self.vsy, self.l_s, self.qx_s, self.qy_s, self.m_s)
        ipr = self.ip.copy_to_host()
        self._syncs += 1
        nsub = np.zeros(B, np.int32)
        for b in np.nonzero(part)[0]:
            st = self.stats[b]; P = self.Ps[b]
            self._inlet_sum[b] = 0.0; self._spill_rate[b] = 0.0
            if act[b]:
                tot, zin, zout = ipr[b, IP_ZTOT], ipr[b, IP_ZIN], ipr[b, IP_ZOUT]
                Zr = 0.0 if tot <= 0 else (1.0 if zout == 0 else float(zin / tot))
                self.last_Z[b] = Zr
                if Zr < 1.0:
                    st["offcup_frames"] += 1
                    st["min_Z"] = min(st["min_Z"], Zr)
                S = inlets[b].S_eff
                if P.offcup == "account":
                    self._spill_rate[b] = (1.0 - Zr) * S
                    self._inlet_sum[b] = chis[b] * S * Zr if Zr > 0 else 0.0
                else:
                    self._inlet_sum[b] = chis[b] * S if zin > 0 else 0.0
            umax = max(float(ipr[b, IP_UMAX]), float(ipr[b, IP_VSMAX]))
            st["max_u"] = max(st["max_u"], umax)
            dt_cfl = num.cfl * g.h / (2.0 * max(umax, 1e-9))
            nsub[b] = max(1, min(int(math.ceil(fdt[b] / dt_cfl)), num.max_substeps))
        run = nsub.copy()
        final = np.zeros(B, np.int32)
        for attempt in range(8):
            dts = np.where(run > 0, fdt / np.maximum(run, 1), 0.0)
            self.nsub_d.copy_to_device(run)
            self.dt_d.copy_to_device(dts)
            if self.P.nu > 0 and not self.sym:
                self._nv = np.maximum(1, np.ceil(np.array([P.nu for P in self.Ps]) * dts / (g.h * g.h)
                                                 / num.explicit_visc_limit)).astype(np.int32)
                self.nv_d.copy_to_device(self._nv)
            if self.mixing:
                self._nm = np.maximum(1, np.ceil(np.array([P.D for P in self.Ps]) * dts / (g.h * g.h) / 0.2)).astype(np.int32)
                self.nm_d.copy_to_device(self._nm)
            self.fa.copy_to_device(np.zeros((B, 2)))
            if attempt > 0:
                L(k_restore, G3, TPB, self.nsub_d, self.l_s, self.qx_s, self.qy_s, self.m_s, self.l, self.qx, self.qy,
                  self.m)
            active_all = run > 0
            for rnd in range(int(run.max())):
                self._round(rnd, run, dts, (run > rnd) & active_all)
            fa = self.fa.copy_to_host()
            self._syncs += 1
            failed = np.zeros(B, bool)
            for b in np.nonzero(active_all)[0]:
                self.stats[b]["clip_mass"] += float(fa[b, 1]) * g.area
                if fa[b, 0] > 0:
                    failed[b] = True
                else:
                    final[b] = run[b]
                    for _ in range(int(run[b])):
                        self.deposited[b] += self._inlet_sum[b] * dts[b]
                    self.stats[b]["steps"] += int(run[b])
            if not failed.any():
                break
            for b in np.nonzero(failed)[0]:
                self.stats[b]["retries"] += 1
            run = np.where(failed, run * 2, 0).astype(np.int32)
        else:
            raise RuntimeError("sub-step failed repeatedly")
        out = []
        for b in range(B):
            if not part[b]:
                out.append(None)
                continue
            st = self.stats[b]
            self.spilled[b] += self._spill_rate[b] * fdt[b]
            self.t[b] += fdt[b]
            st["frames"] += 1
            st["max_substeps"] = max(st["max_substeps"], int(final[b]))
            st["launches"] += L.n - n0
            st["host_syncs"] += self._syncs
            out.append((float(chis[b]), int(final[b])))
        return out


class _Args:
    """Per-iteration argument builder for _pcg: head(it) before the shared (red, st, it[, tol]) block, tail(it) after."""

    def __init__(self, head, tail=None):
        self.head, self._tail = head, tail

    def __call__(self, it):
        return self.head(it)

    def tail(self, it):
        return self._tail(it) if self._tail else ()


class CudaSolver:
    """Drop-in replacement for solver.Solver (the attributes run.py and the tests use), one member of BatchCudaSolver."""

    def __init__(self, params: Params, num: Numerics, dtype=np.float64, **kw):
        self.batch = BatchCudaSolver([params], num, dtype=dtype, **kw)
        self.P, self.num, self.g = params, num, self.batch.g
        self.l_floor = self.batch.l_floor

    t = property(lambda self: float(self.batch.t[0]))
    deposited = property(lambda self: float(self.batch.deposited[0]))
    spilled = property(lambda self: float(self.batch.spilled[0]))
    last_Z = property(lambda self: float(self.batch.last_Z[0]))
    stats = property(lambda self: self.batch.stats[0])

    def advance_frame(self, inlet: Inlet, frame_dt=None):
        return self.batch.advance_frames([inlet], [frame_dt])[0]

    def concentration(self):
        return self.batch.concentration(0)

    def host_state(self):
        return self.batch.host_state(0)

    def ledger(self):
        return self.batch.ledger(0)

    def energy(self):
        return self.batch.energy(0)
