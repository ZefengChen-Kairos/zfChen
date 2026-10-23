"""GPU port (numba.cuda) of pitcher.ForwardPitcher.run, batched over many pose sequences.

One thread block per pose sequence; the block walks the sequence step by step (the pitcher volume couples the steps):
  free surface   heights of the cavity sample points along e_z, 256-bin histogram of the volumes (shared-memory atomics),
                 the bin holding V gathered into shared memory and ranked there -- the same weighted quantile and the
                 same interpolation between consecutive sorted points as v05_pitcher_check.FastPitcher.free_surface
  weir           front-edge segments (heads, projected widths, flux-weighted exit quantities incl. omega x r) and the
                 side overflow, block reductions (ForwardPitcher.weir)
  volume         thread 0: V_P update, spill, optional outflow inertia (ForwardPitcher.run)
Pose rates (np.gradient of the tip and of R(yaw, tilt, roll), omega from dR R^T) are a separate per-step kernel (for
ForwardLaw.smooth_s = 0; with smoothing the CPU rates are uploaded).  The ballistic flight and the arrival binning
are cheap and run on the host with the CPU code (ForwardPitcher.arrivals unchanged).
The CPU class is not modified; results agree with it to rounding (sums in a different order).

    from latte_imex.cuda_pitcher import GPUForwardPitcher
    G = GPUForwardPitcher(ForwardPitcher(geom, law))
    Es = G.run_batch([dict(t=..., tip=..., yaw=..., tilt=..., roll=..., V0=..., dt=...), ...])   # list of E dicts
"""
from __future__ import annotations
import math
import numpy as np
from numba import cuda, float64, int32

NTP = 128          # threads per block
NB = 256           # histogram bins (FastPitcher.NB)
CAP = 768          # points of one bin held in shared memory (the NX cavity at nr = 6 has <= ~400)
NROW = 12          # rows of the reduction scratch
# output columns
O_VP, O_QO, O_QEQ, O_QSP, O_U, O_H, O_HB, O_AO, O_W, O_ZF, O_UST = range(11)
O_ORG, O_VO, O_DO, O_EM = 11, 14, 17, 20
NOUT = 23
U_HEAD = dict(max=0, local=1, bar=2)
WIDTH = dict(projected=0, arc=1)


@cuda.jit(device=True)
def _red(sh, t, k, op):
    """Block reduction of rows 0..k-1 of sh (NROW, NTP); op 0 = sum, 1 = max, 2 = min. Result in sh[:, 0]."""
    cuda.syncthreads()
    w = NTP // 2
    while w > 0:
        if t < w:
            for q in range(k):
                a = sh[q, t]; b = sh[q, t + w]
                if op == 0:
                    sh[q, t] = a + b
                elif op == 1:
                    sh[q, t] = a if a >= b else b
                else:
                    sh[q, t] = a if a <= b else b
        cuda.syncthreads()
        w //= 2


@cuda.jit(device=True, inline=True)
def _rot(yaw, tilt, roll, R):
    """R = Rz(yaw) Ry(tilt) Rx(roll) (pitcher.rotation), row-major into R[0:9]."""
    cy = math.cos(yaw); sy = math.sin(yaw)
    ct = math.cos(tilt); st = math.sin(tilt)
    cr = math.cos(roll); sr = math.sin(roll)
    # Rz @ Ry
    a00 = cy * ct; a01 = -sy; a02 = cy * st
    a10 = sy * ct; a11 = cy; a12 = sy * st
    a20 = -st; a21 = 0.0; a22 = ct
    # (Rz Ry) @ Rx, Rx = [[1,0,0],[0,cr,-sr],[0,sr,cr]]
    R[0] = a00; R[1] = a01 * cr + a02 * sr; R[2] = -a01 * sr + a02 * cr
    R[3] = a10; R[4] = a11 * cr + a12 * sr; R[5] = -a11 * sr + a12 * cr
    R[6] = a20; R[7] = a21 * cr + a22 * sr; R[8] = -a21 * sr + a22 * cr


@cuda.jit
def k_rates(tip, ang, n, dt, v, om):
    """np.gradient of the tip (v) and of R (dR), omega from the antisymmetric part of dR R^T (ForwardPitcher.pose_rates)."""
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    b = cuda.blockIdx.y
    nb = n[b]
    if i >= nb:
        return
    h = dt[b]
    if i == 0:
        i0 = 0; i1 = 1; den = h
    elif i == nb - 1:
        i0 = nb - 2; i1 = nb - 1; den = h
    else:
        i0 = i - 1; i1 = i + 1; den = 2.0 * h
    for c in range(3):
        v[b, i, c] = (tip[b, i1, c] - tip[b, i0, c]) / den
    Ra = cuda.local.array(9, float64); Rb = cuda.local.array(9, float64); Rc = cuda.local.array(9, float64)
    _rot(ang[b, i0, 0], ang[b, i0, 1], ang[b, i0, 2], Ra)
    _rot(ang[b, i1, 0], ang[b, i1, 1], ang[b, i1, 2], Rb)
    _rot(ang[b, i, 0], ang[b, i, 1], ang[b, i, 2], Rc)
    W = cuda.local.array(9, float64)
    for r in range(3):
        for c in range(3):
            acc = 0.0
            for k in range(3):
                acc += ((Rb[r * 3 + k] - Ra[r * 3 + k]) / den) * Rc[c * 3 + k]
            W[r * 3 + c] = acc
    om[b, i, 0] = 0.5 * (W[7] - W[5])
    om[b, i, 1] = 0.5 * (W[2] - W[6])
    om[b, i, 2] = 0.5 * (W[3] - W[1])


@cuda.jit
def k_forward(pts, dv, tipb, midP, segP, dirP, smidP, ssegP, beak, law, tip, ang, vt, om, n, V0, dtv, out, status):
    """ForwardPitcher.run steps 2-7 for pose sequence b = blockIdx.x (see module docstring)."""
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x
    sh = cuda.shared.array((NROW, NTP), float64)
    hist = cuda.shared.array(NB, float64)
    zb = cuda.shared.array(CAP, float64)
    vb = cuda.shared.array(CAP, float64)
    cb = cuda.shared.array(CAP, float64)
    ib = cuda.shared.array(CAP, int32)
    R = cuda.shared.array(9, float64)
    sc = cuda.shared.array(32, float64)      # broadcast scalars
    nbin = cuda.shared.array(1, int32)
    npt = pts.shape[0]
    nf = midP.shape[0]
    ns = smidP.shape[0]
    C_d = law[0]; C_v = law[1]; g = law[2]; h_k = law[3]; uhead = int(law[4]); wmode = int(law[5])
    side_spill = law[6] != 0.0; inertia = law[7] != 0.0; L_e = law[8]
    Kw = (2.0 / 3.0) * C_d * math.sqrt(2 * g)
    dt = dtv[b]
    if t == 0:
        sc[0] = V0[b]       # V
        sc[1] = 0.0         # U state
    nsteps = n[b]
    for i in range(nsteps):
        if t == 0:
            _rot(ang[b, i, 0], ang[b, i, 1], ang[b, i, 2], R)
        cuda.syncthreads()
        V = sc[0]
        w0 = R[6]; w1 = R[7]; w2 = R[8]                    # R^T e_z
        cst = tip[b, i, 2] - (tipb[0] * w0 + tipb[1] * w1 + tipb[2] * w2)
        # ---- free surface: pass 1, range
        lo = 1e300; hi = -1e300
        k = t
        while k < npt:
            z = pts[k, 0] * w0 + pts[k, 1] * w1 + pts[k, 2] * w2 + cst
            lo = min(lo, z); hi = max(hi, z)
            k += NTP
        sh[0, t] = lo
        _red(sh, t, 1, 2)
        lo = sh[0, 0]
        cuda.syncthreads()
        sh[0, t] = hi
        _red(sh, t, 1, 1)
        hi = sh[0, 0]
        s = NB / ((hi - lo) * (1 + 1e-9) + 1e-300)
        # ---- pass 2, histogram of the volumes
        k = t
        while k < NB:
            hist[k] = 0.0
            k += NTP
        cuda.syncthreads()
        k = t
        while k < npt:
            z = pts[k, 0] * w0 + pts[k, 1] * w1 + pts[k, 2] * w2 + cst
            cuda.atomic.add(hist, int((z - lo) * s), dv[k])
            k += NTP
        cuda.syncthreads()
        if t == 0:
            acc = 0.0
            for q in range(NB):
                acc += hist[q]
                hist[q] = acc
            nbin[0] = 0
        cuda.syncthreads()
        total = hist[NB - 1]
        over = 0.0
        if V <= 0:
            zf = lo
        elif V >= total:
            zf = hi
            over = V - total
        else:
            j = 0
            while hist[j] < V:
                j += 1
            below = hist[j - 1] if j > 0 else 0.0
            # pass 3: gather bin j, max height below it
            mb = -1e300
            k = t
            while k < npt:
                z = pts[k, 0] * w0 + pts[k, 1] * w1 + pts[k, 2] * w2 + cst
                ix = int((z - lo) * s)
                if ix == j:
                    m = cuda.atomic.add(nbin, 0, 1)
                    if m < CAP:
                        zb[m] = z; vb[m] = dv[k]; ib[m] = k
                elif ix < j:
                    mb = max(mb, z)
                k += NTP
            sh[0, t] = mb
            _red(sh, t, 1, 1)
            mb = sh[0, 0]
            cnt = nbin[0]
            if cnt > CAP:
                if t == 0:
                    status[b] = 1
                cnt = CAP
            # rank inside the bin: cum_p = below + sum of dv over points sorted before or at p (key (z, index))
            k = t
            while k < cnt:
                zp = zb[k]; ipk = ib[k]
                a = below
                for q in range(cnt):
                    zq = zb[q]
                    if zq < zp or (zq == zp and ib[q] <= ipk):
                        a += vb[q]
                cb[k] = a
                k += NTP
            cuda.syncthreads()
            # k* = first point in sorted order with cum >= V  (min key among those), fallback: the last point
            bz = 1e300; bi = 2147483647.0; lz = -1e300; li = -1.0
            k = t
            while k < cnt:
                zp = zb[k]; ipk = float(ib[k])
                if cb[k] >= V and (zp < bz or (zp == bz and ipk < bi)):
                    bz = zp; bi = ipk
                if zp > lz or (zp == lz and ipk > li):
                    lz = zp; li = ipk
                k += NTP
            sh[0, t] = bz
            _red(sh, t, 1, 2)
            mz = sh[0, 0]
            cuda.syncthreads()
            sh[0, t] = bi if bz == mz else 2147483647.0
            _red(sh, t, 1, 2)
            mi = sh[0, 0]
            cuda.syncthreads()
            if mz >= 1e299:                              # no point reached V (rounding): take the last one
                sh[0, t] = lz
                _red(sh, t, 1, 1)
                mz = sh[0, 0]
                cuda.syncthreads()
                sh[0, t] = li if lz == mz else -1.0
                _red(sh, t, 1, 1)
                mi = sh[0, 0]
                cuda.syncthreads()
            # predecessor of k* in sorted order: max key < (mz, mi)
            pz = -1e300; pi = -1.0; ck = 0.0
            k = t
            while k < cnt:
                zp = zb[k]; ipk = float(ib[k])
                if zp == mz and ipk == mi:
                    ck = cb[k]
                elif (zp < mz or (zp == mz and ipk < mi)) and (zp > pz or (zp == pz and ipk > pi)):
                    pz = zp; pi = ipk
                k += NTP
            sh[0, t] = pz
            sh[1, t] = ck
            _red(sh, t, 2, 1)                            # ck: only one thread holds it (cum >= below >= 0)
            Pz = sh[0, 0]; cumk = sh[1, 0]
            cuda.syncthreads()
            sh[0, t] = pi if pz == Pz else -1.0
            _red(sh, t, 1, 1)
            Pi = sh[0, 0]
            cuda.syncthreads()
            if Pi >= 0.0:
                v0 = 0.0
                k = t
                while k < cnt:
                    if zb[k] == Pz and float(ib[k]) == Pi:
                        v0 = cb[k]
                    k += NTP
                sh[0, t] = v0
                _red(sh, t, 1, 1)
                v0 = sh[0, 0]
                c0 = Pz
                cuda.syncthreads()
            elif mb > -1e299:
                c0 = mb; v0 = below
            else:
                c0 = mz; v0 = 0.0
            f = (V - v0) / max(cumk - v0, 1e-18)
            zf = c0 + f * (mz - c0)
        # ---- weir over the front edge (ForwardPitcher.weir)
        tx = tip[b, i, 0]; ty = tip[b, i, 1]; tz = tip[b, i, 2]
        Hm = 0.0; ex = 0.0; ey = 0.0; zmin = 1e300; amin = 2147483647.0
        k = t
        while k < nf:
            mz_ = ((midP[k, 0] - tipb[0]) * R[6] + (midP[k, 1] - tipb[1]) * R[7] + (midP[k, 2] - tipb[2]) * R[8]) + tz
            h = max(zf - mz_ - h_k, 0.0)
            Hm = max(Hm, h)
            sx = segP[k, 0] * R[0] + segP[k, 1] * R[1] + segP[k, 2] * R[2]
            sy = segP[k, 0] * R[3] + segP[k, 1] * R[4] + segP[k, 2] * R[5]
            dx = dirP[k, 0] * R[0] + dirP[k, 1] * R[1] + dirP[k, 2] * R[2]
            dy = dirP[k, 0] * R[3] + dirP[k, 1] * R[4] + dirP[k, 2] * R[5]
            nn = max(math.sqrt(dx * dx + dy * dy), 1e-12)
            wa = h ** 1.5 * math.sqrt(sx * sx + sy * sy)
            ex += wa * (dx / nn); ey += wa * (dy / nn)
            if mz_ < zmin:
                zmin = mz_; amin = float(k)
            k += NTP
        sh[0, t] = Hm
        _red(sh, t, 1, 1)
        H = sh[0, 0]
        cuda.syncthreads()
        sh[0, t] = ex; sh[1, t] = ey
        _red(sh, t, 2, 0)
        ex = sh[0, 0]; ey = sh[1, 0]
        cuda.syncthreads()
        sh[0, t] = zmin
        _red(sh, t, 1, 2)
        Zm = sh[0, 0]
        cuda.syncthreads()
        sh[0, t] = amin if zmin == Zm else 2147483647.0
        _red(sh, t, 1, 2)
        kmin = int(sh[0, 0])
        cuda.syncthreads()
        if not (H > 0):
            cb_ = math.cos(beak[0]); sb_ = math.sin(beak[0])
            ex = R[0] * cb_ + R[1] * sb_
            ey = R[3] * cb_ + R[4] * sb_
        en = max(math.sqrt(ex * ex + ey * ey), 1e-12)
        ex /= en; ey /= en
        # widths, flows, side overflow
        Qs_ = 0.0; wd = 0.0; hsm = 0.0; ssum = 0.0
        k = t
        while k < nf:
            mz_ = ((midP[k, 0] - tipb[0]) * R[6] + (midP[k, 1] - tipb[1]) * R[7] + (midP[k, 2] - tipb[2]) * R[8]) + tz
            h = max(zf - mz_ - h_k, 0.0)
            sx = segP[k, 0] * R[0] + segP[k, 1] * R[1] + segP[k, 2] * R[2]
            sy = segP[k, 0] * R[3] + segP[k, 1] * R[4] + segP[k, 2] * R[5]
            if wmode == 0:
                pr = sx * ex + sy * ey
                ax = sx - pr * ex; ay = sy - pr * ey
                dw = math.sqrt(ax * ax + ay * ay)
            else:
                dw = math.sqrt(sx * sx + sy * sy)
            Qs_ += Kw * dw * h ** 1.5
            if h > 0:
                wd += dw
            k += NTP
        if side_spill:
            k = t
            while k < ns:
                mz_ = ((smidP[k, 0] - tipb[0]) * R[6] + (smidP[k, 1] - tipb[1]) * R[7] + (smidP[k, 2] - tipb[2]) * R[8]) + tz
                hs = max(zf - mz_ - h_k, 0.0)
                hsm = max(hsm, hs)
                sx = ssegP[k, 0] * R[0] + ssegP[k, 1] * R[1] + ssegP[k, 2] * R[2]
                sy = ssegP[k, 0] * R[3] + ssegP[k, 1] * R[4] + ssegP[k, 2] * R[5]
                ssum += math.sqrt(sx * sx + sy * sy) * hs ** 1.5
                k += NTP
        sh[0, t] = Qs_; sh[1, t] = wd; sh[2, t] = ssum
        _red(sh, t, 3, 0)
        Q = sh[0, 0]; width = sh[1, 0]; ssum = sh[2, 0]
        cuda.syncthreads()
        sh[0, t] = hsm
        _red(sh, t, 1, 1)
        hsm = sh[0, 0]
        cuda.syncthreads()
        Qsp = Kw * ssum if (side_spill and hsm > 0) else 0.0
        wet = Q > 0 and H > 0
        Ub = 0.0
        if wet and uhead == 2:
            hq = 0.0
            k = t
            while k < nf:
                mz_ = ((midP[k, 0] - tipb[0]) * R[6] + (midP[k, 1] - tipb[1]) * R[7] + (midP[k, 2] - tipb[2]) * R[8]) + tz
                h = max(zf - mz_ - h_k, 0.0)
                sx = segP[k, 0] * R[0] + segP[k, 1] * R[1] + segP[k, 2] * R[2]
                sy = segP[k, 0] * R[3] + segP[k, 1] * R[4] + segP[k, 2] * R[5]
                if wmode == 0:
                    pr = sx * ex + sy * ey
                    ax = sx - pr * ex; ay = sy - pr * ey
                    dw = math.sqrt(ax * ax + ay * ay)
                else:
                    dw = math.sqrt(sx * sx + sy * sy)
                hq += Kw * dw * h ** 1.5 * h
                k += NTP
            sh[0, t] = hq
            _red(sh, t, 1, 0)
            Ub = C_v * math.sqrt(2 * g * (sh[0, 0] / Q))
            cuda.syncthreads()
        # flux-weighted exit quantities
        for q in range(NROW):
            sh[q, t] = 0.0
        if wet:
            Umax = C_v * math.sqrt(2 * g * H)
            k = t
            while k < nf:
                mx = ((midP[k, 0] - tipb[0]) * R[0] + (midP[k, 1] - tipb[1]) * R[1] + (midP[k, 2] - tipb[2]) * R[2]) + tx
                my = ((midP[k, 0] - tipb[0]) * R[3] + (midP[k, 1] - tipb[1]) * R[4] + (midP[k, 2] - tipb[2]) * R[5]) + ty
                mz_ = ((midP[k, 0] - tipb[0]) * R[6] + (midP[k, 1] - tipb[1]) * R[7] + (midP[k, 2] - tipb[2]) * R[8]) + tz
                h = max(zf - mz_ - h_k, 0.0)
                sx = segP[k, 0] * R[0] + segP[k, 1] * R[1] + segP[k, 2] * R[2]
                sy = segP[k, 0] * R[3] + segP[k, 1] * R[4] + segP[k, 2] * R[5]
                if wmode == 0:
                    pr = sx * ex + sy * ey
                    ax = sx - pr * ex; ay = sy - pr * ey
                    dw = math.sqrt(ax * ax + ay * ay)
                else:
                    dw = math.sqrt(sx * sx + sy * sy)
                Qi = Kw * dw * h ** 1.5
                if uhead == 0:
                    Ui = Umax
                elif uhead == 1:
                    Ui = C_v * math.sqrt(2 * g * h)
                else:
                    Ui = Ub
                dx = dirP[k, 0] * R[0] + dirP[k, 1] * R[1] + dirP[k, 2] * R[2]
                dy = dirP[k, 0] * R[3] + dirP[k, 1] * R[4] + dirP[k, 2] * R[5]
                dz = dirP[k, 0] * R[6] + dirP[k, 1] * R[7] + dirP[k, 2] * R[8]
                rx = mx - tx; ry = my - ty; rz = mz_ - tz
                vx = vt[b, i, 0] + (om[b, i, 1] * rz - om[b, i, 2] * ry) + Ui * dx
                vy = vt[b, i, 1] + (om[b, i, 2] * rx - om[b, i, 0] * rz) + Ui * dy
                vz = vt[b, i, 2] + (om[b, i, 0] * ry - om[b, i, 1] * rx) + Ui * dz
                w = Qi / Q
                sh[0, t] += w * Ui
                sh[1, t] += w * mx; sh[2, t] += w * my; sh[3, t] += w * mz_
                sh[4, t] += w * vx; sh[5, t] += w * vy; sh[6, t] += w * vz
                sh[7, t] += w * dx; sh[8, t] += w * dy; sh[9, t] += w * dz
                if Qi > 0:
                    sh[10, t] += Qi / Ui
                sh[11, t] += Qi * h
                k += NTP
        _red(sh, t, NROW, 0)
        if t == 0:
            Qeq = Q if wet else 0.0
            if wet:
                U = sh[0, 0]; ox = sh[1, 0]; oy = sh[2, 0]; oz = sh[3, 0]
                vox = sh[4, 0]; voy = sh[5, 0]; voz = sh[6, 0]
                dn = max(math.sqrt(sh[7, 0] ** 2 + sh[8, 0] ** 2 + sh[9, 0] ** 2), 1e-12)
                dox = sh[7, 0] / dn; doy = sh[8, 0] / dn; doz = sh[9, 0] / dn
                Ao = sh[10, 0]; hbar = sh[11, 0] / Q
            else:
                U = 0.0; vox = 0.0; voy = 0.0; voz = 0.0; dox = 0.0; doy = 0.0; doz = 0.0; Ao = 0.0; hbar = 0.0
                kk = kmin
                ox = ((midP[kk, 0] - tipb[0]) * R[0] + (midP[kk, 1] - tipb[1]) * R[1] + (midP[kk, 2] - tipb[2]) * R[2]) + tx
                oy = ((midP[kk, 0] - tipb[0]) * R[3] + (midP[kk, 1] - tipb[1]) * R[4] + (midP[kk, 2] - tipb[2]) * R[5]) + ty
                oz = ((midP[kk, 0] - tipb[0]) * R[6] + (midP[kk, 1] - tipb[1]) * R[7] + (midP[kk, 2] - tipb[2]) * R[8]) + tz
            Qo = Qeq
            Ust = sc[1]
            if inertia:
                a_ = dt / (2 * C_v ** 2 * L_e)
                b_ = Ust + dt * g * max(H, 0.0) / L_e
                if b_ > 0:
                    Ust = (-1 + math.sqrt(1 + 4 * a_ * b_)) / (2 * a_)
                else:
                    Ust = b_ / (1 + a_ * abs(Ust))
                Ueq = C_v * math.sqrt(2 * g * H) if H > 0 else 0.0
                Qo = Qeq / Ueq * max(Ust, 0.0) if (Qeq > 0 and Ueq > 0) else 0.0
            sp = Qsp + over / dt
            o_ = (Qo + sp) * dt
            if o_ > V:
                ff = V / o_
                Qo = Qo * ff; sp = sp * ff
            V -= (Qo + sp) * dt
            if Qeq > 0 and Qo != Qeq:
                Ao *= Qo / Qeq
            out[b, i, O_VP] = V; out[b, i, O_QO] = Qo; out[b, i, O_QEQ] = Qeq; out[b, i, O_QSP] = sp
            out[b, i, O_U] = U; out[b, i, O_H] = H; out[b, i, O_HB] = hbar; out[b, i, O_AO] = Ao
            out[b, i, O_W] = width; out[b, i, O_ZF] = zf; out[b, i, O_UST] = Ust
            out[b, i, O_ORG] = ox; out[b, i, O_ORG + 1] = oy; out[b, i, O_ORG + 2] = oz
            out[b, i, O_VO] = vox; out[b, i, O_VO + 1] = voy; out[b, i, O_VO + 2] = voz
            out[b, i, O_DO] = dox; out[b, i, O_DO + 1] = doy; out[b, i, O_DO + 2] = doz
            out[b, i, O_EM] = ex; out[b, i, O_EM + 1] = ey; out[b, i, O_EM + 2] = 0.0
            sc[0] = V; sc[1] = Ust
        cuda.syncthreads()


class GPUForwardPitcher:
    """Batched GPU replay of ForwardPitcher.run (same geometry precomputation: the CPU object's front edge, segments,
    directions and beak axis are uploaded as they are)."""

    def __init__(self, fp):
        self.fp = fp
        g, law = fp.geom, fp.law
        if law.U_head not in U_HEAD or law.width not in WIDTH:
            raise ValueError("unknown ForwardLaw option")
        if getattr(law, "exit", "lip") != "lip":
            raise ValueError("GPUForwardPitcher: ForwardLaw(exit='wall') (mi27 wall exit) runs on the CPU only")
        self.d = dict(
            pts=cuda.to_device(np.ascontiguousarray(g.points, np.float64)),
            dv=cuda.to_device(np.ascontiguousarray(g.dv, np.float64)),
            tipb=cuda.to_device(np.asarray(g.tip, np.float64)),
            midP=cuda.to_device(np.ascontiguousarray(fp.mid_P)), segP=cuda.to_device(np.ascontiguousarray(fp.seg_P)),
            dirP=cuda.to_device(np.ascontiguousarray(fp.dir_P)),
            smidP=cuda.to_device(np.ascontiguousarray(fp.side_mid_P if len(fp.side_a) else np.zeros((1, 3)))),
            ssegP=cuda.to_device(np.ascontiguousarray(fp.side_seg_P if len(fp.side_a) else np.zeros((1, 3)))),
            beak=cuda.to_device(np.array([fp.beak_phi])),
            law=cuda.to_device(np.array([law.C_d, law.C_v, law.g, law.h_k, U_HEAD[law.U_head], WIDTH[law.width],
                                         float(bool(law.side_spill and len(fp.side_a))), float(bool(law.inertia)),
                                         law.L_e])))

    def run_batch(self, seqs):
        """seqs: list of dict(t, tip (n,3), yaw, tilt, roll, V0, dt=None).  Returns the list of E dicts of
        ForwardPitcher.run (same keys and units), plus E['gpu_status'] (1 = a histogram bin overflowed the shared
        buffer: that sequence was recomputed on the CPU)."""
        fp, law = self.fp, self.fp.law
        B = len(seqs)
        ns = [len(np.asarray(s["t"])) for s in seqs]
        nmax = max(ns)
        tip = np.zeros((B, nmax, 3)); ang = np.zeros((B, nmax, 3))
        dts = np.zeros(B); V0 = np.zeros(B)
        for b, s in enumerate(seqs):
            n = ns[b]
            t = np.asarray(s["t"], float)
            tip[b, :n] = np.asarray(s["tip"], float)
            for c, k in enumerate(("yaw", "tilt", "roll")):
                ang[b, :n, c] = np.broadcast_to(np.asarray(s[k], float), (n,))
            dts[b] = float(t[1] - t[0]) if s.get("dt") is None else float(s["dt"])
            V0[b] = float(s["V0"])
        dtip, dang = cuda.to_device(tip), cuda.to_device(ang)
        dn = cuda.to_device(np.array(ns, np.int32))
        vt = cuda.device_array((B, nmax, 3)); om = cuda.device_array((B, nmax, 3))
        if law.smooth_s > 0:
            vh = np.zeros((B, nmax, 3)); oh = np.zeros((B, nmax, 3))
            for b, s in enumerate(seqs):
                n = ns[b]
                v, o = fp.pose_rates(np.asarray(s["t"], float), tip[b, :n], ang[b, :n, 0], ang[b, :n, 1], ang[b, :n, 2],
                                     law.smooth_s)
                vh[b, :n], oh[b, :n] = v, o
            vt, om = cuda.to_device(vh), cuda.to_device(oh)
        else:
            rdt = np.array([float(np.asarray(s["t"], float)[1] - np.asarray(s["t"], float)[0]) for s in seqs])
            k_rates[((nmax + 127) // 128, B), 128](dtip, dang, dn, cuda.to_device(rdt), vt, om)
        out = cuda.device_array((B, nmax, NOUT))
        status = cuda.to_device(np.zeros(B, np.int32))
        d = self.d
        k_forward[B, NTP](d["pts"], d["dv"], d["tipb"], d["midP"], d["segP"], d["dirP"], d["smidP"], d["ssegP"], d["beak"],
                          d["law"], dtip, dang, vt, om, dn, cuda.to_device(V0), cuda.to_device(dts), out, status)
        O = out.copy_to_host(); st = status.copy_to_host()
        vth, omh = vt.copy_to_host(), om.copy_to_host()
        res = []
        for b, s in enumerate(seqs):
            n = ns[b]
            if st[b]:
                E = fp.run(s["t"], s["tip"], s["yaw"], s["tilt"], s["roll"], s["V0"], s.get("dt"))
                E["gpu_status"] = 1
                res.append(E)
                continue
            o = O[b, :n]
            E = {k: o[:, c].copy() for k, c in (("V_P", O_VP), ("Q_o", O_QO), ("Q_eq", O_QEQ), ("Q_spill", O_QSP),
                                               ("U", O_U), ("H", O_H), ("h_bar", O_HB), ("A_o", O_AO), ("width", O_W),
                                               ("z_f", O_ZF), ("U_state", O_UST))}
            for k, c in (("origin", O_ORG), ("v_o", O_VO), ("d_o", O_DO), ("e_main", O_EM)):
                E[k] = o[:, c:c + 3].copy()
            t = np.asarray(s["t"], float)
            E.update(t=t, dt=float(dts[b]), v_tip=vth[b, :n].copy(), omega=omh[b, :n].copy(), V0=float(V0[b]), gpu_status=0)
            ballistic(E, law.g)
            res.append(E)
        return res


def ballistic(E, g):
    """Step 8 of ForwardPitcher.run (the same formulas): flight to z = 0, hit point, normal speed, arrival time."""
    vz, z0 = E["v_o"][:, 2], E["origin"][:, 2]
    tf = np.where(z0 > 0, (vz + np.sqrt(vz ** 2 + 2 * g * np.maximum(z0, 0.0))) / g, 0.0)
    E["t_f"] = np.where(E["Q_o"] > 0, tf, np.nan)
    E["x_hit"] = E["origin"][:, :2] + E["v_o"][:, :2] * tf[:, None]
    E["U_perp"] = np.sqrt(vz ** 2 + 2 * g * np.maximum(z0, 0.0))
    E["u_h"] = E["v_o"][:, :2].copy()
    E["t_arr"] = E["t"] + tf
    return E


def compare(E_cpu, E_gpu, keys=("V_P", "Q_o", "Q_spill", "U", "H", "z_f", "A_o", "width", "origin", "v_o", "x_hit",
                                 "U_perp", "t_arr")):
    """Max abs difference per key (NaN-aware) and the scale of each key."""
    out = {}
    for k in keys:
        a, b = np.nan_to_num(np.asarray(E_cpu[k], float)), np.nan_to_num(np.asarray(E_gpu[k], float))
        out[k] = (float(np.max(np.abs(a - b))) if a.size else 0.0, float(np.max(np.abs(a))) if a.size else 0.0)
    return out
