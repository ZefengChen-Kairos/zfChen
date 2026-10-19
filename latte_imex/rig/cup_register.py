"""Cup registration: one cup frame (liquid-surface plane, disc centre, radius) for every consumer of the pour videos.
Replaces the depth-camera plane of topview.py and the free per-frame circles of disc_fit.py (V2_SURFACE_MODEL.md 10.3).

1. plane   rectification plane = depth surface (cup_track surf_z_board, median 9, as topview.py) + dz(t).  The RealSense
           depth reads the milk / crema surface 2-5 mm too deep (light enters the liquid before it scatters back), so
           the three cameras' views of the surface disagree by dz cot(elevation) = 2-5 mm.  dz is measured per frame
           from multi-view consistency: every camera's lightness rectified onto z + dz (DZ_GRID), high-passed, pairwise
           NCC on the liquid patch (r < PATCH_R from the depth-surface centre, > TIP_R from the spout tip), best dz with
           a parabola step; dz(t) = NCC-weighted robust smoothing (frames with best NCC < NCC_MIN carry no weight).
2. fuse    top views re-fused at z + dz(t) (topview.py fusion: per-channel median over the unoccluded views) -> top/;
           the previous frames are kept in top_v0/.
3. rigid   the cup is rigid and its wall flares: R(t) = a_p + b (z(t) - zbar_p) with b shared by all pours, fitted
           (Huber) on frames whose free circle (disc_fit edge detector + RANSAC) covers >= COVER_R deg and lies within
           RJUMP of the pour median.  Each frame's centre is then fitted with R fixed to R(t): RANSAC over two-point
           fixed-radius circles on the edge points whose polarity is liquid inside / dark outside (the outer rim edge,
           dark inside, cannot be chosen), least squares on the inliers.  Centres covering < COVER_C deg or more than
           CJUMP from a +-0.5 s running median are rejected; the rest are smoothed by a robust local-linear fit
           (Gaussian weights, sigma SIG_C, Tukey reweighting) and gaps interpolated.
4. write   raw_data/rig/results/<p>/disc.json and pours/<p>/disc.json, same schema as disc_fit.py (t, surf_z corrected,
           disc_centre = the smoothed track, disc_R = R(t), raw, fit_ok = raw centre accepted, tip_rel_mm, tip_rel_norm,
           jet_h_mm corrected) + dz_mm, plane_ncc, L_m (the pour's length unit: median R(t); every consumer uses it for
           coordinates, R(t) only for the edge of the observed disc) and a sheet raw_data/rig/results/<p>/cup_register.jpg.
    python3 -m latte_imex.rig.cup_register [pattern ...]     (needs the session's pours/ data, as stream_foot.py)"""
import json
import os
import shutil
import sys

import cv2
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trimrec import Take  # noqa: E402

SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "raw_data", "rig", "results")
PATTERNS = ("heart", "push_heart", "leaf", "swan", "tulip")
PX, HALF, OCC = 0.0005, 0.07, 0.010           # top-view grid (topview.py)
DZ_GRID = np.arange(-8.0, 4.01, 1.0) * 1e-3  # [m] plane offsets tried (negative = higher, board z points down)
PATCH_R, TIP_R = 0.030, 0.012                # [m] liquid patch for the plane test
NCC_MIN = 0.35                               # frames below this best mean NCC carry no plane information
SIG_DZ = 0.5                                 # [s] smoothing of dz(t)
COVER_R, RJUMP = 200, 3.0e-3                 # radius-model frames: free-circle coverage [deg], |R - median| [m]
COVER_C, CJUMP = 150, 3.0e-3                 # centre frames: coverage [deg], jump from the running median [m]
SIG_C = 0.10                                 # [s] centre smoothing
INL = 2.5                                    # [px] circle inlier band


# ----------------------------------------------------------------------------------------------- 1. plane
def _rect_L(tk, W2C, c, view, cen, z, n):
    img, L, dep = view
    g = (np.arange(n) + 0.5) * PX - n * PX / 2
    X, Y = np.meshgrid(cen[0] + g, cen[1] + g)
    Pw = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    K = tk.K[c]
    Pc = Pw @ W2C[c][:3, :3].T + W2C[c][:3, 3]
    u = (Pc[:, 0] / Pc[:, 2] * K[0, 0] + K[0, 2]).reshape(n, n).astype(np.float32)
    v = (Pc[:, 1] / Pc[:, 2] * K[1, 1] + K[1, 2]).reshape(n, n).astype(np.float32)
    out = cv2.remap(L, u, v, cv2.INTER_LINEAR)
    dm = cv2.remap(dep, u, v, cv2.INTER_NEAREST)
    vis = ((u >= 0) & (u < img.shape[1] - 1) & (v >= 0) & (v < img.shape[0] - 1)
           & ((dm == 0) | (dm > Pc[:, 2].reshape(n, n) - OCC)))
    return out, vis, X, Y


def _hp(a, m):
    w = m.astype(np.float32)
    b = cv2.GaussianBlur(a * w, (0, 0), 8) / np.maximum(cv2.GaussianBlur(w, (0, 0), 8), 1e-3)
    return np.where(m, a - b, 0)


def _ncc(a, b, m):
    a, b = a[m], b[m]
    if len(a) < 500:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def plane_offsets(tk, ct, zs, cxy):
    """dz [m] and its best mean pairwise NCC per cup_track frame."""
    W2C = [np.linalg.inv(T) for T in tk.c2w]
    n = int(2 * (PATCH_R + 0.004) / PX)
    out = []
    for fi, o in enumerate(ct):
        i = int(np.argmin(np.abs(tk.T - o["t"])))
        views = []
        for c, k, _ in tk.sync(i):
            img = tk.color(c, k)
            views.append((c, (img, cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32),
                              tk.depth(c, k).astype(np.float32) * 1e-3)))
        tip = np.array(o["tip"])
        sc = []
        for dz in DZ_GRID:
            ims = [_rect_L(tk, W2C, c, vw, cxy[fi], zs[fi] + dz, n) for c, vw in views]
            X, Y = ims[0][2], ims[0][3]
            inside = (np.hypot(X - cxy[fi][0], Y - cxy[fi][1]) < PATCH_R) & (np.hypot(X - tip[0], Y - tip[1]) > TIP_R)
            vals = []
            for a in range(len(ims)):
                for b in range(a + 1, len(ims)):
                    m = ims[a][1] & ims[b][1] & inside
                    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    vals.append(_ncc(_hp(ims[a][0], m), _hp(ims[b][0], m), m))
            sc.append(np.nanmean(vals) if np.isfinite(vals).any() else np.nan)
        sc = np.array(sc)
        if not np.isfinite(sc).any():
            out.append((0.0, np.nan))
            continue
        j = int(np.nanargmax(sc))
        d = 0.0
        if 0 < j < len(sc) - 1 and np.isfinite(sc[j - 1:j + 2]).all():
            y0, y1, y2 = sc[j - 1], sc[j], sc[j + 1]
            den = y0 - 2 * y1 + y2
            d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den < 0 else 0.0
        out.append((float(DZ_GRID[j] + d * (DZ_GRID[1] - DZ_GRID[0])), float(sc[j])))
    return np.array(out)


def smooth_dz(t, dz, q, support=1.0):
    """NCC-weighted Gaussian smoothing of dz(t), blended towards the pour median where the local support (sum of
    weights) is below `support` (~2-3 informative frames).  Estimates at the edge of DZ_GRID are not informative;
    estimates more than 1.5 mm from the smooth curve (or, with little support, from the pour median) are dropped."""
    edge = (np.abs(dz - DZ_GRID[0]) < 0.5e-3) | (np.abs(dz - DZ_GRID[-1]) < 0.5e-3)
    w = np.where(np.isfinite(q) & (q >= NCC_MIN) & ~edge, q, 0.0)
    if w.sum() == 0:
        return np.zeros_like(dz)
    med = float(np.median(dz[w > 0]))
    for _ in range(3):
        K = np.exp(-0.5 * ((t[:, None] - t[None, :]) / SIG_DZ) ** 2) * w[None, :]
        den = K.sum(1)
        loc = (K @ dz) / np.maximum(den, 1e-9)
        a = np.clip(den / support, 0.0, 1.0)
        sm = a * loc + (1 - a) * med
        w = np.where(np.abs(dz - sm) < 1.5e-3, w, 0.0)
        med = float(np.median(dz[w > 0])) if (w > 0).any() else med
    return sm


# ----------------------------------------------------------------------------------------------- 2. fuse
def fuse(tk, o, c0, z):
    W2C = [np.linalg.inv(T) for T in tk.c2w]
    i = int(np.argmin(np.abs(tk.T - o["t"])))
    n = int(2 * HALF / PX)
    g = (np.arange(n) + 0.5) * PX - HALF
    X, Y = np.meshgrid(c0[0] + g, c0[1] + g)
    Pw = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    cols, vis = [], []
    for c, k, _ in tk.sync(i):
        K = tk.K[c]
        Pc = Pw @ W2C[c][:3, :3].T + W2C[c][:3, 3]
        u = Pc[:, 0] / Pc[:, 2] * K[0, 0] + K[0, 2]
        v = Pc[:, 1] / Pc[:, 2] * K[1, 1] + K[1, 2]
        img = tk.color(c, k)
        dep = tk.depth(c, k).astype(float) * 1e-3
        ok = (u >= 0) & (u < img.shape[1] - 1) & (v >= 0) & (v < img.shape[0] - 1) & (Pc[:, 2] > 0)
        ui, vi = np.clip(u.astype(int), 0, img.shape[1] - 1), np.clip(v.astype(int), 0, img.shape[0] - 1)
        dm = dep[vi, ui]
        cols.append(img[vi, ui].astype(float))
        vis.append(ok & ((dm == 0) | (dm > Pc[:, 2] - OCC)))
    C, V = np.stack(cols), np.stack(vis)
    top = np.nanmedian(np.where(V[..., None], C, np.nan), 0).reshape(n, n, 3)
    return np.nan_to_num(top).astype(np.uint8), V.any(0).reshape(n, n)


# ----------------------------------------------------------------------------------------------- 3. circles
def edges(img, seen):
    """Liquid / dark-rim boundary points (disc_fit.ransac_disc) and the liquid-inside polarity test image."""
    V = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    bright = ((V > 85) & seen).astype(np.uint8)
    dark = ((V < 60) & seen).astype(np.uint8)
    edge = bright & cv2.dilate(dark, np.ones((5, 5), np.uint8)) & (1 - cv2.erode(bright, np.ones((3, 3), np.uint8)))
    return np.argwhere(edge)[:, ::-1].astype(float), V.astype(np.float32)


def _kasa(x, y):
    A = np.stack([x, y, np.ones_like(x)], 1)
    a = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    cx, cy = a[0] / 2, a[1] / 2
    return cx, cy, np.sqrt(max(a[2] + cx * cx + cy * cy, 1e-9))


def _cover(pts, cx, cy):
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return 10 * len(np.unique(np.floor((ang + np.pi) / (np.pi / 18))))


def free_circle(pts, rng, rpx=(56, 100), iters=1500):
    """disc_fit.ransac_disc: (cx, cy, R, cover) in px or None."""
    if len(pts) < 30:
        return None
    best = (0, None)
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        cx, cy, r = _kasa(s[:, 0], s[:, 1])
        if not (rpx[0] < r < rpx[1]):
            continue
        inl = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < 3
        score = inl.sum() * min(_cover(pts[inl], cx, cy) / 10, 24) / 24
        if score > best[0]:
            best = (score, (cx, cy, r, inl))
    if best[1] is None:
        return None
    cx, cy, r, inl = best[1]
    for _ in range(3):
        cx, cy, r = _kasa(pts[inl, 0], pts[inl, 1])
        inl = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < INL
    return cx, cy, r, _cover(pts[inl], cx, cy)


def fixed_circle(pts, Vimg, r, rng, iters=1500):
    """Centre (px) of a circle of radius r through the edge points whose inner side is bright and outer side dark."""
    if len(pts) < 30:
        return None
    best = (0, None)
    for _ in range(iters):
        a, b = pts[rng.choice(len(pts), 2, replace=False)]
        d = np.hypot(*(b - a))
        if d < 0.5 * r or d >= 2 * r:
            continue
        mid, h = 0.5 * (a + b), np.sqrt(r * r - 0.25 * d * d)
        perp = np.array([-(b - a)[1], (b - a)[0]]) / d
        for c in (mid + h * perp, mid - h * perp):
            dist = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
            inl = np.abs(dist - r) < INL
            if inl.sum() < 20:
                continue
            score = inl.sum() * min(_cover(pts[inl], *c) / 10, 24) / 24
            if score > best[0]:
                best = (score, (c, inl))
    if best[1] is None:
        return None
    c, inl = best[1]
    q = pts[inl]
    # polarity: liquid (bright) inside, rim (dark) outside
    rr = np.hypot(q[:, 0] - c[0], q[:, 1] - c[1])[:, None]
    nrm = (q - c) / np.maximum(rr, 1e-9)
    pin, pout = q - 4 * nrm, q + 4 * nrm
    h, w = Vimg.shape
    sample = lambda P: Vimg[np.clip(P[:, 1].astype(int), 0, h - 1), np.clip(P[:, 0].astype(int), 0, w - 1)]
    pol = sample(pin) > sample(pout) + 15
    q = q[pol]
    if len(q) < 20:
        return None
    for _ in range(10):                                   # Gauss-Newton, radius fixed
        d = q - c
        rr = np.hypot(d[:, 0], d[:, 1])
        J = -d / np.maximum(rr, 1e-9)[:, None]
        res = rr - r
        keep = np.abs(res) < INL
        if keep.sum() < 20:
            return None
        step = np.linalg.lstsq(J[keep], -res[keep], rcond=None)[0]
        c = c + step
        if np.abs(step).max() < 1e-3:
            break
    d = q - c
    res = np.hypot(d[:, 0], d[:, 1]) - r
    keep = np.abs(res) < INL
    return c, _cover(q[keep], *c), int(keep.sum()), float(np.sqrt(np.mean(res[keep] ** 2)))


def robust_track(t, C, ok, sig=SIG_C, it=4, c_tukey=2.0e-3):
    """Robust local-linear smoothing of the accepted centres (Tukey reweighting), evaluated at every t."""
    tt, X = t[ok], C[ok]
    w = np.ones(len(tt))
    for _ in range(it):
        fit = np.empty_like(X)
        for j, tj in enumerate(tt):
            k = np.exp(-0.5 * ((tt - tj) / sig) ** 2) * w
            A = np.stack([np.ones_like(tt), tt - tj], 1) * np.sqrt(k)[:, None]
            fit[j] = np.linalg.lstsq(A, X * np.sqrt(k)[:, None], rcond=None)[0][0]
        r = np.hypot(*(X - fit).T)
        w = np.where(r < c_tukey, (1 - (r / c_tukey) ** 2) ** 2, 0.0)
    out = np.empty((len(t), 2))
    for j, tj in enumerate(t):
        k = np.exp(-0.5 * ((tt - tj) / sig) ** 2) * w
        if k.sum() < 1e-3:                                  # far from any accepted centre: nearest accepted values
            k = np.exp(-0.5 * ((tt - tj) / (4 * sig)) ** 2) * np.maximum(w, 1e-6)
        A = np.stack([np.ones_like(tt), tt - tj], 1) * np.sqrt(k)[:, None]
        out[j] = np.linalg.lstsq(A, X * np.sqrt(k)[:, None], rcond=None)[0][0]
    return out, w


# ----------------------------------------------------------------------------------------------- driver
def stage_plane_fuse(p, log=print):
    """Plane offsets, re-fused top views, free circles and edge points of every frame -> pours/<p>/cup_register_raw.npz."""
    tk = Take(f"{SCR}/ghdata/{p}")
    ct = json.load(open(f"{SCR}/{p}/cup_track.json"))
    t = np.array([o["t"] for o in ct])
    zs = median_filter(np.array([o["surf_z_board"] for o in ct]), 9, mode="nearest")
    cxy = np.array([o["surf_centre"] for o in ct])
    cxy = np.stack([median_filter(cxy[:, 0], 15, mode="nearest"), median_filter(cxy[:, 1], 15, mode="nearest")], 1)
    P = plane_offsets(tk, ct, zs, cxy)
    dz = smooth_dz(t, P[:, 0], P[:, 1])
    log(f"{p}: plane dz median {np.median(P[np.isfinite(P[:, 1]) & (P[:, 1] >= NCC_MIN), 0]) * 1e3:+.1f} mm over "
        f"{int((P[:, 1] >= NCC_MIN).sum())}/{len(P)} frames with NCC >= {NCC_MIN}; smoothed {dz.min() * 1e3:+.1f}..{dz.max() * 1e3:+.1f} mm")
    top, old = f"{SCR}/{p}/top", f"{SCR}/{p}/top_v0"
    if os.path.isdir(top) and not os.path.isdir(old):
        shutil.move(top, old)
    os.makedirs(top, exist_ok=True)
    rng = np.random.default_rng(0)
    free, edge_pts = [], []
    for fi, o in enumerate(ct):
        img, seen = fuse(tk, o, cxy[fi], zs[fi] + dz[fi])
        np.savez_compressed(f"{top}/{fi:04d}.npz", img=img, seen=seen, c0=cxy[fi], z=zs[fi] + dz[fi], px=PX, half=HALF)
        pts, _ = edges(img, seen)
        edge_pts.append(pts)
        fc = free_circle(pts, rng)
        free.append(fc if fc is not None else (np.nan,) * 4)
    np.savez_compressed(f"{SCR}/{p}/cup_register_raw.npz", t=t, zs=zs, dz=dz, dz_raw=P[:, 0], ncc=P[:, 1], c0=cxy,
                        free=np.array(free, float))
    return dict(t=t, zs=zs, dz=dz, ncc=P[:, 1], c0=cxy, free=np.array(free, float))


def refuse(p, tol=0.3e-3, log=print):
    """Re-smooth dz(t) from the stored per-frame estimates and re-fuse only the frames whose plane moved by > tol."""
    fn = f"{SCR}/{p}/cup_register_raw.npz"
    Z = dict(np.load(fn))
    dz = smooth_dz(Z["t"], Z["dz_raw"], Z["ncc"])
    redo = np.where(np.abs(dz - Z["dz"]) > tol)[0]
    if len(redo):
        tk = Take(f"{SCR}/ghdata/{p}")
        ct = json.load(open(f"{SCR}/{p}/cup_track.json"))
        rng = np.random.default_rng(0)
        for fi in redo:
            img, seen = fuse(tk, ct[fi], Z["c0"][fi], Z["zs"][fi] + dz[fi])
            np.savez_compressed(f"{SCR}/{p}/top/{fi:04d}.npz", img=img, seen=seen, c0=Z["c0"][fi], z=Z["zs"][fi] + dz[fi],
                                px=PX, half=HALF)
            fc = free_circle(edges(img, seen)[0], rng)
            Z["free"][fi] = fc if fc is not None else (np.nan,) * 4
    Z["dz"] = dz
    np.savez_compressed(fn, **Z)
    log(f"{p}: plane re-smoothed, {len(redo)} frames re-fused")


def radius_model(raws, log=print):
    """R(t) = a_p + b (z - zbar_p), b shared: Huber IRLS on the frames with a well-covered free circle near the pour median."""
    rows = []
    for j, (p, r) in enumerate(raws.items()):
        Rm = r["free"][:, 2] * PX
        ok = np.isfinite(Rm) & (r["free"][:, 3] >= COVER_R)
        med = np.median(Rm[ok])
        ok &= np.abs(Rm - med) < RJUMP
        z = r["zs"] + r["dz"]
        for zz, RR in zip(z[ok], Rm[ok]):
            rows.append((j, zz - np.mean(z[ok]), RR))
    A = np.array(rows)
    npour = len(raws)
    X = np.zeros((len(A), npour + 1))
    X[np.arange(len(A)), A[:, 0].astype(int)] = 1.0
    X[:, -1] = A[:, 1]
    y = A[:, 2]
    w = np.ones(len(y))
    for _ in range(20):
        beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        res = y - X @ beta
        s = 1.4826 * np.median(np.abs(res)) + 1e-6
        w = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(res), 1e-12))
    b = float(beta[-1])
    out = {}
    for j, (p, r) in enumerate(raws.items()):
        z = r["zs"] + r["dz"]
        ok = np.isfinite(r["free"][:, 2]) & (r["free"][:, 3] >= COVER_R)
        zbar = np.mean(z[ok & (np.abs(r["free"][:, 2] * PX - np.median(r["free"][ok, 2] * PX)) < RJUMP)])
        out[p] = beta[j] + b * (z - zbar)
        log(f"{p}: R = {beta[j] * 1e3:.2f} mm {b:+.3f} (z - zbar), R(t) {out[p].min() * 1e3:.2f}..{out[p].max() * 1e3:.2f} mm, "
            f"residual MAD {1.4826 * np.median(np.abs(res[A[:, 0] == j])) * 1e3:.2f} mm")
    return out, b


def stage_track(p, r, Rt, log=print):
    ct = json.load(open(f"{SCR}/{p}/cup_track.json"))
    t, c0 = r["t"], r["c0"]
    rng = np.random.default_rng(1)
    C = np.full((len(t), 2), np.nan)
    info = []
    for fi in range(len(t)):
        Z = np.load(f"{SCR}/{p}/top/{fi:04d}.npz")
        pts, Vimg = edges(Z["img"], Z["seen"])
        fc = fixed_circle(pts, Vimg, Rt[fi] / PX, rng)
        if fc is None:
            info.append(None)
            continue
        c, cov, ninl, rms = fc
        C[fi] = c0[fi] + (c + 0.5) * PX - HALF
        info.append(dict(cx=float(C[fi, 0]), cy=float(C[fi, 1]), cover_deg=int(cov), n_inl=ninl, rms_px=rms))
    cov = np.array([q["cover_deg"] if q else 0 for q in info])
    ok = np.isfinite(C[:, 0]) & (cov >= COVER_C)
    run = np.full_like(C, np.nan)
    for fi in range(len(t)):
        nb = ok & (np.abs(t - t[fi]) <= 0.5)
        if nb.sum() >= 3:
            run[fi] = np.median(C[nb], 0)
    ok &= np.isfinite(run[:, 0]) & (np.hypot(*(C - run).T) < CJUMP)
    track, w = robust_track(t, C, ok)
    acc = np.zeros(len(t), bool)
    acc[np.where(ok)[0]] = w > 0
    zc = r["zs"] + r["dz"]
    L = float(np.median(Rt))
    res = []
    for fi, o in enumerate(ct):
        tip = np.array(o["tip"])
        cen = track[fi]
        res.append(dict(t=o["t"], surf_z=float(zc[fi]), disc_centre=cen.tolist(), disc_R=float(Rt[fi]), raw=info[fi],
                        fit_ok=bool(acc[fi]), tip_rel_mm=((tip[:2] - cen) * 1e3).tolist(),
                        tip_rel_norm=((tip[:2] - cen) / L).tolist(), jet_h_mm=float((zc[fi] - tip[2]) * 1e3),
                        dz_mm=float(r["dz"][fi] * 1e3), plane_ncc=None if not np.isfinite(r["ncc"][fi]) else float(r["ncc"][fi]),
                        L_m=L, source="cup_register.py"))
    v0 = f"{SCR}/{p}/disc_v0.json"                         # the free per-frame fits (disc_fit.py), for comparison
    if not os.path.exists(v0):
        shutil.copy(f"{SCR}/{p}/disc.json", v0)
    for fn in (os.path.join(RES, p, "disc.json"), f"{SCR}/{p}/disc.json"):
        json.dump(res, open(fn, "w"))
    old = json.load(open(v0))
    oc = np.array([d["disc_centre"] for d in old])
    n = min(len(oc), len(track))
    dev = np.hypot(*(oc[:n] - track[:n]).T) * 1e3
    log(f"{p}: centres accepted {acc.sum()}/{len(t)} (coverage >= {COVER_C} deg: {(cov >= COVER_C).sum()}), L = {L * 1e3:.2f} mm; "
        f"old disc.json vs new track: median {np.median(dev):.2f} mm, p95 {np.percentile(dev, 95):.2f}, max {dev.max():.2f} mm")
    _sheet(p, res, old)
    return res


def _sheet(p, res, old, n=12):
    pick = np.linspace(0, len(res) - 1, n).astype(int)
    big = np.argsort([-np.hypot(*(np.array(old[i]["disc_centre"]) - res[i]["disc_centre"])) for i in range(min(len(old), len(res)))])[:4]
    tiles = []
    for fi in sorted(set(pick.tolist()) | set(big.tolist())):
        Z = np.load(f"{SCR}/{p}/top/{fi:04d}.npz")
        img = Z["img"].copy()
        c0 = Z["c0"]
        for d, col in ((old[fi] if fi < len(old) else None, (0, 0, 255)), (res[fi], (0, 255, 0))):
            if d is None or not np.isfinite(d["disc_R"]):
                continue
            pc = ((np.array(d["disc_centre"]) - c0 + HALF) / PX).astype(int)
            cv2.circle(img, (int(pc[0]), int(pc[1])), int(d["disc_R"] / PX), col, 1)
            cv2.drawMarker(img, (int(pc[0]), int(pc[1])), col, cv2.MARKER_CROSS, 10, 1)
        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"{p} f{fi} t{res[fi]['t'] - res[0]['t']:.1f}", (3, 12), 0, 0.38, (0, 255, 255), 1)
        tiles.append(img)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    cv2.imwrite(os.path.join(RES, p, "cup_register.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(argv=None):
    pats = (argv or sys.argv[1:]) or list(PATTERNS)
    raws = {}
    for p in pats:
        fn = f"{SCR}/{p}/cup_register_raw.npz"
        if os.path.exists(fn) and os.environ.get("REUSE", "1") == "1":
            Z = np.load(fn)
            raws[p] = {k: Z[k] for k in ("t", "zs", "dz", "ncc", "c0", "free")}
        else:
            raws[p] = stage_plane_fuse(p)
    Rt, b = radius_model(raws)
    json.dump(dict(b_dRdz=b, patterns=list(raws)), open(os.path.join(RES, "cup_register.json"), "w"), indent=1)
    for p in raws:
        stage_track(p, raws[p], Rt[p])


if __name__ == "__main__":
    main()
