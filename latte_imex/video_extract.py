"""Per-frame extraction of the six reference latte-art clips into the solver/target frame.

    python -m latte_imex.video_extract heart                 # -> latte_imex/video_targets/heart/{frames.npz, observables.json, ...}
    python -m latte_imex.video_extract heart --stages my_stages.json --M 128 --fps 5

One shared pipeline for all six patterns (heart, push_heart, layered_heart, tulip, leaf, swan):

 1. cup rectification, every frame (15 fps): the liquid surface is fitted by an ellipse (RANSAC on the outer boundary
    of the crema region, then a shape-constrained refit with the axis ratio/orientation taken from the clean frames,
    then robust temporal smoothing); frames whose fit is unreliable are flagged and get the interpolated track.  The
    ellipse is mapped affinely to the unit disc exactly as `targets.warp` does it (same axis/sign convention, same
    per-pattern rotation `targets.FRAMES[name]['rot']`), so the reference frame reproduces targets/<name>.npz.
 2. whiteness w in [0, 1] (foam vs crema) in the rectified frame, with a validity mask: pixels hidden by the pitcher,
    the milk stream, the instructor's overlay (red dots + white lines), subtitles/watermarks and glare are INVALID
    (not brown).  Levels per frame: the frame's own foam/crema split (same method as targets.py: Otsu, or 3-class
    k-means for the leaf) where both phases are clearly present (separation >= 0.25, bright class less saturated),
    smoothed over +-0.6 s; frames without foam use the measured crema level + the clip's median contrast offset.
    The reference frame uses exactly the targets.py levels.
 3. observables in physical units (cup diameter D = 8 cm): white area A(t) and dA/dt, first-white delay after each
    pour onset, brown hole radius at the impact during low pours, displacement of existing white blocks during a
    push, brown-gap widths between components, stripe pitch / brown-line width in wiggle regions, cut-line width and
    stem-tip position.  Pour onsets/ends come from the stage script (format below).
 4. standard files in video_targets/<name>/ (+ preview.png contact sheet and the QA movie extraction.mp4: camera crop
    with the ellipse | rectified frame | w with invalid pixels grey):
    frames.npz   t [n] clip time (k / fps_out, the last frame is the targets.FRAMES reference frame), frame_index [n],
                 white [n, M, M] float16 w in [0, 1] (0 where invalid), valid [n, M, M] bool, disk [M, M] (r <= 0.49),
                 findex [n, M, M] float16 raw foam index (1-S)V (to re-threshold), lo/hi [n] levels used
                 (w = clip((f - lo) / (hi - lo))), levels_own [n] (frame's own foam/crema split accepted),
                 ell [n, 5] (cx, cy, a, b, phi) image px / rad in the targets.py axis convention, quality [n]
                 (2 reliable fit, 1 partial fit, 0 propagated by cup-motion registration: tilt phase / heavy occlusion),
                 occ_frac [n] invalid fraction of the disc, tip [n, 2] stream foot / pitcher tip in T (NaN = none),
                 stream_px [n], overlay [n], M, rot, flip, method, fps, fps_out, t_ref, D_cup, and the full 15 fps
                 ellipse track (track_t, track_ell, track_quality, track_cov).
    observables.json  see compute_observables(); lengths in cm (or *_D in cup diameters, *_T / *_M positions).
    Caveat: during a pour the foam is creamier than at the end (its f-histogram mode is ~0.55-0.6 against ~0.85 for
    the finished pattern), and the end frames are three-modal (crema / light-tan thin foam / white foam): targets.py
    counts the light-tan mode as brown, the per-frame split during the pour counts the only foam mode as white, so
    A(t) of leaf / swan / layered_heart drops in the last 1-2 s.  findex lets a user apply another threshold.

COORDINATE FRAMES (verified, see `frame_check` / notes in observables.json)
---------------------------------------------------------------------------
 image  : pixel (col, row), row pointing down, 480 x 640.
 T      : target/solver-comparison frame = targets.py frame = frames.npz frame.  Arrays [iy, ix] on M x M cells covering
          [-0.5, 0.5]^2 in cup diameters, x = (ix+0.5)/M-0.5, y = (iy+0.5)/M-0.5 (row index grows with y: "y up" when
          plotted with origin='lower').  Pattern axis = y; the pours start on the B side (spout / pitcher-handle side) at
          y < 0 and the cut exits (heart tip, stem, D side) at y > 0.  Image -> T:  p_img = c + 2a Xr e_x - 2b Yr e_y with
          (Xr, Yr) = R(rot) (s X, Y), s = -1 if flip else 1, e_x = major axis pointing to image-left, e_y = (e_x[1],
          -e_x[0]) (points image-down).  det[e_x, e_y] = -1: the T frame is the MIRROR image of the physical top view
          (all six targets; e.g. the watermark text is mirrored in targets/swan.npz).
 M      : pitcher.Move frame = solver Inlet frame = solver arrays c[iy, ix] (y = (iy+0.5)/N - 0.5).  Right-handed world
          top view (z up), Barista yaw = -pi/2, i.e. the spout points to -y and the pitcher handle / barista is at +y;
          V0.5 pours start at +y and the cut runs toward -y.
 T -> M :  (x_M, y_M) = (x_T, -y_T)            (a reflection; it also undoes the mirror of the T frame)
 M -> T :  the same map; a solver field c_M[iy, ix] is compared with a target/frames.npz field as c_M[::-1, :].
 Check  :  V0.5 heart through the old solver (action_opt.run_one, model 'old') ends with the stem at y_M < 0 and the
          notch at y_M > 0; targets/heart.npz has the stem at y_T > 0.  Rows flipped, the two agree (frame_check()).
 The video-frame positions written in stage scripts are given in the M frame (cup-diameter units), so they can be
 fed to pitcher.Move directly; use `t_to_m` on positions measured in frames.npz.

STAGE-SCRIPT FORMAT (video_targets/<name>/stages.json, written by the per-pattern agents)
------------------------------------------------------------------------------------------
{"name": "heart", "frame": "M", "Q_ref": 1.5e-05, "t_clip0": 0.0,
 "stages": [ {"t0": 0.0, "t1": 1.5, "kind": "low_pour", "p0": [0.0, 0.20], "p1": [0.0, 0.20],
              "z0": 0.015, "z1": 0.015, "flow0": 1.0, "flow1": 1.0,
              "wiggle": null, "confidence": 0.7, "note": "B point, cup tilted"}, ...]}
 t0, t1   clip time [s] (absolute time in the reference clip); stages are ordered, gaps become dry repositions;
 kind     low_pour | wiggle | push | retreat | lift | high_cut | reposition | stop;
 p0, p1   stream impact point at t0 / t1 in the M frame [cup diameters] (straight path, cosine easing as Barista);
          optional "bezier": [[c1x, c1y], [c2x, c2y]];
 z0, z1   spout height above the surface [m] (low pour ~0.010-0.025, lift -> 0.05-0.07, high cut 0.07-0.10);
 flow0/1  wanted flow relative to Q_ref (1.0 = Q_ref = 15 mL/s; "big flow" 1.3-1.8, "small flow" 0.2-0.4; 0 = dry);
          defaults by kind when omitted (reposition/stop: 0);
 wiggle   null or {"amp": half-amplitude [cup diameters], "amp1": optional end amplitude, "freq": [Hz],
          "axis": "x" | "y" | angle in degrees in the M frame (0 = x, 90 = y)};
 confidence in [0, 1] (how well the video pins this stage), note: free text.
 `stages_to_moves(script)` -> (list[pitcher.Move], t_clip0); a "move away" is appended like pitcher.v05_moves.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from . import targets as TG

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "video_targets")
CLIP_DIRS = [os.environ.get("LATTE_CLIPS", "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/refclips")]
PATTERNS = ["heart", "push_heart", "layered_heart", "tulip", "leaf", "swan"]
D_CUP = 0.08            # cup (liquid-surface) diameter [m]
Q_REF = 15e-6           # flow level 1.0 [m^3/s]
STAGE_KINDS = ("low_pour", "wiggle", "push", "retreat", "lift", "high_cut", "reposition", "stop")


# =========================================================================== video io
def clip_path(name):
    for d in CLIP_DIRS:
        p = os.path.join(d, f"{name}.mp4")
        if os.path.exists(p):
            return p
    p = os.path.join(TG.REF, f"reference_{name}.mp4")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"no clip for {name} in {CLIP_DIRS + [TG.REF]}")


def read_video(path):
    """All frames as uint8 [n, H, W, 3] and the frame rate (frame k is at t = k / fps)."""
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                           "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.strip().split(",")
    W, H = int(info[0]), int(info[1])
    num, den = info[2].split("/")
    fps = float(num) / float(den)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3), fps


# =========================================================================== ellipse geometry
# parameters p = (cx, cy, a, b, phi): centre [px], semi-axes a >= b [px], phi = angle of the major axis in image
# coordinates (x right, y down), normalised so that e_x = (cos phi, sin phi) points to image-left (targets.py convention)
def canon(p):
    cx, cy, a, b, phi = p
    if b > a:
        a, b, phi = b, a, phi + math.pi / 2
    if math.cos(phi) > 0:
        phi += math.pi
    phi = (phi + math.pi) % (2 * math.pi) - math.pi          # (-pi, pi]; e_x points left -> |phi| > pi/2
    return (float(cx), float(cy), float(a), float(b), float(phi))


def basis(p):
    """targets.warp axes: e_x (major, image-left) and e_y = (e_x[1], -e_x[0]) (minor, image-down)."""
    phi = p[4]
    ex = np.array([math.cos(phi), math.sin(phi)])
    return ex, np.array([ex[1], -ex[0]])


def fit_direct(x, y):
    """Halir-Flusser direct least-squares ellipse -> canonical p or None."""
    if len(x) < 6:
        return None
    mx, my = x.mean(), y.mean()
    s = math.sqrt(max(((x - mx) ** 2 + (y - my) ** 2).mean() / 2.0, 1e-12))
    xn, yn = (x - mx) / s, (y - my) / s
    D1 = np.stack([xn * xn, xn * yn, yn * yn], 1)
    D2 = np.stack([xn, yn, np.ones_like(xn)], 1)
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
        Mm = S1 + S2 @ T
        Mm = np.array([Mm[2] / 2.0, -Mm[1], Mm[0] / 2.0])
        _, v = np.linalg.eig(Mm)
    except np.linalg.LinAlgError:
        return None
    v = np.real(v)
    k = np.nonzero(4 * v[0] * v[2] - v[1] ** 2 > 0)[0]
    if len(k) == 0:
        return None
    A, B, C = v[:, k[0]]
    Dd, E, F = T @ v[:, k[0]]
    Q = np.array([[A, B / 2], [B / 2, C]])
    try:
        c = np.linalg.solve(2 * Q, -np.array([Dd, E]))
    except np.linalg.LinAlgError:
        return None
    kk = -(c @ Q @ c + np.array([Dd, E]) @ c + F)
    ev, evec = np.linalg.eigh(Q)
    if ev[0] == 0 or ev[1] == 0 or kk / ev[0] <= 0 or kk / ev[1] <= 0:
        return None
    ax = np.sqrt(kk / ev)
    return canon((c[0] * s + mx, c[1] * s + my, ax[0] * s, ax[1] * s, math.atan2(evec[1, 0], evec[0, 0])))


def ell_uv(p, x, y):
    cx, cy, a, b, phi = p
    c, s = math.cos(phi), math.sin(phi)
    return (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c


def ell_dist(p, x, y):
    """Signed Sampson distance [px] (> 0 outside)."""
    a, b = p[2], p[3]
    u, v = ell_uv(p, x, y)
    F = u * u / (a * a) + v * v / (b * b) - 1.0
    return F / np.maximum(2 * np.hypot(u / (a * a), v / (b * b)), 1e-9)


def ell_rho(p, x, y):
    """Elliptic radius (1 on the ellipse)."""
    u, v = ell_uv(p, x, y)
    return np.sqrt((u / p[2]) ** 2 + (v / p[3]) ** 2)


def coverage(p, x, y, nbin=36):
    """Fraction of angular bins (parametric angle) holding at least one of the points."""
    if len(x) == 0:
        return 0.0
    u, v = ell_uv(p, x, y)
    th = np.arctan2(v / p[3], u / p[2])
    return len(np.unique(((th + math.pi) / (2 * math.pi) * nbin).astype(int) % nbin)) / nbin


def plausible(p, shape):
    if p is None:
        return False
    cx, cy, a, b, phi = p
    H, W = shape
    tilt = abs(((phi + math.pi / 2) % math.pi) - math.pi / 2)
    return 45 < a < 0.7 * W and 0.45 < b / a <= 1.0 and tilt < math.radians(40) and 0 < cx < W and 0 < cy < H


def _to_circle(x, y, ratio, phi):
    c, s = math.cos(phi), math.sin(phi)
    return x * c + y * s, (-x * s + y * c) / ratio


def _from_circle(u, v, ratio, phi):
    v = v * ratio
    c, s = math.cos(phi), math.sin(phi)
    return u * c - v * s, u * s + v * c


def _circle3(u, v):
    """Circles through point triples (vectorised): u, v [k, 3] -> centres [k, 2], radii [k]."""
    ax, ay, bx, by, cx, cy = u[:, 0], v[:, 0], u[:, 1], v[:, 1], u[:, 2], v[:, 2]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    d = np.where(np.abs(d) < 1e-9, np.nan, d)
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return np.stack([ux, uy], 1), np.hypot(ax - ux, ay - uy)


def _circle_lsq(u, v):
    A = np.stack([u, v, np.ones_like(u)], 1)
    sol, *_ = np.linalg.lstsq(A, u * u + v * v, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    return cx, cy, math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))


def ransac_free(x, y, shape, rng, n_iter=300, tol=2.0, wt=None):
    n = len(x)
    wt = np.ones(n) if wt is None else wt
    if n < 20:
        return None
    best, best_s = None, -1
    for _ in range(n_iter):
        idx = rng.choice(n, 6, replace=False)
        p = fit_direct(x[idx], y[idx])
        if not plausible(p, shape):
            continue
        s = float(wt[np.abs(ell_dist(p, x, y)) < tol].sum())
        if s > best_s:
            best, best_s = p, s
    for _ in range(3):
        if best is None:
            break
        inl = np.abs(ell_dist(best, x, y)) < 1.5 * tol
        q = fit_direct(x[inl], y[inl])
        if not plausible(q, shape):
            break
        best = q
    return best


def ransac_shape(x, y, ratio, phi, rng, n_iter=600, tol=2.0, window=None, wt=None):
    """Ellipse with fixed axis ratio b/a and orientation phi (3-point circle RANSAC in the de-squashed plane).
    window = (cx, cy, a, dc, da): accept only centres within dc [px] and a within a factor (1 +- da)."""
    n = len(x)
    if n < 12:
        return None
    wt = np.ones(n) if wt is None else wt
    u, v = _to_circle(x, y, ratio, phi)
    idx = rng.integers(0, n, size=(n_iter, 3))
    C, R = _circle3(u[idx], v[idx])
    ok = np.isfinite(R) & (R > 45)
    if window is not None:
        wc = np.array(_to_circle(window[0], window[1], ratio, phi))
        ok &= (np.hypot(*(C - wc).T) < window[3] / ratio) & (np.abs(R / window[2] - 1) < window[4])
    if not ok.any():
        return None
    C, R = C[ok], R[ok]
    # score in chunks (distance in the de-squashed plane ~ distance along the minor axis / ratio; use tol/ratio)
    best, best_s = None, -1
    for k0 in range(0, len(R), 200):
        d = np.abs(np.hypot(u[None, :] - C[k0:k0 + 200, 0:1], v[None, :] - C[k0:k0 + 200, 1:2]) - R[k0:k0 + 200, None])
        s = ((d < tol) * wt[None, :]).sum(1)
        j = int(np.argmax(s))
        if s[j] > best_s:
            best_s, best = s[j], (C[k0 + j, 0], C[k0 + j, 1], R[k0 + j])
    cu, cv, r = best
    for _ in range(3):
        inl = np.abs(np.hypot(u - cu, v - cv) - r) < 1.5 * tol
        if inl.sum() < 8:
            break
        cu, cv, r = _circle_lsq(u[inl], v[inl])
    cx, cy = _from_circle(cu, cv, ratio, phi)
    return canon((cx, cy, r, r * ratio, phi))


# =========================================================================== pixel classes (image space)
def pixel_classes(rgb):
    """rgb float [H, W, 3] in [0, 1] -> dict of boolean masks + hsv."""
    hsv = rgb_to_hsv(np.clip(rgb, 0, 1))
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    brown = (H > 0.02) & (H < 0.13) & (S > 0.3) & (V > 0.25)              # crema (targets.liquid_ellipse thresholds)
    red = (r - g > 0.4) & (r - b > 0.4) & (r > 0.55)                          # instructor's overlay dots
    pure = (S < 0.07) & (V > 0.9)                                             # overlay lines, subtitles, glare (and clean foam)
    return dict(brown=brown, red=red, pure=pure, H=H, S=S, V=V)


def overlay_lines(cls, min_len=60):
    """The instructor's overlay: red dots + thin pure-white straight lines through them (only when >= 2 red dots)."""
    red = ndi.binary_opening(cls["red"], iterations=1)
    lab, n = ndi.label(red)
    out = np.zeros_like(red)
    if n == 0:
        return out
    sizes = ndi.sum(red, lab, range(1, n + 1))
    keep = [k + 1 for k in range(n) if 15 <= sizes[k] <= 900]
    if len(keep) < 2:
        return out
    dots = np.array(ndi.center_of_mass(red, lab, keep))            # (row, col)
    out = ndi.binary_dilation(np.isin(lab, keep), iterations=10)    # dots + the B/C/D letters next to them
    # thin white lines: pure-white pixels that are thin (removed by an opening) and long
    pure = cls["pure"]
    thin = pure & ~ndi.binary_opening(pure, iterations=3)
    lab2, n2 = ndi.label(ndi.binary_dilation(thin, iterations=1))
    if n2:
        objs = ndi.find_objects(lab2)
        for k, sl in enumerate(objs):
            ext = math.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
            if ext >= min_len:
                out |= lab2 == k + 1
    return ndi.binary_dilation(out, iterations=2)


# =========================================================================== per-frame surface boundary
def surface_component(cls, prior=None, overlay=None):
    """Crema region of the liquid surface (holes filled) -> bool mask or None.  With a prior ellipse, all crema
    components lying mostly inside the (slightly enlarged) prior are kept; otherwise the largest component that
    does not touch the bottom image border (the hand)."""
    brown = cls["brown"].copy()
    if overlay is not None:
        brown |= overlay & ndi.binary_dilation(brown, iterations=6)
    brown = ndi.binary_opening(brown, iterations=2)
    lab, n = ndi.label(brown)
    if n == 0:
        return None
    Hh, W = brown.shape
    sizes = ndi.sum(brown, lab, range(1, n + 1))
    objs = ndi.find_objects(lab)
    if prior is not None:
        yy, xx = np.mgrid[:Hh, :W]
        inside = ell_rho(prior, xx, yy) < 1.08
        frac = ndi.sum(inside, lab, range(1, n + 1)) / np.maximum(sizes, 1)
        keep = [k + 1 for k in range(n) if sizes[k] >= 150 and frac[k] > 0.6]
        if not keep:
            return None
        comp = np.isin(lab, keep)
    else:
        cand = []
        for k in range(n):
            if sizes[k] < 300:
                continue
            bottom = objs[k][0].stop >= Hh - 2
            cand.append((sizes[k] * (0.2 if bottom else 1.0), k + 1))
        if not cand:
            return None
        comp = lab == max(cand)[1]
    return ndi.binary_fill_holes(ndi.binary_closing(comp, iterations=3))


def boundary_candidates(comp, cls, max_pts=700):
    """Rim candidates of the crema region: edge pixels on (within 2.5 px of) the convex hull -- concave notches cut by
    the pitcher drop out -- plus the lower envelope (lowest crema pixel per image column: the near rim, never
    behind the pitcher).  Points whose outside (4-8 px along the outward normal) is not porcelain-like (bright, low
    saturation) are dropped.  Returns x, y, weight (2 for lower-envelope points, 1 otherwise)."""
    edge = comp & ~ndi.binary_erosion(comp)
    iy, ix = np.nonzero(edge)
    if len(ix) < 20:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    pts = np.stack([ix, iy], 1).astype(float)
    try:
        hull = ConvexHull(pts)
        eq = hull.equations
        on_hull = (-(pts @ eq[:, :2].T + eq[:, 2])).min(1) < 2.5
    except Exception:  # noqa
        on_hull = np.ones(len(pts), bool)
    low = np.zeros(len(pts), bool)
    cols = np.full(comp.shape[1], -1)
    np.maximum.at(cols, ix, iy)
    low = iy == cols[ix]
    keep = on_hull | low
    pts, low = pts[keep], low[keep]
    # outward normal from the smoothed mask
    sm = ndi.gaussian_filter(comp.astype(np.float32), 2.0)
    gy, gx = np.gradient(sm)
    px, py = pts[:, 0].astype(int), pts[:, 1].astype(int)
    nx, ny = -gx[py, px], -gy[py, px]
    nn = np.hypot(nx, ny) + 1e-9
    nx, ny = nx / nn, ny / nn
    H, W = comp.shape
    S, V = cls["S"], cls["V"]
    good = np.zeros(len(pts))
    for k in (4, 6, 8):
        qx = np.clip(np.round(pts[:, 0] + k * nx).astype(int), 0, W - 1)
        qy = np.clip(np.round(pts[:, 1] + k * ny).astype(int), 0, H - 1)
        good += (V[qy, qx] > 0.45) & (S[qy, qx] < 0.3)
    keep = good >= 2
    pts, low = pts[keep], low[keep]
    if len(pts) > max_pts:
        sel = np.linspace(0, len(pts) - 1, max_pts).astype(int)
        pts, low = pts[sel], low[sel]
    return pts[:, 0], pts[:, 1], np.where(low, 2.0, 1.0)


def frame_fit(rgb, rng, prior=None, shape=None, window=None):
    """One frame -> dict(p, cov, inl, n, area) (p None if nothing plausible)."""
    cls = pixel_classes(rgb)
    ov = overlay_lines(cls)
    comp = surface_component(cls, prior, ov)
    if comp is None:
        return dict(p=None, cov=0.0, inl=0.0, n=0)
    x, y, wt = boundary_candidates(comp, cls)
    if len(x) < 20:
        return dict(p=None, cov=0.0, inl=0.0, n=len(x))
    tol = 2.0
    if shape is None:
        p = ransac_free(x, y, rgb.shape[:2], rng, tol=tol, wt=wt)
    else:
        p = ransac_shape(x, y, shape[0], shape[1], rng, tol=tol, window=window, wt=wt)
    if p is None:
        return dict(p=None, cov=0.0, inl=0.0, n=len(x))
    inl = np.abs(ell_dist(p, x, y)) < 1.5 * tol + 0.01 * p[2]
    # crema found outside the fitted surface on the near (image-lower) side, where the pitcher never is: fraction of
    # the near half of the crema region beyond rho 1.06 (would mean the ellipse is too small or misplaced)
    yy, xx = np.nonzero(comp)
    xx, yy = xx.astype(float), yy.astype(float)
    ex, ey = basis(p)
    dn = (xx - p[0]) * ey[0] + (yy - p[1]) * ey[1]           # along e_y (image-down): > 0 = near half
    near = dn > 0
    out = float((ell_rho(p, xx[near], yy[near]) > 1.06).mean()) if near.any() else 1.0
    return dict(p=p, cov=coverage(p, x[inl], y[inl]), inl=float(inl.mean()), n=len(x), out=out)


# =========================================================================== temporal track
def _robust_smooth(t, y, w, sigma, it=3, c=4.0, scale=None):
    """Kernel-weighted local-linear smoother with Tukey re-weighting; w = prior weights (0 = missing)."""
    t = np.asarray(t, float); y = np.asarray(y, float); w0 = np.asarray(w, float).copy()
    good = (w0 > 0) & np.isfinite(y)
    yy = np.where(good, y, 0.0)
    rw = np.ones_like(y)
    out = np.full_like(y, np.nan)
    for _ in range(it):
        ww = np.where(good, w0 * rw, 0.0)
        for i in range(len(t)):
            k = np.exp(-0.5 * ((t - t[i]) / sigma) ** 2) * ww
            sk = k.sum()
            if sk < 1e-6:
                continue
            tm = (k * t).sum() / sk
            ym = (k * yy).sum() / sk
            var = (k * (t - tm) ** 2).sum() / sk
            slope = (k * (t - tm) * (yy - ym)).sum() / sk / var if var > 1e-9 else 0.0
            out[i] = ym + slope * (t[i] - tm)
        res = np.where(good, y - out, 0.0)
        s = scale if scale is not None else 1.4826 * np.median(np.abs(res[good])) + 1e-9
        u = res / (c * s)
        rw = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
    if np.isnan(out).any():                       # extrapolate flat outside the supported range
        ok = ~np.isnan(out)
        if ok.any():
            out = np.interp(t, t[ok], out[ok])
    return out


def phase_shift(A, B):
    """Translation d (dx, dy) [px] such that B(x) ~ A(x - d) (phase correlation, sub-pixel by parabola fit)."""
    FA, FB = np.fft.fft2(A), np.fft.fft2(B)
    R = FB * np.conj(FA)
    R /= np.abs(R) + 1e-9
    r = np.fft.ifft2(R).real
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    H, W = r.shape

    def sub(c, m, p):
        den = m - 2 * c + p
        return 0.0 if abs(den) < 1e-12 else 0.5 * (m - p) / den
    dy = iy + sub(r[iy, ix], r[(iy - 1) % H, ix], r[(iy + 1) % H, ix])
    dx = ix + sub(r[iy, ix], r[iy, (ix - 1) % W], r[iy, (ix + 1) % W])
    if dy > H / 2:
        dy -= H
    if dx > W / 2:
        dx -= W
    return dx, dy, float(r.max())


def _grad_img(rgb_u8, box):
    x0, x1, y0, y1 = box
    g = rgb_u8[y0:y1, x0:x1].astype(np.float32).mean(-1)
    g = ndi.gaussian_filter(g, 1.5)
    gm = np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1))
    wy = np.hanning(gm.shape[0])[:, None]; wx = np.hanning(gm.shape[1])[None, :]
    return gm * wy * wx


def cup_box(p, shape):
    """Image box around the cup body below/around the surface (the pitcher is above it)."""
    cx, cy, a, b, _ = p
    H, W = shape
    x0, x1 = int(max(0, cx - 1.35 * a)), int(min(W, cx + 1.35 * a))
    y0, y1 = int(max(0, cy - 0.3 * b)), int(min(H, cy + 2.4 * b))
    return x0, x1, y0, y1


def propagate_track(frames, P, quality, log=None):
    """Fill frames of quality 0 by following the cup's image motion (phase correlation of the cup-body region between
    consecutive frames) from the nearest good frames; returns the new P and a flag array (True = propagated)."""
    n = len(frames)
    P = P.copy()
    prop = np.zeros(n, bool)
    good = quality > 0
    if good.all() or not good.any():
        return P, prop
    shp = frames.shape[1:3]

    def walk(i_from, i_to):
        """Propagate P[i_from] step by step to i_to; returns dict i -> p."""
        out = {}
        p = P[i_from].copy()
        step = 1 if i_to > i_from else -1
        for i in range(i_from + step, i_to + step, step):
            box = cup_box(p, shp)
            if box[1] - box[0] < 32 or box[3] - box[2] < 32:
                dx = dy = 0.0
            else:
                dx, dy, _ = phase_shift(_grad_img(frames[i - step], box), _grad_img(frames[i], box))
            if math.hypot(dx, dy) > 0.25 * p[2]:          # implausible jump (scene change): stop following
                dx = dy = 0.0
            p = p.copy(); p[0] += dx; p[1] += dy
            out[i] = p
        return out

    i = 0
    while i < n:
        if good[i]:
            i += 1
            continue
        j = i
        while j < n and not good[j]:
            j += 1
        left, right = i - 1, j            # good neighbours (may be out of range)
        fw = walk(left, j - 1) if left >= 0 else None
        bw = walk(right, i) if right < n else None
        for k in range(i, j):
            if fw is not None and bw is not None:
                w = (k - left) / (right - left)
                P[k] = (1 - w) * fw[k] + w * bw[k]
            else:
                P[k] = (fw if fw is not None else bw)[k]
            prop[k] = True
        i = j
    if log:
        log(f"propagated {int(prop.sum())} frames by cup-motion registration")
    return P, prop


def scene_cuts(frames, thr=45.0):
    """Indices i where frame i is a different shot from frame i-1 (mean absolute difference > thr grey levels)."""
    d = np.array([np.abs(frames[i].astype(np.int16)[::4, ::4] - frames[i - 1].astype(np.int16)[::4, ::4]).mean()
                  for i in range(1, len(frames))])
    return [i + 1 for i in np.nonzero(d > thr)[0]]


def track_ellipses(frames, fps, ref_index=None, seed=0, log=None):
    """Fit the liquid-surface ellipse in every frame.
    pass 1  free RANSAC fits; frames with >= 80 % angular coverage are anchors for the axis ratio and orientation
            (smoothed over 1 s, held constant outside the anchored range);
    pass 2  shape-constrained fits (3-point circle RANSAC in the de-squashed plane) inside a window around the
            anchor track; tier A = well supported (coverage >= 0.45, no crema outside on the near side) and consistent
            with the smooth tier-A track (4 % of a); frames far from tier A follow the cup's image motion (phase
            correlation of the cup body); tier B = partial fits within 12 % of that track;
    final   robust local-linear smoothing (sigma 2 frames) of centre and size over tier A (weight = coverage) and
            tier B (0.3 x coverage), the propagated track filling the rest.
    quality 2 = tier-A fit within 0.2 s, 1 = tier-B fit within 0.2 s, 0 = propagated only (tilt phase / heavy
            occlusion: geometry approximate), -1 = no cup (after a scene cut that follows ref_index).
    Returns dict: t, p [n, 5] (cx, cy, a, b, phi; px/rad, targets convention), p_raw, p_free, cov, inl, out, rel,
    approx, anchor, quality, ratio, cuts."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    t = np.arange(n) / fps
    shp = frames.shape[1:3]
    # pass 1: free fits
    P1 = np.full((n, 5), np.nan); cov1 = np.zeros(n); out1 = np.ones(n)
    for i in range(n):
        f = frame_fit(frames[i].astype(np.float64) / 255.0, rng)
        if f["p"] is not None:
            P1[i] = f["p"]; cov1[i] = f["cov"]; out1[i] = f["out"]
    anchor = (cov1 >= 0.8) & (out1 < 0.02)
    if anchor.sum() < 3:
        anchor = (cov1 >= 0.6) & (out1 < 0.05)
    if anchor.sum() < 3:
        raise RuntimeError("no clean frames to anchor the ellipse shape")
    # shape (axis ratio, orientation): smooth over anchors, flat extrapolation; sigma 1 s
    ratio_s = _robust_smooth(t, P1[:, 3] / P1[:, 2], anchor.astype(float), 1.0)
    phi_u = np.unwrap(np.where(np.isfinite(P1[:, 4]), P1[:, 4], 0.0))
    phi_s = _robust_smooth(t, phi_u, anchor.astype(float), 1.0)
    # preliminary centre/size track from anchors (used only as a loose window)
    pre = np.stack([_robust_smooth(t, P1[:, k], anchor.astype(float), 0.5) for k in (0, 1, 2)], 1)
    # distance (in time) to the nearest anchor: the window is loose far from anchors
    ia = np.nonzero(anchor)[0]
    dta = np.array([np.min(np.abs(t[ia] - ti)) for ti in t])
    # pass 2: shape-constrained fits with a window around the preliminary track
    P2 = np.full((n, 5), np.nan); cov2 = np.zeros(n); inl2 = np.zeros(n); out2 = np.ones(n)
    for i in range(n):
        rgb = frames[i].astype(np.float64) / 255.0
        loose = dta[i] > 1.0
        win = (pre[i, 0], pre[i, 1], pre[i, 2], (0.6 if loose else 0.25) * pre[i, 2], 0.45 if loose else 0.15)
        prior = canon((pre[i, 0], pre[i, 1], pre[i, 2] * (1.25 if loose else 1.08), pre[i, 2] * ratio_s[i] * (1.25 if loose else 1.08), phi_s[i]))
        f = frame_fit(rgb, rng, prior=prior, shape=(ratio_s[i], phi_s[i]), window=win)
        if f["p"] is None:
            f = frame_fit(rgb, rng, prior=None, shape=(ratio_s[i], phi_s[i]), window=win)
        if f["p"] is not None:
            P2[i] = f["p"]; cov2[i] = f["cov"]; inl2[i] = f["inl"]; out2[i] = f["out"]
    fin = np.isfinite(P2[:, 0])
    okA = fin & (cov2 >= 0.45) & (out2 < 0.05)          # well-supported fit
    okB = fin & (cov2 >= 0.2) & (out2 < 0.12)           # partial support (tilt phase, heavy occlusion)
    sig = 2.0 / fps
    phi_c = np.array([canon((0, 0, 1, 0.5, ph))[4] for ph in phi_s])

    def smooth(w, extra=None):
        Y = P2.copy()
        if extra is not None:                           # pseudo-observations where there is no fit
            m = ~np.isfinite(Y[:, 0]) | (w <= 0)
            Y[m] = extra[m]
            w = np.where(m, 0.05, w)
        return [_robust_smooth(t, Y[:, k], w, sig) for k in (0, 1, 2)]

    def resid(Q):
        return (np.hypot(P2[:, 0] - Q[:, 0], P2[:, 1] - Q[:, 1]) + np.abs(P2[:, 2] - Q[:, 2])) / Q[:, 2]

    def gap(mask):
        idx = np.nonzero(mask)[0]
        return np.array([np.min(np.abs(t[idx] - ti)) if len(idx) else 99.0 for ti in t])

    def assemble(cx, cy, a):
        return np.stack([cx, cy, a, a * ratio_s, phi_c], 1)

    # tier A: well supported and consistent with the smooth track of tier A
    PA = assemble(*smooth(np.where(okA, cov2, 0.0)))
    relA = okA & (resid(PA) < 0.04)
    PA = assemble(*smooth(np.where(relA, cov2, 0.0)))
    # frames far from tier A: follow the cup's image motion from the tier-A frames
    PA, _ = propagate_track(frames, PA, (gap(relA) <= 0.2).astype(int), log=log)
    # tier B: partial fits that agree with the propagated track
    relB = okB & ~relA & (resid(PA) < 0.12)
    w = np.where(relA, cov2, np.where(relB, 0.3 * cov2, 0.0))
    P = assemble(*smooth(w, extra=PA))
    gA, gB = gap(relA), gap(relA | relB)
    quality = np.where(gA <= 0.2, 2, np.where(gB <= 0.2, 1, 0))
    # after a scene cut (outro, cut-away) with no reliable fit: no cup
    cuts = scene_cuts(frames)
    for c in cuts:
        if (ref_index is not None and c > ref_index) or not relA[c:].any():
            quality[c:] = -1
            break
    if log:
        log(f"ellipse track: {n} frames, anchors {int(anchor.sum())}, reliable {int(relA.sum())}, approx {int(relB.sum())}, "
            f"ratio {np.median(ratio_s):.3f}, quality 2/1/0/-1 = {(quality == 2).sum()}/{(quality == 1).sum()}/"
            f"{(quality == 0).sum()}/{(quality == -1).sum()}, scene cuts {cuts}")
    return dict(t=t, p_raw=P2, p_free=P1, p=P, cov=cov2, inl=inl2, out=out2, rel=relA, approx=relB, anchor=anchor,
                quality=quality, ratio=ratio_s, cuts=cuts)


# =========================================================================== rectification (targets.warp convention)
def cup_grid(M):
    g = (np.arange(M) + 0.5) / M - 0.5
    X, Y = np.meshgrid(g, g)
    return X, Y


def sample_coords(p, M, rot=0.0, flip=False):
    """Image coordinates (px, py) [M, M] of the T-frame cells (same formula as targets.warp)."""
    X, Y = cup_grid(M)
    if flip:
        X = -X
    th = math.radians(rot)
    Xr = math.cos(th) * X - math.sin(th) * Y
    Yr = math.sin(th) * X + math.cos(th) * Y
    u, v = 2 * Xr, -2 * Yr
    ex, ey = basis(p)
    cx, cy, a, b = p[:4]
    return cx + a * u * ex[0] + b * v * ey[0], cy + a * u * ex[1] + b * v * ey[1]


def image_to_cup(p, px, py, rot=0.0, flip=False):
    """Inverse of sample_coords: image point(s) -> T-frame (X, Y) [cup diameters]."""
    ex, ey = basis(p)
    dx, dy = np.asarray(px, float) - p[0], np.asarray(py, float) - p[1]
    u = (dx * ex[0] + dy * ex[1]) / p[2]
    v = (dx * ey[0] + dy * ey[1]) / p[3]
    Xr, Yr = u / 2, -v / 2
    th = math.radians(rot)
    X = math.cos(th) * Xr + math.sin(th) * Yr
    Y = -math.sin(th) * Xr + math.cos(th) * Yr
    return (-X if flip else X), Y


def rectify(img, p, M, rot=0.0, flip=False, order=1):
    """Sample an image ([H, W] or [H, W, C]) on the T-frame grid."""
    px, py = sample_coords(p, M, rot, flip)
    if img.ndim == 2:
        return ndi.map_coordinates(img, [py, px], order=order, mode="nearest")
    return np.stack([ndi.map_coordinates(img[..., k], [py, px], order=order, mode="nearest") for k in range(img.shape[2])], -1)


def disk_mask(M, R=0.49):
    X, Y = cup_grid(M)
    return np.hypot(X, Y) <= R


# =========================================================================== occluders / overlays (image space)
def static_text_mask(frames, idx, thr_frac=0.6):
    """Watermark / logo text that stays at a fixed image position: thin bright strokes (white top-hat) present in more
    than thr_frac of the sampled frames.  The cup moves in the frame, so real foam stripes do not stay put."""
    cnt = None
    for i in idx:
        rgb = frames[i].astype(np.float32) / 255.0
        V = rgb.max(-1)
        S = (V - rgb.min(-1)) / (V + 1e-6)
        th = V - ndi.grey_opening(V, size=(7, 7))
        m = (th > 0.10) & (S < 0.15) & (V > 0.75)
        cnt = m.astype(np.int32) if cnt is None else cnt + m
    st = cnt >= thr_frac * len(idx)
    return ndi.binary_dilation(ndi.binary_closing(st, iterations=2), iterations=2)


def surface_validity(rgb, cls, p, rim_barrier=True, extra_barrier=None):
    """Visible liquid surface inside the ellipse (image space).  Returns (valid, occ, stream, tip):
      occ    pitcher + milk stream: non-surface-coloured regions (dark, grey steel, crema-hued but darker than this
             frame's crema) crossing the rim band, with the regions they enclose against the far rim filled in
             (highlights, milk on the lip), plus the thin bright stream hanging from them;
      valid  inside the ellipse, not occluded, and part of the liquid (enclosed by crema / occluder / near rim):
             the porcelain inner wall that shows inside the ellipse while the cup is tilted is invalid;
      tip    lowest point of the stream (else of the pitcher) along e_y, image px: the stream foot (impact) while the
             stream lands on crema; only approximate when it lands inside white (the occluded strip is depth-limited)."""
    H, W = rgb.shape[:2]
    cx, cy, a, b, _ = p
    x0, x1 = int(max(0, cx - 1.3 * a)), int(min(W, cx + 1.3 * a + 1))
    y0, y1 = int(max(0, cy - 1.3 * b)), int(min(H, cy + 1.3 * b + 1))
    valid_full = np.zeros((H, W), bool)
    occ_full = np.zeros((H, W), bool)
    stream_full = np.zeros((H, W), bool)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return valid_full, occ_full, stream_full, None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    rho = ell_rho(p, xx, yy)
    ex, ey = basis(p)
    dn = ((xx - cx) * ey[0] + (yy - cy) * ey[1]) / b                # > 0 near (image-lower) half
    S, V = cls["S"][y0:y1, x0:x1], cls["V"][y0:y1, x0:x1]
    brown = cls["brown"][y0:y1, x0:x1]
    r, bb = rgb[y0:y1, x0:x1, 0], rgb[y0:y1, x0:x1, 2]
    inside = rho < 1.0
    ref = inside & (dn > 0.3) & (rho > 0.6) & brown                  # crema surely on the surface (near annulus)
    if ref.sum() > 50:
        Vc = V[ref]
        v_lo, v_med = np.quantile(Vc, 0.05), np.median(Vc)
    else:
        v_lo, v_med = 0.45, 0.6
    crema = brown & (V > 0.88 * v_lo)
    foam = (S < 0.32) & (V > max(0.55, 1.02 * v_med)) & (r >= bb - 0.02)
    core = ndi.binary_opening(~(crema | foam) & inside, iterations=1)
    lab, n = ndi.label(core)
    occ = np.zeros_like(core)
    if n:
        band = inside & (rho > 0.93)
        touch = ndi.maximum(band, lab, range(1, n + 1))
        sizes = ndi.sum(core, lab, range(1, n + 1))
        occ = np.isin(lab, [k + 1 for k in range(n) if touch[k] and sizes[k] >= 60])
    # pitcher + milk sheet on the lip + falling stream: everything that hangs from the far rim.  In every image column
    # crossing the far rim inside a light/non-crema region (or the pitcher core), the contiguous run of non-crema
    # pixels going down from the rim is occluded (the pitcher and the stream are seen in front of the surface and come
    # from above); the run is cut 0.35 D below the lowest pitcher-core pixel of that column, so the stream landing in
    # a white disc hides only a strip of it.
    ext = rho < 1.12
    stream = np.zeros_like(occ)
    if ext.any():
        ys = np.arange(ext.shape[0])[:, None]
        start = np.where(ext.any(0), np.argmax(ext, axis=0), ext.shape[0])
        C = (~crema | occ) & ext
        run = np.cumprod(np.where(ys < start[None, :], True, C), axis=0).astype(bool) & inside & (ys >= start[None, :])
        low = np.where(occ.any(0), (occ * ys).max(0), -1)
        first = np.where(run.any(0), np.argmax(run, axis=0), ext.shape[0])
        ref_row = np.where(low >= 0, np.maximum(low, first), first)
        run &= ys <= ref_row[None, :] + 0.7 * b
        # slanted stream: light pixels in narrow horizontal runs (< 0.12 D) connected to the vertical run / pitcher
        hl = foam & inside & ~run
        lab_r, nr = ndi.label(hl, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))
        if nr:
            rl = ndi.sum(hl, lab_r, range(1, nr + 1))
            thin = hl & (np.concatenate([[0], rl])[lab_r] <= 0.24 * a)
            seed = ndi.binary_dilation(run | occ, iterations=2) & thin
            if seed.any():
                ext_run = ndi.binary_propagation(seed, mask=thin, structure=np.ones((3, 3), bool))
                ext_run &= ys <= ref_row[None, :] + 0.7 * b
                run |= ext_run
        stream = run & ~occ
        occ = occ | run
    occ_all = ndi.binary_dilation(occ | stream, iterations=2) & inside
    # liquid: enclosed by crema, occluder and the near rim
    barrier = crema | occ_all
    if extra_barrier is not None:          # overlay lines crossing the rim must not connect foam to the porcelain
        barrier = barrier | extra_barrier[y0:y1, x0:x1]
    if rim_barrier:          # reliable ellipse: white touching the near rim (stem) is still surface
        barrier = barrier | ((rho >= 1.0) & (rho < 1.06) & (dn > 0.2))
    liquid = ndi.binary_fill_holes(barrier) & inside
    valid = inside & liquid & ~occ_all
    valid_full[y0:y1, x0:x1] = valid
    occ_full[y0:y1, x0:x1] = occ_all
    stream_full[y0:y1, x0:x1] = stream
    tip = None
    sel = stream & (rho < 0.98)
    if not sel.any():
        sel = occ_all & (rho < 0.98)
    if sel.any():
        k = int(np.argmax(np.where(sel, dn, -9.0)))
        tip = (float(xx.ravel()[k]), float(yy.ravel()[k]))
    return valid_full, occ_full, stream_full, tip


# =========================================================================== whiteness
def foam_index(rgb_w):
    hsv = rgb_to_hsv(np.clip(rgb_w, 0, 1))
    return (1 - hsv[..., 1]) * hsv[..., 2], hsv


def reference_levels(rgb_w, disk, method="otsu2"):
    """Foam/crema levels of the clean reference frame, computed exactly as targets.py does."""
    if method == "foam3":
        w, lo, _, hi = TG.whiteness_foam3(rgb_w, disk)
    else:
        w, _, lo, hi = TG.whiteness(rgb_w, disk)
    return float(lo), float(hi), w


def crema_level(f, hsv, valid):
    """Median foam index of clearly crema pixels (saturated, crema hue) among the valid ones."""
    H, S = hsv[..., 0], hsv[..., 1]
    sel = valid & (S > 0.3) & (H > 0.02) & (H < 0.14)
    return float(np.median(f[sel])) if sel.sum() > 200 else np.nan


def _otsu(vals, nbins=128):
    hist, edges = np.histogram(vals, nbins)
    mids = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = np.roll(np.cumsum((hist * mids)[::-1])[::-1] / np.maximum(w1, 1), -1)
    var = w0[:-1] * w1[:-1] * (m0[:-1] - m1[:-1]) ** 2
    return mids[int(np.argmax(var))]


def frame_levels(f, hsv, valid, method="otsu2", min_sep=0.25, min_frac=0.003):
    """This frame's own crema/foam levels (lighting and white balance change a lot while the pitcher is in view):
    two-class Otsu split of f (three-class k-means for 'foam3', as targets.py) over the valid pixels, accepted only if
    the classes are well separated (>= min_sep), the bright class covers >= min_frac and is clearly less saturated
    than the dark one (foam, not light crema).  Returns (ok, lo, hi)."""
    vals = f[valid]
    if len(vals) < 500:
        return False, np.nan, np.nan
    S = hsv[..., 1][valid]
    if method == "foam3":
        c = TG._kmeans3(vals)
        lo, hi = c[1], c[2]
        up = vals >= 0.5 * (c[1] + c[2])
    else:
        thr = _otsu(vals)
        up = vals >= thr
        if not up.any() or up.all():
            return False, np.nan, np.nan
        lo, hi = np.median(vals[~up]), np.median(vals[up])
    if not up.any() or up.all():
        return False, np.nan, np.nan
    ok = (hi - lo >= min_sep) and up.mean() >= min_frac and np.median(S[up]) < 0.75 * np.median(S[~up])
    return bool(ok), float(lo), float(hi)


def whiteness_frame(f, lo, hi):
    return np.clip((f - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def clean_rim_blobs(w, valid, min_blob=0.01):
    """White blobs touching the rim that are smaller than min_blob of the cup (reflections) -> invalid."""
    M = w.shape[0]
    X, Y = cup_grid(M)
    rim = np.hypot(X, Y) > 0.46
    lab, n = ndi.label((w > 0.5) & valid)
    bad = np.zeros_like(valid)
    if n:
        sizes = ndi.sum(np.ones_like(w), lab, range(1, n + 1))
        touch = ndi.maximum(rim, lab, range(1, n + 1))
        tot = valid.sum()
        for k in range(n):
            if touch[k] and sizes[k] < min_blob * tot:
                bad |= lab == k + 1
    return ndi.binary_dilation(bad, iterations=2)


# =========================================================================== extraction driver
def out_dir_of(name):
    return os.path.join(OUT_ROOT, name)


def extract(name, M=192, fps_out=5.0, t_end=None, out_dir=None, log=print, video=None):
    """Clip -> video_targets/<name>/frames.npz.  Output times are k / fps_out up to t_end (default: the targets.FRAMES
    reference time, whose frame is always included as the last one, so frames.npz[-1] is the target frame)."""
    cfg = TG.FRAMES[name]
    rot, flip, method = cfg["rot"], cfg["flip"], cfg.get("method", "otsu2")
    path = clip_path(name)
    frames, fps = video if video is not None else read_video(path)
    i_ref = int(math.ceil(cfg["t"] * fps - 1e-6))          # the frame ffmpeg -ss t returns (= targets/<name>_frame.png)
    t_end = cfg["t"] if t_end is None else t_end
    n_track = min(len(frames), max(i_ref, int(round(t_end * fps))) + int(fps) + 1)
    later = [c for c in scene_cuts(frames[:n_track]) if c > i_ref]
    if later:                                                  # outro / teaser of another pattern: not this cup
        n_track = later[0]
    tr = track_ellipses(frames[:n_track], fps, ref_index=i_ref, log=log)
    # anchor the track on the targets.py fit of the reference frame (clean frame), blended over +-0.3 s, so that the
    # last frame of frames.npz is rectified exactly as targets/<name>.npz
    try:
        e = TG.liquid_ellipse(frames[i_ref].astype(np.float64) / 255.0)
        ex0 = e["evec"][:, 0]
        p_tg = np.array(canon((e["center"][0], e["center"][1], e["axes"][0], e["axes"][1], math.atan2(ex0[1], ex0[0]))))
        dlt = p_tg - tr["p"][i_ref]
        dlt[4] = (dlt[4] + math.pi) % (2 * math.pi) - math.pi
        wgt = np.exp(-0.5 * ((tr["t"] - tr["t"][i_ref]) / 0.3) ** 2)
        tr["p"] = tr["p"] + wgt[:, None] * dlt[None, :]
        tr["quality"][i_ref] = 2
        log(f"reference frame {i_ref}: track anchored on the targets.py ellipse (shift {np.hypot(*dlt[:2]):.1f} px, "
            f"a {dlt[2]:+.1f} px, b {dlt[3]:+.1f} px, phi {math.degrees(dlt[4]):+.1f} deg)")
    except Exception as ex:  # noqa
        log(f"reference frame {i_ref}: targets.py ellipse failed ({ex}); track kept")
    # output frame indices
    idx = sorted(set(int(round(k * fps / fps_out)) for k in range(int(t_end * fps_out + 1e-6) + 1)) | {i_ref})
    idx = [i for i in idx if i < n_track and tr["quality"][i] >= 0]
    disk = disk_mask(M)
    # reference levels from the target frame
    p_ref = tr["p"][i_ref]
    rgbw_ref = rectify(frames[i_ref].astype(np.float64) / 255.0, p_ref, M, rot, flip)
    lo_ref, hi_ref, w_tg = reference_levels(rgbw_ref, disk, method)
    f_ref, hsv_ref = foam_index(rgbw_ref)
    crema_ref = crema_level(f_ref, hsv_ref, disk)
    stat = static_text_mask(frames, list(range(0, n_track, 5)))
    n = len(idx)
    F = np.zeros((n, M, M), np.float32)
    VAL = np.zeros((n, M, M), bool)
    crema = np.full(n, np.nan)
    lev_ok = np.zeros(n, bool); lev_lo = np.full(n, np.nan); lev_hi = np.full(n, np.nan)
    tip = np.full((n, 2), np.nan)
    occ_frac = np.zeros(n)
    stream_px = np.zeros(n)
    ov_any = np.zeros(n, bool)
    for j, i in enumerate(idx):
        rgb = frames[i].astype(np.float64) / 255.0
        cls = pixel_classes(rgb)
        p = tr["p"][i]
        ov = overlay_lines(cls)
        vis, occ, stream, tp = surface_validity(rgb, cls, p, rim_barrier=bool(tr["quality"][i] == 2),
                                                extra_barrier=ov | stat)
        glare = ndi.binary_dilation(frames[i].min(-1) >= 250, iterations=1)
        bad = ~vis | ov | stat | glare
        rgbw = rectify(rgb, p, M, rot, flip)
        inval = rectify(bad.astype(np.float32), p, M, rot, flip, order=1) > 0.25
        valid = disk & ~inval
        f, hsv = foam_index(rgbw)
        F[j], VAL[j] = f, valid
        crema[j] = crema_level(f, hsv, valid)
        lev_ok[j], lev_lo[j], lev_hi[j] = frame_levels(f, hsv, valid, method)
        if tp is not None:
            tip[j] = image_to_cup(p, tp[0], tp[1], rot, flip)
        occ_frac[j] = 1.0 - valid.sum() / disk.sum()
        stream_px[j] = stream.sum()
        ov_any[j] = ov.any()
    # levels per frame.  The foam is creamier while it is being poured and whitens afterwards, and the light changes
    # while the pitcher is in view, so a fixed (reference-frame) level misses the stripes of a pour.  Each frame gets
    # its own foam/crema split where both phases are clearly present (frame_levels); the threshold thr and
    # half-contrast h are then smoothed in time (running median over 5 frames), frames without an accepted split take
    # them from accepted frames within 0.6 s, else thr = crema level + the clip's median (thr - crema) offset.
    # w = clip(0.5 + (f - thr) / (2 h)); the reference frame keeps exactly the targets.py levels.
    tt = np.array(idx) / fps
    thr_own = 0.5 * (lev_lo + lev_hi)
    h_own = 0.5 * (lev_hi - lev_lo)
    okl = lev_ok & np.isfinite(thr_own)
    jr = idx.index(i_ref)
    thr_ref, h_ref = 0.5 * (lo_ref + hi_ref), 0.5 * (hi_ref - lo_ref)
    okl[jr] = True; thr_own[jr], h_own[jr] = thr_ref, h_ref
    off = np.nanmedian((thr_own - crema)[okl]) if okl.any() else thr_ref - crema_ref
    h_med = np.nanmedian(h_own[okl]) if okl.any() else h_ref
    thr = np.full(n, np.nan); hh = np.full(n, np.nan)
    ia = np.nonzero(okl)[0]
    for j in range(n):
        near = ia[np.abs(tt[ia] - tt[j]) <= 0.6] if len(ia) else ia
        if len(near):
            k = near[np.argsort(np.abs(tt[near] - tt[j]))[:5]]
            thr[j], hh[j] = np.median(thr_own[k]), np.median(h_own[k])
        else:
            c = crema[j] if np.isfinite(crema[j]) else crema_ref
            thr[j], hh[j] = c + off, h_med
    hh = np.maximum(hh, 0.08)
    thr[jr], hh[jr] = thr_ref, h_ref
    lo, hi = thr - hh, thr + hh
    g = crema / crema_ref
    W = np.zeros((n, M, M), np.float32)
    for j in range(n):
        w = whiteness_frame(F[j], lo[j], hi[j])
        VAL[j] &= ~clean_rim_blobs(w, VAL[j])
        W[j] = np.where(VAL[j], w, 0.0)
    t = np.array(idx) / fps
    out_dir = out_dir or out_dir_of(name)
    os.makedirs(out_dir, exist_ok=True)
    res = dict(name=name, clip=path, t=t, frame_index=np.array(idx), white=W.astype(np.float16), valid=VAL, disk=disk,
               findex=np.where(VAL, F, 0.0).astype(np.float16),
               ell=tr["p"][idx], quality=tr["quality"][idx], gain=g, lo=lo, hi=hi, levels_own=lev_ok, occ_frac=occ_frac, tip=tip,
               stream_px=stream_px, overlay=ov_any, M=M, rot=rot, flip=flip, method=method, fps=fps, fps_out=fps_out,
               t_ref=cfg["t"], D_cup=D_CUP, track_t=tr["t"], track_ell=tr["p"], track_quality=tr["quality"],
               track_cov=tr["cov"], ref_white_targets_method=w_tg.astype(np.float32))
    np.savez_compressed(os.path.join(out_dir, "frames.npz"), **res)
    log(f"{name}: {n} frames at {fps_out} fps ({t[0]:.2f}-{t[-1]:.2f} s), M={M}, levels lo {lo_ref:.3f} hi {hi_ref:.3f} "
        f"crema_ref {crema_ref:.3f}, quality 2/1/0 = {(res['quality'] == 2).sum()}/{(res['quality'] == 1).sum()}/"
        f"{(res['quality'] == 0).sum()} -> {out_dir}/frames.npz")
    return res, frames, fps, tr


def load_frames(name, out_dir=None):
    z = np.load(os.path.join(out_dir or out_dir_of(name), "frames.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def compare_with_target(res):
    """IoU / L1 of the last frame (the reference frame) against targets/<name>.npz (resampled to M)."""
    w_t, _ = TG.load(res["name"] if isinstance(res["name"], str) else str(res["name"]))
    M = res["white"].shape[1]
    if w_t.shape[0] != M:
        w_t = np.clip(ndi.zoom(w_t, M / w_t.shape[0], order=1), 0, 1)
    disk = res["disk"]
    w = res["white"][-1].astype(np.float64)
    v = res["valid"][-1] & disk
    a, b = (w > 0.5) & v, (w_t > 0.5) & v
    iou = float((a & b).sum() / max((a | b).sum(), 1))
    l1 = float(np.abs(w - w_t)[v].mean())
    w2 = res["ref_white_targets_method"]
    a2 = (w2 > 0.5) & disk; b2 = (w_t > 0.5) & disk
    iou2 = float((a2 & b2).sum() / max((a2 | b2).sum(), 1))
    return dict(iou=iou, l1=l1, iou_targets_method=iou2, valid_frac=float(v.sum() / disk.sum()))


def preview_sheet(res, frames, path, n_cols=8, max_frames=32):
    """Contact sheet: per selected frame the camera image with the fitted ellipse (green good / cyan approx / magenta
    propagated), the rectified colour image with invalid pixels tinted red, and w (T frame, +y up)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(res["t"])
    sel = np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int))
    rows = (len(sel) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(3 * rows, n_cols, figsize=(2.0 * n_cols, 2.1 * 3 * rows), dpi=80, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    M = res["white"].shape[1]
    for k, j in enumerate(sel):
        r, c = 3 * (k // n_cols), k % n_cols
        i = int(res["frame_index"][j])
        p = res["ell"][j]
        rgb = frames[i]
        ax = axes[r, c]
        ax.imshow(rgb)
        th = np.linspace(0, 2 * np.pi, 120)
        cs, sn = math.cos(p[4]), math.sin(p[4])
        col = {2: "lime", 1: "cyan", 0: "magenta"}.get(int(res["quality"][j]), "red")
        ax.plot(p[0] + p[2] * np.cos(th) * cs - p[3] * np.sin(th) * sn, p[1] + p[2] * np.cos(th) * sn + p[3] * np.sin(th) * cs, color=col, lw=1)
        tp = res["tip"][j]
        ax.set_xlim(max(0, p[0] - 1.6 * p[2]), min(rgb.shape[1], p[0] + 1.6 * p[2]))
        ax.set_ylim(min(rgb.shape[0], p[1] + 2.2 * p[3]), max(0, p[1] - 2.6 * p[3]))
        ax.set_title(f"t={res['t'][j]:.1f}s q{int(res['quality'][j])}", fontsize=7)
        rgbw = rectify(rgb.astype(np.float64) / 255.0, p, M, float(res["rot"]), bool(res["flip"]))
        v = res["valid"][j]
        tint = rgbw.copy()
        tint[~v & res["disk"]] = 0.5 * tint[~v & res["disk"]] + 0.5 * np.array([1.0, 0.0, 0.0])
        tint[~res["disk"]] = 0.15
        axes[r + 1, c].imshow(tint, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5])
        if np.all(np.isfinite(tp)):
            axes[r + 1, c].plot(tp[0], tp[1], "c+", ms=6)
        w = res["white"][j].astype(np.float32)
        img = np.where(v, w, 0.35)
        img[~res["disk"]] = 0.1
        axes[r + 2, c].imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
        A = float(((w > 0.5) & v).sum()) / res["disk"].sum()
        axes[r + 2, c].set_title(f"white {A:.3f} valid {1 - res['occ_frac'][j]:.2f}", fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def qa_video(res, frames, path, size=320):
    """QA movie at the output frame rate: camera crop with the fitted ellipse (green = reliable, cyan = approximate,
    magenta = propagated), rectified colour frame in the T frame (+y up, invalid tinted red), and w (invalid grey)."""
    import shutil
    import tempfile
    from PIL import Image, ImageDraw
    tmp = tempfile.mkdtemp(prefix="vx_qa_")
    M = res["white"].shape[1]
    disk = res["disk"]
    cols = {2: (0, 255, 0), 1: (0, 255, 255), 0: (255, 0, 255)}
    try:
        for j in range(len(res["t"])):
            i = int(res["frame_index"][j])
            p = res["ell"][j]
            rgb = frames[i]
            # panel 1: camera crop
            x0, x1 = p[0] - 1.6 * p[2], p[0] + 1.6 * p[2]
            y0, y1 = p[1] - 2.0 * p[3], p[1] + 2.0 * p[3]
            im = Image.fromarray(rgb)
            d = ImageDraw.Draw(im)
            th = np.linspace(0, 2 * np.pi, 120)
            cs, sn = math.cos(p[4]), math.sin(p[4])
            pts = list(zip(p[0] + p[2] * np.cos(th) * cs - p[3] * np.sin(th) * sn, p[1] + p[2] * np.cos(th) * sn + p[3] * np.sin(th) * cs))
            d.line(pts + [pts[0]], fill=cols.get(int(res["quality"][j]), (255, 0, 0)), width=2)
            side = max(x1 - x0, y1 - y0)
            cx_, cy_ = (x0 + x1) / 2, (y0 + y1) / 2
            p1 = im.crop((int(cx_ - side / 2), int(cy_ - side / 2), int(cx_ + side / 2), int(cy_ + side / 2))).resize((size, size))
            # panel 2: rectified colour, invalid tinted
            rgbw = rectify(rgb.astype(np.float64) / 255.0, p, M, float(res["rot"]), bool(res["flip"]))
            v = res["valid"][j]
            tint = rgbw.copy()
            bad = ~v & disk
            tint[bad] = 0.5 * tint[bad] + 0.5 * np.array([1.0, 0.0, 0.0])
            tint[~disk] = 0.12
            p2 = Image.fromarray((np.clip(tint[::-1], 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
            # panel 3: w
            w = res["white"][j].astype(np.float32)
            img = np.where(v, w, 0.4)
            img[~disk] = 0.12
            p3 = Image.fromarray((img[::-1] * 255).astype(np.uint8)).convert("RGB").resize((size, size), Image.NEAREST)
            tp = res["tip"][j]
            if np.all(np.isfinite(tp)):
                d3 = ImageDraw.Draw(p3)
                qx, qy = (tp[0] + 0.5) * size, (0.5 - tp[1]) * size
                d3.line([(qx - 6, qy), (qx + 6, qy)], fill=(0, 200, 255), width=2)
                d3.line([(qx, qy - 6), (qx, qy + 6)], fill=(0, 200, 255), width=2)
            canvas = Image.new("RGB", (3 * size, size + 28), (20, 20, 20))
            canvas.paste(p1, (0, 28)); canvas.paste(p2, (size, 28)); canvas.paste(p3, (2 * size, 28))
            A = float(((w > 0.5) & v).sum()) / disk.sum()
            ImageDraw.Draw(canvas).text((6, 8), f"{res['name']}  t={res['t'][j]:5.2f}s  ellipse q{int(res['quality'][j])}  "
                                               f"white(valid) {A:.3f}  valid {v[disk].mean():.2f}   [camera | T frame, +y up | w]",
                                        fill=(230, 230, 230))
            canvas.save(os.path.join(tmp, f"f{j:04d}.png"))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", f"{float(res['fps_out'])}", "-i",
                        os.path.join(tmp, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2", path], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================================== frames: T (targets / frames.npz) <-> M (pitcher.Move)
def t_to_m(p):
    """T-frame point(s) [cup diameters] -> pitcher.Move frame: (x, y) -> (x, -y)."""
    p = np.asarray(p, float)
    return p * np.array([1.0, -1.0]) if p.ndim else p


def m_to_t(p):
    """pitcher.Move point(s) -> T frame (the same reflection)."""
    return t_to_m(p)


def sim_to_t(c):
    """Solver field c_M[iy, ix] -> T-frame field (rows flipped), comparable with targets/<name>.npz and frames.npz."""
    return np.asarray(c)[::-1, :]


# =========================================================================== stage scripts
POUR_KINDS = ("low_pour", "wiggle", "push", "retreat")
KIND_DEFAULTS = dict(            # z0, z1 [m], flow0, flow1 (relative to Q_ref)
    low_pour=(0.015, 0.015, 1.0, 1.0), wiggle=(0.02, 0.02, 1.0, 1.0), push=(0.015, 0.015, 1.4, 1.4),
    retreat=(0.02, 0.02, 0.8, 0.8), lift=(0.02, 0.06, 0.6, 0.3), high_cut=(0.07, 0.09, 0.3, 0.05),
    reposition=(0.03, 0.03, 0.0, 0.0), stop=(0.09, 0.09, 0.0, 0.0))


def load_stages(path_or_dict):
    """Read and validate a stage script (dict or JSON path).  Returns the dict with defaults filled in."""
    d = json.load(open(path_or_dict)) if isinstance(path_or_dict, str) else dict(path_or_dict)
    if d.get("frame", "M") != "M":
        raise ValueError("stage positions must be given in the pitcher.Move frame ('frame': 'M')")
    d.setdefault("Q_ref", Q_REF)
    st = []
    last_t1 = -1e9
    for k, s in enumerate(d["stages"]):
        s = dict(s)
        if s["kind"] not in STAGE_KINDS:
            raise ValueError(f"stage {k}: unknown kind {s['kind']!r} (allowed: {STAGE_KINDS})")
        if not s["t1"] > s["t0"]:
            raise ValueError(f"stage {k}: t1 must be > t0")
        if s["t0"] < last_t1 - 1e-6:
            raise ValueError(f"stage {k}: overlaps the previous stage (t0 {s['t0']} < {last_t1})")
        last_t1 = s["t1"]
        z0, z1, f0, f1 = KIND_DEFAULTS[s["kind"]]
        s.setdefault("p1", s["p0"])
        s.setdefault("z0", z0); s.setdefault("z1", s.get("z0", z0) if s["kind"] not in ("lift", "high_cut") else z1)
        s.setdefault("flow0", f0); s.setdefault("flow1", s["flow0"] if s["kind"] not in ("lift", "high_cut") else f1)
        s.setdefault("wiggle", None); s.setdefault("confidence", 0.5); s.setdefault("note", "")
        st.append(s)
    d["stages"] = st
    d.setdefault("t_clip0", st[0]["t0"])
    return d


def stages_to_moves(script, away=True, min_gap=0.02):
    """Stage script -> (list[pitcher.Move], t_clip0).  Simulation time 0 = clip time t_clip0 (first stage t0).
    Gaps between stages become dry repositions; a final 'move away' is appended as in pitcher.v05_moves."""
    from .pitcher import Move, away_point
    d = load_stages(script)
    Qr = d["Q_ref"]
    moves = []
    prev = None
    for s in d["stages"]:
        if prev is not None and s["t0"] - prev["t1"] > min_gap:
            zr = max(prev["z1"], 0.03)
            moves.append(Move(s["t0"] - prev["t1"], tuple(prev["p1"]), tuple(s["p0"]), z0=zr, z1=zr, Q0=0.0, Q1=0.0,
                              name="reposition (gap)"))
        wg = s.get("wiggle") or {}
        kw = {}
        if wg:
            ax = wg.get("axis", "x")
            if isinstance(ax, str):
                kw["wobble_axis"] = ax
            else:
                kw["wobble_dir"] = math.radians(float(ax))
        bez = tuple(tuple(c) for c in s["bezier"]) if s.get("bezier") else None
        moves.append(Move(s["t1"] - s["t0"], tuple(s["p0"]), tuple(s["p1"]), z0=s["z0"], z1=s["z1"],
                          Q0=s["flow0"] * Qr, Q1=s["flow1"] * Qr, wobble=float(wg.get("amp", 0.0)),
                          wobble1=float(wg.get("amp1", wg.get("amp", 0.0))), wfreq=float(wg.get("freq", 0.0)),
                          bezier=bez, name=f"{s['kind']} {s['t0']:.2f}-{s['t1']:.2f}s", **kw))
        prev = s
    if away:
        last = moves[-1]
        moves.append(Move(1.0, last.p1, away_point(last.p1), z0=max(last.z1, 0.09), z1=0.14, Q0=0.0, name="move away"))
    return moves, d["t_clip0"]


def stage_point(script, t):
    """Intended impact point (M frame) at clip time t (cosine easing like pitcher.Barista, wiggle not included)."""
    d = load_stages(script)
    for s in d["stages"]:
        if s["t0"] <= t <= s["t1"]:
            f = (t - s["t0"]) / (s["t1"] - s["t0"])
            e = 0.5 - 0.5 * math.cos(math.pi * f)
            return np.array(s["p0"], float) + e * (np.array(s["p1"], float) - np.array(s["p0"], float)), s
    return None, None


def pour_onsets(script):
    """(t0, stage) of every pour stage that follows a non-pour stage, a gap, or the clip start."""
    d = load_stages(script)
    out, prev = [], None
    for s in d["stages"]:
        if s["kind"] in POUR_KINDS and (prev is None or prev["kind"] not in POUR_KINDS or s["t0"] - prev["t1"] > 0.05):
            out.append((s["t0"], s))
        prev = s
    return out


# =========================================================================== observables
def _interp_cross(t, y, thr, i0):
    """First time >= t[i0] at which y crosses thr upward (linear interpolation), requiring the next sample too."""
    for i in range(max(i0, 1), len(t)):
        if y[i] >= thr and (i + 1 >= len(t) or y[i + 1] >= thr):
            if y[i - 1] < thr and i - 1 >= i0:
                return float(t[i - 1] + (thr - y[i - 1]) / max(y[i] - y[i - 1], 1e-12) * (t[i] - t[i - 1]))
            return float(t[i])
    return None


def filled_white(white, valid, t, hold=0.6):
    """Causal fill: w with invalid pixels replaced by the most recent valid observation of that pixel within `hold`
    seconds (never by a later frame, so timing observables are not advanced).  Returns (w_filled, observed)."""
    n = len(t)
    out = np.zeros(white.shape, np.float32)
    obs = np.zeros(white.shape, bool)
    last = np.zeros(white.shape[1:], np.float32)
    last_t = np.full(white.shape[1:], -1e9)
    for j in range(n):
        v = valid[j]
        last = np.where(v, white[j].astype(np.float32), last)
        last_t = np.where(v, t[j], last_t)
        ok = (t[j] - last_t) <= hold
        out[j] = np.where(ok, last, 0.0)
        obs[j] = ok
    return out, obs


def area_series(res, hold=0.6):
    W, V, D, t = res["white"].astype(np.float32), res["valid"], res["disk"], res["t"]
    M = W.shape[1]
    cell = (D_CUP * 100 / M) ** 2                                   # cm^2 per cell
    Wf, obs = filled_white(W, V & D[None], t, hold)
    A_valid = ((W > 0.5) & V).sum((1, 2)) * cell
    A_fill = ((Wf > 0.5) & obs & D[None]).sum((1, 2)) * cell
    frac_obs = (obs & D[None]).sum((1, 2)) / D.sum()
    As = ndi.gaussian_filter1d(A_fill, 1.0) if len(t) > 3 else A_fill
    dA = np.gradient(As, t) if len(t) > 1 else np.zeros_like(As)
    R = np.sqrt(A_fill / math.pi)
    return dict(t=t.tolist(), A_valid_cm2=A_valid.round(4).tolist(), A_filled_cm2=A_fill.round(4).tolist(),
                frac_valid=(V & D[None]).sum((1, 2)).astype(float).__truediv__(D.sum()).round(4).tolist(),
                frac_observed_filled=frac_obs.round(4).tolist(), dA_dt_cm2_s=dA.round(4).tolist(),
                R_eq_cm=R.round(4).tolist(), dR_dt_cm_s=np.gradient(ndi.gaussian_filter1d(R, 1.0), t).round(4).tolist()
                if len(t) > 3 else [], cup_area_cm2=float(D.sum() * cell)), Wf, obs


def first_white(res, Wf, obs, onsets, a_min_frac=0.004, ring=0.03):
    """For each pour onset t0: first time new white (not within `ring` D of the white present at t0) exceeds
    a_min_frac of the cup area on two consecutive frames (crossing interpolated)."""
    t, D = res["t"], res["disk"]
    M = D.shape[0]
    out = []
    r = max(1, int(round(ring * M)))
    for t0, st in onsets:
        i0 = int(np.searchsorted(t, t0 - 1e-6))
        if i0 >= len(t):
            continue
        jb = max(i0 - 1, 0)                        # white already there just before (or at) the onset
        base = ndi.binary_dilation((Wf[jb] > 0.5) & obs[jb], iterations=r)
        ro = max(1, int(round(0.02 * M)))                # thin streaks (stream edges, overlay) are not a first white
        st_o = np.hypot(*np.mgrid[-ro:ro + 1, -ro:ro + 1]) <= ro
        new = np.array([ndi.binary_opening((Wf[j] > 0.5) & obs[j] & D & ~base, structure=st_o).sum() / D.sum()
                        for j in range(len(t))])
        tf = _interp_cross(t, new, a_min_frac, i0)
        item = dict(t_onset=float(t0), kind=st["kind"] if st else None, t_first_white=tf,
                    delay_s=(None if tf is None else round(tf - t0, 3)), a_min_frac=a_min_frac)
        if tf is not None:
            j = int(np.searchsorted(t, tf - 1e-6))
            j = min(j + 1, len(t) - 1)
            m = ndi.binary_opening((Wf[j] > 0.5) & obs[j] & D & ~base, structure=st_o)
            if m.any():
                X, Y = cup_grid(M)
                c = np.array([X[m].mean(), Y[m].mean()])
                item.update(centroid_T=c.round(4).tolist(), centroid_M=t_to_m(c).round(4).tolist())
        out.append(item)
    return out


def _ray_hits(w, valid, c, steps, th):
    """First visible white along rays from c (T frame); a ray counts if visible brown precedes the hit and no
    occluded stretch > 0.03 D lies just before it.  Returns (radii, hit points)."""
    M = w.shape[0]
    D = np.stack([np.cos(th), np.sin(th)], 1)                        # [nray, 2]
    P = c[None, None, :] + steps[None, :, None] * D[:, None, :]      # [nray, nstep, 2]
    ix = np.clip(((P[..., 0] + 0.5) * M).astype(int), 0, M - 1)
    iy = np.clip(((P[..., 1] + 0.5) * M).astype(int), 0, M - 1)
    vv, ww = valid[iy, ix], w[iy, ix] > 0.5
    hit = vv & ww
    rad, pts = [], []
    for k in range(len(th)):
        if not hit[k].any():
            continue
        j = int(np.argmax(hit[k]))
        if not (vv[k, :j] & ~ww[k, :j]).any():
            continue
        gap = 0
        for kk in range(j - 1, -1, -1):
            if vv[k, kk]:
                break
            gap += 1
        if gap <= 0.03 * M:
            rad.append(steps[j]); pts.append(P[k, j])
    return np.array(rad), np.array(pts)


def hole_radius(w, valid, center_T, rmax=0.2, nray=72, search=0.08, iters=2):
    """Brown hole at the impact, seen as the brown region enclosed by the white ring.  The centre is searched within
    `search` D of center_T (visible brown points maximising the fraction of rays that end on white), then moved to
    the centroid of the ring hits.  Returns dict(r_D, r_cm, center_T, hit_frac) or None."""
    M = w.shape[0]
    th = np.linspace(0, 2 * np.pi, nray, endpoint=False)
    steps = np.arange(1, int(rmax * M) + 1) / M
    c0 = np.asarray(center_T, float)
    best, best_f = None, 0.0
    g = np.arange(-search, search + 1e-9, 0.02)
    for dx in g:
        for dy in g:
            if dx * dx + dy * dy > search * search + 1e-9:
                continue
            c = c0 + (dx, dy)
            ix, iy = int((c[0] + 0.5) * M), int((c[1] + 0.5) * M)
            if not (0 <= ix < M and 0 <= iy < M) or not valid[iy, ix] or w[iy, ix] > 0.5:
                continue
            rad, _ = _ray_hits(w, valid, c, steps, th[::3])
            f = len(rad) / len(th[::3])
            if f > best_f:
                best, best_f = c, f
    if best is None or best_f < 0.4:
        return None
    c, res = best, None
    for _ in range(iters + 1):
        rad, pts = _ray_hits(w, valid, c, steps, th)
        if len(rad) < 0.4 * nray:
            break
        res = dict(r_D=float(np.median(rad)), r_cm=float(np.median(rad) * D_CUP * 100), center_T=c.round(4).tolist(),
                   hit_frac=round(len(rad) / nray, 3))
        c = pts.mean(0)
        ix, iy = int((c[0] + 0.5) * M), int((c[1] + 0.5) * M)
        if not (0 <= ix < M and 0 <= iy < M) or w[iy, ix] > 0.5:
            break
    return res


def track_blocks(res, Wf, obs, t0, t1, impact_T, min_frac=0.005):
    """White blocks present at t0, followed until t1 + 0.4 s: centroid and near/far edge along the axis from the
    impact through each block's initial centroid, and the brown gap to the new white."""
    t, D = res["t"], res["disk"]
    M = D.shape[0]
    X, Y = cup_grid(M)
    i0 = int(np.searchsorted(t, t0 - 1e-6))
    i0 = max(i0 - 1, 0)
    i1 = int(np.searchsorted(t, t1 + 0.4))
    W0 = (Wf[i0] > 0.5) & obs[i0] & D
    lab, n = ndi.label(W0)
    blocks = []
    imp = np.asarray(impact_T, float)
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < min_frac * D.sum():
            continue
        c0 = np.array([X[m].mean(), Y[m].mean()])
        u = c0 - imp
        u = u / (np.linalg.norm(u) + 1e-9)
        cur = m
        ser = []
        for j in range(i0, min(i1 + 1, len(t))):
            Wj = (Wf[j] > 0.5) & obs[j] & D
            lj, nj = ndi.label(Wj)
            ids = np.unique(lj[ndi.binary_dilation(cur, iterations=max(1, int(0.02 * M))) & Wj])
            ids = ids[ids > 0]
            if len(ids) == 0:
                break
            nxt = np.isin(lj, ids)
            big = np.nonzero(ndi.sum(Wj, lj, range(1, nj + 1)) >= 0.002 * D.sum())[0] + 1
            other = np.isin(lj, big) & ~nxt
            s = (X[nxt] - imp[0]) * u[0] + (Y[nxt] - imp[1]) * u[1]
            gap = None
            if other.any():
                dt = ndi.distance_transform_edt(~nxt)
                gap = float(dt[other].min() / M * D_CUP * 100)
            ser.append(dict(t=round(float(t[j]), 3), centroid_u_cm=round(float(s.mean() * D_CUP * 100), 3),
                            near_u_cm=round(float(np.percentile(s, 2) * D_CUP * 100), 3),
                            far_u_cm=round(float(np.percentile(s, 98) * D_CUP * 100), 3),
                            area_cm2=round(float(nxt.sum() * (D_CUP * 100 / M) ** 2), 3), gap_to_other_white_cm=gap,
                            merged=bool(nxt.sum() > 1.6 * cur.sum())))
            cur = nxt
        if ser:
            s0 = ser[0]
            for r_ in ser:
                r_.update(d_centroid_cm=round(r_["centroid_u_cm"] - s0["centroid_u_cm"], 3),
                          d_near_cm=round(r_["near_u_cm"] - s0["near_u_cm"], 3), d_far_cm=round(r_["far_u_cm"] - s0["far_u_cm"], 3))
        blocks.append(dict(centroid0_T=c0.round(4).tolist(), axis_T=u.round(4).tolist(), axis_M=t_to_m(u).round(4).tolist(),
                           series=ser))
    return blocks


def component_gaps(w, valid, min_frac=0.004):
    """Minimum brown gap [cm] between every pair of white components larger than min_frac of the cup."""
    M = w.shape[0]
    D = disk_mask(M)
    lab, n = ndi.label((w > 0.5) & valid & D)
    comps = [k for k in range(1, n + 1) if (lab == k).sum() >= min_frac * D.sum()]
    X, Y = cup_grid(M)
    out = []
    for i, ka in enumerate(comps):
        dt = ndi.distance_transform_edt(lab != ka)
        for kb in comps[i + 1:]:
            mb = lab == kb
            k = int(np.argmin(np.where(mb, dt, np.inf)))
            out.append(dict(a=int(ka), b=int(kb), gap_cm=round(float(dt.ravel()[k] / M * D_CUP * 100), 3),
                            where_T=[round(float(X.ravel()[k]), 3), round(float(Y.ravel()[k]), 3)]))
    sizes = {int(k): round(float((lab == k).sum() / D.sum()), 4) for k in comps}
    return dict(components=sizes, gaps=out)


def axis_runs(w, valid, p0_T, p1_T, band=0.02, step=0.004):
    """White/brown run lengths along the segment p0 -> p1 (T frame), w averaged over +-band across it.  Returns a list
    of (kind, start_D, length_D) and summary stats of the interior runs."""
    M = w.shape[0]
    p0, p1 = np.asarray(p0_T, float), np.asarray(p1_T, float)
    L = np.linalg.norm(p1 - p0)
    if L < 1e-6:
        return None
    u = (p1 - p0) / L
    nrm = np.array([-u[1], u[0]])
    s = np.arange(0, L, step)
    offs = np.linspace(-band, band, 5)
    prof, ok = [], []
    for si in s:
        vals, vs = [], []
        for o in offs:
            q = p0 + si * u + o * nrm
            ix, iy = int((q[0] + 0.5) * M), int((q[1] + 0.5) * M)
            if 0 <= ix < M and 0 <= iy < M:
                vals.append(w[iy, ix]); vs.append(valid[iy, ix])
        prof.append(np.mean(vals) if vals else np.nan)
        ok.append(bool(vs) and np.mean(vs) > 0.6)
    prof, ok = np.array(prof), np.array(ok)
    runs = []
    cur, start = None, 0
    for k in range(len(s)):
        lab_ = "x" if not ok[k] else ("W" if prof[k] > 0.5 else "b")
        if lab_ != cur:
            if cur is not None:
                runs.append((cur, round(float(s[start]), 4), round(float(s[k] - s[start]), 4)))
            cur, start = lab_, k
    runs.append((cur, round(float(s[start]), 4), round(float(L - s[start]), 4)))
    # interior: between the first and last white run
    iw = [k for k, r_ in enumerate(runs) if r_[0] == "W"]
    stats = {}
    if len(iw) >= 2:
        inner = runs[iw[0]:iw[-1] + 1]
        wc = [r_[1] + r_[2] / 2 for r_ in inner if r_[0] == "W"]
        bl = [r_[2] for r_ in inner if r_[0] == "b"]
        stats = dict(n_white=len(wc), pitch_D=round(float(np.median(np.diff(wc))), 4) if len(wc) > 1 else None,
                     brown_line_D=round(float(np.median(bl)), 4) if bl else None,
                     brown_lines_D=[round(float(x), 4) for x in bl])
        for k_ in ("pitch_D", "brown_line_D"):
            if stats.get(k_) is not None:
                stats[k_.replace("_D", "_cm")] = round(stats[k_] * D_CUP * 100, 3)
    return dict(runs=runs, **stats)


def cut_observables(w, valid, p0_T, p1_T):
    """Cut line along p0 -> p1 (T frame): brown line width across the axis inside the white, stem tip (furthest
    white of the main pattern along the cut direction) and stem width just behind the tip."""
    M = w.shape[0]
    X, Y = cup_grid(M)
    p0, p1 = np.asarray(p0_T, float), np.asarray(p1_T, float)
    L = np.linalg.norm(p1 - p0)
    if L < 1e-6:
        return None
    u = (p1 - p0) / L
    nrm = np.array([-u[1], u[0]])
    Wm = (w > 0.5) & valid
    lab, n = ndi.label(Wm)
    if n == 0:
        return None
    sizes = ndi.sum(Wm, lab, range(1, n + 1))
    main = np.isin(lab, np.nonzero(sizes >= 0.0005 * M * M)[0] + 1) & (np.hypot(X, Y) < 0.48)
    s_all = (X - p0[0]) * u[0] + (Y - p0[1]) * u[1]
    o_all = (X - p0[0]) * nrm[0] + (Y - p0[1]) * nrm[1]
    near_axis = np.abs(o_all) < 0.04
    if not (main & near_axis).any():
        return None
    tipk = np.argmax(np.where(main & near_axis, s_all, -9))
    tip = np.array([X.ravel()[tipk], Y.ravel()[tipk]])
    s_tip = float(s_all.ravel()[tipk])
    widths = []
    for sv in np.arange(0.05, max(s_tip - 0.05, 0.05), 0.02):
        q = p0 + sv * u
        o = np.arange(-0.08, 0.0801, 0.0025)
        pts = q[None, :] + o[:, None] * nrm[None, :]
        ix = np.clip(((pts[:, 0] + 0.5) * M).astype(int), 0, M - 1)
        iy = np.clip(((pts[:, 1] + 0.5) * M).astype(int), 0, M - 1)
        if valid[iy, ix].mean() < 0.9:
            continue
        ww = w[iy, ix] > 0.5
        c = len(o) // 2
        if ww[c]:
            continue
        lo_ = c
        while lo_ > 0 and not ww[lo_ - 1]:
            lo_ -= 1
        hi_ = c
        while hi_ < len(o) - 1 and not ww[hi_ + 1]:
            hi_ += 1
        if lo_ > 0 and hi_ < len(o) - 1:                     # brown run bounded by white on both sides
            widths.append((round(float(sv), 3), round(float((hi_ - lo_ + 1) * 0.0025), 4)))
    q = p0 + (s_tip - 0.03) * u
    o = np.arange(-0.06, 0.0601, 0.0025)
    pts = q[None, :] + o[:, None] * nrm[None, :]
    ix = np.clip(((pts[:, 0] + 0.5) * M).astype(int), 0, M - 1)
    iy = np.clip(((pts[:, 1] + 0.5) * M).astype(int), 0, M - 1)
    stem_w = float((w[iy, ix] > 0.5).sum() * 0.0025)
    r_tip = float(np.hypot(*tip))
    return dict(tip_T=tip.round(4).tolist(), tip_M=t_to_m(tip).round(4).tolist(), tip_r_D=round(r_tip, 4),
                tip_to_rim_cm=round((0.5 - r_tip) * D_CUP * 100, 3), stem_width_D=round(stem_w, 4),
                cut_line_widths_D=widths,
                cut_line_width_median_cm=(round(float(np.median([x[1] for x in widths])) * D_CUP * 100, 3) if widths else None))


def _frame_at(t, tq):
    return int(np.clip(np.searchsorted(t, tq - 1e-6), 0, len(t) - 1))


def compute_observables(res, stages=None, hold=0.6):
    """All observables (physical units: cm, s) from frames.npz contents and an optional stage script.
    Invalid pixels are filled causally with the last valid observation of the pixel within `hold` s (never from a
    later frame).  Keys of the returned dict / observables.json:
      area          t, A_valid_cm2 (visible white), A_filled_cm2, frac_valid, frac_observed_filled, dA_dt_cm2_s,
                    R_eq_cm (sqrt(A/pi)), dR_dt_cm_s, cup_area_cm2
      impact_track  t, tip_T, tip_M (stream foot / pitcher tip), stream_px
      first_white   per pour onset (stage script; else clip start): t_onset, t_first_white, delay_s, centroid_T/_M
                    (new white = outside the white present at the onset dilated by 0.03 D, opened with a 0.02 D disc,
                    >= 0.4 % of the cup on two consecutive frames, crossing time interpolated)
      hole          per frame of low_pour/push stages (else frames with a stream): brown hole radius r_cm / r_D around
                    the impact (rays to the white ring), its centre, hit_frac (ring closure), centre source
      push_blocks   per pour onset with white already present: every old block's centroid / near / far edge along the
                    axis impact -> block (cm, and d_* relative to the onset), area, gap to the new white, merged flag
      gaps          per frame: white components (> 0.4 % of the cup) and the minimum brown gap of every pair (cm)
      stripes       at the final frame and at the end of every wiggle/retreat/push/low_pour stage: white/brown run
                    lengths along the pattern axis (y_T, through the stage's mean x) and across it; pitch_D/_cm,
                    brown_line_D/_cm (medians of the interior runs)
      cut           per high_cut stage (else an assumed cut along y_T, exit at +y_T), after the cut and at the end:
                    stem tip (T, M, distance to the rim), stem width, brown cut-line widths across the axis
      quality       ellipse quality, valid fraction, overlay flag, crema gain per frame
    """
    t = res["t"]
    M = res["white"].shape[1]
    W = res["white"].astype(np.float32)
    V = res["valid"] & res["disk"][None]
    area, Wf, obs = area_series(res, hold)
    out = dict(name=str(res["name"]), D_cup_cm=D_CUP * 100, M=int(M), fps_out=float(res["fps_out"]),
               t_ref=float(res["t_ref"]), units=dict(length="cm unless *_D (cup diameters) or *_T/*_M (frame coords, D)",
                                                      time="s, clip time"),
               quality=dict(t=t.round(3).tolist(), ellipse_quality=res["quality"].astype(int).tolist(),
                            frac_valid=area["frac_valid"], overlay=res["overlay"].astype(bool).tolist(),
                            exposure_gain=np.round(res["gain"], 4).tolist()),
               area=area)
    tipT = res["tip"]
    out["impact_track"] = dict(t=t.round(3).tolist(), tip_T=np.round(tipT, 4).tolist(),
                               tip_M=np.round(tipT * np.array([1, -1]), 4).tolist(),
                               stream_px=res["stream_px"].astype(int).tolist(),
                               note="lowest stream (else pitcher) point mapped to the surface; the stream foot while "
                                    "it lands on crema, approximate when it lands on white; NaN = no occluder")
    d = load_stages(stages) if stages is not None else None
    onsets = pour_onsets(d) if d else [(float(t[0]), None)]
    out["first_white"] = first_white(res, Wf, obs, onsets)
    # hole radius
    holes = []
    for j in range(len(t)):
        cands, kind = [], None
        if d:
            pm, st = stage_point(d, float(t[j]))
            if st is None or st["kind"] not in ("low_pour", "push"):
                continue
            kind = st["kind"]
            cands.append(("stage", m_to_t(pm)))
        else:
            kind = "auto"
        if np.all(np.isfinite(tipT[j])) and res["stream_px"][j] > 0:
            cands.append(("stream_foot", tipT[j]))
        best = None
        for src, cen in cands:
            h = hole_radius(Wf[j], obs[j] & res["disk"], cen)
            if h is not None and (best is None or h["hit_frac"] > best["hit_frac"]):
                best = dict(h, center_from=src)
        if best is not None:
            best.update(t=round(float(t[j]), 3), kind=kind)
            holes.append(best)
    out["hole"] = holes
    # blocks pushed by a new pour (every pour onset with white already present)
    blocks = []
    for t0, st in onsets:
        if st is None:
            continue
        i0 = _frame_at(t, t0)
        if i0 == 0 or ((Wf[i0 - 1] > 0.5) & obs[i0 - 1]).sum() < 0.005 * res["disk"].sum():
            continue
        t1 = st["t1"]
        for s2 in d["stages"]:                      # follow through the consecutive pour stages
            if s2["t0"] >= st["t0"] and s2["kind"] in POUR_KINDS and s2["t0"] <= t1 + 0.05:
                t1 = max(t1, s2["t1"])
        bl = track_blocks(res, Wf, obs, t0, t1, m_to_t(st["p0"]))
        blocks.append(dict(t_onset=float(t0), t_end=float(t1), impact_T=m_to_t(st["p0"]).round(4).tolist(), blocks=bl))
    out["push_blocks"] = blocks
    # gaps between components: every frame (valid fraction reported alongside) + the final frame
    gaps = []
    for j in range(len(t)):
        g = component_gaps(Wf[j], obs[j])
        g.update(t=round(float(t[j]), 3), frac_valid=area["frac_valid"][j])
        gaps.append(g)
    out["gaps"] = gaps
    # stripes / runs along the pattern axis (y_T) and across it, at stage ends and at the final frame
    probes = []
    times = [("final", float(t[-1]), None)]
    if d:
        for s in d["stages"]:
            if s["kind"] in ("wiggle", "retreat", "push", "low_pour"):
                times.append((f"{s['kind']}@{s['t1']:.2f}", float(s["t1"]), s))
    for lab_, tq, s in times:
        j = _frame_at(t, tq)
        xs = 0.0
        if s is not None:
            xs = float(m_to_t(np.array(s["p0"], float))[0] + m_to_t(np.array(s["p1"], float))[0]) / 2
        probes.append(dict(at=lab_, t=round(float(t[j]), 3),
                           axis_y=axis_runs(Wf[j], obs[j], (xs, -0.47), (xs, 0.47)),
                           axis_x=axis_runs(Wf[j], obs[j], (-0.47, 0.0), (0.47, 0.0))))
    out["stripes"] = probes
    # cut
    cuts = []
    cut_stages = [s for s in d["stages"] if s["kind"] == "high_cut"] if d else []
    if not cut_stages:
        cut_stages = [dict(kind="high_cut", t0=float(t[-1]), t1=float(t[-1]), p0=[0.0, 0.47], p1=[0.0, -0.47], auto=True)]
    for s in cut_stages:
        p0T, p1T = m_to_t(np.array(s["p0"], float)), m_to_t(np.array(s["p1"], float))
        u = p1T - p0T
        if np.linalg.norm(u) < 1e-6:
            continue
        u /= np.linalg.norm(u)
        a0, a1 = p0T - 0.6 * u, p1T + 0.6 * u                     # extend through the whole cup
        a0 = a0 if np.hypot(*a0) < 0.48 else a0 * 0.48 / np.hypot(*a0)
        a1 = a1 if np.hypot(*a1) < 0.48 else a1 * 0.48 / np.hypot(*a1)
        for lab_, tq in (("after_cut", min(float(s["t1"]) + 0.4, float(t[-1]))), ("final", float(t[-1]))):
            j = _frame_at(t, tq)
            c = cut_observables(Wf[j], obs[j], a0, a1)
            if c is not None:
                c.update(at=lab_, t=round(float(t[j]), 3), stage_t0=float(s["t0"]), auto=bool(s.get("auto", False)))
                cuts.append(c)
    out["cut"] = cuts
    out["stages_used"] = d
    return out


# =========================================================================== frame check (old solver, V0.5 heart)
def frame_check(c_sim=None, N=64, out="runs/frame_check"):
    """Run (or load) the V0.5 heart through the old solver and compare the final orientation with targets/heart.npz:
    width profile along y (lobes vs tip) correlated with the target as is and with rows flipped (T <- M), and the
    y-skewness of the white (> 0: pointed tip / stem toward +y).  Run from the repository root (action_opt reads
    runs/joint_sweep_r2fix).  Result of 2026-10-12 (N=64, 89 s): video_targets/frame_check.json."""
    if c_sim is None:
        from . import action_opt as ao
        r = ao.run_one(("old", "heart", 1.0, 1.0, 1.0, N, out))
        c_sim = np.load(os.path.join(out, "old", f"heart_{ao.tag_of(1.0, 1.0, 1.0)}_N{N}.npy"))
    N = c_sim.shape[0]
    g = (np.arange(N) + 0.5) / N - 0.5
    Xg, Yg = np.meshgrid(g, g)
    mask = np.hypot(Xg, Yg) < 0.49
    s = np.clip(np.nan_to_num(c_sim) / 0.6, 0, 1) * mask
    w_t, _ = TG.load("heart")
    w_t = np.clip(ndi.zoom(w_t, N / w_t.shape[0], order=1), 0, 1)

    def prof(a):
        return (a > 0.5).sum(1).astype(float)

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

    def iou(a, b):
        a, b = a > 0.5, b > 0.5
        return float((a & b).sum() / max((a | b).sum(), 1))

    def skew_y(a):
        """Skewness of the y distribution of the white pixels: > 0 = long tail (pointed tip / stem) toward +y."""
        m = a > 0.5
        y = Yg[m]
        return float(((y - y.mean()) ** 3).mean() / (y.std() ** 3 + 1e-12))

    res = dict(N=N, profile_corr_identity=corr(prof(s), prof(w_t)), profile_corr_rows_flipped=corr(prof(s[::-1]), prof(w_t)),
               iou_identity=iou(s, w_t), iou_rows_flipped=iou(s[::-1], w_t),
               skew_y_sim_M=skew_y(s), skew_y_target_T=skew_y(w_t))
    flip = res["profile_corr_rows_flipped"] > res["profile_corr_identity"] and res["skew_y_sim_M"] * res["skew_y_target_T"] < 0
    res["verdict"] = ("tip/stem of the simulated heart at -y in the solver (M) frame, at +y in targets (T): "
                      "(x, y)_M = (x, -y)_T" if flip else "no flip detected")
    return res


# =========================================================================== CLI
def run_all(name, M=192, fps=5.0, stages=None, out_dir=None, preview=True, video=True, obs_only=False, log=print):
    out_dir = out_dir or out_dir_of(name)
    os.makedirs(out_dir, exist_ok=True)
    if stages is None and os.path.exists(os.path.join(out_dir, "stages.json")):
        stages = os.path.join(out_dir, "stages.json")
    frames = None
    if obs_only and os.path.exists(os.path.join(out_dir, "frames.npz")):
        res = load_frames(name, out_dir)
    else:
        res, frames, fps_v, _ = extract(name, M=M, fps_out=fps, out_dir=out_dir, log=log)
        res = load_frames(name, out_dir)
    match = compare_with_target(res)
    log(f"{name}: final frame vs targets/{name}.npz: IoU {match['iou']:.3f}, L1 {match['l1']:.3f}")
    obs = compute_observables(res, stages)
    obs["final_match_targets"] = match
    obs["frame_convention"] = FRAME_CONVENTION
    with open(os.path.join(out_dir, "observables.json"), "w") as f:
        json.dump(obs, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    if preview or video:
        if frames is None:
            frames, _ = read_video(str(res["clip"]))
    if preview:
        preview_sheet(res, frames, os.path.join(out_dir, "preview.png"))
    if video:
        qa_video(res, frames, os.path.join(out_dir, "extraction.mp4"))
    log(f"{name}: wrote {out_dir}/observables.json" + (f" (stages {stages})" if stages else " (no stage script: auto observables)"))
    return res, obs


FRAME_CONVENTION = dict(
    T="frames.npz / targets/<name>.npz: arrays [iy, ix], x=(ix+0.5)/M-0.5, y=(iy+0.5)/M-0.5 [cup diameters]; pattern axis y, "
      "pours start on the B side (y<0), cut exits at +y; image->T as targets.warp (e_x = major axis to image-left, "
      "e_y = (e_x[1], -e_x[0])), then rotation targets.FRAMES[name]['rot']; mirror image of the physical top view",
    M="pitcher.Move / solver Inlet / solver arrays c[iy, ix]: right-handed top view, spout toward -y (Barista yaw=-pi/2), "
      "pitcher handle at +y; V0.5 pours start at +y and cut toward -y",
    T_to_M="(x, y)_M = (x, -y)_T ; solver field to T: c_M[::-1, :]",
    stage_positions="stage scripts give p0/p1 in the M frame (cup diameters)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="extract per-frame white maps and observables from a reference clip")
    ap.add_argument("name", choices=PATTERNS + ["all"])
    ap.add_argument("--M", type=int, default=192)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--stages", default=None, help="stage-script JSON (default: video_targets/<name>/stages.json if present)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-video", action="store_true", help="skip the QA movie extraction.mp4")
    ap.add_argument("--obs-only", action="store_true", help="recompute observables.json from an existing frames.npz")
    ap.add_argument("--frame-check", action="store_true", help="run the V0.5 heart through the old solver (N=64) and report the T/M flip")
    a = ap.parse_args(argv)
    if a.frame_check:
        print(json.dumps(frame_check(), indent=1))
        return
    names = PATTERNS if a.name == "all" else [a.name]
    for n in names:
        run_all(n, M=a.M, fps=a.fps, stages=a.stages if len(names) == 1 else None, out_dir=a.out if len(names) == 1 else None,
                preview=not a.no_preview, video=not a.no_video, obs_only=a.obs_only)


if __name__ == "__main__":
    main()
