"""Occlusion-aware fusion of the rectified top-view frames of video_obs.py (V2_SURFACE_MODEL.md 10.5).

Why.  cup_register.fuse() samples every camera at the liquid plane and keeps a camera where its depth is not nearer than
the plane by OCC -- and MISSING DEPTH COUNTS AS VISIBLE.  The black, glossy cup rim and the steel pitcher return no
depth, so the oblique cameras 0 and 2 contribute their view of the cup wall (black) along a band at the near rim, and
camera 1 its view of the pitcher spout; the per-pixel median of 2 views then turns grey -> staircase 'cracks' across the
disc and the white (94-98 % of the median's dark pixels in the crack windows are such a leak; the leaks of cams 0 / 2
lie 2 px (median) from their valid near-depth pixels, 72-95 % within 8 px).  The thin in-air stream is not resolved by
stereo: its depth reads as the plane, so every camera keeps it.  Exposure differences between the cameras make seams
where the set of visible views changes.

What (per frame; all constants shared by every pour and frame):
 1. occluders  in each camera's IMAGE (848 x 480):
               near    valid depth nearer than the liquid plane (ray / plane intersection) by > OCC, closed; missing-depth
                       blobs whose rim is mostly near (dropouts inside the hand / rim / pitcher) filled (a whole enclosed
                       hole is NOT filled: the rim ring encloses the liquid in camera 1);
               pitcher the tag-tracked pitcher solid (body of revolution of the measured radius profile + the spout tip,
                       pitcher_model.py pose convention x_w = R(t) x_p + p(t), poses of tip.npz minus temporal outliers,
                       each camera at its own frame time), convex hull of its projection (the stored pitcher cloud holds
                       only the camera-facing half of the wall plus stream / foam points beyond the spout, so it is not used);
               stream  capsules of radius JET_R from the spout tip straight down to the plane and to the measured stream
                       foot (stream_foot.py; running median of the foot - tip offset over +-3 frames) -- the thin jet is not
                       resolved by stereo and reads as the plane;
               all dilated by MARGIN_PX = the width of the dark band at occluder edges (each camera's view divided by the
               depth-confirmed other views, binned by image distance to its near depth: the 10th percentile is back on its
               plateau at 8-10 px; beyond it missing depth on the liquid is harmless and is NOT treated as occluded).
 2. rectify    every camera bilinearly (pixel centres at integer coordinates, as librealsense / OpenCV) onto a 2x
               supersampled disc grid at the plane z of top/<fi>.npz; usable = inside the image & outside the occluders &
               not a per-camera specular glare blob.
 3. fuse       per-camera log-gain alignment to the consensus: a global per-channel offset + a smooth field (block
               medians + median filter, 2 iterations) clamped to +-G_LOC, both estimated on the >= 3-view pixels only
               (>= 2 for a camera without enough of them; a 2-view pair whose aligned views disagree by > DIS is left out
               of the fit -- unclamped, a 2-view fit turned leaks into gains up to |G| 1.6 and carried them into
               single-view regions); weights = resolution ((s_min / s)^4, s = the worst-direction ground sampling step:
               the blur bias of a fine feature ~ sigma^2, inverse squared bias -> s^-4) x outlier rejection when >= 3
               views x a distance feather at usable-mask edges; pixels with no usable view are invalid
               (their colour falls back to the cup_register rule, so the image looks as before there).  Usable-weighted
               2x2 mean -> the 128 video grid.
 4. validity   video_obs.observe() (inside R_IN * edge, not a small saturated glare blob, outside FOOT_R of the spout
               tip) on the usable-view mask instead of cup_register's 'seen'.
Cost: the occluders cost 2-3 % of the disc and the 8 px margin another 13-18 % in the early frames (t < 4 s valid
0.75-0.80 of the disc against 0.94-0.97; whole pour 0.84-0.87 against 0.97), including most of the stream-foot
neighbourhood (0.07-0.26 valid within 4 mm of the foot); the image data do not support a smaller margin (camera 1 is
still 7 % brighter than the other views 12 px from its stream mask).  About a third of the valid pixels are single-view.
In 2-view pixels camera 1 always has the larger weight and no outlier test is possible: its lamp highlight (late leaf)
and dark cup-wall reflections at the left rim (late tulip / leaf) remain (dropping 2-view no-consensus pixels cost tulip
2.2 % of its valid area; not done).
Output (video_obs.write_frames): t, img RGB, L, valid, nview (usable views per pixel), frame_idx, pitcher_depth_conflict
(per camera: share of the pitcher hull where the camera sees valid depth AT the plane; high = wrong pose), pose_rejected_t.
    python3 -m latte_imex.rig.fuse_occ [--out DIR] [--frames a:b] [pour ...]     (needs the session's pours/ data)
"""
import argparse
import json
import os
import tempfile
import time
import warnings

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .trimrec import Take
from .video_obs import EXT, FOOT_R, OUT, POURS, R_IN, REPO, SCR, observe

RES = os.path.join(REPO, "raw_data", "rig", "results")
SS = 2                      # supersampling of the fusion grid (2x2 mean -> OUT)
NS = SS * OUT
# occluders
OCC = 0.010                 # [m] valid depth nearer than the plane by more than this is an occluder (cup_register)
CLOSE_PX = 5                # closing of the near mask (bridges 1-2 px depth speckle)
HOLE_NEAR = 0.6             # a missing-depth blob is filled when this share of its rim is near-valid depth
MARGIN_PX = 8               # [image px] occluder dilation: measured dark-edge width (edge profile, module docstring)
JET_R = 0.003               # [m] stream radius (3-6 mm wide stream, before the margin)
FOOT_HALF, FOOT_MAX = 3, 0.015   # stream drift: running median (+-3 frames) of the measured foot - tip offset
                                  # (stream_foot.json, all frames: single 'trusted' feet can be 20 mm off); not used above 15 mm
POSE_WIN, POSE_TOL = 0.3, 0.020   # [s], [m] a tag-BA pose is dropped when its tip or origin is > POSE_TOL off the local
                                  # linear motion of the neighbouring poses within +-POSE_WIN (2x the p95 residual ~10 mm)
# fusion
GLARE_T, GLARE_A = 240, 0.004     # per-camera glare: components with min channel > GLARE_T smaller than GLARE_A of the disc
SMOOTH_B, SMOOTH_K = 8, 5         # smooth gain field: block size [NS px], median filter [blocks]
TAU = 0.12                        # outlier scale of the aligned log lightness (>= 3 views)
GAIN_MINV, GAIN_MIN_PX = 3, 500   # gain consensus only where >= 3 views are usable (their median rejects one leaking view);
                                  # a camera with < 500 such pixels (early frames) falls back to >= 2 views
G_LOC = 0.15                # clamp of the smooth part of a camera's log gain around its global offset (0.15 = the largest
                            # per-pour median of the per-frame p99 |G| in 3-view regions)
DIS, DIS_SIGMA = 3 * 0.12, 2.0    # 2-view consensus: two usable views whose aligned log lightness (blurred within their masks by
                                  # 2 NS px = 0.7 mm, the oblique cameras' sampling step) differ by > 3 TAU -> neither enters
                                  # the gain fit (both stay in the fusion)
RES_POW = 4.0               # view weight (s_min / s)^RES_POW, s = worst-direction ground sampling step
FEATHER_PX = 4.0            # [px of the NS grid] weight ramp at a camera's usable-mask edge (1.4 mm)
H_IMG, W_IMG = 480, 848


# ------------------------------------------------------------------------------------------------ pour data
class PourData:
    """Take, cup frame (disc.json), plane (top/<fi>.npz z), pitcher solid and stream feet of one pour."""

    def __init__(self, p, scr=SCR):
        self.p, self.scr = p, scr
        self.tk = Take(f"{scr}/ghdata/{p}")
        self.D = json.load(open(f"{scr}/{p}/disc.json"))
        self.ct = json.load(open(f"{scr}/{p}/cup_track.json"))
        self.t0 = self.D[0]["t"]
        self.w2c = [np.linalg.inv(T) for T in self.tk.c2w]
        vv, uu = np.mgrid[0:H_IMG, 0:W_IMG].astype(np.float64)
        # per-pixel ray direction in world coordinates (camera z component = 1 before rotation)
        self.rays_w = []
        for c, K in enumerate(self.tk.K):
            r = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
            self.rays_w.append((r @ self.tk.c2w[c][:3, :3].T)[..., 2].astype(np.float32))
        # pitcher: good tag-BA poses (tip.npz), body profile, spout tip
        T = np.load(f"{scr}/{p}/tip.npz")
        o = np.argsort(T["t"])
        self.tip_p = np.array(T["tip_p"], float)
        keep = pose_inliers(T["t"][o], T["poses"][o], self.tip_p)
        self.pose_rejected = T["t"][o][~keep]
        self.pt, self.pP = T["t"][o][keep], T["poses"][o][keep]
        self.slerp = Slerp(self.pt, Rotation.from_rotvec(self.pP[:, :3]))
        self.solid = pitcher_solid(p, scr, self.tip_p)
        # the stream feet measured from the same pose track as this fusion's pitcher (pours/<p>/tip.npz = the v1 track)
        fn = os.path.join(RES, p, "stream_foot_v1.json")
        if not os.path.exists(fn):
            fn = os.path.join(RES, p, "stream_foot.json")
        self.foot_off = {}                                # frame -> horizontal stream-foot offset from the tip [m]
        if os.path.exists(fn):
            F = json.load(open(fn))["frames"]
            fi = np.array([f["fi"] for f in F])
            off = np.array([np.subtract(f["foot_board"][:2], f["tip_board"][:2]) for f in F], float)
            for j in np.unique(fi):
                m = np.median(off[np.abs(fi - j) <= FOOT_HALF], 0)
                if np.hypot(*m) <= FOOT_MAX:
                    self.foot_off[int(j)] = m

    def z(self, fi):
        return float(np.load(f"{self.scr}/{self.p}/top/{fi:04d}.npz")["z"])

    def sync(self, fi):
        i = int(np.argmin(np.abs(self.tk.T - self.ct[fi]["t"])))
        return self.tk.sync(i)

    def cam_time(self, c, k):
        j = int(np.searchsorted(self.tk.steps[c], k))
        return float(self.tk.ts[c][j] - self.tk.t0)

    def pitcher_pose(self, tt):
        """pitcher -> world (pitcher_model.py: X = (Pw - p) @ R  <=>  Pw = R X + p)."""
        tt = float(np.clip(tt, self.pt[0], self.pt[-1]))
        return self.slerp(tt).as_matrix(), np.array([np.interp(tt, self.pt, self.pP[:, 3 + j]) for j in range(3)])

    def plane_depth(self, c, z):
        """camera-z depth [m] of the plane world-z = z along every pixel ray (inf where the ray does not reach it)."""
        T = self.tk.c2w[c]
        dz = self.rays_w[c]
        s = (z - T[2, 3]) / np.where(np.abs(dz) < 1e-9, 1e-9, dz)
        return np.where(s > 0, s, np.inf).astype(np.float32)

    def project(self, c, Pw):
        K, W = self.tk.K[c], self.w2c[c]
        Pc = Pw @ W[:3, :3].T + W[:3, 3]
        return Pc[..., 0] / Pc[..., 2] * K[0, 0] + K[0, 2], Pc[..., 1] / Pc[..., 2] * K[1, 1] + K[1, 2], Pc[..., 2]


def pose_inliers(t, P, tip_p, win=POSE_WIN, tol=POSE_TOL, passes=2):
    """Temporal consistency of the tag-BA poses (tip.npz keeps the converged ones, a few gross outliers remain, e.g. leaf's
    last pose 146 mm off): the tip and the origin of each pose against a local linear fit in time of the other inlier
    poses within +-win; poses off by more than tol are dropped (interpolated over)."""
    R = Rotation.from_rotvec(P[:, :3]).as_matrix()
    tracks = (np.einsum("nij,j->ni", R, tip_p) + P[:, 3:], P[:, 3:])
    ok = np.ones(len(t), bool)
    for _ in range(passes):
        res = np.zeros(len(t))
        for i in range(len(t)):
            nb = ok & (np.abs(t - t[i]) <= win)
            nb[i] = False
            if nb.sum() < 2:
                continue
            A = np.stack([np.ones(nb.sum()), t[nb] - t[i]], 1)
            res[i] = max(np.linalg.norm(X[i] - np.linalg.lstsq(A, X[nb], rcond=None)[0][0]) for X in tracks)
        ok = res <= tol
    return ok


def pitcher_solid(p, scr, tip_p, n_phi=48):
    """Points of the pitcher solid in the pitcher frame: rings of the body of revolution (radius profile of
    raw_data/rig/results/pitcher_merged3.json, from the pour's wall bottom upwards) + the spout tip.  The body is convex
    (tapered cylinder) and the spout a wedge to the tip, so the convex hull of the projected points is the silhouette."""
    prof = json.load(open(os.path.join(RES, "pitcher_merged3.json")))["body_radius_profile_mm"]
    zz = sorted(int(k) for k in prof)
    zb = json.load(open(f"{scr}/{p}/pitcher_model.json"))["wall_bottom_mm"] * 1e-3
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    pts = [np.stack([prof[str(k)] * 1e-3 * np.cos(ph), prof[str(k)] * 1e-3 * np.sin(ph), np.full(n_phi, zb + k * 1e-3)], 1)
           for k in zz[::3] + [zz[-1]]]
    pts.append(np.asarray(tip_p, float)[None])
    return np.concatenate(pts)


# ------------------------------------------------------------------------------------------------ occluder masks
_DISK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MARGIN_PX + 1, 2 * MARGIN_PX + 1))
_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_PX, CLOSE_PX))


def near_mask(dep, zp):
    """Valid depth nearer than the plane by > OCC, closed, with the depth holes it (mostly) encloses filled."""
    near = ((dep > 0) & (dep < zp - OCC)).astype(np.uint8)
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, _CLOSE)
    miss = ((dep == 0) & (near == 0)).astype(np.uint8)
    n, lab = cv2.connectedComponents(miss, connectivity=8)
    if n > 1:
        ring = cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)   # max label next to a pixel
        rim = (miss == 0) & (ring > 0)
        tot = np.bincount(ring[rim], minlength=n)
        nr = np.bincount(ring[rim & (near > 0)], minlength=n)
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        fill = (nr >= HOLE_NEAR * np.maximum(tot, 1)) & (tot > 0)
        fill[border] = False
        fill[0] = False
        near |= fill[lab].astype(np.uint8)
    return near.astype(bool)


def capsule_hull(P, c, a, b, r, n=16):
    """Image convex hull of a capsule (radius r) between world points a and b."""
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    e1 = np.cross(d, [1.0, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = r * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)
    pts = np.concatenate([a + ring, b + ring, a - r * d + 0 * ring[:1], b + r * d + 0 * ring[:1]])
    return P.project(c, pts)


def _fill_hull(m, u, v, zc):
    ok = zc > 0.02
    if ok.sum() < 3:
        return m
    h = cv2.convexHull(np.stack([u[ok], v[ok]], 1).astype(np.float32))
    cv2.fillConvexPoly(m, np.round(h).astype(np.int32), 1)
    return m


def occluders(P, fi, c, k, dep, z):
    """Occluder mask of camera c (image coordinates) and its parts (near, pitcher, stream) before the margin."""
    zp = P.plane_depth(c, z)
    near = near_mask(dep, zp)
    tt = P.cam_time(c, k)
    R, pp = P.pitcher_pose(tt)
    u, v, zc = P.project(c, P.solid @ R.T + pp)
    pit = _fill_hull(np.zeros((H_IMG, W_IMG), np.uint8), u, v, zc).astype(bool)
    tip = R @ P.tip_p + pp
    jet = np.zeros((H_IMG, W_IMG), np.uint8)
    if z - tip[2] > 0:                                    # board z points into the table: tip above the plane
        # union of the stream straight down and the stream to the (running-median) measured foot
        feet = [np.array([tip[0], tip[1], z])]
        if fi in P.foot_off:
            feet.append(np.array([tip[0] + P.foot_off[fi][0], tip[1] + P.foot_off[fi][1], z]))
        for foot in feet:
            ju, jv, jz = capsule_hull(P, c, tip, foot, JET_R)
            jet = _fill_hull(jet, ju, jv, jz)
    # diagnostic: share of the pitcher silhouette where the camera has valid depth AT the plane (a clear line of sight
    # to the liquid -- the pitcher cannot be there); high values flag a wrong pose
    inp = pit & np.isfinite(zp)
    conflict = float(((dep > 0) & (np.abs(dep - zp) < OCC) & inp).sum() / max(inp.sum(), 1)) if inp.sum() > 500 else np.nan
    occ = near | pit | jet.astype(bool)
    occ = cv2.dilate(occ.astype(np.uint8), _DISK).astype(bool)
    return occ, dict(near=near, pitcher=pit, jet=jet.astype(bool), conflict=conflict)


# ------------------------------------------------------------------------------------------------ rectification
def grid(n=NS):
    """NS grid whose 2x2 means are the OUT video grid (pixel centres linspace(-EXT, EXT, OUT))."""
    d = 2 * EXT / (OUT - 1)
    s = -EXT - d / 4 + np.arange(n) * d / 2 if n == NS else np.linspace(-EXT, EXT, n)
    return np.meshgrid(s, s)


def rectify(P, fi):
    """Per camera: bilinear colour on the NS disc grid, usable mask, cup_register-rule visibility, ground sampling step."""
    z = P.z(fi)
    cen, L = np.array(P.D[fi]["disc_centre"]), P.D[fi]["L_m"]
    U, V = grid()
    Pw = np.stack([cen[0] + U * L, cen[1] + V * L, np.full_like(U, z)], -1).reshape(-1, 3)
    disc = np.hypot(U, V) <= 1.0
    views = []
    for c, k, dt in P.sync(fi):
        img = P.tk.color(c, k)
        dep = P.tk.depth(c, k).astype(np.float32) * 1e-3
        occ, parts = occluders(P, fi, c, k, dep, z)
        u, v, zc = P.project(c, Pw)
        u, v, zc = (a.reshape(NS, NS).astype(np.float32) for a in (u, v, zc))
        col = cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        inimg = (u >= 0) & (u <= W_IMG - 1) & (v >= 0) & (v <= H_IMG - 1) & (zc > 0)
        oc = cv2.remap(occ.astype(np.float32), u, v, cv2.INTER_LINEAR, borderValue=1.0) > 0.0
        dm = cv2.remap(dep, u, v, cv2.INTER_NEAREST)
        old_vis = inimg & ((dm == 0) | (dm > zc - OCC))                  # cup_register rule (fallback colour only)
        vis = inimg & ~oc
        sat = ((col.min(-1) > GLARE_T) & vis & disc).astype(np.uint8)
        nl, lab, st, _ = cv2.connectedComponentsWithStats(sat, 8)
        small = [j for j in range(1, nl) if st[j, cv2.CC_STAT_AREA] < GLARE_A * disc.sum()]
        glare = cv2.dilate(np.isin(lab, small).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        Cw = P.tk.c2w[c][:3, 3]
        X, Y = Pw[:, 0].reshape(NS, NS), Pw[:, 1].reshape(NS, NS)
        dvec = np.stack([Cw[0] - X, Cw[1] - Y, np.full_like(X, Cw[2] - z)], -1)
        cosang = np.abs(dvec[..., 2]) / np.linalg.norm(dvec, axis=-1)
        step = zc / P.tk.K[c][0, 0] / np.maximum(cosang, 0.2)                # worst-direction step (along the camera's tilt)
        views.append(dict(cam=c, k=k, dt=dt, col=col, use=vis & ~glare, old_vis=old_vis, step=step.astype(np.float32),
                          conflict=parts["conflict"]))
    return views, disc


# ------------------------------------------------------------------------------------------------ fusion
def _smooth_field(r, m):
    """Robust smooth field of r (NS, NS, 3) known where m: block medians, hole filling, median filter, upsample."""
    B = SMOOTH_B
    nb = NS // B
    rb = r.reshape(nb, B, nb, B, 3).transpose(0, 2, 1, 3, 4).reshape(nb, nb, B * B, 3)
    mb = m.reshape(nb, B, nb, B).transpose(0, 2, 1, 3).reshape(nb, nb, B * B)
    cnt = mb.sum(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.where(cnt[..., None] >= B * B // 4, np.nanmedian(np.where(mb[..., None], rb, np.nan), 2), np.nan)
    ok = np.isfinite(med[..., 0])
    if ok.sum() < 4:
        return np.zeros_like(r)
    out = np.zeros((nb, nb, 3), np.float32)
    pad = SMOOTH_K // 2
    for ch in range(3):
        a = np.where(ok, med[..., ch], 0).astype(np.float32)
        f = a.copy()
        for s in (1.0, 2.0, 4.0, 8.0):
            num = cv2.GaussianBlur(a * ok, (0, 0), s)
            den = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), s)
            fill = ~ok & (f == 0) & (den > 1e-3)
            f[fill] = (num / np.maximum(den, 1e-6))[fill]
            if not (~ok & (f == 0)).any():
                break
        fp = np.pad(f, pad, mode="edge")
        st = np.stack([fp[i:i + nb, j:j + nb] for i in range(SMOOTH_K) for j in range(SMOOTH_K)], -1)
        out[..., ch] = np.median(st, -1)
    return cv2.resize(out, (NS, NS), interpolation=cv2.INTER_LINEAR)


def _nanmedian0(A, M):
    """per-pixel median over the views where M (NaN where none)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.where(M[..., None] if A.ndim == M.ndim + 1 else M, A, np.nan), 0)


def _gain_fit(lC, M, nv0, local=True):
    """Per-camera log gain against the per-pixel median of the usable views, fitted on the pixels with >= GAIN_MINV usable
    views (>= 2 for a camera with < GAIN_MIN_PX of them): global per-channel offset g0 (median residual) + (local) the
    smooth field (_smooth_field) of the remaining residual, clamped to +-G_LOC.  Returns G (n, NS, NS, 3) and g0 (n, 3)."""
    n = len(lC)
    G = np.zeros_like(lC)
    g0 = np.zeros((n, 3))
    for it in range(2):
        ref = _nanmedian0(lC - G, M)
        for c in range(n):
            m = M[c] & (nv0 >= GAIN_MINV)
            if m.sum() < GAIN_MIN_PX:
                m = M[c] & (nv0 >= 2)
            if m.sum() < GAIN_MIN_PX:
                continue
            r = lC[c] - ref
            g0[c] = np.median(r[m], 0)
            G[c] = g0[c]
            if local:
                f = _smooth_field(np.where(m[..., None], r - g0[c], 0.0), m)
                G[c] = G[c] + np.clip(f, -G_LOC, G_LOC)
    return G, g0


def _no_consensus(A, M, two):
    """Pixels with exactly two usable views whose aligned log lightness (each blurred within its usable mask by DIS_SIGMA)
    differ by more than DIS: no consensus, neither view is trusted."""
    if not two.any():
        return np.zeros_like(two)
    lb = []
    for c in range(len(A)):
        m = M[c].astype(np.float32)
        num = cv2.GaussianBlur(A[c].mean(-1).astype(np.float32) * m, (0, 0), DIS_SIGMA)
        den = cv2.GaussianBlur(m, (0, 0), DIS_SIGMA)
        lb.append(np.where(M[c], num / np.maximum(den, 1e-6), np.nan))
    lb = np.stack(lb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nan_to_num(np.nanmax(lb, 0) - np.nanmin(lb, 0))
    return two & (spread > DIS)


def fuse_views(views, disc):
    """Gain-aligned, resolution / outlier weighted, feathered fusion of the usable views.
    Returns the fused BGR float image (NS), the usable-any mask and the per-camera usable masks."""
    C = np.stack([v["col"] for v in views])
    M = np.stack([v["use"] for v in views]) & disc[None]
    lC = np.log(C + 1.0)
    nv0 = M.sum(0)
    two = nv0 == 2
    # (1) global per-camera, per-channel offsets (exposure / white balance) on the consensus pixels
    g0 = _gain_fit(lC, M, nv0, local=False)[1]
    # (2) 2-view consensus test on the globally aligned views; disagreeing pairs do not enter the smooth gain field
    nocons = _no_consensus(lC - g0[:, None, None, :], M, two)
    Mg = M & ~nocons[None]
    # (3) smooth (bounded) gain field on the consensus pixels
    G = _gain_fit(lC, Mg, nv0, local=True)[0]
    A = lC - G
    step = np.stack([v["step"] for v in views])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = np.nan_to_num((np.nanmin(np.where(M, step, np.nan), 0)[None] / step) ** RES_POW)
    lA = A.mean(-1)
    lmed = _nanmedian0(lA, M)
    nvalid = M.sum(0)
    dev = np.where(nvalid[None] >= 3, np.abs(lA - lmed[None]), 0.0)
    out_w = np.exp(-(np.nan_to_num(dev) / TAU) ** 2)
    feather = np.stack([np.clip(cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3) / FEATHER_PX, 0.05, 1.0)
                        for m in M])
    W = M * res * out_w * feather
    Ws = W.sum(0)
    top = (W[..., None] * (np.exp(A) - 1.0)).sum(0) / np.maximum(Ws, 1e-9)[..., None]
    none = Ws <= 0
    if none.any():                                        # no usable view: cup_register rule colour (invalid anyway)
        Mo = np.stack([v["old_vis"] for v in views])
        fb = np.nan_to_num(_nanmedian0(C, Mo))
        top[none] = fb[none]
    return top, ~none, M


# ------------------------------------------------------------------------------------------------ video grid
def to_video(top, seen, M, D, fi):
    """NS fusion -> OUT frame: usable-weighted 2x2 mean, seen = > half of the 4 sub-pixels usable, nview = cameras usable
    on > half of the sub-pixels; then video_obs.observe()."""
    w = seen.astype(np.float32).reshape(OUT, SS, OUT, SS)
    t4 = top.reshape(OUT, SS, OUT, SS, 3)
    num = (t4 * w[..., None]).sum((1, 3))
    den = w.sum((1, 3))
    img = np.where(den[..., None] > 0, num / np.maximum(den, 1e-9)[..., None], t4.mean((1, 3)))
    seen128 = den / (SS * SS) > 0.5
    nview = (M.reshape(M.shape[0], OUT, SS, OUT, SS).mean((2, 4)) > 0.5).sum(0).astype(np.uint8)
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    im, valid = observe(img, seen128, D, fi)
    return im, valid, nview


def frame(P, fi):
    """Fused frame fi of P: img (BGR uint8, OUT), valid, nview, per-camera pitcher depth conflict."""
    views, disc = rectify(P, fi)
    top, seen, M = fuse_views(views, disc)
    im, valid, nview = to_video(top, seen, M, P.D, fi)
    return dict(img=im, valid=valid, nview=nview, conflict=[v["conflict"] for v in views])


def build(p, frames=None, log=print):
    """All frames of pour p (those with a top/<fi>.npz plane; or the subset `frames`) in the video_obs layout."""
    P = PourData(p)
    fis = [fi for fi in range(len(P.D)) if os.path.exists(f"{SCR}/{p}/top/{fi:04d}.npz")]
    if frames is not None:
        fis = [fi for fi in fis if fi in set(frames)]
    imgs, Ls, vals, nvs, ts, cfs = [], [], [], [], [], []
    tic = time.time()
    for fi in fis:
        r = frame(P, fi)
        imgs.append(r["img"][..., ::-1].copy())
        Ls.append(cv2.cvtColor(r["img"], cv2.COLOR_BGR2LAB)[..., 0])
        vals.append(r["valid"])
        nvs.append(r["nview"])
        cfs.append(r["conflict"])
        ts.append(P.D[fi]["t"] - P.t0)
    cf = np.array(cfs, float)
    log(f"{p}: {len(ts)} frames fused in {time.time() - tic:.0f} s; poses dropped at t = "
        f"{np.round(P.pose_rejected, 2).tolist()}; pitcher depth conflict per camera max {np.nanmax(cf, 0).round(2).tolist()}")
    return dict(t=np.array(ts), img=np.array(imgs), L=np.array(Ls), valid=np.array(vals), nview=np.array(nvs),
                frame_idx=np.array(fis), pitcher_depth_conflict=cf, pose_rejected_t=P.pose_rejected,
                source=f"{p}: fuse_occ.py (occluder masks in camera images: near depth + pitcher solid + stream, "
                       f"margin {MARGIN_PX} px; gain-aligned (3-view consensus, smooth part <= {G_LOC}) weighted fusion; "
                       f"cup_register plane, disc.json track)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pours", nargs="*", default=list(POURS))
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "fuse_occ"),
                    help="directory for <p>.npz (video_obs layout, frames only); video_obs.frames() writes the repo copy")
    ap.add_argument("--frames", default=None, help="a:b (frame index range) for a partial run")
    a = ap.parse_args(argv)
    fr = None
    if a.frames:
        lo, hi = map(int, a.frames.split(":"))
        fr = range(lo, hi)
    os.makedirs(a.out, exist_ok=True)
    for p in a.pours:
        F = build(p, fr)
        np.savez_compressed(os.path.join(a.out, f"{p}.npz"), w=np.zeros(F["L"].shape, np.uint8), extent=EXT, R_in=R_IN,
                            foot_r=FOOT_R, **F)
        print(f"-> {a.out}/{p}.npz")


if __name__ == "__main__":
    main()
