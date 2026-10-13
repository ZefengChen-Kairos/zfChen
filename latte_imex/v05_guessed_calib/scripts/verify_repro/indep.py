"""indep.py -- independent re-implementation of the final-frame metrics of the guessed-action calibration (written from
the definitions in prereg.json and the compare.py docstring, without calling compare.py / cf_runner / the driver's loss):

  target      latte_imex/targets/<p>.npz (w, mask) read directly (heart, tulip: PHI = 0, no frame correction needed)
  sim -> grid model c_M [iy, ix] (M frame, y up), NaN outside the sim cup -> filled with the nearest cup cell
              (KD-tree), then bilinear with clamped indices at p = R(-theta) (q - d); q = physical position of target
              cell (row r, col j) in the T frame = (g_j, -g_r), g = cell centres on [-0.5, 0.5]
  IoU / Dice  (s > thr) & cup vs (w > 0.5) & cup
  boundary    white cell with a 4-neighbour (inside the array) that is non-white and inside the cup; symmetric mean
              nearest-boundary Euclidean distance (KD-tree), cell size 1 / M_t, in D
  alignment   L_fin: argmax IoU(c > 0.4), theta = -25 + 2.5 k deg, dx, dy = (j - 6) / 100 D, first maximum in loop order
              (theta, dx, dy); 'best' (compare.best_align definition): 1 deg / 0.01 D grid, then 0.25 deg / 0.0025 D
              refinement around it (clipped to the window), strict '>' scan
  L_fin       (1 - Dice) + d_bnd / 0.05 + mean_l |n_m - n_v| / max(n_v, 1); n = structnum.stripes(...)['total'] (the only
              reused function: brown-seam detection, as the task asks)
"""
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

REVIEW = os.environ["V05GCAL_REVIEW"]                          # review dir (copy: path from the environment)
sys.path.insert(0, os.path.join(REVIEW, "v05real"))
import structnum as SN                                        # noqa: E402  (seam counts only)

TARGETS = "/home/user/zfChen/latte_imex/targets"
D_MM = 80.0


def load_target(p):
    z = np.load(os.path.join(TARGETS, f"{p}.npz"))
    return z["w"].astype(np.float64), z["mask"].astype(bool)


def fill_nearest(cM, mM):
    """NaN-free field: cup values kept, every other cell = value of the nearest cup cell (KD-tree)."""
    f = np.where(mM, np.nan_to_num(cM), 0.0)
    ii, jj = np.nonzero(mM)
    oi, oj = np.nonzero(~mM)
    if len(oi):
        _, k = cKDTree(np.c_[ii, jj]).query(np.c_[oi, oj])
        f[oi, oj] = f[ii[k], jj[k]]
    return f


def bilinear(f, px, py):
    N = f.shape[0]
    I = np.clip((py + 0.5) * N - 0.5, 0.0, N - 1.0)
    J = np.clip((px + 0.5) * N - 0.5, 0.0, N - 1.0)
    i0 = np.minimum(np.floor(I).astype(int), N - 2)
    j0 = np.minimum(np.floor(J).astype(int), N - 2)
    ti, tj = I - i0, J - j0
    return ((1 - ti) * ((1 - tj) * f[i0, j0] + tj * f[i0, j0 + 1])
            + ti * ((1 - tj) * f[i0 + 1, j0] + tj * f[i0 + 1, j0 + 1]))


class Grid:
    def __init__(self, cM, mM, p, fill=None):
        self.f = fill_nearest(cM, mM) if fill is None else fill
        self.w, self.cup = load_target(p)
        self.Mt = self.w.shape[0]
        g = (np.arange(self.Mt) + 0.5) / self.Mt - 0.5
        self.qx = np.broadcast_to(g[None, :], (self.Mt, self.Mt))
        self.qy = np.broadcast_to(-g[:, None], (self.Mt, self.Mt))
        self.vw = (self.w > 0.5) & self.cup
        self.r, self.c = np.nonzero(self.cup)
        self.cqx, self.cqy = self.qx[self.r, self.c], self.qy[self.r, self.c]
        self.cvw = self.vw[self.r, self.c]

    def field(self, th=0.0, dx=0.0, dy=0.0, cup_only=False):
        t = math.radians(th)
        ct, st = math.cos(t), math.sin(t)
        qx, qy = (self.cqx, self.cqy) if cup_only else (self.qx, self.qy)
        ux, uy = qx - dx, qy - dy
        return bilinear(self.f, ct * ux + st * uy, -st * ux + ct * uy)

    def iou_cup(self, th, dx, dy, thr=0.4):
        a = self.field(th, dx, dy, cup_only=True) > thr
        u = np.count_nonzero(a | self.cvw)
        return np.count_nonzero(a & self.cvw) / u if u else float("nan")


def boundary(A, cup):
    nb = np.zeros_like(A)
    X = ~A & cup
    nb[1:, :] |= X[:-1, :]
    nb[:-1, :] |= X[1:, :]
    nb[:, 1:] |= X[:, :-1]
    nb[:, :-1] |= X[:, 1:]
    return A & nb


def bnd_dist(A, B, cup):
    Mt = A.shape[0]
    bA, bB = np.argwhere(boundary(A, cup)), np.argwhere(boundary(B, cup))
    if len(bA) == 0 or len(bB) == 0:
        return float("nan"), float("nan"), float("nan")
    ab = cKDTree(bB).query(bA)[0].mean() / Mt
    ba = cKDTree(bA).query(bB)[0].mean() / Mt
    return 0.5 * (ab + ba), ab, ba


def seams(s, cup, thr):
    return [int(L["total"]) for L in SN.stripes(s, cup, thr)]


def terms(G, s, n_vid):
    a = (s > 0.4) & G.cup
    a5 = (s > 0.5) & G.cup
    na, nb, ni = int(a.sum()), int(G.vw.sum()), int((a & G.vw).sum())
    dice = 2 * ni / (na + nb)
    iou = ni / int((a | G.vw).sum())
    iou5 = int((a5 & G.vw).sum()) / int((a5 | G.vw).sum())
    bd, ab, ba = bnd_dist(a, G.vw, G.cup)
    bnd = 0.5 if not np.isfinite(bd) else bd
    nm = seams(s, G.cup, 0.4)
    S = float(np.mean([abs(m - v) / max(v, 1) for m, v in zip(nm, n_vid)]))
    return dict(L=(1 - dice) + bnd / 0.05 + S, dice=dice, iou=iou, iou_c05=iou5, bnd_D=bnd, bnd_mm=bnd * D_MM,
                bnd_c05_mm=bnd_dist(a5, G.vw, G.cup)[0] * D_MM, S=S, n_model=nm,
                white_frac=na / int(G.cup.sum()))


def align_prereg(G):
    best = (-1.0, None)
    for k in range(21):
        th = -25.0 + 2.5 * k
        for i in range(13):
            dx = round((i - 6) / 100.0, 4)
            for j in range(13):
                dy = round((j - 6) / 100.0, 4)
                v = G.iou_cup(th, dx, dy)
                if v > best[0]:
                    best = (v, (th, dx, dy))
    return best


def align_compare(G):
    best = (-1.0, None)
    for k in range(51):
        th = -25.0 + 1.0 * k
        for i in range(13):
            dx = round((i - 6) / 100.0, 4)
            for j in range(13):
                dy = round((j - 6) / 100.0, 4)
                v = G.iou_cup(th, dx, dy)
                if v > best[0]:
                    best = (v, (th, dx, dy))
    th0, dx0, dy0 = best[1]
    for a in range(-4, 5):
        th = min(max(th0 + 0.25 * a, -25.0), 25.0)
        for b in range(-4, 5):
            dx = min(max(dx0 + 0.0025 * b, -0.06), 0.06)
            for c in range(-4, 5):
                dy = min(max(dy0 + 0.0025 * c, -0.06), 0.06)
                v = G.iou_cup(th, dx, dy)
                if v > best[0]:
                    best = (v, (th, dx, dy))
    return best


def evaluate(cM, mM, p, fill=None, compare_align=True):
    G = Grid(cM, mM, p, fill)
    n_vid = seams(G.w, G.cup, 0.5)
    out = dict(video=dict(n=n_vid, white_frac=float(G.vw.sum() / G.cup.sum())))
    out["raw"] = terms(G, G.field(), n_vid)
    v, (th, dx, dy) = align_prereg(G)
    al = terms(G, G.field(th, dx, dy), n_vid)
    out["lfin"] = dict(al, transform=dict(theta_deg=th, dx_D=dx, dy_D=dy, iou_search=v))
    if compare_align:
        v, (th, dx, dy) = align_compare(G)
        b = terms(G, G.field(th, dx, dy), n_vid)
        out["best"] = dict(b, transform=dict(theta_deg=th, dx_D=dx, dy_D=dy, iou_search=v),
                           white_ratio=b["white_frac"] / out["video"]["white_frac"])
    out["raw"]["white_ratio"] = out["raw"]["white_frac"] / out["video"]["white_frac"]
    return out
