"""Structure-aware, position-tolerant score of a simulated frame against a video frame (V2_SURFACE_MODEL.md §9).

score(c, w, valid) -> [0, 1], higher = better; c: simulated mixing fraction (soft, N x N), w: video white fraction (soft),
valid: observed cells.  Built for the N = 48 solver grid (cell ~1.8 mm); set_grid(N) keeps R, SHIFT, SPECK, A and MIN_WHITE at
the same physical size on other grids (SIG_C stays in cells: it is the solver's own resolution).
  0. cells outside `valid` take the value of the nearest valid cell (no fake edges at occluders); specks of <= SPECK
     8-connected cells flipped (both); the simulation alone smoothed by SIG_C cells (the solver does not resolve
     single-cell features; without it grid noise in c is rewarded)
  1. soft white      m = clip((x - LO) / (HI - LO), 0, 1)  (= mean of the hard masks x > tau, tau in [LO, HI])
  2. interior brown  b = closing(m, disk R) - m             (brown flanked by white within R cells: lines, layer gaps,
                                                             notches)
  3. per class K in {m, b}: F_K = (sum min(p, N(o)) + sum min(o, N(p)) + 2A) / (sum p + sum o + 2A),
     N(x) = (x + dilate(x, radius 1)) / 2: the share of class mass in sim and video with a same-class partner in the
     same cell (full credit) or one cell away (half credit); A = 1 add-one smoothing
  4. score = max over the rigid shifts |s| <= SHIFT cells (~4 mm) of the sim of sqrt(F_white F_brown)
Validated on the five pours (scratchpad harness): a single disc with the right area and place, the video with its brown
lines filled, or with the lines redrawn at random, all score clearly below the video against itself 0.25 s later (IoU
fails all three); shifts of 1-2 cells cost less than a 0.25 s lag, 5 cells clearly more; robust to speckle and to
occluder borders.  Not differentiable (use CMA-ES, diff/cmafit.py).

pour_scores / lag_reference: the calibration objective -- frame means over the frames with enough observed white,
normalised by the video's own 0.25 s self-consistency (1 = as close as the video is to itself 0.25 s later).
"""
import numpy as np
from scipy import ndimage as ndi
from numpy.lib.stride_tricks import sliding_window_view

LO, HI = 0.3, 0.7      # soft-threshold ramp
R = 2                  # closing radius that defines interior brown (cells)
A = 1.0                # additive prior per class (add-one smoothing, in cells)
SHIFT = 2              # rigid alignment search radius (cells)
SPECK = 2              # specks of <= SPECK cells are removed (both images)
SIG_C = 0.7            # resolution filter on the simulation only (cells)

REF_N = 48             # the grid the constants above are set for
_BASE = dict(R=R, SHIFT=SHIFT, SPECK=SPECK, A=A)

_E8 = np.ones((3, 3), bool)


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


_DR = _disk(R)
_D1 = _disk(1)
_OFF = [(dy, dx) for dy in range(-SHIFT, SHIFT + 1) for dx in range(-SHIFT, SHIFT + 1) if dy * dy + dx * dx <= SHIFT * SHIFT]
_OY = np.array([SHIFT - dy for dy, dx in _OFF]); _OX = np.array([SHIFT - dx for dy, dx in _OFF])


def _despeckle(x):
    b = x > 0.5
    out = x.copy()
    for ph, val in ((b, 0.0), (~b, 1.0)):
        lab, n = ndi.label(ph, structure=_E8)
        if n:
            small = np.bincount(lab.ravel()) <= SPECK
            small[0] = False
            if small.any():
                out[small[lab]] = val
    return out


def _classes(x):
    m = np.clip((x - LO) * (1.0 / (HI - LO)), 0.0, 1.0)
    b = np.clip(ndi.grey_closing(m, footprint=_DR) - m, 0.0, 1.0)
    return m, b


def _near(x):
    return 0.5 * (x + ndi.grey_dilation(x, footprint=_D1))


def score_parts(c, w, valid):
    v = np.asarray(valid, bool)
    c = np.asarray(c, float); w = np.asarray(w, float)
    if not v.any():
        return dict(score=1.0, F_white=1.0, F_brown=1.0, shift=(0, 0))
    if (~v).any():
        idx = tuple(ndi.distance_transform_edt(~v, return_distances=False, return_indices=True))
        c = c[idx]; w = w[idx]
    if SPECK:
        c = _despeckle(c); w = _despeckle(w)
    if SIG_C > 0:
        c = ndi.gaussian_filter(c, SIG_C, mode="nearest")
    vf = v.astype(float)
    (ms, bs), (mv, bv) = _classes(c), _classes(w)
    nms, nbs, nmv, nbv = _near(ms), _near(bs), _near(mv) * vf, _near(bv) * vf
    mv = mv * vf; bv = bv * vf
    anym = (nms > 0) | (nbs > 0) | (nmv > 0) | (nbv > 0)
    if not anym.any():
        return dict(score=1.0, F_white=1.0, F_brown=1.0, shift=(0, 0))
    S = SHIFT
    ys = np.flatnonzero(anym.any(1)); xs = np.flatnonzero(anym.any(0))
    y0, y1 = max(ys[0] - S, 0), min(ys[-1] + S + 1, v.shape[0])
    x0, x1 = max(xs[0] - S, 0), min(xs[-1] + S + 1, v.shape[1])
    H, W = y1 - y0, x1 - x0
    win = lambda a: sliding_window_view(np.pad(a[y0:y1, x0:x1], S), (H, W))[_OY, _OX]   # (21, H, W) shifted sims
    cr = lambda a: a[y0:y1, x0:x1]
    vc = cr(vf)

    def tolF(p, npp, o, no):          # F_K for all 21 shifts of the sim at once
        P = win(p); NP = win(npp); oc = cr(o); noc = cr(no)
        num = np.minimum(P, noc).sum((1, 2)) + np.minimum(NP, oc).sum((1, 2)) + 2 * A
        den = (P * vc).sum((1, 2)) + oc.sum() + 2 * A
        return num / den

    Fw = tolF(ms, nms, mv, nmv)
    Fb = tolF(bs, nbs, bv, nbv)
    k = int(np.argmax(Fw * Fb))
    return dict(score=float(np.sqrt(Fw[k] * Fb[k])), F_white=float(Fw[k]), F_brown=float(Fb[k]), shift=_OFF[k])


def score(c, w, valid):
    return score_parts(c, w, valid)["score"]


# ---------------------------------------------------------------------------- whole-video use
MIN_WHITE = 40          # observed white cells (w > 0.5) a video frame needs to count
_MIN_WHITE0 = MIN_WHITE


def set_grid(N):
    """Scale the cell-based constants to grid N so that they keep their physical size (lengths x N/48, areas x
    (N/48)^2).  At N = 96 with the N = 48 constants the video's own 0.25 s self-consistency drops (tulip 0.92 -> 0.84,
    leaf 0.89 -> 0.79); scaled, it is 0.94 / 0.91 (V2_SURFACE_MODEL.md 10.5)."""
    global R, SHIFT, SPECK, A, MIN_WHITE, _DR, _OFF, _OY, _OX
    f = N / REF_N
    R = max(1, int(round(_BASE['R'] * f)))
    SHIFT = max(1, int(round(_BASE['SHIFT'] * f)))
    SPECK = int(round(_BASE['SPECK'] * f * f))
    A = _BASE['A'] * f * f
    MIN_WHITE = int(round(_MIN_WHITE0 * f * f))
    _DR = _disk(R)
    _OFF = [(dy, dx) for dy in range(-SHIFT, SHIFT + 1) for dx in range(-SHIFT, SHIFT + 1) if dy * dy + dx * dx <= SHIFT * SHIFT]
    _OY = np.array([SHIFT - dy for dy, dx in _OFF]); _OX = np.array([SHIFT - dx for dy, dx in _OFF])
MIN_OBS = 0.6           # ... and the observed share of the cup
LAG = 0.25              # [s] the video self-consistency reference


def frame_ok(w, valid, mask):
    v = np.asarray(valid, bool)
    return ((np.asarray(w) > 0.5) & v).sum() >= MIN_WHITE and v.sum() >= MIN_OBS * np.asarray(mask, bool).sum()


def iou(c, w, valid):
    v = np.asarray(valid, bool)
    a, b = (np.asarray(c) > 0.5) & v, (np.asarray(w) > 0.5) & v
    return float((a & b).sum() / max((a | b).sum(), 1))


def lag_reference(t, W, V, mask, lag=LAG):
    """The video against itself `lag` s later (frames with a partner within 35 % of the lag, both frame_ok):
    (structure score, IoU) means -- the per-pour normalisers."""
    t = np.asarray(t, float)
    st, io = [], []
    for i in range(len(t)):
        j = int(np.argmin(np.abs(t - (t[i] + lag))))
        if j <= i or abs(t[j] - t[i] - lag) > 0.35 * lag or not (frame_ok(W[i], V[i], mask) and frame_ok(W[j], V[j], mask)):
            continue
        v = np.asarray(V[i], bool) & np.asarray(V[j], bool)
        st.append(score(W[i], W[j], v))
        io.append(iou(W[i], W[j], v))
    return (float(np.mean(st)), float(np.mean(io))) if st else (1.0, 1.0)


def pour_scores(C, W, V, wt, mask, ref=None):
    """Time-weighted frame means of the structure score and the IoU over the frame_ok frames; with ref = (structure,
    IoU) of lag_reference also the normalised values."""
    ok = np.array([frame_ok(W[i], V[i], mask) for i in range(len(W))])
    wt = np.asarray(wt, float) * ok
    if wt.sum() <= 0:
        return dict(struct=0.0, viou=0.0, n=0)
    wt = wt / wt.sum()
    st = sum(wt[i] * score(np.clip(C[i], 0, 1), W[i], V[i]) for i in np.flatnonzero(ok))
    io = sum(wt[i] * iou(C[i], W[i], V[i]) for i in np.flatnonzero(ok))
    out = dict(struct=float(st), viou=float(io), n=int(ok.sum()))
    if ref is not None:
        out.update(struct_n=float(st / ref[0]), viou_n=float(io / ref[1]))
    return out
