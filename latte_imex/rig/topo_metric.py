"""Topology-aware evaluation of a video whiteness W against the gap-preserving reference of the finished pattern
(topo_ref.py -> raw_data/rig/targets/<p>_ref.npz).  V2_SURFACE_MODEL.md 10.5.

    python3 -m latte_imex.rig.topo_metric [--viddir D] [--offsets F] [--resdir R] [--pours a,b] <tag> <dir>
    python3 -m latte_imex.rig.topo_metric [...] --sheet <tag> <dir>          -> R/<tag>_sheet.jpg
<dir>/W_<p>.npz: W float (n, H, H) in [0, 1], frames aligned with <viddir>/<p>.npz, extent 1.05, any H (seg_topo.py
writes this layout); <dir> = 'video_obs' scores the stored target w / 255 of the frames.  Optional key last_only=True:
only the last 3 frames are meaningful (temporal metrics skipped, reported as null).  Output R/<tag>.json (R: --resdir,
env TOPO_RESDIR, default <tmp>/topo_metric) and a table.

Reference at H (align-corners area resampling, pixel centres linspace(-EXT, EXT, n) as video_obs and the targets; the
512 reference moved by the frame's measured offset, topo_ref.offset / targets/frame_offsets.json): valid_ref = valid
fraction > 0.5, white_ref = white fraction > 0.5, gaps_ref = gap fraction > GAP_FRAC minus white_ref (keeps 1-2 mm
lines), skeleton = Zhang-Suen thinning of gaps_ref, faint_ref = the reference's sub-PSF hairlines (neutral: IoU
don't-care, accepted as gap-precision hits and boundary matches).  Everything inside m = valid_ref & video valid.
Last N_LAST frames, averaged:
  iou        IoU of W > 0.5 and white_ref (faint excluded)
  gapR       share of the gap skeleton where W < 0.5 (strict; json gap_recall_tol1: within 1 px)
  gapP       share of W's interior brown (built like the reference gaps: W upsampled to 512, > 0.5, closing with the
             2 mm disk minus white, pieces < 1 mm^2 dropped, area fraction > GAP_FRAC at H) within 1 px of gaps / faint
  gapF1      2 P R / (P + R)
  cPred, cRef, |dc|, |dc20|   8-connected white components >= 2 mm^2 (>= 20 mm^2: the pattern's parts)
  bf1        boundary F-score, 1 px tolerance (a boundary is a white pixel with a brown 4-neighbour inside m)
  down       the structure score as the calibration saw the video before run.video_on_grid area-averaged: solver grid of
             wsolver.Geo(48), W and valid at the NEAREST pixel, the reference as c (its white area fraction per cell),
             diff.score.score_parts; dRef = the reference itself through the same pipeline (a reference value)
  dArea      as down with W and valid AREA-averaged over each cell footprint (= run.video_on_grid now)
Whole video: early (white fraction of the first 3 frames), flip (class changes per pixel and frame), midfar (white
farther than 8 px at 128 from the final white_ref in the middle third), blink (one-frame class flips; json blink_px,
flip8_px = pixels flipping >= 8 times in the second half), p20L40 (parts >= 20 mm^2, mean over the last 40 frames),
crack (interior brown of W in the frames t < 4 s, summed, in 128-px units; counts real early gaps too: read it
relative to a baseline).
Thickness / part terms: mass (W > 0.5 over white_ref, last 3 frames, on m & not faint), ME = |ln mass|, PE = mean over
the last 40 frames of |parts20 - R0| / R0 (R0 = round(cRef20) >= 1), json pchg60 (part-count changes over the last 60
frames); J2 = (iou + bf1 + gapF1 + dArea + (1 - min(PE, 1)) + (1 - min(ME, 1))) / 6, the selection objective of
seg_topo.
Caveats: the reference is a different camera 0.5-2.9 s after the last frame; blurring it to the video's edge blur
(sigma ~1 mm) and thresholding gives gapR only 0.62-0.77, so gapR ~0.6-0.8 is about what the frames allow; heart has
~20 skeleton px, its gap terms are noisy; gapP rests on n_pb (json); part counts and crack depend on the valid mask.
"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..diff import score as SS
from .video_obs import footprint_mean

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TGT = os.path.join(REPO, "raw_data", "rig", "targets")
DEFAULT_VIDDIR = os.path.join(REPO, "raw_data", "rig", "video_obs")
VIDDIR = os.environ.get("VIDDIR", DEFAULT_VIDDIR)          # frames the W belongs to (valid, t, registration)
OFFSETS = os.environ.get("FRAME_OFFSETS", "")             # '' -> default rule (see frame_offsets)
RESDIR = os.environ.get("TOPO_RESDIR", os.path.join(tempfile.gettempdir(), "topo_metric"))

POURS = ("tulip", "leaf", "heart", "push_heart", "swan")
CAL, TEST = ("tulip", "leaf"), ("heart", "push_heart", "swan")
EXT = 1.05
L_M_MM = 42.0           # cup radius unit (disc.json L_m 41.5-42.3 mm)
N_LAST = 3              # last frames scored against the finished-pattern reference
WHITE_FRAC = 0.5        # reference white at H: area fraction of reference white > 0.5
GAP_FRAC = 0.25         # reference gap at H: area fraction of reference gaps > 0.25
R_GAP_MM = 2.0          # interior-brown closing radius (same physical radius as the reference gaps)
MIN_GAP_MM2 = 1.0       # interior-brown pieces smaller than this dropped (as topo_ref.py: closing residues at corners)
MIN_COMP_MM2 = 2.0      # components counted
BIG_COMP_MM2 = 20.0     # 'big' components (the pattern's parts, not specks): comps20_*
TOL = 1                 # [px at H] gap-precision and boundary tolerance (Chebyshev)
N_DOWN = 48             # solver grid (cmafit default --N 48)
CUP_R = 0.49            # wsolver.Geo / real_program.CUP_R: disc radius on the unit solver box [-0.5, 0.5]^2
MIDFAR_PX128 = 8        # midfar distance [px at 128]
N_PARTS_LAST = 40       # p20L40: mean parts >= 20 mm^2 over the last 40 frames
N_PARTS_SERIES = 60     # json parts20_series: the last 60 frames
CRACK_T = 4.0           # crack proxy: frames with t < 4 s
FLIP8_N = 8             # flip8_px: pixels flipping >= 8 times in the second half
_E8 = np.ones((3, 3), bool)
_E4 = ndi.generate_binary_structure(2, 1)


def disk(r):
    k = int(np.ceil(r))
    y, x = np.ogrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y) <= r * r + 1e-9


def grid(n):
    s = np.linspace(-EXT, EXT, n)
    return np.meshgrid(s, s)


def area_resample(a, n, shift=(0.0, 0.0)):
    """Area-weighted mean of a (m x m) on the n x n grid, both with pixel centres linspace(-EXT, EXT, .) (align corners:
    the video_obs and targets convention; video_obs.footprint_mean), optionally with the content of a moved by shift =
    (du, dv) [L_m].  Not cv2 INTER_AREA: it assumes pixel edges at +-EXT, which magnifies the 512 -> 128 result by
    0.6 % (the rim moves outwards by up to 1.5 target px = 0.26 mm, 0.37 px at 128)."""
    a = np.asarray(a, np.float32)
    if a.shape[0] == n and shift[0] == 0 and shift[1] == 0:
        return a.astype(float)
    U, V = grid(n)
    d = 2 * EXT / (n - 1) if n < a.shape[0] else 0.0
    return footprint_mean(a, U - shift[0], V - shift[1], d, None if d > 0 else 1)


def masked_mean_resample(x, v, n, shift=(0.0, 0.0)):
    num = area_resample(np.where(v, x, 0.0), n, shift)
    den = area_resample(v.astype(float), n, shift)
    return num / np.maximum(den, 1e-9), den


_OFF_FLY = {}


def frame_offsets(p):
    """{frame index: (du, dv) [L_m]} of the scored frames against the target (topo_ref.offset, image-to-image measurement,
    W not involved).  File: OFFSETS if given, else targets/frame_offsets.json for the default frames (topo_ref.py
    --offsets).  It is used when it holds the pour and its fingerprint of the frames (topo_ref.frames_fp; files without
    one are taken as given) matches VIDDIR; otherwise (other frames, no file, stale file) the offsets are measured now on
    the last N_LAST frames of VIDDIR."""
    from .topo_ref import frames_fp, offset
    fn = OFFSETS or (f"{TGT}/frame_offsets.json" if os.path.abspath(VIDDIR) == os.path.abspath(DEFAULT_VIDDIR) else "")
    if p not in _OFF_FLY:
        Z = np.load(f"{VIDDIR}/{p}.npz")
        F = json.load(open(fn)) if fn and os.path.exists(fn) else {}
        fp = F.get("_frames", {}).get(p)
        if p in F and fp in (None, frames_fp(Z)):
            _OFF_FLY[p] = F[p]
        else:
            T = np.load(f"{TGT}/{p}.npz")
            n = len(Z["t"])
            _OFF_FLY[p] = {str(k): list(offset(T, Z["img"][k], Z["valid"][k])[:2]) for k in range(n - N_LAST, n)}
            why = "stale offsets file (frames changed)" if p in F else "no offsets file"
            print(f"[topo_metric] {p}: frame offsets measured on {VIDDIR} ({why}: {fn or '-'})", flush=True)
    return {int(k): (v[0] / L_M_MM, v[1] / L_M_MM) for k, v in _OFF_FLY[p].items()}


class Ref:
    """Reference of pour p at H x H."""

    def __init__(self, p, H, shift=(0.0, 0.0)):
        R = np.load(f"{TGT}/{p}_ref.npz")
        self.p, self.H, self.shift = p, H, tuple(shift)
        self.px_mm = 2 * EXT * L_M_MM / (H - 1)          # pixel centres linspace(-EXT, EXT, H)
        wf, vf = masked_mean_resample(R["white"].astype(float), R["valid"], H, shift)
        self.wfrac = np.where(vf > 0.5, wf, 0.0)          # the reference as a soft W (ceiling of 'down')
        self.valid = vf > 0.5
        self.white = (wf > WHITE_FRAC) & self.valid
        gf, _ = masked_mean_resample(R["gaps"].astype(float), R["valid"], H, shift)
        self.gaps = (gf > GAP_FRAC) & self.valid & ~self.white
        self.skel = cv2.ximgproc.thinning(self.gaps.astype(np.uint8) * 255) > 0
        # faint hairlines (white in the reference, sub-PSF brown lines): neutral -- IoU don't-care, gap-precision hit
        ff, _ = masked_mean_resample(R["faint"].astype(float), R["valid"], H, shift)
        self.faint = (ff > GAP_FRAC) & self.valid & ~self.gaps
        self.fskel = cv2.ximgproc.thinning(self.faint.astype(np.uint8) * 255) > 0
        self.white512, self.valid512 = R["white"], R["valid"]
        self.n_comp512 = int(R["n_comp"])
        self.r_gap = R_GAP_MM / self.px_mm
        self.min_px = MIN_COMP_MM2 / self.px_mm ** 2
        self.min_gap_px = MIN_GAP_MM2 / self.px_mm ** 2
        self.big_px = BIG_COMP_MM2 / self.px_mm ** 2


def n_components(m, min_px):
    lab, n = ndi.label(m, structure=_E8)
    if not n:
        return 0
    return int((np.bincount(lab.ravel())[1:] >= min_px).sum())


def interior_brown(wb, m, r, min_px=0.0):
    """closing(wb, disk r) - wb inside m, pieces < min_px dropped (at the resolution of wb)."""
    pad = int(np.ceil(r)) + 2
    cl = ndi.binary_closing(np.pad(wb, pad), structure=disk(r))[pad:-pad, pad:-pad]
    pb = cl & ~wb & m
    if min_px > 0 and pb.any():
        lab, n = ndi.label(pb, structure=_E8)
        small = np.bincount(lab.ravel()) < min_px
        small[0] = False
        pb &= ~small[lab]
    return pb


REF_N = 512


def interior_brown_ref_res(w, m, ref):
    """Predicted interior brown with exactly the reference's gap construction: W bilinearly upsampled to the
    512 reference grid and thresholded at 0.5, closing with the reference's disk (round(R_GAP_MM / px) = 12 px), pieces
    < MIN_GAP_MM2 dropped, then back to H by area fraction > GAP_FRAC (as gaps_ref).  At H = 128 a 2 mm disk is
    2.9 px, which discretises to a 5 x 5 SQUARE: it closes diagonal channels up to ~4.9 mm wide, against 4.1 mm for the
    reference, and produced false interior brown even for the reference itself (ref_self leaf gapP 0.88)."""
    return interior_brown_H(w, m, ref.H)


def interior_brown_H(w, m, H):
    """interior_brown_ref_res for a W of size H (no reference needed)."""
    if H == REF_N:
        w5, m5 = w, m
    else:
        U, V = grid(REF_N)
        w5 = footprint_mean(np.asarray(w, np.float32), U, V, 0.0, S=1)
        m5 = footprint_mean(m.astype(np.float32), U, V, 0.0, S=1) > 0.5
    px5 = 2 * EXT * L_M_MM / (REF_N - 1)
    wb5 = (w5 > 0.5) & m5
    pb5 = interior_brown(wb5, m5, int(round(R_GAP_MM / px5)), MIN_GAP_MM2 / px5 ** 2)
    if H == REF_N:
        return pb5
    pf, _ = masked_mean_resample(pb5.astype(float), m5, H)
    return (pf > GAP_FRAC) & m & ~((w > 0.5) & m)


def boundary(wb, m):
    return wb & m & ndi.binary_dilation(~wb & m, structure=_E4)


def near(x, tol=TOL):
    return ndi.binary_dilation(x, structure=np.ones((2 * tol + 1, 2 * tol + 1), bool)) if x.any() else x


def frame_metrics(w, vv, ref):
    """Per-frame metrics of W (w, H x H) with video valid vv against the (registered) reference ref: iou, gap recall /
    precision / F1, component counts, bf1 and the json-only extras (module docstring)."""
    m = vv & ref.valid
    wb = (w > 0.5) & m
    rw = ref.white & m
    out = {}
    dc = ref.faint                                   # don't care in the IoU
    out["iou"] = float((wb & rw & ~dc).sum() / max(((wb | rw) & ~dc).sum(), 1))
    sk = ref.skel & m
    out["gap_recall"] = float((sk & ~wb).sum() / sk.sum()) if sk.any() else float("nan")
    # registration-tolerant variant (json only): a skeleton pixel counts if some pixel within TOL is brown in W
    brown_near = ndi.binary_erosion(wb | ~m, structure=np.ones((2 * TOL + 1,) * 2, bool), border_value=1)
    out["gap_recall_tol1"] = float((sk & ~brown_near).sum() / sk.sum()) if sk.any() else float("nan")
    pb = interior_brown_ref_res(w, m, ref)
    out["gap_precision"] = float((pb & near(ref.gaps | ref.faint)).sum() / pb.sum()) if pb.any() else float("nan")
    out["n_pb"] = int(pb.sum())                      # json only: gapP rests on this many predicted interior-brown px
    fs = ref.fskel & m                               # json only: faint hairlines shown brown (not required)
    out["faint_recall"] = float((fs & ~wb).sum() / fs.sum()) if fs.any() else float("nan")
    r_, p_ = out["gap_recall"], out["gap_precision"]
    out["gap_f1"] = float(2 * p_ * r_ / (p_ + r_)) if (np.isfinite(p_) and np.isfinite(r_) and p_ + r_ > 0) else 0.0
    out["comps_pred"] = n_components(wb, ref.min_px)
    out["comps_ref"] = n_components(rw, ref.min_px)
    out["comps_absdiff"] = abs(out["comps_pred"] - out["comps_ref"])
    out["comps20_pred"] = n_components(wb, ref.big_px)
    out["comps20_ref"] = n_components(rw, ref.big_px)
    out["comps20_absdiff"] = abs(out["comps20_pred"] - out["comps20_ref"])
    out["n_skel"] = int(sk.sum())
    bp, br = boundary(wb, m), boundary(rw, m)
    P = (bp & near(br | ref.faint)).sum() / max(bp.sum(), 1)       # an edge on a faint hairline is not an error
    Rr = (br & near(bp)).sum() / max(br.sum(), 1)
    out["bf1"] = float(2 * P * Rr / (P + Rr)) if P + Rr > 0 else 0.0
    return out


class SolverGrid:
    """The N_DOWN solver grid exactly as latte_imex.diff (wsolver.Geo cell centres on [-0.5, 0.5]^2, disc CUP_R) and the
    sampling of run.video_on_grid: U = x / CUP_R, V = -y / CUP_R, the video W and valid at the NEAREST pixel of the
    linspace(-EXT, EXT, H) grid, inside = U^2 + V^2 <= 1 & disc.  The reference plays the simulation: its white area
    fraction over each cell footprint (h / CUP_R wide), cells with reference valid fraction > 0.5."""

    def __init__(self, ref):
        h = 1.0 / N_DOWN
        c = (np.arange(N_DOWN) + 0.5) * h - 0.5
        x = np.repeat(c[None, :], N_DOWN, 0); y = np.repeat(c[:, None], N_DOWN, 1)
        disc = np.hypot(x, y) <= CUP_R
        U, V = x / CUP_R, -y / CUP_R
        self.inside = (U ** 2 + V ** 2 <= 1.0) & disc
        H = ref.H
        self.col = np.clip(np.round((U + EXT) / (2 * EXT) * (H - 1)).astype(int), 0, H - 1)
        self.row = np.clip(np.round((V + EXT) / (2 * EXT) * (H - 1)).astype(int), 0, H - 1)
        d = h / CUP_R
        su, sv = ref.shift
        num = footprint_mean(np.where(ref.valid512, ref.white512, False).astype(np.float32), U - su, V - sv, d)
        den = footprint_mean(ref.valid512.astype(np.float32), U - su, V - sv, d)
        self.c = num / np.maximum(den, 1e-9)
        self.cvalid = den > 0.5
        self.w_ceil = ref.wfrac[self.row, self.col]
        self.v_ceil = ref.valid[self.row, self.col]
        self.U, self.V, self.d = U, V, d
        self.ref_wfrac, self.ref_valid = ref.wfrac, ref.valid

    def area(self, w, v):
        """W and valid on the cells by the AREA mean over each cell footprint (d = h / CUP_R wide, S x S bilinear
        sub-samples, footprint_mean), valid if > half the footprint is valid (combined/down_test.py Cells.area, zero
        offset)."""
        num = footprint_mean(np.where(v, w, 0.0).astype(np.float32), self.U, self.V, self.d)
        den = footprint_mean(v.astype(np.float32), self.U, self.V, self.d)
        return num / np.maximum(den, 1e-9), (den > 0.5) & self.inside & self.cvalid


def down_grid(sg, w, vv):
    """score.score_parts(c = reference cells, w = W at the cell centres (nearest), valid) and the ceiling: the same call with
    w = the reference's own white fraction at H sampled the same way (what a W equal to the reference scores)."""
    valid = vv[sg.row, sg.col] & sg.inside & sg.cvalid
    parts = SS.score_parts(sg.c, w[sg.row, sg.col], valid)
    ceil = SS.score(sg.c, sg.w_ceil, valid & sg.v_ceil)
    return float(parts["score"]), float(ceil), float(parts["F_white"]), float(parts["F_brown"])


def down_area_grid(sg, w, vv):
    """down with area-mean sampling (SolverGrid.area) and its reference value (the reference's own white fraction at H
    through the same sampler, cells valid for both)."""
    wa, va = sg.area(w, vv)
    parts = SS.score_parts(sg.c, wa, va)
    wr, vr = sg.area(sg.ref_wfrac.astype(np.float32), sg.ref_valid)
    ceil = SS.score(sg.c, wr, va & vr)
    return float(parts["score"]), float(ceil), float(parts["F_white"]), float(parts["F_brown"])


def resample_valid(V, H):
    if V.shape[-1] == H:
        return V.astype(bool)
    return np.array([area_resample(v.astype(np.float32), H) > 0.5 for v in V])


def temporal_extras(W, V, t, ref):
    """blink, flip8_px, p20L40 / parts20_series, crack (module docstring)."""
    n, H = W.shape[0], W.shape[-1]
    wb = W > 0.5
    out = {}
    bl, bpx = [], []
    for k in range(1, n - 1):
        m = V[k - 1] & V[k] & V[k + 1]
        if not m.any():
            continue
        b = m & (wb[k - 1] == wb[k + 1]) & (wb[k] != wb[k - 1])
        bl.append(b.sum() / m.sum()); bpx.append(b.sum())
    out["blink"] = float(np.mean(bl)) if bl else float("nan")
    out["blink_px"] = float(np.mean(bpx)) if bpx else float("nan")
    cnt = np.zeros((H, H), int)
    for k in range(n // 2, n - 1):
        cnt += ((wb[k] ^ wb[k + 1]) & V[k] & V[k + 1])
    out["flip8_px"] = int((cnt >= FLIP8_N).sum())
    ser = [n_components(wb[k] & V[k] & ref.valid, ref.big_px) for k in range(max(0, n - N_PARTS_SERIES), n)]
    out["parts20_series"] = [int(x) for x in ser]
    out["parts20_last40"] = float(np.mean(ser[-N_PARTS_LAST:]))
    early = [k for k in range(n) if t[k] < CRACK_T]
    cr = [int(interior_brown_H(W[k], V[k], H).sum()) if V[k].sum() >= 50 else 0 for k in early]
    s128 = ((H - 1) / 127.0) ** 2                                  # pixels at H per 128-px
    out["crack"] = float(np.sum(cr) / s128)
    out["crack_mm2_per_frame"] = float(np.mean(cr) * (2 * EXT * L_M_MM / (H - 1)) ** 2) if cr else float("nan")
    out["crack_series"] = [int(x) for x in cr]
    return out


def load_w(p, d, Z):
    """(W, last_only) of pour p: <d>/W_<p>.npz, or the stored target w / 255 of the frames if d == 'video_obs'."""
    if d == "video_obs":
        return Z["w"].astype(float) / 255.0, False
    D = np.load(f"{d}/W_{p}.npz")
    return D["W"].astype(float), (bool(D["last_only"]) if "last_only" in D.files else False)


def evaluate_pour(p, d):
    """All metrics of pour p for the W of d (load_w): the last N_LAST frames against the per-frame registered reference
    (frame_metrics, down, dArea, mass_ratio / ME), the whole-video terms (temporal_extras, early, flip, midfar) and PE,
    pchg60, J2.  Returns a flat dict (json keys)."""
    Z = np.load(f"{VIDDIR}/{p}.npz")
    W, last_only = load_w(p, d, Z)
    n, H = W.shape[0], W.shape[-1]
    assert n == len(Z["t"]), f"{p}: W has {n} frames, video_obs {len(Z['t'])}"
    V = resample_valid(Z["valid"], H)
    ref = Ref(p, H)                                  # unshifted: midfar
    off = frame_offsets(p)                           # per-frame reference registration (topo_ref.offset)
    refs = {k: Ref(p, H, off.get(k, (0.0, 0.0))) for k in range(n - N_LAST, n)}
    rows = [frame_metrics(W[k], V[k], refs[k]) for k in range(n - N_LAST, n)]
    out = {k: float(np.nanmean([r[k] for r in rows])) if any(np.isfinite(r[k]) for r in rows) else float("nan")
           for k in rows[0]}
    ds = [down_grid(SolverGrid(refs[k]), W[k], V[k]) for k in range(n - N_LAST, n)]
    out["down"] = float(np.mean([x[0] for x in ds])); out["down_self"] = float(np.mean([x[1] for x in ds]))
    out["down_Fw"] = float(np.mean([x[2] for x in ds])); out["down_Fb"] = float(np.mean([x[3] for x in ds]))
    da = [down_area_grid(SolverGrid(refs[k]), W[k], V[k]) for k in range(n - N_LAST, n)]
    out["down_area"] = float(np.mean([x[0] for x in da])); out["down_area_self"] = float(np.mean([x[1] for x in da]))
    out["down_area_Fw"] = float(np.mean([x[2] for x in da])); out["down_area_Fb"] = float(np.mean([x[3] for x in da]))
    out["comps_ref512"] = ref.n_comp512
    mw = mr = 0                                      # thickness: white mass against the reference (registered, not faint)
    for k in range(n - N_LAST, n):
        m = refs[k].valid & V[k] & ~refs[k].faint
        mw += int(((W[k] > 0.5) & m).sum()); mr += int((refs[k].white & m).sum())
    out["mass_ratio"] = float(mw / max(mr, 1))
    out["ME"] = float(abs(np.log(max(out["mass_ratio"], 1e-3))))
    if last_only:
        out.update(early_white=float("nan"), flip=float("nan"), midfar=float("nan"), blink=float("nan"),
                   parts20_last40=float("nan"), crack=float("nan"), blink_px=float("nan"), flip8_px=float("nan"),
                   crack_mm2_per_frame=float("nan"))
    else:
        out.update(temporal_extras(W, V, Z["t"], ref))
        wb = W > 0.5
        out["early_white"] = float(np.mean([(wb[k] & V[k]).sum() / max(V[k].sum(), 1) for k in range(3)]))
        fl = [((wb[k] ^ wb[k + 1]) & V[k] & V[k + 1]).sum() / max((V[k] & V[k + 1]).sum(), 1) for k in range(n - 1)]
        out["flip"] = float(np.mean(fl))
        far = ndi.distance_transform_edt(~ref.white) > MIDFAR_PX128 * H / 128.0
        mid = range(n // 3, 2 * n // 3)
        out["midfar"] = float(np.mean([(wb[k] & V[k] & far).sum() / max((V[k] & far).sum(), 1) for k in mid]))
    if last_only:
        out.update(PE=float("nan"), pchg60=float("nan"), J2=float("nan"))
    else:
        r0 = max(1, int(round(out["comps20_ref"])))
        ser = np.array(out["parts20_series"])
        out["PE"] = float(np.mean(np.abs(ser[-N_PARTS_LAST:] - r0)) / r0)
        out["pchg60"] = int((np.diff(ser) != 0).sum())
        out["J2"] = float((out["iou"] + out["bf1"] + out["gap_f1"] + out["down_area"] + (1 - min(out["PE"], 1.0))
                           + (1 - min(out["ME"], 1.0))) / 6)
    out["H"] = H
    out["n_frames"] = n
    return out


KEYS = ["iou", "gap_recall", "gap_precision", "gap_f1", "comps_pred", "comps_ref", "comps_absdiff", "comps20_absdiff",
        "bf1", "down", "down_self", "early_white", "flip", "midfar", "down_area", "blink", "parts20_last40", "comps20_ref",
        "crack", "mass_ratio", "PE", "J2"]
HDR = ["iou", "gapR", "gapP", "gapF1", "cPred", "cRef", "|dc|", "|dc20|", "bf1", "down", "dRef", "early", "flip",
       "midfar", "dArea", "blink", "p20L40", "p20ref", "crack", "mass", "PE", "J2"]
JKEYS = KEYS + ["gap_recall_tol1", "faint_recall", "comps20_pred", "down_Fw", "down_Fb", "n_skel", "n_pb",
                "down_area_self", "down_area_Fw", "down_area_Fb", "blink_px", "flip8_px", "crack_mm2_per_frame", "ME",
                "pchg60"]
FMT = {"blink": "{:7.5f}", "crack": "{:7.0f}"}


def mean_row(res, ps):
    return {k: float(np.nanmean([res[p][k] for p in ps])) if any(np.isfinite(res[p][k]) for p in ps) else float("nan")
            for k in JKEYS}


def fmt(v, k=None):
    if not np.isfinite(v):
        return "    - " if k not in FMT else "     - "
    return FMT[k].format(v) if k in FMT else f"{v:6.3f}"


def table(tag, res, pours=POURS):
    """Text table of res (run): one row per pour and the mean / cal / test rows, columns HDR."""
    lines = [f"[{tag}]  last {N_LAST} frames vs reference (cRef at W resolution; 512 reference counts in json); "
             f"frames {VIDDIR}",
             f"{'pour':11s}" + "".join(f"{h:>7s}" if k not in FMT else f"{h:>8s}" for h, k in zip(HDR, KEYS))]
    for p in pours:
        lines.append(f"{p:11s}" + "".join(f" {fmt(res[p][k], k)}" for k in KEYS))
    for nm in ("mean", "cal", "test"):
        if nm in res:
            lines.append(f"{nm:11s}" + "".join(f" {fmt(res[nm][k], k)}" for k in KEYS))
    return "\n".join(lines)


def run(tag, d, pours=POURS):
    """evaluate_pour for each pour, mean / cal / test rows -> RESDIR/<tag>.json (with the settings) and the printed table."""
    res = {p: evaluate_pour(p, d) for p in pours}
    for nm, ps in (("mean", POURS), ("cal", CAL), ("test", TEST)):
        if all(p in res for p in ps):
            res[nm] = mean_row(res, ps)
    os.makedirs(RESDIR, exist_ok=True)
    meta = dict(tag=tag, dir=d if d == "video_obs" else os.path.abspath(d), viddir=os.path.abspath(VIDDIR), offsets=OFFSETS or "default",
                n_last=N_LAST, white_frac=WHITE_FRAC, gap_frac=GAP_FRAC,
                r_gap_mm=R_GAP_MM, min_comp_mm2=MIN_COMP_MM2, min_gap_mm2=MIN_GAP_MM2, tol_px=TOL, midfar_px128=MIDFAR_PX128,
                crack_t=CRACK_T, n_parts_last=N_PARTS_LAST,
                version="align-corners resampling, per-frame reference registration (targets/frame_offsets.json), "
                        "solver-grid down with nearest W sampling and ceiling down_self, interior brown at the reference "
                        "resolution, faint hairlines neutral; selectable frames (VIDDIR), down_area, blink, p20L40, crack; "
                        "mass_ratio, ME, PE, pchg60, J2")
    clean = lambda d: {k: (clean(v) if isinstance(v, dict) else (None if isinstance(v, float) and not np.isfinite(v) else v))
                       for k, v in d.items()}
    json.dump(dict(meta=meta, results=clean(res)), open(f"{RESDIR}/{tag}.json", "w"), indent=1)
    t = table(tag, res, pours)
    print(t)
    return res


def sheet(tag, d, size=256):
    """Contact sheet of the last frame per pour -> RESDIR/<tag>_sheet.jpg: video frame | W | W against the registered
    reference (white agree, red W-only white, blue reference-only white, green gap skeleton, yellow faint hairlines)."""
    rows = []
    for p in POURS:
        if d != "video_obs" and not os.path.exists(f"{d}/W_{p}.npz"):
            continue
        Z = np.load(f"{VIDDIR}/{p}.npz")
        W = load_w(p, d, Z)[0]
        n, H = W.shape[0], W.shape[-1]
        V = resample_valid(Z["valid"][-1:], H)[0]
        ref = Ref(p, H, frame_offsets(p).get(n - 1, (0.0, 0.0)))
        img = cv2.resize(Z["img"][-1][..., ::-1], (size, size), interpolation=cv2.INTER_NEAREST)
        wv = (np.clip(W[-1], 0, 1) * 255).astype(np.uint8)
        wv = cv2.cvtColor(wv, cv2.COLOR_GRAY2BGR); wv[~V] = (60, 0, 60)
        wv = cv2.resize(wv, (size, size), interpolation=cv2.INTER_NEAREST)
        m = V & ref.valid
        wb = (W[-1] > 0.5) & m
        ov = np.zeros((H, H, 3), np.uint8)
        ov[~m] = (40, 40, 40)
        ov[wb & ref.white] = (255, 255, 255)
        ov[wb & ~ref.white & m] = (0, 0, 255)          # pred white, ref brown: red
        ov[~wb & ref.white & m] = (255, 120, 0)        # pred brown, ref white: blue
        ov = cv2.resize(ov, (size, size), interpolation=cv2.INTER_NEAREST)
        sc = size / H
        for y, x in zip(*np.nonzero(ref.skel & m)):
            cv2.circle(ov, (int((x + 0.5) * sc), int((y + 0.5) * sc)), max(1, int(sc * 0.3)), (0, 220, 0), -1)
        for y, x in zip(*np.nonzero(ref.fskel & m)):
            cv2.circle(ov, (int((x + 0.5) * sc), int((y + 0.5) * sc)), max(1, int(sc * 0.2)), (0, 220, 255), -1)
        fm = frame_metrics(W[-1], V, ref)
        cv2.putText(img, f"{p} last frame", (4, 14), 0, 0.45, (0, 255, 255), 1)
        cv2.putText(wv, f"{tag} W", (4, 14), 0, 0.45, (0, 255, 255), 1)
        cv2.putText(ov, f"IoU {fm['iou']:.2f} gapR {fm['gap_recall']:.2f} c {fm['comps_pred']}/{fm['comps_ref']}",
                    (4, 14), 0, 0.42, (0, 255, 255), 1)
        rows.append(np.hstack([img, wv, ov]))
    os.makedirs(RESDIR, exist_ok=True)
    fn = f"{RESDIR}/{tag}_sheet.jpg"
    cv2.imwrite(fn, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(fn)


if __name__ == "__main__":
    a = sys.argv[1:]
    pours = POURS
    while a and a[0] in ("--viddir", "--offsets", "--resdir", "--pours"):     # options override the env vars
        opt, val, a = a[0], a[1], a[2:]
        if opt == "--viddir":
            VIDDIR = val
        elif opt == "--offsets":
            OFFSETS = val
        elif opt == "--resdir":
            RESDIR = val
        else:
            pours = tuple(val.split(","))
    if a and a[0] == "--sheet":
        sheet(a[1], a[2])
    elif len(a) == 2:
        run(a[0], a[1], pours)
    else:
        print(__doc__)
        sys.exit(1)
