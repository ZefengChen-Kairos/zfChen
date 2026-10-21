"""Per-frame validation of a calibrated closure against the gap-preserving video target (V2_SURFACE_MODEL.md 10.5):
frame by frame, does the simulated white field keep the brown gaps the video keeps (tulip: teardrop / heart / wings
separated by brown lines)?

    python3 -m latte_imex.diff.validate <closure.json> --out DIR [--pours tulip,leaf,...] [--N 48] [--frames 8]
                                        [--e0 world|jet|none] [--device cpu]
-> DIR/<p>_frames.json (one record per observed frame), DIR/summary.json (per pour), DIR/<p>_sheet.jpg.

Forward run = the calibration's: cmafit._init / evaluate / run_fields (DiffPour tau 0.05, scales (0.3, 0.15, 0), score
constants at the grid's physical size, the video's 0.25 s lag reference recomputed on the frames); parameters =
DiffPour.s1 + the surface-current defaults + the closure's numbers, switches e0 / exit_cal / upwell / circ / circ_L /
circ_rf / circ_r from the closure (as cmafit --init and the best_closure.json it writes).  The time-weighted frame_ok means of
iou / struct below are checked against evaluate() (summary 'calib_check').

Per observed frame (run.video_on_grid: exposure transients and frames after the simulated time are not observed; the
sim state after the solver frame k whose end time is nearest the video time, cmafit.run_fields):
  solver grid (what the calibration scores): iou, struct (+ F_white, F_brown, shift [cells]), iou_n / struct_n (over
    the video's 0.25 s lag reference), frame_ok (score.frame_ok: counted by the calibration), wt (time weight), white
    area of sim / video on the observed cells [mm^2]
  video grid H = 128 (0.69 mm / px; the grid of video_obs, topo_metric and the stored target W): the sim c clipped to
    [0, 1] (as score.pour_scores), cells outside the disc given the nearest disc cell, bilinear from the cell centres to
    the pixel centres (cell (x, y) is the image point U = x / CUP_R, V = -y / CUP_R, run.video_on_grid), white = c > 0.5
    (the IoU's threshold, the centre of the score's soft ramp LO..HI).  The score's resolution filter SIG_C is not
    applied: it only widens what one cell already cannot resolve.
    Gap topology against the video frame W itself (not the finished-pattern reference): W plays the reference --
    white = W > 0.5, gaps = topo_metric.interior_brown_H(W) (topo_ref's construction: W at 512, > 0.5, closing with the
    2 mm disk, pieces < 1 mm^2 dropped, area fraction > GAP_FRAC at H), Zhang-Suen skeleton, no faint hairlines -- and
    topo_metric.frame_metrics is applied unchanged: gap_recall (video gap skeleton where the sim is brown; _tol1: brown
    within 1 px -- neither has the structure score's +-2 cell shift search),
    gap_precision (sim interior brown within 1 px of the video gaps), gap_f1, bf1 (boundary F, 1 px), iou_H, comps /
    comps20 (8-connected white components >= 2 / >= 20 mm^2) of sim and video, all inside m = video valid & disc.
    The gap terms count on frames whose video gap skeleton has >= MIN_SKEL px (gap_def); topo_metric's convention
    gap_f1 = 0 when the prediction has no interior brown is kept (gap_recall alone is 1 for a sim without any white).
  ceilings, same frame, same function:
    w48  the calibration target itself (the video's area mean on the solver cells, unobserved cells given the nearest
         observed cell) brought to H exactly like the sim: what a simulation equal to its target on the solver grid
         scores -- the limit of the grid, not of the physics
    lag  the video 0.25 s later (score.LAG, partner within 35 % of the lag as in score.lag_reference, without its
         frame_ok condition) as the prediction: the video's own temporal consistency, the yardstick of the
         calibration's normalisation (none in the last 0.25 s)
    (w48 is the key for the target on the N grid whatever N is.)
  threshold sensitivity (thr, gap_def frames): the sim's gap terms, comps20 and part separation with white = c > 0.4
    and c > 0.6 (THR; inside the score's soft ramp LO..HI): the sim's c is soft across its brown lines, so its gap F1
    depends on the threshold; the ceilings stay at 0.5.
  part separation (sep_sim, sep_w48): every pair of video white parts >= 20 mm^2; each part is represented by the
    prediction's 8-connected white component covering most of it (>= SEP_COVER of the part, else missing); the pair is
    kept apart when the two representatives differ, merged when they coincide -- the tulip question (teardrop / heart /
    wings) frame by frame, which equal part counts alone do not answer (two wrong splits count like two right ones).
  gap_width: the video's gap pieces at 512 (the step inside interior_brown_H), width = 2 x distance transform - 1 px on
    their skeleton: median [mm] and the share of the skeleton narrower than one solver cell (gap_sub_cell).
  last N_LAST frames (topo_metric.N_LAST): the same metrics against the finished-pattern reference (targets/<p>_ref.npz,
    moved by the frame's measured offset: topo_metric.frame_offsets / targets/frame_offsets.json) for the sim (ref_sim),
    the w48 ceiling (ref_w48) and the video W (ref_video = topo_metric's video_obs row).
Alignment check (landing_in_occlusion_by_lag, LAGS): share of the active frames whose simulated landing point lies in
the video's occlusion (stream / pitcher, unobserved) at video frame j + lag; time-aligned runs peak at lag 0
(landing_peak: the peak lag [frames, s]; > 0: the simulated landing point is ahead of the video).  Coarse: the
occlusion is the pitcher as well as the stream (base rate ~0.1).
Summary per pour: calibration-weighted iou / struct, plain means over the observed frames and over the last 40
(topo_metric.N_PARTS_LAST) observed frames, gap terms over the gap_def frames (sim, w48, lag; sim_f1_thr: the sim at
the THR thresholds), comps20 sim / video / reference and the share of frames whose sim count equals the video's, the
share of video part pairs kept apart (pooled over the last 40 frames), the last-N_LAST reference rows.  summary.json
collects the pours of one closure / N / e0: a run with another one starts the file afresh.
Sheet: K observed frames from the first frame_ok one to the last; rows video frame (+ the sim's landing point, cyan) /
video W / sim c (the part the video does not see dimmed) / overlay (white both, red sim only, blue video only, brown
both; video gap skeleton green where the sim keeps it brown, yellow where the sim fills it; magenta sim interior brown
off the video gaps); below, the time series (iou, struct, white area, gap F1 with its ceilings and the sim's THR band,
comps20, part pairs kept apart).
Speed: topo_metric.interior_brown_H (scipy closing with a 25 px disk at 512) is swapped in-process for GapTwin (the same
remap tables precomputed, the same closing by cv2); the twin is compared pixel for pixel with the original on the first
N_TWIN gap frames of every pour (summary 'twin_check') and the original is used if it ever differs.
Resolution: N = 48 cells are h / CUP_R x 42 mm = 1.79 mm wide; the video's layer gaps are 1-2 mm (gap_width).  A brown
line narrower than a cell cannot appear in the cell field at full contrast, so read the sim's gap_f1 against w48, not
against 1; the w48 numbers are what N = 48 can show at all.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import cmafit as CF
from . import schedule as SC
from . import score as SS
from ..rig import topo_metric as TM

PATTERNS = CF.PATTERNS
EXT, CUP_R, L_M_MM = TM.EXT, TM.CUP_R, TM.L_M_MM
MIN_SKEL = 10           # [px at H] video gap skeleton a frame needs for its gap terms to count (~7 mm of line)
N_LAST40 = TM.N_PARTS_LAST
N_TWIN = 3              # gap frames per pour on which GapTwin is compared with topo_metric.interior_brown_H
W_IOU = 0.25            # cmafit default objective weight (per-pour objective in the summary)
SEP_COVER = 0.25        # a prediction component represents a video part if it covers >= this share of the part
THR = (0.4, 0.6)        # sim white thresholds of the sensitivity rows (the IoU's 0.5 is the main row)
LAGS = range(-12, 13)   # [video frames] lags of the landing-point alignment check (~ +-0.6 to +-1 s)
THR_KEYS = ("gap_recall", "gap_precision", "gap_f1", "comps20_pred")
GAP_KEYS = ("gap_recall", "gap_recall_tol1", "gap_precision", "gap_f1", "bf1", "iou_H", "comps20_pred", "comps_pred",
            "n_pb")
REF_KEYS = ("iou", "gap_recall", "gap_recall_tol1", "gap_precision", "gap_f1", "bf1", "comps20_pred", "comps20_ref",
            "comps_pred", "comps_ref", "n_skel")
_IB_ORIG = TM.interior_brown_H


# ------------------------------------------------------------------------------------------------ gap construction
class GapTwin:
    """topo_metric.interior_brown_H for W of size H: the remap tables of video_obs.footprint_mean (H -> 512 bilinear,
    512 -> H area mean) computed once with the same expressions, the closing with the same disk and padding by cv2
    instead of scipy.  Same output (checked per pour), ~8x faster.  self.pb5: the 512 interior brown of the last call."""

    def __init__(self, H):
        self.H = H
        n = TM.REF_N
        U, V = TM.grid(n)
        sp = 2 * EXT / (H - 1)
        self.up = (((U + 0.0 + EXT) / sp).astype(np.float32), ((V + 0.0 + EXT) / sp).astype(np.float32))
        Uh, Vh = TM.grid(H)
        d = 2 * EXT / (H - 1)
        sp5 = 2 * EXT / (n - 1)
        S = 2 * int(np.ceil(d / sp5)) + 2 if d > sp5 else 1
        self.S = S
        self.mx = [((Uh + ((i + 0.5) / S - 0.5) * d + EXT) / sp5).astype(np.float32) for i in range(S)]
        self.my = [((Vh + ((j + 0.5) / S - 0.5) * d + EXT) / sp5).astype(np.float32) for j in range(S)]
        self.px5 = 2 * EXT * L_M_MM / (n - 1)
        self.r = int(round(TM.R_GAP_MM / self.px5))
        self.kern = TM.disk(self.r).astype(np.uint8)
        self.min_px = TM.MIN_GAP_MM2 / self.px5 ** 2
        self.pb5 = None

    def _upH(self, a):
        return np.zeros(self.up[0].shape) + cv2.remap(np.asarray(a, np.float32), self.up[0], self.up[1], cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_REPLICATE)

    def _down(self, a):
        a = np.asarray(a, np.float32)
        acc = np.zeros((self.H, self.H))
        for j in range(self.S):
            for i in range(self.S):
                acc += cv2.remap(a, self.mx[i], self.my[j], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return acc / (self.S * self.S)

    def __call__(self, w, m, H=None):
        assert H in (None, self.H)
        w5 = self._upH(w)
        m5 = self._upH(m.astype(np.float32)) > 0.5
        wb5 = (w5 > 0.5) & m5
        pad = self.r + 2
        cl = cv2.morphologyEx(np.pad(wb5, pad).astype(np.uint8), cv2.MORPH_CLOSE, self.kern)[pad:-pad, pad:-pad] > 0
        pb = cl & ~wb5 & m5
        if self.min_px > 0 and pb.any():
            lab, n = ndi.label(pb, structure=TM._E8)
            small = np.bincount(lab.ravel()) < self.min_px
            small[0] = False
            pb &= ~small[lab]
        self.pb5 = pb
        num = self._down(np.where(m5, pb.astype(float), 0.0))
        den = self._down(m5.astype(float))
        pf = num / np.maximum(den, 1e-9)
        return (pf > TM.GAP_FRAC) & m & ~((w > 0.5) & m)


@contextlib.contextmanager
def gap_impl(f):
    """topo_metric.interior_brown_H (looked up by frame_metrics at call time) replaced by f inside the block."""
    TM.interior_brown_H = f
    try:
        yield
    finally:
        TM.interior_brown_H = _IB_ORIG


class FrameRef:
    """A video frame W (H x H) as a topo_metric.Ref: white = W > 0.5, gaps = interior_brown_H(W) (topo_ref's gap rule at
    512), Zhang-Suen skeleton of the gaps, no faint hairlines; gap width from the 512 gap pieces (twin only)."""

    def __init__(self, w, v, gaps_fn):
        H = w.shape[0]
        self.H = H
        self.px_mm = 2 * EXT * L_M_MM / (H - 1)
        self.valid = v
        self.white = (w > 0.5) & v
        self.gaps = gaps_fn(w, v, H)
        self.skel = cv2.ximgproc.thinning(self.gaps.astype(np.uint8) * 255) > 0
        self.faint = np.zeros_like(v)
        self.fskel = self.faint
        self.min_px = TM.MIN_COMP_MM2 / self.px_mm ** 2
        self.big_px = TM.BIG_COMP_MM2 / self.px_mm ** 2
        self.width_mm = None
        pb5 = getattr(gaps_fn, "pb5", None)
        if pb5 is not None and pb5.any():
            sk = cv2.ximgproc.thinning(pb5.astype(np.uint8) * 255) > 0
            if sk.any():
                self.width_mm = (2 * ndi.distance_transform_edt(pb5)[sk] - 1) * gaps_fn.px5


def part_separation(pred, vid, m, big_px):
    """Pairs of video white parts (vid > 0.5 inside m, 8-connected, >= big_px) and how the prediction treats them
    (module docstring): dict(parts, pairs, kept, merged, missing)."""
    lv, nv = ndi.label((vid > 0.5) & m, structure=TM._E8)
    sz = np.bincount(lv.ravel())
    parts = [i for i in range(1, nv + 1) if sz[i] >= big_px]
    lp, npr = ndi.label((pred > 0.5) & m, structure=TM._E8)
    rep = []
    for i in parts:
        ov = np.bincount(lp[lv == i], minlength=npr + 1)
        ov[0] = 0
        rep.append(int(np.argmax(ov)) if ov.max() >= SEP_COVER * sz[i] else 0)
    out = dict(parts=len(parts), pairs=0, kept=0, merged=0, missing=0)
    for a in range(len(rep)):
        for b in range(a + 1, len(rep)):
            out["pairs"] += 1
            out["missing" if rep[a] == 0 or rep[b] == 0 else ("merged" if rep[a] == rep[b] else "kept")] += 1
    return out


def landing_peak(by_lag, tv):
    """Peak of landing_check: lag [frames] and [s] (median video frame interval), its share and the share at lag 0."""
    ok = {int(d): s for d, s in by_lag.items() if np.isfinite(s)}
    if not ok:
        return None
    d = max(ok, key=lambda x: (ok[x], -abs(x)))
    return dict(lag_frames=d, lag_s=float(d * np.median(np.diff(tv))), share=ok[d], share_0=ok.get(0, float("nan")))


def landing_check(frames, Vv, lags=LAGS, r_max=0.8):
    """Time-alignment check: share of the active frames (landing point within r_max disc radii) whose simulated landing
    point lies in the video's unobserved area (the stream / pitcher occlusion, fuse_occ) of video frame j + lag, per
    lag [frames].  Aligned frames peak at lag 0; a peak at lag > 0 puts the simulated landing point ahead of the video.
    Only frames with a partner at every lag count, so the shares of all lags rest on the same frames."""
    H = Vv.shape[-1]
    out = {}
    frames = [f for f in frames if all(0 <= f["j"] + d < len(Vv) for d in lags)]
    for d in lags:
        hit = []
        for f in frames:
            if not f["land_UV"] or f["land_UV"][0] ** 2 + f["land_UV"][1] ** 2 > r_max ** 2 or not 0 <= f["j"] + d < len(Vv):
                continue
            col, row = (int(round((x + EXT) / (2 * EXT) * (H - 1))) for x in f["land_UV"])
            hit.append(not Vv[f["j"] + d][row, col])
        out[str(d)] = float(np.mean(hit)) if hit else float("nan")
    return out


def to_H(c, keep, H):
    """Cell field c (N x N, wsolver.Geo layout) on the H x H video grid: cells outside `keep` take the nearest kept cell,
    then bilinear between the cell centres (U = x / CUP_R, V = -y / CUP_R)."""
    N = c.shape[0]
    idx = ndi.distance_transform_edt(~keep, return_distances=False, return_indices=True)
    cf = np.asarray(c, float)[tuple(idx)]
    U, V = TM.grid(H)
    i = (U * CUP_R + 0.5) * N - 0.5
    j = (-V * CUP_R + 0.5) * N - 0.5
    return ndi.map_coordinates(cf, [j, i], order=1, mode="nearest")


def _pick(m, keys):
    return {k: m[k] for k in keys if k in m}


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _nanmean(a):
    a = np.array([np.nan if v is None else v for v in a], float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


# ------------------------------------------------------------------------------------------------ closure
def load_closure(fn, e0=None):
    """(numbers, switches) of a closure json (cmafit best_closure.json or latte_imex/data/closures/*.json)."""
    J = json.load(open(fn))
    num = {k: float(v) for k, v in J.items() if isinstance(v, (int, float)) and not k.startswith("_")}
    sw = dict(e0=SC.e0_mode(e0 if e0 is not None else num.get("e0", 0.0)), exit_cal=bool(num.get("exit_cal", 0.0)),
              upwell=bool(num.get("upwell", 0.0)), circ=bool(num.get("circ", 0.0)),
              circ_shape=dict(L=num.get("circ_L", 0.4), r_f=num.get("circ_rf", 0.06), r=num.get("circ_r", 1.0)))
    return num, sw


def closure_params(D, num, circ):
    """cmafit main's P: DiffPour.s1, surface-current start (or off), then the closure's numbers."""
    from .fit import CIRC_START
    P = dict(D.s1)
    P.update(CIRC_START if circ else {"kappa_M": 0.0, "eps_f": 0.0, "kappa_w": 0.0})
    P.update(num)
    return P


# ------------------------------------------------------------------------------------------------ one pour
def validate_pour(p, num, sw, N=48, device="cpu", log=print):
    """Forward run of pour p with the closure, then the per-frame metrics (module docstring).  Returns (frames, meta,
    images) -- images: per frame index the arrays the sheet draws."""
    t0 = time.time()
    cfg = dict(pours=[p], N=N, device=device, e0=sw["e0"], exit_cal=sw["exit_cal"], upwell=sw["upwell"], circ=sw["circ"],
               circ_shape=sw["circ_shape"])
    CF._init(cfg)
    D = CF._D[p]
    P = closure_params(D, num, sw["circ"])
    t1 = time.time()
    r = CF.evaluate(P, [p])[p]
    if "err" in r:
        raise RuntimeError(f"{p}: forward run failed: {r['err']}")
    C, Wc, Vc, wt = CF.run_fields(D)
    t2 = time.time()
    log(f"{p}: init {t1 - t0:.0f} s, forward {t2 - t1:.0f} s; calibration view: video IoU {r['viou']:.3f} "
        f"({r['viou_n']:.2f}), structure {r['struct']:.3f} ({r['struct_n']:.2f}), {r['n']} frame_ok frames")
    from ..rig.real_program import REPO
    Z = np.load(os.path.join(REPO, "raw_data", "rig", "video_obs", f"{p}.npz"))      # the file run.video_on_grid reads
    if os.path.abspath(TM.VIDDIR) != os.path.abspath(TM.DEFAULT_VIDDIR):
        log(f"[validate] warning: topo_metric VIDDIR {TM.VIDDIR} is not the frames the calibration reads")
    tv, Wv, Vv, img = Z["t"], Z["w"].astype(float) / 255.0, Z["valid"], Z["img"]
    H = Wv.shape[-1]
    Uh, Vh = TM.grid(H)
    disc = Uh ** 2 + Vh ** 2 <= 1.0
    px_mm = 2 * EXT * L_M_MM / (H - 1)
    cell_mm = D.geo.h / CUP_R * L_M_MM
    msk = D.geo.mask_np
    t_end = np.cumsum(D.S["fdt"])
    tobs = np.array([o["t"] for o in D.obs])
    js = [int(np.argmin(np.abs(tv - o["t"]))) for o in D.obs]
    off = TM.frame_offsets(p)
    twin = GapTwin(H)
    use_twin, twin_check = True, []
    frames, images = [], {}
    for i, o in enumerate(D.obs):
        j, k = js[i], int(o["k"])
        c = np.clip(C[i], 0.0, 1.0)
        wc, vc = Wc[i], Vc[i]
        sp = SS.score_parts(c, wc, vc)
        io = SS.iou(C[i], wc, vc)
        ok = bool(SS.frame_ok(wc, vc, msk))
        act = bool(D.S["active"][k])
        rec = dict(i=i, j=j, t=float(tv[j]), k=k, t_sim=float(t_end[k]), wt=float(wt[i]), frame_ok=ok, active=act,
                   land_UV=[float(D.S["x"][k] / CUP_R), float(-D.S["y"][k] / CUP_R)] if act else None,
                   obs_frac=float(vc.sum() / msk.sum()), iou=io, struct=sp["score"], F_white=sp["F_white"],
                   F_brown=sp["F_brown"], shift=list(sp["shift"]), iou_n=io / D.score_ref[1],
                   struct_n=sp["score"] / D.score_ref[0],
                   white_mm2_sim_N=float(((C[i] > 0.5) & vc).sum() * cell_mm ** 2),
                   white_mm2_video_N=float(((wc > 0.5) & vc).sum() * cell_mm ** 2))
        m = Vv[j] & disc
        su = to_H(c, msk, H)
        wu = to_H(wc, vc, H)
        rec["white_mm2_sim"] = float(((su > 0.5) & m).sum() * px_mm ** 2)
        rec["white_mm2_video"] = float(((Wv[j] > 0.5) & m).sum() * px_mm ** 2)
        if use_twin and len(twin_check) < N_TWIN:
            g0 = _IB_ORIG(Wv[j], m, H)
            if g0.any():                                 # pixel-for-pixel check of the twin on frames with gaps
                g1 = twin(Wv[j], m, H)
                s0, s1 = _IB_ORIG(su, m, H), twin(su, m, H)
                bad = int((g0 != g1).sum() + (s0 != s1).sum())
                twin_check.append(dict(j=j, px_differ=bad))
                if bad:
                    use_twin = False
                    log(f"[validate] {p} frame {j}: GapTwin differs from topo_metric by {bad} px; using the original")
        gfn = twin if use_twin else _IB_ORIG
        with gap_impl(gfn):
            fr = FrameRef(Wv[j], m, gfn)
            ms = TM.frame_metrics(su, m, fr)
            mw = TM.frame_metrics(wu, m, fr)
            il = int(np.argmin(np.abs(tobs - (tobs[i] + SS.LAG))))
            lag = None
            if il > i and abs(tobs[il] - tobs[i] - SS.LAG) <= 0.35 * SS.LAG:
                ml = TM.frame_metrics(Wv[js[il]], Vv[js[il]] & disc, fr)
                lag = dict(_pick(ml, GAP_KEYS), j=js[il], comps20_pred=ml["comps20_pred"])
                lag["iou_H"] = ml["iou"]
            if j in off:                                 # last N_LAST frames: the finished-pattern reference
                ref = TM.Ref(p, H, off[j])
                rec["ref_sim"] = _pick(TM.frame_metrics(su, Vv[j], ref), REF_KEYS)
                rec["ref_w48"] = _pick(TM.frame_metrics(wu, Vv[j], ref), REF_KEYS)
                rec["ref_video"] = _pick(TM.frame_metrics(Wv[j], Vv[j], ref), REF_KEYS)
                rec["ref_offset_mm"] = [off[j][0] * L_M_MM, off[j][1] * L_M_MM]
            if ms["n_skel"] >= MIN_SKEL:                 # threshold sensitivity: c > thr == c + 0.5 - thr > 0.5
                rec["thr"] = {f"{t:.1f}": dict(_pick(TM.frame_metrics(su + (0.5 - t), m, fr), THR_KEYS),
                                               sep=part_separation(su + (0.5 - t), Wv[j], m, fr.big_px)) for t in THR}
        rec.update(n_skel=ms["n_skel"], gap_def=bool(ms["n_skel"] >= MIN_SKEL), iou_H=ms["iou"],
                   gap_recall=ms["gap_recall"], gap_recall_tol1=ms["gap_recall_tol1"],
                   gap_precision=ms["gap_precision"], gap_f1=ms["gap_f1"], bf1=ms["bf1"],
                   n_pb_sim=ms["n_pb"], comps_sim=ms["comps_pred"], comps_video=ms["comps_ref"],
                   comps20_sim=ms["comps20_pred"], comps20_video=ms["comps20_ref"])
        rec["w48"] = dict(_pick(mw, GAP_KEYS), iou_H=mw["iou"])
        rec["sep_sim"] = part_separation(su, Wv[j], m, fr.big_px)
        rec["sep_w48"] = part_separation(wu, Wv[j], m, fr.big_px)
        rec["lag"] = lag
        if fr.width_mm is not None:
            rec["gap_width_mm_med"] = float(np.median(fr.width_mm))
            rec["gap_sub_cell"] = float((fr.width_mm < cell_mm).mean())
            rec["gap_skel5_px"] = int(fr.width_mm.size)
        frames.append(rec)
        images[i] = dict(img=img[j], W=Wv[j], V=Vv[j], m=m, sim=su, skel=fr.skel & m, gaps=fr.gaps)
    t3 = time.time()
    meta = dict(pour=p, N=N, H=H, cell_mm=cell_mm, px_mm=px_mm, params=P, switches=dict(sw, e0=str(sw["e0"])),
                score_ref=dict(struct=D.score_ref[0], iou=D.score_ref[1]), evaluate=r, objective=CF.objective({p: r}, W_IOU),
                n_video_frames=int(len(tv)), n_observed=len(frames),
                unobserved_frames=sorted(set(range(len(tv))) - set(js)), T_sim=float(t_end[-1]), t_video_last=float(tv[-1]),
                max_abs_dt_sim=float(max(abs(f["t_sim"] - f["t"]) for f in frames)), twin_check=twin_check,
                landing_in_occlusion_by_lag=landing_check(frames, Vv),
                frame_dt_s=float(np.median(np.diff(tv))),
                gap_impl="GapTwin" if use_twin else "topo_metric.interior_brown_H", min_skel=MIN_SKEL,
                timing_s=dict(init=t1 - t0, forward=t2 - t1, metrics=t3 - t2))
    meta["landing_peak"] = landing_peak(meta["landing_in_occlusion_by_lag"], tv)
    log(f"{p}: {len(frames)} observed frames analysed in {t3 - t2:.0f} s ({meta['gap_impl']}, twin check {twin_check})")
    del CF._D[p]
    return frames, meta, images


def _share(fs, grp, what="kept"):
    """Pooled share of the video part pairs of frames fs that the prediction grp keeps apart (or merges)."""
    n = sum(f[grp]["pairs"] for f in fs)
    return sum(f[grp][what] for f in fs) / n if n else float("nan")


def summarise(frames, meta):
    """Per-pour summary (module docstring)."""
    F = frames
    ok = [f for f in F if f["frame_ok"]]
    w = np.array([f["wt"] for f in ok])
    w = w / w.sum() if len(ok) else w
    L40 = F[-N_LAST40:]
    gd = [f for f in F if f["gap_def"]]
    gd40 = [f for f in L40 if f["gap_def"]]
    mean = lambda fs, key: _nanmean([f[key] for f in fs])
    sub = lambda fs, grp, key: _nanmean([f[grp][key] for f in fs if f.get(grp)])
    s = dict(calib=dict(viou=float(np.dot(w, [f["iou"] for f in ok])) if ok else None,
                        struct=float(np.dot(w, [f["struct"] for f in ok])) if ok else None,
                        n_frame_ok=len(ok)),
             calib_check=dict(evaluate_viou=meta["evaluate"]["viou"], evaluate_struct=meta["evaluate"]["struct"]),
             objective=meta["objective"], n_observed=len(F), n_gap_def=len(gd), max_abs_dt_sim=meta["max_abs_dt_sim"],
             landing_in_occlusion_by_lag=meta.get("landing_in_occlusion_by_lag"), landing_peak=meta.get("landing_peak"))
    thr = lambda fs, key="gap_f1": dict({f"{t:.1f}": _nanmean([f["thr"][f"{t:.1f}"][key] for f in fs if f.get("thr")])
                                         for t in THR}, **{"0.5": mean(fs, key)})
    thr_sep = lambda fs: {f"{t:.1f}": _share([dict(s=f["thr"][f"{t:.1f}"]["sep"]) for f in fs if f.get("thr")], "s")
                          for t in THR}
    s["all"] = dict(iou=mean(F, "iou"), struct=mean(F, "struct"), iou_H=mean(F, "iou_H"),
                    sep_kept_sim=_share(F, "sep_sim"), sep_kept_w48=_share(F, "sep_w48"),
                    white_mm2_sim=mean(F, "white_mm2_sim"), white_mm2_video=mean(F, "white_mm2_video"))
    s["gaps"] = dict(sim_f1=mean(gd, "gap_f1"), sim_recall=mean(gd, "gap_recall"),
                     sim_recall_tol1=mean(gd, "gap_recall_tol1"), sim_precision=mean(gd, "gap_precision"),
                     sim_bf1=mean(F, "bf1"), w48_f1=sub(gd, "w48", "gap_f1"), w48_recall=sub(gd, "w48", "gap_recall"),
                     w48_bf1=sub(F, "w48", "bf1"), lag_f1=sub(gd, "lag", "gap_f1"), lag_bf1=sub(F, "lag", "bf1"),
                     n_lag=sum(1 for f in gd if f.get("lag")), sim_f1_thr=thr(gd),
                     sim_has_interior_brown=float(np.mean([f["n_pb_sim"] > 0 for f in gd])) if gd else None)
    wd = [f for f in F if "gap_width_mm_med" in f]
    if wd:
        pix = np.array([f["gap_skel5_px"] for f in wd], float)
        s["gap_width"] = dict(median_mm=float(np.median([f["gap_width_mm_med"] for f in wd])),
                              sub_cell_share=float(np.dot(pix, [f["gap_sub_cell"] for f in wd]) / pix.sum()),
                              cell_mm=meta["cell_mm"], px_mm=meta["px_mm"])
    s["last40"] = dict(n=len(L40), t0=L40[0]["t"], iou=mean(L40, "iou"), struct=mean(L40, "struct"),
                       white_mm2_sim=mean(L40, "white_mm2_sim"), white_mm2_video=mean(L40, "white_mm2_video"),
                       comps20_sim=mean(L40, "comps20_sim"), comps20_video=mean(L40, "comps20_video"),
                       comps20_w48=sub(L40, "w48", "comps20_pred"),
                       comps20_match_sim=float(np.mean([f["comps20_sim"] == f["comps20_video"] for f in L40])),
                       comps20_match_w48=float(np.mean([f["w48"]["comps20_pred"] == f["comps20_video"] for f in L40])),
                       sep_kept_sim=_share(L40, "sep_sim"), sep_kept_w48=_share(L40, "sep_w48"),
                       sep_merged_sim=_share(L40, "sep_sim", "merged"),
                       sep_pairs=int(sum(f["sep_sim"]["pairs"] for f in L40)),
                       n_gap_def=len(gd40), sim_f1=mean(gd40, "gap_f1"), w48_f1=sub(gd40, "w48", "gap_f1"),
                       lag_f1=sub(gd40, "lag", "gap_f1"), sim_f1_thr=thr(gd40),
                       sep_kept_sim_thr=dict(thr_sep(gd40), **{"0.5": _share(gd40, "sep_sim")}))
    R = [f for f in F if f.get("ref_sim")]
    if R:
        s["ref_last"] = {g: {k: _nanmean([f[g][k] for f in R]) for k in REF_KEYS}
                         for g in ("ref_sim", "ref_w48", "ref_video")}
        s["ref_last"]["frames"] = [f["j"] for f in R]
    return s


# ------------------------------------------------------------------------------------------------ sheet
SIM_C, VID_C, W48_C, LAG_C = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
STR_C, IOU_C = "#0b0b0b", "#9b4fb0"
INK, INK2, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"


def _panel_gray(a, show, dim, size):
    g = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    pur = np.array([60, 0, 60], float)
    im[dim] = (0.35 * im[dim] + 0.65 * pur).astype(np.uint8)
    im[~show] = (0, 0, 0)
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_NEAREST)


def _overlay(d, size):
    m, wb, sb = d["m"], (d["W"] > 0.5) & d["m"], (d["sim"] > 0.5) & d["m"]
    ov = np.zeros(m.shape + (3,), np.uint8)
    ov[:] = (30, 30, 30)
    ov[m & ~wb & ~sb] = (40, 70, 110)                    # brown in both
    ov[wb & sb] = (235, 235, 235)
    ov[sb & ~wb] = (60, 60, 230)                         # sim white only: red
    ov[wb & ~sb] = (230, 140, 40)                        # video white only: blue
    ov[d["pb_sim"] & ~TM.near(d["gaps"])] = (200, 60, 200)    # sim interior brown off the video gaps: magenta
    ov = cv2.resize(ov, (size, size), interpolation=cv2.INTER_NEAREST)
    sc = size / m.shape[0]
    for msk, col in ((d["skel"] & ~sb, (0, 200, 0)), (d["skel"] & sb, (0, 230, 255))):   # video gap kept / filled
        for y, x in zip(*np.nonzero(msk)):
            cv2.circle(ov, (int((x + 0.5) * sc), int((y + 0.5) * sc)), 2, col, -1)
    return ov


def _put(im, s, y, scale=0.42, col=(0, 255, 255)):
    cv2.putText(im, s, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, s, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def _series(frames, meta, sel, width):
    """Time-series panel (BGR uint8, width px): one axis per measure, the sheet's frames marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([f["t"] for f in frames])
    g = lambda key, grp=None: np.array([np.nan if (grp and not f.get(grp)) else
                                        (f[grp][key] if grp else f[key]) for f in frames], float)
    gdef = np.array([f["gap_def"] for f in frames])
    ok = np.array([f["frame_ok"] for f in frames])
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(5, 1, figsize=(width / 100, 11.5), dpi=100, sharex=True, facecolor="#fcfcfb")
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.grid(True, color=GRID_C, lw=0.8)
        for f in sel:
            a.axvline(frames[f]["t"], color=INK2, lw=0.8, ls=":")
        for x in t[~ok]:                                 # frames the calibration does not count
            a.axvspan(x - 0.03, x + 0.03, color="#efeeea", lw=0)
    a = ax[0]                                            # own colours: blue / orange mean sim / video below
    a.plot(t, g("struct"), color=STR_C, lw=2, label="structure score (solver grid)")
    a.plot(t, g("iou"), color=IOU_C, lw=2, label="video IoU (solver grid)")
    a.axhline(meta["score_ref"]["struct"], color=STR_C, lw=1, ls="--")
    a.axhline(meta["score_ref"]["iou"], color=IOU_C, lw=1, ls="--")
    a.set_ylim(0, 1)
    a.set_ylabel("score")
    leg = dict(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title_fontsize=10)
    a.legend(title="sim vs video\n(dashed: video vs itself\n0.25 s later)", **leg)
    a = ax[1]
    a.plot(t, g("white_mm2_sim"), color=SIM_C, lw=2, label="sim")
    a.plot(t, g("white_mm2_video"), color=VID_C, lw=2, label="video")
    a.set_ylabel("white area [mm²]")
    a.legend(**leg)
    a = ax[2]
    gf = lambda x: np.where(gdef, x, np.nan)
    band = np.array([[f["thr"][f"{x:.1f}"]["gap_f1"] if f.get("thr") else np.nan for x in THR] + [f["gap_f1"]]
                     for f in frames], float)
    a.fill_between(t, gf(np.nanmin(band, 1)), gf(np.nanmax(band, 1)), color=SIM_C, alpha=0.18, lw=0,
                   label=f"sim, white threshold\n{THR[0]:g}-{THR[-1]:g}")
    a.plot(t, gf(g("gap_f1")), color=SIM_C, lw=2, marker="o", ms=3, label="sim (threshold 0.5)")
    a.plot(t, gf(g("gap_f1", "w48")), color=W48_C, lw=1.5, label=f"video through the\nN={meta['N']} grid (ceiling)")
    a.plot(t, gf(g("gap_f1", "lag")), color=LAG_C, lw=1.2, ls="--", label="video 0.25 s later")
    a.set_ylim(-0.02, 1.02)
    a.set_ylabel("gap F1 vs video")
    a.legend(title=f"frames with >= {MIN_SKEL} px\nof video gap skeleton", **leg)
    a = ax[3]
    a.step(t, g("comps20_video"), where="mid", color=VID_C, lw=2, label="video")
    a.step(t, g("comps20_sim"), where="mid", color=SIM_C, lw=2, label="sim")
    a.step(t, g("comps20_pred", "w48"), where="mid", color=W48_C, lw=1.2, label=f"video through\nthe N={meta['N']} grid")
    rr = [f["ref_video"]["comps20_ref"] for f in frames if f.get("ref_video")]
    if rr:
        a.axhline(rr[-1], color=INK2, lw=1, ls="--", label="finished-pattern\nreference")
    a.set_ylabel("white parts ≥ 20 mm²")
    a.legend(**leg)
    a.yaxis.get_major_locator().set_params(integer=True)
    a = ax[4]
    sh = lambda grp: np.array([f[grp]["kept"] / f[grp]["pairs"] if f[grp]["pairs"] else np.nan for f in frames])
    a.plot(t, sh("sep_sim"), color=SIM_C, lw=2, marker="o", ms=3, label="sim")
    a.plot(t, sh("sep_w48"), color=W48_C, lw=1.5, label=f"video through\nthe N={meta['N']} grid")
    a.set_ylim(-0.02, 1.02)
    a.set_ylabel("video part pairs\nkept apart")
    a.legend(title="frames with >= 2 video parts", **leg)
    a.set_xlabel("video time [s]  (dotted: frames shown above; shaded: not counted by the calibration)")
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    fig.canvas.draw()
    im = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    plt.close(fig)
    return cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)


def sheet(fn, p, frames, meta, images, K=8, size=256):
    """DIR/<p>_sheet.jpg (module docstring)."""
    first = next((i for i, f in enumerate(frames) if f["frame_ok"]), 0)
    sel = sorted(set(np.linspace(first, len(frames) - 1, K).round().astype(int).tolist()))
    H, nN = meta["H"], f"N{meta['N']}"
    sc = size / H
    gfn = GapTwin(H) if meta["gap_impl"] == "GapTwin" else _IB_ORIG     # the routine the metrics used
    cols = []
    for i in sel:
        f, d = frames[i], images[i]
        d["pb_sim"] = gfn(d["sim"], d["m"], H)
        rgb = cv2.resize(np.ascontiguousarray(d["img"][..., ::-1]), (size, size), interpolation=cv2.INTER_AREA)
        if f["land_UV"]:                                 # the sim's landing point (solver frame k)
            u, v = f["land_UV"]
            px = (int(((u + EXT) / (2 * EXT) * (H - 1) + 0.5) * sc), int(((v + EXT) / (2 * EXT) * (H - 1) + 0.5) * sc))
            cv2.drawMarker(rgb, px, (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
        _put(rgb, f"t {f['t']:.2f} s  (frame {f['j']})", 14)
        _put(rgb, f"sim k {f['k']}  t_end {f['t_sim']:.2f} s", 30)
        wv = _panel_gray(d["W"], np.ones_like(d["V"]), ~d["V"], size)
        _put(wv, f"video W  white {f['white_mm2_video']:.0f} mm2", 14)
        _put(wv, f"parts>=20mm2 {f['comps20_video']}  gap skel {f['n_skel']} px", 30)
        disc = TM.grid(H)[0] ** 2 + TM.grid(H)[1] ** 2 <= 1.0
        sv = _panel_gray(d["sim"], disc, disc & ~d["m"], size)
        _put(sv, f"sim c  white {f['white_mm2_sim']:.0f} mm2", 14)
        _put(sv, f"parts>=20mm2 {f['comps20_sim']}  (N={meta['N']} grid)", 30)
        ov = _overlay(d, size)
        _put(ov, f"IoU {f['iou']:.2f}  S {f['struct']:.2f}" + ("" if f["frame_ok"] else "  (not scored)"), 14)
        gl = f"gapF1 {f['gap_f1']:.2f} ({nN} ceil {f['w48']['gap_f1']:.2f})" if f["gap_def"] else "gap terms undefined"
        _put(ov, gl, 30)
        q, qw = f["sep_sim"], f["sep_w48"]
        if q["pairs"]:
            _put(ov, f"video parts {q['parts']}: pairs apart sim {q['kept']}/{q['pairs']} ({nN} {qw['kept']})", 46)
        cols.append(np.vstack([rgb, wv, sv, ov]))
    grid = np.hstack(cols)
    leg = np.full((30, grid.shape[1], 3), 252, np.uint8)
    items = [((235, 235, 235), "white both"), ((60, 60, 230), "sim white only"), ((230, 140, 40), "video white only"),
             ((40, 70, 110), "brown both"), ((0, 200, 0), "video gap kept brown by sim"),
             ((0, 230, 255), "video gap filled by sim"), ((200, 60, 200), "sim gap off video gaps"),
             ((255, 255, 0), "+ sim landing point"), ((60, 0, 60), "not observed (dimmed)")]
    x = 8
    for col, s in items:
        cv2.rectangle(leg, (x, 9), (x + 14, 23), col, -1)
        cv2.rectangle(leg, (x, 9), (x + 14, 23), (80, 80, 80), 1)
        cv2.putText(leg, s, (x + 19, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        x += 26 + int(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0])
    head = np.full((34, grid.shape[1], 3), 252, np.uint8)
    cv2.putText(head, f"{p}: closure {meta.get('closure', '')}  N={meta['N']} (cell {meta['cell_mm']:.2f} mm), video "
                f"{meta['H']} px ({meta['px_mm']:.2f} mm)   rows: video frame / video W / sim c / overlay",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    out = np.vstack([head, grid, leg, _series(frames, meta, sel, grid.shape[1])])
    cv2.imwrite(fn, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return fn


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("closure", help="closure json (cmafit best_closure.json or latte_imex/data/closures/*.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pours", default=",".join(PATTERNS))
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--frames", type=int, default=8, help="time points on the sheet")
    ap.add_argument("--e0", default=None, choices=("jet", "world", "none"), help="override the closure's e0 switch")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    log = lambda *x: print(*x, flush=True)
    num, sw = load_closure(a.closure, None if a.e0 is None else (0 if a.e0 == "none" else a.e0))
    sfn = os.path.join(a.out, "summary.json")
    M = dict(closure=os.path.abspath(a.closure), N=a.N, e0=str(sw["e0"]), min_skel=MIN_SKEL, n_last40=N_LAST40,
             thr=list(THR), note="iou / struct on the solver grid as the calibration; gap terms at the video grid against "
                                 "the video frame (w48 = the calibration target through the N grid: resolution ceiling; "
                                 "lag = the video 0.25 s later); ref_last = last frames against the finished-pattern "
                                 "reference")
    S = json.load(open(sfn)) if os.path.exists(sfn) else {}
    if any(S.get("_meta", {}).get(k) != M[k] for k in ("closure", "N", "e0")):
        if S:
            log(f"[validate] {sfn} holds another closure / N / e0 ({S.get('_meta', {}).get('closure')}): started afresh")
        S = {}
    S["_meta"] = M
    tag = os.path.join(os.path.basename(os.path.dirname(M["closure"])), os.path.basename(a.closure))
    for p in a.pours.split(","):
        frames, meta, images = validate_pour(p, num, sw, N=a.N, device=a.device, log=log)
        meta["closure"] = tag
        json.dump(_clean(dict(meta=meta, frames=frames)), open(os.path.join(a.out, f"{p}_frames.json"), "w"), indent=1)
        S[p] = _clean(dict(summarise(frames, meta), timing_s=meta["timing_s"], twin_check=meta["twin_check"],
                           gap_impl=meta["gap_impl"]))
        json.dump(S, open(sfn, "w"), indent=1)
        fn = sheet(os.path.join(a.out, f"{p}_sheet.jpg"), p, frames, meta, images, K=a.frames)
        s = S[p]
        ft, lp = s["gaps"]["sim_f1_thr"], s["landing_peak"] or {}
        q = lambda x, f=".3f": "-" if x is None else format(x, f)      # None: nan in the json (no such frames)
        log(f"{p}: IoU {q(s['calib']['viou'])} struct {q(s['calib']['struct'])} (evaluate "
            f"{q(s['calib_check']['evaluate_viou'])} / {q(s['calib_check']['evaluate_struct'])}); gap F1 sim "
            f"{q(s['gaps']['sim_f1'])} (threshold " + " / ".join(f"{k} {q(ft[k])}" for k in sorted(ft)) + ") | N-grid "
            f"ceiling {q(s['gaps']['w48_f1'])} | lag {q(s['gaps']['lag_f1'])} over {s['n_gap_def']} frames; last 40 "
            f"parts>=20mm2 sim {q(s['last40']['comps20_sim'], '.2f')} video {q(s['last40']['comps20_video'], '.2f')}, "
            f"video part pairs kept apart sim {q(s['last40']['sep_kept_sim'], '.2f')} N-grid "
            f"{q(s['last40']['sep_kept_w48'], '.2f')}; landing point in the occlusion peaks at lag "
            f"{lp.get('lag_frames')} frames ({q(lp.get('lag_s'), '+.2f')} s, {q(lp.get('share'), '.2f')} vs "
            f"{q(lp.get('share_0'), '.2f')} at 0) -> {fn}")


if __name__ == "__main__":
    main()
