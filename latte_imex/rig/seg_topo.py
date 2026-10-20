"""Gap-preserving video whiteness W (the supervision target of video_obs.py), leave-one-pour-out.  V2_SURFACE_MODEL.md 10.5.

    W = carve( veto( temporal( classifier ) ) )

Why.  seg_sup's W filled the thin brown gaps between layers (tulip teardrop / heart / wings merged; 14 % of the tulip
gap centre-line brown): its labels are Otsu masks that fill gaps (gap recall 0.42-0.57 on tulip / leaf / swan) and its
temporal stage blurs the evidence by 1 px.  A brown line 1-2 mm wide under the video's edge blur (sigma ~1 mm) never
falls to the crema level, so no global threshold keeps it; a half-max test against the local white level does.

1. classifier  seg_sup exactly (frame_feats, CFG: features, label construction, weights, camera-gain replay,
               HistGradientBoosting) and seg_gmm for its crema-belief feature, both retrained leave-one-pour-out on the
               video_obs frames (fuse_occ.py); labels = video_obs.hand_mask (the Otsu target masks of the other four pours,
               align-corners area resampling); seg_gmm geometry from the same four masks (sorted order).  The logits are
               rounded to float16 (what the cache stores, so a cached and a fresh run give the same W).
2. temporal    seg_sup.temporal exactly: evidence clip(0.35 (logit - training prior), +-6), blurred 1 px, per-pixel
               two-state HMM with 1 px belief diffusion -> W2.
3. veto        (GATES, 'z2') foam over crema can only ADD reflectance, so white is vetoed where the pixel is darker than
               its local crema by more than the noise: x = linear colour / local crema (normalised convolution over
               SIG_END = 4 mm of the frame's crema seeds = stage-2 brown farther than SEED_D from white; divides out cast
               shadows and the 3x brightness drift over a pour), F' = the frame's median x over its foam seeds (>= N_FOAM
               px, else the nearest such frame's), f = (x - 1).(F' - 1) / |F' - 1|^2 (0 = crema, 1 = foam), median over
               5 frames ('z2'; 'z1' per frame), cap = exp(-min(f, 0)^2 / (2 sigma_k^2)), W2 = min(W2, cap); sigma_k =
               1.4826 MAD of f over frame k's crema seeds.  cap = 1 for every f >= 0 (semi-transparent foam, shaded
               foam), 0.5 at -1.18 sigma: no tuned constant.  Removes the leaf shadow blob next to the pitcher
               (0.81-1.25 s, 161 -> 27 px).  'none' = no veto (ablation).
4. carve       (STAGE3, 'run') half-max valley test on the linear-light foam fraction f against the local crema / foam
               colours (normalised convolution over SIG_END of stage-2 brown / white seeds farther than SEED_D from the
               edge): flank level P = grey closing of f over R_CL = 4 mm (gap half-width ~2 mm + 2 blur sigma), a pixel
               with P >= P_MIN is a valley when f < HALF P (the 50 % point between the local white level and crema is
               where a blurred edge is; a 1-2 mm gap keeps f well below it).  Evidence GAMMA_V (HALF P - f) / S_E,
               clipped +-LLR_V, through a per-pixel two-state valley HMM (switching A_SW per frame); junction bridges =
               Hessian centre-line pixels with f / P < HYST_LOW within HYST_D of a valley, 8-connected to it; then a
               minimum run length MINRUN = 5 frames (~0.3 s) per pixel in time (shorter carved runs dropped, shorter
               uncarved gaps inside a run filled; ends edge-padded).  Carved: W = min(W2, 1 - P(valley)), bridges and
               filled runs min(W2, BRIDGE_W).  The carve only lowers W.  'r1' = no persistence (MINRUN 1), 'off' = none.
Selection (doc 10.5): stage 2 / carve chosen by a rule fixed in advance on tulip + leaf (J2 = mean of iou, bf1, gap F1,
area-sampled structure score, part-count and thickness terms; admissibility bounds on early white, mid-pour false
white, flips, blinks); the rule's tie-break picked gate 'none', 'z2' was adopted after the round's results for the
physics above (cal J2 identical, test J2 0.835 vs 0.833).
Cache: env SEG_TOPO_CACHE (default <tmp>/seg_topo_cache): classifier logits per fold, keyed by a content hash of the
frames and the target masks.
    python3 -m latte_imex.rig.seg_topo <out_dir> [--pours tulip,...] [--gate z2] [--carve run] [--viddir D]
        -> <out_dir>/W_<p>.npz: W float16 (n, 128, 128) in [0, 1] (0 outside valid), W2 (stage 2 after the veto), carved
    video_obs.white(p) stores segment_lopo(p) as the target."""
import argparse
import hashlib
import json
import os
import tempfile
import time
import warnings

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import seg_gmm, seg_sup, video_obs

VID = os.environ.get("SEG_TOPO_VIDDIR", video_obs.VID)       # frames: <VID>/<p>.npz (video_obs layout)
CACHE = os.environ.get("SEG_TOPO_CACHE", os.path.join(tempfile.gettempdir(), "seg_topo_cache"))
POURS = video_obs.POURS
EXT, L_M_MM = video_obs.EXT, 42.0                            # [mm] cup radius unit (disc.json L_m 41.5-42.3 mm)
CFG = dict(seg_sup.CFG)                                      # stages 1-2: seg_sup unchanged
CARVE = dict(SIG_END=4.0, SEED_D=1.5, R_CL=4.0, HALF=0.5, P_MIN=0.5, S_E=0.05, GAMMA_V=0.5, LLR_V=4.0, P0_V=0.01,
             HYST_LOW=0.75, HYST_SIG=1.0, HYST_D=2.0, BRIDGE_W=0.45)
# SIG_END [mm] end-member smoothing; SEED_D [mm] seed distance from the stage-2 edge; R_CL [mm] flank closing radius;
# HALF half-max; P_MIN dimmest plateau treated as foam; S_E evidence scale (crema-seed noise of f ~0.05); GAMMA_V, LLR_V
# evidence tempering / clip; P0_V initial valley belief; HYST_LOW, HYST_SIG [mm], HYST_D [mm] bridges; BRIDGE_W W cap
VETO = dict(SEED_D=CARVE["SEED_D"], SIG_END=CARVE["SIG_END"], N_FOAM=100)   # N_FOAM: foam seeds (~50 mm^2) for F'
GATES = {"none": None, "z1": dict(VETO, TMED=1), "z2": dict(VETO, TMED=5)}
STAGE3 = {"off": None, "r1": dict(A_SW=0.1, MINRUN=1), "run": dict(A_SW=0.1, MINRUN=5)}
GATE, CARVE_PRESET = "z2", "run"                             # the adopted target


# ------------------------------------------------------------------------------------------------ helpers
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lo(q):
    return np.log(q / (1 - q))


def px_mm(H):
    return 2 * EXT * L_M_MM / (H - 1)


def video(p, vid=None):
    Z = np.load(os.path.join(vid or VID, f"{p}.npz"))
    return dict(t=Z["t"], img=Z["img"], valid=Z["valid"].astype(bool))


_FID = {}


def frames_id(vid=None):
    """Content hash of the frames (t, img, valid of every pour) and the target masks: the cache key (the whiteness stored
    in the same files does not enter)."""
    vid = os.path.realpath(vid or VID)
    stamp = tuple((os.path.getsize(f"{vid}/{p}.npz"), os.path.getmtime(f"{vid}/{p}.npz")) for p in POURS)
    if _FID.get(vid, (None,))[0] != stamp:
        h = hashlib.sha1()
        for p in POURS:
            Z = np.load(f"{vid}/{p}.npz")
            T = np.load(f"{video_obs.REPO}/raw_data/rig/targets/{p}.npz")
            for a in (Z["t"], Z["img"], Z["valid"], T["white"], T["valid"]):
                h.update(np.ascontiguousarray(a).tobytes())
        _FID[vid] = (stamp, h.hexdigest()[:10])
    return _FID[vid][1]


# ------------------------------------------------------------------------------------------------ 1. classifier
def classifier_logits(p, vid=None, log=print):
    """Per-frame classifier log-odds of pour p (float16-rounded) and the training prior log-odds, from seg_sup trained on
    the other four pours (and a seg_gmm geometry from their masks).  Cached."""
    vid = vid or VID
    others = tuple(q for q in POURS if q != p)
    key = hashlib.sha1(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()},
                                  sort_keys=True).encode()).hexdigest()[:8]
    fn = f"{CACHE}/logit_{frames_id(vid)}_{key}_{p}.npz"
    if os.path.exists(fn):
        D = np.load(fn)
        return D["L"].astype(np.float32), float(D["prior"])
    masks = {q: video_obs.hand_mask(q) for q in POURS}
    old = seg_sup.VID, seg_gmm.VID
    seg_sup.VID = seg_gmm.VID = vid                     # both read the frames (seg_gmm: the training pours' last frame)
    t0 = time.time()
    try:
        gmm = {}
        for q in others + (p,):
            seg_gmm._GEOM.clear()
            gmm[q] = seg_gmm.segment(q, video(q, vid), {r: masks[r] for r in sorted(others)}).astype(np.float32)
        seg_gmm._GEOM.clear()
        t1 = time.time()
        clf, prior = seg_sup.train(others, gmm, masks, CFG)
        t2 = time.time()
        L = seg_sup.predict_logit(clf, p, gmm[p], CFG).astype(np.float16)
    finally:
        seg_sup.VID, seg_gmm.VID = old
        seg_gmm._GEOM.clear()
    log(f"{p}: seg_gmm {t1 - t0:.0f} s, classifier trained on {', '.join(others)} {t2 - t1:.0f} s, "
        f"predicted {time.time() - t2:.0f} s")
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(fn, L=L, prior=float(prior), train=np.array(others), frames=os.path.realpath(vid))
    return L.astype(np.float32), float(prior)


# ------------------------------------------------------------------------------------------------ colour
def srgb_lin(img):
    x = img.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _disk(rf):
    r = int(np.floor(rf))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ((x * x + y * y) <= rf * rf + 1e-9).astype(np.uint8)


def _normconv(x, m, sig, prior, k=1e-3):
    g = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), sig)
    wm = g(m.astype(np.float32))
    return np.dstack([(g(x[..., c] * m) + k * prior[c]) / (wm + k) for c in range(x.shape[2])])


# ------------------------------------------------------------------------------------------------ 3. darkness veto
def darkness_veto(img, V, W2, g=GATES[GATE]):
    """cap (n, H, H) = exp(-min(f, 0)^2 / (2 sigma_k^2)) on the illumination-normalised foam fraction f (module docstring,
    stage 3) and f; cap = 1 where f is undefined (no crema seeds, < 50 valid px)."""
    n, H = W2.shape[0], W2.shape[-1]
    pm = px_mm(H)
    fg = np.full(W2.shape, np.nan, np.float32)
    CS = np.zeros(W2.shape, bool)                       # crema seeds per frame (noise scale)
    Fx = np.full((n, 3), np.nan, np.float32)
    X = {}
    for k in range(n):
        lin = srgb_lin(img[k])
        wb = (W2[k] > 0.5) & V[k]
        dW = ndi.distance_transform_edt(~wb) * pm
        dB = ndi.distance_transform_edt(wb) * pm
        fs = wb & (dB > g["SEED_D"])
        cs = V[k] & ~wb & (dW > g["SEED_D"])
        if V[k].sum() < 50 or cs.sum() < 20:
            continue
        CS[k] = cs
        C = _normconv(lin, cs, g["SIG_END"] / pm, np.median(lin[cs], 0))
        X[k] = lin / np.maximum(C, 1e-4)
        if fs.sum() >= g["N_FOAM"]:
            Fx[k] = np.median(X[k][fs], 0)
    okF = np.flatnonzero(np.isfinite(Fx[:, 0]))
    if len(okF) == 0:
        return np.ones(W2.shape, np.float32), fg
    for k, x in X.items():
        Fk = Fx[k] if np.isfinite(Fx[k, 0]) else Fx[okF[np.argmin(np.abs(okF - k))]]
        d = (Fk - 1.0)[None, None, :]
        f = ((x - 1.0) * d).sum(2) / max(float((d * d).sum()), 1e-6)
        fg[k] = np.where(V[k], f, np.nan)
    X = None
    fs_ = fg
    if g["TMED"] > 1:                                 # temporal median of f over TMED frames (valid frames only)
        h = g["TMED"] // 2
        fs_ = np.full_like(fg, np.nan)
        for k in range(n):
            if np.isfinite(fg[k]).any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)          # all-nan columns outside valid
                    fs_[k] = np.where(np.isfinite(fg[k]), np.nanmedian(fg[max(0, k - h):k + h + 1], 0), np.nan)
    ok = np.isfinite(fs_)
    sig = np.full(n, np.nan)
    for k in range(n):
        a = fs_[k][CS[k] & ok[k]]
        if a.size >= 50:
            sig[k] = 1.4826 * np.median(np.abs(a - np.median(a)))
    okS = np.flatnonzero(np.isfinite(sig) & (sig > 0))
    if len(okS) == 0:
        return np.ones(W2.shape, np.float32), fs_
    sig = np.array([sig[k] if k in set(okS) else sig[okS[np.argmin(np.abs(okS - k))]] for k in range(n)])
    neg = np.minimum(np.nan_to_num(fs_, nan=0.0), 0.0)
    cap = np.where(ok, np.exp(-neg ** 2 / (2.0 * sig[:, None, None] ** 2)), 1.0).astype(np.float32)
    return cap, fs_


# ------------------------------------------------------------------------------------------------ 4. carve
def foam_fraction(img, v, wb, pm, c=CARVE):
    """linear-light foam fraction f = (x - C).(F - C) / |F - C|^2 against the local crema / foam colours (normalised
    convolution over SIG_END; seeds = stage-2 brown / white farther than SEED_D from its edge); outside valid: the
    nearest valid value."""
    lin = srgb_lin(img)
    dW = ndi.distance_transform_edt(~wb) * pm
    dB = ndi.distance_transform_edt(wb | ~v) * pm
    cs, fs = v & ~wb & (dW > c["SEED_D"]), wb & (dB > c["SEED_D"])
    if cs.sum() < 20:
        cs = v & ~wb
    if fs.sum() < 20:
        if wb.sum() < 5:
            return None
        fs = wb
    s = c["SIG_END"] / pm
    C = _normconv(lin, cs, s, np.median(lin[cs], 0))
    Fo = _normconv(lin, fs, s, np.median(lin[fs], 0))
    d = Fo - C
    f = ((lin - C) * d).sum(2) / np.maximum((d * d).sum(2), 1e-6)
    idx = ndi.distance_transform_edt(~v, return_distances=False, return_indices=True)
    return f[tuple(idx)].astype(np.float32)


def valley_evidence(img, v, w, c=CARVE):
    """(f, P, e) of one frame: e = HALF P - f where P >= P_MIN and valid, -inf elsewhere."""
    pm = px_mm(img.shape[0])
    f = foam_fraction(img, v, (w > 0.5) & v, pm, c)
    if f is None:
        return None
    k = _disk(c["R_CL"] / pm)
    P = cv2.erode(cv2.dilate(f, k), k)
    elig = (P >= c["P_MIN"]) & v
    return f, P, np.where(elig, c["HALF"] * P - f, -np.inf).astype(np.float32)


def carve_evidence(img, V, W2, c=CARVE):
    """per frame (f, P, e) of the valley test (needs only the stage-2 W for the colour seeds)."""
    F, P = np.zeros_like(W2), np.zeros_like(W2)
    E = np.full_like(W2, -np.inf)
    for k in range(len(W2)):
        if V[k].sum() < 50:
            continue
        r = valley_evidence(img[k], V[k], W2[k], c)
        if r is not None:
            F[k], P[k], E[k] = r
    return F, P, E


def hmm_valley(E, V, a_sw, c=CARVE):
    """per-pixel two-state (valley / not) HMM, forward-backward; evidence clip(GAMMA_V e / S_E, +-LLR_V), 0 where
    ineligible or invalid; symmetric switching a_sw per frame -> P(valley) (defined at every frame)."""
    L_ = np.where(np.isfinite(E), np.clip(c["GAMMA_V"] * E / c["S_E"], -c["LLR_V"], c["LLR_V"]), 0.0)
    L_ = np.where(V, L_, 0.0).astype(np.float32)
    F, B = np.zeros_like(L_), np.zeros_like(L_)
    lam = np.full(E.shape[1:], _lo(c["P0_V"]), np.float32)
    for k in range(len(E)):
        if k > 0:
            q = _sig(lam)
            lam = _lo(np.clip(q * (1 - a_sw) + (1 - q) * a_sw, 1e-6, 1 - 1e-6))
        lam = lam + L_[k]
        F[k] = lam
    b = np.zeros(E.shape[1:], np.float32)
    for k in range(len(E) - 2, -1, -1):
        q = _sig(b + L_[k + 1])
        b = np.log(((1 - a_sw) * q + a_sw * (1 - q)) / (a_sw * q + (1 - a_sw) * (1 - q)))
        B[k] = b
    return _sig(F + B).astype(np.float32)


def centre_lines(f, pm, sig_mm):
    """ridge lines of the valleys of f (Hessian of f blurred by sig_mm: positive major curvature, minimum across it)."""
    g = cv2.GaussianBlur(f.astype(np.float32), (0, 0), sig_mm / pm)
    gyy = np.gradient(np.gradient(g, axis=0), axis=0)
    gxx = np.gradient(np.gradient(g, axis=1), axis=1)
    gxy = np.gradient(np.gradient(g, axis=0), axis=1)
    tr, det = gxx + gyy, gxx * gyy - gxy * gxy
    l1 = 0.5 * tr + np.sqrt(np.maximum(0.25 * tr * tr - det, 0))
    vx, vy = gxy, l1 - gxx
    alt = np.hypot(vx, vy) < 1e-9
    vx = np.where(alt, l1 - gyy, vx); vy = np.where(alt, gxy, vy)
    nn = np.maximum(np.hypot(vx, vy), 1e-9)
    vx, vy = (vx / nn).astype(np.float32), (vy / nn).astype(np.float32)
    yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]].astype(np.float32)
    a = cv2.remap(g, xx + vx, yy + vy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    b = cv2.remap(g, xx - vx, yy - vy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (l1 > 0) & (g <= a) & (g <= b)


def bridges(strong, f, P, elig, pm, c=CARVE):
    """junction bridges: centre-line pixels with f / P < HYST_LOW within HYST_D of a strong valley, 8-connected to it."""
    weak = centre_lines(f, pm, c["HYST_SIG"]) & elig & (f / np.maximum(P, 1e-3) < c["HYST_LOW"]) & ~strong
    weak &= ndi.distance_transform_edt(~strong) * pm <= c["HYST_D"]
    lab, n = ndi.label(weak | strong, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.zeros_like(strong)
    has = np.zeros(n + 1, bool)
    has[np.unique(lab[strong])] = True
    has[0] = False
    return weak & has[lab]


def min_run(C, n):
    """per pixel along time: drop True runs shorter than n, then fill False gaps shorter than n (edge-padded)."""
    if n <= 1:
        return C
    pad = np.pad(C, ((n, n), (0, 0), (0, 0)), mode="edge")
    st = np.ones((n, 1, 1), bool)
    pad = ndi.binary_opening(pad, structure=st)
    pad = ndi.binary_closing(pad, structure=st)
    return pad[n:-n]


def carve(img, V, W2, s3, c=CARVE):
    """stage 4 with the persistence preset s3 (STAGE3) -> (W, carved mask)."""
    pm = px_mm(W2.shape[-1])
    F, P, E = carve_evidence(img, V, W2, c)
    post = hmm_valley(E, V, s3["A_SW"], c)
    strong = post > 0.5
    br = np.zeros_like(strong)
    for k in range(len(W2)):
        if np.isfinite(E[k]).any():
            br[k] = bridges(strong[k] & V[k], F[k], P[k], np.isfinite(E[k]), pm, c)
    C = min_run(strong | br, s3["MINRUN"])
    C &= V
    W = np.where(C & strong, np.minimum(W2, 1.0 - post), W2)
    W = np.where(C & ~strong, np.minimum(W, c["BRIDGE_W"]), W)
    return W.astype(np.float32), C


# ------------------------------------------------------------------------------------------------ driver
def segment_lopo(p, gate=GATE, carve_preset=CARVE_PRESET, vid=None, log=print, return_parts=False):
    """W (n, 128, 128) float32 in [0, 1] of pour p (0 outside valid): classifier trained on the other four pours,
    seg_sup.temporal, the darkness veto GATES[gate], the carve STAGE3[carve_preset]."""
    Z = video(p, vid)
    V = Z["valid"]
    L, prior = classifier_logits(p, vid, log)
    W2 = np.where(V, seg_sup.temporal(L, prior, V, CFG), 0.0).astype(np.float32)
    if GATES[gate] is not None:
        W2 = np.minimum(W2, darkness_veto(Z["img"], V, W2, GATES[gate])[0]).astype(np.float32)
    s3 = STAGE3[carve_preset]
    W, C = (W2, np.zeros_like(V)) if s3 is None else carve(Z["img"], V, W2, s3)
    W = np.where(V, np.clip(W, 0, 1), 0.0).astype(np.float32)
    return (W, dict(W2=W2, carved=C, logit=L, prior=prior)) if return_parts else W


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--pours", default=",".join(POURS))
    ap.add_argument("--gate", default=GATE, choices=list(GATES))
    ap.add_argument("--carve", default=CARVE_PRESET, choices=list(STAGE3))
    ap.add_argument("--viddir", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in a.pours.split(","):
        t0 = time.time()
        W, I = segment_lopo(p, a.gate, a.carve, a.viddir, return_parts=True)
        cfg = dict(gate=a.gate, carve=a.carve, frames=os.path.realpath(a.viddir or VID), GATE=GATES[a.gate],
                   S3=STAGE3[a.carve], CARVE=CARVE)
        np.savez_compressed(f"{a.out}/W_{p}.npz", W=W.astype(np.float16), W2=I["W2"].astype(np.float16),
                            carved=I["carved"], cfg=json.dumps(cfg))
        print(f"{p}: W {W.shape} -> {a.out}/W_{p}.npz ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
