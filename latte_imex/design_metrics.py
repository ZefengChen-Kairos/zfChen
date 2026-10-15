"""Objective pieces of the GPU action design (gpu_design), each taken from the CPU code that defined it:

  shape vs the V0.5 reference   iou04 / iou05 / white / r_hp / hp_contrast of v05_allphys/allphys.metrics (review
                                verify_compare.hp_stats: 1.2 mm Gaussian high-pass, correlation and contrast inside the
                                1 mm-eroded white region of the reference), compared against the V0.5 reference field
                                (the V0.5 controls' own run; 128^2: v05audit cf/baseline, 256^2: runs/v05_256), and the
                                screening score of allphys.rank (iou04 + r_hp, higher = better): L_shape = 2 - iou04 - r_hp.
  decisive structures           review ref2/structm.py (per-pattern measures: heart notch, push seams, layered bands /
                                curls, tulip axis bands, leaf veins, swan neck / head), reported, not in the loss.
  S1 inlet terms                v05_pourable.finalize's exact J on the delivered inlet, per flowing stroke n:
                                jc = dep_n / W (logit chi_n - logit chi_t,n)^2  (chi clipped to [0.02, 0.98]),
                                js = dep_n / W (0.4 ln s_n / ln 1.25)^2,
                                momentum m = ln(kappa_t |u_h|_n / (0.7 FS_uh_n)): load-bearing strokes max(m, 0)^2 / 6,
                                others LAM_OTH F_int,n / F_oth m^2;  W, F_oth over all six patterns (as in finalize).
                                chi_n, |u_h|_n = S-weighted over the solver frames of stroke n (as ana128.chi_real /
                                inlet_strokes); s_n = received volume / (S1 design volume / S1 design s) = the flow scale
                                relative to the V0.5 controls' stroke volume (the S1 design's s).
  robustness indicator          autodl_v05.compare_c (IoU of c > 0.5 and white-area ratio against the nominal run).

Reference data (exported once from the review scratch by `gpu_design export`, small): data/gpu_design/
  v05_ref_<pattern>_N128.npz, v05_ref_<pattern>_N256.npz   c (float32), mask
  s1_refs.json                                             v05_pourable design refs.json (chi_t, dep, F_int, FS_uh_ms)
and v05_pourable/data/design_S1_final.json (per-stroke S1 design: s, vol_mL; closure kappa_t).
"""
from __future__ import annotations
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "gpu_design")
S1_FINAL = os.path.join(HERE, "v05_pourable", "data", "design_S1_final.json")
LB_STROKES = (("heart", 1), ("heart", 2), ("tulip", 6), ("leaf", 4), ("swan", 8), ("swan", 9))   # v05_pourable
LAM_OTH = 0.25
CHI_CLIP = (0.02, 0.98)
D_MM = 80.0


# ----------------------------------------------------------------------------- reference fields
_REF = {}


def load_ref(pattern, N, data=DATA):
    """(c, mask) of the V0.5 reference at N: the stored 128^2 / 256^2 field, otherwise the 256^2 field resampled
    bilinearly to the cell centres of the N grid (mask = solver.Grid(N).mask)."""
    key = (pattern, N, data)
    if key in _REF:
        return _REF[key]
    fn = os.path.join(data, f"v05_ref_{pattern}_N{N}.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        out = (z["c"].astype(float), z["mask"].astype(bool))
    else:
        from scipy.ndimage import map_coordinates
        from .solver import Grid
        z = np.load(os.path.join(data, f"v05_ref_{pattern}_N256.npz"))
        c0 = np.nan_to_num(z["c"].astype(float))
        mask = Grid(N, 0.49).mask
        x = (np.arange(N) + 0.5) / N * 256 - 0.5
        J, I = np.meshgrid(x, x, indexing="ij")
        c = map_coordinates(c0, [J, I], order=1, mode="nearest")
        out = (np.where(mask, c, np.nan), mask)
    _REF[key] = out
    return out


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def hp_stats(c, mask, cr, mr, N):
    """verify_compare.hp_stats: (contrast ratio, correlation) of the 1.2 mm high-pass inside the reference white."""
    from scipy import ndimage
    sig = 1.2 / D_MM * N

    def hp(x, m):
        xx = np.where(m, np.nan_to_num(x), 0.0)
        w = ndimage.gaussian_filter(m.astype(float), sig)
        s = ndimage.gaussian_filter(xx, sig) / np.maximum(w, 1e-9)
        return xx - s
    region = ndimage.binary_erosion((np.nan_to_num(cr) > 0.4) & mr, iterations=max(1, int(round(1.0 / D_MM * N)))) & mask
    a, b = hp(c, mask)[region], hp(cr, mr)[region]
    if a.size < 10 or a.std() == 0 or b.std() == 0:
        return None, None
    return float(a.std() / max(b.std(), 1e-12)), float(np.corrcoef(a, b)[0, 1])


def shape_vs_ref(c, pattern, N, struct=False):
    """Shape metrics of c (N x N, NaN or anything outside the cup) against the V0.5 reference; L_shape = 2 - iou04 -
    r_hp (allphys.rank's score; r_hp None -> 0)."""
    cr, mr = load_ref(pattern, N)
    mask = mr
    c = np.nan_to_num(c)
    crn = np.nan_to_num(cr)
    hc, rh = hp_stats(c, mask, crn, mr, N)
    out = dict(iou04=iou((c > 0.4) & mask, (crn > 0.4) & mr), iou05=iou((c > 0.5) & mask, (crn > 0.5) & mr),
               white=float(((c > 0.4) & mask).sum() / mask.sum()), white_ref=float(((crn > 0.4) & mr).sum() / mr.sum()),
               r_hp=rh, hp_contrast=hc)
    out["L_shape"] = 2.0 - out["iou04"] - (rh if rh is not None else 0.0)
    if struct:
        try:
            out["struct"] = STRUCT[pattern](np.where(mask, c, np.nan))
            out["struct_ref"] = STRUCT[pattern](cr)
        except Exception as e:                     # a structure measure that finds nothing (e.g. no white) is reported
            out["struct"] = dict(error=f"{type(e).__name__}: {e}")
    return out


def compare_nominal(c, c0, mask):
    """autodl_v05.compare_c (IoU of c > 0.5 with the nominal run, white-area ratio c > 0.4, mean |dc|)."""
    from .autodl_v05 import compare_c
    return compare_c(np.nan_to_num(c), np.nan_to_num(c0), mask)


# ----------------------------------------------------------------------------- S1 inlet terms
_S1 = {}


def s1_tables(data=DATA):
    """refs (per pattern, per stroke number string), the S1 design (per pattern: {n: dict(s, vol_mL)}), the closure's
    kappa_t, W and F_oth (sums over all six patterns' flowing strokes, as v05_pourable.finalize)."""
    if "t" in _S1:
        return _S1["t"]
    with open(os.path.join(data, "s1_refs.json")) as f:
        refs = json.load(f)
    with open(S1_FINAL) as f:
        fin = json.load(f)
    des = {p: {int(r["n"]): dict(s=float(r["s"]), vol_mL=float(r["vol_mL"]), cls=r["cls"], chi_t=float(r["chi_t"]))
               for r in v["strokes"]} for p, v in fin["patterns"].items()}
    W = sum(refs[p][str(n)]["dep"] for p in des for n in des[p])
    Fo = sum(refs[p][str(n)]["F_int"] for p in des for n in des[p] if (p, n) not in LB_STROKES)
    _S1["t"] = (refs, des, float(fin["closure"]["kappa_t"]), W, Fo)
    return _S1["t"]


def _logit(x):
    x = min(max(float(x), CHI_CLIP[0]), CHI_CLIP[1])
    return math.log(x / (1 - x))


def s1_terms(pattern, strokes, kappa_t=None):
    """S1 design J of one pattern from the realised per-stroke numbers strokes {n: dict(chi, uh_ms, vol_mL)} (missing
    strokes: chi 0, |u_h| 0, volume 0).  Returns dict(J, parts, rows)."""
    refs, des, kt0, W, Fo = s1_tables()
    kt = kt0 if kappa_t is None else kappa_t
    parts = dict(chi=0.0, s=0.0, uh=0.0, oth=0.0)
    rows = {}
    for n, d in sorted(des[pattern].items()):
        rf = refs[pattern][str(n)]
        r = strokes.get(n, dict(chi=0.0, uh_ms=0.0, vol_mL=0.0))
        e = _logit(r["chi"]) - _logit(rf["chi_t"])
        v0 = d["vol_mL"] / max(d["s"], 1e-9)                   # the V0.5 controls' stroke volume (s = 1)
        s = max(r["vol_mL"], 1e-3) / max(v0, 1e-9)
        js = rf["dep"] / W * (0.4 * math.log(s) / math.log(1.25)) ** 2
        m = math.log(max(kt * r["uh_ms"], 1e-9) / (0.7 * rf["FS_uh_ms"]))
        lb = (pattern, n) in LB_STROKES
        ju = max(m, 0.0) ** 2 / len(LB_STROKES) if lb else 0.0
        jo = 0.0 if lb else LAM_OTH * rf["F_int"] / Fo * m * m
        jc = rf["dep"] / W * e * e
        parts["chi"] += jc; parts["s"] += js; parts["uh"] += ju; parts["oth"] += jo
        rows[n] = dict(chi=r["chi"], chi_t=rf["chi_t"], e=e, s=s, vol_mL=r["vol_mL"], vol_S1design_mL=d["vol_mL"],
                       uh_ms=r["uh_ms"], mom_ratio=kt * r["uh_ms"] / (0.7 * rf["FS_uh_ms"]), J=jc + js + ju + jo)
    return dict(J=sum(parts.values()), parts=parts, rows=rows)


# ----------------------------------------------------------------------------- decisive structures (review ref2/structm.py)
def _samp(c, xs, ys):
    from scipy.ndimage import map_coordinates
    h = 1.0 / c.shape[0]
    f = np.nan_to_num(c, nan=0.0)
    j = (np.asarray(ys) + 0.5) / h - 0.5
    i = (np.asarray(xs) + 0.5) / h - 0.5
    return map_coordinates(f, [j, i], order=1, mode="nearest")


def _bands(prof, prom=0.05):
    from scipy.signal import find_peaks
    pk, pp = find_peaks(prof, prominence=prom)
    tr, tp = find_peaks(-prof, prominence=prom)
    amp = float(np.median(pp["prominences"])) if len(pk) else 0.0
    return len(pk), len(tr), amp


def _line(c, x0, y0, x1, y1, n=400):
    t = np.linspace(0, 1, n)
    return _samp(c, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)


def m_heart(c):
    ax = _line(c, 0, -0.30, 0, 0.32, 800)
    ys = np.linspace(-0.30, 0.32, 800)
    white = ys[ax > 0.6]
    if not len(white):
        return dict(tip_y=None, lobe_bottom_y=None, notch_depth_D=0.0, axis_dark_len_D=0.0)
    top = white.min()
    lw = [ys[_line(c, sx, -0.3, sx, 0.32, 800) > 0.6] for sx in (0.08, -0.08)]
    lob = max((v.max() for v in lw if len(v)), default=top)
    sel = (ys < lob) & (ys > top)
    dark = ys[sel & (ax < 0.6)]
    depth = float(lob - dark.min()) if len(dark) else 0.0
    return dict(tip_y=round(float(top), 3), lobe_bottom_y=round(float(lob), 3), notch_depth_D=round(depth, 3),
                axis_dark_len_D=round(float(len(dark) * (0.62 / 799)), 3))


def m_push(c):
    out = {}
    for y in (-0.1, 0.0, 0.1):
        n_pk, n_tr, amp = _bands(_line(c, -0.3, y, 0.3, y, 600), 0.08)
        out[f"y{y:+.1f}"] = dict(white_bands=n_pk, dark_seams=n_tr, amp=round(amp, 2))
    return out


def m_layered(c):
    out = {}
    for x in (-0.1, -0.05, 0.05, 0.1):
        n_pk, n_tr, amp = _bands(_line(c, x, -0.2, x, 0.2, 400), 0.04)
        out[f"x{x:+.2f}"] = dict(bright=n_pk, dark=n_tr, amp=round(amp, 2))
    n_pk, n_tr, amp = _bands(_line(c, -0.25, 0.1, 0.25, 0.1, 500), 0.04)
    out["y+0.10"] = dict(bright=n_pk, dark=n_tr, amp=round(amp, 2))
    vals = []
    for sx in (-1, 1):
        xs, ys = np.meshgrid(np.linspace(sx * 0.02, sx * 0.12, 30), np.linspace(0.03, 0.15, 30))
        vals.append(float(np.percentile(_samp(c, xs.ravel(), ys.ravel()), 95)))
    out["curl_c95"] = [round(v, 3) for v in vals]
    out["axis_upper_c_mean"] = round(float(_line(c, 0, -0.2, 0, 0.1, 300)[:150].mean()), 3)
    return out


def m_tulip(c):
    n_pk, n_tr, amp = _bands(_line(c, 0, -0.25, 0, 0.34, 600), 0.08)
    off = [_bands(_line(c, x, -0.25, x, 0.34, 600), 0.08) for x in (-0.06, 0.06)]
    return dict(axis_bright=n_pk, axis_dark=n_tr, axis_amp=round(amp, 2), off_dark=[o[1] for o in off],
                base_center_c=round(float(_samp(c, [0.0], [0.27])[0]), 3))


def m_leaf(c):
    out = {}
    for x in (-0.08, -0.05, 0.05, 0.08):
        n_pk, n_tr, amp = _bands(_line(c, x, -0.05, x, 0.27, 400), 0.05)
        out[f"x{x:+.2f}"] = dict(veins=n_pk, amp=round(amp, 2))
    out["mean_veins"] = round(float(np.mean([v["veins"] for v in out.values()])), 2)
    out["mean_amp"] = round(float(np.mean([v["amp"] for v in list(out.values())[:4]])), 3)
    out["axis_upper_c_mean"] = round(float(_line(c, 0, -0.2, 0, 0.2, 400)[:200].mean()), 3)
    return out


def m_swan(c):
    h = 1.0 / c.shape[0]
    ys = np.arange(0.0, 0.261, h)
    xg = np.arange(-0.32, -0.06, h / 2)
    prev = None
    cmax, width = [], []
    for y in ys:
        prof = _samp(c, xg, np.full_like(xg, y))
        if prev is not None:
            prof = np.where(np.abs(xg - prev) <= 0.04, prof, 0)
        i = int(np.argmax(prof))
        prev = xg[i]
        cmax.append(prof[i])
        j0 = i
        while j0 > 0 and prof[j0 - 1] > 0.5:
            j0 -= 1
        j1 = i
        while j1 < len(xg) - 1 and prof[j1 + 1] > 0.5:
            j1 += 1
        width.append((j1 - j0 + 1) * h / 2 if prof[i] > 0.5 else 0.0)
    cmax, width = np.array(cmax), np.array(width)
    xs, yy = np.meshgrid(np.arange(-0.32, -0.08, h / 2), np.arange(0.26, 0.34, h / 2))
    wh = _samp(c, xs.ravel(), yy.ravel()).reshape(xs.shape) > 0.6
    beak_left = float(xs[wh].min()) if wh.any() else None
    up = _line(c, -0.21, 0.03, -0.08, -0.07, 200)
    return dict(neck_min_cmax=round(float(cmax.min()), 3), neck_frac_cmax_gt_0p6=round(float((cmax > 0.6).mean()), 3),
                neck_width_med_mm=round(float(np.median(width)) * 80, 2), neck_width_max_mm=round(float(width.max()) * 80, 2),
                head_area_mm2=round(float(wh.sum() * (h / 2) ** 2 * 6400), 1),
                beak_left_x=None if beak_left is None else round(beak_left, 3),
                upper_neck_arc_c_min=round(float(up.min()), 3), upper_neck_arc_c_mean=round(float(up.mean()), 3))


STRUCT = dict(heart=m_heart, push_heart=m_push, layered_heart=m_layered, tulip=m_tulip, leaf=m_leaf, swan=m_swan)
