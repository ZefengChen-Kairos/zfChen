"""Gap-preserving reference of the finished patterns (the yardstick of topo_metric.py; V2_SURFACE_MODEL.md 10.5).

Input: only targets/<p>.npz img and valid (the camera-1 frame 0.5-2.9 s after the pour, registered onto the video grid
by target_register.py: 512 px over [-1.05, 1.05] L_m, 0.173 mm / px).  The stored Otsu mask is not used except that
Otsu is recomputed as the seed.  One rule, one set of constants for all five pours; no video data, no video whiteness.
  0. valid = target valid & r <= R_IN (as video_obs.py: the bright meniscus ring at the wall is excluded)
  1. linear light: sRGB decode (optical blur mixes radiance linearly, not gamma-encoded values)
  2. seed classes: Otsu on LAB lightness inside valid; crema seed = Otsu-brown farther than SEED_D from Otsu-white and
     warmer than the chromaticity midpoint of the two Otsu classes, foam seed = Otsu-white farther than SEED_D from
     Otsu-brown and colder than that midpoint (glare is bright but crema-coloured)
  3. local end-members: C(x), F(x) = normalised Gaussian convolutions (SIG_END) of the linear colour over the crema /
     foam seeds (illumination, exposure, vignetting), with a weak pull to the global seed medians where no seed is near
  4. foam fraction f = min(fl, fc), smoothed SIG_F (noise):
       fl = (x - C).(F - C) / |F - C|^2   linear-light projection (0 = crema, 1 = foam plateau)
       fc = the a of x = (1 - a) C + a F that reproduces x's chromaticity (B - R) / (R + G + B): exact for blur mixtures,
            blind to intensity -> specular glare / reflections on crema (bright but warm) are not foam
  5. local white level P(x) = max of f within R_PEAK (the flanking white of any gap / the peak of a thin white band),
     smoothed SIG_P; half-max rule: white  <=>  P >= P_MIN  and  f >= HALF P (the 50 % point between the local white
     level and crema is where a blurred step edge is; for a thin dark gap between white bands it marks the gap's full
     width at half depth; a thin white band keeps its FWHM)
  6. clean-up: white specks and brown holes < MIN_SPECK_MM2 removed; components counted at >= MIN_COMP_MM2
  7. gaps = closing(white, disk R_GAP) - white, inside valid (brown flanked by white: lines, layer gaps, notches up to
     4 mm wide); pieces < MIN_GAP_MM2 dropped (closing residues at concave corners facing open crema)
  8. information only: faint = white pixels with black top-hat depth of f (radius R_FAINT) >= FAINT_DEPTH (shallow
     layer stripes that stay above half-max: most likely unresolved sub-PSF brown hairlines; neutral in the metric)
Result: tulip 3 parts (wings 768, heart 466, teardrop 303 mm^2; the Otsu mask merges teardrop and heart), leaf 1,
heart 1, push_heart 1, swan 2 parts >= 20 mm^2.  The tulip split holds for half-max fractions 0.35-0.65 (f at the
teardrop / heart saddle <= 0.32) and survives a video-like blur up to sigma ~1.0 mm.
Frame registration (offset()): the target was registered to the LAST video frame only; the two before it drift by up to
1.6 mm, which a strict 1-px gap metric cannot absorb, so each scored frame's translation against the target image is
measured (NCC of high-passed LAB lightness, sigma 3 mm, r < 0.9, both valid; integer search on a 256 grid +-2.8 mm,
refined on the 512 grid +-2 px with a parabolic sub-pixel peak; W never used).
    python3 -m latte_imex.rig.topo_ref [pour ...]      -> raw_data/rig/targets/<p>_ref.npz
    python3 -m latte_imex.rig.topo_ref --offsets       -> raw_data/rig/targets/frame_offsets.json (video_obs frames)
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TGT = os.path.join(REPO, "raw_data", "rig", "targets")
VID = os.path.join(REPO, "raw_data", "rig", "video_obs")
POURS = ("tulip", "leaf", "heart", "push_heart", "swan")
EXT = 1.05
L_M = 0.042          # [m] cup radius unit of the five pours (disc.json L_m 41.5-42.3 mm)
PX_MM = 2 * EXT * L_M * 1e3 / 511      # 0.1726 mm per pixel at 512 (pixel centres linspace(-EXT, EXT, 512))

SEED_D = 0.6         # [mm] seed distance from the Otsu boundary
SIG_END = 4.0        # [mm] end-member smoothing (illumination scale)
SIG_F = 0.25         # [mm] noise smoothing of f
R_PEAK = 1.5         # [mm] local white level radius (half of a 3 mm gap)
SIG_P = 0.5          # [mm] smoothing of the local white level
HALF = 0.5           # half-max
P_MIN = 0.5          # a structure counts as white only if its peak reaches half the foam-crema contrast
MIN_SPECK_MM2 = 0.5  # specks / pin holes removed
MIN_COMP_MM2 = 2.0   # components counted
R_GAP = 2.0          # [mm] closing radius defining the gaps (fills brown up to 4 mm wide between white)
MIN_GAP_MM2 = 1.0    # [mm^2] gap pieces smaller than this are closing residues at concave corners
R_FAINT = 1.25       # [mm] faint-line top-hat radius (lines up to 2.5 mm wide)
FAINT_DEPTH = 0.2    # faint-line minimum dip depth (fraction of the foam-crema contrast)
R_IN = 0.96          # [disc radii] as video_obs.py
CHROMA = True        # whiteness = min(linear-light fraction, chromatic fraction)
# frame registration
N_LAST = 3           # scored frames (topo_metric.N_LAST)
SIG_HP_MM = 3.0      # high-pass of the lightness
L_M_MM, NR = 42.0, 512
PXR = 2 * EXT * L_M_MM / (NR - 1)


def mm(x):
    return x / PX_MM


def disk(r):
    r = int(round(r))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def srgb_lin(img):
    c = img.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def otsu(x):
    h, e = np.histogram(x, 256)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def normconv(x, m, sig, prior, k=1e-3):
    g = lambda a: ndi.gaussian_filter(a, sig, mode="nearest")
    wm = g(m.astype(float))
    if x.ndim == 3:
        return np.dstack([(g(x[..., c] * m) + k * prior[c]) / (wm + k) for c in range(x.shape[2])])
    return (g(x * m) + k * prior) / (wm + k)


def clean(white, valid):
    a = mm(1) ** 2 * MIN_SPECK_MM2
    lab, n = ndi.label(white)
    if n:
        sz = np.bincount(lab.ravel()); small = sz < a; small[0] = False
        white = white & ~small[lab]
    lab, n = ndi.label(~white & valid)
    if n:
        sz = np.bincount(lab.ravel()); small = sz < a; small[0] = False
        white = white | small[lab]
    return white & valid


def components(white, min_mm2=MIN_COMP_MM2, px_mm=PX_MM):
    lab, n = ndi.label(white, structure=np.ones((3, 3), bool))
    if not n:
        return 0, lab
    sz = np.bincount(lab.ravel())[1:]
    return int((sz >= min_mm2 / px_mm ** 2).sum()), lab


def reference(img, valid):
    """img BGR uint8 (512 x 512), valid -> dict(white, gaps, valid, faint, f, P, otsu, otsu_th)."""
    s = np.linspace(-EXT, EXT, img.shape[0])
    v = valid.astype(bool) & (np.hypot(*np.meshgrid(s, s)) <= R_IN)
    lin = srgb_lin(img)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    th = otsu(L[v])
    wo = (L > th) & v
    dw = ndi.distance_transform_edt(~wo)          # distance to Otsu white
    db = ndi.distance_transform_edt(wo | ~v)       # inside white: distance to brown / invalid
    dv = ndi.distance_transform_edt(v)
    # seeds must also agree in colour (chromaticity b = (B - R) / S on the right side of the midpoint of the two
    # Otsu classes' medians): bright crema-coloured glare is not a foam seed, cold foam specks are not crema seeds
    S0 = lin.sum(2); b0 = (lin[..., 0] - lin[..., 2]) / np.maximum(S0, 1e-6)
    bmid = 0.5 * (np.median(b0[wo]) + np.median(b0[v & ~wo]))
    cs = v & ~wo & (dw > mm(SEED_D)) & (dv > mm(SEED_D)) & (b0 < bmid)
    fs = wo & (db > mm(SEED_D)) & (b0 > bmid)
    cmed = np.median(lin[cs], 0); fmed = np.median(lin[fs], 0)
    C = normconv(lin, cs, mm(SIG_END), cmed)
    F = normconv(lin, fs, mm(SIG_END), fmed)
    d = F - C
    fl = ((lin - C) * d).sum(2) / np.maximum((d * d).sum(2), 1e-6)       # linear-light projection
    # chromatic unmixing: the foam fraction a of x = (1 - a) C + a F that reproduces the chromaticity b = (B - R) / S
    # of x (exact for blur mixtures, blind to intensity: specular glare on crema is bright but crema-coloured)
    S = lin.sum(2); b = (lin[..., 0] - lin[..., 2]) / np.maximum(S, 1e-6)
    SC, SF = C.sum(2), F.sum(2); dC, dF = C[..., 0] - C[..., 2], F[..., 0] - F[..., 2]
    den = (dF - dC) - b * (SF - SC)
    fc = (b * SC - dC) / np.where(np.abs(den) < 1e-6, 1e-6, den)
    f = np.minimum(fl, np.clip(fc, -1.0, 2.0)) if CHROMA else fl
    # outside valid: replace by nearest valid value (no fake edges at the disc border)
    idx = ndi.distance_transform_edt(~v, return_distances=False, return_indices=True)
    f = f[tuple(idx)]
    f = ndi.gaussian_filter(f, mm(SIG_F))
    P = ndi.grey_dilation(f, footprint=disk(mm(R_PEAK)))
    P = ndi.gaussian_filter(P, mm(SIG_P))
    white = (P >= P_MIN) & (f >= HALF * P) & v
    white = clean(white, v)
    closed = ndi.binary_closing(np.pad(white, 20), structure=disk(mm(R_GAP)))[20:-20, 20:-20]
    gaps = closed & ~white & v
    lab, n = ndi.label(gaps, structure=np.ones((3, 3), bool))
    if n:
        small = np.bincount(lab.ravel()) < MIN_GAP_MM2 / PX_MM ** 2
        small[0] = False
        gaps &= ~small[lab]
    # information only (neutral in the metric): faint dark lines inside the white whose dip stays above half-max --
    # black top-hat depth of f over R_FAINT >= FAINT_DEPTH of the foam-crema contrast
    bth = ndi.grey_closing(f, footprint=disk(mm(R_FAINT))) - f
    faint = white & (bth >= FAINT_DEPTH)
    return dict(white=white, gaps=gaps, valid=v, faint=faint, f=f.astype(np.float32), P=P.astype(np.float32), otsu=wo,
                otsu_th=th)


def build(pours=POURS):
    for p in pours:
        T = np.load(f"{TGT}/{p}.npz")
        r = reference(T["img"], T["valid"])
        n, _ = components(r["white"])
        no, _ = components(T["white"] & r["valid"])
        np.savez_compressed(f"{TGT}/{p}_ref.npz", white=r["white"], valid=r["valid"], gaps=r["gaps"], faint=r["faint"],
                            n_comp=n, f=r["f"].astype(np.float16), px_mm=PX_MM, extent=EXT, min_comp_mm2=MIN_COMP_MM2,
                            r_gap_mm=R_GAP, min_gap_mm2=MIN_GAP_MM2,
                            rule="topo_ref.py: linear-light crema->foam mixing fraction, local half-max (f >= P/2, P>=0.5)")
        wf = r["white"][r["valid"]].mean(); of = T["white"][r["valid"]].mean()
        print(f"{p}: components ref {n} (otsu target {no}), white frac ref {wf:.3f} otsu {of:.3f}, gaps px {r['gaps'].sum()}")


# ------------------------------------------------------------------------------------------------ frame registration
_s = np.linspace(-EXT, EXT, NR)
_UR, _VR = np.meshgrid(_s, _s)
_g = (np.arange(NR) * 127 / (NR - 1)).astype(np.float32)
_MX, _MY = np.meshgrid(_g, _g)


def _hp(L):
    L = L.astype(np.float32)
    return L - cv2.GaussianBlur(L, (0, 0), SIG_HP_MM / PXR)


def _ncc(A, B, m, dy, dx):
    Bs = np.roll(B, (dy, dx), (0, 1))
    ms = m & np.roll(m, (dy, dx), (0, 1))
    a, b = A[ms], Bs[ms]
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def _peak(M):
    iy, ix = np.unravel_index(np.argmax(M), M.shape)

    def sub(c, i):
        if 0 < i < len(c) - 1:
            d = c[i - 1] - 2 * c[i] + c[i + 1]
            return i + (0.5 * (c[i - 1] - c[i + 1]) / d if d < 0 else 0.0)
        return float(i)
    return sub(M[:, ix], iy), sub(M[iy, :], ix), iy, ix


def offset(T, img128, valid128):
    """(du, dv) [mm] of the video frame content (img128 RGB, 128 grid) relative to the target image T, NCC at the optimum
    and unshifted.  topo_metric samples the reference at (U - du, V - dv) for that frame."""
    A = _hp(cv2.cvtColor(T["img"], cv2.COLOR_BGR2LAB)[..., 0])
    v = cv2.remap(np.ascontiguousarray(img128[..., ::-1]), _MX, _MY, cv2.INTER_CUBIC)      # RGB -> BGR, 128 -> 512
    vv = cv2.remap(valid128.astype(np.uint8), _MX, _MY, cv2.INTER_NEAREST) > 0
    B = _hp(cv2.cvtColor(v, cv2.COLOR_BGR2LAB)[..., 0])
    m = (np.hypot(_UR, _VR) < 0.9) & T["valid"].astype(bool) & vv
    A2, B2 = A[::2, ::2], B[::2, ::2]                  # coarse: 256 grid, +-8 px (2.8 mm)
    m2 = m[::2, ::2]
    R = 8
    M = np.array([[_ncc(A2, B2, m2, dy, dx) for dx in range(-R, R + 1)] for dy in range(-R, R + 1)])
    _, _, iy, ix = _peak(M)
    cy, cx = 2 * (iy - R), 2 * (ix - R)
    r = 2                                              # fine: 512 grid, +-2 px around the coarse peak
    M = np.array([[_ncc(A, B, m, cy + dy, cx + dx) for dx in range(-r, r + 1)] for dy in range(-r, r + 1)])
    fy, fx, iy, ix = _peak(M)
    dy, dx = cy + fy - r, cx + fx - r
    # np.roll(B, d) matches A: B(x - d) = A(x) -> the video content sits at -d relative to the target
    return -dx * PXR, -dy * PXR, float(M.max()), _ncc(A, B, m, 0, 0)


def frame_offsets(vid=VID, fn=os.path.join(TGT, "frame_offsets.json")):
    """{pour: {frame index: [du_mm, dv_mm, ncc, ncc_unshifted]}} of the last N_LAST frames of vid -> fn."""
    out = {}
    for p in POURS:
        T = np.load(f"{TGT}/{p}.npz")
        Z = np.load(f"{vid}/{p}.npz")
        n = len(Z["t"])
        out[p] = {}
        for k in range(n - N_LAST, n):
            du, dv, c, c0 = offset(T, Z["img"][k], Z["valid"][k])
            out[p][str(k)] = [round(du, 4), round(dv, 4), round(c, 4), round(c0, 4)]
            print(f"{p} frame {k} (of {n}): video vs target ({du:+.2f}, {dv:+.2f}) mm, NCC {c:.3f} (unshifted {c0:.3f})")
    json.dump(out, open(fn, "w"), indent=1)
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--offsets"]:
        frame_offsets()
    else:
        build(sys.argv[1:] or POURS)
