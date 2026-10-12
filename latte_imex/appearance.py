"""Appearance rendering anchored on the real initial surface (senior V1 section 4), in the rectified cup frame (T).

    I_hat_t(x) = [1 - alpha(C_t(x); k)] I_0(x) + alpha(C_t(x); k) [E(x) * rho_m]          (linear RGB)
    alpha(C; k) = (1 - exp(-k C)) / (1 - exp(-k)),   alpha(C; 0) = C
    E(x) = Smooth(I_0(x)) / rho_b                                                         (fixed, low-frequency)

C is the model's bounded composition (raft coverage c, through the same camera PSF as calib.observe_model).  The
anchors come from the same clip and are frozen before k is learned (k is the only learned appearance parameter, and
it is learned only after the dynamics are frozen, on the calibration clip, mode B: final unoccluded frame RGB):
  I_0    the clips have no unoccluded pre-pour frame (the pitcher is in view from the first frame), so I_0 is the
         per-point median of the rectified linear RGB over the reliable frames in which the point is visible and
         still brown (before it first turns white); points never seen brown are filled by normalised convolution;
  rho_b  median of I_0 over the disc;   rho_m  median linear RGB of visible white (w > 0.9) in the last frames;
  E      Gaussian(I_0, SIGMA_E_D) / rho_b.
  Default background = E * rho_b (the low-frequency part of I_0; the senior's simplification for a near-constant E):
  the estimated I_0 is patchy (32 % of tulip's surface is never seen brown and unoccluded, and the early frames are
  taken under the tilt), so its texture would be an artefact of the estimate.  background='I0' uses it anyway.
Checks (section 5.4, E5): alpha(0) = 0, alpha(1) = 1, k -> 0 linear limit, monotone; C = 0 gives I_hat = I_0.
"""
import math
import os

import numpy as np
from scipy import ndimage as ndi

from . import calib as CB
from . import video_extract as VE

SIGMA_E_D = 0.08         # illumination-map smoothing [cup diameters]
BROWN_W = 0.15           # a point counts as brown below this whiteness
FILL_SIGMA_D = 0.03


def srgb_to_lin(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def alpha(C, k):
    C = np.clip(np.asarray(C, np.float64), 0.0, 1.0)
    if abs(k) < 1e-6:
        return C
    return (1.0 - np.exp(-k * C)) / (1.0 - math.exp(-k))


def rectified_rgb(clip, idx=None):
    """Linear RGB of the clip's kept frames (physical time axis) on the T grid [n, M, M, 3]."""
    res = clip["res"]
    frames, _ = VE.read_video(VE.clip_path(clip["name"]))
    M = clip["M"]
    idx = range(len(res["t"])) if idx is None else idx
    out = []
    for j in idx:
        img = frames[int(res["frame_index"][j])].astype(np.float64) / 255.0
        out.append(srgb_to_lin(VE.rectify(img, res["ell"][j], M, float(res["rot"]), bool(res["flip"]))))
    return np.array(out, np.float32)


def _fill(img, known, sigma):
    """Normalised-convolution fill of unknown pixels (repeated until all disc pixels are covered)."""
    out = img.copy()
    k = known.astype(np.float64)
    for _ in range(12):
        if k.min() > 0:
            break
        num = np.stack([ndi.gaussian_filter(out[..., c] * k, sigma) for c in range(3)], -1)
        den = ndi.gaussian_filter(k, sigma)[..., None]
        est = np.where(den > 1e-6, num / np.maximum(den, 1e-12), 0.0)
        new = (k == 0) & (den[..., 0] > 1e-3)
        out[new] = est[new]
        k = np.where(new, 1.0, k)
        sigma *= 1.5
    return out


def anchors(clip, rgb=None):
    res = clip["res"]
    M = clip["M"]
    rgb = rectified_rgb(clip) if rgb is None else rgb
    disk = res["disk"].astype(bool)
    W, V = res["white"].astype(np.float32), res["valid"] & disk[None]
    ok_frame = clip["shape_ok"]
    turned = np.zeros((M, M), bool)
    stack = np.full((len(rgb), M, M, 3), np.nan, np.float32)
    for j in range(len(rgb)):
        turned |= V[j] & (W[j] > 0.5)
        if not ok_frame[j]:
            continue
        sel = V[j] & (W[j] < BROWN_W) & ~turned
        stack[j][sel] = rgb[j][sel]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        I0 = np.nanmedian(stack, axis=0)
    known = np.isfinite(I0[..., 0]) & disk
    I0 = np.where(np.isfinite(I0), I0, 0.0)
    I0 = _fill(I0, known, FILL_SIGMA_D * M)
    rho_b = np.median(I0[disk], axis=0)
    last = [j for j in range(len(rgb) - 1, -1, -1) if ok_frame[j]][:3]
    wm = np.concatenate([rgb[j][V[j] & (W[j] > 0.9)] for j in last]) if last else np.zeros((0, 3))
    rho_m = np.median(wm, axis=0) if len(wm) else np.array([0.8, 0.78, 0.72])
    E = np.stack([ndi.gaussian_filter(I0[..., c], SIGMA_E_D * M) for c in range(3)], -1) / rho_b
    return dict(I0=I0.astype(np.float32), known_frac=float(known.sum() / disk.sum()), rho_b=rho_b, rho_m=rho_m,
                E=E.astype(np.float32), disk=disk)


def render(C_T, anc, k, background="smooth"):
    """Model composition on the T grid [M, M] (or [n, M, M]) -> sRGB in [0, 1]."""
    a = alpha(C_T, k)[..., None]
    bg = anc["E"] * anc["rho_b"] if background == "smooth" else anc["I0"]
    lin = (1.0 - a) * bg + a * (anc["E"] * anc["rho_m"])
    out = lin_to_srgb(lin)
    out[..., ~anc["disk"], :] = 1.0
    return out


def rgb_loss(pred_srgb, true_lin, valid, eps=1e-3):
    """Charbonnier L1 in linear RGB on valid pixels."""
    d = srgb_to_lin(pred_srgb) - true_lin
    return float(np.sqrt(d[valid] ** 2 + eps ** 2).mean())


def fit_k(C_final, true_final_lin, valid, anc, ks=None):
    """Mode B: k from the final unoccluded frame only (grid search on a bounded interval; k = 0 is the linear
    baseline and is always reported)."""
    ks = np.concatenate([[0.0], np.geomspace(0.05, 12.0, 40)]) if ks is None else ks
    losses = [rgb_loss(render(C_final, anc, k), true_final_lin, valid) for k in ks]
    i = int(np.argmin(losses))
    return dict(k=float(ks[i]), loss=float(losses[i]), loss_linear=float(losses[0]), ks=ks.tolist(), losses=losses)


def checks(anc):
    """E5 checks: endpoints of alpha, linear limit, monotonicity, C = 0 reproduces I_0."""
    C = np.linspace(0, 1, 101)
    out = dict(alpha0=float(alpha(0.0, 3.0)), alpha1=float(alpha(1.0, 3.0)),
               linear_limit=float(np.abs(alpha(C, 1e-7) - C).max()),
               monotone=bool(all(np.all(np.diff(alpha(C, k)) >= 0) for k in (0.0, 0.5, 3.0, 10.0))))
    r0 = render(np.zeros(anc["I0"].shape[:2]), anc, 3.0, background="I0")
    out["C0_reproduces_I0"] = float(np.abs(srgb_to_lin(r0)[anc["disk"]] - anc["I0"][anc["disk"]]).max())
    return out
