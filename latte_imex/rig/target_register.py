"""Register the hand-checked finished-pattern targets (targets/<p>.npz) onto the video frame of the pour.

The targets were drawn on one camera-1 frame taken 0.5-2.9 s after the pattern phase, with that camera's own depth
plane and disc fit, so they sit in a slightly different cup frame from the video supervision (video_obs.py, disc.json
of cup_register.py): rotation up to ~2 deg, scale 3-8 %, shift up to ~4 mm.  Registration = the similarity transform
(rotation about the disc centre, scale, shift) that maximises the NCC of the high-passed LAB lightness of the target
image and of the last fused top view inside r < 0.9 (coarse grid + pattern search).  This is an image-to-image
measurement registration (the simulation is not involved).  The target's white / valid / img are resampled onto the
video convention (512 grid, extent 1.05 disc radii of L_m); the camera-1 originals are kept as targets/<p>_cam1.npz and
the transforms in targets/registration.json.
    python3 -m latte_imex.rig.target_register [pattern ...]"""
import json
import os
import shutil
import sys

import cv2
import numpy as np

SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TGT = os.path.join(REPO, "raw_data", "rig", "targets")
PATTERNS = ("heart", "push_heart", "leaf", "swan", "tulip")
NR, EXT = 256, 1.05


def _hp(g):
    g = g.astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 10)


def _warp(img, rot, sc, du, dv, n, interp):
    """img on the target grid (n x n, extent EXT) sampled at video coordinates: target = Rot(-a) ((U, V) - d) / sc."""
    s = np.linspace(-EXT, EXT, n)
    U, V = np.meshgrid(s, s)
    a = np.radians(rot)
    X, Y = (U - du) / sc, (V - dv) / sc
    Ut, Vt = np.cos(a) * X + np.sin(a) * Y, -np.sin(a) * X + np.cos(a) * Y
    m = img.shape[0]
    mx = ((Ut + EXT) / (2 * EXT) * (m - 1)).astype(np.float32)
    my = ((Vt + EXT) / (2 * EXT) * (m - 1)).astype(np.float32)
    return cv2.remap(img, mx, my, interp, borderValue=0)


def last_frame(p, n):
    """Last fused top view on the video convention (registered track, length unit L_m), LAB L."""
    D = json.load(open(f"{SCR}/{p}/disc.json"))
    fi = len(D) - 1
    Z = np.load(f"{SCR}/{p}/top/{fi:04d}.npz")
    img, c0, PX, H = Z["img"], Z["c0"], float(Z["px"]), float(Z["half"])
    cen, R = np.array(D[fi]["disc_centre"]), D[fi].get("L_m") or D[fi]["disc_R"]
    s = np.linspace(-EXT, EXT, n)
    U, V = np.meshgrid(s, s)
    mx = ((cen[0] + U * R - c0[0] + H) / PX - 0.5).astype(np.float32)
    my = ((cen[1] + V * R - c0[1] + H) / PX - 0.5).astype(np.float32)
    return cv2.cvtColor(cv2.remap(img, mx, my, cv2.INTER_LINEAR), cv2.COLOR_BGR2LAB)[..., 0]


def register(p, log=print):
    src = os.path.join(TGT, f"{p}_cam1.npz")
    if not os.path.exists(src):
        shutil.copy(os.path.join(TGT, f"{p}.npz"), src)
    T = dict(np.load(src))
    tg = cv2.cvtColor(T["img"], cv2.COLOR_BGR2LAB)[..., 0]
    B = _hp(last_frame(p, NR))
    s = np.linspace(-EXT, EXT, NR)
    U, V = np.meshgrid(s, s)
    M = U ** 2 + V ** 2 < 0.9 ** 2

    def score(x):
        A = _hp(_warp(tg, *x, NR, cv2.INTER_AREA))
        a, b = A[M], B[M]
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))

    best = (-2.0, (0.0, 1.0, 0.0, 0.0))
    for rot in np.arange(-12, 12.1, 2.0):
        for sc in (0.92, 0.96, 1.0, 1.04, 1.08):
            for du in np.arange(-0.12, 0.121, 0.04):
                for dv in np.arange(-0.12, 0.121, 0.04):
                    c = score((rot, sc, du, dv))
                    if c > best[0]:
                        best = (c, (rot, sc, du, dv))
    x, step = np.array(best[1], float), np.array([1.0, 0.02, 0.02, 0.02])
    for _ in range(60):
        moved = False
        for j in range(4):
            for sg in (-1, 1):
                y = x.copy()
                y[j] += sg * step[j]
                c = score(tuple(y))
                if c > best[0]:
                    best, x, moved = (c, tuple(y)), y, True
        if not moved:
            step = step / 2
            if step[0] < 0.05:
                break
    rot, sc, du, dv = best[1]
    n = T["white"].shape[0]
    out = dict(T)
    out["white"] = _warp(T["white"].astype(np.uint8), rot, sc, du, dv, n, cv2.INTER_NEAREST).astype(bool)
    out["valid"] = _warp(T["valid"].astype(np.uint8), rot, sc, du, dv, n, cv2.INTER_NEAREST).astype(bool)
    out["img"] = _warp(T["img"], rot, sc, du, dv, n, cv2.INTER_LINEAR)
    out["source"] = str(T["source"]) + " | registered onto the video frame (target_register.py)"
    np.savez_compressed(os.path.join(TGT, f"{p}.npz"), **out)
    D = json.load(open(f"{SCR}/{p}/disc.json"))
    Lmm = (D[-1].get("L_m") or D[-1]["disc_R"]) * 1e3
    rec = dict(ncc_identity=score((0.0, 1.0, 0.0, 0.0)), ncc=best[0], rot_deg=rot, scale=sc, du=du, dv=dv,
               shift_mm=[du * Lmm, dv * Lmm])
    log(f"{p:10s} NCC {rec['ncc_identity']:.2f} -> {rec['ncc']:.2f}: rotate {rot:+.2f} deg, scale {sc:.3f}, "
        f"shift ({du * Lmm:+.1f}, {dv * Lmm:+.1f}) mm")
    sheet = np.hstack([cv2.resize(tg, (NR, NR), interpolation=cv2.INTER_AREA), last_frame(p, NR),
                       _warp(tg, rot, sc, du, dv, NR, cv2.INTER_AREA)])
    return rec, sheet


def main(argv=None):
    pats = (argv or sys.argv[1:]) or list(PATTERNS)
    fn = os.path.join(TGT, "registration.json")
    R = json.load(open(fn)) if os.path.exists(fn) else {}
    rows = []
    for p in pats:
        R[p], sh = register(p)
        rows.append(sh)
    json.dump(R, open(fn, "w"), indent=1)
    cv2.imwrite(os.path.join(TGT, "registration.jpg"), np.vstack(rows))


if __name__ == "__main__":
    main()
