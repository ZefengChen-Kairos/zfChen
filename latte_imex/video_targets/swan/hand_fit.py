"""Hand-read liquid outline points (image px) -> fixed-shape ellipses for the swan tilt phase -> hand_ellipses.json.

The outline points were read by eye on native-resolution crops with a 10 px grid (tiles/z_*.png): the far contact line
(liquid / far inner wall) and the left / right contact lines; the near side of the liquid is hidden by the near rim and
is not used.  Shape fixed to the level phase (b/a = 0.752, phi = -3.032 rad: the liquid surface is horizontal whatever
the cup tilt, so its image ellipse keeps the camera's axis ratio); free: centre and size.  At 8.0-9.0 s the extractor's
own (quality-2) ellipse is ~12 px too small on the image-left side (tiles/pix_ellcheck_level.png); from 9.6 s on it
matches the outline and is used as is (t_until).

    python latte_imex/video_targets/swan/hand_fit.py      # writes hand_ellipses.json and tiles/hand_ellipses.png
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import video_extract as VX  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RATIO, PHI = 0.752, -3.032

PTS = {   # t: list of (x, y) outline points [px]
    0.00: [(150, 260), (170, 257), (190, 256), (210, 255), (225, 256), (305, 280), (315, 300), (321, 320), (323, 335)],
    1.33: [(130, 272), (150, 267), (170, 265), (190, 264), (210, 264), (222, 264), (322, 295), (328, 310), (331, 330),
           (331, 350)],
    2.00: [(130, 280), (150, 272), (170, 268), (190, 266), (210, 265), (225, 265), (320, 295), (330, 315), (335, 335),
           (335, 350)],
    2.67: [(130, 278), (150, 268), (170, 262), (190, 258), (210, 256), (225, 256), (330, 290), (345, 310), (350, 330),
           (350, 350)],
    3.33: [(113, 300), (130, 280), (150, 268), (170, 260), (190, 257), (210, 255), (230, 255), (345, 310), (350, 330),
           (352, 350), (350, 370)],
    4.00: [(120, 300), (140, 278), (160, 268), (180, 262), (200, 258), (220, 256), (350, 320), (355, 340), (355, 360),
           (350, 380)],
    5.00: [(130, 284), (150, 272), (170, 265), (190, 260), (210, 258), (228, 257), (350, 330), (355, 350), (355, 370)],
    6.00: [(100, 290), (120, 270), (140, 260), (160, 255), (180, 252), (200, 250), (220, 250), (90, 310), (88, 330),
           (350, 340), (355, 360), (352, 385)],
    7.00: [(100, 288), (120, 272), (140, 262), (160, 256), (180, 252), (200, 250), (220, 250), (88, 320), (88, 340),
           (352, 350), (355, 370), (350, 390)],
    7.60: [(100, 290), (120, 270), (140, 262), (160, 256), (180, 252), (200, 250), (220, 250), (240, 252), (88, 320),
           (87, 340), (90, 360), (350, 340), (355, 360), (352, 385), (345, 400)],
    8.00: [(100, 285), (120, 268), (140, 260), (160, 255), (180, 252), (200, 250), (220, 250), (240, 251), (85, 340),
           (85, 360), (357, 360)],
}


# size fixed where the outline (top + right contact lines only, cup strongly tilted) does not constrain it: the
# liquid-surface diameter grows smoothly while the bowl fills (fits: a = 128 at 3.3 s, 137 at 6-8 s, 140-150 later)
A_FIXED = {0.00: 116.0, 1.33: 120.0, 2.00: 123.0, 2.67: 126.0}
# from TRACK_FROM on the extractor's quality-2 track is used (hand fit and track agree within 5 px at 8.6 s; at 8.0 s
# the track is ~12 px too small on the image-left side)
TRACK_FROM = 8.6


def fit(pts, p0=(230, 340, 125), a_fixed=None):
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)

    def res(q):
        a = a_fixed if a_fixed is not None else q[2]
        return VX.ell_dist((q[0], q[1], a, a * RATIO, PHI), x, y)
    q = least_squares(res, p0[:2] if a_fixed is not None else p0, loss="soft_l1", f_scale=2.0).x
    a = a_fixed if a_fixed is not None else q[2]
    p = VX.canon((q[0], q[1], a, a * RATIO, PHI))
    r = res(q)
    return [round(float(v), 2) for v in p[:4]] + [round(float(p[4]), 4)], float(np.sqrt(np.mean(r ** 2)))


def load_extra():
    path = os.path.join(HERE, "hand_points_extra.json")
    if os.path.exists(path):
        return {float(k): v for k, v in json.load(open(path)).items()}
    return {}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = dict(PTS)
    pts.update(load_extra())
    out = {}
    frs = np.load(os.path.join(HERE, "frames.npz"))
    for t in sorted(pts):
        p, rms = fit(pts[t], a_fixed=A_FIXED.get(t))
        out[t] = p
        print(f"t={t:5.2f}  ell=({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}, {p[3]:.1f}, {p[4]:.3f})  rms {rms:.1f} px  n={len(pts[t])}")
    hand_fits = dict(out)
    for t in [k for k in out if k >= TRACK_FROM - 1e-9]:
        del out[t]
    i8 = int(round(TRACK_FROM * 15))
    out[TRACK_FROM] = [round(float(v), 2) for v in frs["track_ell"][i8][:4]] + [round(float(frs["track_ell"][i8][4]), 4)]
    json.dump({"note": __doc__.split("\n\n")[0], "ratio": RATIO, "phi": PHI, "t_until": TRACK_FROM,
               "hand_fits_all": {f"{t:.2f}": v for t, v in hand_fits.items()},
               "ell": {f"{t:.2f}": v for t, v in out.items()}, "points": {f"{t:.2f}": v for t, v in pts.items()}},
              open(os.path.join(HERE, "hand_ellipses.json"), "w"), indent=1)
    frames, fps = VX.read_video(VX.clip_path("swan"))
    fr = np.load(os.path.join(HERE, "frames.npz"))
    out = hand_fits
    ts = sorted(out)
    nc = 4
    nr = int(math.ceil(len(ts) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.9 * nr), dpi=80)
    for t, ax in zip(ts, axs.flat):
        i = int(round(t * fps))
        ax.imshow(frames[i][170:470, 50:420], extent=[49.5, 419.5, 469.5, 169.5])
        for p, c in [(out[t], "lime"), (fr["track_ell"][i], "magenta")]:
            th = np.linspace(0, 2 * np.pi, 200)
            ex, ey = VX.basis(p)
            ax.plot(p[0] + p[2] * np.cos(th) * ex[0] + p[3] * np.sin(th) * ey[0],
                    p[1] + p[2] * np.cos(th) * ex[1] + p[3] * np.sin(th) * ey[1], "-", c=c, lw=0.8)
        P = np.array(pts[t])
        ax.plot(P[:, 0], P[:, 1], "c.", ms=4)
        ax.set_xlim(50, 420); ax.set_ylim(470, 170)
        ax.set_title(f"t={t:.2f} hand (lime) vs track q{int(fr['track_quality'][i])} (magenta)", fontsize=8)
    for ax in axs.flat[len(ts):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "tiles", "hand_ellipses.png"))
