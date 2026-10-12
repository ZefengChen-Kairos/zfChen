"""Tile maker for reading the swan clip (frames for the stage script).  Run from anywhere:

    python latte_imex/video_targets/swan/make_tiles.py cam  t0 t1 fps tag [x0,x1,y0,y1]
    python latte_imex/video_targets/swan/make_tiles.py pix  t0 t1 fps tag x0,x1,y0,y1
    python latte_imex/video_targets/swan/make_tiles.py rect t0 t1 fps tag
    python latte_imex/video_targets/swan/make_tiles.py zoom t0 t1 fps tag [x0,x1,y0,y1 in M]

cam : camera crops with the liquid-surface ellipse and the M-frame grid (0.1 D; red y_M = 0, magenta x_M = 0) projected
      on the surface.  pix: native crops with pixel ticks.  rect / zoom: rectified M-frame tiles (origin lower, x_M right,
      y_M up = toward the pitcher handle / barista; the spout points to -y_M).
Ellipse: the extractor track (frames.npz track_ell) except where HAND overrides it (tilt phase, quality 0): hand-fitted
liquid-surface ellipses (cx, cy, a, b, phi) in px / rad, targets.py convention, interpolated in time.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/home/user/zfChen")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from latte_imex import video_extract as VX  # noqa: E402

NAME = "swan"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tiles")
HAND_FILE = os.path.join(HERE, "hand_ellipses.json")
_cache = {}


def data():
    if not _cache:
        fr = dict(np.load(os.path.join(HERE, "frames.npz"), allow_pickle=True))
        frames, fps = VX.read_video(VX.clip_path(NAME))
        _cache.update(fr=fr, frames=frames, fps=fps)
    return _cache["fr"], _cache["frames"]


def hand():
    if os.path.exists(HAND_FILE):
        h = json.load(open(HAND_FILE))
        return {float(k): v for k, v in h["ell"].items() if not k.startswith("_")}, float(h.get("t_until", -1))
    return {}, -1.0


def ell_of(i):
    """(ellipse, quality tag) for video frame i: 8 = hand-fitted (interpolated), else track quality."""
    fr, _ = data()
    H, t_until = hand()
    t = i / 15.0
    if H and t <= t_until + 1e-9:
        ts = sorted(H)
        p = [float(np.interp(t, ts, [H[k][j] for k in ts])) for j in range(5)]
        return tuple(p), 8
    return tuple(fr["track_ell"][i]), int(fr["track_quality"][i])


def m_to_img(p, xm, ym, rot=0.0, flip=False):
    X, Y = np.asarray(xm, float), -np.asarray(ym, float)
    if flip:
        X = -X
    th = math.radians(rot)
    Xr = math.cos(th) * X - math.sin(th) * Y
    Yr = math.sin(th) * X + math.cos(th) * Y
    ex, ey = VX.basis(p)
    cx, cy, a, b = p[:4]
    u, v = 2 * Xr, -2 * Yr
    return cx + a * u * ex[0] + b * v * ey[0], cy + a * u * ex[1] + b * v * ey[1]


def img_to_m(p, px, py, rot=0.0):
    X, Y = VX.image_to_cup(p, px, py, rot, False)
    return X, -Y


def _idx(t0, t1, fps, times=None):
    _, frames = data()
    ts = times if times is not None else np.arange(t0, t1 + 1e-6, 1.0 / fps)
    idx = sorted(set(int(round(t * 15)) for t in ts))
    return [i for i in idx if 0 <= i < len(frames)]


def _draw_ell(ax, p, c="lime"):
    tt = np.linspace(0, 2 * np.pi, 200)
    ex, ey = VX.basis(p)
    ax.plot(p[0] + p[2] * np.cos(tt) * ex[0] + p[3] * np.sin(tt) * ey[0],
            p[1] + p[2] * np.cos(tt) * ex[1] + p[3] * np.sin(tt) * ey[1], "-", c=c, lw=0.6)


def cam_tiles(t0, t1, fps, tag, box=(40, 440, 60, 480), ncol=5, grid=True, times=None, marks=None):
    fr, frames = data()
    rot = float(fr["rot"])
    idx = _idx(t0, t1, fps, times)
    n = len(idx)
    nrow = int(math.ceil(n / ncol))
    x0, x1, y0, y1 = box
    fig, axs = plt.subplots(nrow, ncol, figsize=((x1 - x0) / 100 * 1.3 * ncol, (y1 - y0) / 100 * 1.3 * nrow), dpi=80)
    axs = np.atleast_2d(axs)
    for k, ax in enumerate(axs.flat):
        ax.axis("off")
        if k >= n:
            continue
        i = idx[k]
        p, q = ell_of(i)
        ax.imshow(frames[i][y0:y1, x0:x1], extent=[x0 - .5, x1 - .5, y1 - .5, y0 - .5], interpolation="lanczos")
        _draw_ell(ax, p, "lime" if q in (2, 8) else "magenta")
        if grid:
            ys = np.linspace(-0.5, 0.5, 60)
            for g in np.arange(-0.4, 0.41, 0.1):
                ok = np.hypot(g, ys) <= 0.5
                gx, gy = m_to_img(p, np.full(ok.sum(), g), ys[ok], rot)
                ax.plot(gx, gy, "-", c="magenta" if abs(g) < 1e-6 else "cyan", lw=0.35, alpha=0.8)
                gx, gy = m_to_img(p, ys[ok], np.full(ok.sum(), g), rot)
                ax.plot(gx, gy, "-", c="red" if abs(g) < 1e-6 else "yellow", lw=0.35, alpha=0.8)
            for (xm, ym, s) in [(0.0, 0.45, "+y"), (0.45, 0.0, "+x")]:
                gx, gy = m_to_img(p, xm, ym, rot)
                ax.text(gx, gy, s, color="white", fontsize=6, ha="center", va="center")
        if marks and i in marks:
            for (mx, my) in marks[i]:
                gx, gy = m_to_img(p, mx, my, rot)
                ax.plot(gx, gy, "r+", ms=8)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_title(f"t={i/15:.2f}s q{q}", fontsize=7)
    fig.suptitle(f"{NAME} camera; grid = M frame on the surface ellipse (0.1 D; red y_M=0, magenta x_M=0)", fontsize=7)
    fig.tight_layout(pad=0.2)
    path = os.path.join(OUT, f"{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def pix_tiles(t0, t1, fps, tag, box, ncol=4, step=10, sc=3.0, times=None, ell=True):
    _, frames = data()
    idx = _idx(t0, t1, fps, times)
    n = len(idx)
    nrow = int(math.ceil(n / ncol))
    x0, x1, y0, y1 = box
    fig, axs = plt.subplots(nrow, ncol, figsize=((x1 - x0) / 100 * sc * ncol, (y1 - y0) / 100 * sc * nrow), dpi=80)
    axs = np.atleast_2d(axs)
    for k, ax in enumerate(axs.flat):
        if k >= n:
            ax.axis("off"); continue
        i = idx[k]
        ax.imshow(frames[i][y0:y1, x0:x1], extent=[x0 - .5, x1 - .5, y1 - .5, y0 - .5], interpolation="nearest")
        if ell:
            p, q = ell_of(i)
            _draw_ell(ax, p, "lime" if q in (2, 8) else "magenta")
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_xticks(np.arange(x0, x1 + 1, step)); ax.set_yticks(np.arange(y0, y1 + 1, step))
        ax.tick_params(labelsize=5, length=2)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.grid(True, lw=0.3, alpha=0.5, c="cyan")
        ax.set_title(f"t={i/15:.2f}s (#{i})", fontsize=8)
    fig.tight_layout(pad=0.3)
    path = os.path.join(OUT, f"{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def rect_tiles(t0, t1, fps, tag, ncol=5, Mres=256, win=(-0.5, 0.5, -0.5, 0.5), times=None, marks=None, step=0.1):
    fr, frames = data()
    rot = float(fr["rot"])
    idx = _idx(t0, t1, fps, times)
    n = len(idx)
    nrow = int(math.ceil(n / ncol))
    x0, x1, y0, y1 = win
    s = 2.8 / max(x1 - x0, y1 - y0)
    fig, axs = plt.subplots(nrow, ncol, figsize=(s * (x1 - x0) * ncol, (s * (y1 - y0) + 0.2) * nrow), dpi=90)
    axs = np.atleast_2d(axs)
    for k, ax in enumerate(axs.flat):
        if k >= n:
            ax.axis("off"); continue
        i = idx[k]
        p, q = ell_of(i)
        rgbM = VX.rectify(frames[i].astype(np.float64) / 255.0, p, Mres, rot, False)[::-1]   # rows grow with y_M
        ax.imshow(np.clip(rgbM, 0, 1), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], interpolation="lanczos")
        for g in np.arange(-0.5, 0.501, step):
            major = abs(round(g / 0.1) * 0.1 - g) < 1e-6
            ax.axhline(g, c="red" if abs(g) < 1e-6 else "yellow", lw=0.5 if major else 0.2, alpha=0.6)
            ax.axvline(g, c="magenta" if abs(g) < 1e-6 else "cyan", lw=0.5 if major else 0.2, alpha=0.6)
        ax.add_patch(plt.Circle((0, 0), 0.5, fill=False, color="lime", lw=0.5))
        if marks and i in marks:
            for mk in marks[i]:
                ax.plot(*mk, "r+", ms=9, mew=1.2)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xticks(np.arange(round(x0, 1), x1 + 1e-6, 0.1)); ax.set_yticks(np.arange(round(y0, 1), y1 + 1e-6, 0.1))
        ax.tick_params(labelsize=5)
        ax.set_title(f"M t={i/15:.2f}s q{q}", fontsize=7)
    fig.tight_layout(pad=0.3)
    path = os.path.join(OUT, f"{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    kind = sys.argv[1]
    a = [float(x) for x in sys.argv[2:5]]
    tag = sys.argv[5]
    extra = [float(v) for v in sys.argv[6].split(",")] if len(sys.argv) > 6 else None
    if kind == "cam":
        print(cam_tiles(a[0], a[1], a[2], tag, box=tuple(int(v) for v in extra) if extra else (40, 440, 60, 480)))
    elif kind == "pix":
        print(pix_tiles(a[0], a[1], a[2], tag, tuple(int(v) for v in extra)))
    elif kind == "rect":
        print(rect_tiles(a[0], a[1], a[2], tag))
    elif kind == "zoom":
        print(rect_tiles(a[0], a[1], a[2], tag, ncol=4, Mres=384, win=tuple(extra), step=0.05))
