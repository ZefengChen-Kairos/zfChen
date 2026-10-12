"""(copied from the session scratchpad; run from this directory with PYTHONPATH=/home/user/zfChen)
Tile maker for reading the layered_heart clip: camera crops with an M-frame grid projected on the liquid surface,
and rectified M-frame tiles.  Usage: python tiles.py cam t0 t1 fps name [--ell override.json]"""
import sys, os, math, json
import numpy as np
sys.path.insert(0, "/home/user/zfChen")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from latte_imex import video_extract as VX

NAME = "layered_heart"
OUT = "/home/user/zfChen/latte_imex/video_targets/layered_heart/tiles"
_cache = {}


def data():
    if not _cache:
        fr = VX.load_frames(NAME)
        frames, fps = VX.read_video(str(fr["clip"]))
        _cache.update(fr=fr, frames=frames, fps=fps)
    return _cache["fr"], _cache["frames"]


HAND = {0.0: (220, 318), 1.2: (220, 318), 1.8: (218, 320), 2.4: (215, 322), 3.0: (213, 325), 3.6: (212, 327),
        4.0: (212, 328), 4.4: (211.2, 329.0)}
HAND_AB = (112.5, 86.0, -3.0)
OVR = {}   # frame index -> ellipse override


def hand_ell(t):
    ts = sorted(HAND)
    cx = np.interp(t, ts, [HAND[k][0] for k in ts]); cy = np.interp(t, ts, [HAND[k][1] for k in ts])
    return (float(cx), float(cy)) + HAND_AB


def ell_of(i):
    fr, _ = data()
    if i in OVR:
        return OVR[i], 9
    if i / 15.0 < 4.4 - 1e-9:
        return hand_ell(i / 15.0), 8
    # nearest override in time if given as ranges
    return tuple(fr["track_ell"][i]), int(fr["track_quality"][i])


def m_to_img(p, xm, ym, rot, flip=False):
    X, Y = np.asarray(xm, float), -np.asarray(ym, float)
    if flip:
        X = -X
    th = math.radians(rot)
    Xr = math.cos(th) * X - math.sin(th) * Y
    Yr = math.sin(th) * X + math.cos(th) * Y
    u, v = 2 * Xr, -2 * Yr
    ex, ey = VX.basis(p)
    cx, cy, a, b = p[:4]
    return cx + a * u * ex[0] + b * v * ey[0], cy + a * u * ex[1] + b * v * ey[1]


def img_to_m(p, px, py, rot):
    X, Y = VX.image_to_cup(p, px, py, rot, False)
    return X, -Y


def cam_tiles(t0, t1, fps, tag, ncol=5, box=None, grid=True, scale=1.0):
    fr, frames = data()
    rot = float(fr["rot"])
    idx = sorted(set(int(round(t * 15)) for t in np.arange(t0, t1 + 1e-6, 1.0 / fps)))
    idx = [i for i in idx if i < len(frames)]
    n = len(idx)
    nrow = int(math.ceil(n / ncol))
    if box is None:
        box = (0, 480, 100, 560)  # x0, x1, y0, y1
    x0, x1, y0, y1 = box
    w = (x1 - x0) / 100 * 1.25 * scale
    h = (y1 - y0) / 100 * 1.25 * scale
    fig, axs = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), dpi=80)
    axs = np.atleast_2d(axs)
    for k, ax in enumerate(axs.flat):
        ax.axis("off")
        if k >= n:
            continue
        i = idx[k]
        p, q = ell_of(i)
        ax.imshow(frames[i][y0:y1, x0:x1], extent=[x0 - .5, x1 - .5, y1 - .5, y0 - .5], interpolation="lanczos")
        tt = np.linspace(0, 2 * np.pi, 200)
        ex, ey = VX.basis(p)
        ax.plot(p[0] + p[2] * np.cos(tt) * ex[0] + p[3] * np.sin(tt) * ey[0],
                p[1] + p[2] * np.cos(tt) * ex[1] + p[3] * np.sin(tt) * ey[1], "-", c="lime", lw=0.6)
        if grid:
            for g in np.arange(-0.4, 0.41, 0.1):
                ys = np.linspace(-0.5, 0.5, 50)
                ok = np.hypot(g, ys) <= 0.5
                gx, gy = m_to_img(p, np.full(ok.sum(), g), ys[ok], rot)
                ax.plot(gx, gy, "-", c="cyan" if abs(g) > 1e-6 else "magenta", lw=0.35, alpha=0.8)
                gx, gy = m_to_img(p, ys[ok], np.full(ok.sum(), g), rot)
                ax.plot(gx, gy, "-", c="yellow" if abs(g) > 1e-6 else "red", lw=0.35, alpha=0.8)
            # label +y_M and +x_M directions
            for (xm, ym, s) in [(0.0, 0.45, "+y"), (0.45, 0.0, "+x")]:
                gx, gy = m_to_img(p, xm, ym, rot)
                ax.text(gx, gy, s, color="white", fontsize=6, ha="center", va="center")
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_title(f"t={i/15:.2f}s  q{q}", fontsize=7)
    fig.suptitle(f"{NAME} camera {t0:.2f}-{t1:.2f}s; grid = M frame on the fitted surface (0.1 D; red y_M=0, magenta x_M=0)",
                 fontsize=7)
    fig.tight_layout(pad=0.2)
    path = os.path.join(OUT, f"{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def pix_tiles(t0, t1, fps, tag, box, ncol=4, step=10, sc=3.0, LS=5, times=None):
    """Native pixel crops with pixel ticks, for picking the stream foot in px."""
    fr, frames = data()
    idx = sorted(set(int(round(t * 15)) for t in (times if times is not None else np.arange(t0, t1 + 1e-6, 1.0 / fps))))
    idx = [i for i in idx if i < len(frames)]
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
        ax.set_xticks(np.arange(x0, x1 + 1, step)); ax.set_yticks(np.arange(y0, y1 + 1, step))
        ax.tick_params(labelsize=LS, length=2)
        ax.grid(True, lw=0.3, alpha=0.5, c="cyan")
        ax.set_title(f"t={i/15:.2f}s (#{i})", fontsize=8)
    fig.tight_layout(pad=0.3)
    path = os.path.join(OUT, f"{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def rect_tiles(t0, t1, fps, tag, ncol=5, Mres=192):
    fr, frames = data()
    rot = float(fr["rot"])
    idx = sorted(set(int(round(t * 15)) for t in np.arange(t0, t1 + 1e-6, 1.0 / fps)))
    idx = [i for i in idx if i < len(frames)]
    n = len(idx)
    nrow = int(math.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.7 * nrow), dpi=90)
    axs = np.atleast_2d(axs)
    for k, ax in enumerate(axs.flat):
        if k >= n:
            ax.axis("off"); continue
        i = idx[k]
        p, q = ell_of(i)
        rgbT = VX.rectify(frames[i].astype(np.float64) / 255.0, p, Mres, rot, False)
        rgbM = rgbT[::-1]          # M frame: row index grows with y_M
        ax.imshow(np.clip(rgbM, 0, 1), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5])
        for g in np.arange(-0.4, 0.41, 0.1):
            ax.axhline(g, c="yellow" if abs(g) > 1e-6 else "red", lw=0.3, alpha=0.6)
            ax.axvline(g, c="cyan" if abs(g) > 1e-6 else "magenta", lw=0.3, alpha=0.6)
        ax.add_patch(plt.Circle((0, 0), 0.5, fill=False, color="lime", lw=0.5))
        ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4]); ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        ax.tick_params(labelsize=5)
        ax.set_title(f"M frame t={i/15:.2f}s q{q}", fontsize=7)
    fig.tight_layout(pad=0.3)
    path = os.path.join(OUT, f"{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    kind = sys.argv[1]
    a = [float(x) for x in sys.argv[2:5]]
    tag = sys.argv[5]
    if kind == "cam":
        print(cam_tiles(a[0], a[1], a[2], tag))
    elif kind == "rect":
        print(rect_tiles(a[0], a[1], a[2], tag))
    elif kind == "pix":
        box = tuple(int(v) for v in sys.argv[6].split(","))
        print(pix_tiles(a[0], a[1], a[2], tag, box))


def rect_zoom(times, tag, ncol=4, Mres=384, win=(-0.35, 0.35, -0.2, 0.45), step=0.05, marks=None):
    fr, frames = data()
    rot = float(fr["rot"])
    idx = [int(round(t * 15)) for t in times]
    n = len(idx)
    nrow = int(math.ceil(n / ncol))
    x0, x1, y0, y1 = win
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol * (x1 - x0) / 0.7, 4.2 * nrow * (y1 - y0) / 0.7 + 0.3 * nrow), dpi=80)
    axs = np.atleast_2d(axs)
    for k, ax in enumerate(axs.flat):
        if k >= n:
            ax.axis("off"); continue
        i = idx[k]
        p, q = ell_of(i)
        rgbM = VX.rectify(frames[i].astype(np.float64) / 255.0, p, Mres, rot, False)[::-1]
        ax.imshow(np.clip(rgbM, 0, 1), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], interpolation="lanczos")
        for g in np.arange(-0.5, 0.501, step):
            major = abs(round(g / 0.1) * 0.1 - g) < 1e-6
            ax.axhline(g, c="red" if abs(g) < 1e-6 else "yellow", lw=0.6 if major else 0.25, alpha=0.7)
            ax.axvline(g, c="magenta" if abs(g) < 1e-6 else "cyan", lw=0.6 if major else 0.25, alpha=0.7)
        if marks and i / 15.0 in marks:
            for mk in marks[i / 15.0]:
                ax.plot(*mk, "r+", ms=10)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xticks(np.arange(round(x0, 1), x1 + 1e-6, 0.1)); ax.set_yticks(np.arange(round(y0, 1), y1 + 1e-6, 0.1))
        ax.tick_params(labelsize=7)
        ax.set_title(f"M t={i/15:.2f}s q{q}", fontsize=9)
    fig.tight_layout(pad=0.3)
    path = os.path.join(OUT, f"{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path
