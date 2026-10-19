"""Surface velocity of the real pours from the rectified top view (video_obs.py frames): dense optical flow
(Farneback) between consecutive frames, in solver units (cup radius 0.49) per second, x = board x, y = -board y (the
solver's axes), on the video_obs disc grid.  Trusted only where the image has texture (smoothed gradient magnitude of
the lightness above TEX_MIN) inside both frames' observation masks, and where the displacement is plausible (< MAX_PX).
Writes raw_data/rig/video_obs/<p>_flow.npz: t (mid times, sim clock), u, v (float16, (n-1, 64, 64), downsampled 2x),
ok (bool), plus the sheet <p>_flow.png (time-mean flow).
    python3 -m latte_imex.rig.video_flow [pattern ...]"""
import os
import sys

import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OBS = os.path.join(REPO, "raw_data", "rig", "video_obs")
PATTERNS = ("heart", "push_heart", "leaf", "swan", "tulip")
TEX_MIN = 6.0            # smoothed |grad L| [grey levels / px] for a trusted flow pixel
MAX_PX = 6.0             # [px / frame] plausible displacement on the 128 px grid


def flow(p):
    from .real_program import CUP_R
    Z = np.load(os.path.join(OBS, f"{p}.npz"))
    img, val, t, ex = Z["img"], Z["valid"], Z["t"], float(Z["extent"])
    n = img.shape[1]
    s = 2 * ex / (n - 1) * CUP_R                       # solver units per pixel
    U, V, OK, T = [], [], [], []
    for j in range(len(t) - 1):
        a = cv2.cvtColor(img[j], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(img[j + 1], cv2.COLOR_BGR2GRAY)
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 9, 3, 5, 1.1, 0)
        gx, gy = cv2.Sobel(a, cv2.CV_32F, 1, 0) / 8, cv2.Sobel(a, cv2.CV_32F, 0, 1) / 8
        tex = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2.0)
        dt = t[j + 1] - t[j]
        ok = val[j] & val[j + 1] & (tex > TEX_MIN) & (np.hypot(fl[..., 0], fl[..., 1]) < MAX_PX)
        u, v = fl[..., 0] * s / dt, -fl[..., 1] * s / dt
        # 2x downsample (mean over trusted pixels of each 2x2 block)
        w = ok.astype(np.float32).reshape(n // 2, 2, n // 2, 2).sum((1, 3))
        U.append((np.where(ok, u, 0).reshape(n // 2, 2, n // 2, 2).sum((1, 3)) / np.maximum(w, 1)).astype(np.float16))
        V.append((np.where(ok, v, 0).reshape(n // 2, 2, n // 2, 2).sum((1, 3)) / np.maximum(w, 1)).astype(np.float16))
        OK.append(w >= 2)
        T.append(0.5 * (t[j] + t[j + 1]))
    U, V, OK, T = np.array(U), np.array(V), np.array(OK), np.array(T)
    np.savez_compressed(os.path.join(OBS, f"{p}_flow.npz"), t=T, u=U, v=V, ok=OK, extent=ex)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cnt = OK.sum(0)
    mu = np.where(OK, U, 0).astype(float).sum(0) / np.maximum(cnt, 1)
    mv = np.where(OK, V, 0).astype(float).sum(0) / np.maximum(cnt, 1)
    m = n // 2
    g = np.linspace(-ex, ex, m)
    X, Y = np.meshgrid(g, -g)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.hypot(mu, mv) * (cnt > 0.2 * len(T)), extent=[-ex, ex, -ex, ex], cmap="viridis")
    ax.quiver(X[::4, ::4], Y[::4, ::4], mu[::4, ::4], mv[::4, ::4], color="w")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="r"))
    ax.set_title(f"{p}: mean surface velocity (trusted pixels), solver axes")
    fig.savefig(os.path.join(OBS, f"{p}_flow.png"), dpi=70)
    print(f"{p}: {len(T)} frame pairs, trusted {OK.mean():.2f} of the grid; mean speed {np.hypot(mu, mv)[cnt > 0.2 * len(T)].mean():.3f} /s")


def main(argv=None):
    for p in (argv or sys.argv[1:]) or PATTERNS:
        flow(p)


if __name__ == "__main__":
    main()
