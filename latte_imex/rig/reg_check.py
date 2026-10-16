"""Horizontal registration check: rasterise each camera's table points (brightness of the printed mat) onto the table
plane and phase-correlate the images camera against camera."""
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z = np.load(sys.argv[1]); out = sys.argv[2]
n, d = Z["n"], float(Z["d"]); px = 0.002; L = 0.5; N = int(2 * L / px)
ims = []
for c in range(3):
    P = Z[f"P{c}"].astype(float); col = Z[f"C{c}"].astype(float).mean(1)
    h = P @ n + d
    m = (np.abs(h) < 0.004) & (np.abs(P[:, 0]) < L) & (np.abs(P[:, 1]) < L)
    i = ((P[m, 0] + L) / px).astype(int); j = ((P[m, 1] + L) / px).astype(int)
    S = np.zeros((N, N)); W = np.zeros((N, N))
    np.add.at(S, (j, i), col[m]); np.add.at(W, (j, i), 1)
    im = np.where(W > 0, S / np.maximum(W, 1), np.nan)
    ims.append(im)
fig, ax = plt.subplots(1, 4, figsize=(22, 5.5))
for c in range(3):
    ax[c].imshow(ims[c], cmap="gray", origin="lower", extent=(-L, L, -L, L)); ax[c].set_title(f"cam{c} table texture")
for a, b in [(0, 1), (0, 2), (1, 2)]:
    both = np.isfinite(ims[a]) & np.isfinite(ims[b])
    ys, xs = np.where(both)
    if len(ys) < 1000: print(a, b, "little overlap"); continue
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    A = np.nan_to_num(ims[a][y0:y1, x0:x1] - np.nanmean(ims[a][both])) * both[y0:y1, x0:x1]
    B = np.nan_to_num(ims[b][y0:y1, x0:x1] - np.nanmean(ims[b][both])) * both[y0:y1, x0:x1]
    win = cv2.createHanningWindow(A.shape[::-1], cv2.CV_64F)
    (dx, dy), resp = cv2.phaseCorrelate(A, B, win)
    print(f"cam{a} vs cam{b}: overlap {both.sum()*px*px*1e4:.0f} cm^2, shift dx {dx*px*1e3:+.1f} mm, dy {dy*px*1e3:+.1f} mm, response {resp:.2f}")
rgb = np.zeros((N, N, 3))
for c in range(3):
    v = ims[c]; rgb[..., c] = np.nan_to_num((v - np.nanpercentile(v, 2)) / (np.nanpercentile(v, 98) - np.nanpercentile(v, 2)))
ax[3].imshow(np.clip(rgb, 0, 1), origin="lower", extent=(-L, L, -L, L)); ax[3].set_title("overlay R=cam0 G=cam1 B=cam2")
fig.tight_layout(); fig.savefig(out, dpi=100)
