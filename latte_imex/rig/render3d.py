"""Crop an object around a board-frame seed (x, y) from fused static clouds and render coloured 3-D views."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, sx, sy, rad = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
srcs = sys.argv[5:]
fig = plt.figure(figsize=(5 * 4, 5 * len(srcs)))
for row, src in enumerate(srcs):
    Z = np.load(src); n, d = Z["n"], float(Z["d"])
    P = np.concatenate([Z[f"P{c}"] for c in range(3)]).astype(float)
    C = np.concatenate([Z[f"C{c}"] for c in range(3)]) / 255.0
    cam = np.concatenate([np.full(len(Z[f"P{c}"]), c) for c in range(3)])
    h = P @ n + d
    m = (np.hypot(P[:, 0] - sx, P[:, 1] - sy) < rad) & (h > 0.003) & (h < 0.25)
    X, Y, H, Cm, cm = P[m, 0], P[m, 1], h[m], C[m], cam[m]
    # drop sparse outliers: keep points with >= 8 neighbours within 5 mm (grid count)
    g = np.floor(np.stack([X, Y, H], 1) / 0.005).astype(int)
    _, inv, cnt = np.unique(g, axis=0, return_inverse=True, return_counts=True)
    k = cnt[inv.ravel()] >= 4
    X, Y, H, Cm, cm = X[k], Y[k], H[k], Cm[k], cm[k]
    print(src, "points", len(X), "height max mm", round(H.max() * 1e3, 1))
    for j, (el, az) in enumerate([(20, -60), (20, 30), (20, 120), (80, -90)]):
        ax = fig.add_subplot(len(srcs), 4, row * 4 + j + 1, projection="3d")
        ax.scatter((X - sx) * 1e3, (Y - sy) * 1e3, H * 1e3, s=0.6, c=Cm, depthshade=False)
        ax.view_init(el, az); ax.set_box_aspect((1, 1, 1))
        L = rad * 1e3
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(0, 2 * L)
        ax.set_xlabel("x mm"); ax.set_ylabel("y mm"); ax.set_zlabel("h mm")
        if j == 0: ax.set_title(src.split("/")[-1].replace("_cloud.npz", ""))
fig.tight_layout(); fig.savefig(out, dpi=90)
