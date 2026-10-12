"""High-zoom crops around the stream foot (5 px grid, 10 px labels) for reading the foot position.
   python foot_zoom.py tag t:cx:cy [t:cx:cy ...]     (crop 80 x 80 px centred at (cx, cy))"""
import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tiles as T  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_, frames = T.data()
tag = sys.argv[1]
items = [tuple(float(v) for v in a.split(":")) for a in sys.argv[2:]]
nc = 4
nr = int(math.ceil(len(items) / nc))
H = 40
fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.7 * nr), dpi=90)
axs = np.atleast_2d(axs)
for ax, (t, cx, cy) in zip(axs.flat, items):
    i = int(round(t * 15))
    x0, x1, y0, y1 = int(cx - H), int(cx + H), int(cy - H), int(cy + H)
    ax.imshow(frames[i][y0:y1, x0:x1], extent=[x0 - .5, x1 - .5, y1 - .5, y0 - .5], interpolation="nearest")
    ax.set_xticks(np.arange((x0 // 10) * 10, x1 + 1, 10)); ax.set_yticks(np.arange((y0 // 10) * 10, y1 + 1, 10))
    ax.set_xticks(np.arange((x0 // 5) * 5, x1 + 1, 5), minor=True); ax.set_yticks(np.arange((y0 // 5) * 5, y1 + 1, 5), minor=True)
    ax.grid(True, which="both", lw=0.3, c="cyan", alpha=0.7)
    ax.tick_params(labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_title(f"t={t:.2f} (#{i})", fontsize=9)
for ax in axs.flat[len(items):]:
    ax.axis("off")
fig.tight_layout()
p = os.path.join(T.OUT, f"fz_{tag}.png")
fig.savefig(p)
print(p)
