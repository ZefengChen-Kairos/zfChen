"""Crops showing the whole stream (spout lip to foot) with a 10 px grid, for heights (image length of the stream)
and flow (stream width).   python stream_zoom.py tag t:x0:x1:y0:y1 [...]"""
import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tiles as T  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_, frames = T.data()
tag = sys.argv[1]
items = [a.split(":") for a in sys.argv[2:]]
nc = 4
nr = int(math.ceil(len(items) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 4.4 * nr), dpi=90)
axs = np.atleast_2d(axs)
for ax, it in zip(axs.flat, items):
    t = float(it[0]); x0, x1, y0, y1 = (int(v) for v in it[1:])
    i = int(round(t * 15))
    ax.imshow(frames[i][y0:y1, x0:x1], extent=[x0 - .5, x1 - .5, y1 - .5, y0 - .5], interpolation="nearest")
    ax.set_xticks(np.arange((x0 // 20) * 20, x1 + 1, 20)); ax.set_yticks(np.arange((y0 // 20) * 20, y1 + 1, 20))
    ax.set_xticks(np.arange((x0 // 10) * 10, x1 + 1, 10), minor=True); ax.set_yticks(np.arange((y0 // 10) * 10, y1 + 1, 10), minor=True)
    ax.grid(True, which="both", lw=0.3, c="cyan", alpha=0.7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_title(f"t={t:.2f} (#{i})", fontsize=9)
for ax in axs.flat[len(items):]:
    ax.axis("off")
fig.tight_layout()
p = os.path.join(T.OUT, f"sz_{tag}.png")
fig.savefig(p)
print(p)
