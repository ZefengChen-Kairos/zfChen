"""Large rectified M-frame views (x_M right, y_M up; the pitcher of this clip is at -y_M) with a 0.05 D grid and
optional marks.   python rect_big.py tag t1,t2,... [x0,x1,y0,y1] [marks: t:x:y;t:x:y]"""
import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tiles as T  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from latte_imex import video_extract as VX  # noqa: E402

fr, frames = T.data()
tag = sys.argv[1]
times = [float(v) for v in sys.argv[2].split(",")]
win = tuple(float(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else (-0.5, 0.5, -0.5, 0.5)
marks = {}
if len(sys.argv) > 4:
    for m in sys.argv[4].split(";"):
        t, x, y = (float(v) for v in m.split(":"))
        marks.setdefault(int(round(t * 15)), []).append((x, y))
nc = min(3, len(times))
nr = int(math.ceil(len(times) / nc))
x0, x1, y0, y1 = win
s = 6.0 / max(x1 - x0, y1 - y0)
fig, axs = plt.subplots(nr, nc, figsize=(s * (x1 - x0) * nc, (s * (y1 - y0) + 0.3) * nr), dpi=80)
axs = np.atleast_1d(axs).ravel()
for ax, t in zip(axs, times):
    i = int(round(t * 15))
    p, q = T.ell_of(i)
    rgbM = VX.rectify(frames[i].astype(np.float64) / 255.0, p, 512, 0.0, False)[::-1]
    ax.imshow(np.clip(rgbM, 0, 1), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], interpolation="lanczos")
    for g in np.arange(-0.5, 0.501, 0.05):
        major = abs(round(g / 0.1) * 0.1 - g) < 1e-6
        ax.axhline(g, c="red" if abs(g) < 1e-6 else "yellow", lw=0.7 if major else 0.25, alpha=0.6)
        ax.axvline(g, c="magenta" if abs(g) < 1e-6 else "cyan", lw=0.7 if major else 0.25, alpha=0.6)
    ax.add_patch(plt.Circle((0, 0), 0.5, fill=False, color="lime", lw=0.6))
    for mk in marks.get(i, []):
        ax.plot(*mk, "r+", ms=14, mew=1.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks(np.arange(round(x0, 1), x1 + 1e-6, 0.1)); ax.set_yticks(np.arange(round(y0, 1), y1 + 1e-6, 0.1))
    ax.tick_params(labelsize=8)
    ax.set_title(f"M frame t={t:.2f}s (#{i}) ell q{q}", fontsize=10)
for ax in axs[len(times):]:
    ax.axis("off")
fig.tight_layout()
p = os.path.join(T.OUT, f"rb_{tag}.png")
fig.savefig(p)
print(p)
