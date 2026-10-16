"""Per-camera activity over time (frame-difference energy on downsampled colour, and fraction of depth pixels above
the table within R of the cup), plus a contact sheet, to locate the pour."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rs_zip import Rec

path, out = sys.argv[1], sys.argv[2]
r = Rec(path)
fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
res = {"t0": r.t0, "cams": {}}
for c in range(r.ncam):
    t = r.ts[c] - r.t0
    prev, E = None, []
    for k in r.steps[c]:
        g = cv2.cvtColor(cv2.resize(r.color(c, k), (212, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(float)
        E.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))
        prev = g
    E = np.array(E)
    ax[0].plot(t, E, label=f"cam{c}")
    dt = np.diff(t)
    ax[1].plot(t[1:], 1 / dt, ".", ms=2, label=f"cam{c}")
    res["cams"][c] = dict(n=len(t), dur=float(t[-1]), fps_eff=float(len(t) / t[-1]), max_gap_s=float(dt.max()),
                          t=t.round(4).tolist(), energy=E.round(3).tolist())
ax[0].set_ylabel("mean |Δ grey|"); ax[0].legend(); ax[1].set_ylabel("instant fps"); ax[1].set_xlabel("t [s]")
fig.tight_layout(); fig.savefig(out + "_activity.png", dpi=100)
json.dump(res, open(out + "_activity.json", "w"))
# contact sheet: every ~1 s, all cams
T = max(res["cams"][c]["dur"] for c in res["cams"])
times = np.arange(0, T, max(1.0, T / 16))
rows = []
for tt in times:
    row = []
    for c in range(r.ncam):
        i = int(np.argmin(np.abs(r.ts[c] - r.t0 - tt)))
        im = cv2.resize(r.color(c, r.steps[c][i]), (283, 160))
        cv2.putText(im, f"c{c} t={tt:.1f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        row.append(im)
    rows.append(np.hstack(row))
cv2.imwrite(out + "_sheet.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
print({c: {k: v for k, v in res["cams"][c].items() if k in ("n", "dur", "fps_eff", "max_gap_s")} for c in res["cams"]})
