"""Liquid surface in the cup per frame from fused depth: points below the spout tip, in a 70 mm column, not black
(cup), not pitcher; height = mode of the vertical histogram; centre / radius = circle fit of the surface patch."""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from trimrec import Take, cloud
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

take = sys.argv[1]
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
tk = Take(f"{B}/ghdata/{take}"); T = np.load(f"{B}/{take}/tip.npz")
UP = np.array([0, 0, -1.0])
def kasa(x, y):
    A = np.stack([x, y, np.ones_like(x)], 1); b = x * x + y * y
    a = np.linalg.lstsq(A, b, rcond=None)[0]; cx, cy = a[0] / 2, a[1] / 2
    return cx, cy, np.sqrt(max(a[2] + cx * cx + cy * cy, 1e-9))
out = []
for i in range(len(tk.T)):
    tt = tk.T[i]
    if tt < T["t"][0] or tt > T["t"][-1]: continue
    tip = np.array([np.interp(tt, T["t"], T["tip_board"][:, j]) for j in range(3)])
    pts = []
    for c, k, dt in tk.sync(i):
        P, C, _ = cloud(tk, c, k, 1)
        h = (P - tip) @ UP                                  # height relative to the tip (negative = below)
        d = np.hypot(P[:, 0] - tip[0], P[:, 1] - tip[1])
        m = (h < -0.01) & (h > -0.15) & (d < 0.07)
        if m.sum() == 0: continue
        hsv = cv2.cvtColor(C[m][None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
        liquid = hsv[:, 2] > 90                            # not the black cup
        pts.append(np.c_[P[m][liquid], h[m][liquid], np.full(liquid.sum(), c)])
    Q = np.concatenate(pts) if pts else np.zeros((0, 5))
    if len(Q) < 200: continue
    hist, e = np.histogram(Q[:, 3], bins=np.arange(-0.15, -0.01, 0.002))
    j = int(np.argmax(hist)); hs = 0.5 * (e[j] + e[j + 1])
    band = np.abs(Q[:, 3] - hs) < 0.004
    if band.sum() < 100: continue
    cx, cy, R = kasa(Q[band, 0], Q[band, 1])
    # refine: points of the band within R + 5 mm
    out.append(dict(t=tt, tip=tip.tolist(), jet_height_m=float(-hs), surf_centre=[cx, cy], surf_R=float(R), n=int(band.sum()),
                    cams=np.bincount(Q[band, 4].astype(int), minlength=3).tolist(),
                    surf_z_board=float(np.median(Q[band, 2]))))
json.dump(out, open(f"{B}/{take}/cup_track.json", "w"))
t = np.array([o["t"] for o in out]); jh = np.array([o["jet_height_m"] for o in out]) * 1e3
cxy = np.array([o["surf_centre"] for o in out]); tipxy = np.array([o["tip"][:2] for o in out]); R = np.array([o["surf_R"] for o in out]) * 1e3
zs = np.array([o["surf_z_board"] for o in out]) * 1e3
rel = (tipxy - cxy) * 1e3
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
ax[0, 0].plot(t, jh, ".-"); ax[0, 0].set_ylabel("spout tip above liquid surface [mm]")
ax[0, 1].plot(t, -zs, ".-"); ax[0, 1].set_ylabel("liquid surface height above board [mm]")
ax[1, 0].plot(t, R, ".-"); ax[1, 0].set_ylabel("fitted surface-patch radius [mm]")
ax[1, 1].plot(rel[:, 0], rel[:, 1], ".-"); ax[1, 1].set_aspect("equal"); ax[1, 1].set_title("tip (xy) relative to surface-patch centre [mm]")
for a in ax.ravel()[:3]: a.set_xlabel("t [s]")
fig.tight_layout(); fig.savefig(f"{B}/{take}/cup_track.png", dpi=80)
print(len(out), "frames; jet height mm median", np.median(jh).round(1), "range", np.percentile(jh, [5, 95]).round(1), "R med", np.median(R).round(1))
