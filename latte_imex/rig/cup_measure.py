"""Segment the cup near a seed (x,y) in the fused static cloud, fit rim circle and outer radius profile."""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out, sx, sy = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
Z = np.load(src)
n, d = Z["n"], float(Z["d"])
# table frame: e3 = n (up), e1/e2 orthonormal
e3 = n / np.linalg.norm(n); e1 = np.cross([0, 1, 0], e3); e1 /= np.linalg.norm(e1); e2 = np.cross(e3, e1)
o = -d * e3
pts, cam = [], []
for c in range(3):
    P = Z[f"P{c}"].astype(float) - o
    Q = np.stack([P @ e1, P @ e2, P @ e3], 1)
    pts.append(Q); cam.append(np.full(len(Q), c))
Q = np.concatenate(pts); cam = np.concatenate(cam)
s = np.array([sx, sy, 0.0]) - o; s = np.array([s @ e1, s @ e2])
r = np.hypot(Q[:, 0] - s[0], Q[:, 1] - s[1])
m = (r < float(sys.argv[5]) if len(sys.argv) > 5 else r < 0.09) & (Q[:, 2] > 0.004) & (Q[:, 2] < 0.2)
C = Q[m]; cm = cam[m]
# iterative circle fit (Kasa) on all object points by height bands -> centre; outer radius profile
def kasa(x, y):
    A = np.stack([x, y, np.ones_like(x)], 1); b = x * x + y * y
    a = np.linalg.lstsq(A, b, rcond=None)[0]; cx, cy = a[0] / 2, a[1] / 2
    return cx, cy, np.sqrt(a[2] + cx * cx + cy * cy)
zmax = np.percentile(C[:, 2], 99.5)
band = C[:, 2] > zmax - 0.006
cx, cy, R = kasa(C[band, 0], C[band, 1])
for _ in range(3):
    rr = np.hypot(C[:, 0] - cx, C[:, 1] - cy)
    keep = band & (np.abs(rr - R) < 0.008)
    cx, cy, R = kasa(C[keep, 0], C[keep, 1])
rr = np.hypot(C[:, 0] - cx, C[:, 1] - cy)
hs = np.arange(0.0, zmax + 0.005, 0.005)
prof = []
for h in hs:
    b = (C[:, 2] >= h) & (C[:, 2] < h + 0.005) & (rr < R + 0.02)
    prof.append(float(np.percentile(rr[b], 90)) if b.sum() > 30 else None)
res = dict(centre_table_xy_m=[float(cx), float(cy)], rim_height_mm=float(zmax * 1e3), rim_outer_radius_mm=float(R * 1e3),
           rim_fit_rms_mm=float(np.sqrt(np.mean((rr[keep] - R) ** 2)) * 1e3), npts=int(m.sum()),
           outer_radius_profile_mm={f"{h*1e3:.0f}": (None if p is None else round(p * 1e3, 1)) for h, p in zip(hs, prof)},
           frame="table frame: origin = board origin projected on the table plane, z up")
print(json.dumps(res, indent=1))
json.dump(res, open(out + ".json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
for c, col in zip(range(3), ["tab:red", "tab:green", "tab:blue"]):
    k = cm == c
    ax[0].scatter(C[k, 0] * 1e3, C[k, 1] * 1e3, s=0.3, c=col, alpha=0.4, label=f"cam{c}")
    ax[1].scatter(rr[k] * 1e3, C[k, 2] * 1e3, s=0.3, c=col, alpha=0.3)
t = np.linspace(0, 2 * np.pi, 200)
ax[0].plot((cx + R * np.cos(t)) * 1e3, (cy + R * np.sin(t)) * 1e3, "k-", lw=1)
ax[0].set_aspect("equal"); ax[0].set_title(f"top view: rim circle R = {R*1e3:.1f} mm"); ax[0].legend(markerscale=20)
ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("y [mm]")
ax[1].axvline(R * 1e3, color="k", lw=0.8); ax[1].set_xlim(0, (R + 0.03) * 1e3)
ax[1].set_xlabel("distance from cup axis [mm]"); ax[1].set_ylabel("height above table [mm]")
ax[1].set_title(f"side profile (all angles): rim height {zmax*1e3:.1f} mm")
fig.tight_layout(); fig.savefig(out + ".png", dpi=110)
