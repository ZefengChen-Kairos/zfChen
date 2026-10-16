import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
Z = np.load(sys.argv[1]); out = sys.argv[2]
P, C, w = Z["P"].astype(float), Z["C"] / 255.0, Z["win"]
r = np.hypot(P[:, 0], P[:, 1]); phi = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360
keep = P[:, 2] > 0.004
P, C, w, r, phi = P[keep], C[keep], w[keep], r[keep], phi[keep]
# body radius profile: per 5 mm height, median radius of outer-shell points (exclude handle/spout via angular median)
hs = np.arange(0, 0.13, 0.005); prof = []
for h in hs:
    k = (P[:, 2] >= h) & (P[:, 2] < h + 0.005)
    if k.sum() < 200: prof.append(None); continue
    rb = np.zeros(36)
    for a in range(36):
        kk = k & (phi >= a * 10) & (phi < a * 10 + 10)
        rb[a] = np.percentile(r[kk], 80) if kk.sum() > 5 else np.nan
    prof.append(float(np.nanmedian(rb)))
# protrusions: angular bins where r exceeds body radius by > 8 mm at mid height
mid = (P[:, 2] > 0.04) & (P[:, 2] < 0.11)
body = np.nanmedian([p for p in prof if p])
prot = []
for a in range(0, 360, 5):
    kk = mid & (phi >= a) & (phi < a + 5)
    if kk.sum() > 20 and np.percentile(r[kk], 95) > body + 0.008: prot.append(a)
top = P[:, 2] > 0.11
spout = []
for a in range(0, 360, 5):
    kk = top & (phi >= a) & (phi < a + 5)
    if kk.sum() > 10 and np.percentile(r[kk], 95) > body + 0.006: spout.append(a)
res = dict(height_mm=round(float(np.percentile(P[:, 2], 99.7)) * 1e3, 1),
           body_radius_profile_mm={f"{h*1e3:.0f}": (None if p is None else round(p * 1e3, 1)) for h, p in zip(hs, prof)},
           protrusion_angles_mid_deg=prot, protrusion_angles_top_deg=spout)
print(json.dumps(res))
json.dump(res, open(out + ".json", "w"), indent=1)
fig = plt.figure(figsize=(20, 10))
tab = plt.get_cmap("tab10")
for j, az in enumerate([0, 90, 180, 270]):
    for row in range(2):
        ax = fig.add_subplot(2, 4, row * 4 + j + 1, projection="3d")
        ax.scatter(P[:, 0] * 1e3, P[:, 1] * 1e3, P[:, 2] * 1e3, s=0.5, c=(C if row == 0 else tab(w % 10)), depthshade=False)
        ax.view_init(10, az); ax.set_box_aspect((1, 1, 1)); ax.set_xlim(-80, 80); ax.set_ylim(-80, 80); ax.set_zlim(0, 160)
        ax.set_title(("colour" if row == 0 else "window: blue=0-1.8s orange=14.7-19.2s green=20.8-26.8s") + f"  view {az}°", fontsize=9)
fig.tight_layout(); fig.savefig(out + ".png", dpi=85)
