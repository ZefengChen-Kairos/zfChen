"""Static scene: per-camera median depth over n frames -> board-frame point clouds; table plane; overlap check; figures."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rs_zip import Rec, backproject

path, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40
T1, T2 = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (-1e9, 1e9)
r = Rec(path)
clouds, cols = [], []
for c in range(r.ncam):
    sel = np.where((r.ts[c] - r.t0 >= T1) & (r.ts[c] - r.t0 <= T2))[0]
    ks = r.steps[c][sel[np.linspace(0, len(sel) - 1, min(n, len(sel))).astype(int)]]
    D = np.stack([r.depth(c, k) for k in ks]).astype(float)
    D[D == 0] = np.nan
    med = np.nanmedian(D, 0)
    spread = np.nanstd(D, 0)
    valid = np.isfinite(med) & (np.sum(np.isfinite(D), 0) > n * 0.6)
    dm = np.where(valid, med, 0) * 1e-3
    P, ok = backproject(dm, r.K[c], r.c2w[c])
    img = r.color(c, ks[len(ks) // 2])[..., ::-1]
    clouds.append(P); cols.append(img[ok] / 255.0)
    print(f"cam{c}: valid {valid.mean():.2f}, temporal std median {np.nanmedian(spread):.2f} mm")
# board frame: z into board -> height above table plane = -z
allP = np.concatenate(clouds)
near = np.linalg.norm(allP[:, :2], axis=1) < 0.6
# RANSAC plane on near points
rng = np.random.default_rng(0)
Q = allP[near]
best = (0, None)
for _ in range(400):
    s = Q[rng.choice(len(Q), 3, replace=False)]
    nrm = np.cross(s[1] - s[0], s[2] - s[0]); L = np.linalg.norm(nrm)
    if L < 1e-9: continue
    nrm /= L; dd = -nrm @ s[0]
    inl = np.abs(Q @ nrm + dd) < 0.004
    if inl.sum() > best[0]: best = (inl.sum(), (nrm, dd))
nrm, dd = best[1]
if nrm[2] > 0: nrm, dd = -nrm, -dd
print("table plane normal (board frame)", np.round(nrm, 4), "offset m", round(dd, 4), "tilt vs board deg",
      round(np.degrees(np.arccos(abs(nrm[2]))), 2))
res = {"plane_normal": nrm.tolist(), "plane_d": dd, "cams": []}
for c, P in enumerate(clouds):
    h = P @ nrm + dd
    on = (np.abs(h) < 0.01) & (np.linalg.norm(P[:, :2], axis=1) < 0.6)
    res["cams"].append(dict(table_rms_mm=float(np.sqrt(np.mean(h[on] ** 2)) * 1e3), table_bias_mm=float(np.mean(h[on]) * 1e3)))
    print(f"cam{c}: table points {on.sum()}, bias {np.mean(h[on])*1e3:.2f} mm, rms {np.sqrt(np.mean(h[on]**2))*1e3:.2f} mm")
np.savez_compressed(out + "_cloud.npz", **{f"P{c}": clouds[c].astype(np.float32) for c in range(r.ncam)},
                    **{f"C{c}": (cols[c] * 255).astype(np.uint8) for c in range(r.ncam)}, n=nrm, d=dd)
# figure: top view coloured by camera and by height; RGB top view
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
cc = ["tab:red", "tab:green", "tab:blue"]
for c, P in enumerate(clouds):
    h = P @ nrm + dd
    m = (np.linalg.norm(P[:, :2], axis=1) < 0.5) & (h > -0.02) & (h < 0.25)
    ax[0].scatter(P[m, 0], P[m, 1], s=0.05, c=cc[c], alpha=0.3, label=f"cam{c}")
    ax[2].scatter(P[m, 0], P[m, 1], s=0.05, c=cols[c][m])
    hm = m & (h > 0.01)
    sc = ax[1].scatter(P[hm, 0], P[hm, 1], s=0.2, c=h[hm] * 1e3, cmap="viridis", vmin=0, vmax=150)
plt.colorbar(sc, ax=ax[1], label="height above table [mm]")
for a, t in zip(ax, ["by camera", "objects (height > 10 mm)", "colour"]):
    a.set_aspect("equal"); a.set_title(t); a.set_xlabel("x board [m]"); a.set_ylabel("y board [m]")
    for c in range(r.ncam):
        a.plot(*r.c2w[c][:2, 3], "k^"); a.text(*r.c2w[c][:2, 3], f" cam{c}")
ax[0].legend(markerscale=40)
fig.tight_layout(); fig.savefig(out + "_top.png", dpi=110)
json.dump(res, open(out + "_static.json", "w"), indent=1)
