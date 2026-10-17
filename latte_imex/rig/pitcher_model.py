"""Accumulate depth points in the pitcher frame (poses from the tag BA) -> pitcher shape incl. rim and spout;
pitcher tilt / down-azimuth over time."""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot, Slerp
from trimrec import Take, cloud
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

take = sys.argv[1]
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
tk = Take(f"{B}/ghdata/{take}"); ba = pickle.load(open(f"{B}/{take}/ba.pkl", "rb"))
o = np.argsort(ba["t"]); t = ba["t"][o]; P = ba["poses"][o]
keep = np.r_[True, np.diff(t) > 1e-4]; t, P = t[keep], P[keep]
R = Rot.from_rotvec(P[:, :3]); sl = Slerp(t, R)
def pose(tt):
    tt = np.clip(tt, t[0], t[-1]); Rm = sl(tt).as_matrix(); p = np.array([np.interp(tt, t, P[:, 3 + j]) for j in range(3)])
    return Rm, p
UP = np.array([0, 0, -1.0])   # board z points into the table
# tilt of the pitcher axis from vertical, and azimuth (pitcher frame) of the downhill direction
tilt, az = [], []
for k in range(len(t)):
    Rm = R[k].as_matrix(); a = Rm[:, 2]
    tilt.append(np.degrees(np.arccos(np.clip(a @ UP, -1, 1))))
    g = Rm.T @ (-UP); az.append(np.degrees(np.arctan2(g[1], g[0])))
tilt, az = np.array(tilt), np.array(az)
# accumulate points
vox = {}; H = 0.002
for c in range(tk.ncam):
    for k, tt in zip(tk.steps[c][::2], (tk.ts[c] - tk.t0)[::2]):
        if tt < t[0] - 0.05 or tt > t[-1] + 0.05: continue
        Pw, C, _ = cloud(tk, c, k, 2)
        Rm, p = pose(tt)
        X = (Pw - p) @ Rm                                   # board -> pitcher frame
        rr = np.hypot(X[:, 0], X[:, 1])
        m = (rr < 0.09) & (X[:, 2] > -0.10) & (X[:, 2] < 0.10)
        hsv = cv2.cvtColor(C[m][None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
        white = (hsv[:, 1] < 70) & (hsv[:, 2] > 110)
        Xm = X[m][white]
        for key in map(tuple, np.floor(Xm / H).astype(int)):
            vox[key] = vox.get(key, 0) + 1
K = np.array(list(vox.keys())); N = np.array(list(vox.values()))
sel = N >= 4
X = (K[sel] + 0.5) * H; Nn = N[sel]
rr = np.hypot(X[:, 0], X[:, 1]); ph = np.degrees(np.arctan2(X[:, 1], X[:, 0]))
wall = np.abs(rr - ba["r"]) < 0.012
ztop = np.percentile(X[wall, 2], 99.5); zbot = np.percentile(X[wall, 2], 0.5)
top = X[:, 2] > ztop - 0.012
prot = top & (rr > ba["r"] + 0.006)
spout_az = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(ph[prot]))), np.mean(np.cos(np.radians(ph[prot])))))) if prot.sum() > 5 else None
pour = tilt > 30
res = dict(take=take, r_mm=ba["r"] * 1e3, wall_top_mm=ztop * 1e3, wall_bottom_mm=zbot * 1e3, height_mm=(ztop - zbot) * 1e3,
           spout_az_from_shape_deg=spout_az, n_protrusion_vox=int(prot.sum()),
           downhill_az_when_tilt_gt30_deg=float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(az[pour]))), np.mean(np.cos(np.radians(az[pour])))))) if pour.any() else None,
           tilt_deg_range=[float(tilt.min()), float(np.percentile(tilt, 50)), float(tilt.max())])
print(json.dumps(res, indent=1))
json.dump(res, open(f"{B}/{take}/pitcher_model.json", "w"), indent=1)
np.savez_compressed(f"{B}/{take}/pitcher_model.npz", X=X.astype(np.float32), N=Nn, t=t, tilt=tilt, az=az, P=P)
fig = plt.figure(figsize=(18, 9))
for j, (el, a) in enumerate([(10, 0), (10, 90), (10, 180), (80, -90)]):
    ax = fig.add_subplot(2, 4, j + 1, projection="3d")
    ax.scatter(X[:, 0] * 1e3, X[:, 1] * 1e3, X[:, 2] * 1e3, s=0.5, c=np.log(Nn), cmap="viridis", depthshade=False)
    ax.view_init(el, a); ax.set_xlim(-90, 90); ax.set_ylim(-90, 90); ax.set_zlim(-100, 80); ax.set_box_aspect((1, 1, 1))
    for tg, (phi, z, _) in ba["layout"].items():
        ax.scatter([ba["r"] * 1e3 * np.cos(phi)], [ba["r"] * 1e3 * np.sin(phi)], [z * 1e3], c="r", s=40); ax.text(ba["r"] * 1e3 * np.cos(phi), ba["r"] * 1e3 * np.sin(phi), z * 1e3, f" tag{tg}")
ax = fig.add_subplot(2, 2, 3); ax.plot(t, tilt, ".-"); ax.set_xlabel("t [s]"); ax.set_ylabel("pitcher tilt from vertical [deg]")
ax = fig.add_subplot(2, 2, 4); ax.plot(t, az, "."); ax.set_xlabel("t [s]"); ax.set_ylabel("downhill azimuth in pitcher frame [deg]")
fig.tight_layout(); fig.savefig(f"{B}/{take}/pitcher_model.png", dpi=80)
