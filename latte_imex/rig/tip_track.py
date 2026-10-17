"""Spout tip in the pitcher frame (from the accumulated shape), outlier-cleaned pitcher poses, tip trajectory in the
board frame, and overlay of the projected tip on the images for checking."""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.ndimage import median_filter
from trimrec import Take

take = sys.argv[1]
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
tk = Take(f"{B}/ghdata/{take}"); ba = pickle.load(open(f"{B}/{take}/ba.pkl", "rb"))
M = np.load(f"{B}/{take}/pitcher_model.npz"); X = M["X"].astype(float); t = M["t"]; P = M["P"]; tilt = M["tilt"]; az = M["az"]
mj = json.load(open(f"{B}/{take}/pitcher_model.json"))
saz = np.radians(mj["downhill_az_when_tilt_gt30_deg"])
ztop = mj["wall_top_mm"] * 1e-3
ph = np.arctan2(X[:, 1], X[:, 0]); rad = X[:, 0] * np.cos(saz) + X[:, 1] * np.sin(saz)
cand = (np.abs(np.angle(np.exp(1j * (ph - saz)))) < np.radians(12)) & (X[:, 2] > ztop - 0.02)
rmax = np.percentile(rad[cand], 97)
near = cand & (rad > rmax - 0.004)
tip_p = np.array([rmax * np.cos(saz), rmax * np.sin(saz), np.percentile(X[near, 2], 90)])
if os.environ.get('TIP'): tip_p = np.array([float(v) for v in os.environ['TIP'].split(',')]) * 1e-3
# clean poses: drop samples whose tilt / azimuth deviate from a running median
mt = median_filter(tilt, 7, mode="nearest"); ma = median_filter(az, 7, mode="nearest")
good = (np.abs(tilt - mt) < 4) & (np.abs(az - ma) < 8)
tg, Pg = t[good], P[good]
sl = Slerp(tg, Rot.from_rotvec(Pg[:, :3]))
def pose(tt):
    tt = np.clip(tt, tg[0], tg[-1])
    return sl(tt).as_matrix(), np.stack([np.interp(tt, tg, Pg[:, 3 + j]) for j in range(3)], -1)
tip_b = np.einsum("nij,j->ni", Rot.from_rotvec(Pg[:, :3]).as_matrix(), tip_p) + Pg[:, 3:]
res = dict(take=take, tip_in_pitcher_mm=(tip_p * 1e3).round(1).tolist(), spout_radius_mm=round(rmax * 1e3, 1),
           n_samples=int(len(t)), n_good=int(good.sum()), tilt_good_deg=[round(float(np.min(tilt[good])), 1), round(float(np.median(tilt[good])), 1), round(float(np.max(tilt[good])), 1)])
print(json.dumps(res))
np.savez_compressed(f"{B}/{take}/tip.npz", t=tg, tip_board=tip_b, tilt=tilt[good], az=az[good], poses=Pg, tip_p=tip_p)
json.dump(res, open(f"{B}/{take}/tip.json", "w"), indent=1)
# overlay: every ~0.8 s, three cameras, projected tip (red) + pitcher axis (yellow)
rows = []
for tt in np.arange(tg[0] + 0.1, tg[-1], 0.8):
    Rm, p = pose(tt); tipb = Rm @ tip_p + p; axb = [Rm @ np.array([0, 0, z]) + p for z in (-0.05, 0.1)]
    ims = []
    for c in range(tk.ncam):
        j = int(np.argmin(np.abs(tk.ts[c] - tk.t0 - tt))); im = tk.color(c, tk.steps[c][j]).copy()
        W2C = np.linalg.inv(tk.c2w[c]); K = tk.K[c]
        def pr(Xb):
            Xc = W2C[:3, :3] @ Xb + W2C[:3, 3]; return tuple(int(v) for v in (Xc[:2] / Xc[2] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]))
        cv2.line(im, pr(axb[0]), pr(axb[1]), (0, 255, 255), 2)
        cv2.circle(im, pr(tipb), 7, (0, 0, 255), 2); cv2.circle(im, pr(tipb), 2, (0, 0, 255), -1)
        cv2.putText(im, f"cam{c} t={tt:.1f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        ims.append(im)
    rows.append(np.hstack(ims))
cv2.imwrite(f"{B}/{take}/tip_overlay.jpg", cv2.resize(np.vstack(rows), None, fx=0.45, fy=0.45), [cv2.IMWRITE_JPEG_QUALITY, 80])
