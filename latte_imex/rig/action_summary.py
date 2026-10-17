"""Extracted action in the cup frame: spout tip relative to the liquid disc (normalised by R), jet height, pitcher
tilt, disc radius; writes action.json and a summary figure with the final top view."""
import sys, json
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
take = sys.argv[1]
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
D = json.load(open(f"{B}/{take}/disc.json")); T = np.load(f"{B}/{take}/tip.npz")
t = np.array([d["t"] for d in D]); rel = np.array([d["tip_rel_norm"] for d in D]); R = np.array([d["disc_R"] for d in D]) * 1e3
jh = np.array([d["jet_h_mm"] for d in D]); zs = np.array([d["surf_z"] for d in D]) * 1e3
tilt = np.interp(t, T["t"], T["tilt"])
act = dict(take=take, t=t.round(4).tolist(), tip_rel_norm=rel.round(4).tolist(), jet_height_mm=jh.round(1).tolist(),
           pitcher_tilt_deg=tilt.round(2).tolist(), disc_R_mm=R.round(2).tolist(),
           frame="cup frame = liquid-disc centre, board x/y axes (horizontal), lengths / disc radius",
           notes="tag edge 18.2 mm assumed; tip = spout lip estimate (0,53,93) mm in the pitcher frame")
json.dump(act, open(f"{B}/{take}/action.json", "w"))
last = len(D) - 1
Z = np.load(f"{B}/{take}/top/{last:04d}.npz"); img = Z["img"][..., ::-1]; c0, PX, HALF = Z["c0"], float(Z["px"]), float(Z["half"])
cen = np.array(D[last]["disc_centre"]); Rl = D[last]["disc_R"]
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(1, 2, 1)
ext = np.array([c0[0] - HALF, c0[0] + HALF, c0[1] - HALF, c0[1] + HALF])
ax.imshow(img, origin="upper", extent=((ext[0] - cen[0]) / Rl, (ext[1] - cen[0]) / Rl, (ext[3] - cen[1]) / Rl, (ext[2] - cen[1]) / Rl))
sc = ax.scatter(rel[:, 0], rel[:, 1], c=t, cmap="plasma", s=12); plt.colorbar(sc, ax=ax, label="t [s]")
ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="lime")); ax.set_xlim(-1.4, 1.4); ax.set_ylim(1.4, -1.4); ax.set_aspect("equal")
ax.set_title(f"{take}: 壶嘴尖在液面上的轨迹（单位=液面半径），背景=最后一帧俯视图")
for j, (y, lab) in enumerate([(jh, "壶嘴尖离液面高度 [mm]"), (tilt, "壶倾角 [°]"), (R, "液面半径 [mm]")]):
    a = fig.add_subplot(3, 2, 2 * j + 2); a.plot(t, y, ".-", ms=3); a.set_ylabel(lab); a.grid(alpha=0.3)
a.set_xlabel("t [s]（从录制开始算）")
fig.tight_layout(); fig.savefig(f"{B}/{take}/action_summary.png", dpi=90)
print("jet h mm pcts", np.percentile(jh, [5, 25, 50, 75, 95]).round(1), "tilt", np.percentile(tilt, [5, 50, 95]).round(1))
