import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import lombscargle
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
P = [("heart", "爱心"), ("push_heart", "推推乐"), ("leaf", "树叶"), ("swan", "天鹅"), ("tulip", "郁金香")]
fig = plt.figure(figsize=(18, 4.2 * len(P)))
summ = {}
for r, (p, nm) in enumerate(P):
    A = json.load(open(f"{B}/{p}/action.json")); D = json.load(open(f"{B}/{p}/disc.json"))
    t = np.array(A["t"]); rel = np.array(A["tip_rel_norm"]); jh = np.array(A["jet_height_mm"]); tl = np.array(A["pitcher_tilt_deg"])
    last = len(D) - 1; Z = np.load(f"{B}/{p}/top/{last:04d}.npz"); img = Z["img"][..., ::-1]; c0, PX, HALF = Z["c0"], float(Z["px"]), float(Z["half"])
    cen = np.array(D[last]["disc_centre"]); Rl = D[last]["disc_R"]
    ax = fig.add_subplot(len(P), 3, 3 * r + 1)
    ax.imshow(img, extent=((c0[0] - HALF - cen[0]) / Rl, (c0[0] + HALF - cen[0]) / Rl, (c0[1] + HALF - cen[1]) / Rl, (c0[1] - HALF - cen[1]) / Rl))
    sc = ax.scatter(rel[:, 0], rel[:, 1], c=t - t[0], cmap="plasma", s=8)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="lime")); ax.set_xlim(-1.3, 1.3); ax.set_ylim(1.3, -1.3); ax.set_aspect("equal")
    ax.set_title(f"{nm}：壶嘴尖轨迹（颜色=时间）"); plt.colorbar(sc, ax=ax, fraction=0.046)
    ax = fig.add_subplot(len(P), 3, 3 * r + 2); ax.plot(t - t[0], jh, ".-", ms=3); ax.set_ylabel("离液面高度 mm"); ax.set_ylim(0, 75); ax.grid(alpha=.3)
    ax2 = ax.twinx(); ax2.plot(t - t[0], tl, "r.-", ms=2, alpha=.6); ax2.set_ylabel("倾角 °", color="r"); ax2.set_ylim(45, 95)
    ax.set_title(f"{nm}：高度（蓝）与倾角（红）"); ax.set_xlabel("t [s]（出图案阶段起）")
    # wiggle: lateral component perpendicular to the overall motion direction, detrended
    d = rel - rel.mean(0); u, s_, vt = np.linalg.svd(d, full_matrices=False); lat = d @ vt[1]
    lat = lat - np.polyval(np.polyfit(t, lat, 3), t)
    f = np.linspace(0.5, 12, 400); pw = lombscargle(t, lat, 2 * np.pi * f, normalize=True)
    ax = fig.add_subplot(len(P), 3, 3 * r + 3); ax.plot(f, pw); ax.set_xlabel("频率 Hz"); ax.set_ylabel("功率（归一化）")
    fpk = float(f[np.argmax(pw)]); amp = float(np.std(lat) * np.sqrt(2) * Rl * 1e3)
    dt = np.diff(t); ax.set_title(f"{nm}：横向摆动频谱，峰 {fpk:.1f} Hz，幅度≈{amp:.1f} mm（帧间隔中位 {np.median(dt)*1e3:.0f} ms）")
    summ[nm] = dict(dur_s=round(t[-1] - t[0], 1), jet_mm_median=round(float(np.median(jh)), 1), jet_mm_p90=round(float(np.percentile(jh, 90)), 1),
                    tilt_med=round(float(np.median(tl)), 1), tilt_range=[round(float(np.percentile(tl, 5)), 1), round(float(np.percentile(tl, 95)), 1)],
                    wiggle_peak_hz=round(fpk, 1), wiggle_peak_power=round(float(pw.max()), 2), wiggle_amp_mm=round(amp, 1), R_mm=round(Rl * 1e3, 1))
fig.tight_layout(); fig.savefig(f"{B}/actions_5.png", dpi=75)
json.dump(summ, open(f"{B}/actions_5.json", "w"), ensure_ascii=False, indent=1)
for k, v in summ.items(): print(k, v)
