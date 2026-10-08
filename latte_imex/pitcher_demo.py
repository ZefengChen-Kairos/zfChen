"""Figures for the level-0 pitcher demo:
    python -m latte_imex.pitcher_demo --out results/pitcher_demo [--run runs/pitcher_heart_256]
  inputs.png   : what the barista intends (tip path, height, wanted flow) and what the pitcher model turns it
                 into (tilt, remaining volume, head, flow, impact speed, jet diameter, hit point)
  side_view.png: pitcher, free surface and jet at four instants (side projection)
  tilt_flow.png: steady weir curve Q(tilt) for three fill volumes
"""
import argparse, math, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .pitcher import Pitcher, Pose, BaristaScript, records_table, rotation

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
MILK = "#f3eee4"; BROWN = "#6b3f1f"; ACC = "#c8552d"; GREY = "0.45"


def inputs_figure(T, path, D_L=0.08):
    t = T["t"]
    fig, axs = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    axs = axs.ravel()
    axs[0].plot(t, np.degrees(T["tilt"]), color=ACC); axs[0].set_ylabel("壶倾角 tilt [°]")
    axs[1].plot(t, T["V"] * 1e6, color=BROWN); axs[1].set_ylabel("壶内剩余 V [mL]")
    axs[2].plot(t, T["h_max"] * 1e3, color=GREY, label="h_max"); axs[2].plot(t, T["h_bar"] * 1e3, color=ACC, label="h̄ (流量加权)")
    axs[2].axhline(0, color="k", lw=0.5); axs[2].set_ylabel("液面高出壶嘴 h [mm]"); axs[2].legend(fontsize=8)
    axs[3].plot(t, T["Q_want"] * 1e6, "--", color=GREY, label="咖啡师想要的 Q")
    axs[3].plot(t, T["Q"] * 1e6, color=ACC, label="堰流模型给出的 Q"); axs[3].set_ylabel("流量 Q [mL/s]"); axs[3].legend(fontsize=8)
    axs[4].plot(t, T["z_tip"] * 100, color=GREY, label="壶嘴高度"); axs[4].plot(t, T["drop"] * 100, color=ACC, label="射流落差")
    axs[4].set_ylabel("[cm]"); axs[4].legend(fontsize=8)
    axs[5].plot(t, T["U_perp"], color=ACC, label="U⊥ 冲击法向速度"); axs[5].plot(t, T["u_h"], color=GREY, label="u_h 水平速度")
    axs[5].set_ylabel("[m/s]"); axs[5].legend(fontsize=8)
    axs[6].plot(t, T["d0"] * 1e3, color=GREY, label="出口 d0"); axs[6].plot(t, T["d_hit"] * 1e3, color=ACC, label="落点 d_jet")
    axs[6].set_ylabel("射流直径 [mm]"); axs[6].legend(fontsize=8); axs[6].set_xlabel("t [s]")
    axs[7].plot(t, T["y_hit"] / D_L, color=ACC, label="y_hit"); axs[7].plot(t, T["x_hit"] / D_L, color=GREY, label="x_hit")
    axs[7].set_ylabel("落点（杯径单位）"); axs[7].legend(fontsize=8); axs[7].set_xlabel("t [s]")
    labels = T["label"]
    for ax in axs:
        ax.grid(alpha=0.25)
        prev = None
        for i, lab in enumerate(labels):
            if lab != prev:
                ax.axvline(t[i], color="k", lw=0.4, alpha=0.4); prev = lab
    prev = None
    for i, lab in enumerate(labels):
        if lab != prev:
            axs[0].text(t[i] + 0.03, axs[0].get_ylim()[1] * 0.97, lab, fontsize=7, va="top", rotation=90); prev = lab
    fig.suptitle("第 0 层壶模型：咖啡师意图（虚线/高度）→ 壶的倾角、液面、堰流流量 → 射流冲击量（2D 模型的六个输入）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120); plt.close(fig)


def side_view(P, S, records, path, times=(1.0, 3.0, 5.0, 6.4)):
    fig, axs = plt.subplots(1, len(times), figsize=(4.2 * len(times), 5.2))
    g = P.geom
    for ax, tq in zip(axs, times):
        k = int(np.argmin(np.abs(np.array([r["t"] for r in records]) - tq)))
        r = records[k]
        pose = Pose(tip=tuple(r["tip"]), yaw=S.barista.yaw, tilt=r["tilt"], roll=0.0)
        # silhouette: the body profile in the body XZ plane (phi = 0 and phi = pi), to world (y, z)
        zz = np.linspace(0, g.height, 60)
        prof = np.concatenate([np.stack([g.r_body(zz), 0 * zz, zz], -1)[::-1],
                               np.stack([-g.r_body(zz), 0 * zz, zz], -1)])
        rim = P.world(pose, g.rim)
        W = P.world(pose, prof)
        ax.plot(W[:, 1] * 100, W[:, 2] * 100, color="0.2", lw=1.5)
        ax.plot(rim[:, 1] * 100, rim[:, 2] * 100, color="0.2", lw=1.0)
        ax.plot([W[0, 1] * 100, W[-1, 1] * 100], [W[0, 2] * 100, W[-1, 2] * 100], color="0.2", lw=1.5)
        # free surface: interior points below c, drawn as a filled band (projection)
        pts = P.world(pose, g.points)
        below = pts[:, 2] < r["c"]
        ax.scatter(pts[below, 1] * 100, pts[below, 2] * 100, s=1.2, color=MILK, alpha=0.5, edgecolors="none")
        ax.axhline(r["c"] * 100, color=ACC, lw=0.6, ls="--")
        J = r["jet"]
        if J is not None:
            poly = P.jet_polyline(J, 40)
            lw = np.clip(J["d_hit"] * 1e3, 0.6, 8)
            ax.plot(poly[:, 1] * 100, poly[:, 2] * 100, color=MILK, lw=lw, solid_capstyle="round", path_effects=None)
            ax.plot(poly[:, 1] * 100, poly[:, 2] * 100, color="0.3", lw=0.4)
        # cup: 8.6 cm wide, 7 cm tall, tilted 25 deg towards the pitcher so that its near rim sits 6 mm above the
        # (horizontal) coffee surface z = 0; the surface centre is the origin of the cup frame
        a = math.radians(25.0); R, H = 4.3, 7.0
        u = np.array([math.sin(a), math.cos(a)]); w = np.array([math.cos(a), -math.sin(a)])
        d = (0.6 + R * math.sin(a)) / math.cos(a)
        rim_c = d * u; bot_c = -(H - d) * u
        corners = np.array([bot_c - R * w, bot_c + R * w, rim_c + R * w, rim_c - R * w])
        ax.plot(*np.vstack([corners[[3, 0, 1, 2]]]).T, color="0.2", lw=1.2)
        def cross(p, q):          # z = 0 crossing on the wall p -> q
            f = -p[1] / (q[1] - p[1]); return p + f * (q - p)
        coffee = np.array([corners[0], corners[1], cross(corners[1], corners[2]), cross(corners[0], corners[3])])
        ax.fill(coffee[:, 0], coffee[:, 1], color=BROWN, alpha=0.9)
        cl = P.clearance(pose) * 100
        ax.text(-8.5, 14.5, f"壶壁最低点离咖啡面 {cl:+.1f} cm" if np.isfinite(cl) else "壶壁不在杯口上方", fontsize=8)
        ax.set_aspect("equal"); ax.set_xlim(-9, 9); ax.set_ylim(-5, 16)
        q = r["Q"] * 1e6
        ax.set_title(f"t = {r['t']:.2f} s   tilt {math.degrees(r['tilt']):.0f}°   Q {q:.1f} mL/s\n{r['label']}", fontsize=9)
        ax.set_xlabel("y [cm]"); ax.grid(alpha=0.2)
    axs[0].set_ylabel("z [cm]")
    fig.suptitle("侧视：壶、水平液面（虚线）、自由射流落到咖啡面；杯子按咖啡师的习惯向壶倾斜 25°", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120); plt.close(fig)


def tilt_flow(P, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    tilts = np.radians(np.linspace(30, 80, 101))
    pose = Pose(tip=(0, 0, 0.03), yaw=-math.pi / 2)
    for V, col in [(250e-6, GREY), (200e-6, ACC), (150e-6, BROWN)]:
        Q = []
        for th in tilts:
            pose.tilt = th
            c, _ = P.free_surface(pose, V)
            Q.append(P.weir(pose, c)["Q_ss"] * 1e6)
        ax.plot(np.degrees(tilts), Q, color=col, label=f"壶内 {V*1e6:.0f} mL")
    ax.axhspan(6, 24, color=ACC, alpha=0.08, label="学长估计的 Q 区间 6–24 mL/s")
    ax.set_ylim(0, 60); ax.set_xlabel("倾角 [°]"); ax.set_ylabel("稳态流量 Q [mL/s]"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_title("V 形堰流：同一倾角下流量取决于壶里还剩多少", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pitcher_demo")
    ap.add_argument("--run", default="runs/pitcher_heart_256")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = BaristaScript(); S.sample(S.T)
    T = records_table(S.records)
    inputs_figure(T, os.path.join(a.out, "inputs.png"))
    side_view(S.pitcher, S, S.records, os.path.join(a.out, "side_view.png"))
    tilt_flow(Pitcher(), os.path.join(a.out, "tilt_flow.png"))
    with open(os.path.join(a.out, "records.csv"), "w") as f:
        keys = ["t", "tilt", "V", "Q_want", "Q", "h_max", "h_bar", "z_tip", "U_perp", "u_h", "d0", "d_hit", "x_hit", "y_hit"]
        f.write(",".join(keys + ["label"]) + "\n")
        for i in range(len(T["t"])):
            f.write(",".join(f"{T[k][i]:.6g}" for k in keys) + f",{T['label'][i]}\n")
    for fn in ("final.png", "movie.mp4", "metrics.json"):
        src = os.path.join(a.run, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, fn))
    print("poured", (S.records[0]["V"] + S.records[0]["Q"] * S.dt - T["V"][-1]) * 1e6, "mL; wrote", a.out)
