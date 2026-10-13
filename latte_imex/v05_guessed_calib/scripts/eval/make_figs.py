"""make_figs.py -- figures of the evaluation (run after analyze.py).

  final_vs_real.png  rows = six patterns (calibration rows marked); columns: real photo, real whiteness w, V0.5 (original
                     actions), FS (all-physical, uncalibrated), CAL (all-physical, calibrated on tulip + leaf); same
                     canonical frame, colour map and metric lines as review/v05real/compare_real.png
  FS_vs_CAL.png      V0.5 | FS | CAL | CAL - FS, colours of v05allphys FS_six.png (c stretched 0.35..1), physical top view
  params.png         the calibrated closure vs FS / FSp2 / the 8 near-optimal samples in the unit box, with bounds
  ident.png          N = 128: FS and the 8 near-optimal closures on six patterns (four tests + tulip, leaf)
  blind/<p>.png      real photo + whiteness, then FS and CAL in a random order labelled A / B (key in blind_key/key.json)
All top views: physical top view, canonical pattern frame (pour start B at top, stem / cut exit at bottom); a solver field
c_M[iy, ix] is drawn with y up (origin lower), a T-frame array with row 0 at the top (as compare_real.png).

    cd /home/user/zfChen && python3 <this> DIR [main fsvcal params ident blind]
"""
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import v05_guessed_calib as G   # noqa: E402

C, SN = G.real()
_spec = importlib.util.spec_from_file_location("v05real_figs", os.path.join(G.REAL, "figs.py"))
RF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RF)                      # photo(), field_rgb() (LATTE), rim(), clean(), rcParams
import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt.rcParams.update({"font.size": 9, "font.family": ["DejaVu Sans", "WenQuanYi Zen Hei"]})
# colours of v05allphys/final/figs.py (FS_six.png)
CM = LinearSegmentedColormap.from_list("s", [(0.25, 0.14, 0.06), (0.62, 0.45, 0.30), (1.0, 0.97, 0.90)])
CM.set_bad((0.86, 0.86, 0.86))
CMW = CM.copy()                                   # same colours, white outside the cup (blind sheets, next to the photo)
CMW.set_bad((1.0, 1.0, 1.0))
BOXZ = dict(heart=(-0.3, 0.3, -0.35, 0.35), push_heart=(-0.35, 0.35, -0.35, 0.35), layered_heart=(-0.25, 0.25, -0.3, 0.3),
            tulip=(-0.3, 0.3, -0.32, 0.38), leaf=(-0.3, 0.3, -0.38, 0.38), swan=(-0.32, 0.3, -0.2, 0.36))
CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="郁金香", leaf="树叶", swan="天鹅")
ROLE = {p: ("校准图" if p in G.NAMES else "检验图") for p in G.PATTERNS}
D_MM = 80.0


def stretch(c):
    return np.clip((c - 0.35) / 0.65, 0, 1)


def show_model(ax, cM, box=None, title=None, cmap=CM):
    """FS_six colours, physical top view (y up)."""
    ax.imshow(stretch(cM), cmap=cmap, vmin=0, vmax=1, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5),
              interpolation="nearest")
    if box:
        x0, x1, y0, y1 = box
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    else:
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(0.49 * np.cos(t), 0.49 * np.sin(t), color="0.55", lw=0.6)
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def fields(root, p):
    cache = os.path.join(root, "cache")
    rec = json.load(open(os.path.join(root, "calib", "calib.json")))
    cal = {k: float(rec["best"]["params"][k]) for k in G.CLOSURE8}
    out = dict(V05=C.load_sim(C.sim_paths(p)[0]["V05"]))
    out["FS"] = G.load_field(cache, G.init_points()["FS"], p, 256)[:2]
    out["CAL"] = G.load_field(cache, cal, p, 256)[:2]
    return out


def mlines(r):
    c, lf = r["compare"], r["lfin"]
    t = c["best"]["transform"]
    return (f"raw  IoU {c['raw']['iou_c04']:.2f} / {c['raw']['iou_c05']:.2f}, bd {c['raw']['bdist_c04_mm']:.1f} mm\n"
            f"best IoU {c['best']['iou_c04']:.2f} / {c['best']['iou_c05']:.2f}, bd {c['best']['bdist_c04_mm']:.1f} mm\n"
            f"(θ {t['theta_deg']:+.1f}°, d ({t['dx_D']:+.3f}, {t['dy_D']:+.3f}) D)\n"
            f"L_fin {lf['L_fin']:.2f}   褐线（五线）{lf['seams_model']}")


# ---------------------------------------------------------------- final_vs_real.png
def main_fig(root, E):
    ed = os.path.join(root, "eval")
    cols = [("V05", "V0.5 原版（原动作）"), ("FS", "FS（物理，未校准）"), ("CAL", "CAL（物理，郁金香+树叶校准）")]
    fig, axs = plt.subplots(6, 5, figsize=(14.0, 23.5))
    for i, p in enumerate(G.PATTERNS):
        R = E["N256"][p]
        V = G.video(p, os.path.join(root, "video"))
        w, wm = V["w"], V["wm"]
        F = fields(root, p)
        ax = axs[i, 0]; ax.imshow(RF.photo(p)); RF.clean(ax)
        rot = "" if C.PHI[p] == 0 else f"（坐标修正 {C.PHI[p]:.0f}°）"
        ax.set_title(f"真实照片，末帧{rot}", fontsize=8)
        ax.set_ylabel(f"{p} ({CN[p]})\n[{ROLE[p]}]", fontsize=11, color="#b03000" if p in G.NAMES else "black")
        ax = axs[i, 1]; ax.imshow(RF.field_rgb(np.where(wm, w, 0), wm)); RF.rim(ax, M=wm.shape[0]); RF.clean(ax)
        ax.set_title(f"真实白度 w   white(w>0.5) {V['white_frac']:.3f}", fontsize=8)
        ax.set_xlabel(f"视频褐线（五线）{V['n']}", fontsize=7.5)
        for j, (k, lab) in enumerate(cols):
            cM, mM = F[k]
            ax = axs[i, 2 + j]
            ax.imshow(RF.field_rgb(cM[::-1, :], mM[::-1, :])); RF.rim(ax, M=cM.shape[0]); RF.clean(ax)
            ax.set_title(f"{lab}\nwhite(c>0.4) {R[k]['compare']['raw']['white_frac_c04']:.3f}", fontsize=8)
            ax.set_xlabel(mlines(R[k]), fontsize=7)
    fig.suptitle("末帧对真实视频（256²，物理俯视、规范坐标：起倒点 B 在上，尾在下）\n"
                 "FS 与 CAL 用同一套猜测的全物理动作 FS；CAL = 只在郁金香 + 树叶（红字行）上校准 8 个闭合参数后冻结；V0.5 = 原版（它自己猜的动作和入口）\n"
                 "IoU = 模型 c>0.4 / c>0.5 对视频 w>0.5（杯内）；bd = c>0.4 白区对称平均边界距离；raw = 原样，"
                 "best = 刚体 |θ|≤25°、|d|≤0.06 D（compare.best_align）\n"
                 "L_fin 与褐线数（x = −0.10, −0.05, 0, 0.05, 0.10 D 五条线）用预注册损失自己的对齐（2.5° / 0.01 D 网格）",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.0)
    fig.savefig(os.path.join(ed, "final_vs_real.png"), dpi=105)
    plt.close(fig)


# ---------------------------------------------------------------- FS_vs_CAL.png
def fsvcal_fig(root, E):
    ed = os.path.join(root, "eval")
    fig = plt.figure(figsize=(10.4, 18.0))
    axs = fig.subplots(6, 4, gridspec_kw=dict(left=0.09, right=0.89, top=0.955, bottom=0.025, hspace=0.32, wspace=0.10))
    for i, p in enumerate(G.PATTERNS):
        F = fields(root, p)
        box = BOXZ[p]
        for j, k in enumerate(("V05", "FS", "CAL")):
            ax = axs[i, j]
            show_model(ax, F[k][0], box)
            r = E["N256"][p][k]
            lab = dict(V05="V0.5 原版（原动作）", FS="FS（未校准）", CAL="CAL（校准后冻结）")[k]
            ax.set_title(f"{lab}" if i == 0 else "", fontsize=9)
            txt = f"white {r['compare']['raw']['white_frac_c04']:.3f}"
            if k != "V05":
                txt += f"   L_fin {r['lfin']['L_fin']:.2f}\n结构判据 {r['auto_verdict']}"
            ax.set_xlabel(txt, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{p} ({CN[p]})\n[{ROLE[p]}]", fontsize=10, color="#b03000" if p in G.NAMES else "black")
        ax = axs[i, 3]
        d = F["CAL"][0] - F["FS"][0]
        im = ax.imshow(d, cmap="coolwarm", vmin=-0.6, vmax=0.6, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5),
                       interpolation="nearest")
        x0, x1, y0, y1 = box
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("CAL − FS（红 = CAL 更白）" if i == 0 else "", fontsize=9)
        ax.set_xlabel(f"|Δc| 杯内均值 {np.nanmean(np.abs(d)):.3f}", fontsize=7.5)
    cax = fig.add_axes([0.915, 0.42, 0.014, 0.16])
    fig.colorbar(im, cax=cax, label="Δc = c_CAL − c_FS")
    fig.suptitle("FS 与 CAL（同一套猜测动作 FS），c 拉伸 0.35..1（FS_six.png 的配色），物理俯视（y 向上，起倒点在上）\n"
                 "红字行 = 校准图（郁金香、树叶），其余四个 = 检验图；结构判据 = V05_ALLPHYS_RESULT §3.1 的可数部分（自动）",
                 fontsize=9.5, y=0.992)
    fig.savefig(os.path.join(ed, "FS_vs_CAL.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- params.png
def params_fig(root, E):
    ed = os.path.join(root, "eval")
    cl = E["closures"]
    I = E.get("ident")
    keys = list(G.CLOSURE8)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for i, k in enumerate(keys):
        y = len(keys) - 1 - i
        ax.plot([0, 1], [y, y], color="0.75", lw=6, solid_capstyle="butt", zorder=0)
        lo, hi, sc = cl["bounds"][k]
        ax.text(-0.02, y, f"{k}\n[{lo:g}, {hi:g}] {sc}", ha="right", va="center", fontsize=8.5)
        if I:
            for nk, s in I["set"].items():
                if nk.startswith("near_") and nk not in ("near_1", "near_3"):
                    ax.plot(s["unit"][k], y, "o", ms=5, color="#e69f00", alpha=0.8, zorder=2)
        for nm, mk, col, ms in (("FS", "s", "black", 8), ("FSp2", "D", "0.45", 7), ("start0_best", "o", "#0072b2", 8),
                                ("CAL", "*", "#d55e00", 14)):
            ax.plot(cl[nm]["unit"][k], y, mk, color=col, ms=ms, zorder=3, mec="white", mew=0.6)
        ax.text(1.02, y, f"CAL {cl['CAL']['values'][k]:.4g}   (FS {cl['FS']['values'][k]:.4g}, "
                         f"起点FS最优 {cl['start0_best']['values'][k]:.4g})", ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("单位盒坐标（0 = 下界，1 = 上界；log 尺度的参数按 log 映射）")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    h = [plt.Line2D([], [], marker=m, color=c, ls="", ms=ms, label=l) for m, c, ms, l in
         (("s", "black", 8, "FS（未校准）"), ("D", "0.45", 7, "FSp2（起点 2）"), ("o", "#0072b2", 8, "起点 FS 的最优（损失 1.312）"),
          ("*", "#d55e00", 13, "CAL = 总最优（起点 FSp2，损失 1.292）"), ("o", "#e69f00", 5, "其余 6 个近最优样本（损失 ≤ 1.333）"))]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ed, "params.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- ident.png
def ident_fig(root, E):
    ed = os.path.join(root, "eval")
    I = E["ident"]
    cache = os.path.join(root, "cache")
    cols = ["FS"] + [k for k in I["names"] if k.startswith("near_")]
    pats = list(G.TEST) + list(G.NAMES)
    fig, axs = plt.subplots(len(pats), len(cols), figsize=(1.85 * len(cols), 2.15 * len(pats)))
    for i, p in enumerate(pats):
        for j, k in enumerate(cols):
            cM = G.load_field(cache, I["set"][k]["params"], p, I["N"])[0]
            ax = axs[i, j]
            show_model(ax, cM, BOXZ[p])
            r = I["per"][p][k]
            ax.set_xlabel(f"L_fin {r['L_fin']:.2f}  白区比 {r['white_ratio']:.2f}", fontsize=6.5)
            if i == 0:
                s = I["set"][k]
                ax.set_title("FS（未校准）" if k == "FS" else
                             f"{k}{'=CAL' if k == 'near_1' else ''}{'=起点FS最优' if k == 'near_3' else ''}\n"
                             f"校准损失 {s['cal_loss']:.3f}", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{CN[p]} [{ROLE[p]}]", fontsize=8)
    fig.suptitle(f"可辨识性：校准损失相差 ≤ 0.041 的 8 个闭合（N = {I['N']}，同一套 FS 动作）对六个图（四检验 + 两校准）的预测；"
                 "FS_six 配色，物理俯视", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(os.path.join(ed, "ident.png"), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- blind material
def crop_T(img, M, h=0.36):
    i0, i1 = int(round((0.5 - h) * M)), int(round((0.5 + h) * M))
    return img[i0:i1, i0:i1]


def blind(root, E):
    bd, kd = os.path.join(root, "blind"), os.path.join(root, "blind_key")
    os.makedirs(bd, exist_ok=True); os.makedirs(kd, exist_ok=True)
    rng = random.SystemRandom()
    key = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), note="A / B -> set; FS = all-physical closure, uncalibrated; "
               "CAL = all-physical closure calibrated on tulip + leaf (frozen); same FS actions", patterns={})
    for p in G.PATTERNS:
        F = fields(root, p)
        V = G.video(p, os.path.join(root, "video"))
        w, wm = V["w"], V["wm"]
        order = ["FS", "CAL"] if rng.random() < 0.5 else ["CAL", "FS"]
        key["patterns"][p] = dict(A=order[0], B=order[1])
        ph = RF.photo(p)
        wimg = RF.field_rgb(np.where(wm, w, 0), wm)
        fig, axs = plt.subplots(2, 4, figsize=(13.2, 7.6))
        for r, full in enumerate((True, False)):
            ax = axs[r, 0]
            ax.imshow(ph if full else crop_T(ph, ph.shape[0]), interpolation="lanczos"); RF.clean(ax)
            ax.set_title("真实照片（末帧）" if full else "真实照片，中央放大", fontsize=9)
            ax = axs[r, 1]
            ax.imshow(wimg if full else crop_T(wimg, wm.shape[0])); RF.clean(ax)
            if full:
                RF.rim(ax, M=wm.shape[0])
            ax.set_title("真实白度 w" if full else "真实白度 w，中央放大", fontsize=9)
            for j, lab in enumerate(("A", "B")):
                cM = F[order[j]][0]
                ax = axs[r, 2 + j]
                show_model(ax, cM, None if full else (-0.36, 0.36, -0.36, 0.36), cmap=CMW)
                RF.clean(ax)
                ax.set_title(f"{lab}" if full else f"{lab}，中央放大", fontsize=12 if full else 9, fontweight="bold")
        fig.suptitle(f"{p}（{CN[p]}）：真实末帧 vs 模型 A / B（同一套动作、同一渲染：c 拉伸 0.35..1，物理俯视，起倒点在上）",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2)
        fig.savefig(os.path.join(bd, f"{p}.png"), dpi=105)
        plt.close(fig)
    G._jdump(key, os.path.join(kd, "key.json"))


def main():
    root = os.path.abspath(sys.argv[1])
    which = sys.argv[2:] or ["main", "fsvcal", "params", "ident", "blind"]
    E = json.load(open(os.path.join(root, "eval", "eval.json")))
    if "main" in which:
        main_fig(root, E)
    if "fsvcal" in which:
        fsvcal_fig(root, E)
    if "params" in which:
        params_fig(root, E)
    if "ident" in which and "ident" in E:
        ident_fig(root, E)
    if "blind" in which:
        blind(root, E)


if __name__ == "__main__":
    main()
