"""One figure for the report: FS actions poured by the CAD (NX) pitcher model vs the redesigned (pitcher-constrained)
actions poured by the same pitcher model.  All fields are stored 256^2 final states; colours / crops as FS_six."""
import os
import sys

import numpy as np

SP = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review"
sys.path.insert(0, os.path.join(SP, "v05allphys", "final"))
import ana as A  # noqa: E402
import figs as F  # noqa: E402  (CM, BOXZ, ALL)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]

NAMES = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="郁金香", leaf="树叶", swan="天鹅")
COLS = [
    ("V05", "V0.5 原版\n（参照）"),
    ("f:FS", "全物理 FS\n设计入口（未经壶核对）"),
    (os.path.join(SP, "v05pitcher/solve/runs/PD_{p}"), "FS 动作，壶模型实倒\n起始装 350 mL"),
    (os.path.join(SP, "v05pitcher/solve/runs/PD250_{p}"), "FS 动作，壶模型实倒\n起始装 250 mL"),
    (os.path.join(SP, "v05pour/loop/runs256/PLS1_{p}_N256"), "按壶约束重新设计的动作\n壶模型实倒（装 100–125 mL）"),
]
# verdicts (V05_ALLPHYS_RESULT 3.1 criteria) from V05_PITCHER_CHECK.md and V05_POURABLE_RESULT.md
VERD = {
    "heart": ["", "是", "部分", "是", "是"],
    "push_heart": ["", "是", "是", "是", "是"],
    "layered_heart": ["", "是", "是", "是", "部分"],
    "tulip": ["", "是", "部分", "是", "是"],
    "leaf": ["", "是", "部分", "是", "是"],
    "swan": ["", "是", "否", "否", "是"],
}
VC = {"是": "#2f7d4f", "部分": "#b07510", "否": "#b23a32"}


def load(src, p):
    if src.startswith("/"):
        z = np.load(os.path.join(src.format(p=p), "c.npz"))
        c, m = z["c"].astype(float), z["mask"]
        return np.where(m, c, np.nan)
    return A.load(src, p)[0]


def main(out):
    rows = F.ALL
    nc = len(COLS)
    fig = plt.figure(figsize=(2.75 * nc + 0.6, 2.75 * len(rows) + 1.9), dpi=140)
    gs = fig.add_gridspec(len(rows), nc, left=0.07, right=0.995, top=0.85, bottom=0.03, wspace=0.05, hspace=0.07)
    for i, p in enumerate(rows):
        x0, x1, y0, y1 = F.BOXZ[p]
        for j, (src, lab) in enumerate(COLS):
            a = fig.add_subplot(gs[i, j]); a.set_xticks([]); a.set_yticks([])
            c = load(src, p)
            a.imshow(np.clip((c - 0.35) / 0.65, 0, 1), cmap=F.CM, vmin=0, vmax=1, origin="upper",
                     extent=(-0.5, 0.5, 0.5, -0.5), interpolation="nearest")
            a.set_xlim(x0, x1); a.set_ylim(y1, y0)
            v = VERD[p][j]
            if v:
                for s in a.spines.values():
                    s.set_edgecolor(VC[v]); s.set_linewidth(2.2)
                a.text(0.03, 0.97, v, transform=a.transAxes, va="top", ha="left", fontsize=9, color="white",
                       bbox=dict(boxstyle="round,pad=0.2", fc=VC[v], ec="none"))
            if i == 0:
                a.set_title(lab, fontsize=9.5, pad=6)
            if j == 0:
                a.set_ylabel(NAMES[p], fontsize=11)
    # group brackets above the columns
    def bracket(j0, j1, text, color):
        b0 = fig.axes[j0].get_position(); b1 = fig.axes[j1].get_position()
        x0, x1, y = b0.x0, b1.x1, 0.895
        fig.add_artist(Rectangle((x0, y), x1 - x0, 0.004, transform=fig.transFigure, color=color, lw=0))
        fig.text((x0 + x1) / 2, y + 0.008, text, ha="center", va="bottom", fontsize=10.5, color=color, fontweight="bold")
    bracket(2, 3, "A：原来的 FS 动作，交给壶模型去倒\n（有几项倒不出来）", "#8a5a2b")
    bracket(4, 4, "B：按壶的约束重新设计\n动作后，交给壶模型去倒", "#2f5f8a")
    fig.text(0.5, 0.985, "同一个 CAD 奶壶模型（NX 壶，519 mL，短鸟嘴）倒出来的入口，喂给 V0.5（256²）",
             ha="center", va="top", fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.958, "A 与 B 用的是同一只壶；区别在动作：B 少装奶、壶倾得陡（63–81°，与视频一致），A 按 FS 原来的高度倒。每一列六图共用一组物理参数（A 用 FS 那组，B 用重新设计的那组）。\n"
             "倒奶的是受人手速度限制的理想手；A 组带脉动的四个图含 5 Hz 倾转（真人很难做到），B 组不用脉动。框色为成形判定（是 / 部分 / 否，参照 V0.5 原版的决定性结构）。",
             ha="center", va="top", fontsize=9, color="#555")
    fig.savefig(out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pitcher_compare.png")
