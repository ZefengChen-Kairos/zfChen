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
    ("f:FS", "① 参数全部在物理范围内\n六个都成形"),
    (os.path.join(SP, "v05pour/loop/runs256/PLS1_{p}_N256"), "② 真实奶壶倒得出的动作\n用壶模型实际倒出：五个成形"),
    (os.path.join(SP, "v05pour/robust/runs/P1_{p}_N256"), "③ 倒的高度偏高 3 mm\n千层心、郁金香走样"),
]
# verdicts (V05_ALLPHYS_RESULT 3.1 criteria) from V05_PITCHER_CHECK.md and V05_POURABLE_RESULT.md
VERD = {
    "heart": ["是", "是", "是"],
    "push_heart": ["是", "是", "是"],
    "layered_heart": ["是", "部分", "否"],
    "tulip": ["是", "是", "部分"],
    "leaf": ["是", "是", "是"],
    "swan": ["是", "是", "是"],
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
    fig = plt.figure(figsize=(3.0 * nc + 0.7, 2.75 * len(rows) + 1.3), dpi=140)
    gs = fig.add_gridspec(len(rows), nc, left=0.1, right=0.99, top=0.905, bottom=0.02, wspace=0.05, hspace=0.07)
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
    fig.text(0.5, 0.988, "V0.5 拉花模拟：六个图案（256²）", ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.968, "框色：绿 = 成形，橙 = 部分成形，红 = 不成形", ha="center", va="top", fontsize=9, color="#555")
    fig.savefig(out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pitcher_compare.png")
