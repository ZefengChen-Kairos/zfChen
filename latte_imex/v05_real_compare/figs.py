"""figs.py -- figures for compare_real.md (run after compare.py + merge).

compare_real.png          rows = patterns; columns: real photo (rectified RGB final frame), real whiteness w, V0.5, all-physical
compare_real_overlay.png  white-mask overlays on the video mask: raw and best alignment for V0.5, all-physical, P1
compare_real_zoom.png     central crops (|x|,|y| <= 0.36 D): photo, w, V0.5, P1, all-physical
compare_real_orientation.png  orientation check (swan, push_heart): camera frame -> rectified -> canonical -> V0.5
All top views: physical top view (no mirror), canonical pattern frame (B / pour start at top, stem / cut exit at bottom);
a T-frame array is drawn with row 0 at the top, a solver field as c_M[::-1, :].
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/user/zfChen")
import compare as C  # noqa: E402
from latte_imex import targets as TG, video_extract as VE  # noqa: E402

plt.rcParams.update({"font.size": 9, "font.family": ["DejaVu Sans", "WenQuanYi Zen Hei"]})
LATTE = LinearSegmentedColormap.from_list("latte", [(0.36, 0.22, 0.11), (0.62, 0.42, 0.24), (0.86, 0.76, 0.62), (0.99, 0.98, 0.95)])
NAMES = dict(heart="heart (大白心)", push_heart="push_heart (推推乐)", layered_heart="layered_heart (千层心)",
             tulip="tulip (郁金香)", leaf="leaf (树叶)", swan="swan (天鹅)")
MET = json.load(open(os.path.join(HERE, "metrics_all.json")))
MH = 384
# flow-pulse phase of the all-physical set (verify.md #2): Codex pi -> 0 for tulip / leaf / swan (P4f patch, made after P4 with pi
# failed) and -> pi/2 for layered_heart (strictfp90); both are post-hoc re-guesses of the action, labelled on the figures
PHASE = dict(heart="", push_heart="", layered_heart=", phase π/2*", tulip=", phase 0*", leaf=", phase 0*", swan=", phase 0*")
PHASE_NOTE = "* flow-pulse peak moved from Codex's π: tulip / leaf / swan to 0 (P4f patch), layered_heart to π/2 (strictfp90): post-hoc action re-guesses"


def photo(p):
    z = np.load(os.path.join(TG.TDIR, f"{p}.npz"))
    f = np.load(os.path.join(TG.HERE, "video_targets", p, "frames.npz"))
    rgb = TG.load_rgb(os.path.join(TG.TDIR, f"{p}_frame.png"))
    rh = VE.rectify(rgb, f["ell"][-1], MH, float(z["rot"]), bool(z["flip"]))
    rh = np.clip(C.rotate_T(rh, C.PHI[p]), 0, 1)
    g = (np.arange(MH) + 0.5) / MH - 0.5
    X, Y = np.meshgrid(g, g)
    rh[np.hypot(X, Y) > 0.49] = 1.0
    return rh


def field_rgb(fT, inside):
    rgba = LATTE(np.clip(np.nan_to_num(fT), 0, 1))[..., :3]
    rgba[~inside] = 1.0
    return rgba


def sim_disp(p, k):
    cM, mM = C.load_sim(C.sim_paths(p)[0][k])
    return field_rgb(cM[::-1, :], mM[::-1, :])


def rim(ax, R=0.49, M=None):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot((R * np.cos(t) + 0.5) * M - 0.5, (R * np.sin(t) + 0.5) * M - 0.5, color="0.55", lw=0.6)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fmt(m):
    return f"IoU {m['iou_c04']:.2f} / {m['iou_c05']:.2f}, bd {m['bdist_c04_D']:.3f} D"


def main_fig():
    fig, axs = plt.subplots(6, 4, figsize=(11.2, 17.6))
    for i, p in enumerate(C.PAT):
        R = MET["patterns"][p]
        w = np.load(os.path.join(HERE, f"fields_{p}.npz"))
        wm = w["wm"]
        tag = R["phys_tag"]
        ax = axs[i, 0]; ax.imshow(photo(p)); clean(ax)
        rot = "" if C.PHI[p] == 0 else f" (frame-corrected, rotated {C.PHI[p]:.0f}°)"
        ax.set_title(f"real photo, final frame{rot}", fontsize=7.5)
        ax.set_ylabel(NAMES[p], fontsize=11)
        ax = axs[i, 1]; ax.imshow(field_rgb(np.where(wm, w["w"], 0), wm)); rim(ax, M=wm.shape[0]); clean(ax)
        ax.set_title(f"real whiteness w   white(w>0.5) {R['video']['white_frac']:.3f}", fontsize=8)
        for j, (k, lab) in enumerate((("V05", "V0.5 (Codex inlet)"), ("PHYS", f"all-physical {tag}{PHASE[p]}"))):
            ax = axs[i, 2 + j]; ax.imshow(sim_disp(p, k)); rim(ax, M=256); clean(ax)
            s = R["sims"][k]
            ax.set_title(f"{lab}   white(c>0.4) {s['raw']['white_frac_c04']:.3f}", fontsize=8)
            ax.set_xlabel(f"raw  {fmt(s['raw'])}\nbest {fmt(s['best'])}\n(θ {s['best']['transform']['theta_deg']:+.1f}°, "
                          f"d ({s['best']['transform']['dx_D']:+.3f}, {s['best']['transform']['dy_D']:+.3f}) D)", fontsize=7.5)
    fig.suptitle("Final pattern vs real video (256², physical top view, canonical frame: pour start B at top, stem at bottom)\n"
                 "IoU = sim c>0.4 / c>0.5 vs video w>0.5 inside the cup; bd = symmetric mean boundary distance (c>0.4); "
                 "raw = as simulated, best = rigid |θ|≤25°, |d|≤0.06 D\n" + PHASE_NOTE, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(os.path.join(HERE, "compare_real.png"), dpi=110)
    plt.close(fig)


def overlay(sT, w, wm):
    a = (sT > 0.4) & wm
    b = (w > 0.5) & wm
    img = np.ones(wm.shape + (3,))
    img[wm] = (0.80, 0.80, 0.80)
    img[a & b] = (0.15, 0.15, 0.15)
    img[a & ~b] = (0.85, 0.25, 0.20)
    img[~a & b] = (0.20, 0.45, 0.85)
    return img


def overlay_fig():
    cols = [("V05", "raw"), ("V05", "best"), ("PHYS", "raw"), ("PHYS", "best"), ("P1", "raw"), ("P1", "best")]
    fig, axs = plt.subplots(6, 6, figsize=(15.5, 16.2))
    for i, p in enumerate(C.PAT):
        R = MET["patterns"][p]
        z = np.load(os.path.join(HERE, f"fields_{p}.npz"))
        for j, (k, a) in enumerate(cols):
            ax = axs[i, j]
            ax.imshow(overlay(z[f"{k}_{a}"], z["w"], z["wm"])); clean(ax)
            m = R["sims"][k][a]
            lab = {"V05": "V0.5", "PHYS": R["phys_tag"] + PHASE[p], "P1": "P1 (c_p 0.3)"}[k]
            extra = "" if a == "raw" else f"  θ {m['transform']['theta_deg']:+.1f}°"
            ax.set_title(f"{lab} {a}{extra}\nIoU {m['iou_c04']:.3f}  bd {m['bdist_c04_D']:.3f} D", fontsize=8)
            if j == 0:
                ax.set_ylabel(NAMES[p], fontsize=10)
    fig.suptitle("White-mask overlays (sim c>0.4 vs video w>0.5): dark = both, red = sim only, blue = video only, grey = both brown\n" + PHASE_NOTE,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(HERE, "compare_real_overlay.png"), dpi=100)
    plt.close(fig)


def crop(img, M, h=0.36):
    i0, i1 = int(round((0.5 - h) * M)), int(round((0.5 + h) * M))
    return img[i0:i1, i0:i1]


def zoom_fig():
    fig, axs = plt.subplots(6, 5, figsize=(14, 17))
    for i, p in enumerate(C.PAT):
        R = MET["patterns"][p]
        z = np.load(os.path.join(HERE, f"fields_{p}.npz"))
        Mt = z["wm"].shape[0]
        panels = [(crop(photo(p), MH), "real photo"), (crop(field_rgb(np.where(z["wm"], z["w"], 0), z["wm"]), Mt), "real w"),
                  (crop(sim_disp(p, "V05"), 256), "V0.5"), (crop(sim_disp(p, "P1"), 256), "P1 (c_p 0.3)"),
                  (crop(sim_disp(p, "PHYS"), 256), f"all-physical {R['phys_tag']}{PHASE[p]}")]
        for j, (im, lab) in enumerate(panels):
            ax = axs[i, j]; ax.imshow(im, interpolation="lanczos"); clean(ax); ax.set_title(lab, fontsize=9)
            if j == 0:
                ax.set_ylabel(NAMES[p], fontsize=10)
    fig.suptitle("Central crops |x|, |y| ≤ 0.36 D (29 mm), same orientation as compare_real.png\n" + PHASE_NOTE, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(HERE, "compare_real_zoom.png"), dpi=100)
    plt.close(fig)


def orientation_fig():
    fig, axs = plt.subplots(2, 5, figsize=(17, 7.2))
    for i, p in enumerate(("swan", "push_heart")):
        z = np.load(os.path.join(TG.TDIR, f"{p}.npz"))
        f = np.load(os.path.join(TG.HERE, "video_targets", p, "frames.npz"))
        rgb = TG.load_rgb(os.path.join(TG.TDIR, f"{p}_frame.png"))
        e = f["ell"][-1]
        ax = axs[i, 0]; ax.imshow(rgb[60:560]); clean(ax); ax.set_title(f"{p}: raw camera frame t = {float(z['t'])} s", fontsize=9)
        rh = VE.rectify(rgb, e, MH, float(z["rot"]), bool(z["flip"]))
        g = (np.arange(MH) + 0.5) / MH - 0.5; X, Y = np.meshgrid(g, g); rh[np.hypot(X, Y) > 0.49] = 1
        ax = axs[i, 1]; ax.imshow(rh); clean(ax); ax.set_title(f"rectified T array, row 0 at top\n(targets rot {float(z['rot']):.0f}°)", fontsize=9)
        ax = axs[i, 2]; ax.imshow(photo(p)); clean(ax); ax.set_title(f"canonical frame (PHI {C.PHI[p]:.0f}°)", fontsize=9)
        ax = axs[i, 3]; ax.imshow(sim_disp(p, "V05")); clean(ax); ax.set_title("V0.5 as simulated, c_M[::-1, :]", fontsize=9)
        ax = axs[i, 4]; ax.imshow(sim_disp(p, "PHYS")); clean(ax); ax.set_title("all-physical strictB3 as simulated", fontsize=9)
    fig.suptitle("Orientation check: the rectified view is a rotation (det > 0, no mirror) of the camera view; the swan target is stored "
                 "un-rotated (rot 0) and needs PHI = 135° to reach the canonical frame of the simulation (head top-left, wing fan to the right)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(HERE, "compare_real_orientation.png"), dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["main", "overlay", "zoom", "orient"]
    if "main" in which: main_fig()
    if "overlay" in which: overlay_fig()
    if "zoom" in which: zoom_fig()
    if "orient" in which: orientation_fig()
