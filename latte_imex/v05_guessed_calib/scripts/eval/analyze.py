"""analyze.py -- metrics of the frozen calibrated closure (CAL) vs the uncalibrated all-physical closure (FS) with the
guessed FS actions, six patterns at N = 256, against the real video's final frame; plus the identifiability check at
N = 128 (ident.py).  Writes DIR/eval/eval.json and DIR/eval/tables.md.

Definitions (nothing new; imported, not copied):
  compare.py   canonical frame (leaf PHI 180 deg, swan 135 deg), compare.Sampler, compare.metrics (IoU c > 0.4 / c > 0.5
               vs video w > 0.5, symmetric boundary distance of c > 0.4, white fraction), compare.best_align (rigid
               |theta| <= 25 deg, |d| <= 0.06 D, 1 deg / 0.01 D grid refined 0.25 deg / 0.0025 D); raw = no transform
  L_fin        v05_guessed_calib.lfin (pre-registered loss, its own 2.5 deg / 0.01 D alignment grid); the brown-seam
               counts per line (structnum.stripes, lines x = -0.10 .. 0.10 D) are the ones at the L_fin alignment
  structure    V05_ALLPHYS_RESULT.md section 3.1: v05allphys/final/ana.py (structm.M measures, feature-box high-pass
               contrast vs V0.5, allphys.metrics IoU / r_hp vs V0.5), on the solver field (control coords, 256^2)
1 D = 80 mm.

    cd /home/user/zfChen && python3 <this> DIR
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import v05_guessed_calib as G   # noqa: E402
from latte_imex import calib as CB              # noqa: E402

C, SN = G.real()
ANA = os.path.join(G.REVIEW, "v05allphys", "final")
sys.path.insert(0, ANA)
import ana as A                                  # noqa: E402
import allphys as AP                             # noqa: E402

D_MM = 80.0
CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="郁金香", leaf="树叶", swan="天鹅")
FSP2 = G.init_points()["FSp2"]


# ---------------------------------------------------------------- metrics of one field
def cmp_metrics(cM, mM, p, vdir):
    """compare.py metrics, raw and best (compare.best_align), plus white fraction ratio model / real."""
    V = G.video(p, vdir)
    w, wm = V["w"], V["wm"]
    smp = C.Sampler(cM, mM, V["Mt"])
    out = {}
    sraw = smp.field_T()
    ba = C.best_align(smp, w, wm)
    sbest = smp.field_T(ba["theta_deg"], ba["dx_D"], ba["dy_D"])
    for tag, sT in (("raw", sraw), ("best", sbest)):
        m = C.metrics(sT, w, wm)
        st = SN.stripes(sT, wm, 0.4)
        out[tag] = dict(iou_c04=m["iou_c04"], iou_c05=m["iou_c05"], bdist_c04_mm=m["bdist_c04_D"] * D_MM,
                        bdist_c05_mm=m["bdist_c05_D"] * D_MM, white_frac_c04=m["white_frac_c04"],
                        white_ratio=m["white_frac_c04"] / V["white_frac"], seams=[s["total"] for s in st])
    out["best"]["transform"] = ba
    out["video"] = dict(white_frac=V["white_frac"], seams=V["n"])
    return out, sraw, sbest


def lfin_metrics(cM, mM, p, vdir):
    r = G.lfin(cM, mM, p, vdir)
    a = r["aligned"]
    return dict(L_fin=r["loss"], dice=a["dice"], bnd_mm=a["bnd_D"] * D_MM, S=a["S"], seams_model=a["n_model"],
                seams_video=r["video"]["n"], gaps_model=a["gaps_model"], dips_model=a["dips_model"],
                transform=r["transform"], raw=dict(L=r["raw"]["L"], iou=r["raw"]["iou"], dice=r["raw"]["dice"],
                                                   bnd_mm=r["raw"]["bnd_D"] * D_MM, S=r["raw"]["S"],
                                                   seams_model=r["raw"]["n_model"]))


def structure(c, p, cref, mref, m):
    """ana.py measures (control coords; c, cref = solver fields 256^2 with NaN outside)."""
    met = AP.metrics(np.nan_to_num(c), m, np.nan_to_num(cref), mref, 256)
    try:
        st = A.SM.M[p](c)
    except ValueError as e:                        # structm.m_heart needs white (c > 0.6) on the axis
        st = dict(error=f"{type(e).__name__}: {e}", **heart_fallback(c))
    return dict(struct=st, hp_rel=A.hp(c, p) / A.hp(cref, p), iou04_vs_V05=met["iou04"],
                r_hp_vs_V05=met["r_hp"], white_c04=met["white"], white_c04_V05=met["white_ref"])


def heart_fallback(c):
    """heart without white on the axis: axis profile and the brown length between the lobes' white extent (same lines
    and threshold c > 0.6 as structm.m_heart)."""
    ys = np.linspace(-0.30, 0.32, 800)
    ax = A.SM.line(c, 0, -0.30, 0, 0.32, 800)
    lob = [ys[A.SM.line(c, x, -0.3, x, 0.32, 800) > 0.6] for x in (-0.08, 0.08)]
    lo = min(v.min() for v in lob if len(v)) if any(len(v) for v in lob) else None
    hi = max(v.max() for v in lob if len(v)) if any(len(v) for v in lob) else None
    sel = (ys >= lo) & (ys <= hi) if lo is not None else np.zeros_like(ys, bool)
    return dict(axis_c_max=round(float(ax.max()), 3), axis_white_c06=int((ax > 0.6).sum()),
                lobe_white_y=[lo and round(float(lo), 3), hi and round(float(hi), 3)],
                axis_dark_len_in_lobes_mm=round(float((sel & (ax < 0.6)).sum() * 0.62 / 799 * D_MM), 1))


def verdict(p, s):
    """operational version of the section 3.1 criteria (the closed outline and the overall shape are judged by eye;
    these are the countable parts).  Returns (auto verdict, reason)."""
    t = s["struct"]
    if p == "heart":
        if "error" in t:
            return "否", (f"中轴上没有 c > 0.6 的白（轴上 c 最大 {t['axis_c_max']:.2f}）：心从凹口到尖整条被褐色劈开，"
                         f"瓣的白区范围内轴上褐段 {t['axis_dark_len_in_lobes_mm']:.0f} mm")
        seam = t["axis_dark_len_D"] * D_MM
        ok = t["tip_y"] <= -0.20 and t["lobe_bottom_y"] > 0.0
        return ("是" if ok else "部分"), f"尖 y {t['tip_y']:+.3f} D，轴上褐缝 {seam:.0f} mm"
    if p == "push_heart":
        n = [t[k]["dark_seams"] for k in ("y-0.1", "y+0.0", "y+0.1")]
        return ("是" if min(n) >= 3 else "部分"), f"三条水平线褐缝 {n[0]}/{n[1]}/{n[2]}（V0.5 3/5/5）"
    if p == "layered_heart":
        b = [t["x-0.05"]["bright"], t["x+0.05"]["bright"]]
        cc = t["curl_c95"]
        ok = min(b) >= 4 and min(cc) >= 0.7
        return ("是" if ok else "部分"), f"x=±0.05 亮带 {b[0]}/{b[1]}（≥4），内卷 c95 {cc[0]:.2f}/{cc[1]:.2f}（≥0.7）"
    if p == "tulip":
        ok = min(t["off_dark"]) >= 2 and t["axis_dark"] >= 2
        return ("是" if ok else "部分"), (f"x=±0.06 瓣间褐缝 {t['off_dark'][0]}/{t['off_dark'][1]}（≥2），轴上 {t['axis_dark']}，"
                                          f"底座中心 c {t['base_center_c']:.2f}")
    if p == "leaf":
        v = [t[k]["veins"] for k in ("x-0.08", "x-0.05", "x+0.05", "x+0.08")]
        ok = min(v) >= 5 and t["mean_amp"] >= 0.15
        return ("是" if ok else "部分"), f"叶脉 {'/'.join(map(str, v))}（每条 ≥5），幅值 {t['mean_amp']:.2f}（≥0.15）"
    if p == "swan":
        ok = t["neck_min_cmax"] >= 0.45 and t["head_area_mm2"] >= 5.0 and s["hp_rel"] >= 0.7
        return ("是" if ok else "部分"), (f"颈脊线 c 最小 {t['neck_min_cmax']:.2f}（>0.6 占 {t['neck_frac_cmax_gt_0p6']:.2f}），"
                                          f"头+喙 {t['head_area_mm2']:.1f} mm²，翅纹高通 {s['hp_rel']:.2f}")
    raise KeyError(p)


def unit(params):
    return {k: float(v) for k, v in zip(G.CLOSURE8, CB.encode(params, G.SPACE))}


# ---------------------------------------------------------------- main
def main():
    root = os.path.abspath(sys.argv[1])
    ed = os.path.join(root, "eval")
    vdir = os.path.join(root, "video")
    cache = os.path.join(root, "cache")
    rec = json.load(open(os.path.join(root, "calib", "calib.json")))
    em = json.load(open(os.path.join(root, "evaluate", "metrics.json")))
    CAL = {k: float(rec["best"]["params"][k]) for k in G.CLOSURE8}
    FS = G.init_points()["FS"]
    sets = dict(FS=FS, CAL=CAL)
    out = dict(meta=dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S"), D_mm=D_MM,
        sets=dict(FS="uncalibrated all-physical closure (V05_ALLPHYS_RESULT section 4.1)",
                  CAL="frozen best of the tulip + leaf calibration (calib.json best, start FSp2, N = 128)",
                  V05="V0.5 original (its own guessed actions and inlet), runs/v05_256 (reference only)"),
        actions="FS actions for FS and CAL (fixed, identical); V0.5 has its own (original guessed) actions",
        definitions=__doc__.split("Definitions")[1].split("    cd ")[0].strip()), closures={}, N256={}, repro={})
    for nm, prm in (("FS", FS), ("FSp2", FSP2), ("CAL", CAL), ("start0_best", rec["inits"][0]["best"]["params"])):
        out["closures"][nm] = dict(values={k: float(prm[k]) for k in G.CLOSURE8}, unit=unit(prm))
    out["closures"]["bounds"] = {k: list(v) for k, v in G.SPACE.items()}
    out["closures"]["fixed"] = G.FIXED
    # ---- N = 256: FS, CAL, V05
    for p in G.PATTERNS:
        t0 = time.time()
        cref, mref = C.load_sim(C.sim_paths(p)[0]["V05"])
        R = {}
        fields = {"V05": (cref, mref)}
        for s, prm in sets.items():
            cM, mM, key = G.load_field(cache, prm, p, 256)
            fields[s] = (cM, mM)
            ev = em["patterns"][p]["best" if s == "CAL" else "FS"]
            R[s] = dict(cache_key=key, inlet=dict(V_received_mL=ev["inlet"]["V_received_mL"],
                                                  V_wanted_mL=ev["inlet"]["V_wanted_mL"],
                                                  clip_low_time_frac=ev["inlet"]["clip_low_time_frac"],
                                                  clip_high_time_frac=ev["inlet"]["clip_high_time_frac"],
                                                  Q_mLs=ev["inlet"]["Q_mLs"]),
                        deposited=ev["ledger"]["deposited"], T=ev["ledger"]["T"],
                        stats=ev["stats"], wall_fwd_s=ev["wall_fwd_s"])
        # FS 256^2 = the stored FS runs (reuse check)
        z = np.load(os.path.join(G.FS_RUNS, f"FS_{p}", "c.npz"))
        c32 = np.where(fields["FS"][1], fields["FS"][0], np.nan).astype(np.float32)
        out["repro"][p] = dict(bitwise_equal=bool(np.array_equal(c32, z["c"], equal_nan=True)),
                               same_mask=bool(np.array_equal(z["mask"], fields["FS"][1])))
        for s, (cM, mM) in fields.items():
            cm, sraw, sbest = cmp_metrics(cM, mM, p, vdir)
            lf = lfin_metrics(cM, mM, p, vdir)
            r = R.setdefault(s, {})
            r.update(compare=cm, lfin=lf)
            if s != "V05":
                assert abs(lf["L_fin"] - em["patterns"][p]["best" if s == "CAL" else "FS"]["loss"]) < 1e-12, (p, s)
                r.update(structure(cM, p, cref, mref, mM))
                r["auto_verdict"], r["auto_reason"] = verdict(p, r)
            else:
                r["struct"] = A.SM.M[p](cM)
            np.savez_compressed(os.path.join(ed, f"T_{s}_{p}.npz"), raw=sraw.astype(np.float32),
                                best=sbest.astype(np.float32))
        out["N256"][p] = R
        print(p, f"{time.time() - t0:.0f}s", " | ".join(
            f"{s} L {R[s]['lfin']['L_fin']:.3f} IoU {R[s]['compare']['raw']['iou_c04']:.2f}/{R[s]['compare']['best']['iou_c04']:.2f}"
            f" {R[s].get('auto_verdict', '')}" for s in ("V05", "FS", "CAL")), flush=True)
    for s in ("FS", "CAL", "V05"):
        L = {p: out["N256"][p][s]["lfin"]["L_fin"] for p in G.PATTERNS}
        out[f"summary_{s}"] = dict(L_fin=L, calibration_mean=float(np.mean([L[p] for p in G.NAMES])),
                                   test_mean=float(np.mean([L[p] for p in G.TEST])),
                                   best_iou_c04_test_mean=float(np.mean([out["N256"][p][s]["compare"]["best"]["iou_c04"]
                                                                          for p in G.TEST])),
                                   raw_iou_c04_test_mean=float(np.mean([out["N256"][p][s]["compare"]["raw"]["iou_c04"]
                                                                         for p in G.TEST])))
    # ---- identifiability (N = 128)
    ij = os.path.join(ed, "ident_runs.json")
    if os.path.exists(ij):
        out["ident"] = ident(root, json.load(open(ij)), json.load(open(os.path.join(ed, "ident_set.json"))), vdir)
    G._jdump(out, os.path.join(ed, "eval.json"))
    tables(out, os.path.join(ed, "tables.md"))


def ident(root, runs, iset, vdir):
    cache = os.path.join(root, "cache")
    names = runs["names"]
    near = [k for k in names if k.startswith("near_")]
    res = dict(N=runs["N"], names=names, set={k: dict(v, unit=unit(v["params"])) for k, v in iset["set"].items()},
               n_cal_samples=iset["n_cal_samples"], per={}, spread={})
    old_h = A.SM.h
    A.SM.h = 1.0 / runs["N"]                       # structm samples by h (written for 256^2)
    try:
        for p in G.TEST + G.NAMES:
            V = G.video(p, vdir)
            masks = {}
            res["per"][p] = {}
            for k in names:
                prm = iset["set"][k]["params"]
                cM, mM, key = G.load_field(cache, prm, p, runs["N"])
                ev = runs["runs"][k][p] if p in G.TEST else json.load(open(os.path.join(cache, f"{key}.json")))["eval"]
                sT = C.Sampler(cM, mM, V["Mt"]).field_T()
                masks[k] = (sT > 0.4) & V["wm"]
                try:
                    st = A.SM.M[p](cM)
                except ValueError as e:
                    st = dict(error=f"{type(e).__name__}: {e}", **heart_fallback(cM))
                res["per"][p][k] = dict(L_fin=ev["loss"], dice=ev["dice"], bnd_mm=ev["bnd_D"] * D_MM, S=ev["S"],
                                        seams=ev["n_model"], iou_raw=ev["raw"]["iou"], white_ratio=ev["white_frac"] /
                                        ev["video_white_frac"], V_received_mL=ev["inlet"]["V_received_mL"],
                                        struct=st)
            # agreement between predictions (raw, target grid, c > 0.4)
            pair = []
            for i, a in enumerate(near):
                for b in near[i + 1:]:
                    pair.append(C.CF.iou(masks[a], masks[b]))
            res["spread"][p] = dict(
                L_fin_near=[res["per"][p][k]["L_fin"] for k in near],
                L_fin_range=[float(min(res["per"][p][k]["L_fin"] for k in near)),
                             float(max(res["per"][p][k]["L_fin"] for k in near))],
                L_fin_FS=res["per"][p]["FS"]["L_fin"],
                pairwise_iou_near=dict(mean=float(np.mean(pair)), min=float(np.min(pair))),
                iou_near_vs_FS=[C.CF.iou(masks[k], masks["FS"]) for k in near],
                iou_near1_vs_near3=C.CF.iou(masks["near_1"], masks["near_3"]),
                white_ratio_near=[res["per"][p][k]["white_ratio"] for k in near])
    finally:
        A.SM.h = old_h
    return res


# ---------------------------------------------------------------- judgement by eye
# written after looking at final_vs_real.png, FS_vs_CAL.png, ident.png and the blind sheets (not blind: the judge knew
# which panel is which); the criteria are those of V05_ALLPHYS_RESULT.md section 3.1
EYE = dict(
    heart=("是（带 34 mm 中缝）", "否",
           "CAL 的心是空心的：从凹口往下一大块褐色楔形，下半只剩灰色细纹，底部是圆的、没有尖。真实是实心大白心，FS 明显更像"),
    push_heart=("是", "是（臂分开）但内心散掉",
                "CAL 的臂更多更细，中间那颗内心变成两道细弧，外弧在底部合成一圈细碗，整体下移；FS 三道臂 + 内心干净。真实是三块分开的大白块，两者布局都不对，FS 更接近（IoU 0.46 对 0.36）"),
    layered_heart=("是", "部分",
                   "CAL 外廓不闭合：两瓣像两只翅膀分开，下半没了，只剩中间一根长茎；内卷亮弧还在但更淡。白区更大（0.199 对 0.164），所以 IoU、边界、L_fin 都更好；但心形本身比 FS 差。两者都远小于真实（0.389）"),
    tulip=("是", "部分",
           "CAL 更大，层间多出很多细 V 条，顶上的两颗白滴变成 3–4 根细条，看不出三层叠瓣，更像一把扇子。褐线数更接近视频（S 0.42→0.16），但这是细条纹带来的，不是叠起来的心。真实是几颗实心心叠起来，两者都不像，FS 的层次更清楚"),
    leaf=("是", "部分",
          "CAL 更高更窄，叶脉变成淡而细的灰线（每条剖线 4–5 道），中间多出一根白茎，叶尖的心形没了。FS 叶脉清楚、有叶尖。真实是一大片白叶，叶片宽，FS 更接近"),
    swan=("是（颈一段淡）", "部分",
          "CAL 的颈变成一条宽的糊带（所以脊线 c 反而更高），头是一个小钩，翅只剩几道糊掉的羽，下面一大片灰色身体。FS 颈细、S 形清楚，翅纹清楚。真实是心形头 + 细 S 颈 + 约 10 道羽，FS 更像"),
)


# ---------------------------------------------------------------- tables
def f2(x):
    return f"{x:.2f}"


def tables(E, path):
    L = []
    w = L.append
    w("# 校准后（郁金香 + 树叶）冻结的闭合 vs 未校准的全物理闭合：六图末帧对真实视频（256²）\n")
    w("动作：两组都用同一套猜测的全物理动作 FS（固定，不随闭合重设计）。CAL = 校准最优闭合（起点 FSp2，N = 128 校准），"
      "FS = 未校准的全物理闭合。V0.5 = 原版（它自己猜的动作和入口），只作参照。1 D = 80 mm。")
    w("定义：compare.py 规范坐标；IoU = 模型 c > 0.4（或 0.5）对视频 w > 0.5（杯内 R = 0.49 D）；未对齐 = 原样；"
      "最佳 = 刚体 |θ| ≤ 25°、|d| ≤ 0.06 D（compare.best_align）；边界距离 = c > 0.4 白区边界的对称平均距离；"
      "白区比 = 模型白区分数 / 视频白区分数；L_fin 按预注册（自带 2.5° / 0.01 D 对齐），褐线数为该对齐下五条线 "
      "x = −0.10, −0.05, 0, 0.05, 0.10 D 上的计数（模型 / 视频）。\n")
    w("## 1. 主表\n")
    w("| 图案 | 组 | 未对齐 IoU 0.4 / 0.5 | 最佳 IoU 0.4 / 0.5 | 边界距离 未对齐 / 最佳 [mm] | 白区比 | 褐线数 模型 | 褐线数 视频 | L_fin | 结构判据（§3.1，自动） |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for p in G.PATTERNS:
        tag = "（校准）" if p in G.NAMES else "（检验）"
        for s in ("FS", "CAL", "V05"):
            r = E["N256"][p][s]
            c, lf = r["compare"], r["lfin"]
            av = r.get("auto_verdict", "–")
            w(f"| {CN[p]}{tag} | {s} | {f2(c['raw']['iou_c04'])} / {f2(c['raw']['iou_c05'])} | "
              f"{f2(c['best']['iou_c04'])} / {f2(c['best']['iou_c05'])} | {c['raw']['bdist_c04_mm']:.1f} / "
              f"{c['best']['bdist_c04_mm']:.1f} | {f2(c['raw']['white_ratio'])} | {lf['seams_model']} | {lf['seams_video']} | "
              f"{lf['L_fin']:.3f} | {av} |")
    w("")
    w("## 2. L_fin 汇总\n")
    w("| 组 | 郁金香 | 树叶 | 校准均值 | 大白心 | 推推乐 | 千层心 | 天鹅 | 检验均值 | 检验最佳 IoU 均值 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for s in ("FS", "CAL", "V05"):
        S = E[f"summary_{s}"]
        w(f"| {s} | " + " | ".join(f"{S['L_fin'][p]:.3f}" for p in G.NAMES) + f" | {S['calibration_mean']:.3f} | "
          + " | ".join(f"{S['L_fin'][p]:.3f}" for p in G.TEST) + f" | {S['test_mean']:.3f} | {S['best_iou_c04_test_mean']:.3f} |")
    w("")
    w("## 3. L_fin 分项（对齐后）\n")
    w("| 图案 | 组 | 1 − Dice | 边界 / 0.05 D | 褐线项 S | L_fin | 对齐 θ / dx / dy |")
    w("|---|---|---|---|---|---|---|")
    for p in G.PATTERNS:
        for s in ("FS", "CAL"):
            lf = E["N256"][p][s]["lfin"]
            t = lf["transform"]
            w(f"| {CN[p]} | {s} | {1 - lf['dice']:.3f} | {lf['bnd_mm'] / D_MM / G.BND_SCALE:.3f} | {lf['S']:.3f} | "
              f"{lf['L_fin']:.3f} | {t['theta_deg']:+.1f}° / {t['dx_D']:+.2f} / {t['dy_D']:+.2f}"
              f"{' (窗口边界)' if any(t['at_bound'].values()) else ''} |")
    w("")
    nb = {s: [p for p in G.PATTERNS if any(E["N256"][p][s]["lfin"]["transform"]["at_bound"].values())] for s in ("FS", "CAL")}
    w(f"窗口边界：12 个对齐里 {len(nb['FS']) + len(nb['CAL'])} 个顶到 ±25° / ±0.06 D（FS {len(nb['FS'])}/6，CAL "
      f"{len(nb['CAL'])}/6；只看检验图 FS {sum(p in G.TEST for p in nb['FS'])}/4，CAL {sum(p in G.TEST for p in nb['CAL'])}/4）。"
      "顶窗口时 Dice、边界距离和褐线项都是在窗口边上算的，所以位置和角度误差被低估；CAL 顶窗口更多，它的这部分误差低估得更多。"
      "N = 128 的校准里 CAL 也是这样：郁金香 dy +0.06，树叶 θ −25°（8 个近最优闭合的树叶全部在 θ = −25°）。")
    w("")
    w("## 4. 结构判据（V05_ALLPHYS_RESULT §3.1，ana.py 的量；控制坐标、256²）\n")
    w("| 图案 | FS | CAL | 对 V0.5 的 IoU / r_hp / 高通对比：FS → CAL |")
    w("|---|---|---|---|")
    for p in G.PATTERNS:
        a, b = E["N256"][p]["FS"], E["N256"][p]["CAL"]
        w(f"| {CN[p]} | {a['auto_verdict']}：{a['auto_reason']} | {b['auto_verdict']}：{b['auto_reason']} | "
          f"{a['iou04_vs_V05']:.2f} / {a['r_hp_vs_V05']:+.2f} / {a['hp_rel']:.2f} → {b['iou04_vs_V05']:.2f} / "
          f"{b['r_hp_vs_V05']:+.2f} / {b['hp_rel']:.2f} |")
    w("")
    w("## 5. 入口（求解器实际收到的流量）\n")
    w("| 图案 | FS 接收量 [mL] | CAL 接收量 [mL] | 低流量夹持（时间占比，求解器采样）FS / CAL | FS 沉积 [D³] | CAL 沉积 [D³] |")
    w("|---|---|---|---|---|---|")
    for p in G.PATTERNS:
        a, b = E["N256"][p]["FS"], E["N256"][p]["CAL"]
        w(f"| {CN[p]} | {a['inlet']['V_received_mL']:.1f} | {b['inlet']['V_received_mL']:.1f} | "
          f"{a['inlet']['clip_low_time_frac']:.3f} / {b['inlet']['clip_low_time_frac']:.3f} | {a['deposited']:.4f} | "
          f"{b['deposited']:.4f} |")
    w("")
    w("夹持占比按求解器实际取样的入口算（CAL 2.0–5.5 %，FS 1.7–4.7 %）；按 1 ms 均匀网格算（`verify/inlet_bounds.json`），"
      "CAL 是 1.9–5.4 %。")
    w("")
    w("## 6. 闭合参数\n")
    w("FSp2 在这里只是一个参照起点：FS 的动作（逐笔高度不变）配 p_dep = 2 的闭合，校准损失 1.736。"
      "`V05_ALLPHYS_RESULT.md` §3.3 里的 FSp2 用的是按 p_dep = 2 重新反设计的逐笔高度（细切抬到 6–7 cm），那一组 6/6 成形。\n")
    cl = E["closures"]
    w("| 参数 | 下界 | 上界 | 尺度 | FS | FSp2 | CAL（最优，起点 FSp2） | CAL 单位盒 | 起点 FS 的最优 | 其单位盒 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for k in G.CLOSURE8:
        lo, hi, sc = cl["bounds"][k]
        w(f"| {k} | {lo:g} | {hi:g} | {sc} | {cl['FS']['values'][k]:.4g} | {cl['FSp2']['values'][k]:.4g} | "
          f"{cl['CAL']['values'][k]:.4g} | {cl['CAL']['unit'][k]:.3f} | {cl['start0_best']['values'][k]:.4g} | "
          f"{cl['start0_best']['unit'][k]:.3f} |")
    w(f"\nν = {G.FIXED['nu']:g}、D = {G.FIXED['D']:g} 固定。\n")
    w("## 7. FS 256² 场复用核对\n")
    w("| 图案 | 与 v05allphys/final/runs/FS_* 逐位相同 |")
    w("|---|---|")
    for p in G.PATTERNS:
        w(f"| {CN[p]} | {'是' if E['repro'][p]['bitwise_equal'] else '否'} |")
    w("")
    if "ident" in E:
        I = E["ident"]
        near = [k for k in I["names"] if k.startswith("near_")]
        w(f"## 8. 可辨识性（N = {I['N']}）：校准损失几乎相同的 {len(near)} 个闭合对六个图（四检验 + 两校准）的预测\n")
        w(f"near_k = 全部 {I['n_cal_samples']} 个 CMA 样本（两条 CMA 共 960 次调用）里校准损失最低的 8 个（near_1 = CAL，"
          "near_3 = 起点 FS 的最优）；损失只差 ≤ 0.041。")
        w("单位盒坐标：")
        w("")
        w("| 闭合 | 校准损失 | 起点 | " + " | ".join(G.CLOSURE8) + " |")
        w("|---|---|---|" + "---|" * len(G.CLOSURE8))
        for k in near:
            s = I["set"][k]
            w(f"| {k} | {s['cal_loss']:.3f} | {'FS' if s['start'] == 0 else 'FSp2'} | "
              + " | ".join(f"{s['unit'][q]:.2f}" for q in G.CLOSURE8) + " |")
        w("")
        w("| 图案 | L_fin：8 个闭合的范围 | L_fin：FS | 8 个闭合两两 IoU 均值 / 最小 | 8 个闭合对 FS 的 IoU（范围） | 白区比（范围） "
          "| 接收奶量 [mL]：8 个闭合 / FS |")
        w("|---|---|---|---|---|---|---|")
        for p in G.TEST + G.NAMES:
            sp = I["spread"][p]
            w(f"| {CN[p]} | {sp['L_fin_range'][0]:.3f} – {sp['L_fin_range'][1]:.3f} | {sp['L_fin_FS']:.3f} | "
              f"{sp['pairwise_iou_near']['mean']:.2f} / {sp['pairwise_iou_near']['min']:.2f} | "
              f"{min(sp['iou_near_vs_FS']):.2f} – {max(sp['iou_near_vs_FS']):.2f} | "
              f"{min(sp['white_ratio_near']):.2f} – {max(sp['white_ratio_near']):.2f} | "
              f"{min(I['per'][p][k]['V_received_mL'] for k in near):.1f} – "
              f"{max(I['per'][p][k]['V_received_mL'] for k in near):.1f} / {I['per'][p]['FS']['V_received_mL']:.1f} |")
        w("")
        w("接收奶量 Q = clip(F / κ_Q, 1, 32 mL/s)：8 个闭合的 κ_Q 是 1.9–4.9 /D，所以同一套奶壶流量 F 下，它们收到的奶相差 "
          "2.6 倍。L_fin 看不到奶量；'动作固定'只对奶壶流量 F 成立。")
        w("")
        w("逐闭合 L_fin（N = 128）：\n")
        w("| 闭合 | " + " | ".join(CN[p] for p in G.TEST + G.NAMES) + " | 检验均值 |")
        w("|---|" + "---|" * (len(G.TEST) + len(G.NAMES) + 1))
        for k in near + ["FS"]:
            Ls = [I["per"][p][k]["L_fin"] for p in G.TEST + G.NAMES]
            w(f"| {k} | " + " | ".join(f"{x:.3f}" for x in Ls) + f" | {np.mean(Ls[:4]):.3f} |")
        w("")
    cj = os.path.join(os.path.dirname(path), "chi_strokes.json")
    if os.path.exists(cj):
        X = json.load(open(cj))
        w("## 9. 逐笔沉积分数 χ（求解器收到的入口，S_eff 加权；chi_strokes.py）\n")
        w("FS 的每笔壶嘴高度是按 FS 闭合反设计的（让每笔 χ 配 V0.5）；动作冻结后换成 CAL 闭合，同样的笔 χ 变成：")
        w("")
        w("| 图案 | 每笔 χ：FS → CAL（# 笔号，类） |")
        w("|---|---|")
        for p in G.PATTERNS:
            w(f"| {CN[p]} | " + "；".join(f"#{k} {X[p]['FS'][k]['cls']} {X[p]['FS'][k]['chi']:.2f}→{X[p]['CAL'][k]['chi']:.2f}"
                                          for k in X[p]["FS"]) + " |")
        w("\nCAL 的沉积开关比 FS 平（p_dep 1.79 对 8，B_dep 1921 对 1550；CAL 的起点 FSp2 是 p_dep 2，"
          "只有起点 FS 的那条 CMA 是从 8 降到 1.29）。同样的笔：")
        w("- '轻柔全沉积'的底座笔（FS 0.94–0.99）降到 0.73–0.81，少沉积约 20 %。")
        w("- 摆动底座，设计目标 0.335（各图 #1、郁金香 #3；这些笔已在 2.50 cm 的高度上界）：FS 0.31–0.34 升到 0.51–0.52。")
        w("- 摆动底座，设计目标 0.52（#0）：FS 0.51 升到 0.57。")
        w("- 细切（FS 0.00–0.02）升到 0.30–0.34，大白心的细切从 0.14 升到 0.40。")
        w("")
        w("也就是说，同一套动作在 CAL 闭合下，底座变薄，摆动和细切都在往液面铺白：细切画出来的不再是褐线，而是细白条。")
        w("κ_r 同时从 0.037 升到 0.172（4.6 倍），冲击点的径向回流更强。")
        w("")
        w("这些 χ 变化不是大白心变空心的原因：单换参数的检验（`ablate/`，N = 128 和 256²）里，FS 只换 (B_dep, p_dep) 时"
          "底座同样少沉积（256² 大白心沉积 −19 %，CAL −21 %），大白心仍是实心；FS 只把 β 换成 1.41 /s 时 χ 一笔不变"
          "（χ 的公式里没有 β），大白心就从凹口往下出现褐色空心楔形（N = 128 整颗空心，256² 空了上半）。"
          "沉积开关的变化主要改变白区大小：千层心、郁金香、树叶的白变大。\n")
    w("## 10. 目判（看 final_vs_real.png、FS_vs_CAL.png、ident.png、blind/；非盲：判的人知道哪张是哪组）\n")
    w("| 图案 | FS | CAL | 说明 |")
    w("|---|---|---|---|")
    for p in G.PATTERNS:
        a, b, t = EYE[p]
        w(f"| {CN[p]}（{'校准' if p in G.NAMES else '检验'}） | {a} | {b} | {t} |")
    w("")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    if "--tables" in sys.argv:                     # rewrite tables.md from eval.json (no recomputation)
        _E = json.load(open(os.path.join(os.path.abspath(sys.argv[1]), "eval", "eval.json")))
        _E["ident"]["n_cal_samples"] = json.load(open(os.path.join(os.path.abspath(sys.argv[1]), "eval",
                                                                   "ident_set.json")))["n_cal_samples"]
        G._jdump(_E, os.path.join(os.path.abspath(sys.argv[1]), "eval", "eval.json"))
        tables(_E, os.path.join(os.path.abspath(sys.argv[1]), "eval", "tables.md"))
    else:
        main()
