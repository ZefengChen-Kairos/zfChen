"""ablate256_analyze.py -- metrics and figure of the 256^2 one-factor swaps (ablate256.py): FS, CAL, FS+beta, FS+dep,
CAL-beta, CAL-dep on the six patterns.  Metrics exactly as eval/analyze.py (compare.py raw / best IoU c > 0.4 and
boundary distance, the pre-registered L_fin, the section 3.1 automatic structure verdict); the FS and CAL fields are the
evaluation runs of the main cache.  Writes ROOT/ablate256/ablate256_metrics.json and ablate256.png.

usage: cd /home/user/zfChen && NUMBA_NUM_THREADS=1 python3 ablate256_analyze.py ROOT     (ROOT = review/v05gcal)
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    root = os.path.abspath(sys.argv[1])
    AN = _load("v05gcal_analyze", os.path.join(root, "eval", "analyze.py"))       # cmp_metrics, lfin_metrics, verdict
    MF = _load("v05gcal_figs", os.path.join(root, "eval", "make_figs.py"))        # show_model, BOXZ, CN (FS_six colours)
    AB = _load("v05gcal_ablate128", os.path.join(root, "review_ablate", "ablate.py"))
    G, C, plt = AN.G, AN.C, MF.plt
    rec = json.load(open(os.path.join(root, "calib", "calib.json")))
    CAL = {k: float(rec["best"]["params"][k]) for k in G.CLOSURE8}
    FS = G.init_points()["FS"]
    for k in G.CLOSURE8:                                    # the swap file holds the same FS / CAL values
        assert AB.FS[k] == FS[k] and AB.CAL[k] == CAL[k], k
    sets = dict(FS=(FS, os.path.join(root, "cache")), CAL=(CAL, os.path.join(root, "cache")))
    sets.update({s: (v, os.path.join(HERE, "cache")) for s, v in AB.SETS.items()})
    vdir = os.path.join(root, "video")
    out = dict(sets={s: v[0] for s, v in sets.items()}, patterns={})
    fields = {}
    for p in G.PATTERNS:
        cref, mref = C.load_sim(C.sim_paths(p)[0]["V05"])
        out["patterns"][p] = {}
        for s, (prm, cdir) in sets.items():
            cM, mM, key = G.load_field(cdir, prm, p, 256)
            fields[(p, s)] = cM
            cm, _, _ = AN.cmp_metrics(cM, mM, p, vdir)
            lf = AN.lfin_metrics(cM, mM, p, vdir)
            r = dict(cache_key=key, compare=cm, lfin=lf)
            r.update(AN.structure(cM, p, cref, mref, mM))
            r["auto_verdict"], r["auto_reason"] = AN.verdict(p, r)
            ev = json.load(open(os.path.join(cdir, f"{key}.json")))["eval"]
            r["deposited"] = ev["ledger"]["deposited"]
            r["V_received_mL"] = ev["inlet"]["V_received_mL"]
            out["patterns"][p][s] = r
            print(p, s, f"L {lf['L_fin']:.3f} IoU {cm['raw']['iou_c04']:.2f}/{cm['best']['iou_c04']:.2f} "
                  f"bnd {cm['raw']['bdist_c04_mm']:.1f}/{cm['best']['bdist_c04_mm']:.1f} dep {r['deposited']:.4f} "
                  f"{r['auto_verdict']} {r['auto_reason']}", flush=True)
    summ = {}
    for s in sets:
        P = out["patterns"]
        summ[s] = dict(cal_loss=float(np.mean([P[p][s]["lfin"]["L_fin"] for p in G.NAMES])),
                       test_L=float(np.mean([P[p][s]["lfin"]["L_fin"] for p in G.TEST])),
                       test_iou_raw=float(np.mean([P[p][s]["compare"]["raw"]["iou_c04"] for p in G.TEST])),
                       test_iou_best=float(np.mean([P[p][s]["compare"]["best"]["iou_c04"] for p in G.TEST])))
    out["summary"] = summ
    G._jdump(out, os.path.join(HERE, "ablate256_metrics.json"))
    cols = list(sets)
    fig, axs = plt.subplots(len(G.PATTERNS), len(cols), figsize=(2.0 * len(cols), 2.15 * len(G.PATTERNS)))
    for i, p in enumerate(G.PATTERNS):
        for j, s in enumerate(cols):
            ax = axs[i, j]
            MF.show_model(ax, fields[(p, s)], MF.BOXZ[p])
            r = out["patterns"][p][s]
            ax.set_xlabel(f"L_fin {r['lfin']['L_fin']:.2f}  IoU {r['compare']['raw']['iou_c04']:.2f}/"
                          f"{r['compare']['best']['iou_c04']:.2f}  {r['auto_verdict']}", fontsize=6.5)
            if i == 0:
                ax.set_title(f"{s}\n校准损失 {summ[s]['cal_loss']:.3f}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{MF.CN[p]} [{MF.ROLE[p]}]", fontsize=8)
    fig.suptitle("单换参数（256²，同一套 FS 动作）：FS+beta = FS 只把 β 换成 CAL 的 1.41；FS+dep = FS 只把 (B_dep, p_dep) 换成 "
                 "CAL 的；CAL-beta、CAL-dep = CAL 把它们换回 FS 的。\n格下：L_fin、IoU 原样/最佳（c > 0.4）、§3.1 自动成形判据；"
                 "FS_six 配色，物理俯视", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(os.path.join(HERE, "ablate256.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
