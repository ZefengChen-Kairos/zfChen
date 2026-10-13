"""struct.py -- a few structure numbers in support of the by-eye judgements (canonical frame, target grid, raw alignment).

  components   white connected components (sim c > 0.4, video w > 0.5) with area >= 4 mm^2, sorted by area [mm^2]
  stripes      brown lines crossing the lines x = -0.10, -0.05, 0, +0.05, +0.10 D: cf_runner.axis_gaps gaps (>= 1 mm, c below
               threshold) + dips (prominence >= 0.1, FWHM >= 0.5 mm) inside the white extent, per line
  heart_axis   (heart, layered_heart) along x = 0 from the top of the lobes down to the tip: notch depth = first white on the
               axis below the lobe top [mm]; brown on axis = total non-white length between the first and last white on the
               axis [mm]; vertical extent of white [mm]
"""
import json, os, sys
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare as C  # noqa: E402
CF = C.CF
D_MM = 80.0
LINES = (-0.10, -0.05, 0.0, 0.05, 0.10)


def components(white, Mt):
    lab, n = ndimage.label(white)
    if n == 0:
        return []
    a = ndimage.sum(white, lab, range(1, n + 1)) * (D_MM / Mt) ** 2
    return sorted([round(float(x), 1) for x in a if x >= 4.0], reverse=True)


def stripes(f, wm, thr):
    g = CF.axis_gaps(np.where(wm, f, np.nan), wm, thr, lines=LINES)
    return [dict(x=L["x"], gaps=L["n_gaps"], dips=L["n_dips"], total=L["n_gaps"] + L["n_dips"]) for L in g["lines"]]


def heart_axis(f, wm, thr):
    Mt = f.shape[0]
    h = D_MM / Mt
    white = (f > thr) & wm
    g = (np.arange(Mt) + 0.5) / Mt - 0.5
    cols = np.abs(g) <= 0.25
    rows_any = np.where(white[:, cols].any(1))[0]
    if len(rows_any) == 0:
        return None
    top = rows_any.min()                                   # top of the lobes (row 0 = top = B side)
    prof = CF.sample(CF.nearest_fill(np.nan_to_num(f), wm), Mt, np.zeros(Mt * 10), np.linspace(-0.49, 0.49, Mt * 10))
    yy = np.linspace(-0.49, 0.49, Mt * 10)                 # T frame y (down in the display)
    wy = prof > thr
    idx = np.where(wy)[0]
    if len(idx) == 0:
        return None
    y_top_lobe = g[top]
    first, last = yy[idx[0]], yy[idx[-1]]
    seg = wy[idx[0]:idx[-1] + 1]
    step = (yy[1] - yy[0]) * D_MM
    return dict(notch_depth_mm=round(float((first - y_top_lobe) * D_MM), 1), brown_on_axis_mm=round(float((~seg).sum() * step), 1),
                axis_white_extent_mm=round(float((last - first) * D_MM), 1), lobe_top_to_tip_mm=round(float((last - y_top_lobe) * D_MM), 1))


def main():
    out = {}
    for p in C.PAT:
        z = np.load(os.path.join(HERE, f"fields_{p}.npz"))
        w, wm = z["w"].astype(float), z["wm"]
        Mt = w.shape[0]
        rec = dict(video=dict(components=components((w > 0.5) & wm, Mt), stripes=stripes(w, wm, 0.5)))
        if p in ("heart", "layered_heart"):
            rec["video"]["heart_axis"] = heart_axis(w, wm, 0.5)
        for k in ("V05", "P1", "PHYS"):
            f = z[f"{k}_raw"].astype(float)
            r = dict(components=components((f > 0.4) & wm, Mt), stripes=stripes(f, wm, 0.4))
            if p in ("heart", "layered_heart"):
                r["heart_axis"] = heart_axis(f, wm, 0.4)
                r["heart_axis_c05"] = heart_axis(f, wm, 0.5)
            rec[k] = r
        out[p] = rec
        print(p, json.dumps({k: (v["components"][:6], [s["total"] for s in v["stripes"]], v.get("heart_axis")) for k, v in rec.items()}))
    json.dump(out, open(os.path.join(HERE, "struct.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
