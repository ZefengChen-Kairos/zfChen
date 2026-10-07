"""Collect the six final runs: overview image, summary table (json + markdown rows)."""
import json, os, numpy as np
from . import render

ORDER = ["heart", "push_heart", "layered_heart", "tulip", "leaf", "swan"]
CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="压纹郁金香", leaf="树叶", swan="天鹅")


def main(root="runs/final"):
    items, rows = [], []
    for p in ORDER:
        d = os.path.join(root, p)
        z = np.load(os.path.join(d, "final_state.npz"))
        c = np.where(z["mask"], z["m"] / np.where(z["mask"], z["l"], 1), np.nan)
        items.append((f"{p} / {CN[p]}", c, z["mask"]))
        m = json.load(open(os.path.join(d, "metrics.json")))
        st, L = m["stats"], m["ledger"]
        rows.append(dict(pattern=p, cn=CN[p], T=m["T"], steps=st["steps"], frames=st["frames"],
                         max_sub=st["max_substeps"], retries=st["retries"], max_u=st["max_u"],
                         cg_mass_per_step=st["cg_mass"] / st["steps"], wall_s=m["elapsed_s"],
                         deposited=L["deposited"], layer_err=L["layer_error"], milk_err=L["milk_error"],
                         clip=st["clip_mass"], white=m["white_area_c_gt_0p4"], lmax=float(z["l"][z["mask"]].max())))
    render.overview(os.path.join(root, "six_patterns.png"), items, ncols=3)
    json.dump(rows, open(os.path.join(root, "summary.json"), "w"), indent=1, ensure_ascii=False)
    for r in rows:
        print(f"{r['cn']}({r['pattern']}): T={r['T']:.2f}s steps={r['steps']} max_sub={r['max_sub']} retries={r['retries']} "
              f"max|u|={r['max_u']:.2f} cg/step={r['cg_mass_per_step']:.1f} wall={r['wall_s']:.0f}s dep={r['deposited']:.3f} "
              f"lerr={r['layer_err']:.1e} merr={r['milk_err']:.1e} clip={r['clip']:.1e} white={r['white']:.3f} lmax={r['lmax']:.2f}")


if __name__ == "__main__":
    main()
