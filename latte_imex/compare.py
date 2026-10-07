"""Side-by-side figure: optimised IMEX result | sim whiteness | target whiteness | reference frame, for several patterns.

    python -m latte_imex.compare runs/zefeng_256 results/zefeng_opt/compare.png
"""
import json, os, sys, numpy as np
from . import targets, render
from .optimize import whiteness_sim, loss_fields
from scipy import ndimage

ORDER = ["heart", "push_heart", "layered_heart", "tulip", "leaf", "swan"]
CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="压纹郁金香", leaf="树叶", swan="天鹅")


def main(root, out, blur=0.015):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [p for p in ORDER if os.path.exists(os.path.join(root, p, "final_state.npz"))]
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)), dpi=100, squeeze=False)
    scores = {}
    for i, p in enumerate(rows):
        z = np.load(os.path.join(root, p, "final_state.npz"))
        mask = z["mask"]; c = np.where(mask, z["m"] / np.where(mask, z["l"], 1), np.nan)
        N = mask.shape[0]
        w, _ = targets.load(p); w = np.clip(ndimage.zoom(w, N / w.shape[0], order=1), 0, 1)
        s = whiteness_sim(c, mask)
        L, parts = loss_fields(s, w, mask, blur * N)
        scores[p] = dict(loss=L, **parts)
        frame = targets.load_rgb(os.path.join(targets.TDIR, f"{p}_frame.png"))
        axes[i, 0].imshow(render.c_to_rgb(c, mask), interpolation="bilinear"); axes[i, 0].set_title(f"{p} / {CN[p]}  IMEX N={N} (optimised controls)")
        axes[i, 1].imshow(s[::-1], cmap="gray", vmin=0, vmax=1); axes[i, 1].set_title(f"sim whiteness   loss {L:.3f}  IoU {parts['iou']:.2f}")
        axes[i, 2].imshow(w[::-1], cmap="gray", vmin=0, vmax=1); axes[i, 2].set_title("target whiteness (warped reference frame)")
        axes[i, 3].imshow(frame); axes[i, 3].set_title("reference clip frame")
        for a in axes[i]: a.axis("off")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    json.dump(scores, open(os.path.splitext(out)[0] + "_scores.json", "w"), indent=1)
    for p, sc in scores.items():
        print(f"{p}: loss={sc['loss']:.3f} iou={sc['iou']:.3f} l1={sc['l1']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
