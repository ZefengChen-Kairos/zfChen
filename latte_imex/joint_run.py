"""Evaluate one named (physics, actions) candidate of the joint chain at a chosen grid and save fields + figure.

    python -m latte_imex.joint_run --from runs/joint_sweep_r2fix/results.json --idx 7 --N 256 --out results/joint_sweep/r2fix_07_N256
    python -m latte_imex.joint_run --physics '{"cp":0.3,...,"kQ":1.8}' --actions-from runs/joint_sweep_r2fix/results.json --idx 7 --N 96 --out ...

The candidate is run through pitcher -> ledger -> IMEX for the heart and the tulip (same chain as joint_sweep),
scored against the video targets, and saved as fields.npz, result.json and a 1x3 figure (target, tulip, heart).
"""
import argparse, json, os
import numpy as np

from .joint_sweep import evaluate
from .optimize import target_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=None, help="results.json of a sweep (physics and actions of --idx)")
    ap.add_argument("--actions-from", default=None, help="results.json whose --idx supplies only the actions")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--physics", default=None, help="JSON dict overriding the physics vector (partial allowed)")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    src = a.src or a.actions_from
    rec = json.load(open(src))[a.idx] if src else dict(physics={}, actions=None)
    physics = dict(rec.get("physics", {}))
    actions = rec.get("actions")
    if a.physics:
        physics.update(json.loads(a.physics))
    if actions is None:
        raise SystemExit("need --from or --actions-from for the actions")
    os.makedirs(a.out, exist_ok=True)
    r = evaluate((a.idx, physics, actions, a.N))
    fields = {name: p.pop("c") for name, p in r["patterns"].items() if "c" in p}
    np.savez_compressed(os.path.join(a.out, "fields.npz"), **fields)
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(r, f, indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .optimize import whiteness_sim
    N = a.N
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    brown, white = np.array([92, 54, 30]) / 255, np.array([246, 240, 226]) / 255

    def show(k, img, mask, title):
        rgb = brown[None, None, :] * (1 - img[..., None]) + white[None, None, :] * img[..., None]
        rgb[~mask] = 0.15
        ax[k].imshow(rgb, origin="lower"); ax[k].set_title(title, fontsize=9); ax[k].axis("off")
    mask = np.hypot(*np.meshgrid(*(2 * [(np.arange(N) + 0.5) / N - 0.5]))) < 0.49
    show(0, target_at("tulip", N), mask, "target tulip")
    show(1, target_at("heart", N), mask, "target heart")
    for k, name in enumerate(("tulip", "heart")):
        p = r["patterns"][name]
        c = fields[name].astype(float)
        show(2 + k, whiteness_sim(c, mask), mask, f"{name} N={N} loss {p.get('loss', 9):.3f} IoU {p.get('iou', 0):.2f}")
    fig.suptitle(a.title or f"{src} idx {a.idx} physics {json.dumps({k: round(v, 4) for k, v in physics.items()})}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "final.png"), dpi=110)
    for name, p in r["patterns"].items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()})


if __name__ == "__main__":
    main()
