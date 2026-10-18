"""Gradient-based calibration of the V0.5 closure with the differentiable Warp solver.

Parameters: v05_adapter.SPACE (8 closure parameters) in the unit box (log-scaled where SPACE says 'log'); nu and D
fixed.  Each iteration runs every calibration pattern forward and backward (exact adjoint, run.DiffPour.loss_grad),
averages the soft-IoU losses and their gradients and takes an Adam step in the unit box (clipped to [0, 1]).  The
soft-IoU temperature is annealed from --tau0 to --tau1.  Every iteration also scores the final fields with the hard
metrics of the Sobol fit (tulip_feas.Target.score: IoU, gap IoU, J); the best iterate by mean J is kept.
Start: --init a closure json (e.g. results/fit/best_closure.json from fit_pipeline, so the gradient refines the
global Sobol search), default the S1 closure.  A failed forward (non-finite state) halves the step and retries.

Outputs in --out: records.jsonl (one line per iteration), best_closure.json (fit_pipeline render reads it),
progress.png, sheet.png (target | start | best), fields/<p>_{start,best}.npy.
    python -m latte_imex.diff.fit --out results/diff_fit [--cal tulip,heart --N 128 --iters 60 --init ...]
    python -m latte_imex.rig.fit_pipeline render --out results/diff_fit          # movies with the result
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from ..v05_adapter import SPACE
from . import wsolver as W

KEYS = tuple(SPACE)


def to_unit(P):
    u = []
    for k in KEYS:
        lo, hi, sc = SPACE[k]
        v = float(P[k])
        u.append(math.log(v / lo) / math.log(hi / lo) if sc == "log" else (v - lo) / (hi - lo))
    return np.clip(np.array(u), 0.0, 1.0)


def from_unit(u, base):
    P = dict(base)
    for k, x in zip(KEYS, np.clip(u, 0.0, 1.0)):
        lo, hi, sc = SPACE[k]
        P[k] = lo * (hi / lo) ** x if sc == "log" else lo + (hi - lo) * x
    return P


def dtheta_du(P):
    """d theta / d u per key (chain rule from the unit box)."""
    out = []
    for k in KEYS:
        lo, hi, sc = SPACE[k]
        out.append(float(P[k]) * math.log(hi / lo) if sc == "log" else hi - lo)
    return np.array(out)


def score(D, final):
    msk = D.geo.mask_np
    c = np.where(msk, final["m"] / np.where(msk, final["l"], 1.0), np.nan)
    M, _, _ = D.T.score(c)
    return M, c


def sheet(out, Ds, cal, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from ..rig.tulip_feas import _rot
    cm = LinearSegmentedColormap.from_list("milk", [(0.36, 0.20, 0.09), (0.66, 0.47, 0.30), (1.0, 0.97, 0.92)])
    cm.set_bad((0.12, 0.12, 0.12))
    fig, ax = plt.subplots(3, len(cal), figsize=(3.3 * len(cal), 10.2), dpi=100, squeeze=False)
    for j, p in enumerate(cal):
        T = Ds[p].T
        ax[0, j].imshow(_rot(T.img[..., ::-1].copy(), T.rot))
        ax[0, j].contour(_rot(T.G.astype(np.float32), T.rot), [0.5], colors="cyan", linewidths=0.7)
        ax[0, j].set_title(f"{p} target", fontsize=9)
        for i, tag in ((1, "start"), (2, "best")):
            c = np.load(os.path.join(out, "fields", f"{p}_{tag}.npy"))
            M, cs, G = T.score(c)
            msk = _rot(np.isfinite(cs).astype(np.float32), T.rot) > 0.5
            ax[i, j].imshow(np.ma.masked_array(_rot(np.nan_to_num(cs).astype(np.float32), T.rot), ~msk), cmap=cm, vmin=0,
                            vmax=0.8)
            ax[i, j].contour(_rot(G.astype(np.float32), T.rot), [0.5], colors="cyan", linewidths=0.7)
            ax[i, j].set_title(f"{tag}: IoU {M['iou']:.2f} gap {M['gap_iou']:.2f}", fontsize=9)
        for i in range(3):
            ax[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sheet.png"))
    plt.close(fig)


def progress(out, recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    it = [r["it"] for r in recs]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=100)
    ax[0].plot(it, [r["loss"] for r in recs], "k.-"); ax[0].set_title("soft-IoU loss (mean)")
    for p in recs[0]["per"]:
        ax[1].plot(it, [r["per"][p]["J"] for r in recs], ".-", label=f"{p} J")
    ax[1].plot(it, [r["J"] for r in recs], "k.-", label="mean J"); ax[1].legend(fontsize=7); ax[1].set_title("IoU + gap IoU")
    U = np.array([r["u"] for r in recs])
    for i, k in enumerate(KEYS):
        ax[2].plot(it, U[:, i], ".-", label=k)
    ax[2].legend(fontsize=6, ncol=2); ax[2].set_title("parameters (unit box)"); ax[2].set_ylim(-0.02, 1.02)
    for x in ax:
        x.grid(alpha=0.3); x.set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "progress.png"))
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/diff_fit")
    ap.add_argument("--cal", default="tulip,heart")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.04, help="Adam step in the unit box")
    ap.add_argument("--tau0", type=float, default=0.10)
    ap.add_argument("--tau1", type=float, default=0.03)
    ap.add_argument("--w-gap", dest="w_gap", type=float, default=2.0)
    ap.add_argument("--init", default=None, help="closure json to start from (default: S1)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--T", type=float, default=None, help="simulate only the first T seconds (tests)")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "fields"), exist_ok=True)
    log = lambda *x: print(*x, flush=True)
    from .run import DiffPour
    cal = a.cal.split(",")
    Ds = {p: DiffPour(p, N=a.N, device=a.device, w_gap=a.w_gap, tau=a.tau0, T_max=a.T, log=log) for p in cal}
    base = dict(Ds[cal[0]].s1)
    if a.init:
        base.update({k: float(v) for k, v in json.load(open(a.init)).items() if k in base})
        log(f"start: {a.init}")
    u = to_unit(base)
    m1, m2 = np.zeros_like(u), np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr = a.lr
    recs, best = [], None
    fn = os.path.join(a.out, "records.jsonl")
    open(fn, "w").close()
    it = 0
    while it < a.iters:
        tau = a.tau0 * (a.tau1 / a.tau0) ** (it / max(a.iters - 1, 1))
        P = from_unit(u, base)
        t0 = time.time()
        try:
            res = {}
            for p, D in Ds.items():
                D.tau = tau
                L, g, final = D.loss_grad(P)
                M, c = score(D, final)
                res[p] = (L, g, M, c)
        except RuntimeError as e:                       # non-finite state: back off
            log(f"it {it}: forward failed ({e}); halving the step")
            lr *= 0.5
            u = u_prev
            continue
        L = float(np.mean([r[0] for r in res.values()]))
        gth = np.mean([[r[1][k] for k in KEYS] for r in res.values()], axis=0)
        gu = gth * dtheta_du(P)
        J = float(np.mean([r[2]["J"] for r in res.values()]))
        rec = dict(it=it, loss=L, J=J, tau=tau, lr=lr, u=u.tolist(), params={k: P[k] for k in KEYS}, grad_u=gu.tolist(),
                   per={p: dict(loss=r[0], iou=r[2]["iou"], gap_iou=r[2]["gap_iou"], J=r[2]["J"]) for p, r in res.items()},
                   wall_s=time.time() - t0)
        recs.append(rec)
        with open(fn, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"it {it:3d}  loss {L:.4f}  J {J:.3f}  " + "  ".join(f"{p} {r[2]['iou']:.2f}/{r[2]['gap_iou']:.2f}" for p, r in res.items())
            + f"  |g| {np.linalg.norm(gu):.2e}  {rec['wall_s']:.0f} s ("
            + ", ".join(f"{p} fwd {D.timing['forward_s']:.0f} + bwd {D.timing['backward_s']:.0f} s" for p, D in Ds.items()) + ")")
        if it == 0:
            for p, r in res.items():
                np.save(os.path.join(a.out, "fields", f"{p}_start.npy"), r[3].astype(np.float32))
        if best is None or J > best["J"]:
            best = rec
            for p, r in res.items():
                np.save(os.path.join(a.out, "fields", f"{p}_best.npy"), r[3].astype(np.float32))
            json.dump(dict(P), open(os.path.join(a.out, "best_closure.json"), "w"), indent=1)
        # Adam (descent on the loss) in the unit box
        m1 = b1 * m1 + (1 - b1) * gu
        m2 = b2 * m2 + (1 - b2) * gu ** 2
        mh, vh = m1 / (1 - b1 ** (it + 1)), m2 / (1 - b2 ** (it + 1))
        u_prev = u.copy()
        u = np.clip(u - lr * mh / (np.sqrt(vh) + eps), 0.0, 1.0)
        it += 1
        progress(a.out, recs)
    sheet(a.out, Ds, cal, log)
    log(f"best: iteration {best['it']}, mean J {best['J']:.3f} (start {recs[0]['J']:.3f}); "
        + " ".join(f"{k} {best['params'][k]:.4g}" for k in KEYS))
    log(f"-> {a.out}/best_closure.json, progress.png, sheet.png")


if __name__ == "__main__":
    main()
