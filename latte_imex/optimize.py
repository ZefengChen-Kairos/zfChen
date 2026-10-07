"""CMA-ES over Zefeng's parametric controls (controls.py) against the reference targets (targets.py).

    python -m latte_imex.optimize heart --N 128 --gens 40 --workers 2

Loss per candidate = L1 distance between Gaussian-blurred whiteness fields (sim vs target, inside the
cup) + 0.5 * (1 - IoU of the thresholded fields).  The simulation runs the full programme at N and
returns the final milk fraction c; s = clip(c/0.6, 0, 1) is compared with the target whiteness w.
"""
import argparse, json, os, sys, time, math, traceback
import numpy as np
from multiprocessing import Pool
from scipy import ndimage
from . import controls, targets
from .solver import Solver, Params, Numerics


def simulate(prog, physics, N, T_max=None):
    """Full programme at resolution N, no snapshots. Returns (c, mask, stats)."""
    P = Params(**physics, return_law="v05")
    num = Numerics(N=N)
    sol = Solver(P, num)
    T = prog.duration if T_max is None else min(T_max, prog.duration)
    while sol.t < T - 1e-9:
        probe = prog(sol.t)
        fdt = num.frame_dt
        if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
            fdt = min(fdt, max(num.scan_safety * min(sol.g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
        inl = prog(sol.t + 0.5 * fdt)
        sol.advance_frame(inl, frame_dt=fdt)
    return sol.concentration(), sol.g.mask, sol.stats


def whiteness_sim(c, mask, c_sat=0.6):
    s = np.clip(np.nan_to_num(c, nan=0.0) / c_sat, 0, 1)
    return np.where(mask, s, 0.0)


def loss_fields(s, w, mask, sigma):
    Gs = ndimage.gaussian_filter(s, sigma); Gw = ndimage.gaussian_filter(w, sigma)
    l1 = float(np.abs(Gs - Gw)[mask].mean())
    a, b = (s > 0.5) & mask, (w > 0.5) & mask
    union = float((a | b).sum()); iou = float((a & b).sum() / union) if union > 0 else 0.0
    area = float(abs(a.sum() - b.sum()) / max(mask.sum(), 1))
    return l1 + 0.5 * (1 - iou) + 0.5 * area, dict(l1=l1, iou=iou, area_diff=area)


def target_at(name, N):
    w, m = targets.load(name)
    M = w.shape[0]
    wN = ndimage.zoom(w, N / M, order=1)
    return np.clip(wN, 0, 1)


def evaluate(args):
    name, x, N, physics, sigma = args
    t0 = time.time()
    try:
        prog, cfg, p = controls.build(name, x, physics=physics)
        c, mask, stats = simulate(prog, physics, N)
        if not np.all(np.isfinite(c[mask])):
            raise FloatingPointError("non-finite c")
        s = whiteness_sim(c, mask)
        w = target_at(name, N)
        L, parts = loss_fields(s, w, mask, sigma)
        return dict(loss=L, ok=True, elapsed=time.time() - t0, T=prog.duration, retries=stats["retries"],
                    steps=stats["steps"], **parts)
    except Exception as e:  # noqa
        return dict(loss=10.0, ok=False, elapsed=time.time() - t0, err=f"{type(e).__name__}: {e}")


def compare_png(path, c, mask, w, title=""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from . import render
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2), dpi=100)
    ax[0].imshow(render.c_to_rgb(c, mask), interpolation="bilinear"); ax[0].set_title(f"IMEX sim  {title}")
    s = whiteness_sim(c, mask)
    ax[1].imshow(s[::-1], cmap="gray", vmin=0, vmax=1); ax[1].set_title("sim whiteness s = clip(c/0.6)")
    ax[2].imshow(w[::-1], cmap="gray", vmin=0, vmax=1); ax[2].set_title("target whiteness w (reference clip)")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--popsize", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--blur", type=float, default=0.015, help="Gaussian sigma in cup diameters")
    ap.add_argument("--x0", default=None, help="json file with a starting physical parameter dict")
    a = ap.parse_args()
    import cma
    name = a.name; sp = controls.SPACES[name]
    out = a.out or f"runs/opt/{name}"; os.makedirs(out, exist_ok=True)
    physics = dict(controls.V05_PHYSICS)
    sigma = a.blur * a.N
    x0 = 0.5 * np.ones(sp.n)
    if a.x0:
        with open(a.x0) as f:
            x0 = sp.encode(json.load(f))
    opts = {"bounds": [0, 1], "seed": a.seed, "verbose": -9}
    if a.popsize:
        opts["popsize"] = a.popsize
    es = cma.CMAEvolutionStrategy(x0, a.sigma0, opts)
    pool = Pool(a.workers)
    best = dict(loss=1e9); history = []
    w = target_at(name, a.N)
    print(f"[{name}] n={sp.n} popsize={es.popsize} N={a.N} workers={a.workers} out={out}", flush=True)
    t_start = time.time()
    for g in range(a.gens):
        if os.path.exists(os.path.join(out, "STOP")):
            print("STOP file found", flush=True); break
        X = es.ask()
        res = pool.map(evaluate, [(name, x, a.N, physics, sigma) for x in X])
        es.tell(X, [r["loss"] for r in res])
        k = int(np.argmin([r["loss"] for r in res]))
        history.append(dict(gen=g, best=res[k]["loss"], mean=float(np.mean([r["loss"] for r in res])),
                            elapsed=time.time() - t_start, fails=sum(not r["ok"] for r in res)))
        if res[k]["loss"] < best["loss"]:
            best = dict(res[k]); best["x"] = [float(v) for v in X[k]]; best["gen"] = g
            prog, cfg, p = controls.build(name, X[k], physics=physics)
            best["params"] = p; best["config"] = cfg
            with open(os.path.join(out, "best.json"), "w") as f:
                json.dump(best, f, indent=1)
            c, mask, stats = simulate(prog, physics, a.N)
            compare_png(os.path.join(out, "best.png"), c, mask, w, title=f"gen {g} loss {best['loss']:.3f} N={a.N}")
        with open(os.path.join(out, "history.json"), "w") as f:
            json.dump(history, f)
        errs = [r.get("err") for r in res if not r["ok"]]
        print(f"[{name}] gen {g:3d} best {res[k]['loss']:.4f} (iou {res[k].get('iou', 0):.3f}) mean {history[-1]['mean']:.4f} "
              f"global {best['loss']:.4f}@{best['gen']} sigma {es.sigma:.3f} eval {np.mean([r['elapsed'] for r in res]):.0f}s "
              f"total {time.time()-t_start:.0f}s" + (f" errs {errs[:1]}" if errs else ""), flush=True)
    pool.close()
    print(f"[{name}] done. best loss {best['loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
