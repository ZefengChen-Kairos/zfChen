"""Small physics fit of the two-layer closure (the old chi closure #9 came out of a joint sweep; the two-layer
constants were only estimated), so that the two closures are compared on equal terms.

    python -m latte_imex.twolayer_fit --n 30 --N 96 --out runs/action_opt     # LHS over the closure constants
    python -m latte_imex.twolayer_fit --actions --N 96 --out runs/action_opt   # action search with the best fit

Free: g_red (reduced gravity of the foam), Fr_c2 (critical Froude^2 of foam survival), phi_foam (foam fraction of
the milk), beta (film drag towards the coffee layer), kappa_c (jet impact traction).  The rest is #9.
Actions: the V0.5 intents (h = q = s = 1).  Objective: mean loss over the six patterns.
Each sample is cached as model label fitNN under <out>.
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np

from .action_opt import run_one, PATTERNS, search, refine
from .joint_sweep import sample, lhs

SPACE = dict(g_red=(25.0, 150.0, "log"), Fr_c2=(5.0, 100.0, "log"), phi_foam=(0.4, 1.0, "lin"),
             beta=(1.0, 10.0, "log"), kappa_c=(0.3, 2.0, "log"))
BASE = dict(closure="twolayer", ent_coef=0.0)
EST = dict(g_red=75.0, Fr_c2=23.0, phi_foam=1.0)        # the estimated constants (model "new"), beta/kappa_c from #9


def candidates(n, seed=7):
    U = lhs(n, len(SPACE), np.random.default_rng(seed))
    return [dict(BASE, **sample(SPACE, u)) for u in U]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--out", default="runs/action_opt")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--actions", action="store_true", help="action search (round 1 + 2) with the best fit")
    a = ap.parse_args()
    fit_path = os.path.join(a.out, f"twolayer_fit_N{a.N}.json")
    if not a.actions:
        C = candidates(a.n)
        jobs = [(f"fit{i:02d}", n, 1.0, 1.0, 1.0, a.N, a.out, c) for i, c in enumerate(C) for n in PATTERNS]
        with Pool(a.procs, maxtasksperchild=4) as pool:
            R = pool.map(run_one, jobs)
        rows = []
        for i, c in enumerate(C):
            rs = [r for r in R if r["model"] == f"fit{i:02d}"]
            rows.append(dict(i=i, over=c, mean=float(np.mean([r["loss"] for r in rs])),
                             per={r["name"]: round(r["loss"], 4) for r in rs}))
        rows.sort(key=lambda r: r["mean"])
        json.dump(dict(best=rows[0], rows=rows), open(fit_path, "w"), indent=1)
        for r in rows[:8]:
            print(f"fit{r['i']:02d} mean {r['mean']:.3f}", {k: round(v, 3) for k, v in r["over"].items() if k in SPACE}, r["per"])
        return
    import latte_imex.action_opt as ao
    ao.MODELS["fit"] = json.load(open(fit_path))["best"]["over"]     # set before the pool forks its workers
    with Pool(a.procs, maxtasksperchild=4) as pool:
        best = search(pool, a.N, a.out, ["fit"], PATTERNS)
        json.dump(best, open(os.path.join(a.out, f"fit_search_N{a.N}.json"), "w"), indent=1)
        best = refine(pool, a.N, a.out, best)
        json.dump(best, open(os.path.join(a.out, f"fit_refine_N{a.N}.json"), "w"), indent=1)
    for k, v in best.items():
        b, z = v["best"], v["base"]
        print(f"{k:20s} base {z['loss']:.3f}/{z['iou']:.2f} -> best {b['loss']:.3f}/{b['iou']:.2f}  h {b['h']} q {b['q']} s {b['s']}")


if __name__ == "__main__":
    main()
