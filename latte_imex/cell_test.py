"""Two-layer closure with the entrainment cell (surface flow to the jet, brown surface where coffee wells up, only brown
liquid subducted at the jet): six patterns with the fitted constants and the two-layer optimized intents.

    python -m latte_imex.cell_test --N 96 --out runs/action_opt          # four fixed variants
    python -m latte_imex.cell_test --fit --n 40 --N 96                   # LHS refit, objective loss + gap-fraction error
"""
import argparse, json, math, os
from multiprocessing import Pool
from .action_opt import run_one, PATTERNS

VARIANTS = {"cellE074C100": dict(ent_coef=0.74, cell_frac=1.0), "cellE074C050": dict(ent_coef=0.74, cell_frac=0.5),
            "cellE074C025": dict(ent_coef=0.74, cell_frac=0.25), "cellE030C050": dict(ent_coef=0.3, cell_frac=0.5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--out", default="runs/action_opt")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    F = json.load(open(os.path.join(a.out, "twolayer_fit_N96.json")))["best"]["over"]
    A = json.load(open(os.path.join(a.out, "fit_refine_N96.json")))
    jobs = []
    for v in a.variants.split(","):
        for n in PATTERNS:
            b = A[f"fit/{n}"]["best"]
            jobs.append((v, n, b["h"], b["q"], b["s"], a.N, a.out, dict(F, **VARIANTS[v])))
    with Pool(4, maxtasksperchild=4) as pool:
        R = pool.map(run_one, jobs)
    json.dump(R, open(os.path.join(a.out, f"cell_test_N{a.N}.json"), "w"), indent=1)




# ---- refit of the two-layer constants with the entrainment cell, gap-aware objective -------------------------------
SPACE = dict(ent_coef=(0.3, 1.5, "log"), cell_frac=(0.1, 0.5, "log"), beta=(1.0, 10.0, "log"), Fr_c2=(10.0, 100.0, "log"),
             g_red=(40.0, 150.0, "log"), phi_foam=(0.5, 1.0, "lin"), kappa_c=(0.3, 2.0, "log"))


def objective(r, N):
    """loss + |gap fraction - target gap fraction| (the plain loss does not see the brown gaps)."""
    import numpy as np
    from .optimize import visible, target_at, gap_stats
    from .action_opt import tag_of
    mask = np.hypot(*np.meshgrid(*(2 * [(np.arange(N) + 0.5) / N - 0.5]))) < 0.49
    if not r.get("ok"):
        return 10.0, {}
    c = np.load(os.path.join(r["out"], r["model"], f"{r['name']}_{tag_of(r['h'], r['q'], r['s'])}_N{N}.npy"))
    gs, gt = gap_stats(visible(c, mask, r.get("closure")), mask, N), gap_stats(target_at(r["name"], N), mask, N)
    return r["loss"] + abs(gs["gap_frac"] - gt["gap_frac"]), gs


def _outline(img, mask, N, close_mm=6.0, thr=0.5, D_mm=80.0):
    import numpy as np
    from scipy import ndimage
    white = (img > thr) & mask
    r = max(1, int(round(close_mm / D_mm * N / 2))); y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return ndimage.binary_closing(np.pad(white, r), structure=x * x + y * y <= r * r)[r:-r, r:-r] & mask


def struct_stats(img, mask, N):
    """Location-free structure of a whiteness image: interior structure density (mean gradient magnitude inside the
    pattern outline, two cells away from its edge), the whiteness quantiles inside the outline, and the gap fraction.
    Location-free on purpose: a pointwise comparison penalizes misplaced structure twice and therefore prefers a
    smooth blob (a disc with the target outline beats every simulation on band-pass correlation)."""
    import numpy as np
    from scipy import ndimage
    from .optimize import gap_stats
    yy, xx = np.meshgrid(*(2 * [(np.arange(N) + 0.5) / N - 0.5]), indexing="ij")
    inner = np.hypot(xx, yy) < 0.45
    o = _outline(img, mask, N); core = ndimage.binary_erosion(o, iterations=2) & inner
    g = ndimage.gaussian_gradient_magnitude(img, 1.0)
    ed = float(g[core].mean()) if core.sum() > 20 else 0.0
    q = np.quantile(img[o], np.linspace(0.05, 0.95, 10)) if o.sum() > 20 else np.zeros(10)
    return ed, q, gap_stats(img, mask, N)["gap_frac"]


_ST = {}


def objective_struct(r, N):
    """J_A = loss + |gap fraction - target| + 0.5 |ln((e + 0.01)/(e_t + 0.01))| + W1(whiteness inside the outline).
    loss places the pattern; the three statistics ask for the right amount of interior structure, gaps and contrast
    without asking where each line is."""
    import numpy as np
    from .optimize import visible, target_at
    from .action_opt import tag_of
    mask = np.hypot(*np.meshgrid(*(2 * [(np.arange(N) + 0.5) / N - 0.5]))) < 0.49
    if not r.get("ok"):
        return 10.0, {}
    if (r["name"], N) not in _ST:
        _ST[(r["name"], N)] = struct_stats(target_at(r["name"], N), mask, N)
    et, qt, gt = _ST[(r["name"], N)]
    c = np.load(os.path.join(r["out"], r["model"], f"{r['name']}_{tag_of(r['h'], r['q'], r['s'])}_N{N}.npy"))
    ed, q, gf = struct_stats(visible(c, mask, r.get("closure")), mask, N)
    t_gap, t_edge, t_w1 = abs(gf - gt), 0.5 * abs(math.log((ed + 0.01) / (et + 0.01))), float(np.abs(q - qt).mean())
    return r["loss"] + t_gap + t_edge + t_w1, dict(gap_frac=gf, edge=ed, t_gap=t_gap, t_edge=t_edge, t_w1=t_w1)


def fit_main():
    import numpy as np
    from .joint_sweep import sample, lhs
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--out", default="runs/action_opt")
    a = ap.parse_args()
    F = json.load(open(os.path.join(a.out, "twolayer_fit_N96.json")))["best"]["over"]
    A = json.load(open(os.path.join(a.out, "fit_refine_N96.json")))
    U = lhs(a.n, len(SPACE), np.random.default_rng(11))
    C = [dict(F, **sample(SPACE, u)) for u in U]
    jobs = [(f"cf{i:02d}", n, A[f"fit/{n}"]["best"]["h"], A[f"fit/{n}"]["best"]["q"], A[f"fit/{n}"]["best"]["s"], a.N, a.out, c)
            for i, c in enumerate(C) for n in PATTERNS]
    with Pool(4, maxtasksperchild=4) as pool:
        R = pool.map(run_one, jobs)
    rows = []
    for i, c in enumerate(C):
        rs = [dict(r, out=a.out) for r in R if r["model"] == f"cf{i:02d}"]
        per = {r["name"]: objective(r, a.N) for r in rs}
        rows.append(dict(i=i, over=c, J=float(np.mean([v[0] for v in per.values()])), loss=float(np.mean([r["loss"] for r in rs])),
                         per={k: dict(J=round(v[0], 4), **{kk: round(vv, 3) for kk, vv in v[1].items()}) for k, v in per.items()}))
    rows.sort(key=lambda r: r["J"])
    json.dump(dict(best=rows[0], rows=rows), open(os.path.join(a.out, f"cell_fit_N{a.N}.json"), "w"), indent=1)
    for r in rows[:8]:
        print(f"cf{r['i']:02d} J {r['J']:.3f} loss {r['loss']:.3f}", {k: round(v, 3) for k, v in r["over"].items() if k in SPACE},
              {k: v["gap_frac"] for k, v in r["per"].items() if "gap_frac" in v})



def actions_main():
    """Action search (round 1 + 2) with the refitted constants, same gap-aware objective."""
    from . import action_opt as ao
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", action="store_true")
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--out", default="runs/action_opt")
    a = ap.parse_args()
    ao.MODELS["cellfit"] = json.load(open(os.path.join(a.out, f"cell_fit_N{a.N}.json")))["best"]["over"]   # before the fork
    key = lambda r: objective(dict(r, out=a.out), a.N)[0]
    with Pool(4, maxtasksperchild=4) as pool:
        best = ao.search(pool, a.N, a.out, ["cellfit"], PATTERNS, key=key)
        json.dump(best, open(os.path.join(a.out, f"cellfit_search_N{a.N}.json"), "w"), indent=1)
        best = ao.refine(pool, a.N, a.out, best, key=key)
        json.dump(best, open(os.path.join(a.out, f"cellfit_refine_N{a.N}.json"), "w"), indent=1)
    for k, v in best.items():
        b, z = v["best"], v["base"]
        print(f"{k:22s} base J {key(z):.3f} loss {z['loss']:.3f} -> J {key(b):.3f} loss {b['loss']:.3f}  h {b['h']} q {b['q']} s {b['s']}")


if __name__ == "__main__":
    import sys
    fit_main() if "--fit" in sys.argv else (actions_main() if "--actions" in sys.argv else main())
