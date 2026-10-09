"""Two-layer closure with the entrainment cell (surface flow to the jet, brown surface where coffee wells up, only brown
liquid subducted at the jet): six patterns with the fitted constants and the two-layer optimized intents.

    python -m latte_imex.cell_test --N 96 --out runs/action_opt
"""
import argparse, json, os
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


if __name__ == "__main__":
    main()
