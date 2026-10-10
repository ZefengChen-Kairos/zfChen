"""Per-pattern action polish with the jointly fitted v2 physics frozen: coordinate search on (h, q, s) around the joint-fit
actions, objective = loss + |gap fraction - target gap fraction| (cell_test.objective).

    python -m latte_imex.v2_actions --runs joint_cma_v2,joint_cma_v2w --rounds 3 --N 96 --out runs/action_opt
"""
import argparse, json, os
from multiprocessing import Pool
from . import action_opt as ao
from .cell_test import objective, objective_struct
from .joint_cma import BASE_V2

STEP = dict(h=(0.85, 1.18), q=(0.88, 1.14), s=(0.95, 1.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="joint_cma_v2")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--out", default="runs/action_opt")
    ap.add_argument("--objective", default="gap", choices=["gap", "struct"])
    ap.add_argument("--model", default="v2best")
    a = ap.parse_args()
    H = [json.load(open(os.path.join(a.out, r, "history.json")))["best"] for r in a.runs.split(",")]
    B = min(H, key=lambda h: h["J"])["best"]
    phys = dict(BASE_V2, **B["phys"])
    ao.MODELS[a.model] = phys                                                  # before the fork
    obj = objective_struct if a.objective == "struct" else objective
    key = lambda r: obj(dict(r, out=a.out), a.N)[0]
    cur = {n: tuple(B["acts"][n]) for n in ao.PATTERNS}
    with Pool(4, maxtasksperchild=8) as pool:
        best = {n: r for n, r in zip(ao.PATTERNS, pool.map(ao.run_one, [(a.model, n, *cur[n], a.N, a.out) for n in ao.PATTERNS]))}
        start = dict(best)
        for rnd in range(a.rounds):
            jobs = []
            for n in ao.PATTERNS:
                b = best[n]
                for i, k in enumerate("hqs"):
                    for f in STEP[k]:
                        v = [b["h"], b["q"], b["s"]]; v[i] = round(v[i] * f, 3)
                        jobs.append((a.model, n, *v, a.N, a.out))
            R = pool.map(ao.run_one, jobs)
            moved = 0
            for n in ao.PATTERNS:
                c = min([r for r in R if r["name"] == n] + [best[n]], key=key)
                moved += c is not best[n]; best[n] = c
            print(f"round {rnd}: {moved} patterns moved, mean J {sum(key(r) for r in best.values()) / 6:.3f}", flush=True)
            if not moved:
                break
    res = {f"{a.model}/{n}": dict(best=best[n], base=start[n], n=0) for n in ao.PATTERNS}
    json.dump(dict(phys=phys, src=B, actions=res), open(os.path.join(a.out, f"{a.model}_actions_N{a.N}.json"), "w"), indent=1)
    for n in ao.PATTERNS:
        print(f"{n:14s} J {key(start[n]):.3f} -> {key(best[n]):.3f}  loss {best[n]['loss']:.3f}  h {best[n]['h']} q {best[n]['q']} s {best[n]['s']}")


if __name__ == "__main__":
    main()
