"""Joint CMA-ES over the shared closure constants and the per-pattern action factors (two-layer closure with the
entrainment cell), objective = mean over the six patterns of loss + |gap fraction - target gap fraction|.

    python -m latte_imex.joint_cma --gens 16 --N 96 --out runs/action_opt

x in [0,1]^25: 7 constants (cell_test.SPACE, log/lin) + 6 x (h, q, s) (log bounds below).  Warm start: the gap-aware
physics fit (cell_fit_N96.json) and the actions of the latest action search (cellfit_refine or fit_refine).
State (CMA pickle, history, best) is saved after every generation; --resume continues.
"""
import argparse, json, math, os, pickle
from multiprocessing import Pool
import numpy as np

from .action_opt import run_one, PATTERNS
from .cell_test import SPACE as SPACE_CELL, objective

# v2 (LATTE_MODEL.md): every physical constant is fitted, within physical ranges; the regularizations are fixed
SPACE_V2 = dict(phi_foam=(0.2, 0.7, "lin"), g_red=(40.0, 110.0, "log"), Fr_c2=(5.0, 100.0, "log"), p_dep=(1.0, 4.0, "lin"),
                ent_coef=(0.2, 1.5, "log"), cell_frac=(0.05, 1.0, "log"), beta=(0.5, 20.0, "log"), kappa_c=(0.1, 3.0, "log"),
                kappa_t=(0.3, 1.0, "lin"), H=(0.03, 0.07, "lin"), H_f=(0.001, 0.006, "log"), m_o=(0.05, 1.0, "log"),
                tau_y=(0.01, 3.0, "log"), nu=(1e-4, 1e-2, "log"))
BASE_V2 = dict(closure="v2", l_skin=0.03, yield_eps=0.02, nu_max=0.5, tau_y_crema=0.0, D=1e-7, D_L=0.08)
X0_V2 = dict(phi_foam=0.45, g_red=75.0, Fr_c2=30.0, p_dep=1.56, ent_coef=0.74, cell_frac=0.3, beta=3.0, kappa_c=1.0,
             kappa_t=0.9, H=0.05, H_f=0.003, m_o=0.2, tau_y=0.3, nu=1e-3)
SPACE = SPACE_CELL

ACT = dict(h=(0.3, 2.0), q=(0.6, 2.0), s=(0.8, 1.3))     # log-scaled action-factor bounds


def to_unit(v, lo, hi, sc):
    return (math.log(v / lo) / math.log(hi / lo)) if sc == "log" else (v - lo) / (hi - lo)


def from_unit(u, lo, hi, sc):
    u = min(max(u, 0.0), 1.0)
    return lo * (hi / lo) ** u if sc == "log" else lo + (hi - lo) * u


def decode(x, base):
    phys = dict(base)
    for k, (lo, hi, sc) in SPACE.items():
        phys[k] = from_unit(x[list(SPACE).index(k)], lo, hi, sc)
    acts, i = {}, len(SPACE)
    for n in PATTERNS:
        acts[n] = tuple(round(from_unit(x[i + j], *ACT[k], "log"), 3) for j, k in enumerate("hqs")); i += 3
    return phys, acts


def encode(phys, acts):
    x = [to_unit(phys[k], lo, hi, sc) for k, (lo, hi, sc) in SPACE.items()]
    for n in PATTERNS:
        x += [to_unit(min(max(v, ACT[k][0]), ACT[k][1]), *ACT[k], "log") for k, v in zip("hqs", acts[n])]
    return np.clip(np.array(x), 0.0, 1.0)


def main():
    import cma
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=16)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--out", default="runs/action_opt")
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--popsize", type=int, default=13)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--spec", default="cell", choices=["cell", "v2"])
    ap.add_argument("--init", default=None, help="history.json of an earlier run: warm start from its best physics and actions")
    ap.add_argument("--name", default=None, help="run directory / job tag (default joint_cma or joint_cma_v2)")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    global SPACE
    name = a.name or ("joint_cma" if a.spec == "cell" else "joint_cma_v2")
    d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
    if a.spec == "v2":
        SPACE = SPACE_V2
        F = dict(BASE_V2, **X0_V2); base = dict(BASE_V2)
    else:
        F = json.load(open(os.path.join(a.out, f"cell_fit_N{a.N}.json")))["best"]["over"]
        base = {k: v for k, v in F.items() if k not in SPACE}
    st_path = os.path.join(d, "state.pkl")
    if a.resume and os.path.exists(st_path):
        es, hist, gen0 = pickle.load(open(st_path, "rb"))
    else:
        if a.init:
            B = json.load(open(a.init))["best"]["best"]; src = a.init
            F = dict(F, **B["phys"]); acts = {n: tuple(B["acts"][n]) for n in PATTERNS}
        elif a.spec == "v2":
            src = "neutral"; acts = {n: (1.0, 1.0, 1.0) for n in PATTERNS}     # no bias from the v1 fits
        else:
            src = next(p for p in (f"cellfit_refine_N{a.N}.json", f"fit_refine_N{a.N}.json") if os.path.exists(os.path.join(a.out, p)))
            A = json.load(open(os.path.join(a.out, src)))
            acts = {n: next((v["best"]["h"], v["best"]["q"], v["best"]["s"]) for k, v in A.items() if k.endswith("/" + n)) for n in PATTERNS}
        x0 = encode(F, acts)
        es = cma.CMAEvolutionStrategy(list(x0), a.sigma0, dict(bounds=[0, 1], popsize=a.popsize, seed=a.seed, verbose=-9))
        hist, gen0 = [dict(gen=-1, J=None, src=src, x0=list(x0))], 0
    with Pool(4, maxtasksperchild=8) as pool:
        for gen in range(gen0, gen0 + a.gens):
            X = es.ask()
            cands = [decode(x, base) for x in X]
            tag = "jc" if a.spec == "cell" else ("v2c" if a.name is None else a.name)
            jobs = [(f"{tag}{gen:02d}_{k:02d}", n, *acts[n], a.N, a.out, phys) for k, (phys, acts) in enumerate(cands) for n in PATTERNS]
            R = pool.map(run_one, jobs)
            Js, rows = [], []
            for k, (phys, acts) in enumerate(cands):
                rs = [dict(r, out=a.out) for r in R if r["model"] == f"{tag}{gen:02d}_{k:02d}"]
                per = {r["name"]: objective(r, a.N) for r in rs}
                J = float(np.mean([v[0] for v in per.values()])); Js.append(J)
                rows.append(dict(k=k, J=J, loss=float(np.mean([r["loss"] for r in rs])), phys={kk: phys[kk] for kk in SPACE}, acts=acts,
                                 gap={n: round(v[1].get("gap_frac", 0.0), 3) for n, v in per.items()}))
            es.tell(X, Js)
            b = min(rows, key=lambda r: r["J"])
            hist.append(dict(gen=gen, J=b["J"], loss=b["loss"], best=b, medianJ=float(np.median(Js))))
            allbest = min((h for h in hist if h.get("best")), key=lambda h: h["J"])
            json.dump(dict(best=allbest, hist=hist), open(os.path.join(d, "history.json"), "w"), indent=1)
            pickle.dump((es, hist, gen + 1), open(st_path, "wb"))
            print(f"gen {gen:02d}  best J {b['J']:.3f} loss {b['loss']:.3f}  median J {np.median(Js):.3f}  "
                  f"overall best J {allbest['J']:.3f}  sigma {es.sigma:.3f}", flush=True)


if __name__ == "__main__":
    main()
