"""Replay the pre-registered CMA-ES runs from the disk cache (no forward calls): checks that the trajectory is the one in
calib.json (generation minima, sigma, n_forward) and reports the final distribution in phenotype (unit box) coordinates,
es.result.xfavorite (the mean as a phenotype; calib.json's history 'mean' is decode(es.mean) of the genotype, which the
bound transform can fold), per-parameter stds, and the last generation's sample range.  Writes DIR/calib_replay.json.

    python3 replay.py DIR
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/user/zfChen")
import cma                                                   # noqa: E402
from latte_imex import calib as CB                           # noqa: E402
from latte_imex import v05_calibrate as VC                   # noqa: E402
from latte_imex import v05_guessed_calib as G                # noqa: E402

D = os.path.abspath(sys.argv[1])
rec = json.load(open(os.path.join(D, "calib", "calib.json")))
cfg = rec["driver"]["cfg"]
spec = G.make_spec()
assert json.loads(json.dumps(spec)) == rec["spec"]
space = rec["space"]
keys = list(space)
names = rec["names"]
N = rec["N"]
cache = os.path.join(D, "cache")
pts = G.init_points()
out = dict(inits=[])
nf = 0
for ii, k in enumerate(cfg["inits"]):
    x0p = pts[k]
    es = cma.CMAEvolutionStrategy(CB.encode(x0p, space), cfg["sigma0"],
                                  dict(bounds=[0, 1], popsize=cfg["popsize"], seed=cfg["seed"] + ii, maxiter=cfg["gens"],
                                       verbose=-9))
    g = 0
    mism = []
    lastX = None
    while not es.stop():
        X = es.ask()
        P = [CB.decode(x, space) for x in X]
        L = []
        for p in P:
            ls = [json.load(open(os.path.join(cache, VC.cache_key(p, n, N, spec) + ".json")))["eval"]["loss"] for n in names]
            nf += len(names)
            li = float(np.mean(ls))
            L.append(li if np.isfinite(li) else 1e6)
        es.tell(X, L)
        h = rec["inits"][ii]["history"][g]
        if abs(h["loss_min"] - min(L)) > 1e-12 or abs(h["sigma"] - es.sigma) > 1e-12:
            mism.append(g)
        lastX = np.array(X)
        g += 1
    xf = np.array(es.result.xfavorite)
    stds = np.array(es.result.stds)
    out["inits"].append(dict(
        init=k, gens=g, trajectory_matches_calib_json=not mism, mismatched_gens=mism, final_sigma=float(es.sigma),
        stop=dict(es.stop()), xfavorite_unit={q: round(float(xf[i]), 4) for i, q in enumerate(keys)},
        xfavorite_phys=CB.decode(xf, space),
        stds_unit={q: round(float(stds[i]), 4) for i, q in enumerate(keys)},
        last_gen_unit_range={q: [round(float(lastX[:, i].min()), 4), round(float(lastX[:, i].max()), 4)]
                             for i, q in enumerate(keys)},
        mean_near_bound_2pct={q: ("lo" if xf[i] <= 0.02 else "hi") for i, q in enumerate(keys)
                              if xf[i] <= 0.02 or xf[i] >= 0.98}))
out["n_forward_replayed"] = nf
out["n_forward_calib_json"] = rec["n_forward"]
json.dump(out, open(os.path.join(D, "calib_replay.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
