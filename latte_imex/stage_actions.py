"""Stage-parameterized pour actions, optimized with the physics frozen (structure-aware objective J_A).

The V0.5 scripts fix the structure of each pattern (where the stages start, whether each stage is closed by its own
lift, whether a through-cut ends the pour); scaling them by (h, q, s) cannot change that structure.  Here a pattern is
a list of stages read off the reference video; each stage is a low pour from a start point to an end point along the
push axis, closed by its own lift (spout up, flow down, moving on along the axis), followed by a dry reposition.

push_heart (reference video): three stages, each from B pushed forward (B->D, B->C with a large flow, a short one at
B), each closed by a lift, no through-cut.

    python -m latte_imex.stage_actions --pattern push_heart --model old --gens 20 --popsize 12 --N 96
"""
import argparse, json, math, os, pickle
from multiprocessing import Pool
import numpy as np

from . import action_opt as ao
from .cell_test import objective_struct
from .joint_cma import to_unit, from_unit

L = 1e-6       # mL/s -> m^3/s

# per stage: end position along the axis, lateral offset, flow [mL/s], spout height [m], duration [s]
STAGE = dict(u_end=(-0.32, 0.28, "lin"), v=(-0.14, 0.14, "lin"), Q=(8.0, 28.0, "log"), z=(0.008, 0.03, "log"),
             dur=(0.5, 2.2, "log"))
GLOBAL = dict(theta=(-0.9, 0.9, "lin"), u_B=(0.08, 0.32, "lin"), z_lift=(0.04, 0.08, "lin"), dur_lift=(0.2, 0.6, "log"),
              Q_lift=(1.0, 6.0, "log"))
SPEC = {"push_heart": dict(n=3, x0=dict(theta=0.0, u_B=0.22, z_lift=0.065, dur_lift=0.4, Q_lift=2.5,
                                        stages=[dict(u_end=-0.12, v=0.0, Q=15.0, z=0.02, dur=1.6),
                                                dict(u_end=0.05, v=0.0, Q=22.0, z=0.018, dur=1.4),
                                                dict(u_end=0.16, v=0.0, Q=15.0, z=0.02, dur=0.9)]))}


def space(n):
    keys = list(GLOBAL.items()) + [(f"{k}{i}", b) for i in range(n) for k, b in STAGE.items()]
    return keys


def encode(p, n):
    flat = {k: p[k] for k in GLOBAL}
    for i, st in enumerate(p["stages"]):
        flat.update({f"{k}{i}": v for k, v in st.items()})
    return np.array([to_unit(flat[k], *b) for k, b in space(n)])


def decode(x, n):
    flat = {k: from_unit(u, *b) for (k, b), u in zip(space(n), x)}
    p = {k: flat[k] for k in GLOBAL}
    p["stages"] = [{k: flat[f"{k}{i}"] for k in STAGE} for i in range(n)]
    return p


def moves_of(p):
    """Stage parameters -> pitcher moves (V0.5 cup frame: barista at +y, pushing towards -y)."""
    from .pitcher import Move
    c, s = math.cos(p["theta"]), math.sin(p["theta"])
    pt = lambda u, v: (float(np.clip(-s * u + c * v, -0.45, 0.45)), float(np.clip(c * u + s * v, -0.45, 0.45)))
    mv, last = [], None
    for i, st in enumerate(p["stages"]):
        a, b = pt(p["u_B"], st["v"]), pt(st["u_end"], st["v"])
        if last is not None:
            mv.append(Move(dur=0.3, p0=last, p1=a, z0=0.03, z1=0.03, Q0=0.0, Q1=0.0, name=f"reposition {i}"))
        mv.append(Move(dur=st["dur"], p0=a, p1=b, z0=st["z"], z1=st["z"], Q0=st["Q"] * L, Q1=st["Q"] * L, name=f"stage {i}"))
        step = -0.06 if st["u_end"] < p["u_B"] else 0.06
        last = pt(st["u_end"] + step, st["v"])
        mv.append(Move(dur=p["dur_lift"], p0=b, p1=last, z0=p["z_lift"], z1=p["z_lift"] + 0.01,
                       Q0=p["Q_lift"] * L, Q1=0.25 * p["Q_lift"] * L, name=f"lift {i}"))
    mv.append(Move(dur=0.35, p0=last, p1=last, z0=0.09, z1=0.09, Q0=0.0, Q1=0.0, name="stop"))
    mv.append(Move(dur=1.0, p0=last, p1=(last[0], -1.8), z0=0.09, z1=0.14, Q0=0.0, Q1=0.0, name="move away"))
    return mv


def tag_x(x):
    return "st" + "".join(f"{int(round(u * 999)):03d}" for u in x)


def main():
    import cma
    from .joint_cma import BASE_V2
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="push_heart")
    ap.add_argument("--model", default="old", help="old (closure #9) or a joint_cma run directory for v2 physics")
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--out", default="runs/action_opt")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    sp = SPEC[a.pattern]; n = sp["n"]
    if a.model == "old":
        over, label = {}, f"stage_old"
    else:
        B = json.load(open(os.path.join(a.out, a.model, "history.json")))["best"]["best"]
        over, label = dict(BASE_V2, **B["phys"]), f"stage_{a.model}"
    d = os.path.join(a.out, f"{label}_{a.pattern}_cma"); os.makedirs(d, exist_ok=True)
    st_path = os.path.join(d, "state.pkl")
    if os.path.exists(st_path):
        es, hist, g0 = pickle.load(open(st_path, "rb"))
    else:
        es = cma.CMAEvolutionStrategy(list(encode(sp["x0"], n)), a.sigma0, dict(bounds=[0, 1], popsize=a.popsize, seed=3, verbose=-9))
        hist, g0 = [], 0
    key = lambda r: objective_struct(dict(r, out=a.out), a.N)
    with Pool(a.procs, maxtasksperchild=8) as pool:
        if g0 == 0:                                            # the video script itself, for reference
            x = encode(sp["x0"], n)
            r0 = pool.map(ao.run_one, [(label, a.pattern, 1.0, 1.0, 1.0, a.N, a.out, over, (tag_x(x), moves_of(decode(x, n))))])[0]
            hist.append(dict(gen=-1, J=key(r0)[0], loss=r0["loss"], x=list(x), p=decode(x, n)))
        for gen in range(g0, g0 + a.gens):
            X = es.ask()
            R = pool.map(ao.run_one, [(label, a.pattern, 1.0, 1.0, 1.0, a.N, a.out, over, (tag_x(x), moves_of(decode(x, n)))) for x in X])
            Js = [key(r)[0] for r in R]
            es.tell(X, Js)
            k = int(np.argmin(Js))
            hist.append(dict(gen=gen, J=Js[k], loss=R[k]["loss"], tag=R[k]["tag"], x=list(X[k]), p=decode(X[k], n),
                             terms=key(R[k])[1], medianJ=float(np.median(Js))))
            best = min(hist, key=lambda h: h["J"])
            json.dump(dict(best=best, hist=hist, over=over), open(os.path.join(d, "history.json"), "w"), indent=1)
            pickle.dump((es, hist, gen + 1), open(st_path, "wb"))
            print(f"gen {gen:02d} best J {Js[k]:.3f} loss {R[k]['loss']:.3f} median {np.median(Js):.3f} overall {best['J']:.3f}", flush=True)


if __name__ == "__main__":
    main()
