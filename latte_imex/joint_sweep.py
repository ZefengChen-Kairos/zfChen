"""Joint forward sweep: physically bounded 2-D closure parameters x physically feasible pitcher actions.

    python -m latte_imex.joint_sweep --n 100 --N 96 --out runs/joint_sweep

No inversion: every sample draws one shared physics vector (within coffee-physics ranges) and one feasible
action per pattern (heart, tulip), runs the full chain pitcher -> ledger -> IMEX, and scores the final
whiteness field against the video targets (latte_imex/targets).  The coupling is purely dimensional
(D_L = 8 cm): S_eff = kQ * Q / D_L^3, U_perp and u_in in D_L/s, physical footprint, so the only effective
constants are in the physics vector.  Outputs: results.json (all samples), fields.npz (final fields), and
best.png (the best tulips with the hearts of the same physics).
"""
import argparse, json, math, os, time
import numpy as np

from .pitcher import Pitcher, GridPitcherGeometry, Coupling, Barista, Move, BaristaScript, Cup
from .solver import Solver, Params, Numerics
from .optimize import whiteness_sim, loss_fields, target_at

D_L = 0.08
# physics: (lo, hi, scale); coffee-physics ranges, effective closure constants with the dimensional coupling
PHYS = dict(cp=(0.3, 4.0, "log"), beta=(1.0, 10.0, "log"), nu=(1e-4, 3e-3, "log"), kappa_c=(0.3, 2.0, "log"),
            kappa_t=(0.1, 1.0, "lin"), kappa_r=(0.003, 0.1, "log"), B_dep=(300.0, 10000.0, "log"), p_dep=(1.0, 4.0, "lin"),
            kQ=(0.8, 3.0, "log"))
# feasible actions (SI heights and flows; cup-diameter positions)
ACT = dict(
    tulip=dict(base_h=(0.010, 0.025, "lin"), base_Q0=(6e-6, 10e-6, "lin"), base_Q1=(10e-6, 16e-6, "lin"),
               wig_amp=(0.002, 0.008, "lin"), wig_f=(2.0, 5.0, "lin"), wig_dir=(0.0, math.pi / 2, "lin"),
               petal_Q=(10e-6, 24e-6, "lin"), petal_h=(0.010, 0.025, "lin"), petal_len=(0.04, 0.12, "lin"),
               spacing=(0.8, 1.2, "lin"), cut_Q0=(2e-6, 5e-6, "lin"), cut_h=(0.06, 0.09, "lin"), cut_dur=(0.6, 1.0, "lin")),
    heart=dict(disc_h=(0.010, 0.025, "lin"), disc_Q=(10e-6, 18e-6, "lin"), cut_Q0=(2e-6, 5e-6, "lin"),
               cut_h=(0.06, 0.09, "lin"), cut_dur=(0.6, 1.0, "lin")),
)


def sample(space, u):
    out = {}
    for (k, (lo, hi, sc)), v in zip(space.items(), u):
        out[k] = float(lo * (hi / lo) ** v) if sc == "log" else float(lo + (hi - lo) * v)
    return out


def lhs(n, d, rng):
    u = np.zeros((n, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def tulip_moves(a):
    sp = a["spacing"]; A = a["wig_amp"] / D_L; f = a["wig_f"]; dr = a["wig_dir"]
    return [
        Move(3.2, (0.0, 0.08 * sp), (0.0, 0.14 * sp), z0=a["base_h"], Q0=a["base_Q0"], Q1=a["base_Q1"], wobble=0.5 * A, wobble1=A,
             wfreq=f, wobble_dir=dr, name="base ramp"),
        Move(3.2, (0.0, 0.14 * sp), (0.0, 0.14 * sp), z0=a["base_h"], Q0=a["base_Q1"], wobble=A, wfreq=f, wobble_dir=dr, name="base hold"),
        Move(0.18, (0.0, 0.14 * sp), (0.0, 0.24 * sp), z0=0.03, Q0=0.0, name="reposition"),
        Move(1.6, (0.0, 0.24 * sp), (0.0, 0.24 * sp - a["petal_len"]), z0=a["petal_h"], Q0=a["petal_Q"], wobble=0.7 * A, wfreq=f,
             wobble_dir=dr, name="petal 2"),
        Move(0.18, (0.0, 0.24 * sp - a["petal_len"]), (0.0, 0.31 * sp), z0=0.03, Q0=0.0, name="reposition"),
        Move(0.6, (0.0, 0.31 * sp), (0.0, 0.31 * sp - 0.6 * a["petal_len"]), z0=a["petal_h"], Q0=0.6 * a["petal_Q"], wobble=0.5 * A,
             wfreq=f, wobble_dir=dr, name="petal 3"),
        Move(0.18, (0.0, 0.31 * sp - 0.6 * a["petal_len"]), (0.0, 0.35 * sp), z0=0.03, Q0=0.0, name="reposition"),
        Move(a["cut_dur"], (0.0, 0.35 * sp), (0.0, -0.27), z0=a["cut_h"], z1=a["cut_h"] + 0.02, Q0=a["cut_Q0"], Q1=1e-6, name="cut"),
        Move(0.5, (0.0, -0.27), (0.0, -0.27), z0=a["cut_h"] + 0.02, Q0=0.0, name="stop"),
        Move(1.0, (0.0, -0.27), (0.0, -1.8), z0=0.09, z1=0.14, Q0=0.0, name="move away"),
    ]


def heart_moves_a(a):
    return [
        Move(4.0, (0.0, 0.20), (0.0, 0.16), z0=a["disc_h"], Q0=a["disc_Q"], name="disc"),
        Move(1.6, (0.0, 0.16), (0.0, 0.00), z0=a["disc_h"], z1=a["disc_h"] + 0.005, Q0=0.6 * a["disc_Q"], name="approach"),
        Move(0.35, (0.0, 0.00), (0.0, 0.00), z0=a["disc_h"] + 0.005, z1=a["cut_h"], Q0=a["cut_Q0"], name="lift"),
        Move(a["cut_dur"], (0.0, 0.00), (0.0, -0.27), z0=a["cut_h"], Q0=a["cut_Q0"], Q1=1e-6, name="cut"),
        Move(0.5, (0.0, -0.27), (0.0, -0.27), z0=a["cut_h"], z1=a["cut_h"] + 0.02, Q0=0.0, name="stop"),
        Move(1.0, (0.0, -0.27), (0.0, -1.8), z0=0.09, z1=0.14, Q0=0.0, name="move away"),
    ]


MOVES = dict(tulip=tulip_moves, heart=heart_moves_a)
_GEOM = None


def run_chain(name, action, physics, N, V0=250e-6):
    """pitcher -> ledger -> IMEX for one pattern; returns (c, mask, stats, T)."""
    global _GEOM
    if _GEOM is None:
        _GEOM = GridPitcherGeometry.load(nr=6)        # coarser cavity sampling: 3x faster pitcher steps, same tilt-flow law
    kQ = physics["kQ"]
    cp = Coupling(c_S=kQ / D_L ** 3, c_U=1.0 / D_L, c_u=1.0 / D_L, footprint="physical")
    script = BaristaScript(moves=MOVES[name](action), pitcher=Pitcher(geom=_GEOM, V0=V0), coupling=cp)
    pd = {k: v for k, v in physics.items() if k != "kQ"}
    pd.update(D=1e-7, kappa_Q=125.0, return_law="v05")
    P = Params(**pd); num = Numerics(N=N)
    sol = Solver(P, num)
    T = script.T
    while sol.t < T - 1e-9:
        probe, _ = script.sample(sol.t)
        fdt = num.frame_dt
        if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
            fdt = min(fdt, max(num.scan_safety * min(sol.g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
        inl, _ = script.sample(sol.t + 0.5 * fdt)
        sol.advance_frame(inl, frame_dt=fdt)
    return sol.concentration(), sol.g.mask, sol.stats, T


def evaluate(args):
    idx, physics, actions, N = args
    out = dict(idx=idx, physics=physics, actions=actions, N=N, patterns={})
    for name in ("tulip", "heart"):
        t0 = time.time()
        try:
            c, mask, stats, T = run_chain(name, actions[name], physics, N)
            if not np.all(np.isfinite(c[mask])):
                raise FloatingPointError("non-finite")
            s = whiteness_sim(c, mask); w = target_at(name, N)
            L, parts = loss_fields(s, w, mask, N / 48)
            out["patterns"][name] = dict(loss=L, ok=True, elapsed=time.time() - t0, retries=stats["retries"],
                                         white=float(((c > 0.4) & mask).sum() / mask.sum()), **parts)
            out["patterns"][name]["c"] = np.nan_to_num(c, nan=-1).astype(np.float16)
        except Exception as e:
            out["patterns"][name] = dict(loss=9.0, ok=False, error=str(e), elapsed=time.time() - t0)
    return out


def design(n, seed, around=None, spread=0.25):
    """Latin hypercube over the unit cube; with `around` (a list of results) the cube is replaced by boxes of
    half-width `spread` (in unit coordinates) centred on those samples, each box getting n/len(around) points."""
    rng = np.random.default_rng(seed)
    U = lhs(n, len(PHYS) + sum(len(v) for v in ACT.values()), rng)
    if around:
        def to_unit(space, vals):
            out = []
            for k, (lo, hi, sc) in space.items():
                v = vals[k]
                out.append(math.log(v / lo) / math.log(hi / lo) if sc == "log" else (v - lo) / (hi - lo))
            return out
        centres = []
        for r in around:
            c = to_unit(PHYS, r["physics"])
            for name, space in ACT.items():
                c += to_unit(space, r["actions"][name])
            centres.append(np.array(c))
        per = int(math.ceil(n / len(centres)))
        rows = []
        for ci, c in enumerate(centres):
            Uc = lhs(per, U.shape[1], np.random.default_rng(seed + 1 + ci))
            rows.append(np.clip(c[None, :] + (Uc - 0.5) * 2 * spread, 0.0, 1.0))
        U = np.vstack(rows)[:n]
    jobs = []
    for i in range(n):
        u = U[i]; k = len(PHYS)
        physics = sample(PHYS, u[:k])
        actions = {}
        for name, space in ACT.items():
            actions[name] = sample(space, u[k:k + len(space)]); k += len(space)
        jobs.append((i, physics, actions))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--out", default="runs/joint_sweep")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--idx", type=int, default=None, help="evaluate one sample of the design (one process per sample)")
    ap.add_argument("--collect", action="store_true", help="merge sample_*.json into results.json / fields.npz")
    ap.add_argument("--around", default=None, help="results.json of a previous sweep: sample around its best samples")
    ap.add_argument("--top", type=int, default=3, help="how many best (joint loss) samples of --around to centre on")
    ap.add_argument("--spread", type=float, default=0.25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    around = None
    if a.around:
        R = json.load(open(a.around)); R = [r for r in R if r["patterns"]["tulip"].get("ok") and r["patterns"]["heart"].get("ok")]
        R.sort(key=lambda r: r["patterns"]["tulip"]["loss"] + r["patterns"]["heart"]["loss"]); around = R[:a.top]
    jobs = design(a.n, a.seed, around=around, spread=a.spread)
    if a.idx is not None:
        i, physics, actions = jobs[a.idx]
        r = evaluate((i, physics, actions, a.N))
        fields = {}
        for name, p in r["patterns"].items():
            if "c" in p:
                fields[name] = p.pop("c")
        with open(os.path.join(a.out, f"sample_{i:03d}.json"), "w") as f:
            json.dump(r, f)
        np.savez_compressed(os.path.join(a.out, f"sample_{i:03d}.npz"), **fields)
        lt = r["patterns"]["tulip"].get("loss", 9); lh = r["patterns"]["heart"].get("loss", 9)
        print(f"idx {i:3d} tulip {lt:.3f} heart {lh:.3f}  tulip {r['patterns']['tulip'].get('elapsed', 0):.0f}s heart {r['patterns']['heart'].get('elapsed', 0):.0f}s", flush=True)
        return
    if a.collect:
        results, fields = [], {}
        for fn in sorted(os.listdir(a.out)):
            if fn.startswith("sample_") and fn.endswith(".json"):
                r = json.load(open(os.path.join(a.out, fn))); results.append(r)
                z = np.load(os.path.join(a.out, fn.replace(".json", ".npz")))
                for name in z.files:
                    fields[f"{r['idx']}_{name}"] = z[name]
        with open(os.path.join(a.out, "results.json"), "w") as f:
            json.dump(results, f)
        np.savez_compressed(os.path.join(a.out, "fields.npz"), **fields)
        print("collected", len(results))
        return
    print("use --idx i (one sample per process; run with xargs -P) and then --collect")


if __name__ == "__main__":
    main()
