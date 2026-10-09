"""Action re-optimization with the physics held fixed (old chi closure #9 vs the two-layer closure).

    python -m latte_imex.action_opt --N 96 --out runs/action_opt            # search, both models, six patterns
    python -m latte_imex.action_opt --N 128 --out runs/action_opt --verify  # best of each search at 128 + baseline

The V0.5-translated barista intent (pitcher.v05_moves) is kept move by move; only three global knobs change:
  h  pour-height factor (every pouring move below 5 cm; the cut and the bridges stay), clipped to 0.8-5 cm
  q  wanted-flow factor of the pouring moves below 5 cm
  s  spatial scale of the path about the cup centre (start/end points, Bezier controls, wiggle amplitude)
Search per (model, pattern): h line (pour height first), then a q x s grid at the best h, then an h refinement.
Every run is cached as <out>/<model>/<pattern>_<tag>_N<N>.{npy,json}, so the search can be resumed.
"""
import argparse, copy, json, math, os, time
from multiprocessing import Pool
import numpy as np

PATTERNS = ["heart", "push_heart", "layered_heart", "tulip", "leaf", "swan"]
MODELS = dict(old={}, new=dict(closure="twolayer", phi_foam=1.0, ent_coef=0.0))
D_L = 0.08
_GEOM = None


def physics9():
    r = json.load(open("runs/joint_sweep_r2fix/results.json"))[9]
    ph = dict(r["physics"]); kQ = ph.pop("kQ")
    return ph, kQ


def scaled_moves(name, h=1.0, q=1.0, s=1.0):
    from .pitcher import v05_moves
    mv = copy.deepcopy(v05_moves(name))
    lim = 0.45
    sc = lambda p: tuple(float(np.clip(c * s, -lim, lim)) for c in p)
    for m in mv[:-1]:                                       # the appended move-away keeps its geometry
        m.p0, m.p1 = sc(m.p0), sc(m.p1)
        if m.bezier:
            m.bezier = tuple(sc(c) for c in m.bezier)
        m.wobble *= s; m.wobble1 *= s
        if m.Q0 > 0 and m.z0 < 0.05:
            m.z0 = float(np.clip(m.z0 * h, 0.008, 0.05))
            if m.z1 < 0.05:
                m.z1 = float(np.clip(m.z1 * h, 0.008, 0.05))
            m.Q0 *= q; m.Q1 *= q
    mv[-1].p0 = mv[-2].p1
    return mv


def tag_of(h, q, s):
    return f"h{h:.3f}_q{q:.3f}_s{s:.3f}"


def run_one(job):
    """job = (model, name, h, q, s, N, out[, over]) -> dict with loss, iou, ...; cached on disk under <out>/<model>.
    over (optional) replaces MODELS[model] as the physics overrides (closure fits use their own model labels)."""
    global _GEOM
    model, name, h, q, s, N, out = job[:7]
    over = job[7] if len(job) > 7 else MODELS[model]
    d = os.path.join(out, model); os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{name}_{tag_of(h, q, s)}_N{N}")
    if os.path.exists(stem + ".json"):
        return json.load(open(stem + ".json"))
    from .pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript
    from .solver import Solver, Params, Numerics
    from .optimize import whiteness_sim, loss_fields, target_at
    if _GEOM is None:
        _GEOM = GridPitcherGeometry.load(nr=6)
    ph, kQ = physics9()
    cp = Coupling(c_S=kQ / D_L ** 3, c_U=1.0 / D_L, c_u=1.0 / D_L, footprint="physical")
    script = BaristaScript(moves=scaled_moves(name, h, q, s), pitcher=Pitcher(geom=_GEOM, V0=250e-6), coupling=cp)
    pd = dict(ph, D=1e-7, kappa_Q=125.0, return_law="v05"); pd.update(over)
    num = Numerics(N=N); sol = Solver(Params(**pd), num); t0 = time.time()
    res = dict(model=model, name=name, h=h, q=q, s=s, N=N)
    try:
        while sol.t < script.T - 1e-9:
            probe, _ = script.sample(sol.t); fdt = num.frame_dt
            if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
                fdt = min(fdt, max(num.scan_safety * min(sol.g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
            inl, _ = script.sample(sol.t + 0.5 * fdt); sol.advance_frame(inl, frame_dt=fdt)
        c = sol.concentration(); mask = sol.g.mask
        if not np.all(np.isfinite(c[mask])):
            raise FloatingPointError("non-finite c")
        L, parts = loss_fields(whiteness_sim(c, mask), target_at(name, N), mask, N / 48)
        np.save(stem + ".npy", c.astype(np.float32))
        V_end = script.pitcher.state.V
        res.update(loss=L, ok=True, V_end=float(V_end), deposited=float(sol.ledger()["deposited"]), **parts)
    except Exception as e:  # noqa
        res.update(loss=10.0, ok=False, err=f"{type(e).__name__}: {e}")
    res["elapsed"] = time.time() - t0
    json.dump(res, open(stem + ".json", "w"))
    print(f"{model:3s} {name:13s} h {h:.2f} q {q:.2f} s {s:.2f}  loss {res['loss']:.3f} iou {res.get('iou', 0):.2f} "
          f"{res['elapsed']:.0f}s", flush=True)
    return res


def search(pool, N, out, models, patterns):
    best = {}
    def batch(jobs):
        return pool.map(run_one, jobs)
    key = lambda r: r["loss"]
    combos = [(m, n) for m in models for n in patterns]
    # stage 1: pour height line
    H = [0.6, 0.8, 1.0, 1.3, 1.6]
    R1 = batch([(m, n, h, 1.0, 1.0, N, out) for m, n in combos for h in H])
    h1 = {(m, n): min([r for r in R1 if r["model"] == m and r["name"] == n], key=key)["h"] for m, n in combos}
    # stage 2: flow x size at the best height
    QS = [(q, s) for q in (0.8, 1.0, 1.25) for s in (0.9, 1.0, 1.15)]
    R2 = batch([(m, n, h1[m, n], q, s, N, out) for m, n in combos for q, s in QS])
    qs = {}
    for m, n in combos:
        b = min([r for r in R2 if r["model"] == m and r["name"] == n], key=key); qs[m, n] = (b["q"], b["s"])
    # stage 3: height refinement at the best flow and size
    R3 = batch([(m, n, round(h1[m, n] * f, 3), *qs[m, n], N, out) for m, n in combos for f in (0.85, 1.15)])
    allr = R1 + R2 + R3
    for m, n in combos:
        rs = [r for r in allr if r["model"] == m and r["name"] == n]
        base = [r for r in rs if r["h"] == 1.0 and r["q"] == 1.0 and r["s"] == 1.0][0]
        best[f"{m}/{n}"] = dict(best=min(rs, key=key), base=base, n=len(rs))
    return best


def refine(pool, N, out, prev):
    """Second round around the first-round optimum (many optima sat on the edge of the first grid):
    h x {0.8, 1}, q x {1, 1.2}, s x {0.93, 1, 1.07}."""
    jobs, keys = [], []
    for k, v in prev.items():
        b = v["best"]
        for fh in (0.8, 1.0):
            for fq in (1.0, 1.2):
                for fs in (0.93, 1.0, 1.07):
                    jobs.append((b["model"], b["name"], round(b["h"] * fh, 3), round(b["q"] * fq, 3), round(b["s"] * fs, 3), N, out))
                    keys.append(k)
    R = pool.map(run_one, jobs)
    best = {}
    for k, v in prev.items():
        rs = [r for r, kk in zip(R, keys) if kk == k] + [v["best"]]
        best[k] = dict(best=min(rs, key=lambda r: r["loss"]), base=v["base"], n=v["n"] + len(rs) - 1)
    return best


def export_web(best, path="web/scripts.json"):
    """Write the optimized intents of the given model into the web bench scripts (moves only; the physics of each
    script is the bench default without the delay closure, the closure the search ran with)."""
    from .pitcher import moves_to_json
    d = json.load(open(path))
    for k, v in best.items():
        b = v["best"]; n = b["name"]
        if n not in d:
            continue
        d[n]["moves"] = moves_to_json(scaled_moves(n, b["h"], b["q"], b["s"]))
        d[n]["opt"] = dict(h=b["h"], q=b["q"], s=b["s"], loss=round(b["loss"], 4), iou=round(b["iou"], 3), N=b["N"],
                           base_loss=round(v["base"]["loss"], 4))
    for k, v in d.items():
        prm = v.get("params") if k == "_default_physics" else (v.get("physics") or {}).get("params")
        if prm is not None:
            prm["tau_d"] = 0.0
    json.dump(d, open(path, "w"), ensure_ascii=False)


def figure(R, N, path):
    """Rows = patterns; columns = target, old before/after, two-layer before/after (final whiteness, loss/IoU)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    from .optimize import whiteness_sim, target_at
    mask = np.hypot(*np.meshgrid(*(2 * [(np.arange(N) + 0.5) / N - 0.5]))) < 0.49
    brown, white = np.array([92, 54, 30]) / 255, np.array([246, 240, 226]) / 255
    cols = [("old", True), ("old", False), ("new", True), ("new", False)]
    head = ["目标", "旧模型 原动作", "旧模型 优化后", "双层 原动作", "双层 优化后"]
    fig, ax = plt.subplots(len(PATTERNS), 5, figsize=(12, 2.5 * len(PATTERNS)))
    def show(a, img, title):
        rgb = brown[None, None, :] * (1 - img[..., None]) + white[None, None, :] * img[..., None]
        rgb[~mask] = 0.15
        a.imshow(rgb, origin="lower"); a.set_title(title, fontsize=9); a.axis("off")
    for i, n in enumerate(PATTERNS):
        show(ax[i, 0], target_at(n, N), f"{n} {head[0]}")
        for j, (m, base) in enumerate(cols):
            rs = [r for r in R if r["model"] == m and r["name"] == n and r["N"] == N
                  and (r["h"] == 1.0 and r["q"] == 1.0 and r["s"] == 1.0) == base]
            if not rs:
                ax[i, j + 1].axis("off"); continue
            r = rs[0]
            c = np.load(os.path.join("runs/action_opt", m, f"{n}_{tag_of(r['h'], r['q'], r['s'])}_N{N}.npy"))
            lab = "" if base else f"\nh×{r['h']:.2f} q×{r['q']:.2f} s×{r['s']:.2f}"
            show(ax[i, j + 1], whiteness_sim(c, mask), f"{head[j + 1]}  {r['loss']:.3f}/{r['iou']:.2f}{lab}")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--out", default="runs/action_opt")
    ap.add_argument("--models", default="old,new")
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--refine", default=None, help="search json of round 1 -> round 2 around its optima")
    ap.add_argument("--export-web", default=None, help="search/refine json -> web/scripts.json moves (model of --models)")
    ap.add_argument("--verify", default=None, help="search json to verify at --N (best and baseline)")
    a = ap.parse_args()
    models, patterns = a.models.split(","), a.patterns.split(",")
    if a.export_web:
        export_web({k: v for k, v in json.load(open(a.export_web)).items() if k.split("/")[0] in models})
        return
    with Pool(a.procs, maxtasksperchild=4) as pool:
        if a.verify:
            S = {}
            for f in a.verify.split(","):        # later files override earlier ones (round 1, then round 2)
                S.update(json.load(open(f)))
            jobs = []
            for k, v in S.items():
                b = v["best"]
                jobs += [(b["model"], b["name"], b["h"], b["q"], b["s"], a.N, a.out),
                         (b["model"], b["name"], 1.0, 1.0, 1.0, a.N, a.out)]
            R = pool.map(run_one, jobs)
            json.dump(R, open(os.path.join(a.out, f"verify_N{a.N}.json"), "w"), indent=1)
            figure(R, a.N, os.path.join(a.out, f"verify_N{a.N}.png"))
            return
        if a.refine:
            prev = {k: v for k, v in json.load(open(a.refine)).items() if k.split("/")[0] in models}
            best = refine(pool, a.N, a.out, prev)
        else:
            best = search(pool, a.N, a.out, models, patterns)
    name = f"refine_N{a.N}.json" if a.refine else f"search_N{a.N}.json"
    json.dump(best, open(os.path.join(a.out, name), "w"), indent=1)
    for k, v in best.items():
        b, z = v["best"], v["base"]
        print(f"{k:20s} base {z['loss']:.3f}/{z['iou']:.2f} -> best {b['loss']:.3f}/{b['iou']:.2f}  h {b['h']} q {b['q']} s {b['s']}")


if __name__ == "__main__":
    main()
