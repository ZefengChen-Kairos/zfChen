"""GPU closure calibration against the real videos (senior V1 5.1-5.4 protocol, the setup of v05_calibrate) and a
hook to fit the forward pitcher's constants to scale-measured flow data.

CLOSURE CALIBRATION = v05_calibrate's pre-registered setup with the forward calls batched on the GPU:
  model     calib.make_model(v05_calibrate.SPEC): v05_adapter.V05Model (the barista's real actions, read off each clip,
            through the physical inlet), closure base FSp2, solver Params defaults (the ALIGNED model: symmetric
            viscosity, off-cup accounting)
  space     the 8 closure parameters of v05_adapter.SPACE (B_dep, p_dep, kappa_r, kappa_Q, c_p, beta, kappa_t, kappa_c;
            physical bounds) + the 2 input nuisances u_exit [0.10, 0.27] m/s and Q_scale [0.5, 2] (v05_calibrate.SPEC)
  loss      calib.loss_of(ev, delay=False) of calib.evaluate (<1 - Dice>_t + <d_bnd>_t / 0.05 D + stage-end fine-line
            count error), mean over tulip + leaf (calib.CAL_AB), equal weights; a failed forward call scores 1e6
  CMA-ES    calib.calibrate's loop (pycma, unit box, bounds [0, 1]): N 96, popsize 12, sigma0 0.25, 20 generations,
            the 3 initial points of v05_calibrate.initial_points (FSp2, FS, box centre; seeds seed + init index); the
            initial points are evaluated once as references.  Here the 3 inits run in LOCKSTEP and one generation of
            all of them (3 x 12 candidates x 2 clips = 72 forward runs) is ONE BatchCudaSolver batch
            (gpu_batch.ClipMember = V05Model.run's frame logic: frames end on the clip's output times); the calib
            metrics run on the host in a process pool (--procs).  Best = lowest mean loss over all CMA samples.
  checkpoint state.pkl (CMA states, numpy RNG, records) after every generation, atomic; `run` resumes.  calib.json has
            calib.calibrate's record format (spec, names, space, inits[history, best], n_forward, n_failed, best).
  predict   the frozen best parameters (and the FSp2 start, for comparison) on all six clips at N = 128 (one batch):
            calibration fit (tulip, leaf) and test prediction (heart, push_heart, layered_heart, swan) reported
            separately (calib.summary); contact sheets; final frames at N = 256 next to the video's last frame.

    python -m latte_imex.gpu_calib run     --out DIR [--N 96 --popsize 12 --gens 20 --inits 0,1,2]
    python -m latte_imex.gpu_calib status  --out DIR
    python -m latte_imex.gpu_calib predict --out DIR [--N 128 --N-fig 256]
    python -m latte_imex.gpu_calib best    --out DIR          # -> DIR/best.json (closure for gpu_design --closure)

PITCHER CONSTANTS (no data yet; the hook and its input format):
    python -m latte_imex.gpu_calib fit_pitcher --scale FILE[,FILE...] --out DIR [--fit C_d,h_k]
    python -m latte_imex.gpu_calib make_scale --pattern heart --law C_d=0.55 --out FILE.npz   (synthetic file, tests)
  Input: one .npz per pour, keys
    t          [n]     pose times [s], uniform step (the pose acts over [t - dt/2, t + dt/2]; ForwardPitcher.run)
    tip        [n, 3]  spout-tip position [m] in the cup frame (coffee surface z = 0), as s1_forward poses
    yaw, tilt, roll [n] [rad], conventions of pitcher.Pose / ForwardPitcher
    fill_mL    ()      milk in the pitcher at t[0] [mL]
    alpha_deg  ()      lip slope of the pitcher geometry (s1_forward.geometry; NX S1: 0)
    m_t        [k]     scale reading times [s] (same clock as t)
    m_g        [k]     scale reading [g]
    scale_on   ()      'cup' (the cup stands on the scale: m = rho V_cup(t), arrivals; default) or 'pitcher'
                       (pitcher on the scale: m = rho V_P(t))
    rho        ()      density [g/mL] (default 1.03, milk)        tau_scale () first-order scale lag [s] (default 0)
  Model: pitcher.ForwardPitcher(geometry(alpha), ForwardLaw(C_d, C_v, h_k)).run(poses) -> V_cup(t) (ForwardPitcher.arrivals)
  or V_P(t), through the scale lag, sampled at m_t; tare: the first reading is subtracted from both.  Loss = RMS of the
  mass error [g] over all files.  Fitted (CMA-ES, unit box): C_d [0.4, 0.9], C_v [0.8, 1.1], h_k [0, 2 mm] (--fit
  chooses; C_v only shifts arrival times by a few ms through the jet speed and is weakly identifiable from a scale).
  Output DIR/pitcher_law.json {law: {...}, rms_g, per_file, ...}: gpu_design --pitcher-law reads it.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import pickle
import time

import numpy as np

from . import calib as CB

_W = {}


def _log_to(path):
    def log(*a):
        line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
        print(line, flush=True)
        if path:
            with open(path, "a") as f:
                f.write(line + "\n")
    return log


def _jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    os.replace(tmp, path)


def spec():
    from . import v05_calibrate as VC
    return dict(VC.SPEC)


# =========================================================================== host-side metrics (process pool)
def _pinit():
    _W["clips"] = {}


def _clip(name):
    c = _W.setdefault("clips", {})
    if name not in c:
        c[name] = CB.load_clip(name)
    return c[name]


def score_one(args):
    """calib._weval's scoring on a finished forward run: (name, c [n, N, N] M frame, ledger) -> compact evaluation."""
    name, c, led, delay = args
    clip = _clip(name)
    try:
        if led is not None and not led.get("ok", True):
            return dict(name=name, loss=float("inf"), ledger=led, failed=True)
        wT = CB.observe_model(c, clip["M"])
        return CB.compact(CB.evaluate(clip, wT, led, delay_in_loss=delay))
    except Exception as e:
        return dict(name=name, loss=float("inf"), failed=True, error=f"{type(e).__name__}: {e}")


class Scorer:
    def __init__(self, procs=4):
        self.pool = None
        if procs > 1:
            import multiprocessing as mp
            self.pool = mp.get_context("spawn").Pool(procs, initializer=_pinit)

    def map(self, jobs):
        if self.pool is None:
            return [score_one(j) for j in jobs]
        return self.pool.map(score_one, jobs, chunksize=1)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def forward(engine, model, items, N, T=None, chunk=0):
    """items: list of (params, clip name) -> list of (name, c, ledger), one batched engine call."""
    from .gpu_batch import ClipMember
    mem = []
    for params, name in items:
        clip = _clip(name)
        mem.append(ClipMember(model, clip["script"], params, N, CB.sample_times(clip), T_max=T, tag=name))
    engine.run(mem, chunk=chunk)
    return [(name, m.result["c"], m.result["ledger"]) for (params, name), m in zip(items, mem)]


# =========================================================================== CMA-ES driver
def run(out, names=CB.CAL_AB, gens=20, popsize=12, N=96, sigma0=0.25, seed=1, inits=(0, 1, 2), backend="cuda",
        T=None, chunk=0, procs=4, max_gens=None):
    import cma
    from . import v05_calibrate as VC
    from .gpu_batch import BatchEngine
    os.makedirs(out, exist_ok=True)
    log = _log_to(os.path.join(out, "log.txt"))
    sp = spec()
    model = CB.make_model(sp)
    space = model.space
    x0s_all = VC.initial_points(space)
    x0s = [x0s_all[i] for i in inits]
    setup = dict(spec=sp, names=list(names), N=N, gens=gens, popsize=popsize, sigma0=sigma0, seed=seed, inits=list(inits),
                 T=T)
    st_path = os.path.join(out, "state.pkl")
    scorer = Scorer(procs)
    engine = BatchEngine(N, backend=backend, log=log)
    delay = bool(sp.get("loss_delay", True))
    try:
        if os.path.exists(st_path):
            with open(st_path, "rb") as f:
                state = pickle.load(f)
            if state["setup"] != setup:
                raise SystemExit(f"{st_path} was written with another setup; use another --out")
            np.random.set_state(state["np_rng"])
            log(f"resume at generation {state['gen']}")
        else:
            rec = dict(spec=sp, names=list(names), space=space, N=N, gens=gens, popsize=popsize, sigma0=sigma0,
                       preregistered=dict(SIGMA_CAM_D=CB.SIGMA_CAM_D, THR=CB.THR, Q_MIN=CB.Q_MIN, FV_MIN=CB.FV_MIN,
                                          BND_SCALE=CB.BND_SCALE, DELAY_SCALE=CB.DELAY_SCALE, EMPTY_BND=CB.EMPTY_BND,
                                          PREROLL=CB.PREROLL, SKIP_FIRST_ONSET=CB.SKIP_FIRST_ONSET, loss_delay=delay),
                       inits=[], n_forward=0, n_failed=0,
                       driver=dict(kind="gpu_calib (v05_calibrate setup, inits in lockstep, batched forward)",
                                   backend=backend, T=T, seed=seed, init_index=list(inits)))
            es = []
            for ii, x0p in zip(inits, x0s):
                es.append(cma.CMAEvolutionStrategy(CB.encode(x0p, space), sigma0,
                                                   dict(bounds=[0, 1], popsize=popsize, seed=seed + ii, maxiter=gens,
                                                        verbose=-9)))
                rec["inits"].append(dict(x0=x0p, history=[], best=None))
            log(f"calibration setup: {'+'.join(names)}, N {N}, popsize {popsize}, gens {gens}, inits {list(inits)}, "
                f"{len(space)} parameters, backend {backend}")
            # the initial points, once, as references (v05_calibrate: never candidates)
            t0 = time.time()
            res = forward(engine, model, [(x0, n) for x0 in x0s for n in names], N, T, chunk)
            evs = scorer.map([(n, c, led, delay) for n, c, led in res])
            refs = []
            for i, x0 in enumerate(x0s):
                e = {ev["name"]: ev for ev in evs[i * len(names):(i + 1) * len(names)]}
                refs.append(dict(params=x0, loss=float(np.mean([e[n]["loss"] for n in names])), evals=e))
                log(f"  reference init {inits[i]}: loss {refs[-1]['loss']:.4f}")
            rec["driver"]["x0_reference"] = refs
            log(f"  references: {time.time() - t0:.1f} s")
            state = dict(setup=setup, gen=0, es=es, rec=rec, np_rng=np.random.get_state(), wall=0.0)
            _save(out, state)
        es, rec = state["es"], state["rec"]
        done = 0
        while state["gen"] < gens and (max_gens is None or done < max_gens):
            g = state["gen"]
            t0 = time.time()
            X = [e.ask() for e in es]
            P = [[CB.decode(x, space) for x in Xi] for Xi in X]
            items = [(p, n) for Pi in P for p in Pi for n in names]
            res = forward(engine, model, items, N, T, chunk)
            tf = time.time() - t0
            evs = scorer.map([(n, c, led, delay) for n, c, led in res])
            ts = time.time() - t0 - tf
            rec["n_forward"] += len(items)
            k = 0
            for ii, (e, Xi, Pi) in enumerate(zip(es, X, P)):
                run_ = rec["inits"][ii]
                L = []
                for p in Pi:
                    ev = {x["name"]: x for x in evs[k:k + len(names)]}
                    k += len(names)
                    rec["n_failed"] += sum(1 for x in ev.values() if x.get("failed"))
                    li = float(np.mean([ev[n]["loss"] for n in names]))
                    L.append(li if np.isfinite(li) else 1e6)
                    if run_["best"] is None or li < run_["best"]["loss"]:
                        run_["best"] = dict(loss=li, params=p, gen=g, evals=ev)
                e.tell(Xi, L)
                run_["history"].append(dict(gen=g, loss_min=float(min(L)), loss_med=float(np.median(L)),
                                            mean=CB.decode(e.mean, space), sigma=float(e.sigma)))
                log(f"[{'+'.join(names)} init {inits[ii]}] gen {g}: min {min(L):.4f} med {np.median(L):.4f} "
                    f"best {run_['best']['loss']:.4f} sigma {e.sigma:.3f} fwd {rec['n_forward']}")
            log(f"gen {g}: {len(items)} forward runs at N {N}: solver {tf:.1f} s, metrics {ts:.1f} s")
            state["gen"] = g + 1
            state["wall"] += time.time() - t0
            state["np_rng"] = np.random.get_state()
            done += 1
            if state["gen"] >= gens:
                b = min(rec["inits"], key=lambda r: r["best"]["loss"])
                rec["best"] = dict(loss=b["best"]["loss"], params=b["best"]["params"],
                                   spread=[r["best"]["loss"] for r in rec["inits"]],
                                   params_by_init=[r["best"]["params"] for r in rec["inits"]])
            _save(out, state)
    finally:
        scorer.close()
    return state


def _save(out, state):
    tmp = os.path.join(out, "state.pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, os.path.join(out, "state.pkl"))
    _jdump(state["rec"], os.path.join(out, "calib.json"))


def load_state(out):
    with open(os.path.join(out, "state.pkl"), "rb") as f:
        return pickle.load(f)


def best_params(out):
    rec = load_state(out)["rec"]
    done = [r for r in rec["inits"] if r["best"] is not None]
    if not done:
        raise SystemExit("no generation finished yet")
    b = min(done, key=lambda r: r["best"]["loss"])
    return b["best"]["params"], b["best"]["loss"]


def write_best(out, quiet=False):
    from .v05_adapter import CLOSURE_KEYS, FSP2
    p, loss = best_params(out)
    clo = dict(FSP2)
    for k, v in p.items():
        k = "cp" if k == "c_p" else k
        if k in CLOSURE_KEYS:
            clo[k] = float(v)
    st = load_state(out)
    obj = dict(params=clo, inputs={k: v for k, v in p.items() if k not in CLOSURE_KEYS}, calib_loss=loss,
               generation=st["gen"], of=st["setup"]["gens"], names=st["setup"]["names"], N=st["setup"]["N"],
               note="closure for gpu_design --closure (u_exit / Q_scale are input nuisances of the real-action inlet, "
                    "not used by the pitcher-driven programs)")
    _jdump(obj, os.path.join(out, "best.json"))
    if not quiet:
        print(json.dumps(obj, indent=1), flush=True)
    return obj


def status(out):
    st = load_state(out)
    rec = st["rec"]
    print(f"generation {st['gen']} of {st['setup']['gens']}, forward runs {rec['n_forward']} (failed {rec['n_failed']}), "
          f"wall {st['wall'] / 60:.1f} min")
    for r in rec["driver"].get("x0_reference", []):
        print(f"  reference loss {r['loss']:.4f}")
    for i, r in enumerate(rec["inits"]):
        if r["best"]:
            print(f"  init {st['setup']['inits'][i]}: best {r['best']['loss']:.4f} (gen {r['best']['gen']}) "
                  + json.dumps({k: round(v, 4) for k, v in r["best"]["params"].items()}))


# =========================================================================== frozen-parameter prediction
def predict(out, N=128, N_fig=256, backend="cuda", T=None, procs=4, chunk=0):
    """Best (and FSp2 start) on the six clips at N, one batch; summary per set; contact sheets; final frames at N_fig."""
    from . import v05_calibrate as VC
    from .gpu_batch import BatchEngine
    log = _log_to(os.path.join(out, "log.txt"))
    sp = spec()
    model = CB.make_model(sp)
    best, loss = best_params(out)
    sets = dict(best=best, FSp2=VC.initial_points(model.space)[0])
    names = list(CB.CAL_AB) + list(CB.TEST)
    items = [(p, n) for p in sets.values() for n in names]
    scorer = Scorer(procs)
    try:
        t0 = time.time()
        res = forward(BatchEngine(N, backend=backend, log=log), model, items, N, T, chunk)
        evs = scorer.map([(n, c, led, False) for n, c, led in res])
    finally:
        scorer.close()
    log(f"predict: {len(items)} runs at N {N} in {time.time() - t0:.1f} s")
    rep = dict(calib_loss=loss, N=N, sets={})
    od = os.path.join(out, "predict")
    os.makedirs(od, exist_ok=True)
    k = 0
    full = {}
    for s, p in sets.items():
        ev = {}
        for n in names:
            ev[n] = evs[k]
            full[(s, n)] = res[k]
            k += 1
        rep["sets"][s] = dict(params=p, calibration_fit=_summary({n: ev[n] for n in CB.CAL_AB}),
                              test_prediction=_summary({n: ev[n] for n in CB.TEST}), evals=ev)
        log(f"  {s}: calibration fit {rep['sets'][s]['calibration_fit']['mean']['loss']:.4f}, test prediction "
            f"{rep['sets'][s]['test_prediction']['mean']['loss']:.4f}")
    _jdump(rep, os.path.join(od, "predict.json"))
    for n in names:                                    # contact sheets of the best set
        nm, c, led = full[("best", n)]
        try:
            CB.contact_sheet(_clip(n), CB.observe_model(c, _clip(n)["M"]), os.path.join(od, f"process_{n}.png"),
                             title=f"calibrated closure (best), N {N}")
        except Exception as e:
            log(f"  contact sheet {n}: {e!r}")
    if N_fig:
        final_frames(out, sets, names, N_fig, backend, T, chunk, model, log)
    return rep


def _summary(evs):
    rows = {}
    for n, e in evs.items():
        rows[n] = dict(loss=e["loss"], iou=(e.get("iou") or {}).get("value"), dice=(e.get("dice") or {}).get("value"),
                       boundary_D=(e.get("boundary_D") or {}).get("value"), final_iou=(e.get("final") or {}).get("iou"),
                       failed=bool(e.get("failed")))
    ok = [r for r in rows.values() if np.isfinite(r["loss"])]
    mean = {k: float(np.mean([r[k] for r in ok if r[k] is not None])) if any(r[k] is not None for r in ok) else None
            for k in ("loss", "iou", "dice", "boundary_D", "final_iou")}
    return dict(patterns=rows, mean=mean)


def final_frames(out, sets, names, N, backend, T, chunk, model, log):
    """Last clip frame only, at N: model (through the observation operator) next to the video, per set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .gpu_batch import BatchEngine, ClipMember
    mem = []
    for s, p in sets.items():
        for n in names:
            clip = _clip(n)
            mem.append(ClipMember(model, clip["script"], p, N, CB.sample_times(clip)[-1:], T_max=T, tag=(s, n)))
    t0 = time.time()
    BatchEngine(N, backend=backend, log=log).run(mem, chunk=chunk)
    log(f"final frames: {len(mem)} runs at N {N} in {time.time() - t0:.1f} s")
    fig, ax = plt.subplots(1 + len(sets), len(names), figsize=(2.2 * len(names), 2.3 * (1 + len(sets))), dpi=120,
                           squeeze=False)
    for j, n in enumerate(names):
        clip = _clip(n)
        ax[0, j].imshow(clip["res"]["white"][-1].astype(float), origin="lower", cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(n, fontsize=8)
    for m in mem:
        s, n = m.tag
        i = 1 + list(sets).index(s)
        j = names.index(n)
        wT = CB.observe_model(m.result["c"], _clip(n)["M"])
        ax[i, j].imshow(wT[-1].astype(float), origin="lower", cmap="gray", vmin=0, vmax=1)
    for i, lab in enumerate(["video"] + list(sets)):
        ax[i, 0].set_ylabel(lab, fontsize=8)
    for x in ax.ravel():
        x.set_xticks([]); x.set_yticks([])
    fig.suptitle(f"Last frame: video vs model (observation operator of calib), N = {N}; calibrated on "
                 f"{'+'.join(CB.CAL_AB)}, tested on the others", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fp = os.path.join(out, "predict", f"final_N{N}.png")
    fig.savefig(fp)
    plt.close(fig)
    log(fp)


# =========================================================================== pitcher constants from scale data
LAW_SPACE = dict(C_d=(0.4, 0.9), C_v=(0.8, 1.1), h_k=(0.0, 2e-3))


def load_scale(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["scale_on"] = str(d.get("scale_on", "cup"))
    d["rho"] = float(d.get("rho", 1.03))
    d["tau_scale"] = float(d.get("tau_scale", 0.0))
    d["name"] = os.path.splitext(os.path.basename(path))[0]
    for k in ("t", "tip", "yaw", "tilt", "roll", "fill_mL", "alpha_deg", "m_t", "m_g"):
        if k not in d:
            raise ValueError(f"{path}: missing key {k!r} (format: gpu_calib docstring)")
    return d


def _lag(t, y, tau):
    if tau <= 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        a = math.exp(-(t[i] - t[i - 1]) / tau)
        out[i] = a * out[i - 1] + (1 - a) * y[i]
    return out


def scale_model(d, E):
    """Predicted scale reading [g] at d['m_t'] from a ForwardPitcher.run result (tare: first reading)."""
    from .pitcher import ForwardPitcher
    dt = float(E["dt"])
    if d["scale_on"] == "pitcher":
        tt, V = E["t"] + 0.5 * dt, E["V_P"]
        m = d["rho"] * V * 1e6                            # pitcher + milk (tare removes the pitcher)
    else:
        t0 = float(np.min(d["t"])) - 0.5 * dt
        n = int(math.ceil((float(np.max(d["m_t"])) - t0) / dt)) + 2
        R = ForwardPitcher.arrivals(E, t0, n, dt)
        tt = t0 + (np.arange(n) + 1) * dt
        m = d["rho"] * R["V_cup"] * 1e6
    m = _lag(tt, m, d["tau_scale"])
    y = np.interp(d["m_t"], tt, m)
    return y - y[0]


def _seqs(ds):
    return [dict(t=np.asarray(d["t"], float) - 0.5 * float(d["t"][1] - d["t"][0]), tip=d["tip"], yaw=d["yaw"],
                 tilt=d["tilt"], roll=d["roll"], V0=float(d["fill_mL"]) * 1e-6, dt=float(d["t"][1] - d["t"][0]))
            for d in ds]


def law_loss(ds, law, backend):
    from .pitcher import ForwardLaw, ForwardPitcher
    from . import s1_forward as SF
    errs, per = [], {}
    by_alpha = {}
    for i, d in enumerate(ds):
        by_alpha.setdefault(float(d["alpha_deg"]), []).append(i)
    for alpha, idx in by_alpha.items():
        fp = ForwardPitcher(SF.geometry(alpha), ForwardLaw(**law))
        Es = SF.run_pitcher(_seqs([ds[i] for i in idx]), fp, backend)
        for i, E in zip(idx, Es):
            d = ds[i]
            y = scale_model(d, E)
            e = y - (np.asarray(d["m_g"], float) - float(d["m_g"][0]))
            errs.append(e)
            per[d["name"]] = float(np.sqrt(np.mean(e ** 2)))
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2))), per


def fit_pitcher(paths, out, fit=("C_d", "h_k"), backend="gpu", gens=12, popsize=8, seed=1, log=print):
    import cma
    os.makedirs(out, exist_ok=True)
    ds = [load_scale(p) for p in paths]
    keys = list(fit)
    base = dict(C_d=0.60, C_v=1.0, h_k=0.0)                 # ForwardLaw defaults (V1)
    unit = lambda k, v: (v - LAW_SPACE[k][0]) / (LAW_SPACE[k][1] - LAW_SPACE[k][0])
    val = lambda k, u: LAW_SPACE[k][0] + (LAW_SPACE[k][1] - LAW_SPACE[k][0]) * min(max(u, 0.0), 1.0)
    L0, per0 = law_loss(ds, base, backend)
    log(f"pitcher fit: {len(ds)} pours, start (V1 defaults) RMS {L0:.3f} g")
    hist = []
    best = (L0, dict(base), per0)
    if len(keys) == 1:                                      # 1-D: bounded scalar search (cheap, deterministic)
        from scipy.optimize import minimize_scalar
        k = keys[0]

        def f(u):
            law = dict(base, **{k: val(k, u)})
            L, per = law_loss(ds, law, backend)
            hist.append(dict(law=law, rms_g=L))
            nonlocal best
            if L < best[0]:
                best = (L, law, per)
            return L
        minimize_scalar(f, bounds=(0.0, 1.0), method="bounded", options=dict(xatol=1e-3, maxiter=25))
    else:
        es = cma.CMAEvolutionStrategy([unit(k, base[k]) for k in keys], 0.25,
                                      dict(bounds=[0, 1], popsize=popsize, seed=seed, maxiter=gens, verbose=-9))
        while not es.stop():
            X = es.ask()
            L = []
            for x in X:
                law = dict(base, **{k: val(k, u) for k, u in zip(keys, x)})
                Li, per = law_loss(ds, law, backend)
                L.append(Li)
                hist.append(dict(law=law, rms_g=Li))
                if Li < best[0]:
                    best = (Li, law, per)
            es.tell(X, L)
            log(f"  gen {len(hist) // popsize}: min {min(L):.3f} g, best {best[0]:.3f} g {best[1]}")
    res = dict(law={k: best[1][k] for k in ("C_d", "C_v", "h_k")}, fitted=keys, rms_g=best[0], per_file=best[2],
               start=dict(law=base, rms_g=L0, per_file=per0), files=[os.path.abspath(p) for p in paths], n_calls=len(hist),
               history=hist, bounds={k: LAW_SPACE[k] for k in keys})
    _jdump(res, os.path.join(out, "pitcher_law.json"))
    log(f"pitcher law {res['law']} RMS {best[0]:.3f} g -> {os.path.join(out, 'pitcher_law.json')}")
    return res


def make_scale(pattern, law, path, T=None, every=0.05, scale_on="cup", backend="cpu", noise_g=0.0, seed=0):
    """A synthetic scale file from the S1 poses of `pattern` poured with ForwardLaw(**law) (format test / recovery)."""
    from .pitcher import ForwardLaw, ForwardPitcher
    from . import s1_forward as SF
    P = SF.load_poses(pattern)
    if T is not None:
        sel = P["t"] <= T
        P = {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(sel) else v) for k, v in P.items()}
    d = dict(t=np.asarray(P["t"], float), tip=P["tip"], yaw=P["yaw"], tilt=P["tilt"], roll=P["roll"],
             fill_mL=float(P["fill_mL"]), alpha_deg=float(P["alpha_deg"]), scale_on=scale_on, rho=1.03, tau_scale=0.0)
    d["m_t"] = np.arange(float(P["t"][0]), float(P["t"][-1]), every)
    d["m_g"] = np.zeros_like(d["m_t"])
    fp = ForwardPitcher(SF.geometry(d["alpha_deg"]), ForwardLaw(**law))
    E = SF.run_pitcher(_seqs([d]), fp, backend)[0]
    y = scale_model(dict(d, name="x"), E)
    d["m_g"] = y + 100.0 + np.random.default_rng(seed).normal(0.0, noise_g, len(y))
    np.savez_compressed(path, **d)
    return path


# =========================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "status", "predict", "best", "fit_pitcher", "make_scale"])
    ap.add_argument("--out", default="results/gpu_calib")
    ap.add_argument("--names", default=",".join(CB.CAL_AB))
    ap.add_argument("--N", type=int, default=None, help="run: 96 (prereg); predict: 128")
    ap.add_argument("--N-fig", type=int, default=256)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--sigma0", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--inits", default="0,1,2")
    ap.add_argument("--T", type=float, default=None, help="truncate every run at T [s] of model time (smoke tests)")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--procs", type=int, default=4, help="processes for the calib metrics on the host")
    ap.add_argument("--max-gens", type=int, default=None)
    ap.add_argument("--solver-backend", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--pitcher-backend", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--scale", default="", help="fit_pitcher: comma-separated scale files")
    ap.add_argument("--fit", default="C_d,h_k")
    ap.add_argument("--pattern", default="heart")
    ap.add_argument("--law", default="C_d=0.6", help="make_scale: k=v,... of ForwardLaw")
    ap.add_argument("--file", default="scale_synthetic.npz")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.out, a.names.split(","), a.gens, a.popsize, a.N or 96, a.sigma0, a.seed,
            tuple(int(v) for v in a.inits.split(",")), a.solver_backend, a.T, a.chunk, a.procs, a.max_gens)
        status(a.out)
        write_best(a.out)
    elif a.cmd == "status":
        status(a.out)
    elif a.cmd == "best":
        write_best(a.out)
    elif a.cmd == "predict":
        predict(a.out, a.N or 128, a.N_fig, a.solver_backend, a.T, a.procs, a.chunk)
    elif a.cmd == "fit_pitcher":
        fit_pitcher([p for p in a.scale.split(",") if p], a.out, tuple(a.fit.split(",")), a.pitcher_backend)
    elif a.cmd == "make_scale":
        law = {k: float(v) for k, v in (kv.split("=") for kv in a.law.split(",") if kv)}
        print(make_scale(a.pattern, law, a.file, T=a.T, backend=a.pitcher_backend))


if __name__ == "__main__":
    main()
