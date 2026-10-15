"""GPU action re-design of the S1 patterns against the forward pose-driven pitcher (V1 mode B), with the aligned model
(Params defaults: symmetric viscosity, off-cup accounting).

Why: the S1 actions were designed for the old pitcher; replayed on the forward pitcher they pour 4-6 mL more and
layered_heart / tulip / leaf / swan get 20-35 % more white area.  Here the per-stroke actions are re-optimised with the
forward pitcher in the loop: candidate actions -> pose sequences -> forward pitcher (batched, GPU) -> arrival records
-> S1 programs (s1_forward.FwdProgram) -> solver (BatchCudaSolver, one batch per generation).

DECISION VARIABLES (per pattern; path, yaw, roll, timings of the S1 poses fixed, as in the S1 prereg 'not_allowed'):
  per flowing stroke k (prereg flowing strokes)
    tilt_k   [deg]  tilt offset of the stroke (the forward-pitcher counterpart of the S1 flow scale s_k: the flow is set
                    by the tilt), s1_forward.tilt_offset profile (held inside the stroke, linear across a bridge, 0.08 s
                    ramps between adjacent strokes, tapered to 0 after the last stroke); bounds [-6, 6] (= the flow-error
                    calibration's MAX_OFFSET)
    z_k      [mm]   tip-height offset (the S1 design's wanted height z_k), s1_forward.profile with the 0.11 m/s ztr
                    transitions; bounds [-4, 8]
    shift_k  [ms]   flow timing: the tilt history shifted against the path inside the stroke (as the P5/P6 hand errors,
                    0.08 s ramps); bounds [-80, 80]
  per pattern
    fill     [mL]   start fill of the pitcher, bounds [100, 125] (the S1 fills)
  (--vars selects a subset; defaults: all four.)
OBJECTIVE (lower = better), per candidate, every term from existing code (design_metrics docstring for the sources):
  J = w_s1 J_S1 + w_shape L_shape + w_rob R + w_pen P
  J_S1     v05_pourable.finalize's S1 design J on the delivered inlet (per-stroke chi vs the V0.5 chi target, flow scale
           vs the V0.5 stroke volume, momentum vs FS), from the solver frames of the nominal run
  L_shape  2 - IoU(c > 0.4) - r_hp against the V0.5 reference field (allphys.rank's screening score)
  R        robustness: mean over the hand-error draws (--rob, default the prereg R1, R2, R3, which combine height
           +-3 mm, flow x0.8..1.2 and +-50 ms per stroke) of (1 - IoU(c > 0.5) with the candidate's own nominal run)
           + |ln white-area ratio|  (autodl_v05.compare_c); the errors are applied on top of the candidate's poses
           with autodl_v05's machinery (s1_forward.perturb_poses, calibrate_flow on the GPU pitcher)
  P        constraint penalties: per-stroke flow-weighted tilt in [63, 81] deg (the S1 design range); flow-weighted tip
           height in the S1 class ranges (low 3-25 mm, cut 12-70 mm: DS_ZLOW / DS_ZCUT); lowest tip while pouring >= 3
           mm; pitcher volume at the end >= 20 mL (DS_VMIN); ideal-hand limits (v05_pourable.hand_check: tilt rate /
           acceleration, tip speed / acceleration, yaw rate; only the excess over max(limit, value of the S1 poses))
  Weights default w_s1 10, w_shape 1, w_rob 0.5, w_pen 1 (judgement call: J_S1 of the S1 design is ~0.01 per pattern, L_shape
  ~0.3-0.8, so 10 J_S1 and L_shape are of one order; robustness at half weight).
OPTIMISER: CMA-ES (pycma, the optimiser of calib.calibrate / v05_calibrate) in the unit box, all requested patterns
in lockstep; one generation = ONE batched evaluation: every candidate x (1 + |rob|) pose sequences of every pattern
through the GPU pitcher (flow-error secant iterations batched), then all programs in one BatchCudaSolver batch at the
screening resolution (--N-lo, default 128).  Multi-fidelity: every --confirm-every generations the --topk best
screened candidates of each pattern (and the CMA mean at the end) are re-run at --N-hi (default 256) from the stored
pitcher records; the design reported is the best confirmed candidate.  Deterministic (cma seed = seed + pattern index,
numpy's global RNG saved in the checkpoint); the checkpoint (state.pkl, atomic) is written after every generation and
`run` resumes from it (AutoDL instances get interrupted).

    python -m latte_imex.gpu_design export                 # once, on the dev machine: reference fields + S1 refs
    python -m latte_imex.gpu_design run --out DIR [--patterns tulip,leaf] [--gens 20 --popsize 12] [--closure FILE]
    python -m latte_imex.gpu_design status --out DIR
    python -m latte_imex.gpu_design write --out DIR        # best designs -> DIR/data (poses_<p>.npz, program_<p>.json)
    python -m latte_imex.gpu_design robust --out DIR       # autodl_v05 robust on the redesigned poses (+ figure)
    python -m latte_imex.gpu_design final --out DIR [--N 256,512] [--video]   # finals / videos / comparison figure
Backends: --solver-backend cuda|cpu, --pitcher-backend gpu|cpu (cpu = the reference implementations, slow).
--closure: 's1' (default: the S1 programs' closure) or a JSON file (gpu_calib best.json, or {key: value}); the
closure keys of v05_adapter.CLOSURE_KEYS are used (kappa_Q also sets the program's kQ).
Outputs under --out: state.pkl, design.json (per pattern: best screened / confirmed, decoded actions, metrics, history),
log.txt; `write`: data/; `robust`: robust_check/; `final`: final/ + fig_design_compare.png.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import time

import numpy as np

from . import s1_forward as SF
from . import design_metrics as DM

VARS = ("tilt", "z", "shift", "fill")
BOUNDS = dict(tilt=(-6.0, 6.0), z=(-4.0, 8.0), shift=(-80.0, 80.0), fill=(100.0, 125.0))
UNITS = dict(tilt="deg", z="mm", shift="ms", fill="mL")
ROB_DEFAULT = ("R1", "R2", "R3")
TILT_RANGE = (63.0, 81.0)                       # [deg] S1 design range (flow-weighted per stroke)
Z_RANGE = dict(low=(3.0, 25.0), cut=(12.0, 70.0))   # [mm] v05_pourable DS_ZLOW / DS_ZCUT
Z_FLOOR = 3.0                                   # [mm] lowest tip while pouring
V_END_MIN = 20.0                                # [mL] v05_pourable DS_VMIN
FAIL_J = 1e6
CLOSURE_KEYS = ("cp", "beta", "nu", "D", "kappa_Q", "kappa_c", "kappa_t", "kappa_r", "B_dep", "p_dep")
DEFAULTS = dict(gens=20, popsize=12, sigma0=0.2, N_lo=128, N_hi=256, topk=2, confirm_every=2, w_s1=10.0, w_shape=1.0,
                w_rob=0.5, w_pen=1.0, seed=1, rob=",".join(ROB_DEFAULT), vars=",".join(VARS), T=None, closure="s1",
                flow_tol=0.003, chunk=0)


def _log_to(path):
    def log(*a):
        line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
        print(line, flush=True)
        if path:
            with open(path, "a") as f:
                f.write(line + "\n")
    return log


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o), ensure_ascii=False)
    os.replace(tmp, path)


# =========================================================================== search space
def flowing(meta):
    return sorted(k - 1 for k in meta["flowing"])


def space_of(meta, vars_=VARS):
    """[(var, stroke index or None, lo, hi)] in a fixed order (per stroke tilt, z, shift; then fill)."""
    sp = []
    for k in flowing(meta):
        for v in ("tilt", "z", "shift"):
            if v in vars_:
                sp.append((v, k, *BOUNDS[v]))
    if "fill" in vars_:
        sp.append(("fill", None, *BOUNDS["fill"]))
    return sp


def decode(x, sp, P):
    d = dict(tilt={}, z={}, shift={}, fill=float(P["fill_mL"]))
    for xi, (v, k, lo, hi) in zip(x, sp):
        val = lo + (hi - lo) * min(max(float(xi), 0.0), 1.0)
        if v == "fill":
            d["fill"] = val
        else:
            d[v][k] = val
    return d


def x0_of(sp, P):
    """Unit coordinates of the S1 poses (all offsets 0, the S1 fill)."""
    x = []
    for v, k, lo, hi in sp:
        val = float(P["fill_mL"]) if v == "fill" else 0.0
        x.append((min(max(val, lo), hi) - lo) / (hi - lo))
    return np.array(x)


def truncate_poses(P, T):
    """Poses up to T + 0.4 s (smoke tests: the pitcher then runs only that far)."""
    if T is None:
        return P
    sel = np.asarray(P["t"]) <= T + 0.4
    return {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(sel) else v) for k, v in P.items()}


def design_poses(P, meta, d):
    """The S1 pose dict P with the actions d applied (same format as s1_forward.load_poses; design_* = the applied
    profiles).  z and shift through s1_forward.perturb_poses' profiles, tilt through tilt_offset, fill replaced."""
    fl = flowing(meta)
    pert = {k: (d["z"].get(k, 0.0), 1.0, d["shift"].get(k, 0.0)) for k in fl}
    offs = {k: math.radians(d["tilt"].get(k, 0.0)) for k in fl}
    s = SF.perturb_poses(P, meta, pert, offs)
    out = dict(P)
    out.update(tip=s["tip"], tilt=s["tilt"], fill_mL=np.float64(d["fill"]), design_dz=s["dz"], design_delta=s["delta"],
               design_dtilt=s["dtilt"])
    return out


# =========================================================================== closure
def load_closure(spec):
    """None for 's1' (the programs' own parameters), else {key: value} of CLOSURE_KEYS from a JSON file (gpu_calib
    best.json: its 'params'; v05_calibrate calib.json: 'best'.'params'; or a flat dict)."""
    if spec in (None, "", "s1"):
        return None
    with open(spec) as f:
        j = json.load(f)
    if "best" in j and isinstance(j["best"], dict) and "params" in j["best"]:
        j = j["best"]["params"]
    elif "params" in j:
        j = j["params"]
    out = {}
    for k, v in j.items():
        k = "cp" if k == "c_p" else k
        if k in CLOSURE_KEYS:
            out[k] = float(v)
    return out


def meta_with_closure(meta, closure):
    if not closure:
        return meta
    m = copy.deepcopy(meta)
    m["config"]["parameters"].update(closure)
    if "kappa_Q" in closure:
        m["kQ"] = float(closure["kappa_Q"])
    return m


# =========================================================================== pitcher stage
_FP = {}


def forward_pitcher(alpha_deg, law=None):
    from .pitcher import ForwardLaw, ForwardPitcher
    key = (float(alpha_deg), json.dumps(law or {}, sort_keys=True))
    if key not in _FP:
        _FP[key] = ForwardPitcher(SF.geometry(alpha_deg), ForwardLaw(**(law or {})))
    return _FP[key]


def pitcher_stage(items, backend, law=None, tol=0.003, log=print):
    """items: list of (P, meta, pert).  Forward pitcher (flow errors calibrated, batched) -> list of
    dict(R, E, seq, info)."""
    from .pitcher import ForwardPitcher
    out = [None] * len(items)
    for alpha in sorted({float(it[1]["alpha_deg"]) for it in items}):
        sel = [i for i, it in enumerate(items) if float(it[1]["alpha_deg"]) == alpha]
        res = SF.calibrate_flow([items[i] for i in sel], forward_pitcher(alpha, law), backend, tol=tol,
                                log=lambda *a: None)
        for i, r in zip(sel, res):
            P, meta, pert = items[i]
            E = r["E"]
            R = ForwardPitcher.arrivals(E, 0.0, meta["n_out"], meta["dt"])
            out[i] = dict(R=R, E=E, seq=r["seq"], info=dict(flow_calibrated=bool(r["converged"]), calib_iters=int(r["iters"]),
                                                             tilt_offset_deg={str(k + 1): math.degrees(v) for k, v in r["offs"].items()},
                                                             stroke_V_mL={str(k + 1): v for k, v in r["V"].items()}))
    return out


def pitcher_summary(P, meta, E, seq):
    """Per flowing stroke: emitted volume, Q-weighted tilt [deg] and tip height [mm], lowest tip while pouring; pitcher
    end volume, poured volume."""
    fl = flowing(meta)
    ev = meta["events"]
    t = E["t"]
    Q = np.asarray(E["Q_o"], float)
    tilt = np.degrees(np.asarray(seq["tilt"], float))
    tz = np.asarray(seq["tip"], float)[:, 2] * 1e3
    rows = {}
    for i, k in enumerate(fl):
        a = -np.inf if i == 0 else ev[k]
        b = np.inf if i == len(fl) - 1 else ev[k + 1]
        m = (t >= a) & (t < b)
        w = Q[m]
        on = m & (Q > 0)
        rows[k + 1] = dict(V_mL=float(w.sum() * E["dt"] * 1e6), cls=meta["cls"][k],
                           tilt_w=float(np.sum(w * tilt[m]) / w.sum()) if w.sum() > 0 else None,
                           z_w_mm=float(np.sum(w * tz[m]) / w.sum()) if w.sum() > 0 else None,
                           z_min_mm=float(tz[on].min()) if on.any() else None)
    return dict(strokes=rows, V_end_mL=float(E["V_P"][-1] * 1e6), poured_mL=float(Q.sum() * E["dt"] * 1e6),
                fill_mL=float(P["fill_mL"]))


_HAND_BASE = {}


def hand_values(P):
    from .v05_pourable import hand_check
    h = hand_check(np.asarray(P["t"], float), tilt_deg=np.degrees(np.asarray(P["tilt"], float)), tip=np.asarray(P["tip"], float),
                   yaw=np.asarray(P["yaw"], float), hand="ideal")
    return {k: h[k] for k in h["limits"]}, h["limits"]


def penalties(pattern, P0, Pd, psum):
    """Constraint penalties (module docstring) and their parts."""
    parts = dict(tilt=0.0, z=0.0, z_floor=0.0, v_end=0.0, hand=0.0)
    for n, r in psum["strokes"].items():
        if r["tilt_w"] is not None:
            ex = max(0.0, r["tilt_w"] - TILT_RANGE[1], TILT_RANGE[0] - r["tilt_w"])
            parts["tilt"] += (ex / 2.0) ** 2
        if r["z_w_mm"] is not None:
            lo, hi = Z_RANGE["cut" if r["cls"] == "cut" else "low"]
            ex = max(0.0, r["z_w_mm"] - hi, lo - r["z_w_mm"])
            parts["z"] += (ex / 2.0) ** 2
        if r["z_min_mm"] is not None and r["z_min_mm"] < Z_FLOOR:
            parts["z_floor"] += (Z_FLOOR - r["z_min_mm"]) ** 2
    if psum["V_end_mL"] < V_END_MIN:
        parts["v_end"] += ((V_END_MIN - psum["V_end_mL"]) / 5.0) ** 2
    if pattern not in _HAND_BASE:
        _HAND_BASE[pattern] = hand_values(P0)
    base, lim = _HAND_BASE[pattern]
    val, _ = hand_values(Pd)
    for k, v in val.items():
        ex = max(0.0, v - max(lim[k], base[k]))
        parts["hand"] += (ex / lim[k] / 0.05) ** 2
    return sum(parts.values()), parts


# =========================================================================== one batched evaluation
class Evaluator:
    """Candidates -> J.  Holds the base poses / metas, the engines (one per N, device buffers reused) and the cfg."""

    def __init__(self, patterns, cfg, solver_backend="cuda", pitcher_backend="gpu", data=SF.DATA, log=print):
        self.cfg = cfg
        self.patterns = list(patterns)
        self.sb, self.pb = solver_backend, pitcher_backend
        self.log = log
        self.rob = [r for r in (cfg["rob"] or "").split(",") if r and r != "nominal"]
        self.vars = tuple(v for v in cfg["vars"].split(",") if v)
        closure = load_closure(cfg["closure"])
        self.closure = closure
        self.law = cfg.get("pitcher_law") or None
        self.base, self.meta, self.space = {}, {}, {}
        for p in self.patterns:
            P = truncate_poses(SF.load_poses(p, data), cfg["T"])
            meta = SF.load_meta(p, data)
            self.base[p] = P
            self.meta[p] = meta_with_closure(meta, closure)
            self.space[p] = space_of(meta, self.vars)
        self.engines = {}

    def engine(self, N):
        from .gpu_batch import BatchEngine
        if N not in self.engines:
            self.engines[N] = BatchEngine(N, backend=self.sb, log=self.log)
        return self.engines[N]

    def kappa_t(self, p):
        return float(self.meta[p]["config"]["parameters"]["kappa_t"])

    def pitcher(self, cands):
        """cands: list of (pattern, x).  Returns per candidate dict(d, Pd, runs={name: pitcher result})."""
        items, keys, out = [], [], []
        for ci, (p, x) in enumerate(cands):
            d = decode(x, self.space[p], self.base[p])
            Pd = design_poses(self.base[p], self.meta[p], d)
            out.append(dict(pattern=p, x=np.asarray(x, float), d=d, Pd=Pd, runs={}))
            for nm in ["nominal"] + self.rob:
                items.append((Pd, self.meta[p], SF.prereg_pert(nm, p)))
                keys.append((ci, nm))
        res = pitcher_stage(items, self.pb, self.law, self.cfg["flow_tol"], self.log)
        for (ci, nm), r in zip(keys, res):
            out[ci]["runs"][nm] = r
        for c in out:
            r = c["runs"]["nominal"]
            c["psum"] = pitcher_summary(c["Pd"], self.meta[c["pattern"]], r["E"], r["seq"])
            c["pen"], c["pen_parts"] = penalties(c["pattern"], self.base[c["pattern"]], c["Pd"], c["psum"])
            for r in c["runs"].values():               # keep only the arrival records (host memory: ~4 MB per run)
                r.pop("E", None)
                r.pop("seq", None)
            c.pop("Pd", None)
        return out

    def solve(self, cands, N):
        """Solver stage for pitcher-evaluated candidates at N; fills c['eval'][N] and returns the J values."""
        from .gpu_batch import ProgMember
        from .solver import Params
        mem, keys = [], []
        for ci, c in enumerate(cands):
            p = c["pattern"]
            for nm, r in c["runs"].items():
                prog = SF.FwdProgram(p, r["R"], N, meta=self.meta[p])
                sc = SF.ProgScript(prog, p)
                mem.append(ProgMember(sc, Params(**sc.params), T_max=self.cfg["T"], tag=(ci, nm)))
                keys.append((ci, nm))
        t0 = time.time()
        self.engine(N).run(mem, chunk=self.cfg.get("chunk") or 0)
        wall = time.time() - t0
        by = {}
        for (ci, nm), m in zip(keys, mem):
            by.setdefault(ci, {})[nm] = m.result
        Js = []
        for ci, c in enumerate(cands):
            ev = self.score(c, by[ci], N)
            c.setdefault("eval", {})[N] = ev
            Js.append(ev["J"])
        return Js, wall, len(mem)

    def score(self, c, res, N):
        cfg = self.cfg
        p = c["pattern"]
        nom = res["nominal"]
        if not nom["ok"]:
            return dict(J=FAIL_J, failed=nom["error"] or "nominal run failed")
        sh = DM.shape_vs_ref(nom["c"], p, N, struct=True)
        s1 = DM.s1_terms(p, nom["strokes"], self.kappa_t(p))
        rob = {}
        mask = DM.load_ref(p, N)[1]
        for nm in self.rob:
            r = res[nm]
            if not r["ok"]:
                rob[nm] = dict(failed=r["error"], term=2.0)
                continue
            q = DM.compare_nominal(r["c"], nom["c"], mask)
            q["term"] = (1.0 - q["iou_white"]) + abs(math.log(max(q["white_area_ratio"], 1e-3)))
            q["flow_calibrated"] = c["runs"][nm]["info"]["flow_calibrated"]
            rob[nm] = q
        R = float(np.mean([v["term"] for v in rob.values()])) if rob else 0.0
        J = cfg["w_s1"] * s1["J"] + cfg["w_shape"] * sh["L_shape"] + cfg["w_rob"] * R + cfg["w_pen"] * c["pen"]
        return dict(J=float(J), J_S1=s1["J"], S1_parts=s1["parts"], S1_rows=s1["rows"], L_shape=sh["L_shape"], shape=sh,
                    R=R, rob=rob, pen=c["pen"], pen_parts=c["pen_parts"], pitcher=c["psum"],
                    ledger=dict(deposited=nom["ledger"]["deposited"], spilled=nom["ledger"]["spilled"],
                                layer_error=nom["ledger"]["layer_error"]),
                    stats=dict(steps=nom["stats"]["steps"], retries=nom["stats"]["retries"]), N=N)


# =========================================================================== CMA-ES driver
def cfg_hash(cfg, patterns):
    keep = {k: cfg[k] for k in sorted(cfg) if k not in ("gens", "chunk", "closure")}
    return hashlib.sha1(json.dumps(dict(cfg=keep, patterns=list(patterns)), sort_keys=True, default=str).encode()).hexdigest()[:16]


def _decoded(x, sp, P):
    d = decode(x, sp, P)
    return dict(tilt_deg={str(k + 1): v for k, v in d["tilt"].items()}, z_mm={str(k + 1): v for k, v in d["z"].items()},
                shift_ms={str(k + 1): v for k, v in d["shift"].items()}, fill_mL=d["fill"])


def _slim(ev):
    return {k: v for k, v in ev.items() if k not in ("S1_rows",)} | dict(S1_rows={str(k): v for k, v in ev.get("S1_rows", {}).items()})


def run(out, patterns, cfg, solver_backend="cuda", pitcher_backend="gpu", data=SF.DATA, max_gens=None):
    """CMA-ES in lockstep over the patterns, resumable (see module docstring).  max_gens stops this session after that
    many new generations (tests)."""
    import cma
    os.makedirs(out, exist_ok=True)
    log = _log_to(os.path.join(out, "log.txt"))
    h = cfg_hash(cfg, patterns)
    st_path = os.path.join(out, "state.pkl")
    ev = Evaluator(patterns, cfg, solver_backend, pitcher_backend, data, log)
    state = None
    if os.path.exists(st_path):
        with open(st_path, "rb") as f:
            state = pickle.load(f)
        if state["hash"] != h:
            raise SystemExit(f"{st_path} was written with another setup (hash {state['hash']} != {h}); use another --out")
        np.random.set_state(state["np_rng"])
        log(f"resume at generation {state['gen']} ({', '.join(patterns)})")
    else:
        es, rec = {}, {}
        for i, p in enumerate(patterns):
            sp = ev.space[p]
            x0 = x0_of(sp, ev.base[p])
            es[p] = cma.CMAEvolutionStrategy(list(x0), cfg["sigma0"], dict(bounds=[0, 1], popsize=cfg["popsize"],
                                                                           seed=cfg["seed"] + i, maxiter=cfg["gens"], verbose=-9))
            rec[p] = dict(pattern=p, space=[dict(var=v, stroke=None if k is None else k + 1, lo=lo, hi=hi, unit=UNITS[v])
                                            for v, k, lo, hi in sp], x0=list(x0), history=[], best_screen=None,
                          best_confirm=None, confirms=[], n_eval=0, n_failed=0)
        state = dict(hash=h, cfg=cfg, patterns=list(patterns), gen=0, es=es, rec=rec, np_rng=None, wall=0.0,
                     backends=dict(solver=solver_backend, pitcher=pitcher_backend))
        # reference: the S1 poses themselves (x0), screened and confirmed once (not a CMA sample)
        log(f"design setup {h}: patterns {', '.join(patterns)}; vars {cfg['vars']}; rob {cfg['rob'] or '-'}; "
            f"popsize {cfg['popsize']}, gens {cfg['gens']}, N {cfg['N_lo']}/{cfg['N_hi']}; closure {cfg['closure']}")
        cands = ev.pitcher([(p, rec[p]["x0"]) for p in patterns])
        ev.solve(cands, cfg["N_lo"])
        ev.solve(cands, cfg["N_hi"])
        for c in cands:
            r = rec[c["pattern"]]
            r["reference_S1"] = dict(x=list(c["x"]), decoded=_decoded(c["x"], ev.space[c["pattern"]], ev.base[c["pattern"]]),
                                     screen=_slim(c["eval"][cfg["N_lo"]]), confirm=_slim(c["eval"][cfg["N_hi"]]))
            log(f"  {c['pattern']}: S1 poses J {c['eval'][cfg['N_lo']]['J']:.4f} (N {cfg['N_lo']}) / "
                f"{c['eval'][cfg['N_hi']]['J']:.4f} (N {cfg['N_hi']}); poured {c['psum']['poured_mL']:.1f} mL")
        state["np_rng"] = np.random.get_state()
        _save(out, state)
    es, rec = state["es"], state["rec"]
    done_now = 0
    while state["gen"] < cfg["gens"] and (max_gens is None or done_now < max_gens):
        g = state["gen"]
        t0 = time.time()
        X = {p: es[p].ask() for p in patterns}
        cands_x = [(p, x) for p in patterns for x in X[p]]
        cands = ev.pitcher(cands_x)
        tp = time.time() - t0
        Js, ws, nm = ev.solve(cands, cfg["N_lo"])
        i = 0
        top = []
        for p in patterns:
            n = len(X[p])
            cs = cands[i:i + n]
            L = [float(c["eval"][cfg["N_lo"]]["J"]) for c in cs]
            es[p].tell(X[p], [v if np.isfinite(v) else FAIL_J for v in L])
            r = rec[p]
            r["n_eval"] += n
            r["n_failed"] += sum(1 for c in cs if c["eval"][cfg["N_lo"]].get("failed"))
            j = int(np.argmin(L))
            if r["best_screen"] is None or L[j] < r["best_screen"]["J"]:
                r["best_screen"] = dict(J=L[j], gen=g, x=list(cs[j]["x"]), decoded=_decoded(cs[j]["x"], ev.space[p], ev.base[p]),
                                        eval=_slim(cs[j]["eval"][cfg["N_lo"]]))
            r["history"].append(dict(gen=g, J_min=min(L), J_med=float(np.median(L)), best=r["best_screen"]["J"],
                                     sigma=float(es[p].sigma), mean=list(map(float, es[p].mean)),
                                     L_shape_min=float(min(c["eval"][cfg["N_lo"]].get("L_shape", 9) for c in cs)),
                                     poured_mL_med=float(np.median([c["psum"]["poured_mL"] for c in cs]))))
            order = np.argsort(L)[:cfg["topk"]]
            top += [cs[o] for o in order]
            i += n
        last = g + 1 >= cfg["gens"]
        tc = 0.0
        if cfg["topk"] > 0 and ((g + 1) % max(cfg["confirm_every"], 1) == 0 or last):
            t1 = time.time()
            extra = []
            if last:                                   # the CMA mean of every pattern, confirmed too
                extra = ev.pitcher([(p, np.clip(es[p].mean, 0, 1)) for p in patterns])
                ev.solve(extra, cfg["N_lo"])
            ev.solve(top + extra, cfg["N_hi"])
            for c in top + extra:
                is_mean = any(c is e for e in extra)
                p = c["pattern"]
                r = rec[p]
                e = c["eval"][cfg["N_hi"]]
                r["confirms"].append(dict(gen=g, J_hi=e["J"], J_lo=c["eval"][cfg["N_lo"]]["J"], x=list(c["x"]),
                                          mean=is_mean))
                if r["best_confirm"] is None or e["J"] < r["best_confirm"]["J"]:
                    r["best_confirm"] = dict(J=e["J"], gen=g, x=list(c["x"]), decoded=_decoded(c["x"], ev.space[p], ev.base[p]),
                                             eval=_slim(e), eval_screen=_slim(c["eval"][cfg["N_lo"]]), cma_mean=is_mean)
            tc = time.time() - t1
        state["gen"] = g + 1
        state["wall"] += time.time() - t0
        state["np_rng"] = np.random.get_state()
        _save(out, state)
        done_now += 1
        for p in patterns:
            hst = rec[p]["history"][-1]
            bc = rec[p]["best_confirm"]
            log(f"gen {g} {p}: J min {hst['J_min']:.4f} med {hst['J_med']:.4f} best {hst['best']:.4f} sigma {hst['sigma']:.3f} "
                f"poured(med) {hst['poured_mL_med']:.1f} mL" + (f"; confirmed best {bc['J']:.4f} (gen {bc['gen']})" if bc else ""))
        log(f"gen {g}: {len(cands)} candidates, {nm} runs at N {cfg['N_lo']}: pitcher {tp:.1f} s, solver {ws:.1f} s, "
            f"confirm {tc:.1f} s")
    return state


def _save(out, state):
    tmp = os.path.join(out, "state.pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, os.path.join(out, "state.pkl"))
    summ = dict(hash=state["hash"], cfg=state["cfg"], gen=state["gen"], wall_s=state["wall"], backends=state["backends"],
                patterns=state["rec"])
    _atomic_json(summ, os.path.join(out, "design.json"))


def load_state(out):
    with open(os.path.join(out, "state.pkl"), "rb") as f:
        return pickle.load(f)


def best_of(r):
    return r["best_confirm"] or r["best_screen"]


def status(out):
    st = load_state(out)
    print(f"setup {st['hash']}: generation {st['gen']} of {st['cfg']['gens']}, wall {st['wall'] / 60:.1f} min, "
          f"backends {st['backends']}")
    for p, r in st["rec"].items():
        b = best_of(r)
        ref = r.get("reference_S1", {})
        print(f"  {p}: S1 poses J {ref.get('confirm', {}).get('J', float('nan')):.4f}; best "
              f"{'confirmed' if r['best_confirm'] else 'screened'} J {b['J'] if b else float('nan'):.4f}; evals {r['n_eval']} "
              f"(failed {r['n_failed']})")
        if b:
            print("     ", json.dumps(b["decoded"], default=lambda v: round(v, 2)))


# =========================================================================== outputs: design data, robustness, finals
def write_data(out, data_in=SF.DATA):
    """Best design of every pattern -> out/data/{poses_<p>.npz, program_<p>.json} in the s1_forward data format (the
    program's closure = the closure the design used), so that autodl_v05 / s1_forward run it unchanged (--data)."""
    st = load_state(out)
    cfg = st["cfg"]
    closure = load_closure(cfg["closure"])
    dd = os.path.join(out, "data")
    os.makedirs(dd, exist_ok=True)
    for p, r in st["rec"].items():
        b = best_of(r)
        if b is None:
            continue
        P = SF.load_poses(p, data_in)
        meta = SF.load_meta(p, data_in)
        sp = space_of(meta, tuple(v for v in cfg["vars"].split(",") if v))
        d = decode(b["x"], sp, P)
        Pd = design_poses(P, meta, d)
        np.savez_compressed(os.path.join(dd, f"poses_{p}.npz"), **{k: Pd[k] for k in ("t", "tip", "yaw", "tilt", "roll",
                                                                                      "fill_mL", "alpha_deg", "dt")})
        m = meta_with_closure(meta, closure)
        m = dict(m, fill_mL=float(d["fill"]), design=dict(source="gpu_design", setup=st["hash"], J=b["J"],
                                                          confirmed=bool(r["best_confirm"]), decoded=b["decoded"],
                                                          closure=cfg["closure"]))
        with open(os.path.join(dd, f"program_{p}.json"), "w") as f:
            json.dump(m, f, indent=1)
        print(f"{p}: design J {b['J']:.4f} -> {dd}", flush=True)
    return dd


def cmd_robust(a):
    from . import autodl_v05 as AV
    dd = write_data(a.out)
    pats = [p for p in a.patterns.split(",") if os.path.exists(os.path.join(dd, f"poses_{p}.npz"))]
    o = os.path.join(a.out, "robust_check")
    argv = ["robust", "--N", str(a.N.split(",")[0]), "--out", o, "--data", dd, "--patterns", ",".join(pats),
            "--pitcher-backend", a.pitcher_backend, "--solver-backend", a.solver_backend, "--chunk", str(a.chunk)]
    if a.T:
        argv += ["--T", str(a.T)]
    AV.main(argv)
    AV.main(["figures", "--out", o])


def cmd_final(a):
    from . import autodl_v05 as AV
    dd = write_data(a.out)
    pats = [p for p in a.patterns.split(",") if os.path.exists(os.path.join(dd, f"poses_{p}.npz"))]
    common = ["--N", a.N, "--patterns", ",".join(pats), "--pitcher-backend", a.pitcher_backend, "--solver-backend",
              a.solver_backend] + (["--T", str(a.T)] if a.T else []) + (["--video"] if a.video else [])
    o = os.path.join(a.out, "final")
    AV.main(["aligned", "--out", o, "--data", dd] + common)
    AV.main(["figures", "--out", o])
    o0 = os.path.join(a.out, "final_s1")                # the S1 poses through the same path, for the comparison
    if not a.no_s1:
        st = load_state(a.out)
        cl = st["cfg"]["closure"]
        d0 = SF.DATA
        if cl not in (None, "", "s1"):                  # S1 poses with the design's closure
            d0 = os.path.join(a.out, "data_s1")
            os.makedirs(d0, exist_ok=True)
            closure = load_closure(cl)
            for p in pats:
                import shutil
                shutil.copy(os.path.join(SF.DATA, f"poses_{p}.npz"), d0)
                with open(os.path.join(d0, f"program_{p}.json"), "w") as f:
                    json.dump(meta_with_closure(SF.load_meta(p), closure), f, indent=1)
        AV.main(["aligned", "--out", o0, "--data", d0] + [x for x in common if x != "--video"])
    compare_figure(a.out, pats)


def compare_figure(out, pats):
    """V0.5 reference | S1 poses | redesign, per pattern, at the largest common N; labels: IoU(c>0.4) with the reference,
    white area, poured volume."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .autodl_v05 import _cmap, _show, _load_c
    cm = _cmap()
    roots = dict(S1=os.path.join(out, "final_s1", "aligned"), design=os.path.join(out, "final", "aligned"))
    Ns = sorted(int(d[1:]) for d in os.listdir(roots["design"])) if os.path.isdir(roots["design"]) else []
    if not Ns:
        return None
    N = Ns[0] if 256 not in Ns else 256
    fig, ax = plt.subplots(3, len(pats), figsize=(max(2.3 * len(pats), 5.0), 7.6), dpi=130, squeeze=False)
    res = {}
    for j, p in enumerate(pats):
        cr, mr = DM.load_ref(p, N)
        _show(ax[0, j], cr, p, cm)
        ax[0, j].set_title(p, fontsize=8)
        res[p] = {}
        for i, (lab, root) in enumerate((("S1", roots["S1"]), ("design", roots["design"])), start=1):
            d = os.path.join(root, f"N{N}", p)
            x = ax[i, j]
            x.set_xticks([]); x.set_yticks([])
            if not os.path.exists(os.path.join(d, "c.npz")):
                x.text(0.5, 0.5, "missing", ha="center", va="center", transform=x.transAxes, fontsize=7)
                continue
            c, mask = _load_c(d)
            _show(x, c, p, cm)
            sh = DM.shape_vs_ref(c, p, N)
            m = json.load(open(os.path.join(d, "metrics.json")))
            poured = m.get("extra", {}).get("poured_mL", float("nan"))
            res[p][lab] = dict(iou04=sh["iou04"], r_hp=sh["r_hp"], white=sh["white"], white_ref=sh["white_ref"],
                               poured_mL=poured)
            x.text(0.03, 0.97, f"IoU {sh['iou04']:.2f}  W {sh['white']:.3f}/{sh['white_ref']:.3f}\n{poured:.1f} mL",
                   color="white", fontsize=5.5, ha="left", va="top", transform=x.transAxes,
                   bbox=dict(boxstyle="round,pad=0.15", fc=(0.2, 0.2, 0.2), ec="none"))
    for i, lab in enumerate(("V0.5 reference", "S1 poses (forward pitcher)", "re-designed")):
        ax[i, 0].set_ylabel(lab, fontsize=8)
    fig.suptitle(f"Re-designed actions on the forward pitcher, aligned model, {N}$^2$\n(c stretched 0.35..1; IoU of c > 0.4 "
                 f"with the V0.5 reference,\nwhite area model/reference, poured volume)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fp = os.path.join(out, "fig_design_compare.png")
    fig.savefig(fp)
    plt.close(fig)
    _atomic_json(res, os.path.join(out, "design_compare.json"))
    print(fp, flush=True)
    return fp


# =========================================================================== export (dev machine, once)
def export(data=DM.DATA):
    """Reference fields and the S1 refs from the review scratch / local runs into latte_imex/data/gpu_design (small)."""
    from . import v05_pitcher_check as PC
    import shutil
    os.makedirs(data, exist_ok=True)
    rev = PC.REVIEW
    shutil.copy(os.path.join(rev, "v05pour", "design", "refs.json"), os.path.join(data, "s1_refs.json"))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in SF.PATTERNS:
        z = np.load(os.path.join(rev, "v05audit", "cf", "baseline", f"{p}_N128.npz"))
        np.savez_compressed(os.path.join(data, f"v05_ref_{p}_N128.npz"), c=z["c"].astype(np.float32), mask=z["mask"])
        z = np.load(os.path.join(repo, "runs", "v05_256", p, "final_state.npz"))
        mask = z["mask"]
        c = np.where(mask, z["m"] / np.where(mask, z["l"], 1.0), np.nan)
        np.savez_compressed(os.path.join(data, f"v05_ref_{p}_N256.npz"), c=c.astype(np.float32), mask=mask)
    with open(os.path.join(data, "SOURCES.json"), "w") as f:
        json.dump(dict(s1_refs="review v05pour/design/refs.json (v05_pourable._refs)",
                       v05_ref_N128="review v05audit/cf/baseline/<p>_N128.npz (allphys.load_ref, N = 128): c, mask",
                       v05_ref_N256="runs/v05_256/<p>/final_state.npz (allphys.load_ref, N = 256): c = m / l, mask",
                       written=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=1)
    print(f"exported to {data}", flush=True)


# =========================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["export", "run", "status", "write", "robust", "final"])
    ap.add_argument("--out", default="results/gpu_design")
    ap.add_argument("--patterns", default=",".join(SF.PATTERNS))
    ap.add_argument("--gens", type=int, default=DEFAULTS["gens"])
    ap.add_argument("--popsize", type=int, default=DEFAULTS["popsize"])
    ap.add_argument("--sigma0", type=float, default=DEFAULTS["sigma0"])
    ap.add_argument("--N-lo", type=int, default=DEFAULTS["N_lo"])
    ap.add_argument("--N-hi", type=int, default=DEFAULTS["N_hi"])
    ap.add_argument("--topk", type=int, default=DEFAULTS["topk"])
    ap.add_argument("--confirm-every", type=int, default=DEFAULTS["confirm_every"])
    ap.add_argument("--rob", default=DEFAULTS["rob"], help="hand-error draws of the robustness term ('' = none)")
    ap.add_argument("--vars", default=DEFAULTS["vars"])
    ap.add_argument("--w-s1", type=float, default=DEFAULTS["w_s1"])
    ap.add_argument("--w-shape", type=float, default=DEFAULTS["w_shape"])
    ap.add_argument("--w-rob", type=float, default=DEFAULTS["w_rob"])
    ap.add_argument("--w-pen", type=float, default=DEFAULTS["w_pen"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--closure", default=DEFAULTS["closure"])
    ap.add_argument("--pitcher-law", default="", help="JSON of ForwardLaw constants (gpu_calib fit_pitcher output)")
    ap.add_argument("--T", type=float, default=None, help="truncate every run (and the poses) at T [s] (smoke tests)")
    ap.add_argument("--chunk", type=int, default=0, help="members per GPU batch (0 = a whole generation)")
    ap.add_argument("--max-gens", type=int, default=None, help="stop this session after that many generations")
    ap.add_argument("--N", default="256,512", help="final: resolutions; robust: the first one")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--no-s1", action="store_true", help="final: skip the S1-poses comparison runs")
    ap.add_argument("--pitcher-backend", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--solver-backend", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args(argv)
    if a.cmd == "export":
        return export()
    if a.cmd == "status":
        return status(a.out)
    if a.cmd == "write":
        return write_data(a.out)
    if a.cmd == "robust":
        return cmd_robust(a)
    if a.cmd == "final":
        return cmd_final(a)
    law = None
    if a.pitcher_law:
        with open(a.pitcher_law) as f:
            j = json.load(f)
        law = {k: float(v) for k, v in j.get("law", j).items() if k in ("C_d", "C_v", "h_k")}
    cfg = dict(DEFAULTS, gens=a.gens, popsize=a.popsize, sigma0=a.sigma0, N_lo=a.N_lo, N_hi=a.N_hi, topk=a.topk,
               confirm_every=a.confirm_every, w_s1=a.w_s1, w_shape=a.w_shape, w_rob=a.w_rob, w_pen=a.w_pen, seed=a.seed,
               rob=a.rob, vars=a.vars, T=a.T, closure=a.closure, chunk=a.chunk, pitcher_law=law)
    if cfg["closure"] not in ("", "s1"):
        cfg["closure"] = os.path.abspath(cfg["closure"])
    cfg["closure_values"] = load_closure(cfg["closure"])          # part of the setup hash (resume check)
    run(a.out, a.patterns.split(","), cfg, a.solver_backend, a.pitcher_backend, max_gens=a.max_gens)
    status(a.out)


if __name__ == "__main__":
    main()
