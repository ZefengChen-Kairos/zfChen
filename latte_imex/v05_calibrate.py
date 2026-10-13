"""Calibration driver for the V0.5 model with the barista's real actions (senior V1 5.1-5.4): CMA-ES on the shared
closure + inlet-nuisance parameters on tulip + leaf (AB) and on tulip only (A, baseline), same space / loss / budget,
through `calib.calibrate`; parameters are then frozen for the test predictions (`evaluate`, a later step).

    python -m latte_imex.v05_calibrate run      --out DIR [--tags AB,A]       # pre-registered run (resumable)
    python -m latte_imex.v05_calibrate smoke    --out DIR                     # 1 generation, popsize 4, same driver
    python -m latte_imex.v05_calibrate status   --out DIR                     # progress (calib.json, cache, logs)
    python -m latte_imex.v05_calibrate report   --out DIR                     # history.png + report.json + tables
    python -m latte_imex.v05_calibrate evaluate --out DIR [--N 128 --N_fig 256]  # frozen parameters, six patterns

PRE-REGISTERED SETUP (`PREREG`, fixed before any calibration run, written to DIR/prereg.json and into every
calib.json under 'driver'):
  spec      dict(kind='v05', base='FSp2', loss_delay=False, space_extra=dict(u_exit=(0.10, 0.27, 'lin'),
            Q_scale=(0.5, 2.0, 'log'))): 10 parameters = the 8 closure parameters of v05_adapter.SPACE (B_dep, p_dep,
            kappa_r, kappa_Q, cp, beta, kappa_t, kappa_c, physical bounds) + 2 input nuisances (exit horizontal speed
            u_exit [m/s], global flow scale Q_scale).  loss_delay=False: V0.5 has no first-white delay mechanism, so
            the delay term is left out of the loss (it is still computed and reported).
  loss      calib.loss_of: <1 - Dice>_t + <d_bnd>_t / 0.05 D + stage-end fine-line count error, mean over the
            calibration patterns (equal weights).
  CMA-ES    calib.calibrate, N = 96, popsize 12, sigma0 0.25 (unit box), 20 generations per initial point, 3 initial
            points: (1) FSp2 closure + u_exit 0.15 + Q_scale 1.0, (2) FS closure + u_exit 0.15 + Q_scale 1.0, (3) the
            centre of the unit box; procs 4.  seed = 1 (inits use seeds 1, 2, 3): calib.calibrate passes seed + ii to
            cma, and cma treats seed 0 as "seed from the clock", which would make init 0 irreproducible.
            Same for AB (tulip + leaf) and A (tulip).  Best = lowest mean loss over all CMA samples of the three
            inits (calib.calibrate rec['best']); the initial points themselves are evaluated once as REFERENCES
            (rec['driver']['x0_reference']), never as candidates and not counted in the CMA budget.
  failures  a forward call that raises, returns a ledger with ok = False, or exceeds TIMEOUT_S wall seconds is scored
            +inf by calib (1e6 for CMA) and counted (rec['n_failed']).
  test      heart, push_heart, layered_heart, swan are never loaded by `run` / `smoke`.
  evaluate  frozen best parameters of AB and A at N = 128 on all six patterns; final-frame figures at N = 256.

RESUME.  Every forward call is cached in DIR/cache/<sha1>.json (key: spec, parameter values (repr), pattern, N).  The
driver wraps calib's worker (`calib._weval`, `calib._winit`) with the cache and the timeout while calib.calibrate runs
(restored afterwards; no file of calib is changed); the wrapper calls the original `calib._weval` and returns the
JSON round-trip of its result, so a fresh call and a cached call return identical values.  cma with a fixed seed asks
the same points when it is told the same losses, so a restarted `run` replays the finished generations from the cache in
seconds and continues where it stopped.  A tag whose calib.json is complete with the same setup is skipped.
Bookkeeping: DIR/run_meta.json (sessions, per-tag segments with wall times and cache counts), DIR/calib_<tag>.log.
`patched(..., inner='module:function')` puts another worker with calib._weval's signature inside the same cache /
timeout wrapper (used by v05_actfit for its loss); without it the wrapper calls calib._weval as before.
The cache is shared by AB and A: A's first generation asks the same tulip points as AB's (same seeds, same x0), so
those tulip evaluations are cache hits (deterministic solver, NUMBA_NUM_THREADS = 1); calib's n_forward counts them
(the budget), cache records count what was computed and by which tag.
"""
import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import signal
import socket
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

from . import calib as CB
from .v05_adapter import FS, FSP2

CLOSURE8 = ("B_dep", "p_dep", "kappa_r", "kappa_Q", "cp", "beta", "kappa_t", "kappa_c")
SPEC = dict(kind="v05", base="FSp2", loss_delay=False,
            space_extra=dict(u_exit=(0.10, 0.27, "lin"), Q_scale=(0.5, 2.0, "log")))
TAGS = dict(AB=CB.CAL_AB, A=CB.CAL_A)
TIMEOUT_S = 1200
AT_BOUND = 0.02            # a parameter "sits at a bound" if its unit-box coordinate is <= 0.02 or >= 0.98
PREREG = dict(
    date="2026-10-13", spec=SPEC, tags={k: list(v) for k, v in TAGS.items()}, order=["AB", "A"],
    test=list(CB.TEST),
    loss="calib.loss_of(ev, delay=False): <1-Dice>_t + <d_bnd>_t/BND_SCALE + stage-end fine-line count error; "
         "mean over the calibration patterns (equal weights)",
    cma=dict(N=96, popsize=12, sigma0=0.25, gens=20, procs=4, seed=1,
             inits=["FSp2 closure + u_exit 0.15 + Q_scale 1.0", "FS closure + u_exit 0.15 + Q_scale 1.0",
                    "centre of the unit box"]),
    selection="best = lowest mean calibration loss over all CMA samples of the three inits (calib.calibrate); "
              "initial points evaluated as references only",
    failures=f"exception / ledger ok=False / wall > {TIMEOUT_S} s -> +inf (1e6 for CMA), counted",
    at_bound=f"unit-box coordinate <= {AT_BOUND} or >= {1 - AT_BOUND}",
    evaluate=dict(N=128, patterns="all six", N_fig=256, figure="final frame"),
    never="test patterns are not used to choose initialisation, parameters, thresholds or time alignment",
)
SMOKE = dict(gens=1, popsize=4)

_ORIG_WEVAL = CB._weval
_ORIG_WINIT = CB._winit
ENV_CACHE, ENV_TIMEOUT, ENV_TAG, ENV_INNER = "V05CAL_CACHE", "V05CAL_TIMEOUT", "V05CAL_TAG", "V05CAL_INNER"


# =========================================================================== cached / time-limited worker
def cache_key(params, name, N, spec):
    s = json.dumps(dict(spec=spec, params={k: repr(float(v)) for k, v in sorted(params.items())}, name=name,
                        N=int(N)), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()


_WS = {}


def _winit(spec):
    _ORIG_WINIT(spec)
    _WS["spec"] = spec


def _inner():
    """The worker the cache wraps: calib._weval (default), or the 'module:function' named in $V05CAL_INNER (same
    signature and return value; v05_actfit uses it for its own loss).  Read from the environment, so pool workers see
    it under fork and spawn alike."""
    s = os.environ.get(ENV_INNER)
    if not s:
        return _ORIG_WEVAL
    import importlib
    mod, fn = s.split(":")
    return getattr(importlib.import_module(mod), fn)


def _alarm(signum, frame):
    raise TimeoutError(f"forward call exceeded {os.environ.get(ENV_TIMEOUT, TIMEOUT_S)} s")


def _weval(args):
    """calib._weval with a disk cache and a wall-time limit (results JSON round-tripped: fresh == cached)."""
    params, name, N = args
    cdir = os.environ[ENV_CACHE]
    key = cache_key(params, name, N, _WS["spec"])
    f = os.path.join(cdir, key + ".json")
    if os.path.exists(f):
        try:
            return name, json.load(open(f))["eval"]
        except Exception:
            pass
    t0 = time.time()
    lim = int(os.environ.get(ENV_TIMEOUT, TIMEOUT_S))
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(lim)
    try:
        n, ev = _inner()(args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    ev = json.loads(json.dumps(ev, default=float))
    rec = dict(key=key, name=name, N=int(N), params=params, tag=os.environ.get(ENV_TAG, ""), pid=os.getpid(),
               t_start=t0, t_end=time.time(), wall_s=time.time() - t0, eval=ev)
    tmp = f"{f}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, default=float)
    os.replace(tmp, f)
    return name, ev


@contextlib.contextmanager
def patched(cache_dir, tag, inner=None):
    """calib's pool worker / initializer -> the cached, time-limited ones while the block runs.  inner = optional
    'module:function' replacing calib._weval inside the wrapper (default: calib._weval)."""
    os.makedirs(cache_dir, exist_ok=True)
    env_old = {k: os.environ.get(k) for k in (ENV_CACHE, ENV_TIMEOUT, ENV_TAG, ENV_INNER)}
    os.environ[ENV_CACHE] = cache_dir
    os.environ[ENV_TIMEOUT] = str(TIMEOUT_S)
    os.environ[ENV_TAG] = tag
    if inner:
        os.environ[ENV_INNER] = inner
    else:
        os.environ.pop(ENV_INNER, None)
    CB._weval, CB._winit = _weval, _winit
    try:
        yield
    finally:
        CB._weval, CB._winit = _ORIG_WEVAL, _ORIG_WINIT
        for k, v in env_old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def n_cache(cache_dir):
    return len(glob.glob(os.path.join(cache_dir, "*.json")))


# =========================================================================== setup
def space_of(spec=SPEC):
    return CB.make_model(spec).space


def initial_points(space):
    x1 = {k: FSP2[k] for k in CLOSURE8}
    x1.update(u_exit=0.15, Q_scale=1.0)
    x2 = {k: FS[k] for k in CLOSURE8}
    x2.update(u_exit=0.15, Q_scale=1.0)
    x3 = CB.decode([0.5] * len(space), space)
    for x in (x1, x2, x3):
        assert set(x) == set(space), (set(x) ^ set(space))
        for k, (lo, hi, _) in space.items():
            assert lo - 1e-12 <= x[k] <= hi + 1e-12, (k, x[k], lo, hi)
    return [x1, x2, x3]


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def code_provenance():
    """git HEAD of the repo (working tree may hold this new file uncommitted) + sha1 of the files the run uses."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = None
    files = ("v05_calibrate.py", "calib.py", "v05_adapter.py", "solver.py", "kernels.py", "v3min_adapter.py",
             "video_extract.py")
    sha = {f: hashlib.sha1(open(os.path.join(here, f), "rb").read()).hexdigest() for f in files
           if os.path.exists(os.path.join(here, f))}
    import cma
    return dict(git_head=head, sha1=sha, cma=cma.__version__, numpy=np.__version__)


def _jload(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


def setup_matches(rec, names, x0s, cfg):
    if rec is None:
        return False
    same = (rec.get("spec") == json.loads(json.dumps(SPEC)) and rec.get("names") == list(names)
            and rec.get("N") == cfg["N"] and rec.get("gens") == cfg["gens"] and rec.get("popsize") == cfg["popsize"]
            and abs(rec.get("sigma0", -1) - cfg["sigma0"]) < 1e-12 and len(rec.get("inits", [])) >= 1)
    if not same:
        return False
    for r, x in zip(rec["inits"], x0s):
        if any(abs(r["x0"][k] - x[k]) > 1e-12 * max(1.0, abs(x[k])) for k in x):
            return False
    drv = rec.get("driver") or {}
    return drv.get("seed", cfg["seed"]) == cfg["seed"]


# =========================================================================== run
class Meta:
    """DIR/run_meta.json: sessions and per-tag segments (start, last update, cache counts)."""

    def __init__(self, out, argv):
        self.path = os.path.join(out, "run_meta.json")
        self.d = _jload(self.path, None) or dict(sessions=[], tags={})
        self.sid = len(self.d["sessions"])
        self.d["sessions"].append(dict(id=self.sid, start=time.time(), start_str=_now(), argv=argv,
                                       host=socket.gethostname(), pid=os.getpid(), cpus=os.cpu_count(),
                                       python=sys.version.split()[0]))
        self.save()

    def save(self):
        _jdump(self.d, self.path)

    def segment(self, tag, n0):
        seg = dict(session=self.sid, start=time.time(), start_str=_now(), end=time.time(), end_str=_now(),
                   cache_start=n0, cache_end=n0, log_lines=0, completed=False)
        self.d["tags"].setdefault(tag, dict(segments=[]))["segments"].append(seg)
        self.save()
        return seg

    def touch(self, seg, ncache, completed=False):
        seg.update(end=time.time(), end_str=_now(), cache_end=ncache, completed=bool(seg.get("completed") or completed))
        seg["log_lines"] += 0 if completed else 1
        self.save()


def run(out, tags=("AB", "A"), gens=20, popsize=12, N=96, procs=4, sigma0=0.25, seed=1, label="run", force=False):
    from multiprocessing import Pool
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "cache")
    cfg = dict(gens=gens, popsize=popsize, N=N, procs=procs, sigma0=sigma0, seed=seed, label=label)
    space = space_of()
    x0s = initial_points(space)
    pre = dict(PREREG, run_config=cfg, space={k: list(v) for k, v in space.items()}, x0s=x0s, timeout_s=TIMEOUT_S,
               numba_threads=os.environ.get("NUMBA_NUM_THREADS"), code=code_provenance())
    pp = os.path.join(out, "prereg.json")
    old = _jload(pp)
    if old is not None and old.get("run_config") != json.loads(json.dumps(cfg)) and not force:
        raise SystemExit(f"{pp} holds a different run_config ({old.get('run_config')}); use another --out or --force")
    _jdump(pre, pp)
    meta = Meta(out, sys.argv)
    for tag in tags:
        names = TAGS[tag]
        d = os.path.join(out, f"calib_{tag}")
        cj = os.path.join(d, "calib.json")
        logp = os.path.join(out, f"calib_{tag}.log")
        rec_old = _jload(cj)
        if rec_old is not None and "best" in rec_old and setup_matches(rec_old, names, x0s, cfg):
            print(f"[{_now()}] {tag}: complete with the same setup, skipped", flush=True)
            continue
        if rec_old is not None and not setup_matches(rec_old, names, x0s, cfg) and not force:
            raise SystemExit(f"{cj} was written with a different setup; use another --out or --force")
        n0 = n_cache(cache)
        seg = meta.segment(tag, n0)
        t0 = time.time()
        state = dict(last=n0, tlast=t0)

        def log(line, tag=tag, seg=seg):
            nc = n_cache(cache)
            new = nc - state["last"]
            dt = time.time() - state["tlast"]
            state.update(last=nc, tlast=time.time())
            s = (f"[{_now()} +{time.time() - t0:7.0f}s] {line} | computed {new} in {dt:.0f}s"
                 f"{' (replayed from cache)' if new == 0 else ''}, cache {nc}")
            print(s, flush=True)
            with open(logp, "a") as fh:
                fh.write(s + "\n")
            meta.touch(seg, nc)

        log(f"{tag} start: session {meta.sid}, names {names}, N {N}, gens {gens}, popsize {popsize}, sigma0 {sigma0},"
            f" seed {seed}, procs {procs}")
        with patched(cache, tag):
            # reference: the initial points themselves (not candidates, not in n_forward)
            with Pool(procs, initializer=_winit, initargs=(SPEC,)) as pool:
                outs = pool.map(_weval, [(x, n, N) for x in x0s for n in names], chunksize=1)
            refs = []
            for i, x in enumerate(x0s):
                evs = dict(outs[i * len(names):(i + 1) * len(names)])
                refs.append(dict(x0=x, loss=float(np.mean([evs[n]["loss"] for n in names])),
                                 per_pattern={n: dict(loss=evs[n]["loss"], failed=bool(evs[n].get("failed")),
                                                      dice=(evs[n].get("dice") or {}).get("value"),
                                                      boundary_D=(evs[n].get("boundary_D") or {}).get("value"))
                                              for n in names}))
                log(f"{tag} x0 {i} reference loss {refs[-1]['loss']:.4f} "
                    + " ".join(f"{n} {evs[n]['loss']:.4f}" for n in names))
            rec = CB.calibrate(SPEC, names, d, x0s, gens=gens, popsize=popsize, sigma0=sigma0, N=N, procs=procs,
                               seed=seed, log=log)
        nc = n_cache(cache)
        meta.touch(seg, nc, completed=True)
        segs = meta.d["tags"][tag]["segments"]
        comp = [_jload(f) for f in glob.glob(os.path.join(cache, "*.json"))]
        comp = [c for c in comp if c and c.get("tag") == tag]
        rec["driver"] = dict(
            prereg=pre, seed=seed, label=label, x0_reference=refs,
            wall_s=float(sum(s["end"] - s["start"] for s in segs)),
            segments=segs, computed_by_tag=len(comp), computed_failed=sum(bool(c["eval"].get("failed")) for c in comp),
            compute_s_by_tag=float(sum(c["wall_s"] for c in comp)),
            note="n_forward = forward calls requested by CMA-ES (the budget, cache hits included); computed_by_tag = "
                 "calls computed during this tag (cache misses); x0 references are extra and not in n_forward")
        _jdump(rec, cj)
        log(f"{tag} done: best {rec['best']['loss']:.4f}, spread {[round(v, 4) for v in rec['best']['spread']]}, "
            f"n_forward {rec['n_forward']}, n_failed {rec['n_failed']}")
    meta.d["sessions"][meta.sid].update(end=time.time(), end_str=_now(), finished=True)
    meta.save()


# =========================================================================== status
def status(out):
    out = os.path.abspath(out)
    cache = os.path.join(out, "cache")
    print(f"cache entries: {n_cache(cache)}")
    for tag in TAGS:
        rec = _jload(os.path.join(out, f"calib_{tag}", "calib.json"))
        if rec is None:
            print(f"{tag}: no calib.json")
            continue
        done = "best" in rec
        print(f"{tag}: {'COMPLETE' if done else 'running'}  n_forward {rec['n_forward']}  n_failed {rec['n_failed']}")
        for i, r in enumerate(rec["inits"]):
            h = r["history"]
            print(f"   init {i}: gens {len(h)}/{rec['gens']}  best {r['best']['loss']:.4f} (gen {r['best']['gen']})"
                  f"  last min {h[-1]['loss_min']:.4f} med {h[-1]['loss_med']:.4f} sigma {h[-1]['sigma']:.3f}")
    meta = _jload(os.path.join(out, "run_meta.json"))
    if meta:
        for tag, t in meta["tags"].items():
            w = sum(s["end"] - s["start"] for s in t["segments"])
            print(f"{tag}: wall {w / 3600:.2f} h over {len(t['segments'])} segment(s)")


# =========================================================================== report
def unit_of(v, lo, hi, sc):
    return CB.to_unit(v, lo, hi, sc)


def param_table(params, space):
    rows = []
    for k, (lo, hi, sc) in space.items():
        u = unit_of(params[k], lo, hi, sc)
        at = "lower" if u <= AT_BOUND else ("upper" if u >= 1 - AT_BOUND else "")
        rows.append(dict(name=k, value=float(params[k]), lo=lo, hi=hi, scale=sc, unit=float(u), at_bound=at))
    return rows


def report(out):
    """history.png, report.json (tables for calibration.md) from calib_AB / calib_A / run_meta / cache."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.abspath(out)
    cache = os.path.join(out, "cache")
    space = space_of()
    meta = _jload(os.path.join(out, "run_meta.json")) or {}
    cached = [c for c in (_jload(f) for f in glob.glob(os.path.join(cache, "*.json"))) if c]
    rep = dict(space={k: list(v) for k, v in space.items()}, tags={})
    for tag, names in TAGS.items():
        rec = _jload(os.path.join(out, f"calib_{tag}", "calib.json"))
        if rec is None:
            continue
        segs = (meta.get("tags", {}).get(tag) or {}).get("segments", [])
        mine = [c for c in cached if c.get("tag") == tag]
        r = dict(complete="best" in rec, n_forward=rec["n_forward"], n_failed=rec["n_failed"],
                 n_candidates=sum(len(i["history"]) for i in rec["inits"]) * rec["popsize"],
                 wall_s=float(sum(s["end"] - s["start"] for s in segs)), segments=len(segs),
                 computed=len(mine), computed_failed=sum(bool(c["eval"].get("failed")) for c in mine),
                 compute_s=float(sum(c["wall_s"] for c in mine)),
                 forward_wall_s=dict(median=float(np.median([c["wall_s"] for c in mine])) if mine else None,
                                     p95=float(np.percentile([c["wall_s"] for c in mine], 95)) if mine else None,
                                     max=float(max(c["wall_s"] for c in mine)) if mine else None),
                 failures=[dict(name=c["name"], params=c["params"], error=c["eval"].get("error"),
                                ledger_error=(c["eval"].get("ledger") or {}).get("error"))
                           for c in mine if c["eval"].get("failed")],
                 x0_reference=(rec.get("driver") or {}).get("x0_reference"), inits=[])
        for i, ini in enumerate(rec["inits"]):
            b = ini["best"]
            r["inits"].append(dict(init=i, x0=ini["x0"], best_loss=b["loss"], best_gen=b["gen"],
                                   gens=len(ini["history"]), params=param_table(b["params"], space),
                                   per_pattern={n: dict(loss=e.get("loss"), dice=(e.get("dice") or {}).get("value"),
                                                        iou=(e.get("iou") or {}).get("value"),
                                                        boundary_D=(e.get("boundary_D") or {}).get("value"),
                                                        first_white_s=(e.get("first_white") or {}).get("value"),
                                                        profiles=[(p["at"], p["n_white_video"], p["n_white_model"])
                                                                  for p in e.get("profiles", [])],
                                                        final_iou=(e.get("final") or {}).get("iou"),
                                                        ledger_ok=(e.get("ledger") or {}).get("ok"))
                                                for n, e in b["evals"].items()},
                                   history=[(h["gen"], h["loss_min"], h["loss_med"], h["sigma"]) for h in ini["history"]],
                                   mean_final=ini["history"][-1]["mean"] if ini["history"] else None))
        if r["inits"]:
            ib = int(np.argmin([i["best_loss"] for i in r["inits"]]))
            r["best_init"] = ib
            r["best_loss"] = r["inits"][ib]["best_loss"]
            r["best_params"] = r["inits"][ib]["params"]
            r["spread"] = dict(loss=[i["best_loss"] for i in r["inits"]],
                               unit_range={p["name"]: float(np.ptp([i["params"][j]["unit"] for i in r["inits"]]))
                                           for j, p in enumerate(r["inits"][0]["params"])})
        rep["tags"][tag] = r
    _jdump(rep, os.path.join(out, "report.json"))

    # ---- history figure: per tag, per init: best-so-far (solid), generation minimum (thin), median (dashed)
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    labels = ["init 1: FSp2", "init 2: FS", "init 3: box centre"]
    tags_present = [t for t in ("AB", "A") if t in rep["tags"]]
    if tags_present:
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator
        fig, axes = plt.subplots(1, len(tags_present), figsize=(6.2 * len(tags_present), 4.5), dpi=130,
                                 squeeze=False)
        for ax, tag in zip(axes[0], tags_present):
            r = rep["tags"][tag]
            for i, ini in enumerate(r["inits"]):
                h = np.array(ini["history"], float)
                if not len(h):
                    continue
                g = h[:, 0] + 1
                ax.plot(g, np.minimum.accumulate(h[:, 1]), color=cols[i], lw=2.0, zorder=3)
                ax.plot(g, h[:, 1], color=cols[i], lw=0.9, alpha=0.6, marker="o", ms=2.5, zorder=2)
                ax.plot(g, h[:, 2], color=cols[i], lw=0.9, ls="--", alpha=0.6, zorder=1)
                ref = (r.get("x0_reference") or [None] * 3)[i]
                if ref is not None and np.isfinite(ref["loss"]):
                    ax.plot([0.25 * (i - 1)], [ref["loss"]], marker="D", ms=6, color=cols[i], mec="white", mew=1.0,
                            zorder=4)
            ax.set_xlabel("generation (0 = initial point, reference evaluation)")
            ax.set_ylabel("loss L (mean over " + " + ".join(TAGS[tag]) + ")")
            ax.set_title(f"{tag}: calibrate on {' + '.join(TAGS[tag])}  (N = 96, popsize 12)", fontsize=10)
            ax.grid(True, color="#e4e3df", lw=0.6)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            lo = min(min(np.array(i["history"])[:, 1]) for i in r["inits"] if i["history"])
            hi = np.percentile([x for i in r["inits"] for x in np.array(i["history"])[:, 2]], 90)
            refs = [x["loss"] for x in (r.get("x0_reference") or []) if np.isfinite(x["loss"])]
            hi = max([hi] + refs)
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))
        hs = [Line2D([], [], color=c, lw=2.0, label=lab) for c, lab in zip(cols, labels)]
        hs += [Line2D([], [], color="#52514e", lw=2.0, label="best so far"),
               Line2D([], [], color="#52514e", lw=0.9, marker="o", ms=2.5, label="generation minimum"),
               Line2D([], [], color="#52514e", lw=0.9, ls="--", label="generation median (clipped at the top)"),
               Line2D([], [], color="#52514e", lw=0, marker="D", ms=6, label="initial point itself")]
        fig.legend(handles=hs, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(os.path.join(out, "history.png"))
        plt.close(fig)
    return rep


def tables_md(rep):
    """Markdown tables (numbers only) for calibration.md."""
    L = []
    def f1(v):
        return "-" if v is None else f"{v:.1f}"
    for tag, r in rep["tags"].items():
        fw = r["forward_wall_s"]
        L.append(f"### {tag}\n")
        L.append(f"- complete {r['complete']}, n_forward {r['n_forward']}, candidates {r['n_candidates']}, "
                 f"failed {r['n_failed']}, computed {r['computed']} (failed {r['computed_failed']}), "
                 f"wall {r['wall_s'] / 3600:.2f} h in {r['segments']} segment(s), compute {r['compute_s'] / 3600:.2f} "
                 f"CPU-h, forward wall median {f1(fw['median'])} s p95 {f1(fw['p95'])} s max {f1(fw['max'])} s\n")
        L.append("| init | x0 loss | best loss | gen of best | " + " | ".join(p["name"] for p in r["inits"][0]["params"]) + " |")
        L.append("|" + "---|" * (4 + len(r["inits"][0]["params"])))
        for i, ini in enumerate(r["inits"]):
            ref = (r.get("x0_reference") or [None] * 3)[i]
            L.append(f"| {i + 1} | {ref['loss']:.4f} | {ini['best_loss']:.4f} | {ini['best_gen']} | "
                     + " | ".join(f"{p['value']:.4g} ({p['unit']:.2f}){'*' if p['at_bound'] else ''}"
                                  for p in ini["params"]) + " |")
        L.append("")
    return "\n".join(L)


# =========================================================================== evaluate (frozen parameters)
_EV = {}


def _ev_init():
    from .v05_adapter import V05Model  # noqa: F401
    _EV["model"] = CB.make_model(SPEC)


def _ev_full(args):
    tag, params, name, N = args
    clip = CB.load_clip(name)
    t0 = time.time()
    try:
        wT, led = CB.simulate_clip(_EV["model"], params, clip, N)
        ev = CB.evaluate(clip, wT, led, delay_in_loss=False)
        out = CB.compact(ev)
        out["failed"] = not (led or {}).get("ok", True)
    except Exception as e:
        out = dict(name=name, loss=float("inf"), failed=True, error=f"{type(e).__name__}: {e}")
    out["wall_s"] = time.time() - t0
    return tag, name, json.loads(json.dumps(out, default=float))


def _ev_final(args):
    tag, params, name, N = args
    clip = CB.load_clip(name)
    t = clip["res"]["t"]
    r = _EV["model"].run(clip["script"], params, N, t[-1:], keep_log=False)
    wT = CB.observe_model(r["c"], clip["M"])
    return tag, name, wT[0], r["ledger"].get("ok", False)


def evaluate(out, N=128, N_fig=256, procs=4, tags=("AB", "A")):
    """Frozen best parameters of each tag on all six patterns at N (process metrics, calib.evaluate with
    delay_in_loss=False, delay still reported), final-frame figure at N_fig.  Calibration fit and test prediction
    are reported separately."""
    from multiprocessing import Pool
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.abspath(out)
    names = list(CB.CAL_AB) + list(CB.TEST)
    best = {t: _jload(os.path.join(out, f"calib_{t}", "calib.json"))["best"]["params"] for t in tags}
    with Pool(procs, initializer=_ev_init) as pool:
        res = pool.map(_ev_full, [(t, best[t], n, N) for t in tags for n in names], chunksize=1)
    evs = {t: {} for t in tags}
    for t, n, e in res:
        evs[t][n] = e
    rep = {}
    for t in tags:
        _jdump(evs[t], os.path.join(out, f"eval_{t}_N{N}.json"))
        cal = list(TAGS[t])
        held = [n for n in CB.CAL_AB if n not in cal]

        def summ(ns):
            rows = {n: dict(loss=evs[t][n]["loss"], iou=(evs[t][n].get("iou") or {}).get("value"),
                            dice=(evs[t][n].get("dice") or {}).get("value"),
                            boundary_D=(evs[t][n].get("boundary_D") or {}).get("value"),
                            first_white_err_s=(evs[t][n].get("first_white") or {}).get("value"),
                            final_iou=(evs[t][n].get("final") or {}).get("iou"),
                            final_boundary_D=(evs[t][n].get("final") or {}).get("bnd_D"),
                            failed=bool(evs[t][n].get("failed"))) for n in ns}
            keys = ["loss", "iou", "dice", "boundary_D", "first_white_err_s", "final_iou", "final_boundary_D"]
            mean = {k: (float(np.mean([r[k] for r in rows.values() if r[k] is not None]))
                        if any(r[k] is not None for r in rows.values()) else None) for k in keys}
            return dict(patterns=rows, mean=mean)
        rep[t] = dict(params=best[t], calibration_fit=summ(cal), test_prediction=summ(list(CB.TEST)),
                      **({"held_out_calibration_clip": summ(held)} if held else {}))
    _jdump(rep, os.path.join(out, f"protocol_N{N}.json"))
    if N_fig:
        with Pool(procs, initializer=_ev_init) as pool:
            fin = pool.map(_ev_final, [(t, best[t], n, N_fig) for t in tags for n in names], chunksize=1)
        W = {(t, n): (w, ok) for t, n, w, ok in fin}
        fig, axes = plt.subplots(1 + len(tags), len(names), figsize=(2.2 * len(names), 2.3 * (1 + len(tags))),
                                 dpi=110, squeeze=False)
        for c, n in enumerate(names):
            clip = CB.load_clip(n)
            res = clip["res"]
            disk = res["disk"].astype(bool)
            rows = [("video", res["white"][-1].astype(float), True)] + [(f"{t} N={N_fig}", *W[(t, n)]) for t in tags]
            for rr, (lab, w, ok) in enumerate(rows):
                img = np.stack([w, w, w], -1) * 0.85 + 0.08
                img[~res["valid"][-1]] = [0.7, 0.3, 0.3]
                img[~disk] = 1.0
                ax = axes[rr, c]
                ax.imshow(img, origin="upper")
                ax.set_xticks([]); ax.set_yticks([])
                role = "calib" if n in CB.CAL_AB else "TEST"
                ax.set_title(f"{n} ({role})\n{lab}" + ("" if ok else " FAILED"), fontsize=7)
        fig.suptitle("final frame: video whiteness vs frozen-parameter model (same observation operator, T frame)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"final_frames_N{N_fig}.png"))
        plt.close(fig)
        np.savez_compressed(os.path.join(out, f"final_frames_N{N_fig}.npz"),
                            **{f"{t}_{n}": W[(t, n)][0].astype(np.float16) for t in tags for n in names})
    return rep


# =========================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "smoke", "status", "report", "evaluate"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", default="AB,A")
    ap.add_argument("--procs", type=int, default=PREREG["cma"]["procs"])
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--N_fig", type=int, default=PREREG["evaluate"]["N_fig"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    tags = tuple(t for t in a.tags.split(",") if t)
    c = PREREG["cma"]
    if a.cmd == "run":
        run(a.out, tags, c["gens"], c["popsize"], a.N or c["N"], a.procs, c["sigma0"], c["seed"], "run", a.force)
    elif a.cmd == "smoke":
        run(a.out, tags, SMOKE["gens"], SMOKE["popsize"], a.N or c["N"], a.procs, c["sigma0"], c["seed"], "smoke",
            a.force)
    elif a.cmd == "status":
        status(a.out)
    elif a.cmd == "report":
        rep = report(a.out)
        print(tables_md(rep))
    elif a.cmd == "evaluate":
        rep = evaluate(a.out, a.N or PREREG["evaluate"]["N"], a.N_fig, a.procs, tags)
        print(json.dumps({t: dict(calibration_fit=r["calibration_fit"]["mean"], test=r["test_prediction"]["mean"])
                          for t, r in rep.items()}, indent=1, default=float))


if __name__ == "__main__":
    # run through the package module so that pool workers pickle functions as latte_imex.v05_calibrate.*
    from latte_imex import v05_calibrate as _me
    _me.main()
