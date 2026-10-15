"""Tests of the GPU design / calibration drivers (gpu_batch, gpu_design, gpu_calib) and the solver-reuse path.

    NUMBA_ENABLE_CUDASIM=1 python -m latte_imex.test_gpu_design --quick      # no GPU: simulator, tiny grids (~30 min)
    python -m latte_imex.test_gpu_design                                       # on the GPU machine (larger grids)
    python -m latte_imex.test_gpu_design --only cpu --quick                    # CPU-only parts (no CUDA at all)

Checks (exit code 1 on failure):
  reset      BatchCudaSolver.reset(params) == a freshly constructed solver (same members, relative 1e-12)
  clip       gpu_batch.ClipMember on the CPU == v05_adapter.V05Model.run (bitwise: c snapshots and ledger)
  design     one tiny design generation (2 candidates + 1 hand-error draw, heart, CPU pitcher): CUDA vs CPU solver,
             every objective term and the final fields agree (|dJ| <= 1e-6 J, max |dc| <= 1e-7)
  calib      one tiny calibration generation (2 candidates on leaf, truncated, milk deposited): CUDA vs CPU losses agree
  smoke      CPU smoke run of gpu_design (2 patterns, 2 generations, tiny N, truncated T) interrupted after one
             generation and resumed == the uninterrupted run (CMA history and best values identical); gpu_calib run
             smoke (1 generation) and its best.json feeding gpu_design --closure
  pitcher    synthetic scale data (ForwardLaw C_d 0.55) -> fit_pitcher recovers C_d within 0.01 (1-D search, CPU)
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np


def _tmp():
    return tempfile.mkdtemp(prefix="gpu_design_test_")


def test_reset(N=20, frames=2):
    from .solver import Params, Numerics
    from .cuda_solver import BatchCudaSolver
    from .test_cuda import _inlet
    num = Numerics(N=N)
    P1 = [Params(cp=1.0, nu=2e-3, closure="chi", return_law="v05", kappa_r=0.1),
          Params(cp=0.8, nu=2e-3, closure="chi", return_law="v05", kappa_r=0.05, B_dep=900.0)]
    P2 = [P1[1], Params(cp=1.2, nu=2e-3, closure="chi", return_law="v05", kappa_r=0.2, beta=1.0)]
    plan = [[_inlet(k) for k in range(frames)], [_inlet(k, 1.3) for k in range(frames)]]
    S = BatchCudaSolver(P1, num)
    for k in range(frames):
        S.advance_frames([plan[b][k] for b in range(2)])
    S.reset(P2)
    for k in range(frames):
        S.advance_frames([plan[b][k] for b in range(2)])
    F = BatchCudaSolver(P2, num)
    for k in range(frames):
        F.advance_frames([plan[b][k] for b in range(2)])
    worst = 0.0
    for b in range(2):
        a, f = S.host_state(b), F.host_state(b)
        for key in a:
            worst = max(worst, float(np.max(np.abs(a[key] - f[key])) / max(np.max(np.abs(f[key])), 1e-30)))
        worst = max(worst, abs(S.t[b] - F.t[b]), abs(S.deposited[b] - F.deposited[b]))
    ok = worst <= 1e-12
    print(f"[reset] reset solver vs fresh solver: max rel diff {worst:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def test_clip(N=24):
    from . import calib as CB
    from . import v05_calibrate as VC
    from .gpu_batch import BatchEngine, ClipMember
    m = CB.make_model(VC.SPEC)
    x0 = VC.initial_points(m.space)[0]
    clip = CB.load_clip("leaf")
    t_out = CB.sample_times(clip)[:4]
    r = m.run(clip["script"], x0, N, t_out)
    mem = [ClipMember(m, clip["script"], x0, N, t_out)]
    BatchEngine(N, "cpu").run(mem)
    r2 = mem[0].result
    dc = float(np.max(np.abs(r["c"] - r2["c"])))
    keys = ("ok", "deposited", "layer_error", "milk_error", "steps", "frames", "V_received_mL", "chi_flow_weighted")
    same = all(r["ledger"][k] == r2["ledger"][k] for k in keys)
    ok = dc == 0.0 and same
    print(f"[clip] ClipMember (CPU engine) vs V05Model.run: max |dc| {dc:.1e}, ledger identical {same} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def _design_cfg(**kw):
    from .gpu_design import DEFAULTS
    cfg = dict(DEFAULTS, popsize=2, gens=1, N_lo=16, N_hi=20, topk=1, rob="P1", T=0.04, closure="s1",
               closure_values=None, pitcher_law=None)
    cfg.update(kw)
    return cfg


def test_design(N=16, T=0.04):
    from . import gpu_design as GD
    cfg = _design_cfg(N_lo=N, T=T)
    rng = np.random.default_rng(3)
    out = {}
    for sb in ("cpu", "cuda"):
        ev = GD.Evaluator(["heart"], cfg, solver_backend=sb, pitcher_backend="cpu", log=lambda *a: None)
        sp = ev.space["heart"]
        xs = [GD.x0_of(sp, ev.base["heart"]), np.clip(GD.x0_of(sp, ev.base["heart"]) + 0.1 * np.random.default_rng(3).standard_normal(len(sp)), 0, 1)]
        cands = ev.pitcher([("heart", x) for x in xs])
        Js, wall, n = ev.solve(cands, N)
        out[sb] = (Js, cands)
    del rng
    Jc, cc = out["cpu"]
    Jg, cg = out["cuda"]
    dJ = max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(Jc, Jg))
    terms = max(abs(a["eval"][N][k] - b["eval"][N][k]) for a, b in zip(cc, cg) for k in ("J_S1", "L_shape", "R", "pen"))
    ok = dJ <= 1e-6 and terms <= 1e-6
    print(f"[design] tiny generation (heart, 2 candidates x 2 runs, N {N}, T {T}): J cpu {['%.6f' % v for v in Jc]} "
          f"cuda {['%.6f' % v for v in Jg]}; max rel dJ {dJ:.1e}, max term diff {terms:.1e} -> {'PASS' if ok else 'FAIL'}",
          flush=True)
    return ok


def test_calib(N=16, T=0.06):
    from . import calib as CB
    from . import gpu_calib as GC
    from .gpu_batch import BatchEngine
    sp = GC.spec()
    model = CB.make_model(sp)
    from . import v05_calibrate as VC
    x0 = VC.initial_points(model.space)
    items = [(x0[0], "leaf"), (x0[2], "leaf")]
    res = {}
    for sb in ("cpu", "cuda"):
        r = GC.forward(BatchEngine(N, sb), model, items, N, T=T)
        res[sb] = r
    dc = max(float(np.max(np.abs(a[1] - b[1]))) for a, b in zip(res["cpu"], res["cuda"]))
    Lc = [GC.score_one((n, c, led, False))["loss"] for n, c, led in res["cpu"]]
    Lg = [GC.score_one((n, c, led, False))["loss"] for n, c, led in res["cuda"]]
    dL = max(abs(a - b) for a, b in zip(Lc, Lg))
    dep = [led["deposited"] for _, _, led in res["cuda"]]
    ok = dc <= 1e-6 and dL <= 1e-6 and min(dep) > 0
    print(f"[calib] tiny generation (2 runs, N {N}, T {T}): deposited {['%.2e' % v for v in dep]}, max |dc| {dc:.1e}, "
          f"losses cpu {['%.4f' % v for v in Lc]} cuda {['%.4f' % v for v in Lg]} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def test_smoke(quick=True):
    from . import gpu_design as GD
    from . import gpu_calib as GC
    root = _tmp()
    try:
        cfg = _design_cfg(gens=2, popsize=3, N_lo=20 if quick else 32, N_hi=24 if quick else 48, confirm_every=1,
                          T=0.25 if quick else 0.5)
        a, b = os.path.join(root, "a"), os.path.join(root, "b")
        kw = dict(solver_backend="cpu", pitcher_backend="cpu")
        GD.run(a, ["heart", "tulip"], dict(cfg), max_gens=1, **kw)
        GD.run(a, ["heart", "tulip"], dict(cfg), **kw)            # resume
        GD.run(b, ["heart", "tulip"], dict(cfg), **kw)            # uninterrupted
        ja, jb = json.load(open(os.path.join(a, "design.json"))), json.load(open(os.path.join(b, "design.json")))
        same = all([(h["J_min"], h["mean"]) for h in ja["patterns"][p]["history"]] ==
                   [(h["J_min"], h["mean"]) for h in jb["patterns"][p]["history"]] for p in ("heart", "tulip"))
        same &= all(ja["patterns"][p]["best_confirm"]["J"] == jb["patterns"][p]["best_confirm"]["J"] for p in ("heart", "tulip"))
        dd = GD.write_data(a)
        wrote = all(os.path.exists(os.path.join(dd, f)) for f in ("poses_heart.npz", "program_tulip.json"))
        # calibration smoke -> closure -> design with that closure
        c = os.path.join(root, "cal")
        GC.run(c, gens=1, popsize=2, N=16, inits=(0,), backend="cpu", T=0.3, procs=1)
        best = GC.write_best(c, quiet=True)
        cfg2 = _design_cfg(gens=1, popsize=2, N_lo=16, N_hi=20, rob="", T=0.2, closure=os.path.join(c, "best.json"))
        cfg2["closure_values"] = GD.load_closure(cfg2["closure"])
        st = GD.run(os.path.join(root, "joint"), ["heart"], cfg2, **kw)
        used = abs(st["rec"]["heart"]["best_confirm"]["eval"]["J_S1"]) >= 0 and \
            GD.load_closure(cfg2["closure"])["B_dep"] == best["params"]["B_dep"]
        ok = same and wrote and used
        print(f"[smoke] CPU design run: resume == uninterrupted {same}; design data written {wrote}; calibration -> "
              f"closure -> design {used} -> {'PASS' if ok else 'FAIL'}", flush=True)
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pitcher(T=0.6):
    from . import gpu_calib as GC
    root = _tmp()
    try:
        f = GC.make_scale("heart", dict(C_d=0.55), os.path.join(root, "scale_heart.npz"), T=T, backend="cpu")
        res = GC.fit_pitcher([f], os.path.join(root, "fit"), fit=("C_d",), backend="cpu", log=lambda *a: None)
        err = abs(res["law"]["C_d"] - 0.55)
        ok = err <= 0.01 and res["rms_g"] < res["start"]["rms_g"]
        print(f"[pitcher] synthetic scale (C_d 0.55): fitted C_d {res['law']['C_d']:.4f}, RMS {res['rms_g']:.4f} g "
              f"(start {res['start']['rms_g']:.3f} g), {res['n_calls']} pitcher calls -> {'PASS' if ok else 'FAIL'}",
              flush=True)
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma list of: reset, clip, design, calib, smoke, pitcher; 'cpu' = the "
                                                 "parts without CUDA")
    a = ap.parse_args(argv)
    sim = os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1"
    only = set(a.only.split(",")) if a.only else set()
    if "cpu" in only:
        only = {"clip", "smoke", "pitcher"}
    small = a.quick or sim
    tests = [("reset", lambda: test_reset(16 if small else 64)),
             ("clip", lambda: test_clip(24)),
             ("design", lambda: test_design(16 if small else 48, 0.04 if small else 0.3)),
             ("calib", lambda: test_calib(16 if small else 48, 0.2 if small else 0.5)),
             ("smoke", lambda: test_smoke(small)),
             ("pitcher", lambda: test_pitcher(0.6))]
    ok = True
    for name, fn in tests:
        if only and name not in only:
            continue
        t0 = time.time()
        r = fn()
        print(f"    ({name}: {time.time() - t0:.0f} s)", flush=True)
        ok &= r
    print("ALL PASS" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
