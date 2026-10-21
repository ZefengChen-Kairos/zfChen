"""CMA-ES calibration of the closure on whole-video scores (forward runs only; the structure score is not differentiable).

Objective: the mean over the calibration pours of  (1 - w_iou) S/S_ref + w_iou IoU/IoU_ref: S the structure score
(diff/score.py), IoU the video IoU, each the time-weighted mean over the frames with enough observed white
(score.frame_ok), and S_ref, IoU_ref the video's own values 0.25 s apart (score.lag_reference: 1 = as close to the video
as the video is to itself 0.25 s later).  w_iou <= 0.3 keeps the structure guarantee (a filled-in or blob-like field
must not win).  Parameters in the unit box of fit.SPACE (+ the
mechanism spaces of the chosen model).  After the fit the frozen closure is scored on the other pours (test.json).

    python -m latte_imex.diff.cmafit --out results/cma_AB --cal tulip,leaf --model dm --e0 world --N 48 \
        --gens 16 --pop 12 [--device cuda:0] [--workers 2] [--w-iou 0.25] [--init closure.json] [--set k=v,...]

--model letters: d (tau_s: foam surfaces ahead of the landing point), m (mu_s: jet momentum the surfacing foam has
lost), c (surface current driven by the stream's momentum: kappa_M, eps_f, kappa_w, xi_X; its shape --circ-r (spin-down
rate [1/s]), --circ-L, --circ-rf), a (alpha_u: underplating, new foam surfacing under an existing raft thickens it),
u (upwell), e / j / k / y (fit.SINK / VJ / KIN / YIELD_SPACE), o (two-phase open floating layer: xi_b brown softness,
lam_b brown subduction rate, fit.OPEN_SPACE); letters compose (e.g. 'co'); 'v05' = none.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from ..v05_adapter import SPACE
from . import schedule as SC
from .fit import (CIRC_SPACE, CIRC_START, DEP_SPACE, DEP_START, KIN_SPACE, MECH_START, OPEN_SPACE, OPEN_START, RAFT_SPACE,
                  RAFT_START, SINK_SPACE, VJ_SPACE, YIELD_SPACE)
from . import score as SS

PATTERNS = ("tulip", "leaf", "heart", "push_heart", "swan")


def model_space(model):
    f = "" if model == "v05" else model
    SP = dict(SPACE)
    if "c" in f:
        SP.update(CIRC_SPACE)
    if "d" in f:
        SP["tau_s"] = DEP_SPACE["tau_s"]
    if "m" in f:
        SP["mu_s"] = DEP_SPACE["mu_s"]
    if "a" in f:
        SP.update(RAFT_SPACE)
    fix = {}
    if "e" in f:                                     # entrainment sink of the diving stream
        SP.update(SINK_SPACE)
    if "j" in f:                                     # traction towards the landing point's velocity V_j
        SP.update(VJ_SPACE)
        fix["vj"] = 1.0
    if "k" in f:                                     # kinematic deposition + landing crater
        SP.update(KIN_SPACE)
        fix["mu_s"] = 1.0
    if "y" in f:                                     # yield stress of the floating layer
        SP.update(YIELD_SPACE)
    if "o" in f:                                     # two-phase open floating layer (soft, subducting brown surface)
        SP.update(OPEN_SPACE)
    return SP, dict(circ="c" in f, upwell="u" in f, fix=fix)


def to_unit(P, SP):
    return np.array([math.log(P[k] / SP[k][0]) / math.log(SP[k][1] / SP[k][0]) if SP[k][2] == "log"
                     else (P[k] - SP[k][0]) / (SP[k][1] - SP[k][0]) for k in SP]).clip(0, 1)


def from_unit(u, base, SP):
    P = dict(base)
    for k, x in zip(SP, np.clip(u, 0, 1)):
        lo, hi, sc = SP[k]
        P[k] = lo * (hi / lo) ** x if sc == "log" else lo + (hi - lo) * x
    return P


def run_fields(D):
    """Simulated c at the observed frames of the last forward run of DiffPour D, with the video frames and weights."""
    msk = D.geo.mask_np
    C = [np.where(msk, D.sol.snaps[o["k"]]["m"].numpy() / np.where(msk, D.sol.snaps[o["k"]]["l"].numpy(), 1.0), 0.0)
         for o in D.obs]
    return C, [o["wo_np"] for o in D.obs], [o["vo_np"] for o in D.obs], [o["wt"] for o in D.obs]


_D = {}
_CFG = {}


def _init(cfg):
    from .run import DiffPour
    _CFG.update(cfg)
    SS.set_grid(cfg["N"])                  # score constants at the same physical size on every grid
    for p in cfg["pours"]:
        D = DiffPour(p, N=cfg["N"], device=cfg["device"], tau=0.05, scales=(0.3, 0.15, 0.0), e0=cfg["e0"],
                     exit_cal=cfg["exit_cal"], upwell=cfg["upwell"], circ=cfg["circ"], circ_shape=cfg["circ_shape"],
                     log=lambda *a: None)
        D.score_ref = SS.lag_reference([o["t"] for o in D.obs], [o["wo_np"] for o in D.obs], [o["vo_np"] for o in D.obs],
                                       D.geo.mask_np)
        _D[p] = D


def evaluate(P, pours=None):
    res = {}
    for p in pours or _CFG["pours"]:
        D = _D[p]
        try:
            D.loss(P)
            C, W, V, wt = run_fields(D)
            res[p] = SS.pour_scores(C, W, V, wt, D.geo.mask_np, D.score_ref)
        except Exception as e:                       # a closure that breaks the solver scores 0
            res[p] = dict(viou=0.0, struct=0.0, viou_n=0.0, struct_n=0.0, err=str(e)[:160])
    return res


def objective(r, w_iou):
    return float(np.mean([w_iou * x["viou_n"] + (1 - w_iou) * x["struct_n"] for x in r.values()]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cal", default="tulip,leaf")
    ap.add_argument("--test", default=None, help="pours scored with the frozen closure (default: the other patterns)")
    ap.add_argument("--model", default="v05")
    ap.add_argument("--e0", default="world", choices=("jet", "world", "none"))
    ap.add_argument("--exit-cal", dest="exit_cal", action="store_true")
    ap.add_argument("--circ-L", dest="circ_L", type=float, default=0.4, help="surface current spreading length (circulation.L_C)")
    ap.add_argument("--circ-rf", dest="circ_rf", type=float, default=0.06, help="surface current forcing radius (circulation.R_F)")
    ap.add_argument("--circ-r", dest="circ_r", type=float, default=1.0,
                    help="surface current spin-down rate [1/s] (circulation.R_C); the steady current goes as kappa_M / r, "
                         "so the kappa_M box and start are scaled by r (the same current amplitudes as at r = 1)")
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--gens", type=int, default=16)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--w-iou", dest="w_iou", type=float, default=0.25)
    ap.add_argument("--init", default=None)
    ap.add_argument("--set", default="")
    ap.add_argument("--bound", default="",
                    help="override parameter bounds, e.g. B_dep=1230:2210 (the deposition threshold measured from the "
                         "videos: foam survives below Fr_c^2 = 11-20, V2_SURFACE_MODEL.md 10)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0, help="0: in-process (one device); >0: forked worker processes (CPU)")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    import cma
    os.makedirs(a.out, exist_ok=True)
    log = lambda *x: print(*x, flush=True)
    cal = a.cal.split(",")
    test = [p for p in (a.test.split(",") if a.test else PATTERNS) if p not in cal]
    SP, sw = model_space(a.model)
    circ_start = dict(CIRC_START)
    if sw["circ"] and a.circ_r != 1.0:               # U_ss = (r - r L^2 lap + ...)^-1 F: amplitude kappa_M / r
        lo, hi, sc = SP["kappa_M"]
        SP["kappa_M"] = (lo * a.circ_r, hi * a.circ_r, sc)
        circ_start["kappa_M"] *= a.circ_r
    for k, v in (x.split("=") for x in a.bound.split(",") if x):
        lo, hi = map(float, v.split(":"))
        SP[k] = (lo, hi, SP[k][2] if k in SP else "lin")
    e0 = SC.e0_mode(0 if a.e0 == "none" else a.e0)
    cfg = dict(pours=cal, N=a.N, device=a.device, e0=e0, exit_cal=a.exit_cal,
               circ_shape=dict(L=a.circ_L, r_f=a.circ_rf, r=a.circ_r), **sw)
    from .run import DiffPour
    base = dict(DiffPour(cal[0], N=16, device="cpu", log=lambda *x: None).s1)
    base.update(circ_start if sw["circ"] else {"kappa_M": 0.0, "eps_f": 0.0, "kappa_w": 0.0})
    base.update({k: v for k, v in dict(DEP_START, **RAFT_START, **MECH_START, **OPEN_START).items() if k in SP})
    if a.init:
        base.update({k: float(v) for k, v in json.load(open(a.init)).items() if isinstance(v, (int, float)) and not k.startswith("_")})
    base.update(sw.pop("fix"))                       # switches of the model letters (vj, mu_s = 1)
    for k, v in (x.split("=") for x in a.set.split(",") if x):
        base[k] = float(v)
    for k in SP:                                     # start inside the box
        lo, hi, _ = SP[k]
        base[k] = min(max(base[k], lo), hi)
    switches = dict(e0=float(SC.e0_code(e0)), exit_cal=float(a.exit_cal), upwell=float(sw["upwell"]), circ=float(sw["circ"]),
                    circ_L=a.circ_L, circ_rf=a.circ_rf, circ_r=a.circ_r)
    log(f"CMA-ES {a.model}: {len(SP)} parameters, calibration {cal}, test {test}, N {a.N}, e0 {a.e0}, "
        f"objective {a.w_iou:g} IoU + {1 - a.w_iou:g} structure (normalised by the video's 0.25 s self-consistency), "
        f"{a.gens} x {a.pop}" + (f"; surface current r {a.circ_r:g} /s, L {a.circ_L:g}, r_f {a.circ_rf:g}, kappa_M box "
                                 f"{SP['kappa_M'][0]:g}-{SP['kappa_M'][1]:g}" if sw["circ"] else ""))
    es = cma.CMAEvolutionStrategy(to_unit(base, SP), a.sigma, dict(bounds=[0, 1], popsize=a.pop, seed=a.seed, verbose=-9))
    best = None
    pool = None
    if a.workers > 0:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(cfg,))
    else:
        _init(cfg)
    try:
        for g in range(a.gens):
            t0 = time.time()
            U = es.ask()
            Ps = [from_unit(u, base, SP) for u in U]
            R = pool.map(evaluate, Ps) if pool else [evaluate(P) for P in Ps]
            f = [-objective(r, a.w_iou) for r in R]
            es.tell(U, f)
            i = int(np.argmin(f))
            if best is None or -f[i] > best["score"]:
                best = dict(score=-f[i], gen=g, P={k: Ps[i][k] for k in SP}, per=R[i])
                json.dump(dict(Ps[i], **switches), open(os.path.join(a.out, "best_closure.json"), "w"), indent=1)
            with open(os.path.join(a.out, "gens.jsonl"), "a") as fh:
                fh.write(json.dumps(dict(gen=g, best_gen=-f[i], mean_gen=-float(np.mean(f)), P={k: Ps[i][k] for k in SP},
                                         per=R[i])) + "\n")
            log(f"gen {g:2d}: best {-f[i]:.3f} mean {-np.mean(f):.3f} | overall {best['score']:.3f} (gen {best['gen']}) "
                + " ".join(f"{p} {R[i][p]['viou']:.3f}/{R[i][p]['struct']:.3f}" for p in cal) + f"  {time.time() - t0:.0f} s")
    finally:
        if pool:
            pool.close()
    json.dump(best, open(os.path.join(a.out, "best.json"), "w"), indent=1)
    P = dict(base, **best["P"])
    if test:
        log(f"frozen-closure test on {test}")
        if pool:
            cfg_t = dict(cfg, pours=test)
            import multiprocessing as mp
            with mp.get_context("fork").Pool(min(a.workers, len(test)), initializer=_init, initargs=(cfg_t,)) as pt:
                parts = pt.starmap(evaluate, [(P, [p]) for p in test])
            rt = {k: v for d in parts for k, v in d.items()}
        else:
            _init(dict(cfg, pours=test))
            rt = evaluate(P, test)
        for p, r in rt.items():
            log(f"  test {p:10s} video IoU {r['viou']:.3f} ({r.get('viou_n', 0):.2f} of the video's own)  structure "
                f"{r['struct']:.3f} ({r.get('struct_n', 0):.2f})  objective {objective({p: r}, a.w_iou):.3f}")
        log(f"  held-out mean: video IoU {np.mean([r['viou'] for r in rt.values()]):.3f}  "
            f"structure {np.mean([r['struct'] for r in rt.values()]):.3f}  objective {objective(rt, a.w_iou):.3f}")
        json.dump(dict(cal=cal, calibration=best["per"], test=rt, params=P, switches=switches),
                  open(os.path.join(a.out, "test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
