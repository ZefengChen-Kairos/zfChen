"""inlet_bounds.py -- (1) the frozen calibrated closure against the pre-registered physical bounds (unit-box position by
its own log / lin formula); (2) the received flow of the FS actions under that closure for all six patterns: the inlet
program (drvf.PhysF built by v05_guessed_calib.fs_program) is a function of t alone (the spout lag is pre-filtered on a
fixed 0.5 ms grid), so it is evaluated here on a dense 1 ms grid without the solver; Q = clip(F / kappa_Q, 1, 32 mL/s)
is checked against [1, 32] mL/s and the clip activity (fraction of stream time with F / kappa_Q outside [1, 32]) is
reported, next to the solver-sampled summary of the N = 256 evaluate runs (evaluate/metrics.json).

    python3 inlet_bounds.py OUT.json
"""
import json
import math
import os
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, "/home/user/zfChen")
import numpy as np                                            # noqa: E402
from latte_imex import v05_guessed_calib as G                # noqa: E402

ROOT = os.path.join(os.environ["V05GCAL_REVIEW"], "v05gcal")   # review dir (copy: path from the environment)
ML = 512.0
DT = 1e-3


def unit(v, lo, hi, sc):
    return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if sc == "log" else (v - lo) / (hi - lo)


def dense_inlet(p, closure):
    prog = G.fs_program(p, closure)
    ts = np.arange(0.0, prog.duration, DT)
    n_on = 0
    for t in ts:
        if prog(float(t)).active:
            n_on += 1
    L = np.asarray(prog.log)
    qraw, q = L[:, 4], L[:, 5]
    lo, hi = qraw < 1.0 - 1e-12, qraw > 32.0 + 1e-12
    mod = [float(ph.get("flow_modulation", 0.0)) for ph in prog.phases]
    return dict(duration_s=float(prog.duration), n_samples=int(len(ts)), stream_time_s=float(n_on * DT),
                Q_mLs=[float(q.min()), float(q.max())], Qraw_mLs=[float(qraw.min()), float(qraw.max())],
                Q_within_1_32=bool(q.min() >= 1.0 - 1e-9 and q.max() <= 32.0 + 1e-9),
                clip_low_time_frac=float(lo.mean()), clip_high_time_frac=float(hi.mean()),
                clip_low_time_s=float(lo.sum() * DT),
                V_wanted_mL=float(qraw.sum() * DT), V_received_mL=float(q.sum() * DT),
                U_perp_ms=[float(L[:, 6].min()), float(L[:, 6].max())], exit_speed_min_ms=float(np.nanmin(L[:, 13])),
                jet_r_mm=[float(L[:, 14].min()), float(L[:, 14].max())], pulsation_max=max(mod), kQ=float(prog.kQ))


def main():
    out_path = sys.argv[1]
    pre = json.load(open(os.path.join(ROOT, "prereg.json")))
    rec = json.load(open(os.path.join(ROOT, "calib", "calib.json")))
    em = json.load(open(os.path.join(ROOT, "evaluate", "metrics.json")))
    cal = {k: float(rec["best"]["params"][k]) for k in G.CLOSURE8}
    space = pre["calibrated"]["space"]
    bounds = {}
    for k, (lo, hi, sc) in space.items():
        u = unit(cal[k], lo, hi, sc)
        bounds[k] = dict(value=cal[k], lo=lo, hi=hi, scale=sc, unit=u, inside=bool(lo <= cal[k] <= hi))
    closure = G.closure_of(cal)
    res = dict(cal=cal, closure=closure, bounds=bounds, all_inside=all(b["inside"] for b in bounds.values()),
               fixed_ok=bool(closure["nu"] == 1e-3 and closure["D"] == 1e-7),
               evaluate_closure_equal=em["closures"]["best"] == closure, patterns={})
    for p in G.PATTERNS:
        d = dense_inlet(p, closure)
        ev = em["patterns"][p]["best"]
        ii = ev["inlet"]
        d["solver_sampled_N256"] = dict(Q_mLs=ii["Q_mLs"], Qraw_mLs=ii["Qraw_mLs"],
                                        clip_low_time_frac=ii["clip_low_time_frac"],
                                        clip_high_time_frac=ii["clip_high_time_frac"], V_wanted_mL=ii["V_wanted_mL"],
                                        V_received_mL=ii["V_received_mL"])
        d["solver_params_eq_closure"] = all(ev["solver_params"][k] == closure[k] for k in G.CLOSURE_KEYS)
        d["evaluate_failed"] = bool(ev.get("failed"))
        res["patterns"][p] = d
        print(p, json.dumps({k: d[k] for k in ("Q_mLs", "Qraw_mLs", "Q_within_1_32", "clip_low_time_frac",
                                                "clip_high_time_frac", "pulsation_max", "exit_speed_min_ms")}),
              "| N256:", json.dumps(d["solver_sampled_N256"]), flush=True)
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    print("bounds:", json.dumps({k: round(b["unit"], 4) for k, b in bounds.items()}), "all inside", res["all_inside"])


if __name__ == "__main__":
    main()
