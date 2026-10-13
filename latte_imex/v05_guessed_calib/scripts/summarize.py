"""Summary of the pre-registered calibration (DIR/calib/calib.json + DIR/cache): references, best per init and overall,
best closure (physical), parameters within 2 % of a bound (unit box), distance between the inits' bests, final sigma,
n_forward, n_failed, wall / compute time.  Writes DIR/calib_summary.json and prints it.

    python3 summarize.py DIR
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import calib as CB          # noqa: E402

D = os.path.abspath(sys.argv[1])
rec = json.load(open(os.path.join(D, "calib", "calib.json")))
space = rec["space"]
keys = list(space)


def unit(p):
    return np.array(CB.encode(p, space))


def near_bound(p, tol=0.02):
    u = unit(p)
    return {k: ("lo" if u[i] <= tol else "hi") for i, k in enumerate(keys) if u[i] <= tol or u[i] >= 1 - tol}


def per_pat(evs):
    out = {}
    for n, e in evs.items():
        if e.get("failed"):
            out[n] = dict(loss=e["loss"], failed=True)
            continue
        out[n] = dict(L=e["loss"], dice=e["dice"], bnd_D=e["bnd_D"], S=e["S"], n_model=e["n_model"], n_video=e["n_video"],
                      transform={k: e["transform"][k] for k in ("theta_deg", "dx_D", "dy_D")},
                      at_bound=e["transform"].get("at_bound"), iou_raw=e["raw"]["iou"], bnd_raw_D=e["raw"]["bnd_D"],
                      L_raw=e["raw"].get("L"), V_received_mL=e["inlet"]["V_received_mL"],
                      clip_low=e["inlet"]["clip_low_time_frac"], clip_high=e["inlet"]["clip_high_time_frac"])
    return out


summ = dict(complete="best" in rec, N=rec["N"], gens=rec["gens"], popsize=rec["popsize"], sigma0=rec["sigma0"],
            n_forward=rec["n_forward"], n_failed=rec["n_failed"], inits=[])
drv = rec.get("driver", {})
if drv:
    summ["references"] = [dict(init=r["init"], loss=r["loss"],
                               per_pattern={n: r["per_pattern"][n]["loss"] for n in r["per_pattern"]})
                          for r in drv["x0_reference"]]
    summ["wall_s_last_process"] = drv["wall_s"]
for i, r in enumerate(rec["inits"]):
    h = r["history"]
    b = r["best"]
    summ["inits"].append(dict(init=i, gens_done=len(h), best_loss=b["loss"], best_gen=b["gen"],
                              best_params=b["params"], best_unit=[round(float(v), 4) for v in unit(b["params"])],
                              near_bound=near_bound(b["params"]), per_pattern=per_pat(b["evals"]),
                              final_sigma=h[-1]["sigma"] if h else None, last_gen_min=h[-1]["loss_min"] if h else None,
                              last_gen_med=h[-1]["loss_med"] if h else None,
                              final_mean=h[-1]["mean"] if h else None,
                              final_mean_unit=[round(float(v), 4) for v in unit(h[-1]["mean"])] if h else None,
                              gen_min=[round(x["loss_min"], 4) for x in h]))
if len(rec["inits"]) >= 2:
    U = [unit(r["best"]["params"]) for r in rec["inits"]]
    summ["best_unit_distance"] = dict(euclid=float(np.linalg.norm(U[0] - U[1])),
                                      per_param={k: round(float(U[1][i] - U[0][i]), 4) for i, k in enumerate(keys)})
    summ["best_loss_spread"] = [r["best"]["loss"] for r in rec["inits"]]
if "best" in rec:
    b = rec["best"]
    summ["best"] = dict(loss=b["loss"], params=b["params"], near_bound=near_bound(b["params"]),
                        from_init=int(np.argmin(b["spread"])))
recs = [json.load(open(f)) for f in glob.glob(os.path.join(D, "cache", "*.json"))]
cal = [r for r in recs if r.get("tag") == "cal"]
if cal:
    w = np.array([r["wall_s"] for r in cal])
    summ["cal_cache"] = dict(records=len(cal), failed=sum(bool(r["eval"].get("failed")) for r in cal),
                             wall_median_s=float(np.median(w)), wall_max_s=float(w.max()),
                             compute_s=float(w.sum()), first=min(r["t_start"] for r in cal),
                             last=max(r["t_end"] for r in cal),
                             span_h=(max(r["t_end"] for r in cal) - min(r["t_start"] for r in cal)) / 3600)
    errs = sorted({r["eval"].get("error", "")[:120] for r in cal if r["eval"].get("failed")})
    summ["cal_cache"]["errors"] = errs
json.dump(summ, open(os.path.join(D, "calib_summary.json"), "w"), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
