"""ident.py -- identifiability check of the calibrated closure: do closures that fit tulip + leaf (almost) equally well
predict the same four test patterns?  Forward calls through the driver's cached / time-limited worker (same cache, same
cache key, N = 128 = the calibration grid), tag 'ident'.

Closure set (fixed before any test pattern is run at N = 128):
  init0_best  best CMA sample of start 0 (FS closure)           (calib.json inits[0].best)
  init1_best  best CMA sample of start 1 (FSp2 closure) = CAL    (calib.json inits[1].best = rec['best'])
  near_k      the 8 distinct CMA samples with the lowest calibration loss (tulip + leaf, N = 128), ranks 1..8; both
              init bests are among them (ranks 1 and 3)
  FS          the FS closure (reference, not calibrated)
Patterns: heart, push_heart, layered_heart, swan (tests) at N = 128.  tulip / leaf at N = 128 are already in the cache.

    cd /home/user/zfChen && NUMBA_NUM_THREADS=1 python3 <this> DIR [--procs 4]
writes DIR/eval/ident_set.json (closure set, before the runs) and DIR/eval/ident_runs.json (the eval records).
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import v05_guessed_calib as G   # noqa: E402

N_ID = 128
TOP = 8
# start 1 (FSp2) began after start 0's last generation was logged (calib.log line 29, 17:26:17 local); cache records
# computed later belong to start 1
T_SPLIT = "2026-10-13 17:26:17"


def closure_set(root):
    rec = json.load(open(os.path.join(root, "calib", "calib.json")))
    t_split = time.mktime(time.strptime(T_SPLIT, "%Y-%m-%d %H:%M:%S"))
    by = {}
    for f in glob.glob(os.path.join(root, "cache", "*.json")):
        d = json.load(open(f))
        if d["N"] != N_ID or d["tag"] != "cal" or d["eval"].get("failed"):
            continue
        k = json.dumps(d["params"], sort_keys=True)
        e = by.setdefault(k, dict(params=d["params"], L={}, t_end=d["t_end"]))
        e["L"][d["name"]] = d["eval"]["loss"]
        e["t_end"] = min(e["t_end"], d["t_end"])
    rows = [dict(v, loss=0.5 * (v["L"]["tulip"] + v["L"]["leaf"]), start=0 if v["t_end"] <= t_split else 1)
            for v in by.values() if "tulip" in v["L"] and "leaf" in v["L"]]
    rows.sort(key=lambda r: r["loss"])
    sel = {}
    for i, r in enumerate(rows[:TOP]):
        sel[f"near_{i + 1}"] = dict(params=r["params"], cal_loss=r["loss"], L_tulip=r["L"]["tulip"], L_leaf=r["L"]["leaf"],
                                    start=r["start"], rank=i + 1)
    b0, b1 = rec["inits"][0]["best"]["params"], rec["inits"][1]["best"]["params"]
    for nm, b in (("init0_best", b0), ("init1_best", b1)):
        hit = [k for k, v in sel.items() if all(abs(v["params"][q] - b[q]) <= 1e-12 * abs(b[q]) for q in G.CLOSURE8)]
        sel[nm] = dict(params=b, same_as=hit[0] if hit else None)
    sel["FS"] = dict(params=G.init_points()["FS"], reference=True)
    return sel, len(rows)


def main():
    root = os.path.abspath(sys.argv[1])
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
    pre = json.load(open(os.path.join(root, "prereg.json")))
    G.check_unchanged(pre, root, G.PATTERNS)
    ed = os.path.join(root, "eval")
    os.makedirs(ed, exist_ok=True)
    sel, n_rows = closure_set(root)
    G._jdump(dict(N=N_ID, top=TOP, n_cal_samples=n_rows, set=sel, written=time.strftime("%Y-%m-%d %H:%M:%S")),
             os.path.join(ed, "ident_set.json"))
    names = [k for k in sel if not (k.startswith("init") and sel[k].get("same_as"))]
    jobs = [(sel[k]["params"], p, N_ID) for k in names for p in G.TEST]
    t0 = time.time()
    evs = G.forward_many(jobs, root, procs, "ident")
    out = {}
    for (params, p, _), ev, k in zip(jobs, evs, [k for k in names for _ in G.TEST]):
        out.setdefault(k, {})[p] = ev
    G._jdump(dict(N=N_ID, names=names, runs=out, wall_s=time.time() - t0, procs=procs),
             os.path.join(ed, "ident_runs.json"))
    for k in names:
        print(k, " ".join(f"{p} {out[k][p]['loss']:.3f}" for p in G.TEST), flush=True)


if __name__ == "__main__":
    main()
