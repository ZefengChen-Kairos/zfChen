"""fresh_run.py -- one forward run of the frozen calibrated closure (calib.json best) with the FS actions, in a fresh
process, through the driver's own run function (v05_guessed_calib.run_fs); no calibration cache is read or written.
Saves the float64 final field, mask, inlet log and what the solver received.

    NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=<fresh dir> python3 fresh_run.py <pattern> <N> <outdir>
"""
import json
import os
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, "/home/user/zfChen")
import numpy as np                                            # noqa: E402
from latte_imex import v05_guessed_calib as G                # noqa: E402

ROOT = os.path.join(os.environ["V05GCAL_REVIEW"], "v05gcal")   # review dir (copy: path from the environment)


def main():
    p, N, out = sys.argv[1], int(sys.argv[2]), os.path.abspath(sys.argv[3])
    os.makedirs(os.path.join(out, "tmp"), exist_ok=True)
    rec = json.load(open(os.path.join(ROOT, "calib", "calib.json")))
    params = {k: float(rec["best"]["params"][k]) for k in G.CLOSURE8}
    closure = G.closure_of(params)
    t0 = time.time()
    r = G.run_fs(p, closure, N, workdir=os.path.join(out, "tmp"))
    np.savez(os.path.join(out, f"{p}_N{N}.npz"), c=r["c"], mask=r["mask"], log=r["log"])
    meta = dict(pattern=p, N=N, params=params, closure=closure, solver_params=r["solver_params"], kQ=r["kQ"],
                program_params=r["program_params"], inlet=r["inlet"], ledger={k: float(v) for k, v in r["ledger"].items()
                                                                               if np.isscalar(v)},
                stats={k: r["stats"].get(k) for k in ("steps", "frames", "max_substeps", "retries", "clip_mass", "max_u")},
                t_end=r["t_end"], wall_s=r["wall_s"], wall_total_s=time.time() - t0,
                env=dict(NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS"),
                         NUMBA_CACHE_DIR=os.environ.get("NUMBA_CACHE_DIR"), pid=os.getpid()))
    json.dump(meta, open(os.path.join(out, f"{p}_N{N}.json"), "w"), indent=1, default=float)
    print(json.dumps(dict(pattern=p, N=N, wall_s=r["wall_s"], inlet=r["inlet"]), default=float), flush=True)


if __name__ == "__main__":
    main()
