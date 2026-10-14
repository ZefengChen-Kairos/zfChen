"""Codex-style movie (render.make_video via latte_imex.run.run, unmodified) of the pitcher-delivered S1 runs (PLS1):
the inlet the CAD pitcher model + ideal hand actually delivered, V0.5 closure S1, 256^2.  Checks that the final field
reproduces the stored PLS1 run before keeping the movie.
    python3 make_pls1_video.py <pattern> <out_dir>
"""
import json
import os
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, "/home/user/zfChen")
import numpy as np  # noqa: E402

import latte_imex.run as RR  # noqa: E402
import latte_imex.v05_pourable as VP  # noqa: E402

p, outdir = sys.argv[1], sys.argv[2]
prog = VP.pl_program(f"S1_{p}_ideal", "pitcher", 256)
RR.v05_program = lambda nm: prog
od = os.path.join(outdir, p)
os.makedirs(od, exist_ok=True)
sol, m = RR.run(p, N=256, out=od, video=True, quiet=True, control="v05")
c = sol.concentration()
ref = np.load(os.path.join(VP.PLOUT, "runs256", f"PLS1_{p}_N256", "c.npz"))
d = np.nanmax(np.abs(np.where(ref["mask"], c.astype(np.float32) - ref["c"], 0.0)))
res = dict(pattern=p, max_abs_diff_vs_PLS1=float(d), T=m["T"], frames=m["stats"].get("frames"))
with open(os.path.join(od, "check.json"), "w") as f:
    json.dump(res, f)
print(json.dumps(res), flush=True)
