"""chi_strokes.py -- per-stroke deposition fraction chi of the inlet the solver receives (no solver run): the FS action
program (drvf.PhysF via v05_guessed_calib.fs_program) sampled every 1 ms, chi = 1 / (1 + (U_perp^2 / (B_dep d))^p_dep)
(solver.py 'chi' closure), weighted by S_eff, per stroke; for the FS closure (for which the per-stroke heights were
designed) and for CAL (same actions, new closure).  Writes DIR/eval/chi_strokes.json.

    cd /home/user/zfChen && python3 <this> DIR
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import v05_guessed_calib as G   # noqa: E402


def strokes(pattern, closure, dt=1e-3):
    prog = G.fs_program(pattern, closure)
    B, p = closure["B_dep"], closure["p_dep"]
    acc = {}
    for t in np.arange(0.0, prog.duration, dt):
        inl = prog(float(t))
        if not inl.active or inl.S_eff <= 0:
            continue
        k = prog.phase_index(float(t))
        zc = inl.U_perp ** 2 / (B * max(inl.d_jet, 1e-6))
        chi = 1.0 / (1.0 + zc ** p)
        a = acc.setdefault(k, [0.0, 0.0, 0.0])
        a[0] += inl.S_eff * chi * dt
        a[1] += inl.S_eff * dt
        a[2] += dt
    return {str(k): dict(chi=a[0] / a[1], S_int=a[1], dur_s=a[2], cls=prog.cls[k]) for k, a in sorted(acc.items())}


def main():
    root = os.path.abspath(sys.argv[1])
    rec = json.load(open(os.path.join(root, "calib", "calib.json")))
    sets = dict(FS=G.closure_of({}), CAL=G.closure_of({k: rec["best"]["params"][k] for k in G.CLOSURE8}))
    out = {}
    for pat in G.PATTERNS:
        out[pat] = {s: strokes(pat, c) for s, c in sets.items()}
        print(pat, " ".join(f"#{k}:{out[pat]['FS'][k]['chi']:.2f}->{out[pat]['CAL'][k]['chi']:.2f}" for k in out[pat]["FS"]))
    G._jdump(out, os.path.join(root, "eval", "chi_strokes.json"))


if __name__ == "__main__":
    main()
