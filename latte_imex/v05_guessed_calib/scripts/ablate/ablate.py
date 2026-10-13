"""ablate.py -- review check: one-factor swaps between the FS closure and the calibrated closure (CAL), N = 128,
tulip + leaf (calibration loss) and heart (largest test change).  Which parameter carries the calibration gain and
the hollow heart: (B_dep, p_dep) (the deposition switch) or beta?  Own cache dir; nothing in the main run is touched.

usage: cd /home/user/zfChen && NUMBA_NUM_THREADS=1 python3 ablate.py ROOT [--procs 4]
"""
import json
import os
import sys

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import v05_guessed_calib as G                # noqa: E402

FS = dict(B_dep=1550.4150287146037, p_dep=8.0, kappa_r=0.03725531625168605, kappa_Q=2.0, cp=1.36, beta=3.0,
          kappa_t=0.7, kappa_c=1.0)
CAL = dict(B_dep=1921.4703209826816, p_dep=1.786057475115062, kappa_r=0.171824364079106, kappa_Q=2.2438438653616903,
           cp=1.0611528759578956, beta=1.4099865535403595, kappa_t=0.8299556750844005, kappa_c=0.9710440057074519)
SETS = {
    "FS+beta": dict(FS, beta=CAL["beta"]),
    "FS+dep": dict(FS, B_dep=CAL["B_dep"], p_dep=CAL["p_dep"]),
    "CAL-beta": dict(CAL, beta=FS["beta"]),
    "CAL-dep": dict(CAL, B_dep=FS["B_dep"], p_dep=FS["p_dep"]),
}
NAMES = ("tulip", "leaf", "heart")


def main():
    root = os.path.abspath(sys.argv[1])
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
    os.makedirs(os.path.join(root, "cache"), exist_ok=True)
    jobs = [(prm, n, 128) for prm in SETS.values() for n in NAMES]
    evs = G.forward_many(jobs, root, procs, "review_ablate")
    out, i = {}, 0
    for s in SETS:
        out[s] = {}
        for n in NAMES:
            ev = evs[i]
            i += 1
            out[s][n] = {k: ev.get(k) for k in ("loss", "dice", "bnd_D", "S", "n_model", "white_frac", "failed", "error")}
        out[s]["cal_loss"] = 0.5 * (out[s]["tulip"]["loss"] + out[s]["leaf"]["loss"])
    out["sets"] = SETS
    with open(os.path.join(root, "ablate.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    for s in SETS:
        print(s, "cal %.3f tulip %.3f leaf %.3f heart %.3f" % (out[s]["cal_loss"], out[s]["tulip"]["loss"],
                                                              out[s]["leaf"]["loss"], out[s]["heart"]["loss"]))


if __name__ == "__main__":
    main()
