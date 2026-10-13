"""ablate256.py -- the one-factor swaps of review_ablate/ablate.py (FS <-> CAL: beta alone, (B_dep, p_dep) alone) at
256^2 on all six patterns.  Which parameter carries the test-image changes (hollow heart, layered_heart, push_heart,
swan) at the evaluation grid?  Own cache dir; nothing in the main run is touched.

usage: cd /home/user/zfChen && NUMBA_NUM_THREADS=1 python3 ablate256.py ROOT [--procs 4]
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import v05_guessed_calib as G                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_A128 = os.path.join(HERE, "ablate.py")                       # repo copy: next to this file
if not os.path.exists(_A128):
    _A128 = os.path.join(HERE, "..", "review_ablate", "ablate.py")  # review layout: review/v05gcal/{ablate256,review_ablate}
_spec = importlib.util.spec_from_file_location("ablate128", _A128)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
SETS = A.SETS                                                 # FS+beta, FS+dep, CAL-beta, CAL-dep (same values)
ORDER = ("swan", "leaf", "tulip", "layered_heart", "heart", "push_heart")   # longest first (pool order)


def main():
    root = os.path.abspath(sys.argv[1])
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
    os.makedirs(os.path.join(root, "cache"), exist_ok=True)
    jobs = [(SETS[s], n, 256) for n in ORDER for s in SETS]
    evs = G.forward_many(jobs, root, procs, "review_ablate256")
    out = {s: {} for s in SETS}
    for (prm, n, _), ev, s in zip(jobs, evs, [s for _ in ORDER for s in SETS]):
        out[s][n] = {k: ev.get(k) for k in ("loss", "dice", "iou", "bnd_D", "S", "n_model", "white_frac", "failed",
                                            "error", "transform")}
        if not ev.get("failed"):
            out[s][n]["iou_raw"] = ev["raw"]["iou"]
            out[s][n]["bnd_raw_D"] = ev["raw"]["bnd_D"]
            out[s][n]["V_received_mL"] = ev["inlet"]["V_received_mL"]
            out[s][n]["deposited"] = ev["ledger"]["deposited"]
    for s in SETS:
        out[s]["cal_loss"] = 0.5 * (out[s]["tulip"]["loss"] + out[s]["leaf"]["loss"])
        out[s]["test_mean"] = sum(out[s][n]["loss"] for n in G.TEST) / len(G.TEST)
    out["sets"] = SETS
    with open(os.path.join(root, "ablate256.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    for s in SETS:
        print(s, "cal %.3f test %.3f " % (out[s]["cal_loss"], out[s]["test_mean"])
              + " ".join("%s %.3f" % (n, out[s][n]["loss"]) for n in G.PATTERNS))


if __name__ == "__main__":
    main()
