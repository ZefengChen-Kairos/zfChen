"""Tulip petal-opening diagnostic: rerun the measured tulip pour with one change at a time."""
import os, sys, json
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, "/home/user/zfChen")
import latte_imex.rig.real_program as RP
name, V0 = sys.argv[1], (float(sys.argv[2]) if sys.argv[2] != "cal" else None)
over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
orig = RP.phases_meta
def pm(*a, **k):
    m = orig(*a, **k)
    for key, f in over.items(): m["config"]["parameters"][key] *= f
    return m
RP.phases_meta = pm
RP.SIM_SUB = "diag_" + name
od, info = RP.solve("tulip", 256, "pitcher", video=False, V0=None if V0 is None else V0 * 1e-6)
M, cs, T = RP.metrics("tulip", "pitcher", 256)
print(name, json.dumps(dict(M, poured=info["poured_mL"], V0=info["V0_mL"])), flush=True)
