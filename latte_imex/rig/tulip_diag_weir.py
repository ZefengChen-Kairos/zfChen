import os, sys, json
os.environ.setdefault("NUMBA_NUM_THREADS", "3")
sys.path.insert(0, "/home/user/zfChen")
import latte_imex.rig.real_program as RP
RP.SIM_SUB = "diag_weir_V200"
od, info = RP.solve("tulip", 256, "weir", video=False, V0=200e-6)
M, cs, T = RP.metrics("tulip", "weir", 256)
print("weir", json.dumps(dict(M, poured=info["poured_mL"])), flush=True)
