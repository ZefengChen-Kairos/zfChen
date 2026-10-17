import os, sys, json
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ["PITCHER_GEOM"] = "/home/user/zfChen/latte_imex/pitcher_geom/pitcher_real.json"
sys.path.insert(0, "/home/user/zfChen")
import latte_imex.rig.real_program as RP
p = sys.argv[1]
fill = json.load(open("/home/user/zfChen/raw_data/rig/results/full_pour_rho0.8.json"))[p]["fill_at_cut_mL"]
RP.SIM_SUB = "sim_realpitcher"
od, info = RP.solve(p, 256, "pitcher", video=True, V0=fill * 1e-6)
M, cs, T = RP.metrics(p, "pitcher", 256)
print(p, json.dumps(dict(M, poured=info["poured_mL"], V0=info["V0_mL"])), flush=True)
