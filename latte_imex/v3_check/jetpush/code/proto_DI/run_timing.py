"""Single-process timing (run alone): wall per simulated second and ms per step, T1 (13 s) and T4 (8.5 s), eta_J = 0 / 1e-4 / 1e-3."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import DI_tests as DT
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
for eta in (0.0, 1e-4, 1e-3):
    for name, fn in (('T1', lambda: DT.T1x(eta=eta, N=N)), ('T4', lambda: DT.T4x(eta=eta, N=N))):
        t0 = time.time(); o, _ = fn(); w = time.time() - t0
        print(json.dumps(dict(test=name, N=N, eta=eta, wall_per_sim_s=round(o['wall_per_sim_s'], 3), ms_per_step=round(o['ms_per_step'], 2),
                              steps=o['steps'], wall_total=round(w, 1))), flush=True)
