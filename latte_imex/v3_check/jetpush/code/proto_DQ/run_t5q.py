"""T5 ledgers / boundedness / interface width + single-process timing (run alone).  usage: python3 -I run_t5q.py N VARS"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dq_tests as Q
import px_tests as T
from px_model import V2
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
vars_ = sys.argv[2].split(',') if len(sys.argv) > 2 else ['v2', 'q']
KEYS = ['foam_rel_err', 'surf_err_cm2', 'max_divres', 'clip_net_cm2', 'redist_cm2', 'redist_per_supply', 'max_c', 'min_c',
        'c_max_after', 'c_min_after', 'iface_width_cells', 'supply_cm2', 'pressed_cm2', 'core_cm2', 'max_courant', 'steps',
        'wall_per_sim_s', 'ms_per_step', 'ms_poisson', 'ms_vof', 'ms_src']
for var in vars_:
    p = dict(V2); p['jetU'] = 1.0 if var == 'q' else 0.0
    for name in ('T1', 'T2h', 'T2m', 'T3', 'T3d', 'T4f', 'T4b'):
        t0 = time.time()
        with open('/proc/loadavg') as f: la = f.read().split()[0]
        if name == 'T1': o, _ = T.T1(params=p, N=N)
        elif name == 'T2h': o, _ = T.T2(params=p, N=N, s0=0.15, travel=0.0)
        elif name == 'T2m': o, _ = T.T2(params=p, N=N, s0=0.15, travel=0.10)
        elif name == 'T3': o, _ = Q.T3Q(params=p, N=N)
        elif name == 'T3d': o, _ = Q.T3Q(params=p, N=N, dwell=0.7)
        elif name == 'T4f': o, _ = Q.T4Q(params=p, N=N)
        elif name == 'T4b': o, _ = Q.T4Q(params=p, N=N, base=True)
        out = {k: o.get(k) for k in KEYS}
        out['load_before'] = float(la); out['wall'] = time.time() - t0
        print(json.dumps(dict(var=var, test=name, N=N, obs=out)), flush=True)
