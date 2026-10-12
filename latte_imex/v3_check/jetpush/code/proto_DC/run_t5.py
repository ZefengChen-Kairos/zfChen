"""T5 ledger / boundedness numbers at full precision + single-process timing (run alone)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px_tests as T
from px_model import V2
import px_model
orig_finish = T.finish
def finish(sim, obs):
    o = orig_finish(sim, obs)
    L = sim.ledgers()
    o['_clip_exact_cm2'] = sim.L['clip_c'] * 1e4
    o['_redist_cm2'] = sim.L.get('redist', 0.0) * 1e4
    o['_foam_rel'] = L['foam_rel']; o['_surf_err_cm2'] = L['surf_err'] * 1e4
    o['_A_r_cm2'] = L['A_r'] * 1e4
    o['_divres'] = sim.stats['max_divres']
    return o
T.finish = finish
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
tests = sys.argv[2].split(',') if len(sys.argv) > 2 else ['T1', 'T2', 'T3', 'T4']
for name in tests:
    t0 = time.time()
    if name == 'T1': o, _ = T.T1(params=dict(V2), N=N)
    elif name == 'T2': o, _ = T.T2(params=dict(V2), N=N, travel=0.0)
    elif name == 'T3': o, _ = T.T3(params=dict(V2), N=N)
    elif name == 'T4': o, _ = T.T4(params=dict(V2), N=N)
    keys = ['_clip_exact_cm2', '_redist_cm2', '_foam_rel', '_surf_err_cm2', '_A_r_cm2', '_divres', 'max_c', 'min_c', 'wall_per_sim_s', 'ms_per_step', 'ms_poisson', 'ms_vof', 'ms_src', 'iface_width_cells']
    print(name, N, json.dumps({k: o.get(k) for k in keys}), 'wall %.1f' % (time.time() - t0), flush=True)
