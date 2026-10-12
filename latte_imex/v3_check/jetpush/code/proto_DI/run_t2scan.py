import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
def run(arg):
    s0, travel, aj = arg
    import px_tests as T
    from px_model import V2
    p = dict(V2); p['alpha_j'] = aj
    o, _ = T.T2(params=p, s0=s0, travel=travel)
    keys = ['near_disp_1', 'gap_min_2', 'gap_min_4', 'gap_min_6', 'contact_4', 'mat_solidity_4', 'mat_area_ratio_4', 'old_raft_area_4', 'old_raft_area_0', 'clip_cm2', 'max_c']
    return s0, travel, aj, {k: (round(o[k], 4) if isinstance(o[k], float) else o[k]) for k in keys}
cases = [(s, 0.0, 8.0) for s in (0.20, 0.25, 0.30)] + [(0.15, 0.10, 8.0), (0.25, 0.0, 5.0), (0.25, 0.0, 12.0), (0.15, 0.10, 5.0)]
with Pool(4) as pool:
    for r in pool.imap(run, cases):
        print('T2scan', *r[:3], json.dumps(r[3]), flush=True)
