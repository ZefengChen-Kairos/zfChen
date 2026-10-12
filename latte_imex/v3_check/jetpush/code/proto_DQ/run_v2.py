"""usage: python3 -I run_v2.py TEST VAR1,VAR2 N1,N2   (TEST in T1,T1L,T1b,T2,T2s,T2m,T3,T3d,T4)"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
VARS = {'v2': {}, 'v1': None, 'a5': dict(alpha_j=5.0), 'a12': dict(alpha_j=12.0), 'de08': dict(delta_e=0.8e-3),
        'de12': dict(delta_e=1.2e-3), 'de16': dict(delta_e=1.6e-3), 'aF0': dict(alpha_F=1e-5), 'aF2': dict(alpha_F=0.018),
        'G-': dict(Gamma=0.0316), 'G+': dict(Gamma=0.0474), 'dc025': dict(delta_c=0.25e-3), 'dc075': dict(delta_c=0.75e-3),
        'wcol': dict(w_rule='col'), 'psub2': dict(p_sub=2.0), 'psub23': dict(p_sub=23.0), 'ct1': dict(c_t=1.0), 'ct3': dict(c_t=3.0),
        'noobst': dict(obst=None), 'ring': dict(Ke_mode=None), 'Pi240': dict(Pi_coh=240.0), 'Pi360': dict(Pi_coh=360.0),
        'tw-': dict(tau_w=1.12), 'tw+': dict(tau_w=1.68), 're_all': dict(obst='re'), 're6': dict(delta_e=3.28e-3), 'aF3': dict(alpha_F=0.027), 'obst_rc': dict(obst='rc'), 'obst_rsub': dict(obst='rsub')}
KEYS = {
 'T1': ['t_first_raw', 't_first_vis', 'ring_open_deg_tw+0.1', 'open_deg_tw+0.6', 'open_deg_3', 'hole_r_2', 'hole_r_3', 'hole_r_6',
        'hole_rmax_3', 'c_centre_3', 'dAdt_raft', 'dAdt_vis', 'refill_raft', 'refill_vis', 'far_disp_0.30', 'far_disp_0.30_before_white',
        'specks_n_3', 'specks_mm2_3', 'grey_cells_3', 'umax_after_stop_tr', 'dA_after_stop_tr', 'hole_deq_3', 'hole_deq_6',
        'max_c', 'clip_cm2', 'foam_rel_err', 'surf_err_cm2', 'iface_width_cells', 'wall_per_sim_s', 'ms_per_step'],
 'T2': ['near_disp_1', 'near_disp_2', 'near_disp_4', 'far_disp_4', 'gap_min_2', 'gap_min_4', 'gap_min_6', 'contact_4',
        'mat_solidity_4', 'mat_area_ratio_4', 'old_raft_area_4', 'old_raft_area_0', 'clip_cm2', 'max_c'],
 'T3': ['line_def_imm', 'line_def_post_tr', 'line_def_final', 'two_rd', 'stem_width_entry', 'stem_width_exit',
        'white_on_brown_cm2', 'disc_area_change_cm2', 'notch_req', 'core_cm2', 'supply_cm2', 'max_c', 'clip_cm2'],
 'T4': ['n_stripes', 'pitch', 'n_enclosed', 'enclosed_area_mm2', 'mixed_frac_of_white', 'n_stripes_final', 'max_abs_dc_after',
        'A_vis_end', 'max_c', 'clip_cm2'],
}
def params(var):
    from px_model import V2
    if var == 'v1':
        return dict(geom='pxx', emit_canvas=True)
    p = dict(V2); p.update(VARS[var]); return p
def run(arg):
    test, var, N = arg
    import px_tests as T, v2_tests as V
    p = params(var); t0 = time.time()
    if test == 'T1':
        o, _ = T.T1(params=p, N=N); out = {k: o.get(k) for k in KEYS['T1']}
        L = o
        out['redist_cm2'] = None
    elif test == 'T1L':
        o = V.T1L(params=p, N=N)
        out = {'t_first': o['t_first'], 'clip_cm2': o['clip_cm2'], 'redist_cm2': o['redist_cm2'], 'max_c': o['max_c'],
               'foam_rel_err': o['foam_rel_err'], 'surf_err_cm2': o['surf_err_cm2']}
        for k in ('at+0.6', 'at+1.5', 'at+2.5', 'at+4.5'):
            m = o[k]; out[k] = {kk: m[kk] for kk in ('r_med', 'r_vis_med', 'r_ins', 'open', 'c0', 'lost_share_emit', 'raft_share_sink', 'pressed_per_supply')}
        rows = o['rows']; tl = max(rows); out['after_stop'] = {kk: rows[tl][kk] for kk in ('r_med', 'r_vis_med', 'open', 'c0')}
    elif test == 'T1b':
        o = V.T1b(params=p, N=N); out = {str(k): v for k, v in o.items()}
    elif test in ('T2', 'T2s', 'T2m'):
        s0 = {'T2': 0.15, 'T2s': float(os.environ.get('S0', '0.25')), 'T2m': 0.15}[test]
        travel = 0.10 if test == 'T2m' else 0.0
        o, _ = T.T2(params=p, N=N, s0=s0, travel=travel); out = {k: o.get(k) for k in KEYS['T2']}; out['s0'] = s0
        out['gap_series'] = {t: round(v['gap_min'], 4) if v['gap_min'] == v['gap_min'] else None for t, v in o['series'].items()}
    elif test in ('T3', 'T3d'):
        o, _ = T.T3(params=p, N=N, gate='lift', dwell=0.7 if test == 'T3d' else 0.0); out = {k: o.get(k) for k in KEYS['T3']}
        out['jet'] = {k: round(o['jet'][k] * 1e3, 3) for k in ('r0', 'rd', 'rc', 're', 'De', 'rp', 'rsub')}
        out['jet']['w'] = o['jet']['w']
    elif test == 'T4':
        o, _ = T.T4(params=p, N=N); out = {k: o.get(k) for k in KEYS['T4']}
    out['wall'] = time.time() - t0
    return test, var, N, out
def rnd(x):
    if isinstance(x, float): return round(x, 4)
    if isinstance(x, dict): return {k: rnd(v) for k, v in x.items()}
    return x
if __name__ == '__main__':
    test = sys.argv[1]; vars_ = sys.argv[2].split(','); Ns = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [128]
    cases = [(test, v, N) for v in vars_ for N in Ns]
    with Pool(min(4, len(cases))) as pool:
        for test, var, N, out in pool.imap(run, cases):
            print(test, var, N, json.dumps(rnd(out)), flush=True)
