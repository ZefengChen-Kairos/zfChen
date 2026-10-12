"""Key numbers 128^2 vs 192^2 (eta_J = 0 and 1e-3)."""
import os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
def load(N):
    rows = []
    for f in glob.glob(os.path.join(HERE, 'out', f'scan_N{N}_*.jsonl')):
        rows += [json.loads(l) for l in open(f)]
    return rows
A = {(r['kind'], json.dumps(r['kw'], sort_keys=True)): r['o'] for r in load(128)}
B = {(r['kind'], json.dumps(r['kw'], sort_keys=True)): r['o'] for r in load(192)}
KEYS = {'T1': ['t_first_vis', 'dAdt_raft', 'hole_r_3', 'hole_ins', 'open_deg_3', 'specks_mm2_3', 'far_disp_0.30_before_white',
               'far_disp_0.30', 'umax_after_tr_mm_s', 'drift_0.30_tr_end', 'drift_0.20_tr_end', 'iface_width_cells'],
        'T2': ['near_1.4', 'near_4', 'far_4', 'gap_min_2', 'gap_min_4', 'gap_min_6', 'sol_mat_4', 'sol_mat_10', 'sol_old_4',
               'wrap_mat_4', 'creep_near_4.6_10'],
        'T3': ['notch_req', 'entry_bulge_max', 'exit_bulge_max', 'line_def_final', 'line_vis_width_final', 'stem_width_entry', 'stem_width_exit'],
        'T4': ['n_stripes', 'pitch', 'n_enclosed', 'enclosed_area_mm2', 'mixed_frac_of_white', 'n_stripes_final', 'max_abs_dc_after']}
for key in sorted(B):
    kind, kw = key
    if key not in A: continue
    a, b = A[key], B[key]
    print(kind, kw)
    for k in KEYS[kind]:
        print('   %-28s 128: %-10s 192: %-10s' % (k, a.get(k), b.get(k)))
    if kind == 'T3':
        print('   V cut+3 0.005/0.02:', a['vshape']['cut+3']['0.005'], a['vshape']['cut+3']['0.02'], '|', b['vshape']['cut+3']['0.005'], b['vshape']['cut+3']['0.02'],
              'hw', a['vshape']['cut+3']['half_width'], b['vshape']['cut+3']['half_width'])
    print('   wall/sim-s', a.get('wall_per_sim_s'), b.get('wall_per_sim_s'), 'steps', a.get('steps'), b.get('steps'))
