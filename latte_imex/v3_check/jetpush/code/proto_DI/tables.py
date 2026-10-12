"""Print compact tables from out/scan_*.jsonl for test_Iinertia.md.  usage: python3 -I tables.py [N]"""
import os, sys, json, glob, math
HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
rows = []
for f in glob.glob(os.path.join(HERE, 'out', f'scan_N{N}_*.jsonl')):
    rows += [json.loads(l) for l in open(f)]


def f(x, nd=4):
    if x is None: return '—'
    if isinstance(x, bool): return str(x)
    if isinstance(x, int): return str(x)
    if isinstance(x, float):
        if x != x: return 'nan'
        if x == 0: return '0'
        if abs(x) < 1e-3 or abs(x) >= 1e4: return f'{x:.2e}'
        return f'{x:.{nd}g}'
    return str(x)


def eta_s(e):
    return '0' if e == 0 else f'{e:.0e}'.replace('e-0', 'e-')


T1 = sorted([r for r in rows if r['kind'] == 'T1'], key=lambda r: r['kw']['eta'])
if T1:
    print('\n## T1  (N=%d)' % N)
    keys = ['t_first_raw', 't_first_vis', 'dAdt_raft', 'dAdt_vis', 'ring_rin_tw+0.1_mm', 'ring_open_deg_tw+0.1', 'hole_r_2', 'hole_r_3',
            'hole_ins', 'open_deg_2', 'open_deg_3', 'specks_n_3', 'specks_mm2_3', 'far_disp_0.30_before_white', 'disp_0.15_pre_white',
            'far_disp_0.30', 'hole_r_6', 'open_deg_6', 'c_centre_6',
            'umax_after_tr_mm_s', 'umax_6_mm_s', 'umax_end_mm_s', 'max_dc_tr_6', 'max_dc_tr_end', 'dA_after_stop_tr_end',
            'drift_0.10_tr_6', 'drift_0.10_tr_end', 'drift_0.15_tr_end', 'drift_0.20_tr_end', 'drift_0.30_tr_end', 'drift_0.40_tr_end',
            'drift_0.30_mean_x_end', 'iface_width_cells_3', 'iface_width_cells', 'foam_rel_err', 'surf_err_cm2', 'impulse_rel_err',
            'max_divres', 'max_c', 'min_c', 'redist_cm2', 'wall_per_sim_s', 'steps']
    print('| 量 | ' + ' | '.join('η_J = ' + eta_s(r['kw']['eta']) for r in T1) + ' |')
    print('|---|' + '---|' * len(T1))
    for k in keys:
        print(f'| {k} | ' + ' | '.join(f(r['o'].get(k)) for r in T1) + ' |')

T2 = [r for r in rows if r['kind'] == 'T2']
if T2:
    print('\n## T2  (N=%d)' % N)
    T2.sort(key=lambda r: (r['kw']['travel'], r['kw']['s0'], r['kw']['eta']))
    cols = ['near_1.4', 'near_1', 'near_2', 'near_4', 'far_1', 'far_2', 'far_4', 'gap_min_2', 'gap_min_4', 'gap_min_6', 'gap_min_10',
            'gap_axis_4', 'sol_mat_4', 'sol_mat_10', 'sol_old_4', 'sol_old_10', 'wrap_mat_4', 'wrap_old_4', 'span_y_4', 'thick_x_4',
            'area_ratio_4', 'old_raft_4', 'creep_near_4.6_10', 'creep_far_4.6_10', 'iface_width_cells', 'impulse_rel_err', 'max_c']
    print('| 情形 | η_J | ' + ' | '.join(cols) + ' |')
    print('|---|---|' + '---|' * len(cols))
    for r in T2:
        kw = r['kw']
        lab = ('移动' if kw['travel'] > 0 else '定点') + f' s0={kw["s0"]:g}'
        print(f'| {lab} | {eta_s(kw["eta"])} | ' + ' | '.join(f(r['o'].get(c), 3) for c in cols) + ' |')

T3 = [r for r in rows if r['kind'] == 'T3']
if T3:
    print('\n## T3  (N=%d)' % N)
    T3.sort(key=lambda r: (r['kw']['dwell'], r['kw']['eta']))
    cols = ['notch_req', 'notch_area_mm2', 'entry_bulge_max', 'exit_bulge_max', 'entry_bdev_min', 'line_def_imm', 'line_def_final',
            'line_vis_width_final', 'two_rd', 'stem_width_entry', 'stem_width_exit', 'disc_area_change_cm2', 'iface_width_cells', 'impulse_rel_err']
    print('| 驻留 | η_J | ' + ' | '.join(cols) + ' | V: Δx(|y|=0.005/0.01/0.02/0.04 D) 过后 0.1 s / 切末 / 切后 3 s | 半宽 (切后 3 s) |')
    print('|---|---|' + '---|' * len(cols) + '---|---|')
    for r in T3:
        o = r['o']; vs = o['vshape']
        vv = ' / '.join(','.join(f(vs[l][y], 2) for y in ('0.005', '0.01', '0.02', '0.04')) for l in ('pass+0.1', 'cut_end', 'cut+3') if l in vs)
        print(f'| {r["kw"]["dwell"]:g} | {eta_s(r["kw"]["eta"])} | ' + ' | '.join(f(o.get(c), 3) for c in cols) + f' | {vv} | {f(vs["cut+3"]["half_width"], 3)} |')

T4 = [r for r in rows if r['kind'] == 'T4']
if T4:
    print('\n## T4  (N=%d)' % N)
    T4.sort(key=lambda r: (r['kw']['base'], r['kw']['eta']))
    cols = ['n_stripes', 'pitch', 'def_mean', 'n_enclosed', 'enclosed_area_mm2', 'n_stripes_final', 'pitch_final', 'def_mean_final',
            'n_enclosed_final', 'survival', 'max_abs_dc_after', 'mixed_frac_of_white', 'A_vis_end', 'iface_width_cells', 'impulse_rel_err']
    print('| 底 | η_J | ' + ' | '.join(cols) + ' | stripe_x |')
    print('|---|---|' + '---|' * len(cols) + '---|')
    for r in T4:
        o = r['o']
        print(f'| {"白底" if r["kw"]["base"] else "新画布"} | {eta_s(r["kw"]["eta"])} | ' + ' | '.join(f(o.get(c), 3) for c in cols)
              + ' | ' + ','.join(f(x, 3) for x in o.get('stripe_x', [])) + ' |')

print('\n## T5 ledgers over all runs (N=%d)' % N)
agg = {}
for r in rows:
    o = r['o']
    for k in ('foam_rel_err', 'surf_err_cm2', 'impulse_rel_err', 'max_divres', 'clip_cm2', 'redist_cm2', 'max_c', 'iface_width_cells'):
        if o.get(k) is not None and o.get(k) == o.get(k):
            agg.setdefault(k, []).append(o[k])
    agg.setdefault('min_c', []).append(o.get('min_c', 0.0))
for k, v in agg.items():
    print(f'{k}: max {max(v):.3e}  min {min(v):.3e}  n {len(v)}')
wall = {}
for r in rows:
    wall.setdefault(r['kind'], []).append((r['kw'].get('eta'), r['o'].get('wall_per_sim_s'), r['o'].get('steps')))
for k, v in wall.items():
    print(k, 'wall/sim-s by eta:', sorted(set((e, round(w, 2)) for e, w, s in v))[:12])
