"""usage: python3 -I tables.py out/scan_A128.jsonl [more.jsonl]  -> prints the T1-T5 tables of test_Ccompres.md"""
import sys, json, math
ORDER = ['v2', 'q', 'sv20', 'sv4', 'sv2', 'sv1', 'sv05', 'sv20n', 'sv2n', 'sv1n', 'sv05n', 'cd1e3', 'cd1e4', 'cd3e5', 'cd1e4n', 'cp', 'cpn']
R = {}
for fn in sys.argv[1:]:
    for l in open(fn):
        r = json.loads(l); R[(r['test'], r['var'], r['arg'], r['N'])] = r['obs']


def f(x, d=3):
    if x is None: return '—'
    if isinstance(x, bool): return str(x)
    if isinstance(x, int): return str(x)
    if isinstance(x, float):
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5): return '%.1e' % x
        return ('%.' + str(d) + 'f') % x
    return str(x)


def rows(test, arg, keys, N, d=3):
    vs = sorted({k[1] for k in R if k[0] == test and k[2] == arg and k[3] == N}, key=lambda v: ORDER.index(v) if v in ORDER else 99)
    print(f'\n## {test} {arg or ""} N={N}')
    print('| var | ' + ' | '.join(keys) + ' |')
    for v in vs:
        o = R[(test, v, arg, N)]
        print(f'| {v} | ' + ' | '.join(f(o.get(k), d) for k in keys) + ' |')


Ns = sorted({k[3] for k in R})
for N in Ns:
    rows('T1', None, ['t_first_vis', 'dAdt_raft', 'dAdt_vis', 'hole_r_2', 'hole_r_3', 'hole_r_6', 'hole_ins', 'open_deg_2', 'open_deg_3', 'open_deg_6',
                      'specks_n_3', 'specks_mm2_3', 'specks_mm2_6', 'disp_0.15_pre_white', 'disp_0.30_pre_white', 'disp_0.30_6', 'disp_0.45_6',
                      'umax_after_tr_mm_s', 'umax_end_mm_s', 'max_dc_tr_end', 'drift_0.30_tr_end', 'drift_0.15_tr_end', 'dA_after_stop_tr_end',
                      'ring_rin_tw+0.1_mm', 'ring_open_deg_tw+0.1'], N, 4)
    rows('T1', None, ['A_abs_cm2', 'src_int_cm2', 'ledger1_rel', 'foam_rel_err', 'surf_err_cm2', 'max_divres', 'max_c', 'min_c', 'c_max_after', 'c_min_after',
                      'redist_cm2', 'clip_cm2', 'iface_width_cells_3', 'iface_width_cells', 'steps', 'wall_per_sim_s', 'ms_per_step', 'ms_poisson',
                      'cg_its_per_solve', 'n_fact', 'active_its_max', 'active_mismatch'], N, 3)
    for test in ('T2h', 'T2m'):
        for s0 in ('0.1', '0.10', '0.15', '0.25'):
            if not any(k[0] == test and k[2] == s0 and k[3] == N for k in R): continue
            rows(test, s0, ['near_1', 'near_1.4', 'near_2', 'near_4', 'near_6', 'far_1', 'far_1.4', 'far_2', 'far_4', 'loop_ratio_1', 'loop_ratio_1.4',
                            'gap_min_1.4', 'gap_min_2', 'gap_min_4', 'gap_min_6', 'gap_min_10', 'gap_axis_2', 'gap_axis_4', 'gap_axis_6',
                            'contact_4', 'sol_mat_4', 'sol_mat_6', 'sol_mat_10', 'sol_old_4', 'wrap_mat_4', 'wrap_old_4', 'thick_x_4', 'span_y_4',
                            'thick_x_6', 'span_y_6', 'area_ratio_4', 'old_raft_4', 'creep_near_4.6_10', 'creep_far_4.6_10', 'max_divres', 'ledger1_rel',
                            'wall_per_sim_s'], N, 3)
    for test in ('T3', 'T3d'):
        rows(test, None, ['two_rd', 'line_def_imm', 'line_def_final', 'line_vis_width_final', 'notch_req', 'notch_area_mm2', 'entry_bulge_max',
                          'entry_bdev_min', 'exit_bulge_max', 'exit_bdev_min', 'stem_width_entry', 'stem_width_exit', 'disc_area_change_cm2', 'core_cm2',
                          'ledger1_rel', 'wall_per_sim_s'], N, 4)
        vs = sorted({k[1] for k in R if k[0] == test and k[3] == N}, key=lambda v: ORDER.index(v) if v in ORDER else 99)
        print(f'\nV shape {test} N={N} (forward drag of x=0 material line, D; at |y| = 0.005/0.01/0.02/0.04/0.08 D, half width)')
        for v in vs:
            vsd = R[(test, v, None, N)].get('vshape', {})
            for lab in ('pass+0.1', 'cut_end', 'cut+3'):
                if lab in vsd:
                    p = vsd[lab]
                    print(f'  {v:6s} {lab:9s} ' + ' '.join(f(p.get(k), 4) for k in ('0.005', '0.01', '0.02', '0.04', '0.08', 'half_width')))
    for test in ('T4f', 'T4b'):
        rows(test, None, ['n_stripes', 'pitch', 'def_mean', 'n_stripes_final', 'survival', 'n_lines_median', 'n_lines_max', 'n_lines_median_fin',
                          'end_y+0.00_n', 'end_y+0.02_n', 'end_y-0.02_n', 'end_y+0.04_n', 'end_y-0.04_n', 'end_y+0.02_pitch', 'end_y+0.02_w',
                          'fin_y+0.02_n', 'n_enclosed', 'enclosed_area_mm2', 'mixed_frac_of_white', 'max_abs_dc_after', 'A_vis_end', 'ledger1_rel',
                          'iface_width_cells', 'wall_per_sim_s', 'ms_per_step', 'steps'], N, 3)
