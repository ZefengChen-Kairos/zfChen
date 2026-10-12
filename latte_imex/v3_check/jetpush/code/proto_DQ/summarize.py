"""Print the key numbers of all logs (for test_Qquasist.md).  usage: python3 -I summarize.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(HERE, 'logs', name); out = {}
    if os.path.exists(p):
        for l in open(p):
            try:
                d = json.loads(l)
            except Exception:
                continue
            out[(d['test'], d['var'])] = d['obs']
    return out


def f(x, n=4):
    if x is None: return 'nan'
    if isinstance(x, bool): return str(x)
    if isinstance(x, (int,)): return str(x)
    if isinstance(x, float):
        if x != x: return 'nan'
        return f'{x:.{n}g}'
    return str(x)


def T1rows(L, tag):
    ks = ['t_first_raw', 't_first_vis', 'ring_open_deg_tw+0.1', 'open_deg_tw+0.6', 'open_deg_3', 'hole_r_2', 'hole_r_3', 'hole_r_6',
          'hole_rmin_3', 'c_centre_3', 'dAdt_raft', 'dAdt_vis', 'far_disp_0.30_before_white', 'far_disp_0.30', 'far_disp_0.45',
          'specks_n_2', 'specks_mm2_2', 'specks_n_3', 'specks_mm2_3', 'specks_n_6', 'specks_mm2_6', 'grey_cells_3',
          'umax_after_stop_tr', 'dA_after_stop_tr', 'refill_raft', 'iface_width_cells', 'up_white_mm2_tw+0.6', 'down_white_mm2_tw+0.6',
          'wall_per_sim_s', 'steps', 'max_courant', 'redist_cm2', 'max_c', 'min_c']
    for v in [k[1] for k in L if k[0] == 'T1']:
        o = L[('T1', v)]
        print(f'[{tag}] T1 {v}: ' + ', '.join(f'{k}={f(o.get(k))}' for k in ks))


def T2rows(L, tag):
    ks = ['near_disp_prewhite', 'near_disp_1', 'near_disp_2', 'near_disp_4', 'far_disp_1', 'far_disp_2', 'far_disp_4',
          'gap_min_2', 'gap_min_4', 'gap_min_6', 'gap_axis_2', 'gap_axis_4', 'gap_axis_6', 'contact_4', 'mat_solidity_4', 'old_solidity_4',
          'mat_area_ratio_4', 'cr_axial_thick_6', 'cr_transverse_span_6', 'cr_span_from_jet_deg_4', 'cr_bay_cos_jet_4',
          'cr_new_in_hull_frac_4', 'cr_crescent_4', 'gap_pass', 'wall_per_sim_s']
    for (t, v) in sorted(k for k in L if k[0].startswith('T2')):
        o = L[(t, v)]
        print(f'[{tag}] {t} {v}: ' + ', '.join(f'{k}={f(o.get(k), 3)}' for k in ks))
        gs = o.get('gap_min_series', {})
        print('      gap series: ' + ' '.join(f'{k}:{f(x, 2)}' for k, x in sorted(gs.items(), key=lambda a: float(a[0]))))


def T3rows(L, tag):
    ks = ['line_def_imm', 'line_def_post_tr', 'line_def_final', 'line_vis_width_final', 'heal_ratio', 'two_rd', 'notch_req',
          'notch_area_mm2', 'entry_bulge_10_15', 'entry_bulge_20_25', 'entry_dev_5', 'entry_dev_10', 'entry_bulge_max', 'V_amp',
          'lat_out_near', 'stem_width_entry', 'stem_width_exit', 'white_on_brown_cm2', 'wall_per_sim_s', 'min_c', 'max_c']
    for (t, v) in sorted(k for k in L if k[0] in ('T3', 'T3d')):
        o = L[(t, v)]
        print(f'[{tag}] {t} {v}: ' + ', '.join(f'{k}={f(o.get(k), 3)}' for k in ks))
        if 'entry_bdev_profile' in o:
            pr = o['entry_bdev_profile']
            print('      entry profile (deg: dev D) ' + ' '.join(f'{a - 40}:{f(pr[a], 2)}' for a in range(0, 81, 4)))


def T4rows(L, tag):
    for (t, v) in sorted(k for k in L if k[0] in ('T4f', 'T4b')):
        o = L[(t, v)]
        s = []
        for ph in ('end', 'fin'):
            for y in (0.0, 0.02, -0.02, 0.04, -0.04):
                k = f'{ph}_y{y:+.2f}'
                s.append(f'{k}: n={o.get(k + "_n")} p={f(o.get(k + "_pitch"), 3)} w={f(o.get(k + "_w"), 3)}')
            s.append(f'{ph}_enclosed={o.get(ph + "_n_enclosed")}/{f(o.get(ph + "_enclosed_mm2"), 4)}mm2')
        print(f'[{tag}] {t} {v}: ' + '; '.join(s) + f'; mixed={f(o.get("mixed_frac_of_white"), 3)} dc_after={o.get("max_abs_dc_after")} wall/s={f(o.get("wall_per_sim_s"), 3)}')


def T1L(L, tag):
    for (t, v) in sorted(k for k in L if k[0] == 'T1L'):
        o = L[(t, v)]
        print(f'[{tag}] T1L {v}: t_first={f(o["t_first"])} ' + ' '.join(
            f'{k}: r_med={f(o[k]["r_med"], 3)} r_ins={f(o[k]["r_ins"], 3)} open={o[k]["open"]}' for k in ('at+1.5', 'at+2.5', 'at+4.5')))


if __name__ == '__main__':
    for name in sys.argv[1:] or ['A_128.jsonl', 'A_192.jsonl', 'lam_128.jsonl', 'aj_128.jsonl']:
        L = load(name)
        if not L: continue
        print(f'================= {name}')
        T1rows(L, name); T1L(L, name); T2rows(L, name); T3rows(L, name); T4rows(L, name)
