"""Variant-C identifiability: +-20 % central differences of the five fitted parameters (Gamma, tau_w, r_e,ref via
delta_e as in v3-min-2 sens2.py, Pi_coh, l_s) on T1-T4 observables, at nominal l_s (C-V + L5Q).
usage: OMP_NUM_THREADS=1 python3 -I sens/sens_C.py ELL_NOM[,ELL_NOM2] [NPROC]  -> sens/sensC_<ells>.json"""
import sys, os, json, math, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from multiprocessing import Pool
NOM = dict(Gamma=0.0395, tau_w=1.4, delta_e=1.0e-3, Pi_coh=300.0)


def run_case(arg):
    name, p = arg
    import px_tests as T
    import DC_tests as DT
    from px_model import V2, PXSim, jet_derived, Grid
    q = dict(V2); q.update(dict(jetU=1.0, canvas='screen')); q.update(p)
    t0 = time.time(); out = {}
    # T1 (3 s pour, stop, observe to 13 s): g1, g2, opening angle, specks, far field, stop-flow
    o1, _ = DT.T1x(params=q, t_end=13.0)
    for k in ('t_first_vis', 'dAdt_vis', 'dAdt_raft', 'open_deg_3', 'hole_r_3', 'specks_mm2_3', 'umax_after_tr_mm_s',
              'disp_0.30_pre_white', 'disp_0.15_pre_white', 'disp_0.30_6', 'disp_0.45_6', 'drift_0.30_tr_end', 'max_dc_tr_end'):
        out[k] = o1[k]
    # g3: v3-min-2 definition (sens2.py): 4.5 s stationary pour, inscribed radius averaged over t_f + 1.0 .. 2.5 s
    g = Grid(128)
    sim = PXSim(grid=g, params=q); J = jet_derived(T.Q_LOW, T.V_LOW, sim.P)
    inp = T.jet_inp(lambda t: 0.0, lambda t: 0.0, lambda t: T.Q_LOW, T.V_LOW)
    tf = o1['t_first_raw']; ts = [round(tf + d, 3) for d in (1.0, 1.5, 2.0, 2.5)]
    vals = []
    sim.run(inp, ts[-1] + 1e-6, lambda t: 0.0, callback=lambda s, t: vals.append(DT.hole_ins(s.c, g, J['r0'])), cb_times=ts)
    out['hole_ins'] = float(np.mean(vals)) / 0.08
    # T2 held s0 = 0.15 D: pre-white push, canvas loop (l_s anchor), gap, crescent; s0 = 0.25 D gap; moving s0 = 0.15 D
    o2, _ = DT.T2x(params=q, s0=0.15, travel=0.0, t_end=10.0)
    for k in ('near_1.4', 'far_1.4', 'far_over_near_1.4', 'loop_ratio_1', 'loop_ratio_1.4', 'near_4', 'far_4', 'gap_min_2',
              'gap_min_4', 'gap_min_6', 'sol_mat_4', 'wrap_mat_4', 'thick_x_4', 'span_y_4'):
        out['T2s15_' + k] = o2.get(k)
    o2b, _ = DT.T2x(params=q, s0=0.25, travel=0.0, t_end=6.0)
    for k in ('gap_min_4', 'gap_min_6', 'sol_mat_4', 'near_1.4', 'loop_ratio_1.4'):
        out['T2s25_' + k] = o2b.get(k)
    o2m, _ = DT.T2x(params=q, s0=0.15, travel=0.10, t_end=6.0)
    for k in ('sol_mat_4', 'area_ratio_4', 'near_1.4', 'gap_min_4', 'thick_x_4'):
        out['T2m15_' + k] = o2m.get(k)
    # T3 cut without dwell (g4, stem) and with a 0.7 s entry dwell (notch, bulges)
    o3, _ = DT.T3x(params=q, dwell=0.0)
    out['line_def_final'] = o3['line_def_final']; out['line_vis_width_final'] = o3['line_vis_width_final']
    out['two_rd'] = o3['two_rd']; out['stem'] = 0.5 * (o3['stem_width_entry'] + o3['stem_width_exit'])
    o3d, _ = DT.T3x(params=q, dwell=0.7)
    out['notch_req'] = o3d['notch_req']; out['entry_bulge'] = o3d['entry_bulge_max']
    out['line_def_final_dwell'] = o3d['line_def_final']
    # T4 wiggle on fresh canvas
    o4, _ = DT.T4x(params=q, base=False)
    for k in ('n_stripes', 'n_lines_median', 'n_enclosed', 'enclosed_area_mm2', 'mixed_frac_of_white', 'max_abs_dc_after'):
        out['T4_' + k] = o4[k]
    out['_wall'] = time.time() - t0
    return name, out


def P(ell, **kw):
    q = dict(NOM); q['ell_s'] = ell; q.update(kw); return q


if __name__ == '__main__':
    ells = [float(x) for x in sys.argv[1].split(',')]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cases = []
    for e in ells:
        cases.append((f'{e:g}|nominal', P(e)))
        for k in NOM:
            for s, f in (('+', 1.2), ('-', 0.8)):
                cases.append((f'{e:g}|{k}{s}', P(e, **{k: NOM[k] * f})))
        for s, f in (('+', 1.2), ('-', 0.8)):
            cases.append((f'{e:g}|ell_s{s}', P(e * f)))
    t0 = time.time()
    fn = os.path.join(HERE, 'sensC_%s.json' % '_'.join(f'{e:g}' for e in ells))
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    cases = [c for c in cases if c[0] not in res]
    with Pool(nproc, maxtasksperchild=1) as pool:
        for name, out in pool.imap_unordered(run_case, cases):
            res[name] = out
            print(name, round(out['_wall'], 1), flush=True)
            json.dump(res, open(fn, 'w'), indent=1)
    print('done', round(time.time() - t0, 1), 's')
