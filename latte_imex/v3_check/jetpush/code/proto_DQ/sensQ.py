"""Finite-difference (+-20 %) sensitivities of T1-T4 observables for v3-min-2 (jetU = 0) and variant Q (jetU = 1).
usage: python3 -I sensQ.py VARIANT(v2|q) [NPROC]   -> sens/sensQ_<variant>.json
Cases: nominal; Gamma, tau_w, delta_e (-> r_e,ref), Pi_coh x 1.2 / 0.8; nuisances alpha_j 12/5 deg, alpha_F x1.2/0.8,
Q scale x1.2/0.8 (Q and Gamma together), c_t x1.5/0.5."""
import sys, os, json, math, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from multiprocessing import Pool

NOM = dict(Gamma=0.0395, tau_w=1.4, delta_e=1.0e-3, Pi_coh=300.0, alpha_F=0.009)


def hole_ins(c, g, r0, x0=0.0):
    """inscribed radius of (brown U jet column), centre within r0 of x_j (as spec_v2/sens/sens2.py)."""
    import px_tests as T
    best = 0.0
    for dx in np.linspace(-r0, r0, 9):
        for dy in np.linspace(-r0, r0, 9):
            if dx * dx + dy * dy > r0 * r0: continue
            th, r, v = T.rays(c, g, x0 + dx, dy, nang=72, rmax=0.02, dr=5e-5)
            colmask = np.hypot(x0 + dx + r[None, :] * np.cos(th)[:, None] - x0, dy + r[None, :] * np.sin(th)[:, None]) < r0
            hit = (v >= 0.5) & ~colmask
            rf = np.where(hit.any(1), r[np.argmax(hit, 1)], 0.02)
            best = max(best, float(rf.min()))
    return best


def run_case(arg):
    name, p, mods, jetU = arg
    import px_tests as T
    import dq_tests as Q
    from px_model import v_from_zs, V2, PXSim, jet_derived
    T.Q_LOW = 15e-6; T.Q_THIN = 3e-6; T.V_LOW = v_from_zs(0.015); T.V_THIN = 1.2
    if 'Qs' in mods: T.Q_LOW = 15e-6 * mods['Qs']; T.Q_THIN = 3e-6 * mods['Qs']
    q = dict(V2); q.update(p); q['jetU'] = jetU
    t0 = time.time(); g = T.grid(128); out = {}
    o1, e1 = T.T1(params=q, keep=True)
    for k in ('t_first_vis', 'dAdt_vis', 'open_deg_3', 'specks_n_3', 'specks_mm2_3', 'far_disp_0.30_before_white',
              'far_disp_0.30', 'hole_r_3'):
        out[k] = o1[k]
    out['hole_growth_ray'] = o1['hole_r_6'] - o1['hole_r_3']
    sim = PXSim(grid=g, params=q); J = jet_derived(T.Q_LOW, T.V_LOW, sim.P)
    inp = T.jet_inp(lambda t: 0.0, lambda t: 0.0, lambda t: T.Q_LOW, T.V_LOW)
    tf = o1['t_first_raw']; ts = [round(tf + d, 3) for d in (1.0, 1.5, 2.0, 2.5)]
    vals = []
    sim.run(inp, ts[-1] + 1e-6, lambda t: 0.0, callback=lambda s, t: vals.append(hole_ins(s.c, g, J['r0'])), cb_times=ts)
    out['hole_ins'] = float(np.mean(vals)) / 0.08
    o2, _ = Q.T2Q(params=q, s0=0.15, travel=0.0)
    out['T2h_gap_2'] = o2['gap_min_2']; out['T2h_gap_6'] = o2['gap_min_6']; out['T2h_sol_4'] = o2['mat_solidity_4']
    out['T2h_near_prewhite'] = o2['near_disp_prewhite']; out['T2h_near_4'] = o2['near_disp_4']
    o2m, _ = Q.T2Q(params=q, s0=0.15, travel=0.10)
    out['T2m_sol_4'] = o2m['mat_solidity_4']; out['T2m_matarea_4'] = o2m['mat_area_ratio_4']
    out['T2m_thick_6'] = o2m['cr_axial_thick_6']; out['T2h_thick_6'] = o2['cr_axial_thick_6']
    out['T2m_gap_4'] = o2m['gap_min_4']; out['T2m_near_prewhite'] = o2m['near_disp_prewhite']
    o3, _ = Q.T3Q(params=q, gate='lift', dwell=0.0)
    out['line_def_final'] = o3['line_def_final']; out['two_rd'] = o3['two_rd']; out['T3_notch'] = o3['notch_req']
    out['stem'] = 0.5 * (o3['stem_width_entry'] + o3['stem_width_exit'])
    o3d, _ = Q.T3Q(params=q, gate='lift', dwell=0.7)
    out['T3d_notch'] = o3d['notch_req']; out['T3d_bulge'] = o3d['entry_bulge_10_15']; out['T3d_line'] = o3d['line_def_final']
    out['T3d_V'] = o3d['V_amp']
    o4, _ = Q.T4Q(params=q, base=False)
    out['T4_lines'] = o4['n_lines_median']; out['T4_enclosed'] = o4['end_n_enclosed']
    out['T4_line_axis'] = o4['end_y+0.00_n']
    out['_wall'] = time.time() - t0
    return name, out


def P(**kw):
    q = dict(NOM); q.update(kw); return q


if __name__ == '__main__':
    var = sys.argv[1]; nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    jetU = {'v2': 0.0, 'q': 1.0}[var]
    cases = [('nominal', P(), {}, jetU)]
    for k in ('Gamma', 'tau_w', 'delta_e', 'Pi_coh', 'alpha_F'):
        for s, f in (('+', 1.2), ('-', 0.8)):
            cases.append((k + s, P(**{k: NOM[k] * f}), {}, jetU))
    for s in ('+', '-'):
        f = 1.2 if s == '+' else 0.8
        cases.append(('Qs' + s, P(Gamma=NOM['Gamma'] * f), {'Qs': f}, jetU))
        cases.append(('alpha_j' + s, P(alpha_j=12.0 if s == '+' else 5.0), {}, jetU))
        cases.append(('c_t' + s, P(c_t=2.0 * (1.5 if s == '+' else 0.5)), {}, jetU))
    t0 = time.time()
    res = {}
    with Pool(nproc) as pool:
        for name, out in pool.imap_unordered(run_case, cases):
            res[name] = out
            print(name, json.dumps({k: (float('%.6g' % v) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    print('done', round(time.time() - t0, 1), 's')
    json.dump(res, open(os.path.join(HERE, f'sensQ_{var}.json'), 'w'), indent=1)
