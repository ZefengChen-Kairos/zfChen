"""v3-min-2 sensitivity of the objective observables on the final geometry (V2 defaults), incl. nuisances.
g1 first white, g2 dA/dt, g3 hole (inscribed radius of brown U column, mean over t_f+1.0..2.5 s of a 4.5 s pour),
g4 cut-line integrated deficit, g5a bite growth after stop (brown area within 0.06 D, 3 s pour), g5b pre-white near-edge pull."""
import sys, os, json, math, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from multiprocessing import Pool
NOM = dict(Gamma=0.0395, tau_w=1.4, delta_e=1.0e-3, Pi_coh=300.0, alpha_F=0.009)

def bite(c, g, rad):
    r = np.hypot(g.X, g.Y); m = (r < rad) & g.inside
    return float(((1 - np.clip(c, 0, 1)) * m).sum() * g.cellA * 1e4)

def hole_ins(c, g, r0, x0=0.0):
    """sub-cell inscribed radius of (brown U jet column): max over centres within r0 of x_j of the min first-white
    distance along 72 rays (bilinear c, dr = 0.05 mm)."""
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
    name, p, mods = arg
    import px_tests as T
    from px_model import v_from_zs, V2, PXSim, jet_derived
    T.Q_LOW = 15e-6; T.Q_THIN = 3e-6; T.V_LOW = v_from_zs(0.015); T.V_THIN = 1.2
    if 'Qs' in mods: T.Q_LOW = 15e-6 * mods['Qs']; T.Q_THIN = 3e-6 * mods['Qs']
    if 'zs' in mods: T.V_LOW = v_from_zs(0.015 + mods['zs'])
    if 'vthin' in mods: T.V_THIN = mods['vthin']
    q = dict(V2); q.update(p)
    t0 = time.time(); g = T.grid(128); out = {}
    # T1 (3 s pour, stop): first white, dA/dt, bite growth
    o1, e1 = T.T1(params=q, keep=True)
    out['t_first_vis'] = o1['t_first_vis']; out['dAdt_vis'] = o1['dAdt_vis']
    s3, s6 = e1['snaps'][3.0], e1['snaps'][6.0]
    out['bite_0.06'] = bite(s6, g, 0.06 * 0.08) - bite(s3, g, 0.06 * 0.08)
    out['bite_0.10'] = bite(s6, g, 0.10 * 0.08) - bite(s3, g, 0.10 * 0.08)
    out['hole_growth_ray'] = o1['hole_r_6'] - o1['hole_r_3']
    out['open_3'] = o1['open_deg_3']
    # hole: 4.5 s stationary pour, inscribed radius averaged over t_f+1.0 .. t_f+2.5
    sim = PXSim(grid=g, params=q); J = jet_derived(T.Q_LOW, T.V_LOW, sim.P)
    inp = T.jet_inp(lambda t: 0.0, lambda t: 0.0, lambda t: T.Q_LOW, T.V_LOW)
    tf = o1['t_first_raw']; ts = [round(tf + d, 3) for d in (1.0, 1.5, 2.0, 2.5)]
    vals = []; rays = []
    def cb(s, t):
        vals.append(hole_ins(s.c, g, J['r0'])); rays.append(T.ring_stats(s.c, g, 0.0, 0.0, rmax=0.02)['r_hole_med'])
    sim.run(inp, ts[-1] + 1e-6, lambda t: 0.0, callback=cb, cb_times=ts)
    out['hole_ins'] = float(np.mean(vals)) / 0.08; out['hole_ray'] = float(np.mean(rays)) / 0.08
    o2, _ = T.T2(params=q, s0=0.15, travel=0.0); out['near_disp_1'] = o2['near_disp_1']
    o2b, _ = T.T2(params=q, s0=0.25, travel=0.0); out['gap_6'] = o2b['gap_min_6']
    o3, _ = T.T3(params=q, gate='lift'); out['line_def_final'] = o3['line_def_final']; out['two_rd'] = o3['two_rd']
    out['stem'] = 0.5 * (o3['stem_width_entry'] + o3['stem_width_exit'])
    out['_wall'] = time.time() - t0
    return name, out

def P(**kw):
    q = dict(NOM); q.update(kw); return q

if __name__ == '__main__':
    cases = [('nominal', P(), {})]
    for k in NOM:
        for s, f in (('+', 1.2), ('-', 0.8)):
            cases.append((k + s, P(**{k: NOM[k] * f}), {}))
    for s in ('+', '-'):
        f = 1.2 if s == '+' else 0.8
        cases.append(('Qs' + s, P(Gamma=NOM['Gamma'] * f), {'Qs': f}))
        cases.append(('delta_c' + s, P(delta_c=0.5e-3 * (1.5 if s == '+' else 0.5)), {}))
        cases.append(('c_t' + s, P(c_t=2.0 * (1.5 if s == '+' else 0.5)), {}))
        cases.append(('zs' + s, P(), {'zs': 0.005 if s == '+' else -0.005}))
        cases.append(('vthin' + s, P(), {'vthin': 1.30 if s == '+' else 1.14}))
        cases.append(('alpha_j' + s, P(alpha_j=12.0 if s == '+' else 5.0), {}))
        cases.append(('p_sub' + s, P(p_sub=20.0 if s == '+' else 5.0), {}))
    t0 = time.time()
    with Pool(4) as pool:
        res = dict(pool.map(run_case, cases))
    print('done', round(time.time() - t0, 1), 's')
    json.dump(res, open(os.path.join(HERE, 'sens2.json'), 'w'), indent=1)
    for k, v in res.items():
        print(k, json.dumps({kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}))
