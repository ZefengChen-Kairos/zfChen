"""v3-min-2 tests on top of px_tests: T1L (long stationary pour), T1b (move onto white), hole metrics."""
import math, json, sys, time
import numpy as np
from scipy import ndimage
import px_tests as T
from px_model import PXSim, D, R, jet_derived, FIXED, NOMINAL, V2, white_area


def P2(**kw):
    p = dict(V2); p.update(kw); return p


def hole_metrics(c, g, x0, y0, r0):
    rs = T.ring_stats(c, g, x0, y0, rmax=0.02)
    rf = rs['rfirst']
    vis = rf - r0                                     # first white beyond the (occluding) jet column
    brown = (c < 0.5) & g.inside
    edt = ndimage.distance_transform_edt(brown) * g.h
    near = np.hypot(g.X - x0, g.Y - y0) <= r0
    r_ins = float(edt[near].max()) if near.any() else float('nan')
    ann = (np.hypot(g.X - x0, g.Y - y0) < r0 + 0.004) & (np.hypot(g.X - x0, g.Y - y0) >= r0) & g.inside
    return dict(r_med=rs['r_hole_med'] / D, r_vis_med=float(np.nanmedian(vis)) / D if np.isfinite(rf).any() else float('nan'),
                r_ins=r_ins / D, open=rs['open_deg'], c0=rs['c_centre'],
                brown_vis_cm2=float(((1 - np.clip(c, 0, 1)) * ann).sum() * g.cellA * 1e4))


def T1L(params=None, N=128, t_pour=6.5, t_end=9.5, x0=0.0, dt_s=0.25):
    g = T.grid(N)
    sim = PXSim(grid=g, params=params)
    inp = T.jet_inp(lambda t: x0, lambda t: 0.0, lambda t: T.Q_LOW if t < t_pour else 0.0, T.V_LOW)
    J = jet_derived(T.Q_LOW, T.V_LOW, sim.P)
    ts = sorted(set([round(x, 3) for x in np.arange(1.0, t_end + 1e-9, dt_s)] + [round(x, 3) for x in np.arange(1.3, 1.7, 0.01)]))
    rows = {}
    prev = {}
    def cb(s, t):
        m = hole_metrics(s.c, g, x0, 0.0, J['r0'])
        L = s.L
        cur = dict(sink=L['sink_all'], refl=L['refloat'], lost=L.get('lost_under_raft', 0.0), emit=L['emitted'],
                   pressed=L.get('pressed', 0.0), sup=L['supply_area'])
        d = {k: cur[k] - prev.get(k, 0.0) for k in cur}
        prev.update(cur)
        m['A'] = float(s.c.sum() * g.cellA * 1e4)
        m['raft_share_sink'] = d['refl'] / max(d['sink'], 1e-30)
        m['lost_share_emit'] = d['lost'] / max(d['lost'] + d['emit'], 1e-30)
        m['pressed_per_supply'] = d['pressed'] / max(d['sup'], 1e-30)
        rows[round(t, 3)] = m
    sim.run(inp, t_end, lambda t: 0.0, breakpoints=[t_pour], callback=cb, cb_times=ts)
    tt = np.array(sorted(rows)); A = np.array([rows[t]['A'] for t in tt])
    i = np.argmax(A >= 1e-3); tf = float(tt[i - 1] + (1e-3 - A[i - 1]) / (A[i] - A[i - 1]) * (tt[i] - tt[i - 1]))
    out = dict(t_first=tf, rows=rows, jet={k: float(v) for k, v in J.items()})
    for dtf in (0.6, 1.0, 1.5, 2.5, 3.5, 4.5):
        tq = min(tt, key=lambda x: abs(x - (tf + dtf)))
        out[f'at+{dtf}'] = rows[tq]
    out['clip_cm2'] = abs(sim.L['clip_c']) * 1e4
    out['redist_cm2'] = sim.L.get('redist', 0.0) * 1e4
    out['max_c'] = sim.stats['max_c']
    L = sim.ledgers(); out['foam_rel_err'] = abs(L['foam_rel']); out['surf_err_cm2'] = abs(L['surf_err']) * 1e4
    return out


def T1b(params=None, N=128, t_move=2.0, V=0.1, t_mv=1.0, t_end=4.0, ex=1.0):
    g = T.grid(N)
    sim = PXSim(grid=g, params=params)
    xf = lambda t: ex * V * D * min(max(t - t_move, 0.0), t_mv)
    inp = T.jet_inp(xf, lambda t: 0.0, lambda t: T.Q_LOW if t < t_end else 0.0, T.V_LOW)
    J = jet_derived(T.Q_LOW, T.V_LOW, sim.P)
    snaps = {}
    def cb(s, t):
        snaps[round(t, 3)] = s.c.copy()
    sim.run(inp, t_end, lambda t: 0.0, breakpoints=[t_move, t_move + t_mv], callback=cb,
            cb_times=[t_move, t_move + 0.5, t_move + t_mv, t_end])
    out = {}
    for t in sorted(snaps):
        m = hole_metrics(snaps[t], g, xf(t), 0.0, J['r0'])
        c = snaps[t]
        r = np.hypot(g.X - xf(t), g.Y)
        core = (r < J['r0'] + 0.003) & g.inside
        m['brown_core_cm2'] = float(((1 - np.clip(c, 0, 1)) * core).sum() * g.cellA * 1e4)
        m['brown_core_vis_cm2'] = float(((1 - np.clip(c, 0, 1)) * core * (r >= J['r0'])).sum() * g.cellA * 1e4)
        h = T.hole_and_specks(c, g, xf(t), 0.0)
        m['specks_n'] = h['n_specks']; m['specks_mm2'] = h['specks_mm2']
        out[t] = m
    out['clip_cm2'] = abs(sim.L['clip_c']) * 1e4
    return out
