"""Mechanism tests T1-T5 for the PX prototype.  Each test returns (obs, extra):
obs   = dict of scalar observables (lengths in D, areas in cm^2, times in s)
extra = snapshots / time series for the figures.
"""
import math, time, json
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from px_model import (PXSim, Grid, NOMINAL, FIXED, D, R, v_from_zs, jet_derived, white_area,
                      sample_bilinear, interface_width_cells)

_GRID = {}


def grid(N=128):
    if N not in _GRID:
        _GRID[N] = Grid(N)
    return _GRID[N]


Q_LOW = 15e-6
ZS_LOW = 0.015
V_LOW = v_from_zs(ZS_LOW)
Q_THIN = 3e-6
V_THIN = 1.2


def rays(c, g, x0, y0, nang=72, rmax=0.015, dr=1e-4):
    th = np.arange(nang) * 2 * np.pi / nang
    r = np.arange(0, rmax, dr)
    X = x0 + r[None, :] * np.cos(th)[:, None]
    Y = y0 + r[None, :] * np.sin(th)[:, None]
    ok = X ** 2 + Y ** 2 < (R - g.h) ** 2
    v = sample_bilinear(c, g, X, Y) * ok
    return th, r, v


def ring_stats(c, g, x0, y0, rmax=0.015):
    th, r, v = rays(c, g, x0, y0, rmax=rmax)
    hit = (v >= 0.5)
    has = hit.any(1)
    rfirst = np.where(has, r[np.argmax(hit, 1)], np.nan)
    open_frac = 1 - has.mean()
    if (~has).any():
        a = th[~has]
        open_dir = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))
    else:
        open_dir = float('nan')
    c_centre = float(sample_bilinear(c, g, np.array([x0]), np.array([y0]))[0])
    return dict(r_hole_med=float(np.nanmedian(rfirst)) if has.any() else float('nan'),
                r_hole_min=float(np.nanmin(rfirst)) if has.any() else float('nan'),
                r_hole_max=float(np.nanmax(rfirst)) if has.any() else float('nan'),
                open_frac=float(open_frac), open_deg=float(open_frac * 360), open_dir=open_dir,
                c_centre=c_centre, rfirst=rfirst, th=th)


def hole_and_specks(c, g, x0, y0):
    brown = (c < 0.5) & g.inside
    lab, n = ndimage.label(brown)
    r = np.hypot(g.X, g.Y)
    rim = set(np.unique(lab[(r > R - 2 * g.h) & brown]).tolist())
    i = int(np.argmin(np.abs(g.x1 - x0))); j = int(np.argmin(np.abs(g.x1 - y0)))
    hole = lab[i, j]
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) if n else np.zeros(0)
    out = {}
    if hole > 0 and hole not in rim:
        A = sizes[hole - 1] * g.cellA
        out['hole_area_cm2'] = float(A * 1e4); out['hole_deq'] = float(2 * math.sqrt(A / math.pi) / D)
    else:
        out['hole_area_cm2'] = float('nan'); out['hole_deq'] = float('nan')
    sp = [k for k in range(1, n + 1) if k not in rim and k != hole]
    out['n_specks'] = len(sp)
    out['specks_mm2'] = float(sum(sizes[k - 1] for k in sp) * g.cellA * 1e6)
    out['grey_cells'] = int(((c > 0.05) & (c < 0.95) & g.inside).sum())
    return out


def sector_asym(c, g, x0, y0, rmax=0.010):
    th = np.arctan2(g.Y - y0, g.X - x0)
    r = np.hypot(g.X - x0, g.Y - y0)
    near = (r < rmax) & g.inside
    up = near & (np.abs(np.abs(th) - np.pi) < np.pi / 4)
    dn = near & (np.abs(th) < np.pi / 4)
    return dict(up=float((c * up).sum() * g.cellA * 1e6), down=float((c * dn).sum() * g.cellA * 1e6))


def jet_inp(xf, yf, Qf, v, ex=1.0, ey=0.0):
    def inp(t):
        return dict(x=xf(t), y=yf(t), Q=Qf(t), v=v, ex=ex, ey=ey)
    return inp


def finish(sim, obs):
    L = sim.ledgers()
    obs['foam_rel_err'] = abs(L['foam_rel'])
    obs['surf_err_cm2'] = abs(L['surf_err']) * 1e4
    obs['clip_cm2'] = abs(L['clip_c']) * 1e4
    obs['max_c'] = sim.stats['max_c']
    obs['min_c'] = sim.stats['min_c']
    obs['max_courant'] = float(sim.stats['max_courant'])
    obs['max_srcdt'] = float(sim.stats['max_srcdt'])
    obs['max_divres'] = float(sim.stats['max_divres'])
    obs['steps'] = sim.stats['steps']
    obs['wall_s'] = sim.stats['t_wall']
    obs['wall_per_sim_s'] = sim.stats['t_wall'] / sim.t
    obs['ms_per_step'] = 1e3 * sim.stats['t_wall'] / sim.stats['steps']
    obs['ms_poisson'] = 1e3 * sim.stats['t_poisson'] / sim.stats['steps']
    obs['ms_vof'] = 1e3 * sim.stats['t_vof'] / sim.stats['steps']
    obs['ms_src'] = 1e3 * sim.stats['t_src'] / sim.stats['steps']
    return obs


# ============================================================================================ T1
def T1(params=None, N=128, keep=False):
    g = grid(N)
    sim = PXSim(grid=g, params=params)
    t_stop, t_end = 3.0, 6.0
    inp = jet_inp(lambda t: 0.0, lambda t: 0.0, lambda t: Q_LOW if t < t_stop else 0.0, V_LOW)
    tk = lambda t: 0.0
    # far-field markers: 32 per ring at r = 0.30, 0.35, 0.40, 0.45 D from the jet
    radii = [0.30, 0.35, 0.40, 0.45]
    th = np.arange(32) * 2 * np.pi / 32
    mk = np.array([[rr * D * math.cos(a), rr * D * math.sin(a)] for rr in radii for a in th])
    sim.markers = mk.copy()
    ts = np.round(np.arange(0.05, t_end + 1e-9, 0.05), 4)
    ser = dict(t=[], A_r=[], A_vis=[], umax=[], foam=[], surf=[])
    snaps = {}
    tw = sim.P['tau_w']
    t_r1, t_r2 = round(tw + 0.1, 4), round(tw + 0.6, 4)
    snap_t = sorted(set([1.45, 1.5, 1.6, 2.0, 3.0, 3.6, 6.0, t_r1, t_r2]))
    first = dict(raw=None, vis=None)

    def cb(s, t):
        L = s.ledgers()
        ser['t'].append(t); ser['A_r'].append(L['A_r'] * 1e4); ser['A_vis'].append(white_area(s, 0.5) * 1e4)
        ser['umax'].append(float(max(np.abs(s.ux).max(), np.abs(s.uy).max())))
        ser['foam'].append(L['foam_rel']); ser['surf'].append(L['surf_err'])
        if any(abs(t - u) < 1e-6 for u in snap_t):
            snaps[round(t, 3)] = s.c.copy()
        if abs(t - 1.4) < 1e-6:
            mk14[:] = s.markers
    mk14 = mk.copy()
    # fine sampling around the onset to resolve the first-white time
    fine = np.round(np.arange(1.30, 1.70, 0.01), 4)
    alltimes = sorted(set(list(ts) + list(fine) + snap_t))
    sim.run(inp, t_end, tk, breakpoints=[t_stop], callback=cb, cb_times=alltimes)
    tt = np.array(ser['t']); Ar = np.array(ser['A_r']); Av = np.array(ser['A_vis'])
    # first white: raft area > 1e-3 cm^2 (raw onset) and visible c>=0.5 area >= 0.05 cm^2 (~13 cells)
    def tcross(A, thr):
        i = np.argmax(A >= thr)
        if A[i] < thr: return float('nan')
        if i == 0: return float(tt[0])
        return float(tt[i - 1] + (thr - A[i - 1]) / (A[i] - A[i - 1]) * (tt[i] - tt[i - 1]))
    obs = {}
    obs['t_first_raw'] = tcross(Ar, 1e-3)
    obs['t_first_vis'] = tcross(Av, 0.05)
    # ring at first white
    rs = ring_stats(snaps[t_r1], g, 0.0, 0.0)
    obs['ring_c_centre_tw+0.1'] = rs['c_centre']
    obs['ring_rin_tw+0.1_mm'] = rs['r_hole_med'] * 1e3
    obs['ring_rmin_tw+0.1_mm'] = rs['r_hole_min'] * 1e3
    obs['ring_rmax_tw+0.1_mm'] = rs['r_hole_max'] * 1e3
    obs['ring_open_deg_tw+0.1'] = rs['open_deg']
    obs['ring_open_dir_tw+0.1'] = rs['open_dir']
    a = sector_asym(snaps[t_r1], g, 0.0, 0.0)
    obs['up_white_mm2_tw+0.1'] = a['up']; obs['down_white_mm2_tw+0.1'] = a['down']
    a = sector_asym(snaps[t_r2], g, 0.0, 0.0)
    obs['up_white_mm2_tw+0.6'] = a['up']; obs['down_white_mm2_tw+0.6'] = a['down']
    rs = ring_stats(snaps[t_r2], g, 0.0, 0.0)
    obs['open_deg_tw+0.6'] = rs['open_deg']
    for tsn in (2.0, 3.0, 6.0):
        rs = ring_stats(snaps[tsn], g, 0.0, 0.0)
        obs[f'hole_r_{tsn:.0f}'] = rs['r_hole_med'] / D
        obs[f'hole_rmin_{tsn:.0f}'] = rs['r_hole_min'] / D
        obs[f'hole_rmax_{tsn:.0f}'] = rs['r_hole_max'] / D
        obs[f'open_deg_{tsn:.0f}'] = rs['open_deg']
        obs[f'open_dir_{tsn:.0f}'] = rs['open_dir']
        obs[f'c_centre_{tsn:.0f}'] = rs['c_centre']
    # growth rate, fit from t_first+0.5 to stop
    t_a = (obs['t_first_raw'] if np.isfinite(obs['t_first_raw']) else 1.4) + 0.5
    m = (tt >= t_a - 1e-9) & (tt <= t_stop + 1e-9)
    obs['dAdt_raft'] = float(np.polyfit(tt[m], Ar[m], 1)[0]) if m.sum() > 2 else float('nan')
    obs['dAdt_vis'] = float(np.polyfit(tt[m], Av[m], 1)[0]) if m.sum() > 2 else float('nan')
    iA3 = np.argmin(abs(tt - t_stop)); iA6 = len(tt) - 1
    obs['A_raft_3'] = float(Ar[iA3]); obs['A_vis_3'] = float(Av[iA3])
    obs['A_raft_6'] = float(Ar[iA6]); obs['A_vis_6'] = float(Av[iA6])
    obs['refill_raft'] = float(Ar[iA6] - Ar[iA3]); obs['refill_vis'] = float(Av[iA6] - Av[iA3])
    i2 = np.argmin(abs(tt - 2.0))
    obs['A_vis_2'] = float(Av[i2])
    obs['spot_req_2'] = math.sqrt(Av[i2] / math.pi) * 1e-2 / D
    obs['spot_req_3'] = math.sqrt(Av[iA3] / math.pi) * 1e-2 / D
    jr = jet_derived(Q_LOW, V_LOW, sim.P)
    t_fr = t_stop + jr['tr']
    obs['umax_after_stop_tr'] = float(max(u for t, u in zip(tt, ser['umax']) if t > t_fr + 0.02))
    obs['dA_after_stop_tr'] = float(Ar[iA6] - Ar[np.argmin(abs(tt - (t_fr + 0.05)))])
    # far field
    disp = np.hypot(*(sim.markers - mk).T)
    for k, rr in enumerate(radii):
        obs[f'far_disp_{rr:.2f}'] = float(disp[k * 32:(k + 1) * 32].max() / D)
    obs['far_disp_max_ge_0.3'] = float(disp.max() / D)
    d14 = np.hypot(*(mk14 - mk).T)
    obs['far_disp_0.30_before_white'] = float(d14[:32].max() / D)
    for tsn in (2.0, 3.0, 6.0):
        h = hole_and_specks(snaps[tsn], g, 0.0, 0.0)
        obs[f'hole_area_{tsn:.0f}_cm2'] = h['hole_area_cm2']
        obs[f'hole_deq_{tsn:.0f}'] = h['hole_deq']
        obs[f'specks_n_{tsn:.0f}'] = h['n_specks']
        obs[f'specks_mm2_{tsn:.0f}'] = h['specks_mm2']
        obs[f'grey_cells_{tsn:.0f}'] = h['grey_cells']
    obs['hole_growth_after_stop'] = obs['hole_deq_6'] - obs['hole_deq_3']
    wc, Lc = interface_width_cells(sim.c, g)
    obs['iface_width_cells'] = float(wc)
    obs['jet'] = {k: float(v) for k, v in jr.items()}
    finish(sim, obs)
    extra = dict(ser=ser, snaps=snaps, markers0=mk, markers=sim.markers.copy()) if keep else None
    return obs, extra


# ============================================================================================ T2
def _poly_axis_cross(P):
    x, y = P[:, 0], P[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    s = (y <= 0) != (y2 <= 0)
    xc = x[s] + (0 - y[s]) * (x2[s] - x[s]) / (y2[s] - y[s])
    return xc


def _poly_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def old_new_masks(sim):
    c, o = sim.c, sim.o
    white = c >= 0.5
    old = white & (o >= 0.5 * c)
    new = white & (o < 0.5 * c)
    return old, new


def gap_metrics(sim):
    g = sim.g
    old, new = old_new_masks(sim)
    out = {}
    if new.sum() == 0:
        out.update(gap_min=float('nan'), gap_axis=float('nan'), contact=0.0, new_area=0.0)
    else:
        dist = ndimage.distance_transform_edt(~new) * g.h
        out['gap_min'] = float((dist[old].min() - g.h) / D) if old.any() else float('nan')
        nb = np.zeros_like(new)
        nb[1:, :] |= new[:-1, :]; nb[:-1, :] |= new[1:, :]; nb[:, 1:] |= new[:, :-1]; nb[:, :-1] |= new[:, 1:]
        out['contact'] = float((old & nb).sum() * g.h / D)
        out['new_area'] = float(new.sum() * g.cellA * 1e4)
        # axis gap (rows j = N/2-1, N/2 straddle y = 0)
        j0 = g.N // 2 - 1
        ro = old[:, j0] | old[:, j0 + 1]
        rn = new[:, j0] | new[:, j0 + 1]
        if ro.any():
            io = np.argmax(ro)
            cand = np.nonzero(rn[:io])[0]
            out['gap_axis'] = float((io - cand.max() - 1) * g.h / D) if len(cand) else float('nan')
        else:
            out['gap_axis'] = float('nan')
    out['old_area'] = float(old.sum() * g.cellA * 1e4)
    out['old_raft_area'] = float(sim.o.sum() * g.cellA * 1e4)
    # o-field axis edges
    j0 = g.N // 2 - 1
    prof = 0.5 * (sim.o[:, j0] + sim.o[:, j0 + 1])
    idx = np.nonzero(prof >= 0.5)[0]
    out['old_near_x'] = float(g.x1[idx.min()] / D) if len(idx) else float('nan')
    out['old_far_x'] = float(g.x1[idx.max()] / D) if len(idx) else float('nan')
    # solidity of the old region
    pts = np.c_[g.X[old], g.Y[old]]
    if len(pts) > 10:
        hull = ConvexHull(pts)
        out['old_solidity'] = float(old.sum() * g.cellA / hull.volume)
    else:
        out['old_solidity'] = float('nan')
    return out


def T2(params=None, N=128, keep=False, s0=0.15, travel=0.10, t_move=1.5, t_stop=4.0, t_end=6.0):
    g = grid(N)
    sim = PXSim(grid=g, params=params)
    cx, rad = 0.12 * D, 0.165 * D
    sim.add_disc(cx, 0.0, rad, old=True)
    x_near0 = cx - rad
    xs = x_near0 - s0 * D
    xf = lambda t: xs + travel * D * min(t, t_move) / t_move
    inp = jet_inp(xf, lambda t: 0.0, lambda t: Q_LOW if t < t_stop else 0.0, V_LOW)
    tk = lambda t: 0.0
    th = np.arange(720) * 2 * np.pi / 720
    mline = np.c_[cx + rad * np.cos(th), rad * np.sin(th)]
    sim.markers = mline.copy()
    A_mat0 = _poly_area(mline)
    rec = {}
    snaps = {}
    times = [0.5, 1.0, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

    def cb(s, t):
        m = gap_metrics(s)
        P = s.markers
        xc = _poly_axis_cross(P)
        m['mat_near_x'] = float(xc.min() / D) if len(xc) else float('nan')
        m['mat_far_x'] = float(xc.max() / D) if len(xc) else float('nan')
        m['mat_area_ratio'] = float(_poly_area(P) / A_mat0)
        try:
            hull = ConvexHull(P)
            m['mat_solidity'] = float(_poly_area(P) / hull.volume)
        except Exception:
            m['mat_solidity'] = float('nan')
        # angular span of the old block seen from the jet
        xj = xf(t)
        old, new = old_new_masks(s)
        ang = np.degrees(np.arctan2(g.Y[old], g.X[old] - xj))
        if len(ang):
            hist = np.histogram(ang, bins=72, range=(-180, 180))[0] > 0
            m['old_span_deg'] = float(hist.sum() * 5)
        else:
            m['old_span_deg'] = 0.0
        m['jet_x'] = float(xj / D)
        rec[round(t, 3)] = m
        snaps[round(t, 3)] = (s.c.copy(), s.o.copy(), s.markers.copy())
    sim.run(inp, t_end, tk, breakpoints=[t_move, t_stop], callback=cb, cb_times=times)
    obs = {}
    near0 = (x_near0) / D; far0 = (cx + rad) / D
    for t in (1.0, 2.0, 4.0, 6.0):
        m = rec[t]
        obs[f'near_disp_{t:.0f}'] = m['mat_near_x'] - near0
        obs[f'far_disp_{t:.0f}'] = m['mat_far_x'] - far0
        obs[f'near_disp_o_{t:.0f}'] = m['old_near_x'] - near0
        obs[f'far_disp_o_{t:.0f}'] = m['old_far_x'] - far0
        obs[f'gap_min_{t:.0f}'] = m['gap_min']
        obs[f'gap_axis_{t:.0f}'] = m['gap_axis']
        obs[f'contact_{t:.0f}'] = m['contact']
        obs[f'old_solidity_{t:.0f}'] = m['old_solidity']
        obs[f'mat_solidity_{t:.0f}'] = m['mat_solidity']
        obs[f'old_span_{t:.0f}'] = m['old_span_deg']
        obs[f'new_area_{t:.0f}'] = m['new_area']
        obs[f'old_raft_area_{t:.0f}'] = m['old_raft_area']
        obs[f'mat_area_ratio_{t:.0f}'] = m['mat_area_ratio']
    obs['old_raft_area_0'] = float(np.pi * rad ** 2 * 1e4)
    obs['series'] = {str(t): {k: rec[t][k] for k in ('gap_min', 'gap_axis', 'contact', 'mat_near_x', 'mat_far_x',
                                                    'mat_solidity', 'old_solidity', 'jet_x', 'new_area', 'mat_area_ratio')}
                     for t in sorted(rec)}
    obs['near0'] = near0; obs['far0'] = far0
    wc, _ = interface_width_cells(sim.c, g)
    obs['iface_width_cells'] = float(wc)
    finish(sim, obs)
    extra = dict(rec=rec, snaps=snaps, cx=cx, rad=rad, xs=xs) if keep else None
    return obs, extra


# ============================================================================================ T3
def T3(params=None, N=128, keep=False, gate='lift', dwell=0.0, t_end=5.0):
    """thin stream crossing a white disc (radius 0.25 D) along y = 0 at 0.4 D/s.
    gate='lift': the stream follows a flow decrease, the supply gate is already open;
    gate='rest': the stream starts from Q = 0 at t = 0 (flow-increase event)."""
    g = grid(N)
    sim = PXSim(grid=g, params=params)
    rad = 0.25 * D
    sim.add_disc(0.0, 0.0, rad, old=True)
    x0, V = -0.35 * D, 0.4 * D
    xd = -rad                      # dwell position (entry edge)
    t_d0 = (xd - x0) / V
    def xf(t):
        if dwell > 0:
            if t < t_d0: return x0 + V * t
            if t < t_d0 + dwell: return xd
            return xd + V * (t - t_d0 - dwell)
        return x0 + V * t
    t_cut = 0.7 * D / V + dwell
    inp = jet_inp(xf, lambda t: 0.0, lambda t: Q_THIN if t < t_cut else 0.0, V_THIN)
    tk = (lambda t: -100.0) if gate == 'lift' else (lambda t: 0.0)
    J = jet_derived(Q_THIN, V_THIN, sim.P)
    stations = np.array([-0.15, -0.05, 0.05, 0.15]) * D
    t_pass = np.array([(s - x0) / V + (dwell if (dwell > 0 and s > xd) else 0.0) for s in stations])
    t_imm = t_pass + 0.15
    t_post = t_pass + J['tr'] + 0.15
    times = sorted(set([round(x, 4) for x in list(t_imm) + list(t_post)] +
                       [round(t_cut, 4), 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]))
    times = [t for t in times if t <= t_end + 1e-9]
    snaps = {}

    def cb(s, t):
        snaps[round(t, 4)] = s.c.copy()
    sim.run(inp, t_end, tk, breakpoints=[t_cut, t_d0, t_d0 + dwell], callback=cb, cb_times=times)
    yy = g.x1
    band = np.abs(yy) < 0.006

    def deficit(c, xs):
        i = int(np.argmin(np.abs(g.x1 - xs)))
        prof = 0.5 * (c[i, :] + c[i + 1 if g.x1[i] < xs else i - 1, :])
        return float(((1 - prof[band]).sum() * g.h) / D), prof
    obs = dict(jet={k: float(v) for k, v in J.items()}, two_rd=2 * J['rd'] / D)
    di, dp, dfin, vw = [], [], [], []
    for s_, a, b in zip(stations, t_imm, t_post):
        di.append(deficit(snaps[round(a, 4)], s_)[0])
        dp.append(deficit(snaps[round(b, 4)], s_)[0] if round(b, 4) in snaps else float('nan'))
        d, prof = deficit(snaps[round(t_end, 4)], s_)
        dfin.append(d)
        vw.append(float((prof[band] < 0.5).sum() * g.h / D))
    obs['line_def_imm'] = float(np.mean(di)); obs['line_def_post_tr'] = float(np.nanmean(dp))
    obs['line_def_final'] = float(np.mean(dfin)); obs['line_vis_width_final'] = float(np.mean(vw))
    obs['line_def_stations_imm'] = di; obs['line_def_stations_final'] = dfin
    cF = snaps[round(t_end, 4)]
    r = np.hypot(g.X, g.Y)
    outside = (r > rad + 1.5e-3) & g.inside
    obs['white_on_brown_cm2'] = float((cF * outside).sum() * g.cellA * 1e4)
    # stem: deposit outside the disc along the path, entry side (x<-rad) and exit side (x>rad)
    for side, msk in (('entry', g.X < -rad - 1.5e-3), ('exit', g.X > rad + 1.5e-3)):
        m = outside & msk & (np.abs(g.Y) < 0.006)
        A = (cF * m).sum() * g.cellA
        Lx = (np.abs(g.x1) > rad + 1.5e-3) & ((g.x1 < 0) if side == 'entry' else (g.x1 > 0))
        Lx = Lx & (np.abs(g.x1) < 0.35 * D)
        obs[f'stem_width_{side}'] = float(A / max(Lx.sum() * g.h, 1e-12) / D)
    obs['disc_area_change_cm2'] = float((cF.sum() * g.cellA - np.pi * rad ** 2) * 1e4)
    # boundary radius along rays from the centre (notch/tip), excluding the line itself
    th = np.radians(np.arange(-40, 40.1, 1.0))
    rr = np.arange(0.15 * D, 0.40 * D, 1e-4)
    def rb(c, th0):
        X = rr[None, :] * np.cos(th0 + th)[:, None]; Y = rr[None, :] * np.sin(th0 + th)[:, None]
        ok = X ** 2 + Y ** 2 < (R - g.h) ** 2
        v = sample_bilinear(c, g, X, Y) * ok
        last = np.array([rr[np.nonzero(row >= 0.5)[0].max()] if (row >= 0.5).any() else np.nan for row in v])
        return last
    for name, th0 in (('exit', 0.0), ('entry', np.pi)):
        b = rb(cF, th0)
        lat = np.abs(rad * np.sin(th))
        keep_ = lat > 2.5e-3    # outside the line half-width zone
        dev = (b - rad) / D
        obs[f'{name}_bdev_max'] = float(np.nanmax(dev[keep_]))
        obs[f'{name}_bdev_min'] = float(np.nanmin(dev[keep_]))
        obs[f'{name}_bdev_profile'] = [float(x) for x in dev]
    # notch around the dwell / entry point: brown area inside the original disc within 8 mm of (-rad, 0),
    # minus the line's own contribution (deficit x path length inside the 8 mm circle)
    near = (np.hypot(g.X + rad, g.Y) < 0.008) & (r < rad) & g.inside
    brown = ((1 - cF) * near).sum() * g.cellA
    line_contrib = obs['line_def_final'] * D * 0.008
    A_notch = max(brown - line_contrib, 0.0)
    obs['notch_area_mm2'] = float(A_notch * 1e6)
    obs['notch_req'] = float(math.sqrt(2 * A_notch / math.pi) / D)
    obs['line_widening'] = obs['line_def_final'] - obs['line_def_imm']
    obs['heal_ratio'] = obs['line_def_final'] / obs['line_def_imm'] if obs['line_def_imm'] > 0 else float('nan')
    wc, _ = interface_width_cells(sim.c, g)
    obs['iface_width_cells'] = float(wc)
    obs['core_cm2'] = sim.L['core'] * 1e4
    obs['supply_cm2'] = sim.L['supply_area'] * 1e4
    finish(sim, obs)
    extra = dict(snaps=snaps, rad=rad, stations=stations, t_imm=t_imm, t_post=t_post, t_cut=t_cut) if keep else None
    return obs, extra


# ============================================================================================ T4
def stripe_metrics(c, g, x_lo, x_hi, halfband=1.25e-3):
    jm = np.abs(g.x1) <= halfband
    prof = c[:, jm].mean(1)
    sel = (g.x1 >= x_lo) & (g.x1 <= x_hi)
    xs = g.x1[sel]; p = prof[sel]
    pk, props = find_peaks(1 - p, height=0.5, prominence=0.2)
    out = dict(n=int(len(pk)), x=[float(v / D) for v in xs[pk]])
    out['pitch'] = float(np.mean(np.diff(xs[pk])) / D) if len(pk) > 1 else float('nan')
    # deficit of each minimum between neighbouring maxima
    mx, _ = find_peaks(p)
    defs = []
    for k in pk:
        left = mx[mx < k]; right = mx[mx > k]
        a = left.max() if len(left) else max(k - 4, 0)
        b = right.min() if len(right) else min(k + 4, len(p) - 1)
        defs.append(float((1 - p[a:b + 1]).sum() * g.h / D))
    out['deficits'] = defs
    out['def_mean'] = float(np.mean(defs)) if defs else float('nan')
    out['prof_x'] = xs / D; out['prof'] = p
    # 2-D: enclosed brown components inside the white (not connected to the outer canvas)
    brown = (c < 0.5) & g.inside
    lab, n = ndimage.label(brown)
    rim = np.unique(lab[(np.hypot(g.X, g.Y) > R - 2 * g.h) & brown])
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    enclosed = [k for k in range(1, n + 1) if k not in rim and sizes[k - 1] >= 2]
    out['n_enclosed'] = len(enclosed)
    out['enclosed_area_mm2'] = float(sum(sizes[k - 1] for k in enclosed) * g.cellA * 1e6)
    return out


def T4(params=None, N=128, keep=False, t_pre=1.5, amp=4e-3, freq=3.0, Vr=0.15, t_w=2.0, t_after=5.0):
    g = grid(N)
    sim = PXSim(grid=g, params=params)
    x0 = 0.15 * D
    def xf(t):
        return x0 if t < t_pre else x0 - Vr * D * min(t - t_pre, t_w)
    def yf(t):
        return 0.0 if t < t_pre else amp * math.sin(2 * math.pi * freq * min(t - t_pre, t_w))
    t_stop = t_pre + t_w
    t_end = t_stop + t_after
    inp = jet_inp(xf, yf, lambda t: Q_LOW if t < t_stop else 0.0, V_LOW)
    tk = lambda t: 0.0
    snaps = {}
    times = [t_pre, t_pre + 0.5, t_pre + 1.0, t_pre + 1.5, t_stop, t_stop + 1.0, t_end]

    def cb(s, t):
        snaps[round(t, 4)] = s.c.copy()
    sim.run(inp, t_end, tk, breakpoints=[t_pre, t_stop], callback=cb, cb_times=times)
    x_lo, x_hi = x0 - Vr * D * t_w - 0.02 * D, x0 + 0.06 * D
    s_end = stripe_metrics(snaps[round(t_stop, 4)], g, x_lo, x_hi)
    s_fin = stripe_metrics(snaps[round(t_end, 4)], g, x_lo, x_hi)
    obs = dict(n_stripes=s_end['n'], pitch=s_end['pitch'], def_mean=s_end['def_mean'],
               n_enclosed=s_end['n_enclosed'], enclosed_area_mm2=s_end['enclosed_area_mm2'],
               n_stripes_final=s_fin['n'], def_mean_final=s_fin['def_mean'], n_enclosed_final=s_fin['n_enclosed'],
               stripe_x=s_end['x'], deficits=s_end['deficits'],
               half_cycles=2 * freq * t_w, retreat_per_cycle=Vr / freq)
    obs['survival'] = (s_fin['n'] / s_end['n']) if s_end['n'] else float('nan')
    obs['max_abs_dc_after'] = float(np.abs(snaps[round(t_end, 4)] - snaps[round(t_stop + 1.0, 4)]).max())
    obs['A_vis_end'] = white_area(sim, 0.5) * 1e4
    obs['A_raft_end'] = white_area(sim) * 1e4
    mixed = ((sim.c > 0.05) & (sim.c < 0.95) & g.inside).sum()
    obs['mixed_frac_of_white'] = float(mixed / max((sim.c >= 0.5).sum(), 1))
    wc, _ = interface_width_cells(sim.c, g)
    obs['iface_width_cells'] = float(wc)
    finish(sim, obs)
    extra = dict(snaps=snaps, s_end=s_end, s_fin=s_fin, x_lo=x_lo, x_hi=x_hi, t_stop=t_stop, t_end=t_end,
                 path=[(xf(t), yf(t)) for t in np.arange(0, t_stop, 0.005)]) if keep else None
    return obs, extra
