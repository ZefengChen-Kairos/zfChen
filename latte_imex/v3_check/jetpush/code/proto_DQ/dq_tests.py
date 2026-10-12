"""Variant-Q test layer on top of the unchanged px_tests / v2_tests (spec_v2 copies).

TQ0  unit check of the discrete Leray projection of a footprint traction (uniform U on a disc of radius a).
T1   px_tests.T1 (3 s pour + 3 s stop) + T1L (6.5 s pour; v2_tests).
T2Q  px_tests.T2 (old 0.33 D disc, second low pour s0 behind its rear edge, moving 0.1 D in 1.5 s or held) + crescent /
     wrap metrics from the material line, gap series.
T3Q  copy of px_tests.T3 (thin stream across a 0.5 D disc at 0.4 D/s, optional 0.7 s dwell at the entry) + Lagrangian
     marker lines across the path (V shape) + side-bulge metrics.
T4Q  copy of px_tests.T4 (pre-pour 1.5 s, wiggle 4 mm x 3 Hz while retreating 0.15 D/s for 2 s, 5 s after) with an
     optional white base, brown-line counts on several lines parallel to the retreat axis.
All lengths returned in D unless the key says otherwise.  SI inside.
"""
import math
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
import px_tests as T
import v2_tests as V2T
from px_model import (PXSim, Grid, D, R, V2, VQ, jet_derived, white_area, sample_bilinear, interface_width_cells)

_orig_finish = T.finish


def _finish(sim, obs):
    """px_tests.finish + T5 extras: conservative redistribution moved [cm^2], net clip [cm^2], post-correction bounds,
    supply / pressed-in / dispersed areas [cm^2]."""
    o = _orig_finish(sim, obs)
    o['redist_cm2'] = sim.L.get('redist', 0.0) * 1e4
    o['clip_net_cm2'] = sim.L['clip_c'] * 1e4
    o['c_max_after'] = float(sim.c.max()); o['c_min_after'] = float(sim.c.min())
    o['supply_cm2'] = sim.L['supply_area'] * 1e4
    o['pressed_cm2'] = sim.L.get('pressed', 0.0) * 1e4
    o['core_cm2'] = sim.L['core'] * 1e4
    o['redist_per_supply'] = o['redist_cm2'] / max(o['supply_cm2'] + o['pressed_cm2'], 1e-30)
    return o


T.finish = _finish

try:
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    HAVE_SHAPELY = True
except Exception:          # pragma: no cover
    HAVE_SHAPELY = False


# ============================================================================================ TQ0
def TQ0(N=128, U=0.09, a=2.72e-3, x0=0.0):
    """uniform U (along +x) on a disc of radius a centred at (x0, 0); discrete Leray projection as in px_model."""
    g = T.grid(N)
    sim = PXSim(grid=g)
    i0, i1, j0, j1, Kw = sim.kernel(x0, 0.0, 'ann', 0.0, a)
    W = np.zeros((N, N)); W[i0:i1, j0:j1] = U * Kw * math.pi * a * a
    Ufx = np.zeros((N + 1, N)); Ufy = np.zeros((N, N + 1))
    Ufx[1:-1, :] = 0.5 * (W[1:, :] + W[:-1, :]); Ufx[~g.mx] = 0.0
    phi, imb = g.solve(-g.div(Ufx, Ufy) * g.inside)
    ux, uy = g.face_vel(phi); ux = ux + Ufx
    out = dict(N=N, max_div=float(np.abs(g.div(ux, uy))[g.inside].max()), imb=float(imb))
    m = U * math.pi * a * a
    out['m'] = m
    # centre of the disc (average of the cell-centred velocity in r < a/2)
    vx, vy = g.interp(ux, uy, np.array([x0]), np.array([0.0]))
    out['u_centre'] = float(vx[0]); out['u_centre_theory'] = U / 2
    for r in (0.006, 0.012, 0.02):
        vx, _ = g.interp(ux, uy, np.array([x0 + r]), np.array([0.0]))
        out[f'u_axis_{r*1e3:.0f}mm'] = float(vx[0])
        out[f'u_axis_{r*1e3:.0f}mm_theory_cup'] = m / (2 * math.pi) * (1 / r ** 2 - 1 / R ** 2) if x0 == 0 else float('nan')
        vx, _ = g.interp(ux, uy, np.array([x0]), np.array([r]))
        out[f'u_lat_{r*1e3:.0f}mm'] = float(vx[0])
        out[f'u_lat_{r*1e3:.0f}mm_theory_cup'] = -m / (2 * math.pi) * (1 / r ** 2 + 1 / R ** 2) if x0 == 0 else float('nan')
    # vorticity outside the footprint must vanish (potential dipole): discrete curl on cell corners
    cz = (uy[1:, 1:-1] - uy[:-1, 1:-1]) / g.h - (ux[1:-1, 1:] - ux[1:-1, :-1]) / g.h
    Xc = (np.arange(1, N) * g.h - R)[:, None]; Yc = (np.arange(1, N) * g.h - R)[None, :]
    rc = np.hypot(Xc - x0, Yc)
    ins = (Xc ** 2 + Yc ** 2) < (R - 2 * g.h) ** 2
    out['max_curl_outside_2a'] = float(np.abs(cz[(rc > 2 * a) & ins]).max())
    out['max_curl_inside'] = float(np.abs(cz[(rc < a) & ins]).max())
    return out


# ============================================================================================ T1
def T1(params=None, N=128, keep=False):
    return T.T1(params=params, N=N, keep=keep)


def T1L(params=None, N=128):
    return V2T.T1L(params=params, N=N)


# ============================================================================================ T2
def _largest(geom):
    if geom.is_empty:
        return None
    if geom.geom_type == 'Polygon':
        return geom
    parts = [p for p in getattr(geom, 'geoms', []) if p.geom_type == 'Polygon']
    return max(parts, key=lambda p: p.area) if parts else None


def crescent_metrics(P, xj, new_mask, g):
    """material polygon P (n,2) of the old block, jet x position xj (y = 0), boolean new-white mask.
    Returns solidity, axial thickness, transverse span, angular span seen from the jet, concavity direction (cos between
    the bay centroid offset and the direction to the jet), share of the new white inside the convex hull."""
    out = {}
    A = T._poly_area(P)
    try:
        hull = ConvexHull(P)
        out['solidity'] = float(A / hull.volume)
    except Exception:
        out['solidity'] = float('nan'); return out
    xc = T._poly_axis_cross(P)
    out['axial_thick'] = float((xc.max() - xc.min()) / D) if len(xc) else float('nan')
    out['transverse_span'] = float(np.ptp(P[:, 1]) / D)
    ang = np.degrees(np.arctan2(P[:, 1], P[:, 0] - xj))
    # span of directions occupied (5 deg bins), robust to wrap-around
    hist = np.histogram(ang, bins=72, range=(-180, 180))[0] > 0
    out['span_from_jet_deg'] = float(hist.sum() * 5)
    out['dist_jet_min'] = float(np.hypot(P[:, 0] - xj, P[:, 1]).min() / D)
    if HAVE_SHAPELY:
        poly = Polygon(P).buffer(0)
        hp = Polygon(P[hull.vertices])
        bay = _largest(hp.difference(poly))
        cen = poly.centroid
        if bay is not None and bay.area > 0:
            bc = bay.centroid
            dvec = np.array([bc.x - cen.x, bc.y - cen.y]); jv = np.array([xj - cen.x, 0.0 - cen.y])
            out['bay_area_frac'] = float(bay.area / poly.area)
            out['bay_cos_jet'] = float(dvec @ jv / max(np.linalg.norm(dvec) * np.linalg.norm(jv), 1e-30))
        else:
            out['bay_area_frac'] = 0.0; out['bay_cos_jet'] = float('nan')
        if new_mask is not None and new_mask.any():
            from matplotlib.path import Path
            pts = np.c_[g.X[new_mask], g.Y[new_mask]]
            inside = Path(P[hull.vertices]).contains_points(pts)
            out['new_in_hull_frac'] = float(inside.mean())
        else:
            out['new_in_hull_frac'] = float('nan')
    sol = out['solidity']; cosb = out.get('bay_cos_jet', float('nan'))
    out['crescent'] = bool(sol < 0.85 and np.isfinite(cosb) and cosb > 0.5 and out.get('new_in_hull_frac', 0) > 0.05)
    return out


def T2Q(params=None, N=128, keep=False, s0=0.15, travel=0.10, t_move=1.5, t_stop=4.0, t_end=6.0):
    o, ex = T.T2(params=params, N=N, keep=True, s0=s0, travel=travel, t_move=t_move, t_stop=t_stop, t_end=t_end)
    g = T.grid(N)
    xs = ex['xs']
    xjf = lambda t: xs + travel * D * min(t, t_move) / t_move
    out = {k: v for k, v in o.items() if k != 'series'}
    out['series'] = o['series']
    out['s0'] = s0; out['travel'] = travel
    for t in (1.4, 2.0, 4.0, 6.0):
        c, oo, P = ex['snaps'][t]
        white = c >= 0.5; new = white & (oo < 0.5 * c)
        cm = crescent_metrics(P, xjf(t), new, g)
        for k, v in cm.items():
            out[f'cr_{k}_{t:g}'] = v
    # pre-white near-edge displacement (first white at 1.4 s)
    m14 = ex['rec'][1.4]
    out['near_disp_prewhite'] = m14['mat_near_x'] - o['near0']
    # gap series (min distance old-new white and axial gap)
    out['gap_min_series'] = {str(t): ex['rec'][t]['gap_min'] for t in sorted(ex['rec'])}
    out['gap_axis_series'] = {str(t): ex['rec'][t]['gap_axis'] for t in sorted(ex['rec'])}
    out['old_area_ratio_4'] = o['old_raft_area_4'] / o['old_raft_area_0']
    gaps = [ex['rec'][t]['gap_min'] for t in (2.0, 4.0, 6.0)]
    out['gap_pass'] = bool(all(np.isfinite(x) and 0.04 <= x <= 0.13 for x in gaps))
    return out, (ex if keep else None)


# ============================================================================================ T3
def T3Q(params=None, N=128, keep=False, gate='lift', dwell=0.0, t_end=5.0):
    """copy of px_tests.T3 + marker lines across the path."""
    g = T.grid(N)
    sim = PXSim(grid=g, params=params)
    rad = 0.25 * D
    sim.add_disc(0.0, 0.0, rad, old=True)
    x0, V = -0.35 * D, 0.4 * D
    xd = -rad
    t_d0 = (xd - x0) / V

    def xf(t):
        if dwell > 0:
            if t < t_d0: return x0 + V * t
            if t < t_d0 + dwell: return xd
            return xd + V * (t - t_d0 - dwell)
        return x0 + V * t
    t_cut = 0.7 * D / V + dwell
    inp = T.jet_inp(xf, lambda t: 0.0, lambda t: T.Q_THIN if t < t_cut else 0.0, T.V_THIN)
    tk = (lambda t: -100.0) if gate == 'lift' else (lambda t: 0.0)
    J = jet_derived(T.Q_THIN, T.V_THIN, sim.P)
    # marker lines x = const across the path (V shape), y in [-0.22, 0.22] D, plus two lines along the path at
    # y = +-0.06 D (lateral displacement = side bulge of the material)
    st_x = np.array([-0.15, -0.05, 0.05, 0.15]) * D
    yy = np.linspace(-0.22, 0.22, 177) * D
    mk = [np.c_[np.full_like(yy, s), yy] for s in st_x]
    mk = np.vstack(mk)
    sim.markers = mk.copy()
    stations = np.array([-0.15, -0.05, 0.05, 0.15]) * D
    t_pass = np.array([(s - x0) / V + (dwell if (dwell > 0 and s > xd) else 0.0) for s in stations])
    t_imm = t_pass + 0.15
    t_post = t_pass + J['tr'] + 0.15
    times = sorted(set([round(x, 4) for x in list(t_imm) + list(t_post)] +
                       [round(t_cut, 4), 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] +
                       ([round(t_d0 + dwell, 4)] if dwell > 0 else [])))
    times = [t for t in times if t <= t_end + 1e-9]
    snaps = {}; msnaps = {}

    def cb(s, t):
        snaps[round(t, 4)] = s.c.copy(); msnaps[round(t, 4)] = s.markers.copy()
    sim.run(inp, t_end, tk, breakpoints=[t_cut, t_d0, t_d0 + dwell], callback=cb, cb_times=times)
    band = np.abs(g.x1) < 0.006

    def deficit(c, xs):
        i = int(np.argmin(np.abs(g.x1 - xs)))
        prof = 0.5 * (c[i, :] + c[i + 1 if g.x1[i] < xs else i - 1, :])
        return float(((1 - prof[band]).sum() * g.h) / D), prof
    obs = dict(jet={k: float(v) for k, v in J.items()}, two_rd=2 * J['rd'] / D, dwell=dwell)
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
    for side, msk in (('entry', g.X < -rad - 1.5e-3), ('exit', g.X > rad + 1.5e-3)):
        m = outside & msk & (np.abs(g.Y) < 0.006)
        A = (cF * m).sum() * g.cellA
        Lx = (np.abs(g.x1) > rad + 1.5e-3) & ((g.x1 < 0) if side == 'entry' else (g.x1 > 0))
        Lx = Lx & (np.abs(g.x1) < 0.35 * D)
        obs[f'stem_width_{side}'] = float(A / max(Lx.sum() * g.h, 1e-12) / D)
    obs['disc_area_change_cm2'] = float((cF.sum() * g.cellA - np.pi * rad ** 2) * 1e4)
    th = np.radians(np.arange(-40, 40.1, 1.0))
    rr = np.arange(0.15 * D, 0.40 * D, 1e-4)

    def rb(c, th0):
        X = rr[None, :] * np.cos(th0 + th)[:, None]; Y = rr[None, :] * np.sin(th0 + th)[:, None]
        ok = X ** 2 + Y ** 2 < (R - g.h) ** 2
        v = sample_bilinear(c, g, X, Y) * ok
        return np.array([rr[np.nonzero(row >= 0.5)[0].max()] if (row >= 0.5).any() else np.nan for row in v])
    adeg = np.degrees(th)
    for name, th0 in (('exit', 0.0), ('entry', np.pi)):
        b = rb(cF, th0)
        dev = (b - rad) / D
        obs[f'{name}_bdev_profile'] = [float(x) for x in dev]
        side = (np.abs(adeg) >= 5) & (np.abs(adeg) <= 30)
        obs[f'{name}_bulge_max'] = float(np.nanmax(dev[side]))          # outward = side bulge
        obs[f'{name}_bulge_10_15'] = float(np.nanmean(dev[(np.abs(adeg) >= 10) & (np.abs(adeg) <= 15)]))
        obs[f'{name}_bulge_20_25'] = float(np.nanmean(dev[(np.abs(adeg) >= 20) & (np.abs(adeg) <= 25)]))
        obs[f'{name}_axis_dev'] = float(np.nanmean(dev[np.abs(adeg) <= 1]))
        obs[f'{name}_dev_5'] = float(np.nanmean(dev[(np.abs(adeg) >= 4) & (np.abs(adeg) <= 6)]))
        obs[f'{name}_dev_10'] = float(np.nanmean(dev[(np.abs(adeg) >= 9) & (np.abs(adeg) <= 11)]))
    near = (np.hypot(g.X + rad, g.Y) < 0.008) & (r < rad) & g.inside
    brown = ((1 - cF) * near).sum() * g.cellA
    line_contrib = obs['line_def_final'] * D * 0.008
    A_notch = max(brown - line_contrib, 0.0)
    obs['notch_area_mm2'] = float(A_notch * 1e6)
    obs['notch_req'] = float(math.sqrt(2 * A_notch / math.pi) / D)
    obs['heal_ratio'] = obs['line_def_final'] / obs['line_def_imm'] if obs['line_def_imm'] > 0 else float('nan')
    # V shape from the marker lines: forward (+x) displacement close to the path minus far from it
    Mf = msnaps[round(t_end, 4)]
    n = len(yy); vamp, lat = [], []
    for k in range(len(st_x)):
        dxk = (Mf[k * n:(k + 1) * n, 0] - mk[k * n:(k + 1) * n, 0]) / D
        dyk = (Mf[k * n:(k + 1) * n, 1] - mk[k * n:(k + 1) * n, 1]) / D
        yD = yy / D
        nearp = (np.abs(yD) >= 0.02) & (np.abs(yD) <= 0.04)
        far = (np.abs(yD) >= 0.12) & (np.abs(yD) <= 0.18)
        vamp.append(float(dxk[nearp].mean() - dxk[far].mean()))
        lat.append(float(np.mean(np.sign(yD[nearp]) * dyk[nearp])))
    obs['V_amp_stations'] = vamp; obs['V_amp'] = float(np.mean(vamp))
    obs['lat_out_near_stations'] = lat; obs['lat_out_near'] = float(np.mean(lat))
    wc, _ = interface_width_cells(sim.c, g)
    obs['iface_width_cells'] = float(wc)
    obs['core_cm2'] = sim.L['core'] * 1e4
    obs['supply_cm2'] = sim.L['supply_area'] * 1e4
    T.finish(sim, obs)
    extra = dict(snaps=snaps, msnaps=msnaps, mk0=mk, rad=rad, stations=stations, t_cut=t_cut, nline=n,
                 st_x=st_x) if keep else None
    return obs, extra


# ============================================================================================ T4
def line_counts(c, g, x_lo, x_hi, yD, halfband=1.25e-3, prom=0.3):
    """brown lines (minima of c flanked by white c >= 0.5 on both sides) along the line y = yD*D, x in [x_lo, x_hi]."""
    jm = np.abs(g.x1 - yD * D) <= halfband
    prof = c[:, jm].mean(1)
    sel = (g.x1 >= x_lo) & (g.x1 <= x_hi)
    xs = g.x1[sel]; p = prof[sel]
    pk, _ = find_peaks(1 - p, height=0.5, prominence=prom)
    keep = []
    for k in pk:
        if (p[:k] >= 0.5).any() and (p[k + 1:] >= 0.5).any():
            keep.append(k)
    keep = np.array(keep, int)
    mx, _ = find_peaks(np.r_[0.0, p, 0.0])
    mx = mx - 1
    defs = []
    for k in keep:
        left = mx[mx < k]; right = mx[mx > k]
        a = left.max() if len(left) else max(k - 4, 0)
        b = right.min() if len(right) else min(k + 4, len(p) - 1)
        defs.append(float((1 - p[a:b + 1]).sum() * g.h / D))
    xk = xs[keep] / D
    return dict(n=int(len(keep)), x=[float(v) for v in xk],
                pitch=float(np.mean(np.diff(np.sort(xk)))) if len(keep) > 1 else float('nan'),
                widths=defs, width_mean=float(np.mean(defs)) if defs else float('nan'))


def T4Q(params=None, N=128, keep=False, t_pre=1.5, amp=4e-3, freq=3.0, Vr=0.15, t_w=2.0, t_after=5.0, base=False,
        base_rad=0.25):
    g = T.grid(N)
    sim = PXSim(grid=g, params=params)
    if base:
        sim.add_disc(0.0, 0.0, base_rad * D, old=True)
    x0 = 0.15 * D

    def xf(t):
        return x0 if t < t_pre else x0 - Vr * D * min(t - t_pre, t_w)

    def yf(t):
        return 0.0 if t < t_pre else amp * math.sin(2 * math.pi * freq * min(t - t_pre, t_w))
    t_stop = t_pre + t_w
    t_end = t_stop + t_after
    inp = T.jet_inp(xf, yf, lambda t: T.Q_LOW if t < t_stop else 0.0, T.V_LOW)
    tk = lambda t: 0.0
    snaps = {}
    times = [t_pre, t_pre + 0.5, t_pre + 1.0, t_pre + 1.5, t_stop, t_stop + 1.0, t_end]

    def cb(s, t):
        snaps[round(t, 4)] = s.c.copy()
    sim.run(inp, t_end, tk, breakpoints=[t_pre, t_stop], callback=cb, cb_times=times)
    x_lo, x_hi = x0 - Vr * D * t_w - 0.02 * D, x0 + 0.06 * D
    obs = dict(base=base, cycles=freq * t_w, retreat_per_cycle=Vr / freq)
    for tag, tt in (('end', t_stop), ('fin', t_end)):
        cS = snaps[round(tt, 4)]
        for yD in (0.0, 0.02, -0.02, 0.04, -0.04):
            lc = line_counts(cS, g, x_lo, x_hi, yD)
            key = f'{tag}_y{yD:+.2f}'
            obs[key + '_n'] = lc['n']; obs[key + '_pitch'] = lc['pitch']; obs[key + '_w'] = lc['width_mean']
            obs[key + '_x'] = lc['x']
        st = T.stripe_metrics(cS, g, x_lo, x_hi)
        obs[f'{tag}_n_enclosed'] = st['n_enclosed']; obs[f'{tag}_enclosed_mm2'] = st['enclosed_area_mm2']
    ns = [obs[f'end_y{y:+.2f}_n'] for y in (0.0, 0.02, -0.02, 0.04, -0.04)]
    obs['n_lines_median'] = float(np.median(ns))
    obs['n_lines_max'] = int(max(ns))
    obs['max_abs_dc_after'] = float(np.abs(snaps[round(t_end, 4)] - snaps[round(t_stop + 1.0, 4)]).max())
    obs['A_vis_end'] = white_area(sim, 0.5) * 1e4
    mixed = ((sim.c > 0.05) & (sim.c < 0.95) & g.inside).sum()
    obs['mixed_frac_of_white'] = float(mixed / max((sim.c >= 0.5).sum(), 1))
    wc, _ = interface_width_cells(sim.c, g)
    obs['iface_width_cells'] = float(wc)
    T.finish(sim, obs)
    extra = dict(snaps=snaps, x_lo=x_lo, x_hi=x_hi, t_stop=t_stop, t_end=t_end,
                 path=[(xf(t), yf(t)) for t in np.arange(0, t_stop, 0.005)]) if keep else None
    return obs, extra
