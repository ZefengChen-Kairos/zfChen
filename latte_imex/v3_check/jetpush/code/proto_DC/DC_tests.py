"""Variant-C tests on proto_DC (test_Ccompres.md).  All lengths in D unless stated, areas cm^2, times s.
Harness = proto_DI/DI_tests.py (T1x-T4x, same observables so that I, Q and C are comparable) with the variant-I parts
removed and the variant-C checks added:
  unit_C0   T-C0: screened point source in an empty cup vs u_r = S K1(r/l_s)/(2 pi l_s)
  T1x       T1 regression (3 s pour + stop) with markers 0.10-0.45 D, observed to t_end (stop-flow drift)
  T2x       push stand-off: old 0.33 D disc, second low pour s0 behind its rear edge, moving or held; + a canvas
            material loop between jet and block (DC.md 4: brown-line / white-arc area ratio before the white)
  T3x       thin cut across a 0.5 D disc (optional entry dwell) + material line for the V shape
  T4x       wiggle 4 mm x 3 Hz while retreating 0.15 D/s for 2 s, fresh canvas or on a white base; brown lines on
            y = 0, +-0.02, +-0.04 D (DQ line_counts)
"""
import math, time, json
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
import px_tests as T
from px_model import PXSim, V2, VQ, D, R, jet_derived, white_area, interface_width_cells

# variant table (DC.md naming): v2 = v3-min-2, q = variant Q, svL = C-V l_s = L cm with W, svLn = without W,
# cdX = C-D control (D_Pi = X m^2/s) with W, cp = C-P diagnostic (l_c = 1 mm compression, l_a = 4 cm expansion) with W
VARS = {'v2': {}, 'q': dict(jetU=1.0)}
for tag, L in (('20', 0.20), ('4', 0.04), ('2', 0.02), ('1', 0.01), ('05', 0.005)):
    VARS['sv' + tag] = dict(jetU=1.0, canvas='screen', ell_s=L)
    VARS['sv' + tag + 'n'] = dict(canvas='screen', ell_s=L)
for tag, Dp in (('1e3', 1e-3), ('1e4', 1e-4), ('3e5', 3e-5)):
    VARS['cd' + tag] = dict(jetU=1.0, canvas='elastic', D_Pi=Dp)
    VARS['cd' + tag + 'n'] = dict(canvas='elastic', D_Pi=Dp)
VARS['cp'] = dict(jetU=1.0, canvas='plateau', ell_c=1e-3, ell_a=0.04)
VARS['cpn'] = dict(canvas='plateau', ell_c=1e-3, ell_a=0.04)


def P(var='v2', **kw):
    p = dict(V2); p.update(VARS[var]); p.update(kw); return p


def ledg(sim, obs):
    """T5: ledgers 1 (area / divergence, with the variant-C absorption A_abs), 2 (foam), 3 (surface), bounds, cost."""
    L = sim.ledgers()
    obs['foam_rel_err'] = abs(L['foam_rel']); obs['surf_err_cm2'] = abs(L['surf_err']) * 1e4
    obs['clip_cm2'] = abs(sim.L['clip_c']) * 1e4; obs['redist_cm2'] = sim.L.get('redist', 0.0) * 1e4
    obs['max_c'] = sim.stats['max_c']; obs['min_c'] = sim.stats['min_c']
    obs['c_max_after'] = float(sim.c.max()); obs['c_min_after'] = float(sim.c.min())
    obs['max_divres'] = float(sim.stats['max_divres'])
    si, da, ab = sim.L.get('src_int', 0.0), sim.L.get('div_int', 0.0), sim.L.get('A_abs', 0.0)
    obs['src_int_cm2'] = si * 1e4; obs['A_abs_cm2'] = ab * 1e4
    obs['ledger1_rel'] = abs(si - ab - da) / max(abs(si), abs(ab), 1e-30)
    obs['steps'] = sim.stats['steps']; obs['wall_per_sim_s'] = sim.stats['t_wall'] / max(sim.t, 1e-9)
    obs['ms_per_step'] = 1e3 * sim.stats['t_wall'] / max(sim.stats['steps'], 1)
    obs['ms_poisson'] = 1e3 * sim.stats['t_poisson'] / max(sim.stats['steps'], 1)
    g = sim.g
    obs['n_helm'] = getattr(g, 'n_helm', 0); obs['cg_its_per_solve'] = getattr(g, 'cg_its', 0) / max(getattr(g, 'n_helm', 0), 1)
    obs['n_fact'] = getattr(g, 'n_fact', 0)
    obs['active_its_max'] = sim.stats.get('max_active_its', 0); obs['active_mismatch'] = sim.stats.get('active_mismatch', 0)
    return obs


def fresh_grid(N):
    """a private Grid per run (the variant-C LU preconditioner is cached on the grid)."""
    from px_model import Grid
    return Grid(N)


# ============================================================================================ unit test
def unit_C0(N=128, S=3.8e-4, ells=(0.005, 0.01, 0.02), radii=(0.004, 0.008, 0.012, 0.02)):
    from scipy.special import k1
    g = fresh_grid(N); sim = PXSim(grid=g)
    sig = np.zeros((N, N)); kw = sim.kernel(0.0, 0.0, 'ann', 0.0, 1.0e-3); sim._add(sig, kw, S)
    rows = []
    for ell in ells:
        kap2 = np.ones((N, N)) * g.inside / ell ** 2
        phi, _ = g.solve_helm(sig * g.inside, kap2)
        ux, uy = g.face_vel(phi)
        for r in radii:
            vx, vy = g.interp(ux, uy, np.array([r]), np.array([0.0]))
            th = S / (2 * math.pi) * k1(r / ell) / ell
            rows.append(dict(N=N, ell_cm=ell * 100, r_mm=r * 1e3, u_mm_s=float(vx[0] * 1e3), th_mm_s=float(th * 1e3),
                             rel=float(vx[0] / th - 1)))
        div = g.div(ux, uy) - sig - kap2 * phi
        rows.append(dict(N=N, ell_cm=ell * 100, max_div_res=float(np.abs(div[g.inside]).max())))
    return rows


# ============================================================================================ T1x
def T1x(var='v2', N=128, t_stop=3.0, t_end=13.0, keep=False, params=None):
    g = fresh_grid(N)
    p = params if params is not None else P(var)
    sim = PXSim(grid=g, params=p)
    inp = T.jet_inp(lambda t: 0.0, lambda t: 0.0, lambda t: T.Q_LOW if t < t_stop else 0.0, T.V_LOW)
    tk = lambda t: 0.0
    radii = [0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.45]
    th = np.arange(32) * 2 * np.pi / 32
    mk = np.array([[rr * D * math.cos(a), rr * D * math.sin(a)] for rr in radii for a in th])
    sim.markers = mk.copy()
    J = jet_derived(T.Q_LOW, T.V_LOW, sim.P)
    t_fr = round(t_stop + J['tr'] + 0.05, 3)
    tw = sim.P['tau_w']
    t_r1, t_r2 = round(tw + 0.1, 4), round(tw + 0.6, 4)
    snap_t = sorted(set([1.4, 2.0, 2.4, 2.65, 2.9, 3.0, 6.0, t_r1, t_r2, t_fr, t_end, 4.0, 5.0]))
    ser = dict(t=[], A_r=[], A_vis=[], umax=[])
    snaps, mks = {}, {}

    def cb(s, t):
        ser['t'].append(t); ser['A_r'].append(s.c.sum() * g.cellA * 1e4); ser['A_vis'].append(white_area(s, 0.5) * 1e4)
        ser['umax'].append(float(max(np.abs(s.ux).max(), np.abs(s.uy).max())))
        if any(abs(t - u) < 1e-6 for u in snap_t):
            snaps[round(t, 3)] = s.c.copy(); mks[round(t, 3)] = s.markers.copy()
    ts = np.round(np.arange(0.05, t_end + 1e-9, 0.05), 4)
    fine = np.round(np.arange(1.30, 1.70, 0.01), 4)
    alltimes = sorted(set(list(ts) + list(fine) + snap_t))
    sim.run(inp, t_end, tk, breakpoints=[t_stop], callback=cb, cb_times=alltimes)
    tt = np.array(ser['t']); Ar = np.array(ser['A_r']); Av = np.array(ser['A_vis']); um = np.array(ser['umax'])

    def tcross(A, thr):
        i = np.argmax(A >= thr)
        if A[i] < thr: return float('nan')
        if i == 0: return float(tt[0])
        return float(tt[i - 1] + (thr - A[i - 1]) / (A[i] - A[i - 1]) * (tt[i] - tt[i - 1]))
    o = dict(var=var)
    o['t_first_raw'] = tcross(Ar, 1e-3); o['t_first_vis'] = tcross(Av, 0.05)
    rs = T.ring_stats(snaps[round(t_r1, 3)], g, 0.0, 0.0)
    o['ring_rin_tw+0.1_mm'] = rs['r_hole_med'] * 1e3; o['ring_open_deg_tw+0.1'] = rs['open_deg']
    for tsn in (2.0, 3.0, 6.0):
        rs = T.ring_stats(snaps[tsn], g, 0.0, 0.0)
        o[f'hole_r_{tsn:.0f}'] = rs['r_hole_med'] / D; o[f'open_deg_{tsn:.0f}'] = rs['open_deg']
        o[f'open_dir_{tsn:.0f}'] = rs['open_dir']; o[f'c_centre_{tsn:.0f}'] = rs['c_centre']
        h = T.hole_and_specks(snaps[tsn], g, 0.0, 0.0)
        o[f'specks_n_{tsn:.0f}'] = h['n_specks']; o[f'specks_mm2_{tsn:.0f}'] = h['specks_mm2']
    # inscribed hole radius (v3-min-2 g3 definition, sub-cell), mean over 2.4 / 2.65 / 2.9 s (t_f + 1.0 .. 1.5)
    import sys, os
    o['hole_ins'] = float(np.mean([hole_ins(snaps[t_], g, J['r0']) for t_ in (2.4, 2.65, 2.9)])) / D
    t_a = (o['t_first_raw'] if np.isfinite(o['t_first_raw']) else 1.4) + 0.5
    m = (tt >= t_a - 1e-9) & (tt <= t_stop + 1e-9)
    o['dAdt_raft'] = float(np.polyfit(tt[m], Ar[m], 1)[0]); o['dAdt_vis'] = float(np.polyfit(tt[m], Av[m], 1)[0])
    i3 = int(np.argmin(abs(tt - t_stop))); i6 = int(np.argmin(abs(tt - 6.0))); ie = len(tt) - 1
    o['A_raft_3'] = float(Ar[i3]); o['A_raft_6'] = float(Ar[i6]); o['A_raft_end'] = float(Ar[ie])
    o['refill_raft_3_6'] = float(Ar[i6] - Ar[i3])
    ifr = int(np.argmin(abs(tt - t_fr)))
    o['dA_after_stop_tr_6'] = float(Ar[i6] - Ar[ifr]); o['dA_after_stop_tr_end'] = float(Ar[ie] - Ar[ifr])
    o['umax_after_tr_mm_s'] = float(um[tt > t_fr + 0.02].max() * 1e3)
    o['umax_6_mm_s'] = float(um[i6] * 1e3); o['umax_end_mm_s'] = float(um[ie] * 1e3)
    o['max_dc_tr_6'] = float(np.abs(snaps[6.0] - snaps[t_fr]).max())
    o['max_dc_tr_end'] = float(np.abs(snaps[round(t_end, 3)] - snaps[t_fr]).max())
    # markers: far field (v3-min-2 T1 definitions at 6 s) and stop-flow drift [t_s + t_r, 6 s] and [.., t_end]
    for k, rr in enumerate(radii):
        sl = slice(k * 32, (k + 1) * 32)
        o[f'disp_{rr:.2f}_pre_white'] = float(np.hypot(*(mks[1.4][sl] - mk[sl]).T).max() / D)
        o[f'disp_{rr:.2f}_pour'] = float(np.hypot(*(mks[3.0][sl] - mk[sl]).T).max() / D)
        o[f'disp_{rr:.2f}_6'] = float(np.hypot(*(mks[6.0][sl] - mk[sl]).T).max() / D)
        o[f'drift_{rr:.2f}_tr_6'] = float(np.hypot(*(mks[6.0][sl] - mks[t_fr][sl]).T).max() / D)
        o[f'drift_{rr:.2f}_tr_end'] = float(np.hypot(*(mks[round(t_end, 3)][sl] - mks[t_fr][sl]).T).max() / D)
        dxm = (mks[round(t_end, 3)][sl, 0] - mks[t_fr][sl, 0]) / D
        o[f'drift_{rr:.2f}_mean_x_end'] = float(dxm.mean())
    o['far_disp_0.30'] = o['disp_0.30_6']; o['far_disp_0.30_before_white'] = o['disp_0.30_pre_white']
    o['t_fr'] = t_fr
    wc, _ = interface_width_cells(sim.c, g); o['iface_width_cells'] = float(wc)
    wc3, _ = interface_width_cells(snaps[3.0], g); o['iface_width_cells_3'] = float(wc3)
    ledg(sim, o)
    extra = dict(snaps=snaps, mks=mks, mk0=mk, radii=radii, ser=ser, t_fr=t_fr) if keep else None
    return o, extra


def hole_ins(c, g, r0, x0=0.0):
    """v3-min-2 sens2.py g3: inscribed radius of (brown U jet column), centres within r0 of x_j, 72 rays, dr 0.05 mm."""
    best = 0.0
    for dx in np.linspace(-r0, r0, 9):
        for dy in np.linspace(-r0, r0, 9):
            if dx * dx + dy * dy > r0 * r0: continue
            th, r, v = T.rays(c, g, x0 + dx, dy, nang=72, rmax=0.02, dr=5e-5)
            colmask = np.hypot(dx + r[None, :] * np.cos(th)[:, None], dy + r[None, :] * np.sin(th)[:, None]) < r0
            hit = (v >= 0.5) & ~colmask
            rf = np.where(hit.any(1), r[np.argmax(hit, 1)], 0.02)
            best = max(best, float(rf.min()))
    return best


# ============================================================================================ T2x
def wrap_span(old, new, g):
    """angular span (deg, 5 deg bins) of the old-white cells seen from the centroid of the new white."""
    if new.sum() == 0 or old.sum() == 0:
        return float('nan')
    cx, cy = g.X[new].mean(), g.Y[new].mean()
    ang = np.degrees(np.arctan2(g.Y[old] - cy, g.X[old] - cx))
    return float((np.histogram(ang, bins=72, range=(-180, 180))[0] > 0).sum() * 5)


def poly_span(Pm, new, g):
    if new.sum() == 0:
        return float('nan')
    cx, cy = g.X[new].mean(), g.Y[new].mean()
    ang = np.degrees(np.arctan2(Pm[:, 1] - cy, Pm[:, 0] - cx))
    return float((np.histogram(ang, bins=72, range=(-180, 180))[0] > 0).sum() * 5)


def T2x(var='v2', s0=0.15, travel=0.10, t_move=1.5, t_stop=4.0, t_end=10.0, N=128, keep=False, params=None, loop_re=None):
    g = fresh_grid(N)
    p = params if params is not None else P(var)
    sim = PXSim(grid=g, params=p)
    cx, rad = 0.12 * D, 0.165 * D
    sim.add_disc(cx, 0.0, rad, old=True)
    x_near0 = cx - rad; xs = x_near0 - s0 * D
    xf = lambda t: xs + travel * D * min(t, t_move) / t_move
    inp = T.jet_inp(xf, lambda t: 0.0, lambda t: T.Q_LOW if t < t_stop else 0.0, T.V_LOW)
    th = np.arange(720) * 2 * np.pi / 720
    mline = np.c_[cx + rad * np.cos(th), rad * np.sin(th)]
    # canvas material loop (DC.md 4, the l_s anchor): circle of radius 0.025 D centred halfway between the edge of the
    # jet's exclusion zone (x_j + r_e) and the near edge of the old block, at the start
    J0 = jet_derived(T.Q_LOW, T.V_LOW, sim.P)
    re_l = J0['re'] if loop_re is None else loop_re      # loop_re: fixed loop placement (independent of delta_e)
    xl = 0.5 * ((xs + re_l) + x_near0); rl = min(0.025 * D, 0.45 * (x_near0 - xs - re_l))
    th2 = np.arange(360) * 2 * np.pi / 360
    cloop = np.c_[xl + rl * np.cos(th2), rl * np.sin(th2)]
    sim.markers = np.r_[mline, cloop]; A0 = T._poly_area(mline); Al0 = T._poly_area(cloop)
    rec, snaps = {}, {}

    def cb(s, t):
        m = T.gap_metrics(s)
        Pm = s.markers[:720]; xc = T._poly_axis_cross(Pm)
        m['loop_area_ratio'] = float(T._poly_area(s.markers[720:]) / Al0)
        m['near'] = float(xc.min() / D - x_near0 / D) if len(xc) else float('nan')
        m['far'] = float(xc.max() / D - (cx + rad) / D) if len(xc) else float('nan')
        m['area_ratio'] = float(T._poly_area(Pm) / A0)
        try:
            m['solidity'] = float(T._poly_area(Pm) / ConvexHull(Pm).volume)
        except Exception:
            m['solidity'] = float('nan')
        ys = Pm[:, 1]; m['span_y'] = float((ys.max() - ys.min()) / D)
        xs_ = Pm[:, 0]; m['thick_x'] = float((xs_.max() - xs_.min()) / D)
        old, new = T.old_new_masks(s)
        m['wrap_old'] = wrap_span(old, new, g); m['wrap_mat'] = poly_span(Pm, new, g)
        m['jet_x'] = float(xf(t) / D)
        rec[round(t, 3)] = m
        if round(t, 3) in (1.0, 1.4, 2.0, 4.0, 6.0, 10.0):
            snaps[round(t, 3)] = (s.c.copy(), s.o.copy(), s.markers[:720].copy(), s.markers[720:].copy())
    t_pw = round(sim.P['tau_w'] - 0.1, 3)                 # just before the first white (t_k = 0)
    times = sorted(set([0.5, 1.0, 1.4, 1.5, 2.0, 2.5, 3.0, 4.0, 4.6, 5.0, 6.0, 8.0, 10.0] + ([t_pw] if loop_re is not None else [])))
    times = [t for t in times if t <= t_end + 1e-9]
    sim.run(inp, t_end, lambda t: 0.0, breakpoints=[t_move, t_stop], callback=cb, cb_times=times)
    o = dict(var=var, s0=s0, travel=travel)
    for t in (1.0, 1.4, 2.0, 4.0, 6.0, 10.0):
        if t not in rec: continue
        m = rec[t]; tg = f'{t:g}'
        o[f'near_{tg}'] = m['near']; o[f'far_{tg}'] = m['far']
        o[f'gap_min_{tg}'] = m['gap_min']; o[f'gap_axis_{tg}'] = m['gap_axis']; o[f'contact_{tg}'] = m['contact']
        o[f'sol_mat_{tg}'] = m['solidity']; o[f'sol_old_{tg}'] = m['old_solidity']
        o[f'wrap_mat_{tg}'] = m['wrap_mat']; o[f'wrap_old_{tg}'] = m['wrap_old']
        o[f'area_ratio_{tg}'] = m['area_ratio']; o[f'span_y_{tg}'] = m['span_y']; o[f'thick_x_{tg}'] = m['thick_x']
        o[f'new_area_{tg}'] = m['new_area']; o[f'old_raft_{tg}'] = m['old_raft_area']
        o[f'loop_ratio_{tg}'] = m['loop_area_ratio']
    if t_pw in rec:
        o['loop_ratio_pw'] = rec[t_pw]['loop_area_ratio']; o['near_pw'] = rec[t_pw]['near']; o['far_pw'] = rec[t_pw]['far']
        o['t_pw'] = t_pw
    if 1.4 in rec and abs(rec[1.4]['near']) > 1e-6:
        o['far_over_near_1.4'] = rec[1.4]['far'] / rec[1.4]['near']
    if 4.6 in rec and t_end >= 10.0:
        o['creep_near_4.6_10'] = rec[10.0]['near'] - rec[4.6]['near']
        o['creep_far_4.6_10'] = rec[10.0]['far'] - rec[4.6]['far']
    o['series'] = {f'{t:g}': {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec[t].items()
                              if k in ('near', 'far', 'gap_min', 'gap_axis', 'contact', 'solidity', 'old_solidity',
                                       'wrap_mat', 'wrap_old', 'area_ratio', 'new_area', 'span_y', 'thick_x',
                                       'loop_area_ratio')} for t in sorted(rec)}
    o['loop_r0'] = rl / D; o['loop_x0'] = xl / D
    wc, _ = interface_width_cells(sim.c, g); o['iface_width_cells'] = float(wc)
    ledg(sim, o)
    extra = dict(snaps=snaps, cx=cx, rad=rad, xs=xs) if keep else None
    return o, extra


# ============================================================================================ T3x
def T3x(var='v2', dwell=0.0, N=128, keep=False, params=None, t_end=5.0):
    """px_tests.T3 (gate 'lift') + a material line at x = 0 across the cut (V shape)."""
    g = fresh_grid(N)
    p = params if params is not None else P(var)
    sim = PXSim(grid=g, params=p)
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
    tk = lambda t: -100.0
    J = jet_derived(T.Q_THIN, T.V_THIN, sim.P)
    ys = np.linspace(-0.2, 0.2, 161) * D
    mk = np.c_[np.zeros_like(ys), ys]; sim.markers = mk.copy()
    t_x0 = (0.0 - x0) / V + dwell
    stations = np.array([-0.15, -0.05, 0.05, 0.15]) * D
    t_pass = np.array([(s - x0) / V + (dwell if (dwell > 0 and s > xd) else 0.0) for s in stations])
    t_imm = t_pass + 0.15
    t_post = t_pass + J['tr'] + 0.15
    times = sorted(set([round(x, 4) for x in list(t_imm) + list(t_post)] +
                       [round(t_cut, 4), round(t_x0 + 0.1, 4), round(t_cut + 3.0, 4), 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]))
    t_end = max(t_end, t_cut + 3.0)
    times = [t for t in times if t <= t_end + 1e-9]
    snaps, mks = {}, {}

    def cb(s, t):
        snaps[round(t, 4)] = s.c.copy(); mks[round(t, 4)] = s.markers.copy()
    sim.run(inp, t_end, tk, breakpoints=[t_cut, t_d0, t_d0 + dwell], callback=cb, cb_times=times)
    band = np.abs(g.x1) < 0.006

    def deficit(c, xs_):
        i = int(np.argmin(np.abs(g.x1 - xs_)))
        prof = 0.5 * (c[i, :] + c[i + 1 if g.x1[i] < xs_ else i - 1, :])
        return float(((1 - prof[band]).sum() * g.h) / D), prof
    o = dict(var=var, dwell=dwell, two_rd=2 * J['rd'] / D)
    di, dfin, vw = [], [], []
    tE = round(t_end, 4)
    for s_, a in zip(stations, t_imm):
        di.append(deficit(snaps[round(a, 4)], s_)[0])
        d, prof = deficit(snaps[tE], s_); dfin.append(d)
        vw.append(float((prof[band] < 0.5).sum() * g.h / D))
    o['line_def_imm'] = float(np.mean(di)); o['line_def_final'] = float(np.mean(dfin))
    o['line_vis_width_final'] = float(np.mean(vw))
    cF = snaps[tE]
    r = np.hypot(g.X, g.Y)
    outside = (r > rad + 1.5e-3) & g.inside
    for side, msk in (('entry', g.X < -rad - 1.5e-3), ('exit', g.X > rad + 1.5e-3)):
        m = outside & msk & (np.abs(g.Y) < 0.006)
        A = (cF * m).sum() * g.cellA
        Lx = (np.abs(g.x1) > rad + 1.5e-3) & ((g.x1 < 0) if side == 'entry' else (g.x1 > 0)) & (np.abs(g.x1) < 0.35 * D)
        o[f'stem_width_{side}'] = float(A / max(Lx.sum() * g.h, 1e-12) / D)
    o['disc_area_change_cm2'] = float((cF.sum() * g.cellA - np.pi * rad ** 2) * 1e4)
    th = np.radians(np.arange(-40, 40.1, 1.0))
    rr = np.arange(0.15 * D, 0.40 * D, 1e-4)

    def rb(c, th0):
        X = rr[None, :] * np.cos(th0 + th)[:, None]; Y = rr[None, :] * np.sin(th0 + th)[:, None]
        ok = X ** 2 + Y ** 2 < (R - g.h) ** 2
        v = T.sample_bilinear(c, g, X, Y) * ok
        return np.array([rr[np.nonzero(row >= 0.5)[0].max()] if (row >= 0.5).any() else np.nan for row in v])
    for name, th0 in (('exit', 0.0), ('entry', np.pi)):
        b = rb(cF, th0)
        keep_ = np.abs(rad * np.sin(th)) > 2.5e-3
        dev = (b - rad) / D
        o[f'{name}_bulge_max'] = float(np.nanmax(dev[keep_])); o[f'{name}_bdev_min'] = float(np.nanmin(dev[keep_]))
        o[f'{name}_bdev_profile'] = [float(x) for x in dev]
    near = (np.hypot(g.X + rad, g.Y) < 0.008) & (r < rad) & g.inside
    brown = ((1 - cF) * near).sum() * g.cellA
    A_notch = max(brown - o['line_def_final'] * D * 0.008, 0.0)
    o['notch_area_mm2'] = float(A_notch * 1e6); o['notch_req'] = float(math.sqrt(2 * A_notch / math.pi) / D)
    # V shape: forward drag of the material line x = 0 vs |y|, right after passage, at cut end, 3 s after
    vs = {}
    for lab, t_ in (('pass+0.1', round(t_x0 + 0.1, 4)), ('cut_end', round(t_cut, 4)), ('cut+3', round(t_cut + 3.0, 4))):
        if t_ not in mks: continue
        dx = (mks[t_][:, 0] - mk[:, 0]) / D
        prof = {}
        for yy in (0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.15):
            i = int(np.argmin(np.abs(ys / D - yy))); j = int(np.argmin(np.abs(ys / D + yy)))
            prof[f'{yy:g}'] = float(0.5 * (dx[i] + dx[j]))
        # half-width of the drag profile: |y| at which the drag falls to half its value at |y| = 0.005 D
        yy_ = np.abs(ys / D); dxs = dx
        cen = prof['0.005']
        hw = float('nan')
        if cen > 1e-4:
            order = np.argsort(yy_)
            yq, dq = yy_[order], dxs[order]
            below = np.nonzero((dq < 0.5 * cen) & (yq > 0.005))[0]
            hw = float(yq[below[0]]) if len(below) else float('nan')
        prof['half_width'] = hw
        vs[lab] = prof
    o['vshape'] = vs
    wc, _ = interface_width_cells(sim.c, g); o['iface_width_cells'] = float(wc)
    o['core_cm2'] = sim.L['core'] * 1e4
    ledg(sim, o)
    extra = dict(snaps=snaps, mks=mks, mk0=mk, rad=rad, t_cut=t_cut, t_end=t_end) if keep else None
    return o, extra


# ============================================================================================ T4x
def T4x(var='v2', base=False, N=128, keep=False, params=None, t_pre=1.5, amp=4e-3, freq=3.0, Vr=0.15, t_w=2.0,
        t_after=5.0, base_cx=0.0, base_rad=0.25):
    g = fresh_grid(N)
    p = params if params is not None else P(var)
    sim = PXSim(grid=g, params=p)
    if base:
        sim.add_disc(base_cx * D, 0.0, base_rad * D, old=True)
    x0 = 0.15 * D
    xf = lambda t: x0 if t < t_pre else x0 - Vr * D * min(t - t_pre, t_w)
    yf = lambda t: 0.0 if t < t_pre else amp * math.sin(2 * math.pi * freq * min(t - t_pre, t_w))
    t_stop = t_pre + t_w; t_end = t_stop + t_after
    inp = T.jet_inp(xf, yf, lambda t: T.Q_LOW if t < t_stop else 0.0, T.V_LOW)
    snaps = {}
    times = [t_pre, t_pre + 0.5, t_pre + 1.0, t_pre + 1.5, t_stop, t_stop + 1.0, t_end]

    def cb(s, t):
        snaps[round(t, 4)] = s.c.copy()
    sim.run(inp, t_end, lambda t: 0.0, breakpoints=[t_pre, t_stop], callback=cb, cb_times=times)
    x_lo, x_hi = x0 - Vr * D * t_w - 0.02 * D, x0 + 0.06 * D
    s_end = T.stripe_metrics(snaps[round(t_stop, 4)], g, x_lo, x_hi)
    s_fin = T.stripe_metrics(snaps[round(t_end, 4)], g, x_lo, x_hi)
    o = dict(var=var, base=base, n_stripes=s_end['n'], pitch=s_end['pitch'], def_mean=s_end['def_mean'],
             n_enclosed=s_end['n_enclosed'], enclosed_area_mm2=s_end['enclosed_area_mm2'],
             n_stripes_final=s_fin['n'], pitch_final=s_fin['pitch'], def_mean_final=s_fin['def_mean'],
             n_enclosed_final=s_fin['n_enclosed'], stripe_x=s_end['x'], deficits=s_end['deficits'],
             cycles=freq * t_w, target_pitch=Vr / freq)
    for tag, tt in (('end', t_stop), ('fin', t_end)):
        cS = snaps[round(tt, 4)]
        for yD in (0.0, 0.02, -0.02, 0.04, -0.04):
            lc = line_counts(cS, g, x_lo, x_hi, yD)
            key = f'{tag}_y{yD:+.2f}'
            o[key + '_n'] = lc['n']; o[key + '_pitch'] = lc['pitch']; o[key + '_w'] = lc['width_mean']; o[key + '_x'] = lc['x']
    ns = [o[f'end_y{y:+.2f}_n'] for y in (0.0, 0.02, -0.02, 0.04, -0.04)]
    nf = [o[f'fin_y{y:+.2f}_n'] for y in (0.0, 0.02, -0.02, 0.04, -0.04)]
    o['n_lines_median'] = float(np.median(ns)); o['n_lines_max'] = int(max(ns)); o['n_lines_median_fin'] = float(np.median(nf))
    o['survival'] = (s_fin['n'] / s_end['n']) if s_end['n'] else float('nan')
    o['max_abs_dc_after'] = float(np.abs(snaps[round(t_end, 4)] - snaps[round(t_stop + 1.0, 4)]).max())
    o['A_vis_end'] = white_area(sim, 0.5) * 1e4
    mixed = ((sim.c > 0.05) & (sim.c < 0.95) & g.inside).sum()
    o['mixed_frac_of_white'] = float(mixed / max((sim.c >= 0.5).sum(), 1))
    wc, _ = interface_width_cells(sim.c, g); o['iface_width_cells'] = float(wc)
    ledg(sim, o)
    extra = dict(snaps=snaps, s_end=s_end, s_fin=s_fin, x_lo=x_lo, x_hi=x_hi, t_stop=t_stop, t_end=t_end,
                 path=[(xf(t), yf(t)) for t in np.arange(0, t_stop, 0.005)]) if keep else None
    return o, extra


def line_counts(c, g, x_lo, x_hi, yD, halfband=1.25e-3, prom=0.3):
    """(proto_DQ dq_tests.line_counts) brown lines = minima of c flanked by white c >= 0.5 on both sides, along y = yD*D."""
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
