"""Figures for test_Qquasist.md (written next to it in review/jetpush/).  usage: python3 -I make_figs.py [N]"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

D = 0.08; R = D / 2
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
OUT = os.path.join(HERE, 'out'); FIG = os.path.dirname(HERE)
BROWN = np.array([0.42, 0.25, 0.15]); WHITE = np.array([0.98, 0.97, 0.94]); OLD = np.array([0.96, 0.72, 0.42])
cmap_c = LinearSegmentedColormap.from_list('latte', [BROWN, WHITE])
x1 = (np.arange(N) + 0.5) * D / N - R
EXT = [x1[0] / D, x1[-1] / D, x1[0] / D, x1[-1] / D]
INSIDE = (x1[:, None] ** 2 + x1[None, :] ** 2) < R ** 2


def load_log(path):
    out = {}
    if os.path.exists(path):
        for l in open(path):
            d = json.loads(l); out[(d['test'], d['var'])] = d['obs']
    return out


LOG = load_log(os.path.join(HERE, 'logs', f'A_{N}.jsonl'))


def rgb(c, o=None):
    c = np.clip(c, 0, 1)
    img = BROWN[None, None, :] * (1 - c[..., None]) + WHITE[None, None, :] * c[..., None]
    if o is not None:
        f = ((o >= 0.5 * c) & (c >= 0.5)).astype(float)       # old = same classification as the gap metric
        img = img * (1 - f[..., None]) + OLD[None, None, :] * f[..., None]
    img[~INSIDE] = 1.0
    return np.transpose(img, (1, 0, 2))      # imshow: rows = y


def show(ax, c, o=None, lim=None, title=None):
    ax.imshow(rgb(c, o), origin='lower', extent=EXT, interpolation='nearest')
    if lim: ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=8)


def fig_T1():
    fig, axs = plt.subplots(2, 5, figsize=(15, 6.4))
    lim = (-0.16, 0.16, -0.16, 0.16)
    for r, var in enumerate(('v2', 'q')):
        z = np.load(os.path.join(OUT, f'T1_{var}_{N}.npz'))
        o = LOG.get(('T1', var), {})
        for k, t in enumerate((1.6, 2.0, 3.0, 6.0)):
            show(axs[r, k], z[f'c{t}'], lim=lim, title=f'{"v3-min-2" if var == "v2" else "Q"}  t = {t:g} s')
            axs[r, k].plot([0], [0], '+', color='c', ms=6)
        m0, m1 = z['markers0'], z['markers']
        ax = axs[r, 4]
        show(ax, z['c6.0'], lim=(-0.5, 0.5, -0.5, 0.5),
             title=f'6 s, markers r=0.30-0.45 D (x5 disp.)\nopen3 {o.get("open_deg_3")} deg, specks3 {o.get("specks_n_3")}')
        d = (m1 - m0) * 5
        ax.quiver(m0[:, 0] / D, m0[:, 1] / D, d[:, 0] / D, d[:, 1] / D, angles='xy', scale_units='xy', scale=1, color='tab:blue',
                  width=0.004)
    fig.suptitle('T1: fixed low pour 0-3 s, stop, observe to 6 s (128^2, e = +x; pitcher on the -x side)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, 'test_Qquasist_T1.png'), dpi=110); plt.close(fig)


def fig_T2():
    cases = ['T2h0.10', 'T2h0.15', 'T2h0.25', 'T2m0.10', 'T2m0.15', 'T2m0.25']
    fig, axs = plt.subplots(len(cases), 5, figsize=(15, 3.0 * len(cases)))
    lim = (-0.42, 0.42, -0.30, 0.30)
    for r, case in enumerate(cases):
        for k, (var, t) in enumerate((('v2', 4.0), ('q', 1.4), ('q', 2.0), ('q', 4.0), ('q', 6.0))):
            p = os.path.join(OUT, f'{case}_{var}_{N}.npz')
            if not os.path.exists(p): continue
            z = np.load(p)
            o = LOG.get((case, var), {})
            ttl = f'{case} {"v3-min-2" if var == "v2" else "Q"} t={t:g}'
            if t in (4.0, 6.0) or t == 2.0:
                g = o.get(f'gap_min_{t:.0f}') if t in (2.0, 4.0, 6.0) else None
                ttl += f'\ngap {g if g is None else round(g, 3)}  sol {o.get("mat_solidity_4") and round(o.get("mat_solidity_4"), 2)}'
            show(axs[r, k], z[f'c{t}'], z[f'o{t}'], lim=lim, title=ttl)
            m = z[f'm{t}']
            axs[r, k].plot(np.r_[m[:, 0], m[0, 0]] / D, np.r_[m[:, 1], m[0, 1]] / D, '-', color='tab:blue', lw=0.6)
            trav = 0.10 if case[2] == 'm' else 0.0
            xj = float(z['xs']) + trav * D * min(t, 1.5) / 1.5
            axs[r, k].plot([xj / D], [0], 'x', color='c', ms=6)
    fig.suptitle('T2 push: old 0.33 D disc (orange), new white (white), material line of the old disc (blue), jet (x).\n'
                 'T2h = jet held, T2m = jet moves +0.10 D in 1.5 s; number = stand-off s0 behind the rear edge [D]', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, 'test_Qquasist_T2.png'), dpi=100); plt.close(fig)
    # gap / displacement curves
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for case, ls in (('T2h0.15', '-'), ('T2h0.25', '--'), ('T2m0.15', ':'), ('T2m0.25', '-.')):
        for var, col in (('v2', 'tab:gray'), ('q', 'tab:red')):
            o = LOG.get((case, var))
            if not o: continue
            ser = o['series']
            tt = sorted(float(t) for t in ser)
            g = [ser[f'{t:g}' if f'{t:g}' in ser else str(t)]['gap_min'] for t in tt]
            g = [np.nan if v is None else v for v in g]
            axs[0].plot(tt, g, ls, color=col, label=f'{case} {var}')
            nx = [ser[f'{t:g}' if f'{t:g}' in ser else str(t)]['mat_near_x'] - o['near0'] for t in tt]
            axs[1].plot(tt, nx, ls, color=col, label=f'{case} {var}')
            so = [ser[f'{t:g}' if f'{t:g}' in ser else str(t)]['mat_solidity'] for t in tt]
            axs[2].plot(tt, so, ls, color=col, label=f'{case} {var}')
    axs[0].axhspan(0.04, 0.13, color='g', alpha=0.12, label='video gap band'); axs[0].set_ylabel('gap old-new white [D]')
    axs[1].axhspan(-0.03, 0.03, color='g', alpha=0.12); axs[1].set_ylabel('near-edge displacement [D]')
    axs[2].axhline(0.85, color='g', lw=0.8); axs[2].set_ylabel('material-line solidity')
    for a in axs: a.set_xlabel('t [s]'); a.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, 'test_Qquasist_T2curves.png'), dpi=110); plt.close(fig)


def fig_T3():
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.4))
    for r, test in enumerate(('T3', 'T3d')):
        for k, var in enumerate(('v2', 'q')):
            z = np.load(os.path.join(OUT, f'{test}_{var}_{N}.npz'))
            o = LOG.get((test, var), {})
            show(axs[r, 2 * k], z['c5.0'], lim=(-0.30, 0.30, -0.30, 0.30),
                 title=f'{test} {"v3-min-2" if var == "v2" else "Q"} 5 s: line def {o.get("line_def_final", 0):.4f} D, notch {o.get("notch_req", 0):.3f} D')
            m = z['m5.0']; m0 = z['mk0']; n = int(z['nline'])
            for s in range(4):
                axs[r, 2 * k].plot(m[s * n:(s + 1) * n, 0] / D, m[s * n:(s + 1) * n, 1] / D, '-', color='tab:blue', lw=0.6)
            show(axs[r, 2 * k + 1], z['c5.0'], lim=(-0.33, -0.17, -0.08, 0.08),
                 title=f'entry zoom: bulge(10-15deg) {o.get("entry_bulge_10_15", 0):+.3f} D, V {o.get("V_amp", 0):.4f} D')
            for s in range(4):
                axs[r, 2 * k + 1].plot(m[s * n:(s + 1) * n, 0] / D, m[s * n:(s + 1) * n, 1] / D, '-', color='tab:blue', lw=0.6)
            th = np.linspace(0, 2 * np.pi, 400)
            for a in (axs[r, 2 * k], axs[r, 2 * k + 1]):
                a.plot(0.25 * np.cos(th), 0.25 * np.sin(th), ':', color='k', lw=0.5)
    fig.suptitle('T3 cut: thin stream (3 mL/s, 1.2 m/s) at 0.4 D/s along +x across a 0.5 D white disc; top no dwell, bottom 0.7 s dwell at the entry.\n'
                 'Blue: marker lines x = -0.15, -0.05, 0.05, 0.15 D (V shape); dotted: original disc', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, 'test_Qquasist_T3.png'), dpi=100); plt.close(fig)


def fig_T4():
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.6))
    lim = (-0.30, 0.30, -0.22, 0.22)
    for r, test in enumerate(('T4f', 'T4b')):
        for k, var in enumerate(('v2', 'q')):
            z = np.load(os.path.join(OUT, f'{test}_{var}_{N}.npz'))
            o = LOG.get((test, var), {})
            for j, t in enumerate((3.5, 8.5)):
                ax = axs[r, 2 * k + j]
                show(ax, z[f'c{t}'], lim=lim, title=f'{test} {"v3-min-2" if var == "v2" else "Q"} t={t:g} s '
                     f'({"stop" if t == 3.5 else "stop+5 s"})\nlines (y=0,+-0.02,+-0.04 D) '
                     + ','.join(str(o.get(f'{"end" if t == 3.5 else "fin"}_y{y:+.2f}_n')) for y in (0.0, 0.02, -0.02, 0.04, -0.04))
                     + f'; islands {o.get("end_n_enclosed" if t == 3.5 else "fin_n_enclosed")}')
                p = z['path']
                if t == 3.5:
                    ax.plot(p[:, 0] / D, p[:, 1] / D, '-', color='c', lw=0.4)
    fig.suptitle('T4 wiggle: pre-pour 1.5 s, then 4 mm x 3 Hz wiggle while retreating 0.15 D/s (-x) for 2 s; top fresh canvas, '
                 'bottom on a 0.5 D white base', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, 'test_Qquasist_T4.png'), dpi=100); plt.close(fig)


def fig_flow():
    """velocity field at t = 2.5 s of T1 (Q): area-source part grad(phi), projected footprint part P[W], total u."""
    import math
    import px_tests as T
    import dq_tests  # noqa: F401  (patched finish)
    from px_model import PXSim, VQ, jet_derived
    g = T.grid(N)
    sim = PXSim(grid=g, params=dict(VQ))
    inp = T.jet_inp(lambda t: 0.0, lambda t: 0.0, lambda t: T.Q_LOW, T.V_LOW)
    sim.run(inp, 2.5, lambda t: 0.0)
    ux, uy = sim.ux, sim.uy
    J = jet_derived(T.Q_LOW, T.V_LOW, sim.P)
    ut = J['v'] * math.sin(math.radians(sim.P['alpha_j']))
    i0, i1, j0, j1, K = sim.kernel(0.0, 0.0, 'ann', 0.0, J['r0'])
    W = np.zeros((N, N)); W[i0:i1, j0:j1] = ut * K * math.pi * J['r0'] ** 2
    Ufx = np.zeros((N + 1, N)); Ufy = np.zeros((N, N + 1))
    Ufx[1:-1, :] = 0.5 * (W[1:, :] + W[:-1, :]); Ufx[~g.mx] = 0.0
    psi, _ = g.solve(-g.div(Ufx, Ufy) * g.inside)
    wx, wy = g.face_vel(psi); wx = wx + Ufx                       # P[W] (zero normal flux at the wall)
    vx, vy = ux - wx, uy - wy                                       # grad(phi): area-source part
    to_c = lambda a, b: (0.5 * (a[1:, :] + a[:-1, :]), 0.5 * (b[:, 1:] + b[:, :-1]))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (U, V, ttl) in zip(axs, (to_c(vx, vy) + ('area-source part grad(phi)',),
                                    to_c(wx, wy) + ('footprint part P[W] (potential dipole outside r_0)',),
                                    to_c(ux, uy) + ('Q total u = grad(phi) + P[W]',))):
        show(ax, sim.c, lim=(-0.16, 0.16, -0.16, 0.16), title=ttl + '\n(t = 2.5 s, background = Q state; arrows: speed, log colour)')
        sp = np.hypot(U, V) * INSIDE
        lw = 0.4 + 1.6 * np.clip(np.log10(np.maximum(sp, 1e-5) / 1e-3) / 2, 0, 1)
        ax.streamplot(x1 / D, x1 / D, (U * INSIDE).T, (V * INSIDE).T, density=1.4, color=np.log10(np.maximum(sp, 1e-5)).T,
                      cmap='viridis', linewidth=lw.T, arrowsize=0.7)
        th = np.linspace(0, 2 * np.pi, 200)
        for rr, cc in ((J['r0'], 'c'), (J['re'], 'm'), (1.5 * J['re'], 'm')):
            ax.plot(rr * np.cos(th) / D, rr * np.sin(th) / D, '--', color=cc, lw=0.7)
        ax.set_xlim(-0.16, 0.16); ax.set_ylim(-0.16, 0.16)
    fig.suptitle('Fixed low pour (e = +x, pitcher at -x): r_0 (cyan), r_e and 1.5 r_e (magenta; supply ring centred at x_j + Delta_e). '
                 'm = %.2e m^3/s, S = wQ/Gamma = %.2e m^2/s' % (ut * math.pi * J['r0'] ** 2, T.Q_LOW / sim.P['Gamma']), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, 'test_Qquasist_flow.png'), dpi=100); plt.close(fig)


def fig_ident():
    R = json.load(open(os.path.join(HERE, 'sens', 'fisherQ.json')))
    vars_ = [v for v in ('v2', 'q') if v in R]
    fig, axs = plt.subplots(2, len(vars_), figsize=(7.5 * len(vars_), 13), gridspec_kw=dict(height_ratios=[2.2, 1]))
    if len(vars_) == 1: axs = axs[:, None]
    for k, v in enumerate(vars_):
        r = R[v]; A = np.array(r['A']); rows = r['rows']; cols = r['cols']
        ax = axs[0, k]
        vmax = 5
        im = ax.imshow(np.clip(A, -vmax, vmax), cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{o} ({r["nominal"].get(o, float("nan")):.3g})' for o in rows], fontsize=7)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f'{A[i, j]:.1f}', ha='center', va='center', fontsize=6)
        ax.axvline(3.5, color='k', lw=1)
        ax.set_title(f'{"v3-min-2" if v == "v2" else "variant Q"}: dO/dln(theta)/sigma (+-20 %; nuisances right of the bar)', fontsize=9)
        ax = axs[1, k]
        txt = []
        for sname, sr in r['sets'].items():
            cos = np.array(sr['cos'])
            ab = ['G', 'tw', 're', 'Pi']
            txt.append(f'{sname}: col-norm cond {sr["cond"]:.3g} (raw {sr["cond_raw"]:.3g}); |col| ' + ' '.join(f'{x:.2f}' for x in sr['norms'])
                       + (f'  NOT anchored: {sr["weak"]}' if sr['weak'] else ''))
            txt.append('   cos ' + ' '.join(f'{ab[a]}-{ab[b]} {cos[a, b]:+.2f}' for a in range(4) for b in range(a + 1, 4)))
            txt.append('   sd(ln) meas ' + ' '.join(f'{x:.2g}' for x in sr['sd']) + ' | +nuis ' + ' '.join(f'{x:.2g}' for x in sr['sdz']))
        ax.axis('off'); ax.text(0, 1, '\n'.join(txt), va='top', family='monospace', fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'test_Qquasist_ident.png'), dpi=100); plt.close(fig)


def fig_res():
    """128 vs 192: T1 (3 s), T2 moving s0 = 0.15 (4 s), T3d (5 s), T4 fresh (stop), variant Q."""
    global x1, EXT, INSIDE
    cases = [('T1', 'c3.0', None, (-0.16, 0.16, -0.16, 0.16)), ('T2m0.15', 'c4.0', 'o4.0', (-0.42, 0.42, -0.30, 0.30)),
             ('T3d', 'c5.0', None, (-0.33, -0.13, -0.10, 0.10)), ('T4f', 'c3.5', None, (-0.30, 0.30, -0.22, 0.22))]
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5))
    for r, n in enumerate((128, 192)):
        x1 = (np.arange(n) + 0.5) * D / n - R
        EXT = [x1[0] / D, x1[-1] / D, x1[0] / D, x1[-1] / D]
        INSIDE = (x1[:, None] ** 2 + x1[None, :] ** 2) < R ** 2
        for k, (test, key, okey, lim) in enumerate(cases):
            p = os.path.join(OUT, f'{test}_q_{n}.npz')
            if not os.path.exists(p): continue
            z = np.load(p)
            show(axs[r, k], z[key], z[okey] if okey else None, lim=lim, title=f'Q {test} {key[1:]} s, {n}^2')
    fig.suptitle('Variant Q at 128^2 (top) and 192^2 (bottom)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(FIG, 'test_Qquasist_res.png'), dpi=100); plt.close(fig)


if __name__ == '__main__':
    which = sys.argv[2].split(',') if len(sys.argv) > 2 else ['T1', 'T2', 'T3', 'T4', 'flow', 'ident']
    for w in which:
        globals()['fig_' + w]()
        print('fig', w, 'done', flush=True)
