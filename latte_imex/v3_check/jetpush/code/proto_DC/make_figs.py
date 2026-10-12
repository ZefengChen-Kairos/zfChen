"""Figures for test_Ccompres.md (PNG next to the report).  usage: python3 -I make_figs.py [unit,T1,T2,T2gap,T3,T4]"""
import os, sys, json, glob, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = os.path.join(HERE, 'out')
FIG = os.path.dirname(HERE)
D = 0.08; R = D / 2
CMAP = LinearSegmentedColormap.from_list('latte', ['#5a3420', '#f7f1e3'])
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e4e2dc'; SURF = '#fcfcfb'
plt.rcParams.update({'font.family': ['DejaVu Sans', 'WenQuanYi Zen Hei'], 'axes.unicode_minus': False, 'font.size': 8.5,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'lines.linewidth': 2.0, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'savefig.facecolor': SURF})
LAB = {'v2': 'v3-min-2', 'q': 'Q (ℓ_s=∞)', 'sv20': 'C-V ℓ_s=20 cm', 'sv4': 'C-V ℓ_s=4 cm', 'sv2': 'C-V ℓ_s=2 cm', 'sv1': 'C-V ℓ_s=1 cm',
       'sv05': 'C-V ℓ_s=0.5 cm', 'sv1n': 'C-V 1 cm 无 W', 'sv2n': 'C-V 2 cm 无 W', 'sv05n': 'C-V 0.5 cm 无 W', 'cd1e3': 'C-D 1e-3',
       'cd1e4': 'C-D D_Π=1e-4', 'cd3e5': 'C-D D_Π=3e-5', 'cd1e4n': 'C-D 1e-4 无 W', 'cp': 'C-P (ℓ_c=1 mm)', 'cpn': 'C-P 无 W'}
COL = {'v2': INK2, 'q': CAT[0], 'sv4': CAT[2], 'sv2': CAT[3], 'sv1': CAT[1], 'sv05': CAT[7], 'cd1e4': CAT[6], 'cd3e5': CAT[4], 'cp': CAT[5],
       'sv1n': '#b4b2ab'}


def load(pattern='scan_*128*.jsonl'):
    R_ = {}
    for fn in sorted(glob.glob(os.path.join(OUT, pattern))):
        for l in open(fn):
            r = json.loads(l); R_[(r['test'], r['var'], r['arg'], r['N'])] = r['obs']
    return R_


def snap(name):
    return np.load(os.path.join(OUT, 'snap_' + name + '.npz'))


def key(z, prefix, t):
    ks = [k for k in z.files if k.startswith(prefix) and k[len(prefix):].replace('.', '', 1).isdigit()]
    return min(ks, key=lambda k: abs(float(k[len(prefix):]) - t))


def show_c(ax, c, lim=None, title=None, o=None):
    ext = [-0.5, 0.5, -0.5, 0.5]
    if o is None:
        ax.imshow(np.clip(c, 0, 1).T, origin='lower', extent=ext, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest')
    else:   # old white tinted blue-grey, new white cream, canvas brown
        cc = np.clip(c, 0, 1); fo = np.where(cc > 1e-9, np.clip(o, 0, None) / np.maximum(cc, 1e-9), 0.0)
        brown = np.array([0x5a, 0x34, 0x20]) / 255; cream = np.array([0xf7, 0xf1, 0xe3]) / 255; old = np.array([0xa9, 0xc4, 0xe0]) / 255
        white = cream[None, None] * (1 - fo[..., None]) + old[None, None] * fo[..., None]
        img = brown[None, None] * (1 - cc[..., None]) + white * cc[..., None]
        ax.imshow(np.transpose(img, (1, 0, 2)), origin='lower', extent=ext, interpolation='nearest')
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(0.5 * np.cos(th), 0.5 * np.sin(th), color=INK2, lw=0.8)
    if lim: ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    ax.set_aspect('equal'); ax.grid(False); ax.tick_params(labelsize=6.5)
    if title: ax.set_title(title, fontsize=7.5, color=INK)


def fig_unit():
    from px_model import Grid, PXSim
    from scipy.special import k1
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8))
    ax = axs[0]
    N = 128; g = Grid(N); sim = PXSim(grid=g); S = 3.8e-4
    sig = np.zeros((N, N)); kw = sim.kernel(0.0, 0.0, 'ann', 0.0, 1.0e-3); sim._add(sig, kw, S)
    rr = np.linspace(0.002, 0.036, 120)
    for k, ell in enumerate((0.005, 0.01, 0.02)):
        kap2 = np.ones((N, N)) * g.inside / ell ** 2
        phi, _ = g.solve_helm(sig * g.inside, kap2); ux, uy = g.face_vel(phi)
        rs = np.arange(0.003, 0.036, 0.0015); vx, _ = g.interp(ux, uy, rs, 0 * rs)
        ax.plot(rr / D, S / (2 * np.pi) * k1(rr / ell) / ell * 1e3, color=CAT[k], lw=1.6, label=f'ℓ_s = {ell*100:g} cm：S K₁(r/ℓ_s)/(2πℓ_s)')
        ax.plot(rs / D, vx * 1e3, 'o', ms=3.5, color=CAT[k], mec=SURF, mew=0.6)
    ax.plot(rr / D, S / (2 * np.pi * rr) * 1e3, '--', color=INK2, lw=1.2, label='等压 S/(2πr)')
    ax.set_yscale('log'); ax.set_ylim(1e-3, 40); ax.set_xlabel('r / D'); ax.set_ylabel('u_r (mm/s)')
    ax.set_title('T-C0：空杯屏蔽点源（线 = 解析，点 = 128²）', fontsize=8.5); ax.legend(fontsize=7)
    # Proposition G: gap ratio gap(l_s)/gap(isobaric) <= 1, radial model vs prototype
    ax = axs[1]
    from scipy.special import k1 as K1
    psi = lambda x: x * K1(x) if x > 0 else 1.0
    Rr = load()

    def ode(ell, rb0, R0=0.0065, t_end=2.6):
        R2, rb2, t, dt = R0 ** 2, rb0 ** 2, 0.0, 1e-3
        while t < t_end - 1e-9:
            Rn = math.sqrt(R2); rb = math.sqrt(rb2)
            pr = psi(Rn / ell) if ell else 1.0; fb = psi(rb / ell) / pr if ell else 1.0
            R2 += 3.8e-4 / math.pi * dt; rb2 = max(rb2 + 3.8e-4 / math.pi * fb * dt, R2); t += dt
        return math.sqrt(rb2) - math.sqrt(R2)
    ells = np.array([0.005, 0.007, 0.01, 0.014, 0.02, 0.028, 0.04, 0.06, 0.1])
    for k, (rb0, lab) in enumerate(((0.022, 's₀=0.25 D'), (0.014, 's₀=0.15 D'))):
        g_inf = ode(None, rb0)
        ax.plot(ells * 100, [ode(e, rb0) / g_inf for e in ells], color=CAT[k], lw=1.6, label=f'径向模型 {lab}（出白后 2.6 s）')
    pts = [('0.25', 4.0, CAT[0], 'o', '原型 s₀=0.25 D，4 s'), ('0.25', 6.0, CAT[0], 's', '原型 s₀=0.25 D，6 s'),
           ('0.15', 2.0, CAT[1], 'o', '原型 s₀=0.15 D，2 s')]
    for s0, t, col, mk, lab in pts:
        gq = Rr[('T2h', 'q', s0, 128)][f'gap_min_{t:g}']
        xs, ys = [], []
        for var, ell in (('sv4', 4), ('sv2', 2), ('sv1', 1), ('sv05', 0.5)):
            o = Rr.get(('T2h', var, s0, 128))
            if o: xs.append(ell); ys.append(o[f'gap_min_{t:g}'] / gq)
        ax.plot(xs, ys, mk, ms=5, color=col, mec=SURF, mew=0.6, label=lab + f'（Q：{gq:.3f} D）')
    ax.axhline(1.0, color=INK2, lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_xlabel('ℓ_s (cm)'); ax.set_ylabel('缝(ℓ_s) / 缝(等压)'); ax.set_ylim(-0.05, 1.15)
    ax.set_title('命题 G：可压只让缝变窄（比值 ≤ 1，定点推挤）', fontsize=8.5); ax.legend(fontsize=6.5, loc='lower right')
    # stop-flow umax
    ax = axs[2]
    for var in ('v2', 'q', 'sv1', 'sv05', 'cd1e4', 'cd3e5', 'cp'):
        o = Rr.get(('T1', var, None, 128))
        if not o: continue
        ax.plot(o['series_t'], np.array(o['series_umax']) * 1e3, color=COL[var], lw=1.4, label=LAB[var])
    ax.axvline(3.0, color=INK2, lw=0.8, ls=':'); ax.text(3.05, 0.5, '停倒', fontsize=7, color=INK2)
    ax.set_yscale('symlog', linthresh=0.1); ax.set_ylim(0, 2000); ax.set_xlabel('t (s)'); ax.set_ylabel('max|u| (mm/s)，0.1 以下线性')
    ax.set_title('T1 停倒后冻结（引理 F）：C-V、C-P 精确为 0，C-D 回弹', fontsize=8.5); ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'test_Ccompres_unit.png'), dpi=150); plt.close(fig)


def fig_T1():
    vs = ['v2', 'q', 'sv2', 'sv1', 'sv05', 'cd1e4', 'cp']
    fig, axs = plt.subplots(2, len(vs), figsize=(2.0 * len(vs), 4.4))
    for j, v in enumerate(vs):
        z = snap(f'T1_{v}_N128')
        for i, t in enumerate((3.0, 13.0)):
            ax = axs[i, j]
            show_c(ax, z[key(z, 'c', t)], (-0.25, 0.25, -0.25, 0.25), LAB[v] if i == 0 else None)
            if j == 0: ax.set_ylabel(f't = {t:g} s', fontsize=8)
            m0 = z['mk0'] / D; m = z[key(z, 'm', t)] / D
            sel = slice(0, 64)
            ax.plot(m[sel, 0], m[sel, 1], 'o', ms=1.4, color=CAT[0])
    fig.suptitle('T1：杯心低位倒 3 s（壶在 −x 侧，ê = +x），3 s 与 13 s；蓝点 = 0.10/0.15 D 标记环', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'test_Ccompres_T1.png'), dpi=150); plt.close(fig)


def fig_T2():
    vs = ['v2', 'q', 'sv2', 'sv1', 'sv05', 'cd1e4', 'cp']
    cases = [('T2h', '0.15', 4.0, '定点 s₀=0.15 D，4 s'), ('T2h', '0.15', 10.0, '定点 s₀=0.15 D，10 s'),
             ('T2h', '0.25', 6.0, '定点 s₀=0.25 D，6 s'), ('T2m', '0.15', 4.0, '移动 s₀=0.15 D，4 s'), ('T2m', '0.25', 6.0, '移动 s₀=0.25 D，6 s')]
    fig, axs = plt.subplots(len(cases), len(vs), figsize=(1.95 * len(vs), 2.05 * len(cases)))
    for i, (test, s0, t, lab) in enumerate(cases):
        for j, v in enumerate(vs):
            ax = axs[i, j]
            fn = os.path.join(OUT, f'snap_{test}_{v}_{s0}_N128.npz')
            if not os.path.exists(fn): ax.axis('off'); continue
            z = np.load(fn)
            show_c(ax, z[key(z, 'c', t)], (-0.35, 0.35, -0.3, 0.3), LAB[v] if i == 0 else None, o=z[key(z, 'o', t)])
            if j == 0: ax.set_ylabel(lab, fontsize=8)
            m = z[key(z, 'm', t)] / D; l = z[key(z, 'l', t)] / D
            ax.plot(np.r_[m[:, 0], m[:1, 0]], np.r_[m[:, 1], m[:1, 1]], color=CAT[1], lw=0.8)
            ax.plot(np.r_[l[:, 0], l[:1, 0]], np.r_[l[:, 1], l[:1, 1]], color=CAT[0], lw=0.8)
    fig.suptitle('T2 推挤：褐 = 画布，奶油 = 新白，灰蓝 = 旧色筏；橙线 = 旧块物质线，蓝线 = 画布物质环', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'test_Ccompres_T2.png'), dpi=140); plt.close(fig)


def fig_T2gap():
    Rr = load()
    vs = ['v2', 'q', 'sv4', 'sv2', 'sv1', 'sv05', 'cd1e4', 'cp']
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 6.4))
    for j, s0 in enumerate(('0.10', '0.15', '0.25')):
        for i, test in enumerate(('T2h', 'T2m')):
            ax = axs[i, j]
            for v in vs:
                o = Rr.get((test, v, s0, 128))
                if not o: continue
                tt = sorted(float(x) for x in o['series']); gg = [o['series'][f'{x:g}']['gap_min'] for x in tt]
                ax.plot([x for x, y in zip(tt, gg) if y is not None], [y for y in gg if y is not None], '-o', ms=3, lw=1.4, color=COL[v], label=LAB[v])
            ax.axhspan(0.04, 0.13, color=GRID, alpha=0.6, lw=0)
            ax.set_ylim(-0.005, 0.15); ax.set_xlim(1.0, 10.2)
            ax.set_title(('定点' if test == 'T2h' else '移动 0.1 D/1.5 s') + f'，s₀ = {s0} D：新白—旧色白最小距离', fontsize=8.5)
            ax.set_xlabel('t (s)（倒到 4 s）'); ax.set_ylabel('缝 / D')
    axs[0, 2].legend(fontsize=7, loc='upper right'); axs[0, 0].text(9.9, 0.085, '视频带', fontsize=7, color=INK2, ha='right')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'test_Ccompres_T2gap.png'), dpi=150); plt.close(fig)


def fig_T3():
    vs = ['v2', 'q', 'sv2', 'sv1', 'sv05', 'cd1e4', 'cp']
    fig, axs = plt.subplots(2, len(vs), figsize=(2.0 * len(vs), 4.3))
    for i, test in enumerate(('T3', 'T3d')):
        for j, v in enumerate(vs):
            ax = axs[i, j]; fn = os.path.join(OUT, f'snap_{test}_{v}_N128.npz')
            if not os.path.exists(fn): ax.axis('off'); continue
            z = np.load(fn); tE = float(z['t_end'])
            show_c(ax, z[key(z, 'c', tE)], (-0.33, 0.33, -0.2, 0.2), LAB[v] if i == 0 else None)
            if j == 0: ax.set_ylabel('驻留 0.7 s' if test == 'T3d' else '不驻留', fontsize=8)
            m = z[key(z, 'm', tE)] / D
            ax.plot(m[:, 0], m[:, 1], color=CAT[0], lw=0.8)
    fig.suptitle('T3 高位细切（3 mL/s，1.2 m/s，0.4 D/s 自左向右穿过 0.5 D 白盘）终态；蓝线 = x = 0 处的物质线（V 形）', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'test_Ccompres_T3.png'), dpi=150); plt.close(fig)


def fig_T4():
    vs = ['v2', 'q', 'sv2', 'sv1', 'sv05', 'cd1e4', 'cp']
    rows = [('T4f', 3.5, '新画布，停时'), ('T4f', 8.5, '新画布，停后 5 s'), ('T4b', 3.5, '白底，停时'), ('T4b', 8.5, '白底，停后 5 s')]
    fig, axs = plt.subplots(len(rows), len(vs), figsize=(1.95 * len(vs), 2.0 * len(rows)))
    for i, (test, t, lab) in enumerate(rows):
        for j, v in enumerate(vs):
            ax = axs[i, j]; fn = os.path.join(OUT, f'snap_{test}_{v}_N128.npz')
            if not os.path.exists(fn): ax.axis('off'); continue
            z = np.load(fn)
            show_c(ax, z[key(z, 'c', t)], (-0.25, 0.30, -0.22, 0.22), LAB[v] if i == 0 else None)
            if j == 0: ax.set_ylabel(lab, fontsize=8)
            if i in (0, 2):
                p = z['path'] / D; ax.plot(p[:, 0], p[:, 1], color=CAT[0], lw=0.5, alpha=0.7)
    fig.suptitle('T4 摆动（4 mm × 3 Hz，后退 0.15 D/s × 2 s，预倒 1.5 s）；蓝线 = 落点路径', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'test_Ccompres_T4.png'), dpi=140); plt.close(fig)


if __name__ == '__main__':
    which = sys.argv[1].split(',') if len(sys.argv) > 1 else ['unit', 'T1', 'T2', 'T2gap', 'T3', 'T4']
    for w in which:
        globals()['fig_' + w](); print('fig', w, flush=True)
