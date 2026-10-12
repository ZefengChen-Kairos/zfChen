"""Figures for test_Iinertia.md (PNG next to the report)."""
import os, sys, json, glob, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = os.path.join(HERE, 'out')
FIG = os.path.dirname(HERE)            # jetpush/  (next to test_Iinertia.md)
D = 0.08; R = D / 2
CMAP = LinearSegmentedColormap.from_list('latte', ['#5a3420', '#f7f1e3'])
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e4e2dc'
plt.rcParams.update({'font.family': ['DejaVu Sans', 'WenQuanYi Zen Hei'], 'axes.unicode_minus': False, 'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'lines.linewidth': 2.0, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                     'savefig.facecolor': '#fcfcfb'})
ETAS = [0.0, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3]
ECOL = {e: CAT[i] for i, e in enumerate(ETAS)}


def elab(e):
    return 'η_J = 0 (v3-min-2)' if e == 0 else f'η_J = {e:.0e}'.replace('e-0', 'e-')


def load(fn):
    rows = []
    for f in glob.glob(os.path.join(OUT, fn)):
        rows += [json.loads(l) for l in open(f)]
    return rows


def snap(name):
    return np.load(os.path.join(OUT, 'snap_' + name + '.npz'))


def show_c(ax, c, N, extent_D=None, title=None):
    h = D / N
    ext = [-0.5, 0.5, -0.5, 0.5]
    ax.imshow(np.clip(c, 0, 1).T, origin='lower', extent=ext, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest')
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(0.5 * np.cos(th), 0.5 * np.sin(th), color=INK2, lw=0.8)
    if extent_D:
        ax.set_xlim(extent_D[0], extent_D[1]); ax.set_ylim(extent_D[2], extent_D[3])
    ax.set_aspect('equal'); ax.grid(False)
    ax.tick_params(labelsize=7)
    if title: ax.set_title(title, fontsize=8, color=INK)


def fig_T1():
    rows = {r['kw']['eta']: r for r in load('scan_N128_T1.jsonl')}
    es = [0.0, 1e-4, 1e-3]
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
    for j, e in enumerate(es):
        s = snap(rows[e]['name'])
        for i, (t, lab) in enumerate(((3.0, '3 s（停倒）'), (13.0, '13 s（停倒后 10 s）'))):
            ax = axs[i, j]
            show_c(ax, s[f'c_{t:g}'], 128, (-0.3, 0.3, -0.3, 0.3), f'{elab(e)}, t = {lab}')
            m0 = s['mk0'] / D; m = s[f'm_{t:g}'] / D
            sel = slice(0, 32 * 2)          # rings 0.10, 0.15 D
            ax.plot(m0[sel, 0], m0[sel, 1], 'o', ms=1.5, color=INK2)
            ax.plot(m[sel, 0], m[sel, 1], 'o', ms=2.0, color=CAT[1])
    axs[1, 0].set_xlabel('x / D（ê = +x，壶在 −x 侧）')
    # drift vs radius (markers already inside 7.5 mm of the stopped jet at t_s + t_r are swallowed -> excluded)
    ax = axs[0, 3]
    radii = [0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.45]
    elist = [e for e in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3] if e in rows]
    xg = (np.arange(128) + 0.5) / 128 * D - R; Xg, Yg = np.meshgrid(xg, xg, indexing='ij'); rg = np.hypot(Xg, Yg)
    chA, umx = [], []
    for k, e in enumerate(elist):
        s_ = snap(rows[e]['name']); mf = s_['m_3.576']; me = s_['m_13']
        rf = np.hypot(mf[:, 0], mf[:, 1]); d = np.hypot(*(me - mf).T) / D
        vals = []
        for kk, rr in enumerate(radii):
            sl = slice(kk * 32, (kk + 1) * 32); keep = rf[sl] > 0.0075
            vals.append(d[sl][keep].max())
        ax.plot(radii, vals, '-o', ms=3.5, color=CAT[k], label=elab(e))
        chA.append((np.abs(s_['c_13'] - s_['c_3.576']) > 0.5).sum() * (D / 128) ** 2 * 1e6)
    ax.axhline(0.01, color=INK2, lw=1, ls='--'); ax.text(0.31, 0.0115, 'V2：10 s 内 ≤ 0.01 D', fontsize=7, color=INK2)
    ax.set_yscale('log'); ax.set_xlabel('标记环初始半径 r / D'); ax.set_ylabel('停倒 + t_r 后 10 s 最大漂移（D）')
    ax.set_ylim(1.5e-4, 12); ax.set_title('停流位移（η_J = 0 时恒为 0）', fontsize=8); ax.legend(fontsize=6.2, ncol=3, loc='upper center')
    ax = axs[1, 3]
    ax.loglog(elist, np.maximum(chA, 1e-2), '-o', ms=5, color=CAT[0])
    for e, a in zip(elist, chA):
        ax.annotate(f'{a:.1f}', (e, max(a, 1e-2)), textcoords='offset points', xytext=(4, -10), fontsize=7, color=INK2)
    ax.axhline(0.4, color=INK2, lw=1, ls='--'); ax.text(2.5e-4, 0.5, '1 格 = 0.39 mm²', fontsize=7, color=INK2)
    ax.set_xlabel('η_J'); ax.set_ylabel('停倒后 10 s 内 |Δc| > 0.5 的面积（mm²）')
    ax.set_title('冻结：停倒后翻色面积（v3-min-2：0）', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'Iinertia_T1.png'), dpi=150); plt.close(fig)


def fig_T2():
    rows = load('scan_N128_T2_T3_T4.jsonl')
    T2 = {(r['kw']['eta'], r['kw']['s0'], r['kw']['travel']): r for r in rows if r['kind'] == 'T2'}
    es = [0.0, 1e-3, 5e-3]
    fig, axs = plt.subplots(2, 4, figsize=(11.5, 5.8), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.3]))
    for j, e in enumerate(es):
        r = T2.get((e, 0.15, 0.0))
        if r is None: continue
        s = snap(r['name'])
        for i, t in enumerate((4.0, 10.0)):
            ax = axs[i, j]
            c = s[f'c_{t:g}']; o = s[f'o_{t:g}']; m = s[f'm_{t:g}'] / D
            show_c(ax, c, 128, (-0.35, 0.45, -0.4, 0.4), f'定点 s₀ = 0.15 D，{elab(e)}，t = {t:g} s')
            ax.contour(np.linspace(-0.5, 0.5, 128), np.linspace(-0.5, 0.5, 128), ((o >= 0.5 * c) & (c >= 0.5)).T.astype(float),
                       levels=[0.5], colors=[CAT[0]], linewidths=1.0)
            ax.plot(np.r_[m[:, 0], m[:1, 0]], np.r_[m[:, 1], m[:1, 1]], color=CAT[1], lw=0.9)
    axs[1, 0].set_xlabel('x / D；蓝线 = 旧色白边界，橙线 = 旧块物质线')
    ax = axs[0, 3]
    for e in [0.0, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3]:
        for s0, ls in ((0.15, '-'), (0.25, '--')):
            r = T2.get((e, s0, 0.0))
            if r is None: continue
            ser = r['o']['series']; ts = sorted(float(t) for t in ser)
            g = [ser[f'{t:g}']['gap_min'] for t in ts]
            ax.plot(ts, g, ls, color=ECOL[e], lw=1.6, label=elab(e) if s0 == 0.15 else None)
    ax.axhspan(0.04, 0.13, color=GRID, alpha=0.7, lw=0); ax.text(4.3, 0.135, '视频缝 0.04–0.13 D（灰带）', fontsize=7, color=INK2)
    ax.set_xlabel('t（s）；倒 0–4 s'); ax.set_ylabel('新旧白最小褐缝（D）'); ax.set_ylim(-0.005, 0.16)
    ax.set_title('定点：缝宽（实线 s₀ = 0.15，虚线 0.25 D）', fontsize=8); ax.legend(fontsize=6.2, loc='center right', bbox_to_anchor=(1.0, 0.6))
    ax = axs[1, 3]
    for e in [0.0, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3]:
        r = T2.get((e, 0.15, 0.0))
        if r is None: continue
        ser = r['o']['series']; ts = sorted(float(t) for t in ser)
        ax.plot(ts, [ser[f'{t:g}']['solidity'] for t in ts], color=ECOL[e], lw=1.6, label=elab(e))
    ax.axhline(0.85, color=INK2, lw=1, ls='--'); ax.text(0.3, 0.86, '月牙线 0.85', fontsize=7, color=INK2)
    ax.set_xlabel('t（s）'); ax.set_ylabel('旧块物质线 solidity'); ax.set_title('定点 s₀ = 0.15 D：solidity', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'Iinertia_T2.png'), dpi=150); plt.close(fig)


def fig_T3():
    rows = load('scan_N128_T2_T3_T4.jsonl')
    T3 = {(r['kw']['eta'], r['kw']['dwell']): r for r in rows if r['kind'] == 'T3'}
    es = [0.0, 1e-3, 2e-3]
    fig, axs = plt.subplots(2, 4, figsize=(11.5, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.3]))
    for j, e in enumerate(es):
        for i, dw in enumerate((0.0, 0.7)):
            r = T3.get((e, dw))
            if r is None: continue
            s = snap(r['name']); keys = sorted([k for k in s.files if k.startswith('c_')], key=lambda k: float(k[2:]))
            kE = keys[-1]; tE = float(kE[2:])
            ax = axs[i, j]
            show_c(ax, s[kE], 128, (-0.42, 0.42, -0.3, 0.3), f'{elab(e)}，驻留 {dw:g} s，t = {tE:.2f} s')
            mk = s[f'm_{tE:g}'] / D if f'm_{tE:g}' in s.files else None
            if mk is not None:
                ax.plot(mk[:, 0], mk[:, 1], color=CAT[1], lw=0.9)
    axs[1, 0].set_xlabel('x / D；橙线 = 初始在 x = 0 的物质线（射流沿 +x）')
    ax = axs[0, 3]
    for e in [0.0, 1e-4, 3e-4, 1e-3, 2e-3]:
        r = T3.get((e, 0.0))
        if r is None: continue
        v = r['o']['vshape']['cut+3']
        yy = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.15]
        ax.plot(yy, [v[f'{y:g}'] for y in yy], '-o', ms=3.5, color=ECOL[e], label=elab(e))
    ax.set_xlabel('|y| / D（离切线距离）'); ax.set_ylabel('物质线前拖 Δx（D），切后 3 s')
    ax.set_title('"V 形"剖面（无驻留）；视频：~0.1 D 宽人字', fontsize=8); ax.legend(fontsize=6.5)
    ax = axs[1, 3]
    es5 = [0.0, 1e-4, 3e-4, 1e-3, 2e-3]
    ax.plot([1e3 * e for e in es5], [T3[(e, 0.7)]['o']['notch_req'] for e in es5], '-o', color=CAT[0], ms=4, label='凹口当量半径')
    ax.plot([1e3 * e for e in es5], [T3[(e, 0.7)]['o']['entry_bulge_max'] for e in es5], '-o', color=CAT[1], ms=4, label='入口外鼓')
    ax.plot([1e3 * e for e in es5], [T3[(e, 0.0)]['o']['line_def_final'] for e in es5], '-o', color=CAT[2], ms=4, label='切线积分亏量（无驻留）')
    ax.axhline(0.05, color=INK2, lw=1, ls='--'); ax.text(0.05, 0.053, '视频凹口 / 外鼓 ~0.05 D', fontsize=7, color=INK2)
    ax.set_xlabel('η_J × 10³'); ax.set_ylabel('D'); ax.set_title('驻留 0.7 s 的凹口与外鼓', fontsize=8); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'Iinertia_T3.png'), dpi=150); plt.close(fig)


def fig_T4():
    rows = load('scan_N128_T2_T3_T4.jsonl')
    T4 = {(r['kw']['eta'], r['kw']['base']): r for r in rows if r['kind'] == 'T4'}
    es = [0.0, 1e-3, 2e-3]
    fig, axs = plt.subplots(2, 4, figsize=(11.5, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.3]))
    for i, b in enumerate((False, True)):
        for j, e in enumerate(es):
            r = T4.get((e, b))
            if r is None: continue
            s = snap(r['name'])
            ax = axs[i, j]
            show_c(ax, s['c_3.5'], 128, (-0.3, 0.35, -0.25, 0.25), f'{"白底上" if b else "新画布"}，{elab(e)}，停摆 3.5 s')
            p = s['path'] / D
            ax.plot(p[:, 0], p[:, 1], color=CAT[0], lw=0.5, alpha=0.8)
        ax = axs[i, 3]
        import px_tests as T
        from px_model import Grid
        for e in [0.0, 1e-4, 3e-4, 1e-3, 2e-3]:
            r = T4.get((e, b))
            if r is None: continue
            s = snap(r['name']); c = s['c_3.5']
            x1 = (np.arange(128) + 0.5) / 128 - 0.5
            jm = np.abs(x1 * D) <= 1.25e-3
            ax.plot(x1, c[:, jm].mean(1), color=ECOL[e], lw=1.3, label=elab(e))
        ax.set_xlim(-0.2, 0.25); ax.set_xlabel('x / D（中线 |y| ≤ 1.25 mm）'); ax.set_ylabel('c')
        ax.set_title(f'{"白底上" if b else "新画布"}：停摆时中线剖面（视频：6 条褐线，间距 0.03–0.045 D）', fontsize=7.5)
        ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'Iinertia_T4.png'), dpi=150); plt.close(fig)


def fig_unit():
    u0 = json.load(open(os.path.join(OUT, 'unit_I0.json')))
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axs[0]
    ts = sorted(set(r['t'] for r in u0['rows']))
    for k, t in enumerate(ts):
        rr = [r for r in u0['rows'] if r['t'] == t]
        ax.plot([r['r_mm'] for r in rr], [r['ratio'] for r in rr], '-o', ms=4, color=CAT[k], label=f't = {t:g} s')
    ax.axhspan(0.97, 1.03, color=GRID, lw=0); ax.set_ylim(0.9, 1.02)
    ax.set_xlabel('离射流 r（mm）'); ax.set_ylabel('u_x 数值 / 解析'); ax.set_title('T-I0：投影 Rayleigh 偶极（验收 r ≥ 3r₀ 处 ±3%）', fontsize=8)
    ax.legend(fontsize=7)
    ax = axs[1]
    for k, e in enumerate((1e-9, 1e-3, 1e-1)):
        f = os.path.join(OUT, f'unit_I1_{e:g}.json')
        if not os.path.exists(f): continue
        u1 = json.load(open(f))
        ax.semilogy(u1['t'], np.maximum(u1['du'], 1e-17), '-o', ms=3.5, color=CAT[k], label=f'max|Δu|，η_J = {e:g}')
        ax.semilogy(u1['t'], u1['Gbar'], ':', color=CAT[k], lw=1.4, label=f'max|Ḡ|，η_J = {e:g}')
    ax.set_xlabel('t（s）；倒 0–3 s'); ax.set_ylabel('m/s')
    ax.set_title('定理 I：无旋足迹力 + 记忆，对 v3-min-2 的速度差（1.5–3.5 s 为 VOF 舍入放大）', fontsize=7.5); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'Iinertia_unit.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    which = sys.argv[1].split(',') if len(sys.argv) > 1 else ['unit', 'T1', 'T2', 'T3', 'T4']
    for w in which:
        globals()['fig_' + w]()
        print('fig', w)
