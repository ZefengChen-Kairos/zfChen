"""Summary sheet: per pour, time series of tilt / heading / spin / spout tip for the current track and v2 with their
uncertainty bands, the intervals where the pose is not (fully) recoverable, and the held-out errors; plus a summary row.
    python3 plot_sheet.py -> ../sheet.jpg"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
V2 = f'{SCR}/pose_v2'
R = f'{V2}/report'
POURS = ['tulip', 'leaf', 'heart', 'push_heart', 'swan']
NB = json.load(open(f'{R}/out/numbers.json'))
C2S = json.load(open(f'{V2}/audit/out/cam2_summary.json'))
LOCO = json.load(open(f'{V2}/audit/out/loco_check.json'))
STR = json.load(open(f'{V2}/audit/out/stream_audit.json'))
FOOT = json.load(open(f'{V2}/out/foot_check.json'))
AH = json.load(open(f'{SCR}/pose_audit/holdout/summary.json'))
CUR_LOCO = {'tulip': (15.2, 10.8), 'leaf': (16.3, 11.9), 'heart': (11.1, 4.6), 'push_heart': (13.1, 17.5), 'swan': (14.5, 10.1)}

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
GOOD, WARN, SERIOUS, CRIT = '#0ca30c', '#fab219', '#ec835a', '#d03b3b'
COV_COL = {3: '#e1e0d9', 2: '#a9a79e', 1: WARN, 0: CRIT}
SPD_COL = {0: '#f4f3ef', 1: '#a9a79e', 2: CRIT}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'legend.frameon': False})


def runs(mask, t):
    m = np.r_[False, mask, False].astype(int); d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    dt = np.median(np.diff(t))
    return [(t[a] - dt / 2, t[b] - t[a] + dt) for a, b in zip(s, e)]


def fillband(x, ok):
    x = np.where(ok & np.isfinite(x), x, np.nan)
    rep = np.nanpercentile(x, 90)
    return np.where(np.isfinite(x), x, rep)


fig = plt.figure(figsize=(26, 30.5), dpi=90)
outer = fig.add_gridspec(7, 1, height_ratios=[1.35, 3.25, 3.25, 3.25, 3.25, 3.25, 3.6], hspace=0.34,
                         left=0.035, right=0.99, top=0.985, bottom=0.052)
# ------------------------------------------------------------------ header
axh = fig.add_subplot(outer[0]); axh.axis('off')
axh.text(0, 1.0, 'Can the milk pitcher\'s pose be fully recovered from the five recorded pours?  No.\nSlow motion is recovered to ~0.2° tilt / ~1 mm tip per instant.  Not recoverable: absolute orientation (camera calibration), spin beyond the tags, fast tilt-backs, anything above 2-5 Hz.',
         fontsize=13.5, fontweight='bold', color=INK, va='top', transform=axh.transAxes)
axh.text(0, 0.60, 'Angles: tilt = jug axis from vertical; heading = azimuth of the jug axis about the vertical (board frame); spin = downhill direction in the pitcher frame minus the spout (0 = spout points straight downhill).\nTime = take clock (s from the cut).  Camera 2 never sees a tag; it is held out of the v2 fit and used for validation only.',
         fontsize=10, color=INK2, va='top', transform=axh.transAxes)
leg1 = [Line2D([], [], color=BLUE, lw=2, label='v2 (pose_v2, 10 ms)'),
        Patch(color=BLUE, alpha=0.45, label='v2 per-instant band: rms(cam-0-only − cam-1-only refit)/√2, 0.25 s window'),
        Patch(color=BLUE, alpha=0.14, label='v2 total band: + systematic (v2 in camera-1 / camera-2 world; spin ±2.5° spout azimuth)'),
        Line2D([], [], color=ORANGE, marker='o', ms=3, lw=0.6, label='current track (tip.npz samples; tip = TIP_KONA)'),
        Patch(color=ORANGE, alpha=0.2, label='current ± per-sample error (audit time-holdout p50)')]
leg2 = [Patch(color=COV_COL[3], label='tags: both cameras within ±0.6 frame'), Patch(color=COV_COL[2], label='one camera, ≥2 tags'),
        Patch(color=COV_COL[1], label='one camera, single tag (per-instant flip ambiguity; resolved by continuity + depth)'),
        Patch(color=COV_COL[0], label='no tag in cams 0/1 (depth + spline only; spin unobserved)'),
        Patch(color=SPD_COL[1], label='speed 10-60 °/s'), Patch(color=SPD_COL[2], label='speed > 60 °/s (also shaded red in the plots)'),
        Line2D([], [], color=INK, marker='v', lw=0, ms=6, label='flip suspect: another pose branch fits the detection > 4× better than v2')]
l1 = axh.legend(handles=leg1, loc='upper left', bbox_to_anchor=(0.0, 0.40), ncol=3, fontsize=9, handlelength=2.2, columnspacing=1.5)
axh.add_artist(l1)
axh.legend(handles=leg2, loc='upper left', bbox_to_anchor=(0.0, 0.14), ncol=4, fontsize=9, handlelength=2.2, columnspacing=1.5)

# ------------------------------------------------------------------ per pour rows
for ip, p in enumerate(POURS):
    S = np.load(f'{R}/out/series_{p}.npz')
    nb = NB[p]
    t = S['t']; ok = S['ok']
    T0 = t[0]; x = t - T0
    g = outer[ip + 1].subgridspec(2, 6, height_ratios=[3.0, 0.62], hspace=0.08, wspace=0.2, width_ratios=[1, 1, 1, 1, 1, 1.22])
    fast = runs(S['speed'] == 2, x); none = runs(S['cls'] == 0, x)
    tc = S['tc'] - T0
    cb = nb['cur_band']
    panels = [('tilt', 'tilt [deg]', S['tilt_c'], cb['tilt']), ('head', 'heading [deg]', S['head_c'], cb['rot']),
              ('spin', 'spin (spout − downhill) [deg]', S['spin_c'], cb['rot'])]
    axes = []
    for j, (k, lab, yc, bc) in enumerate(panels):
        ax = fig.add_subplot(g[0, j]); axes.append(ax)
        y = S[k]; bi = fillband(S[f'inst_{k}'], ok); bs = np.sqrt(bi ** 2 + S[f'sys_{k}'] ** 2)
        ax.fill_between(x, y - bs, y + bs, color=BLUE, alpha=0.14, lw=0)
        ax.fill_between(x, y - bi, y + bi, color=BLUE, alpha=0.45, lw=0)
        ax.fill_between(tc, yc - bc, yc + bc, color=ORANGE, alpha=0.18, lw=0)
        ax.plot(tc, yc, '-', color=ORANGE, lw=0.5, alpha=0.7)
        ax.plot(tc, yc, 'o', color=ORANGE, ms=2.2, mew=0)
        ax.plot(x, y, color=BLUE, lw=1.6)
        if k == 'spin':
            lo, hi = np.percentile(np.r_[y - bs, yc], [0.5, 99.5]); ax.set_ylim(lo - 1.5, hi + 1.5)
        elif k == 'head':
            lo, hi = np.percentile(np.r_[y, yc], [0.5, 99.5]); ax.set_ylim(lo - 4, hi + 4)
        ax.set_ylabel(lab)
        ax.set_title(f'{p}: {lab.split(" [")[0]}' if j == 0 else lab.split(' [')[0], loc='left')
    # tip absolute
    ax = fig.add_subplot(g[0, 3]); axes.append(ax)
    tip = S['tip']; bi = np.stack([fillband(S['inst_tip'][:, i], ok) for i in range(3)], 1); bs = np.sqrt(bi ** 2 + S['sys_tip'] ** 2)
    for i, nm in enumerate('xyz'):
        ax.fill_between(x, tip[:, i] - bs[:, i], tip[:, i] + bs[:, i], color=BLUE, alpha=0.14, lw=0)
        ax.fill_between(x, tip[:, i] - bi[:, i], tip[:, i] + bi[:, i], color=BLUE, alpha=0.45, lw=0)
        ax.plot(tc, S['tip_c'][:, i], 'o', color=ORANGE, ms=1.8, mew=0)
        ax.plot(x, tip[:, i], color=BLUE, lw=1.3)
        ax.text(x[-1] + 0.01 * (x[-1] - x[0]), tip[-1, i], nm, color=INK2, fontsize=9, va='center')
    ax.set_ylabel('spout tip, board frame [mm]'); ax.set_title('spout tip position (x, y, z)', loc='left')
    # |current - v2| tip
    ax = fig.add_subplot(g[0, 4]); axes.append(ax)
    bi = fillband(S['inst_tipn'], ok); bs = np.sqrt(bi ** 2 + S['sys_tipn'] ** 2)
    ax.fill_between(x, 0, bs, color=BLUE, alpha=0.14, lw=0)
    ax.fill_between(x, 0, bi, color=BLUE, alpha=0.45, lw=0)
    dn = np.linalg.norm(S['dtip_c'], axis=1)
    ax.plot(S['tc_in'] - T0, dn, 'o', color=ORANGE, ms=2.4, mew=0)
    ax.axhline(cb['tip'], color=ORANGE, lw=1.0, ls='--')
    ax.text(x[0], cb['tip'] + 0.4, f'current per-sample error p50 {cb["tip"]:.1f} mm', color=INK2, fontsize=8, va='bottom')
    ax.set_ylim(0, max(16, np.percentile(dn, 98) * 1.1))
    ax.set_ylabel('[mm]'); ax.set_title('tip: |current − v2| vs v2 uncertainty', loc='left')
    for ax in axes:
        for a, w in fast:
            ax.axvspan(a, a + w, color=CRIT, alpha=0.13, lw=0)
        for a, w in none:
            ax.axvspan(a, a + w, color=MUTED, alpha=0.25, lw=0)
        ax.set_xlim(x[0], x[-1] * 1.0 + 0.01)
        ax.tick_params(labelbottom=False)
    # strips
    for j in range(5):
        st = fig.add_subplot(g[1, j], sharex=axes[j])
        for c in (3, 2, 1, 0):
            st.broken_barh(runs(S['cls'] == c, x), (1.05, 0.9), color=COV_COL[c], lw=0)
        st.broken_barh(runs(S['amb'], x), (1.05, 0.9), color=WARN, lw=0, hatch='////', edgecolor='#9a6a00')
        for c in (1, 2):
            st.broken_barh(runs(S['speed'] == c, x), (0.05, 0.9), color=SPD_COL[c], lw=0)
        for ts in S['sus_t']:
            st.plot([ts - T0], [1.75], 'v', color=INK, ms=6)
        st.set_ylim(0, 2.0); st.set_yticks([0.5, 1.5]); st.set_yticklabels(['speed', 'tags'], fontsize=8)
        st.grid(False); st.spines['left'].set_visible(False)
        st.set_xlabel(f't − {T0:.2f} s  [s]', fontsize=8.5)
    # held-out column
    gh = g[:, 5].subgridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.12)
    ax = fig.add_subplot(gh[0])
    C2 = json.load(open(f'{V2}/audit/out/cam2_{p}.json'))
    tt = np.array([r['t'] for r in C2]) - T0
    for src, col, nm in (('cur', ORANGE, 'current'), ('cur_pl', AQUA, 'current + own placement'), ('v2out', BLUE, 'v2')):
        yv = np.array([r[src].get('res_absmed', np.nan) if src in r else np.nan for r in C2], float)
        ax.plot(tt, yv, '-', color=col, lw=1.1, label=nm, alpha=0.95)
    ax.set_ylim(0, 20); ax.set_xlim(x[0], x[-1])
    ax.set_title('held-out camera 2: depth |residual| per frame', loc='left', fontsize=9.5)
    ax.set_ylabel('[mm]')
    if ip == 0:
        ax.legend(fontsize=8, loc='upper right', ncol=1)
    axt = fig.add_subplot(gh[1]); axt.axis('off')
    c2 = C2S[p]; lo = LOCO[p]; s_ = STR[p]; f_ = FOOT[p] if p in FOOT else None
    cl = CUR_LOCO[p]
    lines = [
        f"cam-2 IoU med    cur {c2['cur']['iou_med']:.3f} | +pl {c2['cur_pl']['iou_med']:.3f} | v2 {c2['v2out']['iou_med']:.3f}",
        f"cam-2 depth med  cur {c2['cur']['res_med']:.1f} | +pl {c2['cur_pl']['res_med']:.1f} | v2 {c2['v2out']['res_med']:.1f} mm",
        f"LOCO px c0/c1    cur {cl[0]:.1f}/{cl[1]:.1f} | v2 raw {lo['cam0_from_1_noext1'][0]:.1f}/{lo['cam1_from_0_noext1'][0]:.1f} | v2 {lo['cam0_from_1'][0]:.1f}/{lo['cam1_from_0'][0]:.1f}",
        f"stream origin    cur {s_['cur_kona']['horiz']['median']:.1f} | +pl {s_['cur_pl']['horiz']['median']:.1f} | v2 {s_['v2out']['horiz']['median']:.1f} mm (n={s_['n_frames']})",
        f"landing-foot p50 cur {f_['cur']['absdiff_p50']:.1f} | v2 {f_['v2']['absdiff_p50']:.1f} mm (n={f_['v2']['n']})",
        f"x-cam tilt p50   cur {nb['cur_xcam']['tilt'][0]:.2f} | v2 {nb['inst']['raw_abs_diff']['tilt'][0]:.2f} deg",
        f"x-cam tip p50    cur {nb['cur_xcam']['tip'][0]:.1f} | v2 {nb['inst']['raw_abs_diff']['tip'][0]:.1f} mm",
    ]
    axt.text(0.0, 1.0, '\n'.join(lines), fontsize=7.6, family='DejaVu Sans Mono', color=INK, va='top', transform=axt.transAxes, linespacing=1.45)

# ------------------------------------------------------------------ summary row
gs = outer[6].subgridspec(1, 4, wspace=0.28, width_ratios=[1.25, 1, 1, 1.15])
ax = fig.add_subplot(gs[0])
yb = np.arange(len(POURS))[::-1]
keys = [('both', COV_COL[3], 'tags in both cams'), ('one_multi', COV_COL[2], 'one cam ≥2 tags'), ('one_single', COV_COL[1], 'one cam single tag'), ('none', COV_COL[0], 'no tag')]
for i, p in enumerate(POURS):
    ft = NB[p]['frac_time']; left = 0
    for k, col, nm in keys:
        v = ft[k] * 100
        ax.barh(yb[i] + 0.2, v, left=left, height=0.36, color=col, edgecolor='#fcfcfb', lw=1.5, label=nm if i == 0 else None)
        left += v
    left = 0
    for k, col, nm in (('slow', SPD_COL[0], '< 10 °/s'), ('mid', SPD_COL[1], '10-60 °/s'), ('fast', SPD_COL[2], '> 60 °/s')):
        v = ft[k] * 100
        ax.barh(yb[i] - 0.2, v, left=left, height=0.36, color=col, edgecolor='#c3c2b7' if k == 'slow' else '#fcfcfb', lw=1.0, label=nm if i == 0 else None)
        if k == 'fast':
            ax.text(101, yb[i] - 0.2, f'{v:.1f}% fast', va='center', fontsize=8, color=INK2)
        left += v
    ax.text(101, yb[i] + 0.2, f"{(ft['one_single'] + ft['none']) * 100:.1f}% weak", va='center', fontsize=8, color=INK2)
ax.set_yticks(yb); ax.set_yticklabels(POURS); ax.set_xlim(0, 118); ax.set_xlabel('share of pour time [%]  (upper bar: tag coverage; lower bar: v2 angular speed)')
ax.set_title('Where the time goes', loc='left'); ax.grid(axis='y', visible=False)
ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.13))


def dotrow(ax, rows, title, xlabel, logx=True):
    for i, p in enumerate(POURS):
        for (lab, col, mk, val) in rows(p):
            ax.plot(val, yb[i], mk, color=col, ms=8, mec='#fcfcfb', mew=1.2, label=lab if i == 0 else None)
    ax.set_yticks(yb); ax.set_yticklabels(POURS)
    if logx:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title, loc='left'); ax.set_xlabel(xlabel); ax.grid(axis='y', visible=False)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)


dotrow(fig.add_subplot(gs[1]), lambda p: [
    ('current, per instant (x-cam p50)', ORANGE, 'o', NB[p]['cur_xcam']['tilt'][0]),
    ('v2, per instant (x-cam p50)', BLUE, 'o', NB[p]['inst']['raw_abs_diff']['tilt'][0]),
    ('v2, per instant p90', BLUE, 'D', NB[p]['inst']['raw_abs_diff']['tilt'][1]),
    ('v2, fastest bin p90', CRIT, '^', NB[p]['inst']['by_speed']['fast']['tilt_p90'] or np.nan),
    ('v2 systematic (camera worlds), median', INK, 's', NB[p]['sys']['tilt_med'])],
    'Tilt: what is known per instant vs in absolute terms', 'tilt uncertainty [deg] (log)')
dotrow(fig.add_subplot(gs[2]), lambda p: [
    ('current, per instant (x-cam p50)', ORANGE, 'o', NB[p]['cur_xcam']['tip'][0]),
    ('v2, per instant (x-cam p50)', BLUE, 'o', NB[p]['inst']['raw_abs_diff']['tip'][0]),
    ('v2, per instant p90', BLUE, 'D', NB[p]['inst']['raw_abs_diff']['tip'][1]),
    ('v2, fastest bin p90', CRIT, '^', NB[p]['inst']['by_speed']['fast']['tip_p90'] or np.nan),
    ('v2 systematic (camera worlds + spout az), median', INK, 's', NB[p]['sys']['tip_med'])],
    'Spout tip: per instant vs absolute', 'tip uncertainty [mm] (log)')
dotrow(fig.add_subplot(gs[3]), lambda p: [
    ('cam-2 depth |res| med: current', ORANGE, 'o', C2S[p]['cur']['res_med']),
    ('… current + own placement', AQUA, 'o', C2S[p]['cur_pl']['res_med']),
    ('… v2', BLUE, 'o', C2S[p]['v2out']['res_med']),
    ('stream origin med: current (production)', ORANGE, 's', STR[p]['cur_kona']['horiz']['median']),
    ('… v2', BLUE, 's', STR[p]['v2out']['horiz']['median']),
    ('landing vs measured foot p50: current', ORANGE, '^', FOOT[p]['cur']['absdiff_p50']),
    ('… v2', BLUE, '^', FOOT[p]['v2']['absdiff_p50'])],
    'Held-out / independent checks (lower is better)', 'error [mm]', logx=False)
fig.savefig(f'{R}/sheet.jpg', dpi=90, pil_kwargs=dict(quality=88))
print('ok')
