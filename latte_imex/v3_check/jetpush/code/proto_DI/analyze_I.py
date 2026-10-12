"""Normalised sensitivity matrix, column cosines, condition numbers and Fisher sd for variant I.
usage: python3 -I sens/analyze_I.py sens/sensI_<...>.json  -> prints tables, writes sens/analysisI_<eta>.json and a heatmap"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

PAR = ['Gamma', 'tau_w', 'r_e,ref', 'Pi_coh', 'eta_J']
KEY = {'Gamma': 'Gamma', 'tau_w': 'tau_w', 'r_e,ref': 'delta_e', 'Pi_coh': 'Pi_coh', 'eta_J': 'eta_J'}
L = math.log(1.2 / 0.8)
STEP = {'Gamma': L, 'tau_w': L, 'r_e,ref': L * 1.0e-3 / 3.72e-3, 'Pi_coh': L, 'eta_J': L}
# observables: (key, label, kind, scale)  kind 'log' -> d ln O / d ln theta ; 'abs' -> dO/d ln theta / scale
OBS = [
    ('t_first_vis', 'g1 首白 (T1)', 'log', None),
    ('dAdt_vis', 'g2 dA/dt (T1)', 'log', None),
    ('hole_ins', 'g3 褐洞内切半径 (T1L 窗)', 'log', None),
    ('line_def_final', 'g4 切线终亏量 (T3)', 'log', None),
    ('drift_0.30_tr_end', 'g5 停流漂移 0.30 D/10 s (T1)', 'log', None),
    ('T2s15_near_1.4', 'g6 白前近缘位移 (T2 s0=0.15)', 'abs', 0.03),
    ('open_deg_3', '开口角 3 s (T1)', 'log', None),
    ('umax_after_tr_mm_s', '停倒后 max|u| (T1)', 'log', None),
    ('drift_0.20_tr_end', '停流漂移 0.20 D (T1)', 'log', None),
    ('T2s15_near_4', '近缘位移 4 s (T2 s0=0.15)', 'abs', 0.05),
    ('T2s25_gap_min_6', '缝 6 s (T2 s0=0.25)', 'abs', 0.02),
    ('T2s15_sol_mat_4', 'solidity 4 s (T2 s0=0.15)', 'abs', 0.05),
    ('T2s15_creep_near_4.6_10', '停倒后近缘蠕变 (T2)', 'abs', 0.01),
    ('notch_req', '凹口 (T3 驻留)', 'log', None),
    ('entry_bulge', '入口外鼓 (T3 驻留)', 'log', None),
    ('stem', '茎宽 (T3)', 'log', None),
    ('T4_enclosed_area_mm2', '摆动褐岛面积 (T4)', 'log', None),
    ('T4_mixed_frac_of_white', '摆动灰格比 (T4)', 'log', None),
]
# Fisher set: sigma and counts (all six clips / tulip+leaf only)
FISH = {
    't_first_vis': (lambda x: 0.12, 9, 3),
    'dAdt_vis': (lambda x: 0.10 * abs(x), 2, 2),
    'hole_ins': (lambda x: math.hypot(0.004, 0.10 * abs(x)), 3, 2),
    'line_def_final': (lambda x: 0.004, 9, 3),
    'drift_0.30_tr_end': (lambda x: 0.004, 12, 5),
    'T2s15_near_1.4': (lambda x: 0.03, 2, 1),
}


def main(fn):
    R = json.load(open(fn))
    etas = sorted(set(k.split('|')[0] for k in R))
    out = {}
    for e in etas:
        n0 = R[f'{e}|nominal']
        S = np.full((len(OBS), len(PAR)), np.nan)
        for i, (k, lab, kind, sc) in enumerate(OBS):
            for j, p in enumerate(PAR):
                a, b = R.get(f'{e}|{KEY[p]}+'), R.get(f'{e}|{KEY[p]}-')
                if a is None or b is None or a.get(k) is None or b.get(k) is None or n0.get(k) is None: continue
                d = (a[k] - b[k]) / STEP[p]
                if kind == 'log':
                    S[i, j] = d / abs(n0[k]) if n0[k] else np.nan
                else:
                    S[i, j] = d / sc
        print(f'\n=== eta_nom = {e}: normalised sensitivity (log rows: d ln O/d ln theta; abs rows: dO/d ln theta / scale)')
        print('%-34s' % 'obs' + ''.join('%10s' % p for p in PAR) + '   nominal')
        for i, (k, lab, kind, sc) in enumerate(OBS):
            print('%-34s' % lab[:34] + ''.join('%10.3f' % x for x in S[i]) + '   %.5g' % n0.get(k, float('nan')))
        ok = np.all(np.isfinite(S), axis=1)
        Sn = S[ok] / np.linalg.norm(S[ok], axis=0)
        cos = Sn.T @ Sn
        sv = np.linalg.svd(Sn, compute_uv=False)
        print('column cosines (all finite rows, %d rows):' % ok.sum())
        print(np.round(cos, 3)); print('condition number (column-normalised) %.2f' % (sv[0] / sv[-1]))
        # T1-T4 4-parameter only, same rows, for comparison
        sv4 = np.linalg.svd(Sn[:, :4] / np.linalg.norm(Sn[:, :4], axis=0), compute_uv=False)
        res = dict(S=S.tolist(), rows=[o[1] for o in OBS], keys=[o[0] for o in OBS], cos=cos.tolist(), cond_all=float(sv[0] / sv[-1]),
                   cond_all_4par=float(sv4[0] / sv4[-1]), nominal={k: n0.get(k) for k, *_ in OBS})
        # Fisher (v3-min-2 convention: unweighted chi2, sigma, sqrt(n)); two count sets
        for setname, col in (('six', 1), ('tulip_leaf', 2)):
            rows = [k for k in FISH]
            A = []
            for k in rows:
                sig, n6, ntl = FISH[k]; n = (n6, ntl)[col - 1]
                row = []
                for p in PAR:
                    a, b = R[f'{e}|{KEY[p]}+'], R[f'{e}|{KEY[p]}-']
                    row.append((a[k] - b[k]) / STEP[p] / sig(n0[k]) * math.sqrt(n))
                A.append(row)
            A = np.array(A)
            for npar in (4, 5):
                Ap = A[:, :npar]; rws = [r_ for r_ in range(len(rows)) if rows[r_] not in (('drift_0.30_tr_end', 'T2s15_near_1.4') if npar == 4 else ())]
                Ap = Ap[rws]
                Fm = Ap.T @ Ap; C = np.linalg.inv(Fm); sd = np.sqrt(np.diag(C)); corr = C / np.outer(sd, sd)
                An = Ap / np.linalg.norm(Ap, axis=0); s_ = np.linalg.svd(An, compute_uv=False)
                print(f'Fisher [{setname}, {npar} par, rows {[rows[r_] for r_ in rws]}]: cond {s_[0]/s_[-1]:.2f}; sd(ln theta) {np.round(sd, 3)}')
                print('   corr\n', np.round(corr, 2))
                res[f'fisher_{setname}_{npar}'] = dict(cond=float(s_[0] / s_[-1]), sd=sd.tolist(), corr=corr.tolist(),
                                                      cos=(An.T @ An).tolist(), rows=[rows[r_] for r_ in rws])
        out[e] = res
    json.dump(out, open(fn.replace('sensI_', 'analysisI_'), 'w'), indent=1)
    # heatmap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'WenQuanYi Zen Hei'], 'axes.unicode_minus': False, 'font.size': 8,
                         'figure.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
    cm = LinearSegmentedColormap.from_list('div', ['#e34948', '#f2f1ee', '#2a78d6'])
    fig, axs = plt.subplots(1, len(etas), figsize=(5.2 * len(etas), 6.2))
    axs = np.atleast_1d(axs)
    for ax, e in zip(axs, etas):
        S = np.array(out[e]['S'], dtype=float)
        Sc = np.clip(S, -2, 2)
        ax.imshow(Sc, cmap=cm, vmin=-2, vmax=2, aspect='auto')
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if np.isfinite(S[i, j]):
                    ax.text(j, i, f'{S[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='#0b0b0b')
        ax.set_xticks(range(len(PAR))); ax.set_xticklabels(['Γ', 'τ_w', 'r_e,ref', 'Π_coh', 'η_J'])
        ax.set_yticks(range(len(OBS))); ax.set_yticklabels([o[1] for o in OBS], fontsize=7)
        ax.set_title(f'η_J,nom = {float(e):g}：归一化灵敏度（色标截在 ±2）\n条件数 {out[e]["cond_all"]:.2f}（全部 {len(OBS)} 行，5 参数）', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.dirname(HERE)), 'Iinertia_sens.png'), dpi=150)
    print('heatmap written')


if __name__ == '__main__':
    main(sys.argv[1])
