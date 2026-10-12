"""Normalised sensitivity matrix, column cosines, condition numbers and Fisher sd for variant C (C-V + L5Q).
usage: python3 -I sens/analyze_C.py sens/sensC_<...>.json  -> prints tables, writes sens/analysisC_<...>.json + heatmap"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

PAR = ['Gamma', 'tau_w', 'r_e,ref', 'Pi_coh', 'ell_s']
KEY = {'Gamma': 'Gamma', 'tau_w': 'tau_w', 'r_e,ref': 'delta_e', 'Pi_coh': 'Pi_coh', 'ell_s': 'ell_s'}
L = math.log(1.2 / 0.8)
STEP = {'Gamma': L, 'tau_w': L, 'r_e,ref': L * 1.0e-3 / 3.72e-3, 'Pi_coh': L, 'ell_s': L}
# (key, label, kind, scale): 'log' -> d ln O / d ln theta ; 'abs' -> (dO / d ln theta) / scale (scale = video tolerance)
OBS = [
    ('t_first_vis', 'g1 首白 (T1)', 'log', None),
    ('dAdt_vis', 'g2 dA/dt (T1)', 'log', None),
    ('hole_ins', 'g3 褐洞内切半径 (T1L 窗)', 'log', None),
    ('line_def_final', 'g4 切线终亏量 (T3) /0.004 D', 'abs', 0.004),
    ('T2s15_loop_fixed_pw', 'gℓ 白前画布环面积比 (定位环, t_k+τ_w−0.1 s) /0.1', 'abs', 0.1),
    ('T2s15_near_pw', 'g6 白前近缘位移 (t_k+τ_w−0.1 s) /0.03 D', 'abs', 0.03),
    ('T2s15_loop_ratio_1.4', '[旧口径] 画布环面积比 1.4 s, 环随 r_e 移 /0.1', 'abs', 0.1),
    ('T2s15_far_over_near_1.4', '白前远/近缘位移比 (T2) /0.1', 'abs', 0.1),
    ('open_deg_3', '开口角 3 s (T1) /10°', 'abs', 10.0),
    ('specks_mm2_3', '白内褐斑 3 s (T1) /2 mm²', 'abs', 2.0),
    ('disp_0.30_6', '0.30 D 处 6 s 位移 (T1)', 'log', None),
    ('disp_0.45_6', '0.45 D 处 6 s 位移 (T1)', 'log', None),
    ('T2s15_near_4', '近缘位移 4 s (T2 s0=0.15) /0.05 D', 'abs', 0.05),
    ('T2s15_gap_min_4', '缝 4 s (T2 s0=0.15) /0.02 D', 'abs', 0.02),
    ('T2s25_gap_min_6', '缝 6 s (T2 s0=0.25) /0.02 D', 'abs', 0.02),
    ('T2s15_sol_mat_4', 'solidity 4 s 定点 /0.05', 'abs', 0.05),
    ('T2m15_sol_mat_4', 'solidity 4 s 移动 /0.05', 'abs', 0.05),
    ('T2m15_area_ratio_4', '旧块物质面积比 4 s 移动', 'log', None),
    ('notch_req', '凹口 (T3 驻留)', 'log', None),
    ('entry_bulge', '入口外鼓 (T3 驻留)', 'log', None),
    ('stem', '茎宽 (T3)', 'log', None),
    ('T4_enclosed_area_mm2', '摆动褐岛面积 (T4)', 'log', None),
    ('T4_mixed_frac_of_white', '摆动灰格比 (T4)', 'log', None),
    ('T4_n_lines_median', '摆动褐线数 (T4) /1', 'abs', 1.0),
]
# Fisher set: (sigma(x), count six clips, count tulip+leaf)
FISH = {
    't_first_vis': (lambda x: 0.12, 9, 3),
    'dAdt_vis': (lambda x: 0.10 * abs(x), 2, 2),
    'hole_ins': (lambda x: math.hypot(0.004, 0.10 * abs(x)), 3, 2),
    'line_def_final': (lambda x: 0.004, 9, 3),
    'T2s15_loop_fixed_pw': (lambda x: SIG_L, 3, 1),
    'T2s15_near_pw': (lambda x: 0.03, 2, 1),
    'T2s15_loop_ratio_1.4': (lambda x: SIG_L, 3, 1),
}
SIG_L = 0.2


def deriv(R, e, p, k):
    a, b = R.get(f'{e}|{KEY[p]}+'), R.get(f'{e}|{KEY[p]}-')
    if a is None or b is None or a.get(k) is None or b.get(k) is None: return None
    return (a[k] - b[k]) / STEP[p]


def fisher(R, e, rows, pars, counts_col, sigL=None):
    global SIG_L
    if sigL is not None: SIG_L = sigL
    n0 = R[f'{e}|nominal']
    A = []
    for k in rows:
        sig, n6, ntl = FISH[k]; n = (n6, ntl)[counts_col]
        A.append([deriv(R, e, p, k) / sig(n0[k]) * math.sqrt(n) for p in pars])
    A = np.array(A)
    nrm = np.linalg.norm(A, axis=0)
    out = dict(rows=rows, pars=pars, colnorm=nrm.tolist())
    if np.any(nrm < 1e-9):
        out.update(cond=float('inf'), sd=[float('inf')] * len(pars), note='singular: zero column ' + str([p for p, x in zip(pars, nrm) if x < 1e-9]))
        return out
    An = A / nrm; s_ = np.linalg.svd(An, compute_uv=False)
    Fm = A.T @ A
    try:
        C = np.linalg.inv(Fm); sd = np.sqrt(np.diag(C)); corr = C / np.outer(sd, sd)
    except np.linalg.LinAlgError:
        sd = np.full(len(pars), np.inf); corr = np.full((len(pars),) * 2, np.nan)
    out.update(cond=float(s_[0] / s_[-1]), sd=sd.tolist(), corr=corr.tolist(), cos=(An.T @ An).tolist(), sv=s_.tolist())
    return out


def main(fn):
    R = json.load(open(fn))
    ells = sorted(set(k.split('|')[0] for k in R), key=float)
    out = {}
    for e in ells:
        if f'{e}|nominal' not in R: continue
        n0 = R[f'{e}|nominal']
        S = np.full((len(OBS), len(PAR)), np.nan)
        for i, (k, lab, kind, sc) in enumerate(OBS):
            for j, p in enumerate(PAR):
                d = deriv(R, e, p, k)
                if d is None or n0.get(k) is None: continue
                if kind == 'log':
                    S[i, j] = d / abs(n0[k]) if n0[k] else np.nan
                else:
                    S[i, j] = d / sc
        print(f'\n=== ell_s,nom = {e} m: normalised sensitivity (log rows: d ln O/d ln theta; abs rows: dO/d ln theta / scale)')
        print('%-40s' % 'obs' + ''.join('%10s' % p for p in PAR) + '   nominal')
        for i, (k, lab, kind, sc) in enumerate(OBS):
            nv = n0.get(k)
            print('%-40s' % lab[:40] + ''.join('%10.3f' % x for x in S[i]) + '   %s' % ('%.5g' % nv if nv is not None else '—'))
        ok = np.all(np.isfinite(S), axis=1)
        Sn = S[ok] / np.linalg.norm(S[ok], axis=0)
        cos = Sn.T @ Sn
        sv = np.linalg.svd(Sn, compute_uv=False)
        sv4 = np.linalg.svd(S[ok][:, :4] / np.linalg.norm(S[ok][:, :4], axis=0), compute_uv=False)
        print('rows used:', int(ok.sum()), 'of', len(OBS), '; dropped:', [OBS[i][0] for i in range(len(OBS)) if not ok[i]])
        print('column cosines:\n', np.round(cos, 3))
        print('condition number (column-normalised, %d rows): 5 par %.2f ; 4 old par %.2f' % (ok.sum(), sv[0] / sv[-1], sv4[0] / sv4[-1]))
        res = dict(S=[[None if not np.isfinite(x) else float(x) for x in r] for r in S], rows=[o[1] for o in OBS], keys=[o[0] for o in OBS],
                   cos=cos.tolist(), cond_all=float(sv[0] / sv[-1]), cond_all_4par=float(sv4[0] / sv4[-1]), rows_used=int(ok.sum()),
                   nominal={k: n0.get(k) for k, *_ in OBS})
        # without the T4 rows and the quantised rows (opening angle, lines count)
        sel = [i for i in range(len(OBS)) if ok[i] and OBS[i][0] not in ('open_deg_3', 'T4_n_lines_median', 'specks_mm2_3', 'T2s15_loop_ratio_1.4', 'line_def_final')]
        Sq0 = S[sel]; res['rows_noquant'] = [OBS[i][0] for i in sel]
        res['cos_noquant'] = ((Sq0 / np.linalg.norm(Sq0, axis=0)).T @ (Sq0 / np.linalg.norm(Sq0, axis=0))).tolist()
        sv4q = np.linalg.svd(Sq0[:, :4] / np.linalg.norm(Sq0[:, :4], axis=0), compute_uv=False); res['cond_noquant_4par'] = float(sv4q[0] / sv4q[-1])
        print('cosines without quantised/noisy rows (%d rows):' % len(sel)); print(np.round(np.array(res['cos_noquant']), 3))
        Sq = S[sel] / np.linalg.norm(S[sel], axis=0); s2 = np.linalg.svd(Sq, compute_uv=False)
        res['cond_noquant'] = float(s2[0] / s2[-1]); print('condition number without quantised/noisy rows: 5 par %.2f ; 4 old par %.2f' % (res['cond_noquant'], res['cond_noquant_4par']))
        for setname, col in (('six', 0), ('tulip_leaf', 1)):
            for pars, rows in ((PAR[:4], ['t_first_vis', 'dAdt_vis', 'hole_ins', 'line_def_final']),
                               (PAR, ['t_first_vis', 'dAdt_vis', 'hole_ins', 'line_def_final', 'T2s15_loop_fixed_pw']),
                               (PAR, ['t_first_vis', 'dAdt_vis', 'hole_ins', 'line_def_final', 'T2s15_loop_fixed_pw', 'T2s15_near_pw']),
                               ([p for p in PAR if p != 'Pi_coh'], ['t_first_vis', 'dAdt_vis', 'hole_ins', 'T2s15_loop_fixed_pw']),
                               ([p for p in PAR if p != 'Pi_coh'], ['t_first_vis', 'dAdt_vis', 'hole_ins', 'T2s15_loop_fixed_pw', 'T2s15_near_pw']),
                               ([p for p in PAR if p != 'Pi_coh'], ['t_first_vis', 'dAdt_vis', 'hole_ins', 'T2s15_loop_ratio_1.4'])):
                for sigL in (0.1, 0.2, 0.35):
                    if not any(r_.startswith('T2s15_loop') for r_ in rows) and sigL != 0.2: continue
                    fo = fisher(R, e, rows, pars, col, sigL)
                    tag = f'fisher_{setname}_{len(pars)}p_{len(rows)}r_sigL{sigL}'
                    res[tag] = fo
                    print(f'Fisher [{setname}] pars {pars} rows {rows} sig_l={sigL}: cond {fo["cond"]:.2f}; sd(ln) {np.round(fo["sd"], 3)}'
                          + (('  ' + fo.get('note', '')) if 'note' in fo else ''))
                    if 'corr' in fo and len(pars) >= 4:
                        C = np.array(fo['corr']); iu = np.triu_indices(len(pars), 1)
                        kmax = int(np.nanargmax(np.abs(C[iu])))
                        print('     max |corr|: %s-%s %.2f' % (pars[iu[0][kmax]], pars[iu[1][kmax]], C[iu][kmax]))
        out[e] = res
    json.dump(out, open(fn.replace('sensC_', 'analysisC_'), 'w'), indent=1)
    # heatmap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'WenQuanYi Zen Hei'], 'axes.unicode_minus': False, 'font.size': 8,
                         'figure.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
    cm = LinearSegmentedColormap.from_list('div', ['#e34948', '#f2f1ee', '#2a78d6'])
    es = [e for e in ells if e in out]
    fig, axs = plt.subplots(1, len(es), figsize=(5.6 * len(es), 7.4))
    axs = np.atleast_1d(axs)
    for ax, e in zip(axs, es):
        S = np.array([[np.nan if x is None else x for x in r] for r in out[e]['S']], dtype=float)
        ax.imshow(np.clip(S, -2, 2), cmap=cm, vmin=-2, vmax=2, aspect='auto')
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                ax.text(j, i, '—' if not np.isfinite(S[i, j]) else f'{S[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='#0b0b0b')
        ax.set_xticks(range(len(PAR))); ax.set_xticklabels(['Γ', 'τ_w', 'r_e,ref', 'Π_coh', 'ℓ_s'])
        ax.set_yticks(range(len(OBS))); ax.set_yticklabels([o[1] for o in OBS], fontsize=6.8)
        ax.set_title(f'ℓ_s,nom = {float(e)*100:g} cm：归一化灵敏度（色标截在 ±2）\n列归一化条件数 {out[e]["cond_all"]:.2f}（全部 {out[e]["rows_used"]} 行）/ {out[e]["cond_noquant"]:.2f}（去掉量化与噪声行，{len(out[e]["rows_noquant"])} 行）', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.dirname(HERE)), 'test_Ccompres_sens.png'), dpi=150)
    print('heatmap written')


if __name__ == '__main__':
    main(sys.argv[1])
