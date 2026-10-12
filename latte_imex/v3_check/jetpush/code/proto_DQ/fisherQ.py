"""Normalised sensitivity matrix, column cosines, condition numbers for v3-min-2 and variant Q (from sensQ_<var>.json).
A_ij = [O_i(theta_j up) - O_i(theta_j down)] / (Delta ln theta_j * sigma_i)  (per unit ln theta, in units of the
measurement scale sigma_i); relative S_ij = A_ij * sigma_i / |O_i(nominal)| where O_i(nominal) != 0.
Observable sets: ANCHOR6 (v3-min section 4.3 counts), TL (tulip + leaf protocol: anchors that exist in those clips),
TL+ (TL plus tulip/leaf process observables), ALL (every T1-T4 row once)."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = math.log(1.2 / 0.8)
PAR = {'Gamma': ('Gamma+', 'Gamma-', L), 'tau_w': ('tau_w+', 'tau_w-', L),
       'r_e,ref': ('delta_e+', 'delta_e-', L * 1.0e-3 / 3.72e-3), 'Pi_coh': ('Pi_coh+', 'Pi_coh-', L)}
NUI = {'alpha_j': ('alpha_j+', 'alpha_j-', math.log(12 / 5), 0.25), 'alpha_F': ('alpha_F+', 'alpha_F-', L, 0.9),
       'Q scale': ('Qs+', 'Qs-', L, 0.69), 'c_t': ('c_t+', 'c_t-', math.log(3), 0.4)}
SIG = {  # measurement scale (absolute unless callable)
    't_first_vis': 0.12, 'dAdt_vis': lambda x: 0.10 * abs(x), 'hole_ins': lambda x: math.hypot(0.004, 0.10 * abs(x)),
    'line_def_final': 0.004, 'open_deg_3': 15.0, 'specks_mm2_3': 2.0, 'far_disp_0.30_before_white': 0.01,
    'T2h_gap_6': 0.01, 'T2h_sol_4': 0.05, 'T2h_near_prewhite': 0.01, 'T2h_thick_6': 0.02,
    'T2m_sol_4': 0.05, 'T2m_matarea_4': 0.1, 'T2m_gap_4': 0.01, 'T2m_near_prewhite': 0.01, 'T2m_thick_6': 0.02,
    'T3d_notch': 0.01, 'T3d_bulge': 0.01, 'T3d_V': 0.01, 'T3d_line': 0.004, 'T4_lines': 1.0, 'T4_enclosed': 5.0}
ANCHOR6 = {'t_first_vis': 9, 'dAdt_vis': 2, 'hole_ins': 3, 'line_def_final': 9}
TL = {'t_first_vis': 2, 'dAdt_vis': 2, 'hole_ins': 1, 'line_def_final': 6}
# smooth process observables that exist in tulip + leaf: pour-2 pre-white push (tulip 6.6-8.5 s), cut-entry notch and
# side bulges (tulip 14.85 s, leaf 8.8 s entries)
TLP = dict(TL, **{'T2h_near_prewhite': 1, 'T3d_notch': 2, 'T3d_bulge': 2})
SMOOTH = ['t_first_vis', 'dAdt_vis', 'hole_ins', 'line_def_final', 'far_disp_0.30_before_white', 'T2h_gap_6', 'T2h_sol_4',
          'T2h_near_prewhite', 'T2h_thick_6', 'T2m_sol_4', 'T2m_near_prewhite', 'T2m_thick_6', 'T3d_notch', 'T3d_bulge', 'T3d_V']
ALLS = {k: 1 for k in SMOOTH}
TONLY = {'t_first_vis': 2, 'dAdt_vis': 1, 'hole_ins': 1, 'line_def_final': 3}   # A-only baseline (tulip)
SETS = {'ANCHOR6': ANCHOR6, 'TL': TL, 'TL+': TLP, 'ALL-smooth': ALLS, 'T-only': TONLY}


def sig(o, x):
    s = SIG[o]
    return s(x) if callable(s) else s


def matrix(R, rows, cols_spec):
    n0 = R['nominal']
    A = np.zeros((len(rows), len(cols_spec)))
    for i, o in enumerate(rows):
        for j, (a, b, h) in enumerate(cols_spec):
            va, vb = R[a].get(o), R[b].get(o)
            va = 0.0 if va is None or va != va else va
            vb = 0.0 if vb is None or vb != vb else vb
            A[i, j] = (va - vb) / h / sig(o, n0[o] if n0[o] == n0[o] and n0[o] is not None else 0.0)
    return A


def analyse(var, out):
    R = json.load(open(os.path.join(HERE, f'sensQ_{var}.json')))
    n0 = R['nominal']
    P = list(PAR); Nn = list(NUI)
    rows = list(SIG)
    NONSMOOTH = [o for o in rows if o not in SMOOTH]
    Afull = matrix(R, rows, [PAR[c] for c in P] + [NUI[c][:3] for c in Nn])
    out.write(f'\n==== variant {var}: per-sigma sensitivity A = dO/dln(theta)/sigma  (rows: observables)\n')
    out.write('%-26s' % 'obs (nominal)' + ''.join('%10s' % c[:9] for c in P + Nn) + '\n')
    for i, o in enumerate(rows):
        v = n0.get(o)
        out.write('%-26s' % (f'{o} ({v:.4g})' if isinstance(v, (int, float)) and v == v else f'{o} (nan)')
                  + ''.join('%10.2f' % x for x in Afull[i]) + '\n')
    out.write('relative S = dlnO/dln(theta) (only O != 0)\n')
    for i, o in enumerate(rows):
        v = n0.get(o)
        if isinstance(v, (int, float)) and v == v and abs(v) > 1e-9:
            out.write('%-26s' % o + ''.join('%10.2f' % (x * sig(o, v) / abs(v)) for x in Afull[i]) + '\n')
    out.write('non-smooth rows (thresholded / integer / zero; reported, not used in the sets): %s\n' % NONSMOOTH)
    for o in NONSMOOTH:
        out.write('   %-22s ' % o + ' '.join(f'{c}: +{R[PAR[c][0]].get(o)} / -{R[PAR[c][1]].get(o)}' for c in P) + f'  (nominal {n0.get(o)})\n')
    res = {}
    for sname, cnt in SETS.items():
        rws = [o for o in cnt if o in rows]
        idx = [rows.index(o) for o in rws]
        w = np.sqrt(np.array([cnt[o] for o in rws], float))
        A = Afull[idx, :len(P)] * w[:, None]
        B = Afull[idx, len(P):] * w[:, None]
        nrm = np.linalg.norm(A, axis=0)
        out.write(f'\n-- set {sname} (counts {cnt})\n   column norms: ' + ', '.join(f'{c} {x:.2f}' for c, x in zip(P, nrm)) + '\n')
        An = A / np.where(nrm > 0, nrm, 1)
        cos = An.T @ An
        out.write('   column cosines (4 fitted):\n')
        for j, c in enumerate(P):
            out.write('   %-8s' % c + ''.join('%7.2f' % x for x in cos[j]) + '\n')
        sv = np.linalg.svd(An, compute_uv=False)
        cond = sv[0] / sv[-1] if sv[-1] > 1e-12 else float('inf')
        svr = np.linalg.svd(A, compute_uv=False)
        cond_raw = svr[0] / svr[-1] if svr[-1] > 1e-12 else float('inf')
        weak = [c for c, x in zip(P, nrm) if x < 1.0]
        # column-normalised condition number over the anchored columns only (norm >= 1 sigma per unit ln theta)
        keep = [j for j, x in enumerate(nrm) if x >= 1.0]
        svk = np.linalg.svd(An[:, keep], compute_uv=False) if keep else np.array([np.nan])
        cond_k = svk[0] / svk[-1] if len(keep) > 0 and svk[-1] > 1e-12 else float('inf')
        out.write(f'   singular values (column-normalised) {np.round(sv, 3)}; condition number {cond:.3g}'
                  f' [meaningless if a column is noise]; raw sigma-scaled condition number {cond_raw:.3g};'
                  f' columns with norm < 1 sigma per unit ln(theta) (not anchored): {weak};'
                  f' column-normalised cond over anchored columns {cond_k:.3g}\n')
        # cosines of nuisances against the fitted columns
        bn = np.linalg.norm(B, axis=0)
        Bn = B / np.where(bn > 0, bn, 1)
        cnu = An.T @ Bn
        out.write('   cos(fitted, nuisance): ' + '  '.join(f'{Nn[k]}: [' + ' '.join('%.2f' % x for x in cnu[:, k]) + ']'
                                                    for k in range(len(Nn))) + '\n')
        # Fisher (sigma units), measurement only and with nuisance priors
        F = A.T @ A
        try:
            C = np.linalg.inv(F); sd = np.sqrt(np.diag(C)); corr = C / np.outer(sd, sd)
        except np.linalg.LinAlgError:
            sd = np.full(len(P), np.inf); corr = np.full((len(P), len(P)), np.nan)
        pri = np.array([NUI[c][3] for c in Nn])
        Z = np.hstack([A, B]); Fz = Z.T @ Z + np.diag(np.r_[np.zeros(len(P)), 1 / pri ** 2])
        try:
            Cz = np.linalg.inv(Fz); sdz = np.sqrt(np.diag(Cz))[:len(P)]
            cz = Cz[:len(P), :len(P)] / np.outer(sdz, sdz)
        except np.linalg.LinAlgError:
            sdz = np.full(len(P), np.inf); cz = np.full((len(P), len(P)), np.nan)
        out.write(f'   sd(ln theta) meas. only {np.round(sd, 3)}  + nuisances {np.round(sdz, 3)}\n')
        out.write('   max |corr| meas./+nuis.: %.2f / %.2f\n' % (np.nanmax(np.abs(corr - np.eye(len(P)))),
                                                             np.nanmax(np.abs(cz - np.eye(len(P))))))
        res[sname] = dict(cond=cond, cond_raw=cond_raw, cond_anchored=cond_k, weak=weak, cos=cos.tolist(), sv=sv.tolist(), norms=nrm.tolist(), sd=sd.tolist(), sdz=sdz.tolist(),
                          corr=np.nan_to_num(corr).tolist(), corrz=np.nan_to_num(cz).tolist(), cos_nui=cnu.tolist())
    return dict(rows=rows, cols=P + Nn, A=Afull.tolist(), nominal=n0, sets=res)


if __name__ == '__main__':
    allres = {}
    with open(os.path.join(HERE, 'fisherQ.log'), 'w') as f:
        for var in sys.argv[1:] or ['v2', 'q']:
            if os.path.exists(os.path.join(HERE, f'sensQ_{var}.json')):
                allres[var] = analyse(var, f)
    json.dump(allres, open(os.path.join(HERE, 'fisherQ.json'), 'w'), indent=1, default=float)
    print(open(os.path.join(HERE, 'fisherQ.log')).read())
