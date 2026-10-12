"""Relative sensitivities and Fisher analysis (unweighted chi2, true sigma, nuisances marginalised) for v3-min-2."""
import json, math, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'sens2.json'))); n0 = R['nominal']
OBJ = ['t_first_vis', 'dAdt_vis', 'hole_ins', 'line_def_final']
CHK = ['hole_growth_ray', 'bite_0.10', 'near_disp_1', 'open_3', 'gap_6', 'stem', 'hole_ray']
L = math.log(1.2 / 0.8)
PAR = {'Gamma': ('Gamma+', 'Gamma-', L), 'tau_w': ('tau_w+', 'tau_w-', L), 'r_e,ref': ('delta_e+', 'delta_e-', L * 1.0e-3 / 3.72e-3),
       'Pi_coh': ('Pi_coh+', 'Pi_coh-', L)}
NUI = {'alpha_F': ('alpha_F+', 'alpha_F-', L, 0.9), 'Q scale': ('Qs+', 'Qs-', L, 0.69), 'z_s low/0.5cm': ('zs+', 'zs-', 2.0, 1.0),
       'z_s thin/1cm': ('vthin+', 'vthin-', 2.0, 1.0), 'delta_c': ('delta_c+', 'delta_c-', math.log(3), 0.4),
       'c_t': ('c_t+', 'c_t-', math.log(3), 0.4), 'alpha_j': ('alpha_j+', 'alpha_j-', math.log(12 / 5), 0.25),
       'p_sub': ('p_sub+', 'p_sub-', math.log(4), 0.6)}
def sig(o, x):
    return {'t_first_vis': 0.12, 'dAdt_vis': 0.10 * abs(x), 'hole_ins': math.hypot(0.004, 0.10 * abs(x)),
            'line_def_final': 0.004}[o]
NREP = {'t_first_vis': 9, 'dAdt_vis': 2, 'hole_ins': 3, 'line_def_final': 9}
def der(spec, o):
    a, b, h = spec[:3]; return (R[a][o] - R[b][o]) / h
np.set_printoptions(precision=2, suppress=True, linewidth=180)
cols = list(PAR) + list(NUI)
print('relative sensitivity d ln O / d(ln theta) [z_s columns: per 0.5 cm low / 1 cm thin]')
print('%-15s' % 'obs' + ''.join('%11s' % c[:10] for c in cols) + '   nominal')
for o in OBJ + CHK:
    row = []
    for c in cols:
        spec = PAR.get(c) or NUI[c]
        d = der(spec, o); row.append(d / abs(n0[o]) if n0[o] else float('nan'))
    print('%-15s' % o + ''.join('%11.2f' % x for x in row) + '   %.4g' % n0[o])
A = np.array([[der(PAR[c], o) / sig(o, n0[o]) * math.sqrt(NREP[o]) for c in PAR] for o in OBJ])
F = A.T @ A; C = np.linalg.inv(F); sd = np.sqrt(np.diag(C)); corr = C / np.outer(sd, sd)
An = A / np.linalg.norm(A, axis=0); sv = np.linalg.svd(An, compute_uv=False)
B = np.array([[der(NUI[c], o) / sig(o, n0[o]) * math.sqrt(NREP[o]) for c in NUI] for o in OBJ])
pri = np.array([NUI[c][3] for c in NUI])
Z = np.hstack([A, B]); Fz = Z.T @ Z + np.diag(np.r_[np.zeros(len(PAR)), 1 / pri ** 2])
Cz = np.linalg.inv(Fz); sdz = np.sqrt(np.diag(Cz))[:len(PAR)]; cz = Cz[:len(PAR), :len(PAR)] / np.outer(sdz, sdz)
print('\nunweighted chi2, counts', NREP, ' column-normalised condition number %.2f' % (sv[0] / sv[-1]))
print('sd(ln theta) measurement only:', np.round(sd, 3), list(PAR))
print('corr (measurement only)\n', np.round(corr, 2))
print('sd(ln theta) + nuisances     :', np.round(sdz, 3))
print('corr (+ nuisances)\n', np.round(cz, 2))
# which nuisance dominates each parameter's inflation: drop one nuisance at a time
for k, c in enumerate(NUI):
    keep = [j for j in range(len(NUI)) if j != k]
    Z2 = np.hstack([A, B[:, keep]]); F2 = Z2.T @ Z2 + np.diag(np.r_[np.zeros(len(PAR)), 1 / pri[keep] ** 2])
    s2 = np.sqrt(np.diag(np.linalg.inv(F2)))[:len(PAR)]
    print('  without %-14s sd:' % c, np.round(s2, 3))
