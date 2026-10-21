import sys, pickle, numpy as np, json
from scipy.spatial.transform import Rotation as Rot
from pv2lib import *
from spline import Traj
from ctfit import PourProblem, Fit
T = pickle.load(open(f'{V2}/out/ct_tags.pkl', 'rb'))
LEAN = np.radians(-2.41)
CEN = np.array([0.15, 0.22, -0.03])         # common rotation centre for the extrinsic correction (near the jug) [board]
out = {}
for p in sys.argv[1:]:
    tr = T['pours'][p]
    for mode in ('A1', 'B1'):
        L = Layout.from_dict(T['layout'])
        for t in L.tp: L.tp[t][4] = LEAN
        L.ext_cen = CEN.copy()
        D = PourData(p, 'edge', tags=set(L.tp))
        P = PourProblem(D, Traj(tr['T0'], tr['dt'], tr['R'], tr['P']), D.rows, None, nuis_free=('c1',)); P.nuis['c1'] = tr['nuis']['c1']
        oth = [t for t in (0, 1) if t in {d['tag'] for d in D.rows}]
        lfree = [(4, 2)] + [(5, j) for j in range(3)] + [(t, j) for t in oth for j in range(3)]
        if mode == 'A1':
            lfree += [(1, 3), (1, 4), (1, 5)]
        else:
            lfree += [('e1', j) for j in range(6)]
        F = Fit([P], L, sig_t=T['sig_t'], k_t=T['k_t'], sig_alpha=T['sig_alpha'], sig_a=T['sig_a'], layout_free=lfree, use_depth=False)
        F.run(iters=150, tol=1e-11, verbose=False)
        r = F._frame_tags(P, F.L, False)[0] * F.sig_t
        rms = {c: round(float(np.sqrt(np.mean(r[P.tg['cam'] == c] ** 2))), 3) for c in (0, 1)}
        out[f'{p}/{mode}'] = dict(rms=rms, c1=P.nuis['c1'], layout=F.L.pretty(), ext1=F.L.ext1.tolist(), tag_cost=F.cost(P, F.L)['tag'])
        print(p, mode, 'rms', rms, 'cost', round(F.cost(P, F.L)['tag'], 1), 'c1 ms', round(P.nuis['c1'] * 1e3, 1), 'tag1', F.L.pretty()[1],
              'ext1 deg/mm', [round(np.degrees(x), 2) for x in F.L.ext1[:3]] + [round(x * 1e3, 1) for x in F.L.ext1[3:]] if mode == 'B1' else '', flush=True)
json.dump(out, open(f'{V2}/out/diag_ab.json', 'w'), indent=1, default=float)
