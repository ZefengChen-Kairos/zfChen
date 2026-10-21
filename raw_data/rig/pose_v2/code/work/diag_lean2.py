import sys, pickle, numpy as np
from pv2lib import *
from spline import Traj
from ctfit import PourProblem, Fit
T = pickle.load(open(f'{V2}/out/ct_tags.pkl', 'rb'))
for p in sys.argv[1:]:
    tr = T['pours'][p]
    L = Layout.from_dict(T['layout'])
    for t in L.tp: L.tp[t][4] = np.radians(-1.5)
    D = PourData(p, 'edge', tags=set(L.tp))
    P = PourProblem(D, Traj(tr['T0'], tr['dt'], tr['R'], tr['P']), D.rows, None, nuis_free=('c1',)); P.nuis['c1'] = tr['nuis']['c1']
    oth = [t for t in (0, 1) if t in {d['tag'] for d in D.rows}]
    lfree = [(4, 2), ('lean', 0)] + [(t, j) for t in oth + [5] for j in range(3)]
    F = Fit([P], L, sig_t=T['sig_t'], k_t=T['k_t'], sig_alpha=T['sig_alpha'], sig_a=T['sig_a'], layout_free=lfree, use_depth=False)
    F.run(iters=100, tol=1e-10, verbose=False)
    r = F._frame_tags(P, F.L, False)[0] * F.sig_t
    print(p, 'free lean deg', round(np.degrees(F.L.tp[4][4]), 2), 'rms', {c: round(float(np.sqrt(np.mean(r[P.tg['cam'] == c] ** 2))), 3) for c in (0, 1)},
          'tag cost', round(F.cost(P, F.L)['tag'], 1), 'c1 ms', round(P.nuis['c1'] * 1e3, 1), 'layout', F.L.pretty(), flush=True)
