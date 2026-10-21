import sys, pickle, numpy as np
from pv2lib import *
from spline import Traj
from ctfit import PourProblem, Fit
T = pickle.load(open(f'{V2}/out/ct_tags.pkl', 'rb'))
p = sys.argv[1]; mode = sys.argv[2]; fac = float(sys.argv[3])
L0 = Layout.from_dict(T['layout'])
c0 = [pickle.load(open(f'{V2}/out/layout_{q}_flat_cam0_edge.pkl', 'rb'))['layout'] for q in POURS]
tr = T['pours'][p]
D = PourData(p, 'edge', tags=set(L0.tp))
cams = (0,) if mode == 'cam0' else (1,) if mode == 'cam1' else (0, 1)
D.rows = [d for d in D.rows if d['cam'] in cams]
L = L0.copy()
if mode == 'cam0':
    L.tp[4] = np.mean([d['tp'][4] for d in c0], 0); L.tp[5] = np.mean([d['tp'][5] for d in c0], 0); L.tp[4][[0, 1, 3]] = 0
traj = Traj(tr['T0'], tr['dt'], tr['R'], tr['P'])
P = PourProblem(D, traj, D.rows, None, nuis_free=('c1',) if cams == (0, 1) else ())
P.nuis['c1'] = tr['nuis']['c1']
lfree = [(4, 2)] + [(t, j) for t in (0, 1, 5) for j in range(3)] if mode == 'both_free' else []
F = Fit([P], L, sig_t=T['sig_t'], k_t=T['k_t'], sig_alpha=T['sig_alpha'] * fac, sig_a=T['sig_a'] * fac, layout_free=lfree, use_depth=False)
F.run(iters=80, tol=1e-10, verbose=False)
r = F._frame_tags(P, F.L, False)[0] * F.sig_t
print(p, mode, 'prior x', fac, {c: round(float(np.sqrt(np.mean(r[P.tg['cam'] == c] ** 2))), 3) for c in cams},
      'med', {c: round(float(np.median(np.abs(r[P.tg['cam'] == c]))), 3) for c in cams}, 'c1 ms', round(P.nuis['c1'] * 1e3, 2),
      'layout', F.L.pretty() if lfree else '', flush=True)
