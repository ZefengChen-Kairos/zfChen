"""Cause of the camera-0 / camera-1 inconsistency: per pour, tags-only CT fits with
 base: tags 1/5/0 (phi,z,psi) + psi4 free;  A: + tag 1 / tag 0 body-frame 6-dof (u, tilts);  B: tags 4/5 fixed to the
 single-camera (cam-0) geometry, tag 1/0 (phi,z,psi) free + 6-dof world-frame correction of camera 1's extrinsics."""
import sys, pickle, json, numpy as np
from scipy.spatial.transform import Rotation as Rot
from pv2lib import *
from spline import Traj
from ctfit import PourProblem, Fit
T = pickle.load(open(f'{V2}/out/ct_tags.pkl', 'rb'))
c0 = [pickle.load(open(f'{V2}/out/layout_{q}_flat_cam0_edge.pkl', 'rb'))['layout'] for q in POURS]
res = {}
p = sys.argv[1]
tr = T['pours'][p]
for mode in ('base', 'A', 'B'):
    L = Layout.from_dict(T['layout'])
    D = PourData(p, 'edge', tags=set(L.tp))
    traj = Traj(tr['T0'], tr['dt'], tr['R'], tr['P'])
    L.ext_cen = traj.P.mean(0)
    P = PourProblem(D, traj, D.rows, None, nuis_free=('c1',)); P.nuis['c1'] = tr['nuis']['c1']
    tags_here = sorted({d['tag'] for d in D.rows})
    oth = [t for t in (0, 1) if t in tags_here]
    if mode == 'base':
        lfree = [(4, 2)] + [(t, j) for t in oth + [5] for j in range(3)]
    elif mode == 'A':
        lfree = [(4, 2)] + [(t, j) for t in [5] for j in range(3)] + [(t, j) for t in oth for j in range(6)]
    else:
        L.tp[4] = np.r_[np.mean([d['tp'][4] for d in c0], 0)[:4], 0, 0]; L.tp[5] = np.r_[np.mean([d['tp'][5] for d in c0], 0)[:4], 0, 0]
        L.tp[4][[0, 1, 3]] = 0
        lfree = [(t, j) for t in oth for j in range(3)] + [('e1', j) for j in range(6)]
    F = Fit([P], L, sig_t=T['sig_t'], k_t=T['k_t'], sig_alpha=T['sig_alpha'], sig_a=T['sig_a'], layout_free=lfree, use_depth=False)
    F.run(iters=100, tol=1e-10, verbose=False)
    r = F._frame_tags(P, F.L, False)[0] * F.sig_t
    rms = {c: round(float(np.sqrt(np.mean(r[P.tg['cam'] == c] ** 2))), 3) for c in (0, 1)}
    c = F.cost(P, F.L)
    res[mode] = dict(rms=rms, cost=c, c1_ms=P.nuis['c1'] * 1e3, layout=F.L.pretty(), ext1=F.L.ext1.tolist(), ext_cen=F.L.ext_cen.tolist())
    print(p, mode, 'rms', rms, 'cost', {k: round(v, 1) for k, v in c.items()}, 'c1 ms', round(P.nuis['c1'] * 1e3, 2), 'layout', F.L.pretty(), flush=True)
json.dump(res, open(f'{V2}/out/diag_conflict_{p}.json', 'w'), indent=1, default=float)
