"""Leave-one-camera-out refit with the joint model (layout, camera-1 correction, CAD placement fixed from joint.pkl):
the held-out camera's tags AND depth are removed; the trajectory is initialised from the remaining camera's own
per-sample poses (Viterbi-cleaned), so the held-out camera never influences it.  Camera-1 time offset fixed at the
joint value (camera 0 is the time origin).
    python3 loco_fit.py <pour> <kept cam> -> ../out/full_<pour>_v2j_cam<kept>.pkl"""
import sys, time, pickle, json
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import *
from spline import Traj
from ctfit import PourProblem, Fit
from init_track import viterbi_samples, slerp_fn
from fit_joint import load_depth_raw
from fit_full import neff_from_floor, SIG_D

p, keep = sys.argv[1], int(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else 'v2j'
J = pickle.load(open(f'{V2}/out/joint_{OUT}.pkl' if OUT != 'v2j' else f'{V2}/out/joint.pkl', 'rb'))
Told = pickle.load(open(f'{V2}/out/ct_tags_lean_e1.pkl', 'rb'))
PL = J['place']
L = Layout.from_dict(J['layout']); L.pl = np.zeros(6)            # placement correction already composed into PL
cad = gcm.CAD(z0_mm=PL['z0'], az_deg=PL['az']); cad.set_place(PL['z0'], PL['az'], np.array(PL['extra']))
jr = J['pours'][p]
D = PourData(p, 'edge', tags=set(L.tp))
D.rows = [d for d in D.rows if d['cam'] == keep]
samples, tt, by = build_samples(D, cams=(keep,))
X, path, cands = viterbi_samples(D, L, samples, tt, by, sig_t=J['sig_t'], verbose=True)
if keep == 1:                                                    # camera-1 samples: correct their frame times
    tt = tt + jr['nuis']['c1']
f = slerp_fn(tt, X)
traj = Traj(jr['T0'], jr['dt'], *f(Traj(jr['T0'], jr['dt'], jr['R'], jr['P']).ctrl_times()))
dp = load_depth_raw(p, 't1', Layout.from_dict(Told['layout']), (keep,))
P = PourProblem(D, traj, D.rows, dp, nuis_free=(), cad=cad)
P.nuis['c1'] = jr['nuis']['c1']
P.associate(L)
P.neff, _ = neff_from_floor(P, cad, PL['az'])
F = Fit([P], L, sig_t=J['sig_t'], k_t=1.5, sig_d=SIG_D, c_d=0.008, sig_alpha=J['sig_alpha'], sig_a=J['sig_a'], layout_free=(), use_depth=True)
t0 = time.time()
F.run(iters=60, tol=1e-9, outer=4, verbose=False, c_sched=(0.010, 0.008, 0.006, 0.005))
rt = F._frame_tags(P, L, False)[0] * F.sig_t
rd, _ = F._depth(P, False, L)
out = dict(pour=p, cams=(keep,), T0=P.traj.T0, dt=P.traj.dt, R=P.traj.R, P=P.traj.P, nuis=dict(P.nuis), cost=F.cost(P, L), neff=P.neff,
           layout=L.as_dict(), place=PL, tag_res=rt, tag_cam=P.tg['cam'], tag_id=P.tg['tag'], tag_step=P.tg['step'])
pickle.dump(out, open(f'{V2}/out/full_{p}_{OUT}_cam{keep}.pkl', 'wb'))
md = P.dp['assoc']
print(p, 'kept cam', keep, 'tag rms px', round(float(np.sqrt(np.mean(rt ** 2))), 3), 'depth med|r| mm', round(float(np.median(np.abs(rd[md]))) * 1e3, 3),
      f'{time.time() - t0:.0f}s', flush=True)
