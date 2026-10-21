"""Per-detection flip check of v2 with TIMES (the builder's eval_misc.py loop kept only counts).
For every camera-0/1 tag detection: v2's corner cost c_v2 (sum r^2, px^2) at that time, the single-detection pose
branches (multi-start Gauss-Newton from both IPPE solutions, fixed v2 layout), and
  * ambiguous: the detection alone has a second branch > 10 deg from its best one with cost < 4x the best
               (it cannot tell the two branches apart; continuity / other camera / depth must decide);
  * suspect:   eval_misc criterion: a branch > 10 deg from v2 fits > 4x better than v2 and v2's rms > 0.6 px
               (v2 possibly on the wrong branch at that detection = flip not resolved).
    python3 flip_times.py <pour> -> ../out/flips_<pour>.json"""
import sys, json, pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from pv2lib import *
from spline import Traj
from ctfit import tag_fun
import jax.numpy as jnp

p = sys.argv[1]
OUT = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/report/out'
F = pickle.load(open(f'{V2}/out/full_{p}_v2k.pkl', 'rb'))
tr = Traj(F['T0'], F['dt'], F['R'], F['P'])
L = Layout.from_dict(F['layout'])
D = PourData(p, 'edge', tags=set(L.tp))
f_, _ = tag_fun(L.a, L.flat)
c1 = F['nuis']['c1']
ntag = {}
for d in D.rows:
    ntag[(d['cam'], d['step'])] = ntag.get((d['cam'], d['step']), 0) + 1
rows = []
for d in D.rows:
    t = d['t'] + (c1 if d['cam'] == 1 else 0.0)
    if t < tr.T0 or t > tr.t_end:
        continue
    Rv, pv = tr.eval(np.array([t]))
    r = np.asarray(f_(jnp.asarray(Rv), jnp.asarray(pv), jnp.zeros(6), jnp.asarray(L.tp[d['tag']][None]), L.s,
                      D.W2C[d['cam']][None], D.Kv[d['cam']][None], d['q'][None], jnp.asarray(L.ext1),
                      np.array([float(d['cam'] == 1)]), jnp.asarray(L.ext_cen)))
    c_v2 = float((r ** 2).sum())
    sols = frame_candidates(D, L, [d], iters=30)
    x2 = np.r_[Rot.from_matrix(Rv[0]).as_rotvec(), pv[0]]
    alt = [s for s in sols if rot_deg(s[1], x2) > 10]
    suspect = bool(alt and 2 * alt[0][0] * 2 < c_v2 * 0.5 and c_v2 > 8 * 0.3 ** 2 * 4)
    amb = False
    if len(sols) >= 2:
        b = sols[0]
        sec = [s for s in sols[1:] if rot_deg(s[1], b[1]) > 10]
        amb = bool(sec and sec[0][0] < 4 * max(b[0], 1e-9) + 0.5)
    rows.append(dict(t=float(t), cam=int(d['cam']), tag=int(d['tag']), ntag_frame=int(ntag[(d['cam'], d['step'])]),
                     c_v2=c_v2, c_alt=float(2 * alt[0][0]) if alt else None, suspect=suspect, ambiguous=amb))
json.dump(rows, open(f'{OUT}/flips_{p}.json', 'w'))
print(p, len(rows), 'suspect', sum(r['suspect'] for r in rows), 'ambiguous', sum(r['ambiguous'] for r in rows),
      'ambiguous single-tag frames', sum(r['ambiguous'] and r['ntag_frame'] == 1 for r in rows))
