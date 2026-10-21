"""A5: leave-one-camera-out re-evaluation from the builder's single-camera refits (full_<p>_v2k_cam<k>.pkl):
held-out corners (median per-detection corner error, px) with the jointly fitted camera-1 extrinsic correction (as the
builder) and WITHOUT it (raw calibration, comparable to the audit's current-chain LOCO); cross-camera tip / tilt bias;
time lag between the cam-0-only and cam-1-only trajectories (residual camera-1 clock error) from the fast segments."""
import sys, json, pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from pv2lib import *
from spline import Traj
from ctfit import tag_fun
import jax.numpy as jnp
A = f'{V2}/audit'
res = {}
def tr_of(d): return Traj(d['T0'], d['dt'], d['R'], d['P'])
for p in POURS:
    F = pickle.load(open(f'{V2}/out/full_{p}_v2k.pkl', 'rb'))
    L = Layout.from_dict(F['layout'])
    D = PourData(p, 'edge', tags=set(L.tp))
    f_, _ = tag_fun(L.a, L.flat)
    Z = np.load(f'{V2}/out/{p}_pose_v2.npz'); tip_tag = Z['tip_p_tagframe']
    c1 = F['nuis']['c1']
    fits = {k: tr_of(pickle.load(open(f'{V2}/out/full_{p}_v2k_cam{k}.pkl', 'rb'))) for k in (0, 1)}
    fits['joint'] = tr_of(F)
    R = {}
    for held, src in ((1, 0), (0, 1), (0, 'joint'), (1, 'joint')):
        to = fits[src]
        rows = [d for d in D.rows if d['cam'] == held]
        for use_e in (True, False):
            if src == 'joint' and not use_e:
                continue
            e = L.ext1 if use_e else np.zeros(6)
            errs = []
            for d in rows:
                t = d['t'] + (c1 if held == 1 else 0.0)
                if t < to.T0 or t > to.t_end: continue
                Rv, pv = to.eval(np.array([t]))
                if use_e:     # fitted correction: camera-1 observations see C(X); camera 0 sees X
                    w1 = float(held == 1); ee = L.ext1
                else:         # raw calibration: a fit WITHOUT the correction puts the trajectory in the believed world of
                              # its camera; cam-1-only -> C(X) seen by cam 0; cam-0-only -> X seen by cam 1
                    w1 = float(src == 1); ee = L.ext1
                r = np.asarray(f_(jnp.asarray(Rv), jnp.asarray(pv), jnp.zeros(6), jnp.asarray(L.tp[d['tag']][None]), L.s, D.W2C[held][None],
                                  D.Kv[held][None], d['q'][None], jnp.asarray(ee), np.array([w1]), jnp.asarray(L.ext_cen))).reshape(4, 2)
                errs.append(np.median(np.linalg.norm(r, axis=1)))
            errs = np.array(errs)
            R[f'cam{held}_from_{src}' + ('' if use_e else '_noext1')] = [round(float(np.percentile(errs, q)), 2) for q in (50, 90)] + [len(errs)]
    # cross-camera agreement and lag
    a, b = fits[0], fits[1]
    tg = np.arange(max(a.T0, b.T0) + 0.1, min(a.t_end, b.t_end) - 0.1, 0.005)
    Ra, pa = a.eval(tg)
    tipa = np.einsum('nij,j->ni', Ra, tip_tag) + pa
    w = np.linalg.norm((Rot.from_matrix(Ra[:-1]).inv() * Rot.from_matrix(Ra[1:])).as_rotvec(), axis=1) / 0.005
    fast = np.r_[np.degrees(w) > 10, False]
    lags = np.arange(-0.060, 0.0601, 0.0025); cost = []
    for lag in lags:
        Rb, pb = b.eval(np.clip(tg + lag, b.T0, b.t_end - 1e-6))
        tipb = np.einsum('nij,j->ni', Rb, tip_tag) + pb
        dt_ = tipa - tipb; dt_ -= np.median(dt_[fast], 0)
        cost.append(float(np.median(np.linalg.norm(dt_[fast], axis=1))) * 1e3)
    j = int(np.argmin(cost))
    Rb, pb = b.eval(tg); tipb = np.einsum('nij,j->ni', Rb, tip_tag) + pb
    ta, _ = tilt_az(Ra); tb, _ = tilt_az(Rb)
    R['lag_cam1fit_minus_cam0fit_ms'] = round(float(lags[j] * 1e3), 1)
    R['lag_cost_mm'] = dict(at0=round(cost[len(lags) // 2], 3), best=round(cost[j], 3), curve={f'{l*1e3:+.0f}': round(c, 3) for l, c in zip(lags[::4], cost[::4])})
    R['n_fast_5ms'] = int(fast.sum())
    R['tip_bias_vec_mm'] = (np.median(tipa - tipb, 0) * 1e3).round(2).tolist()
    R['tip_absdiff_p50_p90_mm'] = np.percentile(np.linalg.norm(tipa - tipb, axis=1) * 1e3, [50, 90]).round(2).tolist()
    R['tilt_diff_med_p90abs_deg'] = [round(float(np.median(ta - tb)), 2), round(float(np.percentile(np.abs(ta - tb), 90)), 2)]
    res[p] = R
    print(p, json.dumps(R), flush=True)
json.dump(res, open(f'{A}/out/loco_check.json', 'w'), indent=1)
