"""A8: cross-camera disagreement (cam-0-only vs cam-1-only refits, global calibration shared) vs angular speed of v2:
tilt difference and spout-tip distance, binned by |omega|; plus the lag that would explain the fast-bin tilt difference."""
import numpy as np, pickle, sys, json
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from spline import Traj
from pv2lib import tilt_az
from scipy.spatial.transform import Rotation as Rot
V2 = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2'
res = {}
allb = {}
for p in ['tulip', 'leaf', 'heart', 'push_heart', 'swan']:
    Z = np.load(f'{V2}/out/{p}_pose_v2.npz'); B = Z['B_out_to_tag']; tip = Z['tip_p_tagframe']
    tr = {k: pickle.load(open(f'{V2}/out/full_{p}_v2k{s}.pkl', 'rb')) for k, s in (('a', '_cam0'), ('b', '_cam1'), ('j', ''))}
    tr = {k: Traj(v['T0'], v['dt'], v['R'], v['P']) for k, v in tr.items()}
    # restrict to the times both cameras observe (first..last frame of each camera) to avoid extrapolated ends
    tg = np.arange(Z['t'][0] + 0.05, Z['t'][-1] - 0.05, 0.005)
    Rj, pj = tr['j'].eval(tg)
    w = np.r_[np.degrees(np.linalg.norm((Rot.from_matrix(Rj[:-1]).inv() * Rot.from_matrix(Rj[1:])).as_rotvec(), axis=1)) / 0.005, 0]
    Ra, pa = tr['a'].eval(tg); Rb, pb = tr['b'].eval(tg)
    ta, _ = tilt_az(Ra @ B[:3, :3]); tb, _ = tilt_az(Rb @ B[:3, :3]); tj, _ = tilt_az(Rj @ B[:3, :3])
    dtip = np.linalg.norm((np.einsum('nij,j->ni', Ra, tip) + pa) - (np.einsum('nij,j->ni', Rb, tip) + pb), axis=1) * 1e3
    dth = np.gradient(tj, tg)
    out = {}
    for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 1000)):
        m = (w >= lo) & (w < hi)
        if m.sum() < 10: continue
        d = ta[m] - tb[m]
        out[f'w{lo}-{hi}'] = dict(n_s=round(m.sum() * 0.005, 2), tilt_med=round(float(np.median(d)), 2), tilt_p90abs=round(float(np.percentile(np.abs(d), 90)), 2),
                                  tilt_max=round(float(np.abs(d).max()), 2), tip_p50=round(float(np.median(dtip[m])), 2), tip_p90=round(float(np.percentile(dtip[m], 90)), 2))
        allb.setdefault(f'w{lo}-{hi}', []).append(np.abs(d))
    res[p] = out
    print(p, json.dumps(out))
for k, v in allb.items():
    v = np.concatenate(v); print('ALL', k, 'n_s', round(len(v) * 0.005, 1), 'tilt |diff| p50/p90/max', np.percentile(v, [50, 90, 100]).round(2))
json.dump(res, open(f'{V2}/audit/out/xcam_speed.json', 'w'), indent=1)
