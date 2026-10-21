"""A2: fair baseline for the held-out camera-2 check: the CURRENT track (tip.npz minus pose_inliers, as gcm.Pour.track)
gets its own best constant CAD placement (spin-locked robust point-to-plane ICP, z0 -28.5 / az 90 start) from the
camera-0/1 depth points only (raw calibration world, i.e. camera-1 points WITHOUT v2's extrinsic correction).
Per pour (most generous to the current track) and pooled. -> ../out/cur_place.json"""
import sys, json, pickle
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import V2, POURS, Layout
from fit_joint import load_depth_raw
A = f'{V2}/audit'
Lold = Layout.from_dict(pickle.load(open(f'{V2}/out/ct_tags_lean_e1.pkl', 'rb'))['layout'])
cad = gcm.CAD(z0_mm=-28.5, az_deg=90.0)
out, allQ = {}, []
for p in POURS:
    P = gcm.Pour(p)
    dp = load_depth_raw(p, 't1', Lold, (0, 1))
    Q = []
    for f in range(len(dp['ft'])):
        m = dp['pf'] == f
        R, pp = P.track(dp['ft'][f])
        Q.append((dp['Xraw'][m] - pp) @ R)
    Q = np.concatenate(Q)
    Q = Q[np.random.default_rng(0).choice(len(Q), min(len(Q), 60000), replace=False)]
    allQ.append(Q[::2])
    cad.set_place(-28.5, 90.0)
    D, st = gcm.icp(cad, Q, iters=40, lock_spin=True)
    de = gcm.describe(D, cad)
    out[p] = dict(D=D.tolist(), rot_xyz_deg=[de['rot_x_deg'], de['rot_y_deg'], de['rot_z_deg']], centre_mm=de['tr_centre'], tip_mm=de['tr_tip_mm'],
                  rms_mm=st['rms_mm'], med_abs_mm=st['med_abs_mm'])
    print(p, 'rot xyz deg', np.round(out[p]['rot_xyz_deg'], 2), 'centre mm', np.round(de['tr_centre'], 2), 'tip mm', round(de['tr_tip_mm'], 2),
          'rms', round(st['rms_mm'], 3), 'med', round(st['med_abs_mm'], 3), flush=True)
Q = np.concatenate(allQ)
cad.set_place(-28.5, 90.0)
D, st = gcm.icp(cad, Q, iters=40, lock_spin=True); de = gcm.describe(D, cad)
out['pooled'] = dict(D=D.tolist(), rot_xyz_deg=[de['rot_x_deg'], de['rot_y_deg'], de['rot_z_deg']], centre_mm=de['tr_centre'], rms_mm=st['rms_mm'])
print('pooled rot', np.round(out['pooled']['rot_xyz_deg'], 2), 'centre', np.round(de['tr_centre'], 2), 'rms', round(st['rms_mm'], 3))
json.dump(out, open(f'{A}/out/cur_place.json', 'w'), indent=1, default=float)
