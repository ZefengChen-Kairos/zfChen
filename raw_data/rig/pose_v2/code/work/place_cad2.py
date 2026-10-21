"""CAD placement in the v2 tag frame (spout azimuth FIXED at 77.14 deg = 3 flats from tag 4; spin about the jug axis is
not determined by depth): z0 profile, then a spin-locked robust ICP correction (axis tilt about x / y, translation)
of the accumulated camera-0/1 depth points (camera-1 points already corrected by the fitted camera-1 extrinsic),
per pour and pooled.  Writes ../out/placement.json (pooled; used by the full fit).
    python3 place_cad2.py <traj.pkl> <depth tag> [rx0_deg]"""
import sys, json, pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import V2, POURS
from place_cad import pitcher_points, score

AZ = 77.14
trf, tag = sys.argv[1], sys.argv[2]
rx0 = float(sys.argv[3]) if len(sys.argv) > 3 else -3.0
OUTNAME = sys.argv[4] if len(sys.argv) > 4 else 'placement'
T = pickle.load(open(trf, 'rb'))
Ex = np.eye(4); Ex[:3, :3] = Rot.from_euler('x', rx0, degrees=True).as_matrix()
cad = gcm.CAD(z0_mm=-30.0, az_deg=AZ)
out = {}
allQ = []
for p in POURS:
    Q = pitcher_points(p, T, tag)
    Q = Q[np.random.default_rng(0).choice(len(Q), min(len(Q), 120000), replace=False)]
    allQ.append(Q[::3])
    zs = np.arange(-34.0, -25.9, 0.5)
    prof = []
    for z0 in zs:
        cad.set_place(z0, AZ, Ex); prof.append(score(cad, Q)[0])
    z0b = float(zs[int(np.argmin(prof))])
    cad.set_place(z0b, AZ, Ex)
    D, st = gcm.icp(cad, Q[::2], iters=40, lock_spin=True)
    de = gcm.describe(D, cad)
    out[p] = dict(z0=z0b, D=D.tolist(), rot_xyz_deg=[de['rot_x_deg'], de['rot_y_deg'], de['rot_z_deg']], centre_mm=de['tr_centre'],
                  tip_mm=de['tr_tip_mm'], rms_mm=st['rms_mm'], med_abs_mm=st['med_abs_mm'])
    print(p, 'z0', z0b, 'correction on top of rx0: rot xyz', np.round(out[p]['rot_xyz_deg'], 2), 'centre mm', np.round(de['tr_centre'], 2),
          'rms', round(st['rms_mm'], 3), 'med', round(st['med_abs_mm'], 3), flush=True)
Q = np.concatenate(allQ)
zs = np.arange(-34.0, -25.9, 0.5); prof = []
for z0 in zs:
    cad.set_place(z0, AZ, Ex); prof.append(score(cad, Q)[0])
z0b = float(zs[int(np.argmin(prof))])
cad.set_place(z0b, AZ, Ex)
D, st = gcm.icp(cad, Q, iters=60, lock_spin=True)
de = gcm.describe(D, cad)
extra = D @ Ex
cad.set_place(z0b, AZ, extra)
print('POOLED z0', z0b, 'rot xyz on top', np.round([de['rot_x_deg'], de['rot_y_deg'], de['rot_z_deg']], 2), 'centre', np.round(de['tr_centre'], 2),
      'rms', round(st['rms_mm'], 3), 'tip in tag frame mm', np.round(cad.tip_pf * 1e3, 1))
out['pooled'] = dict(z0=z0b, az=AZ, extra=extra.tolist(), rms_mm=st['rms_mm'], tip_tagframe_mm=(cad.tip_pf * 1e3).tolist())
json.dump(out, open(f'{V2}/out/{OUTNAME}_detail_{tag}.json', 'w'), indent=1, default=float)
json.dump(dict(z0=z0b, az=AZ, extra=extra.tolist(), source=f'place_cad2.py {trf} {tag}'), open(f'{V2}/out/{OUTNAME}.json', 'w'), indent=1)
