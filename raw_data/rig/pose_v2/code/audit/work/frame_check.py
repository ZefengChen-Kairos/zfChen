"""A4: the delivered output frame reproduces the fitted model: CAD(z0 -29, az 90) posed with the npz poses vs the
tag-frame CAD (fitted placement) posed with the spline, world points max difference."""
import sys, json, pickle
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from spline import Traj
V2 = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2'
V = pickle.load(open(f'{V2}/out/v2_all.pkl', 'rb'))
PL = V['place']
cad_t = gcm.CAD(z0_mm=PL['z0'], az_deg=PL['az']); cad_t.set_place(PL['z0'], PL['az'], np.array(PL['extra']))
cad_o = gcm.CAD(z0_mm=-29.0, az_deg=90.0)
from scipy.spatial.transform import Rotation as Rot
for p in V['pours']:
    tr = V['pours'][p]; traj = Traj(tr['T0'], tr['dt'], tr['R'], tr['P'])
    Z = np.load(f'{V2}/out/{p}_pose_v2.npz')
    idx = np.linspace(0, len(Z['t']) - 1, 25).astype(int)
    Rt, pt = traj.eval(Z['t'][idx])
    err = []
    for j, i in enumerate(idx):
        Ro = Rot.from_rotvec(Z['poses'][i, :3]).as_matrix(); po = Z['poses'][i, 3:]
        a = cad_t.world(Rt[j], pt[j]); b = cad_o.world(Ro, po)
        err.append(np.abs(a - b).max())
        tipa = Rt[j] @ cad_t.tip_pf + pt[j]; tipb = Ro @ Z['tip_p'] + po
    print(p, 'max world point diff mm', round(max(err) * 1e3, 4), 'tip diff mm', round(np.linalg.norm(tipa - tipb) * 1e3, 4),
          'cad_o tip', (cad_o.tip_pf * 1e3).round(2), 'npz tip_p', (Z['tip_p'] * 1e3).round(2))
