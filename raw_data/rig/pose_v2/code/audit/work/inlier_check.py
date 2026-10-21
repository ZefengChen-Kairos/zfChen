"""A13: consumer filter check: fuse_occ.pose_inliers (win 0.3 s, tol 20 mm, local LINEAR fit of tip and origin) applied to
the delivered v2 tip.npz-format file, as fuse_occ.PourData does on load. Rejected samples are interpolated over (Slerp)."""
import sys, numpy as np
sys.path.insert(0, '/home/user/zfChen')
from latte_imex.rig.fuse_occ import pose_inliers
from scipy.spatial.transform import Rotation as Rot, Slerp
V2 = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2'
SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
UP = np.array([0, 0, -1.0])
for p in ['tulip', 'leaf', 'heart', 'push_heart', 'swan']:
    for name, f in (('v2', f'{V2}/out/{p}_pose_v2.npz'), ('cur', f'{SCR}/pours/{p}/tip.npz')):
        T = np.load(f); o = np.argsort(T['t']); t, P = T['t'][o], T['poses'][o]
        keep = pose_inliers(t, P, np.array(T['tip_p'], float))
        rej = t[~keep]
        msg = f'{p:10s} {name:3s} n {len(t):5d} rejected {(~keep).sum():4d}'
        if (~keep).any():
            # effect on tilt after interpolation over the rejected samples
            sl = Slerp(t[keep], Rot.from_rotvec(P[keep, :3]))
            tt = np.clip(t, t[keep][0], t[keep][-1])
            R2 = sl(tt).as_matrix(); tilt2 = np.degrees(np.arccos(np.clip(R2[:, :, 2] @ UP, -1, 1)))
            R1 = Rot.from_rotvec(P[:, :3]).as_matrix(); tilt1 = np.degrees(np.arccos(np.clip(R1[:, :, 2] @ UP, -1, 1)))
            d = np.abs(tilt2 - tilt1)
            segs = np.split(rej, np.where(np.diff(rej) > 0.05)[0] + 1)
            msg += f'; segments {[(round(s[0], 2), round(s[-1], 2)) for s in segs][:8]}; tilt change after interpolation max {d.max():.2f} deg'
        print(msg, flush=True)
