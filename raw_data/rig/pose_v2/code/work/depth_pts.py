"""Observed pitcher depth points of cameras 0 and 1 (camera 2 never read) for the CT fit.
For every frame: the audit's masks (pose_audit/geom/work/gcm.observe: ROI around the rendered CAD, liquid plane,
cup-rim zone from disc.json, stream capsule, handle zone, occluders in front of the body) evaluated at the given
trajectory and CAD placement; points = gcm.icp_points (no lid / opening, no handle zone), stride 2, at most NMAX per
frame (random, fixed seed); stored in the WORLD (board) frame with the frame's camera, step and nominal time.
    python3 depth_pts.py <pour> <traj.pkl> <z0_mm> <az_deg> <out.npz> [nmax]"""
import sys, pickle, time
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from spline import Traj

p, trf, z0, az, outf = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
NMAX = int(sys.argv[6]) if len(sys.argv) > 6 else 1500
EXTRA_RX = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0       # CAD axis tilt about the tag-frame x [deg] (placement)
T = pickle.load(open(trf, 'rb'))
from scipy.spatial.transform import Rotation as Rot
e1 = np.array(T['layout'].get('ext1', np.zeros(6)), float); cen = np.array(T['layout'].get('ext_cen', np.zeros(3)), float)
Re = Rot.from_rotvec(e1[:3]).as_matrix()          # camera-1 correction C: true world -> camera-1 'believed' world
tr = T['pours'][p]
traj = Traj(tr['T0'], tr['dt'], tr['R'], tr['P'])
off = tr['nuis']
P = gcm.Pour(p)
cad = gcm.CAD(z0_mm=z0, az_deg=az)
if EXTRA_RX:
    Ex = np.eye(4); Ex[:3, :3] = Rot.from_euler('x', EXTRA_RX, degrees=True).as_matrix()
    cad.set_place(z0, az, Ex)
P.tip_p = cad.tip_pf.copy()                         # stream capsule from the CAD spout tip in this frame
rng = np.random.default_rng(0)
X, pf, fcam, fstep, ft, fn_all = [], [], [], [], [], []
t0 = time.time()
nf = 0
for c in (0, 1):
    for k, ts in zip(P.tk.steps[c], P.tk.ts[c] - P.tk.t0):
        if T.get('depth_offsets'):                  # offsets of the depth streams estimated by a depth fit
            tt = ts + off[f'd{c}']
        else:                                       # tags-only trajectory: depth assumed synchronous with colour
            tt = ts + (off.get('c1', 0.0) if c == 1 else 0.0)
        if tt < traj.T0 or tt > traj.t_end:
            continue
        R, pp = traj.eval(np.array([tt]))
        R, pp = R[0], pp[0]
        if c == 1:                                   # render / mask in camera 1's believed world
            R, pp = Re @ R, Re @ (pp - cen) + cen + e1[3:]
        O = gcm.observe(P, cad, c, k, tt, R, pp)
        if O['body'].sum() < 300:
            continue
        Q = gcm.icp_points(P, c, O, R, pp, stride=2)
        if len(Q) < 50:
            continue
        fn_all.append(len(Q))
        if len(Q) > NMAX:
            Q = Q[rng.choice(len(Q), NMAX, replace=False)]
        Xw = Q @ R.T + pp
        if c == 1:                                   # believed world -> true world (C^-1)
            Xw = (Xw - cen - e1[3:]) @ Re + cen
        X.append(Xw.astype(np.float32)); pf.append(np.full(len(Q), nf, np.int32))
        fcam.append(c); fstep.append(int(k)); ft.append(float(ts)); nf += 1
        if nf % 50 == 0:
            print(p, 'frames', nf, f'{time.time() - t0:.0f}s', flush=True)
np.savez_compressed(outf, X=np.concatenate(X), pf=np.concatenate(pf), fcam=np.array(fcam), fstep=np.array(fstep),
                    ft=np.array(ft), n_raw=np.array(fn_all), z0=z0, az=az)
print('done', p, nf, 'frames', sum(len(x) for x in X), 'points', f'{time.time() - t0:.0f}s', flush=True)
