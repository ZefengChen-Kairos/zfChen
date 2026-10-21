"""Held-out camera 2, calibration-robust: camera 2's own extrinsic error (the camera-1 / camera-0 inconsistency was
1.7 deg) makes its raw depth residual a weak absolute judge.  For a pose source: (1) camera-2 pitcher points of every
frame (audit masks at that source's pose); (2) ONE constant world-frame rigid correction C2 of camera 2 fitted to all
frames of the pour (robust point-to-plane to the posed CAD) -- it absorbs any constant calibration offset and cannot
favour the source's time variation; (3) per frame, a spin-locked single-view ICP of the corrected points -> the
per-frame correction the source still needs: rotation about the pouring axis (tilt), total rotation, spout-tip
displacement.  Same masks / CAD / ICP for every source.
    python3 cam2_icp.py <pour> -> ../out/cam2icp_<pour>.json"""
import sys, json, pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import V2
from heldout_cam import v2_source

p = sys.argv[1]
V = pickle.load(open(f'{V2}/out/v2_all.pkl', 'rb'))
P = gcm.Pour(p)
pose_v, cad_v, rng_v = v2_source(p, V)
cad_c = gcm.CAD(z0_mm=-28.5, az_deg=90.0)
a_ = np.radians(V['place']['az'])
spout_v = np.array(V['place']['extra'])[:3, :3] @ np.array([np.cos(a_), np.sin(a_), 0.0])
SRC = dict(cur=(P.track, cad_c, P.tip_p.copy(), np.array([0, 1.0, 0])), v2=(pose_v, cad_v, cad_v.tip_pf.copy(), spout_v))
CEN = np.array([0.15, 0.22, -0.03])
out = {}
for name, (pose, cad, tip_p, spout) in SRC.items():
    P.tip_p = tip_p
    frames = []
    for k, ts in zip(P.tk.steps[2], P.tk.ts[2] - P.tk.t0):
        R, pp = pose(ts)
        O = gcm.observe(P, cad, 2, k, ts, R, pp)
        if O['body'].sum() < 2000:
            continue
        Q = gcm.icp_points(P, 2, O, R, pp, stride=2)
        if len(Q) < 400:
            continue
        Q = Q[np.random.default_rng(int(k)).choice(len(Q), min(len(Q), 2500), replace=False)]
        frames.append((float(ts), R, pp, Q @ R.T + pp))                    # world points
    # (2) constant correction C2: X_true = R2^T (X_b - CEN - t2) + CEN  (small-angle Gauss-Newton, Tukey 5 mm)
    x = np.zeros(6)
    for it in range(12):
        Rc = Rot.from_rotvec(x[:3]).as_matrix()
        H = np.zeros((6, 6)); g = np.zeros(6)
        for ts, R, pp, Xw in frames:
            Y = Xw - CEN - x[3:]
            Xt = Y @ Rc + CEN
            y = (Xt - pp) @ R
            d, j = cad.kd.query(y, distance_upper_bound=0.02)
            ok = np.isfinite(d)
            if ok.sum() < 50:
                continue
            j = j[ok]; n = cad.Np[j]; s = cad.Sp[j]
            r = np.einsum('ij,ij->i', n, y[ok] - s)
            w = np.where(np.abs(r) < 0.005, (1 - (r / 0.005) ** 2) ** 2, 0.0)
            a = (n @ R.T) @ Rc.T                                              # R_c (R n)
            Jr = np.cross(a, Y[ok]); Jt = -a
            J = np.c_[Jr, Jt]
            H += (J * w[:, None]).T @ J; g += (J * w[:, None]).T @ r
        dx = -np.linalg.solve(H + 1e-6 * np.eye(6) * np.trace(H) / 6, g)
        x = x + dx
        if np.abs(dx[:3]).max() < 1e-6 and np.abs(dx[3:]).max() < 1e-6:
            break
    Rc = Rot.from_rotvec(x[:3]).as_matrix()
    # (3) per-frame residual corrections
    rows = []
    for ts, R, pp, Xw in frames:
        Xt = (Xw - CEN - x[3:]) @ Rc + CEN
        Qc = (Xt - pp) @ R
        D, st = gcm.icp(cad, Qc, lock_spin=True)
        if D is None:
            continue
        rv = Rot.from_matrix(D[:3, :3]).as_rotvec()
        upour = np.cross(spout, [0, 0, 1.0]); upour /= np.linalg.norm(upour)       # body-frame pouring-tilt axis
        tip_d = D[:3, :3] @ tip_p + D[:3, 3] - tip_p
        rows.append(dict(t=ts, tilt_deg=float(np.degrees(rv @ upour)), rot_deg=float(np.degrees(np.linalg.norm(rv))),
                         tip_mm=float(np.linalg.norm(tip_d) * 1e3), rms_mm=st['rms_mm']))
    q = lambda v: [round(float(np.percentile(v, 50)), 3), round(float(np.percentile(v, 90)), 3)] if len(v) else None
    allrows = rows
    rows = [r for r in rows if np.isfinite(r['rms_mm']) and r['rot_deg'] < 15.0]   # ICP failures counted separately
    out[name] = dict(n_frames=len(allrows), n_icp_fail=len(allrows) - len(rows), C2_rot_deg=np.degrees(x[:3]).round(3).tolist(), C2_t_mm=(x[3:] * 1e3).round(2).tolist(),
                     abs_tilt_corr_deg_p50_p90=q(np.abs([r['tilt_deg'] for r in rows])), rot_corr_deg_p50_p90=q([r['rot_deg'] for r in rows]),
                     tip_corr_mm_p50_p90=q([r['tip_mm'] for r in rows]), icp_rms_mm_p50=q([r['rms_mm'] for r in rows])[0], rows=rows)
    print(p, name, {k: v for k, v in out[name].items() if k != 'rows'}, flush=True)
json.dump(out, open(f'{V2}/out/cam2icp_{p}.json', 'w'), indent=1)
