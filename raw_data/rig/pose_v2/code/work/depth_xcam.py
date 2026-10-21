"""Depth-only cross-camera consistency (no tags in the comparison): for every camera-0 depth frame paired with the
nearest camera-1 depth frame, a single-camera spin-locked CAD ICP per camera (pose_audit gcm.icp, Tukey) relative to
the tags-only trajectory at each frame's own time -> body-frame corrections D0, D1 -> world-frame disagreement at the
camera-1 time  Delta = T(t1) D1 (T(t1) D0)^-1  (the trajectory carries the motion between the two frames).
Jug axis direction and CAD centre in the world from each camera; a constant world-frame rigid transform C
(camera-0 world -> camera-1 world) is fitted to all pairs of a pour (axis directions + centres, robust).
    python3 depth_xcam.py <depth tag> [pours] -> ../out/depth_xcam.json"""
import sys, json, pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import V2, POURS
from spline import Traj

CEN = np.array([0.15, 0.22, -0.03])
tag = sys.argv[1]
pours = sys.argv[2:] or list(POURS)
T = pickle.load(open(f'{V2}/out/ct_tags.pkl', 'rb'))
cad = gcm.CAD(z0_mm=-28.5, az_deg=77.14)
res = {}
allpairs = []
for p in pours:
    tr = T['pours'][p]
    traj = Traj(tr['T0'], tr['dt'], tr['R'], tr['P'])
    Z = np.load(f'{V2}/data/depth_{p}_{tag}.npz')
    ft = Z['ft'] + np.where(Z['fcam'] == 1, tr['nuis']['c1'], 0.0)
    f0 = np.nonzero(Z['fcam'] == 0)[0]; f1 = np.nonzero(Z['fcam'] == 1)[0]
    pairs = []
    for a in f0:
        b = f1[np.argmin(np.abs(ft[f1] - ft[a]))]
        if abs(ft[b] - ft[a]) > 0.025:
            continue
        Ds = []
        for f in (a, b):
            X = Z['X'][Z['pf'] == f].astype(float)
            R, pp = traj.eval(np.array([ft[f]]))
            Q = (X - pp[0]) @ R[0]
            D, st = gcm.icp(cad, Q, lock_spin=True)
            if D is None or st['n_inl'] < 300:
                Ds = None; break
            Ds.append(D)
        if Ds is None:
            continue
        R1, p1 = traj.eval(np.array([ft[b]]))
        Tt = np.eye(4); Tt[:3, :3] = R1[0]; Tt[:3, 3] = p1[0]
        W0 = Tt @ Ds[0]; W1 = Tt @ Ds[1]                         # CAD-placed body -> world, from camera 0 / camera 1
        cen_p = cad.centre_p
        pairs.append(dict(t=float(ft[a]), dt=float(ft[b] - ft[a]), a0=W0[:3, 2], a1=W1[:3, 2],
                          c0=W0[:3, :3] @ cen_p + W0[:3, 3], c1=W1[:3, :3] @ cen_p + W1[:3, 3]))
    A0 = np.array([q['a0'] for q in pairs]); A1 = np.array([q['a1'] for q in pairs])
    # decomposition of the axis disagreement along camera 0's viewing direction (perp. to the axis) and sideways
    cam0 = np.linalg.inv(np.linalg.inv(T['_c2w0']))[:3, 3] if '_c2w0' in T else None
    import sys as _s; _s.path.insert(0, '/home/user/zfChen/latte_imex/rig')
    from trimrec import Take
    tk = Take(f'/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours/ghdata/{p}')
    o0 = tk.c2w[0][:3, 3]; o1 = tk.c2w[1][:3, 3]
    comp = []
    for q in pairs:
        a = q['a0']; v = q['c0'] - o0; v = v - (v @ a) * a; ev = v / np.linalg.norm(v); es = np.cross(a, ev)
        d = q['a1'] - q['a0']
        v1 = q['c1'] - o1; v1 = v1 - (v1 @ a) * a; ev1 = v1 / np.linalg.norm(v1)
        comp.append((np.degrees(d @ ev), np.degrees(d @ es), float(ev @ ev1)))
    comp = np.array(comp)
    print(p, 'axis disagreement along cam-0 view dir: median %.2f sd %.2f deg; sideways: median %.2f sd %.2f deg; cos(view0, view1) median %.2f' % (
        np.median(comp[:, 0]), comp[:, 0].std(), np.median(comp[:, 1]), comp[:, 1].std(), np.median(comp[:, 2])), flush=True)
    C0 = np.array([q['c0'] for q in pairs]); C1 = np.array([q['c1'] for q in pairs])
    allpairs.append((A0, A1, C0, C1))

    def fun(x, A0=A0, A1=A1, C0=C0, C1=C1):
        Rr = Rot.from_rotvec(x[:3]).as_matrix()
        ra = (A0 @ Rr.T - A1) / np.radians(1.0)                   # 1 deg ~ 1 mm
        rc = ((C0 - CEN) @ Rr.T + CEN + x[3:] - C1) / 0.001
        return np.concatenate([ra.ravel(), rc.ravel()])
    sol = least_squares(fun, np.zeros(6), loss='soft_l1', f_scale=3.0)
    ang = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', A0, A1), -1, 1)))
    dc = (C1 - C0) * 1e3
    Rr_ = Rot.from_rotvec(sol.x[:3]).as_matrix(); resid_ax = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', A0 @ Rr_.T, A1), -1, 1)))
    res[p] = dict(n_pairs=len(pairs), constC_axis_resid_deg_p50=float(np.median(resid_ax)), along_view_med_deg=float(np.median(comp[:, 0])),
                  sideways_med_deg=float(np.median(comp[:, 1])), along_view_sd=float(comp[:, 0].std()), sideways_sd=float(comp[:, 1].std()), axis_angle_deg_p50_p90=np.percentile(ang, [50, 90]).round(2).tolist(),
                  centre_diff_median_mm=np.median(dc, 0).round(2).tolist(), centre_diff_abs_p50=float(np.median(np.linalg.norm(dc, axis=1))),
                  C_rot_deg=np.degrees(sol.x[:3]).round(3).tolist(), C_t_mm=(sol.x[3:] * 1e3).round(2).tolist())
    print(p, json.dumps(res[p]), flush=True)
# pooled fit
A0, A1, C0, C1 = [np.concatenate([x[k] for x in allpairs]) for k in range(4)]


def fun(x):
    Rr = Rot.from_rotvec(x[:3]).as_matrix()
    return np.concatenate([((A0 @ Rr.T - A1) / np.radians(1.0)).ravel(), (((C0 - CEN) @ Rr.T + CEN + x[3:] - C1) / 0.001).ravel()])


sol = least_squares(fun, np.zeros(6), loss='soft_l1', f_scale=3.0)
J = sol.jac; cov = np.linalg.pinv(J.T @ J) * np.median(np.abs(sol.fun)) ** 2 * 2.2
res['pooled'] = dict(n=len(A0), C_rot_deg=np.degrees(sol.x[:3]).round(3).tolist(), C_t_mm=(sol.x[3:] * 1e3).round(2).tolist(),
                     sd_rot_deg=np.degrees(np.sqrt(np.diag(cov)[:3])).round(3).tolist(), sd_t_mm=(np.sqrt(np.diag(cov)[3:]) * 1e3).round(2).tolist())
print('pooled', res['pooled'])
json.dump(res, open(f'{V2}/out/depth_xcam_{tag}.json', 'w'), indent=1)
