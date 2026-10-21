"""Write pose track v2 in the downstream formats.
Output (pitcher) frame = CAD-aligned: z = CAD axis towards the rim, +y = spout (CAD +x), origin on the CAD axis at the
height of tag 4's centre (CAD z = -z0).  This is the convention every consumer assumes ('pitcher z = axis to the rim,
spout +y'); the v2 TAG frame differs from it by the fitted CAD placement (spout at 77.14 deg, axis tilt ~3.7 deg).
Files per pour p (10 ms grid between the first and last tag / depth observation):
  out/<p>_pose_v2.npz         tip.npz keys (t, poses (n,6) rotvec+pos, tip_board, tilt, az, tip_p) = what fuse_occ.PourData,
                              cup_track.py, action_summary.py, barista_view.py read, + the pitcher_pose.npz keys
                              (rotvec, pos, tilt_deg, az_deg, note) = what real_program.load_take / cup_poses read,
                              + extras: sd_* (formal 1-sigma per sample), spline (tag frame), offsets, layout, placement.
  out/<p>_pitcher_pose_v2.npz exactly the raw_data/rig/results/<p>/pitcher_pose.npz key set.
  out/<p>_tip_v2.json         tip.json fields (tip_in_pitcher_mm, n_samples, tilt_good_deg) + frame notes.
    python3 write_out.py <fit tag> [pours]"""
import sys, json, pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import *
from spline import Traj
from ctfit import PourProblem, Fit
from fit_joint import load_depth_raw

NOTE = ("v2 (pose_v2): board frame (up = -z); pitcher frame = CAD-aligned: z = jug (CAD) axis to the rim, spout +y, origin "
        "on the axis at the tag-4 centre height; x_board = R(rotvec) x_p + pos; continuous-time spline (tags+depth+prior) "
        "sampled every 10 ms")


def out_transform(PL):
    """B: output frame -> tag frame (4x4) and the CAD tip in the output frame."""
    cad = gcm.CAD(z0_mm=PL['z0'], az_deg=PL['az'])
    cad.set_place(PL['z0'], PL['az'], np.array(PL['extra']))
    S = np.eye(4)
    S[:3, :3] = Rot.from_euler('z', -90, degrees=True).as_matrix() * 1000.0       # output [m] -> CAD [mm]
    S[:3, 3] = [0, 0, -PL['z0']]
    B = cad.A @ S
    B[:3, :3] /= np.cbrt(np.linalg.det(B[:3, :3]))                              # exact rotation (A has the mm->m scale)
    tip_out = Rot.from_euler('z', 90, degrees=True).as_matrix() @ (cad.tip_cad - np.array([0, 0, -PL['z0']])) * 1e-3
    return B, tip_out, cad


def formal_sd(F, P, tg, upour):
    """posterior 1-sigma (data noise only) of pouring tilt, spin, tip position at the grid times."""
    Hpp, gp, Hpl, Hll, gl, _ = F.linearize(P, F.L)
    C = np.linalg.inv(Hpp + 1e-9 * np.eye(len(gp)))
    i, u, R, p, G, gu = P.traj.eval_jac(tg)
    out = np.zeros((len(tg), 3))
    for n_ in range(len(tg)):
        s0 = 6 * i[n_]
        Cp = G[n_] @ C[s0:s0 + 24, s0:s0 + 24] @ G[n_].T                        # eps (body rot, world trans)
        out[n_, 0] = np.sqrt(upour @ Cp[:3, :3] @ upour)
        ez = np.array([0, 0, 1.0])
        out[n_, 1] = np.sqrt(ez @ Cp[:3, :3] @ ez)
    return out


if __name__ == '__main__':
    tag = sys.argv[1]
    pours = sys.argv[2:] or list(POURS)
    PL = json.load(open(f'{V2}/out/{sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".json") else "placement_joint.json"}'))
    pours = [p for p in pours if not p.endswith('.json')] or list(POURS)
    B, tip_out, cad = out_transform(PL)
    tip_tag = B[:3, :3] @ tip_out + B[:3, 3]
    print('B rot (deg, rotvec)', np.degrees(Rot.from_matrix(B[:3, :3]).as_rotvec()).round(2), 'B t mm', (B[:3, 3] * 1e3).round(2),
          'tip out mm', (tip_out * 1e3).round(1), 'tip tag mm', (tip_tag * 1e3).round(1))
    for p in pours:
        F0 = pickle.load(open(f'{V2}/out/full_{p}_{tag}.pkl', 'rb'))
        traj = Traj(F0['T0'], F0['dt'], F0['R'], F0['P'])
        L = Layout.from_dict(F0['layout']); L.pl = np.zeros(6)     # placement correction already composed into PL
        D = PourData(p, 'edge', tags=set(L.tp))
        tobs = np.r_[[d['t'] + (F0['nuis']['c1'] if d['cam'] == 1 else 0.0) for d in D.rows]]
        dp = load_depth_raw(p, 't1', Layout.from_dict(pickle.load(open(f'{V2}/out/ct_tags_lean_e1.pkl', 'rb'))['layout']), (0, 1))
        tdep = dp['ft'] + np.where(dp['fcam'] == 0, 0.0, F0['nuis']['c1'])
        t_lo, t_hi = max(min(tobs.min(), tdep.min()), traj.T0), min(max(tobs.max(), tdep.max()), traj.t_end - 1e-6)
        tg = np.arange(np.ceil(t_lo * 100) / 100, t_hi, 0.01)
        Rt, pt = traj.eval(tg)
        Ro = Rt @ B[:3, :3]
        po = np.einsum('nij,j->ni', Rt, B[:3, 3]) + pt
        rv = Rot.from_matrix(Ro).as_rotvec()
        tipb = np.einsum('nij,j->ni', Ro, tip_out) + po
        tilt, az = tilt_az(Ro)
        # formal uncertainty
        P = PourProblem(D, traj, D.rows, dp, nuis_free=(), cad=cad)
        P.nuis.update(F0['nuis']); P.associate(L); P.neff = F0['neff']
        F = Fit([P], L, sig_t=F0['sig_t'], k_t=1.5, sig_d=F0['sig_d'], c_d=0.005, neff=None, sig_alpha=F0['sig_alpha'],
                sig_a=F0['sig_a'], use_depth=True)
        a = np.radians(PL['az'] - 90.0)
        upour = np.array([np.cos(a), np.sin(a), 0.0])
        sd = formal_sd(F, P, tg, upour)
        # tip position sd: propagate translation + rotation (approx: rotation sd x lever arm)
        np.savez_compressed(f'{V2}/out/{p}_pose_v2.npz', t=tg, poses=np.c_[rv, po], tip_board=tipb, tilt=tilt, az=az, tip_p=tip_out,
                            rotvec=rv, pos=po, tilt_deg=tilt, az_deg=az, note=np.array(NOTE),
                            sd_tilt_deg=np.degrees(sd[:, 0]), sd_spin_deg=np.degrees(sd[:, 1]),
                            tip_p_tagframe=tip_tag, B_out_to_tag=B, spline_T0=traj.T0, spline_dt=traj.dt, spline_R_tag=traj.R,
                            spline_P_tag=traj.P,
                            # (c1, d0, d1): depth is tied to its camera's colour clock (PourProblem.tie_depth), so the
                            # effective depth offsets are d0 = 0 and d1 = c1 (the stored nuis d0/d1 are unused zeros)
                            offsets_s=np.array([F0['nuis']['c1'], 0.0, F0['nuis']['c1']]),
                            layout_json=np.array(json.dumps(F0['layout'])), placement_json=np.array(json.dumps(PL)))
        np.savez_compressed(f'{V2}/out/{p}_pitcher_pose_v2.npz', t=tg, rotvec=rv, pos=po, tip_p=tip_out, tip_board=tipb,
                            tilt_deg=tilt, az_deg=az, note=np.array(NOTE))
        json.dump(dict(take=p, tip_in_pitcher_mm=(tip_out * 1e3).round(1).tolist(), n_samples=int(len(tg)), n_good=int(len(tg)),
                       tilt_good_deg=[round(float(tilt.min()), 1), round(float(np.median(tilt)), 1), round(float(tilt.max()), 1)],
                       frame='CAD-aligned pitcher frame (spout +y, z = CAD axis, origin on the axis at tag-4 height)', source='pose_v2'),
                  open(f'{V2}/out/{p}_tip_v2.json', 'w'), indent=1)
        print(p, 'samples', len(tg), 't', round(tg[0], 3), round(tg[-1], 3), 'tilt p5/50/95', np.percentile(tilt, [5, 50, 95]).round(1),
              'formal sd tilt deg p50/p90', np.percentile(np.degrees(sd[:, 0]), [50, 90]).round(3),
              'spin', np.percentile(np.degrees(sd[:, 1]), [50, 90]).round(3), flush=True)
