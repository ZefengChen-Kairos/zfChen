"""Dynamics plausibility, flips, cross-camera bias and leave-one-camera-out tag prediction, v2 vs current.
  dynamics: body angular acceleration |alpha| [deg/s^2] and spout-tip acceleration |a_tip| [m/s^2] on a 5 ms grid by
            central differences of the pose series: v2 spline; current production series (real_program.cup_poses
            style: Slerp + linear on 1 ms, gaussian 40 ms) and current raw (Slerp of tip.npz, no smoothing).
  flips:    current tip.npz / ba.pkl samples whose rotation differs from v2 at the same time by > 10 deg; v2: tag
            detections for which the best alternative single-frame pose branch (multi-start, fixed layout) fits the
            corners better than v2 by > 2x and lies > 10 deg from v2.
  LOCO:     corners of the held-out camera predicted by the fit without that camera (tags AND depth removed),
            reprojection error per corner [px]; cross-camera bias: cam-0-only vs cam-1-only fit, tip / origin
            difference (median vector = systematic part, its norm; median of |diff|).
    python3 eval_misc.py <v2 tag> -> ../out/eval_misc.json"""
import sys, json, pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from pv2lib import *
from spline import Traj
from ctfit import tag_fun
import jax.numpy as jnp


def series_dyn(t, Rm, p, tip_p, dt=0.005):
    """central differences on a uniform grid: |alpha| deg/s^2, |a_tip| m/s^2, |omega| deg/s"""
    R = Rot.from_matrix(Rm)
    w = (R[:-1].inv() * R[1:]).as_rotvec() / dt                       # body angular velocity at mid points
    al = np.linalg.norm(np.diff(w, axis=0), axis=1) / dt
    tip = R.apply(tip_p) + p
    a = np.linalg.norm(tip[2:] - 2 * tip[1:-1] + tip[:-2], axis=1) / dt ** 2
    return np.degrees(al), a, np.degrees(np.linalg.norm(w, axis=1))


def cur_series(p, tg, smooth=True):
    z = np.load(f'{SCR}/pours/{p}/tip.npz')
    t, P = z['t'], z['poses']
    o = np.argsort(t); t, P = t[o], P[o]
    k = np.r_[True, np.diff(t) > 1e-4]; t, P = t[k], P[k]
    tg = tg[(tg >= t[0]) & (tg <= t[-1])]
    q = Slerp(t, Rot.from_rotvec(P[:, :3]))(tg).as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0: q[i] = -q[i]
    pos = np.stack([np.interp(tg, t, P[:, 3 + j]) for j in range(3)], 1)
    if smooth:
        dt = tg[1] - tg[0]
        q = gaussian_filter1d(q, 0.04 / dt, axis=0, mode='nearest'); q /= np.linalg.norm(q, axis=1)[:, None]
        pos = gaussian_filter1d(pos, 0.04 / dt, axis=0, mode='nearest')
    return tg, Rot.from_quat(q).as_matrix(), pos, z['tip_p']


def pct(x, qs=(50, 95, 100)):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return [float(np.percentile(x, q)) for q in qs] if len(x) else None


def traj_of(d):
    return Traj(d['T0'], d['dt'], d['R'], d['P'])


if __name__ == '__main__':
    tag = sys.argv[1]
    res = {}
    for p in POURS:
        F = pickle.load(open(f'{V2}/out/full_{p}_{tag}.pkl', 'rb'))
        tr = traj_of(F)
        L = Layout.from_dict(F['layout'])
        Z = np.load(f'{V2}/out/{p}_pose_v2.npz')
        tip_tag = Z['tip_p_tagframe']
        D = PourData(p, 'edge', tags=set(L.tp))
        tobs = np.array([d['t'] for d in D.rows])
        tg = np.arange(tobs.min(), tobs.max(), 0.005)
        R2, p2 = tr.eval(tg)
        al2, a2, w2 = series_dyn(tg, R2, p2, tip_tag)
        tgc, Rc, pc, tipc = cur_series(p, tg, smooth=True)
        alc, ac, wc = series_dyn(tgc, Rc, pc, tipc)
        tgr, Rr, pr, _ = cur_series(p, tg, smooth=False)
        alr, ar, wr = series_dyn(tgr, Rr, pr, tipc)
        R = dict(dyn=dict(v2=dict(alpha_deg_s2=pct(al2), a_tip_ms2=pct(a2), omega_deg_s=pct(w2)),
                          cur_production_40ms=dict(alpha_deg_s2=pct(alc), a_tip_ms2=pct(ac), omega_deg_s=pct(wc)),
                          cur_raw=dict(alpha_deg_s2=pct(alr), a_tip_ms2=pct(ar), omega_deg_s=pct(wr))))
        # flips of the current samples relative to v2
        z = np.load(f'{SCR}/pours/{p}/tip.npz')
        ba = pickle.load(open(f'{SCR}/pours/{p}/ba.pkl', 'rb'))
        out_fl = {}
        for name, (t_, P_) in dict(tip_npz=(z['t'], z['poses']), ba_pkl=(np.asarray(ba['t']), np.asarray(ba['poses']))).items():
            m = (t_ >= tr.T0) & (t_ <= tr.t_end)
            Rv, pv = tr.eval(t_[m])
            # the current frame differs from v2's tag frame by a constant transform: estimate it robustly first
            Rcur = Rot.from_rotvec(P_[m, :3])
            dR = (Rot.from_matrix(Rv).inv() * Rcur)
            # constant X: median rotation vector of dR (frames differ by a fixed body-frame rotation)
            X = Rot.from_rotvec(np.median(dR.as_rotvec(), 0))
            err = np.degrees((dR * X.inv()).magnitude())
            out_fl[name] = dict(n=int(m.sum()), n_gt10deg=int((err > 10).sum()), n_gt5deg=int((err > 5).sum()),
                                frame_rot_deg=float(np.degrees(X.magnitude())), rot_dev_p50_p90=pct(err, (50, 90)))
        R['flips_current_vs_v2'] = out_fl
        # v2 flips: per detection, best alternative branch vs v2
        f_, fj_ = tag_fun(L.a, L.flat)
        nflip, nsingle = 0, 0
        offs = F['nuis']
        for d in D.rows:
            t = d['t'] + (offs['c1'] if d['cam'] == 1 else 0.0)
            if t < tr.T0 or t > tr.t_end:
                continue
            Rv, pv = tr.eval(np.array([t]))
            r = np.asarray(f_(jnp.asarray(Rv), jnp.asarray(pv), jnp.zeros(6), jnp.asarray(L.tp[d['tag']][None]), L.s,
                              D.W2C[d['cam']][None], D.Kv[d['cam']][None], d['q'][None], jnp.asarray(L.ext1),
                              np.array([float(d['cam'] == 1)]), jnp.asarray(L.ext_cen)))
            c_v2 = float((r ** 2).sum())
            sols = frame_candidates(D, L, [d], iters=30)
            x2 = np.r_[Rot.from_matrix(Rv[0]).as_rotvec(), pv[0]]
            alt = [s for s in sols if rot_deg(s[1], x2) > 10]
            nsingle += 1
            if alt and 2 * alt[0][0] * 2 < c_v2 * 0.5 and c_v2 > 8 * 0.3 ** 2 * 4:
                nflip += 1
        R['v2_flip_suspects'] = dict(n_detections=nsingle, n_alt_branch_fits_better=nflip)
        # LOCO + cross-camera bias
        try:
            F0 = pickle.load(open(f'{V2}/out/full_{p}_{tag}_cam0.pkl', 'rb'))     # fitted with camera 0 only
            F1 = pickle.load(open(f'{V2}/out/full_{p}_{tag}_cam1.pkl', 'rb'))     # fitted with camera 1 only
        except FileNotFoundError:
            F0 = F1 = None
        if F0 is not None:
            loco = {}
            for held, Fo in ((1, F0), (0, F1)):
                to = traj_of(Fo)
                rows = [d for d in D.rows if d['cam'] == held]
                t_other = np.array(sorted({d['t'] for d in D.rows if d['cam'] != held}))
                errs, joined = [], []
                for d in rows:
                    t = d['t'] + (F['nuis']['c1'] if held == 1 else 0.0)
                    if t < to.T0 or t > to.t_end:
                        continue
                    Rv, pv = to.eval(np.array([t]))
                    r = np.asarray(f_(jnp.asarray(Rv), jnp.asarray(pv), jnp.zeros(6), jnp.asarray(L.tp[d['tag']][None]), L.s,
                                      D.W2C[d['cam']][None], D.Kv[d['cam']][None], d['q'][None], jnp.asarray(L.ext1),
                                      np.array([float(d['cam'] == 1)]), jnp.asarray(L.ext_cen))).reshape(4, 2)
                    errs.append(np.median(np.linalg.norm(r, axis=1)))
                    joined.append(bool(len(t_other) and np.min(np.abs(t_other - d['t'])) < 0.012))
                errs, joined = np.array(errs), np.array(joined)
                loco[f'cam{held}_heldout_px'] = pct(errs, (50, 90, 100))
                loco[f'cam{held}_heldout_px_joined12ms'] = pct(errs[joined], (50, 90, 100)) if joined.any() else None
                loco[f'cam{held}_n'] = len(errs)
            R['loco'] = loco
            t0_, t1_ = traj_of(F0), traj_of(F1)
            tt = tg[(tg >= max(t0_.T0, t1_.T0)) & (tg <= min(t0_.t_end, t1_.t_end))]
            Ra, pa = t0_.eval(tt); Rb, pb = t1_.eval(tt)
            tipa = np.einsum('nij,j->ni', Ra, tip_tag) + pa; tipb = np.einsum('nij,j->ni', Rb, tip_tag) + pb
            dtip = (tipa - tipb) * 1e3; dorg = (pa - pb) * 1e3
            drot = np.degrees((Rot.from_matrix(Ra).inv() * Rot.from_matrix(Rb)).magnitude())
            tilta, _ = tilt_az(Ra); tiltb, _ = tilt_az(Rb)
            R['cross_camera'] = dict(tip_bias_mm=float(np.linalg.norm(np.median(dtip, 0))), tip_bias_vec_mm=np.median(dtip, 0).round(2).tolist(),
                                     tip_absdiff_p50_p90=pct(np.linalg.norm(dtip, axis=1), (50, 90)),
                                     origin_bias_mm=float(np.linalg.norm(np.median(dorg, 0))),
                                     rot_diff_deg_p50_p90=pct(drot, (50, 90)), tilt_bias_deg=float(np.median(tilta - tiltb)),
                                     tilt_absdiff_p50_p90=pct(np.abs(tilta - tiltb), (50, 90)))
        res[p] = R
        print(p, json.dumps(R), flush=True)
    json.dump(res, open(f'{V2}/out/eval_misc.json', 'w'), indent=1)
