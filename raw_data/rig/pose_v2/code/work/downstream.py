"""Downstream effect of v2 vs the current track on the production inlet chain (as pose_audit/holdout/sensitivity.py):
real_program.cup_poses (Slerp + linear on 1 ms, gaussian 40 ms) -> ForwardPitcher (kona700 geometry, ForwardLaw
defaults, 2 ms step), V0 = fill at the cut from full_pour_kona_poured165.json.
  current: raw_data/rig/results/<p>/pitcher_pose.npz (= tip.npz) with TIP_OVERRIDE = full_pour.TIP_KONA (production)
  v2:      the v2 samples (out/<p>_pose_v2.npz, 10 ms grid) with their own tip_p (CAD spout tip in the v2 frame)
Reports flow rms difference [mL/s] over the time either streams, poured volume, on/off mismatch, landing point
difference p50/p90 [mm]; plots Q(t), tilt(t) and the landing track.
    python3 downstream.py [pours...] -> ../out/downstream.json, ../img/downstream_<p>.jpg"""
import os, sys, json, math
os.environ["PITCHER_GEOM"] = "/home/user/zfChen/latte_imex/pitcher_geom/pitcher_kona700.json"
sys.path.insert(0, "/home/user/zfChen")
import numpy as np
import latte_imex.rig.real_program as RP
from latte_imex.rig.full_pour import TIP_KONA
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

V2 = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2'
FILL = json.load(open("/home/user/zfChen/raw_data/rig/results/full_pour_kona_poured165.json"))
_orig_load = RP.load_take
SUB = 2


def run(p, P=None, tip=None):
    def lt(q):
        T = _orig_load(q)
        if P is not None:
            T = dict(T); T.update({k: np.asarray(v, float) for k, v in P.items()})
        return T
    RP.load_take = lt
    RP.TIP_OVERRIDE = TIP_KONA if tip is None else np.asarray(tip, float)
    try:
        C = RP.cup_poses(p)
    finally:
        RP.load_take = _orig_load
    fp = RP.forward_pitcher()
    V0 = FILL[p]["fill_at_cut_mL"] * 1e-6
    s = slice(None, None, SUB)
    E = fp.run(C["t"][s], C["tip"][s], C["yaw"][s], C["tilt"][s], C["roll"][s], V0, C["dt"] * SUB)
    E["t_abs"] = E["t"] + C["t0_abs"]
    E["axis_tilt"] = C["axis_tilt_deg"][s]
    return C, E


def compare(E0, E1, dt):
    ta = E0["t_abs"]; ok = (ta >= E1["t_abs"][0]) & (ta <= E1["t_abs"][-1])
    Q0 = E0["Q_o"][ok] * 1e6
    Q1 = np.interp(ta[ok], E1["t_abs"], E1["Q_o"]) * 1e6
    X0 = E0["x_hit"][ok]
    X1 = np.stack([np.interp(ta[ok], E1["t_abs"], E1["x_hit"][:, j]) for j in range(2)], 1)
    on = (Q0 > 1) | (Q1 > 1); both = (Q0 > 1) & (Q1 > 1) & np.isfinite(X0).all(1) & np.isfinite(X1).all(1)
    dx = np.linalg.norm(X1 - X0, axis=1) * 1e3
    return dict(span_s=float(ok.sum() * dt), dQ_rms=float(np.sqrt(np.mean((Q1 - Q0)[on] ** 2))), dQ_mean=float(np.mean((Q1 - Q0)[on])),
                Q_mean_cur=float(np.mean(Q0[on])), V_cur_mL=float(Q0.sum() * dt), V_v2_mL=float(Q1.sum() * dt),
                dV_mL=float((Q1.sum() - Q0.sum()) * dt), on_mismatch_frac=float(np.mean((Q0 > 1) != (Q1 > 1))),
                dx_hit_p50=float(np.percentile(dx[both], 50)) if both.any() else None,
                dx_hit_p90=float(np.percentile(dx[both], 90)) if both.any() else None,
                dx_hit_mean_vec_mm=((X1 - X0)[both].mean(0) * 1e3).round(2).tolist() if both.any() else None)


if __name__ == '__main__':
    pours = sys.argv[1:] or ["tulip", "leaf", "heart", "push_heart", "swan"]
    res = {}
    for p in pours:
        Z = np.load(f'{V2}/out/{p}_pose_v2.npz')
        C0, E0 = run(p)
        C1, E1 = run(p, dict(t=Z['t'], rotvec=Z['poses'][:, :3], pos=Z['poses'][:, 3:]), tip=Z['tip_p'])
        dt = C0['dt'] * SUB
        R = compare(E0, E1, dt)
        # tilt difference on the production chain (axis tilt after the 40 ms smoothing)
        ta = E0['t_abs']; ok = (ta >= E1['t_abs'][0]) & (ta <= E1['t_abs'][-1])
        dtilt = np.interp(ta[ok], E1['t_abs'], E1['axis_tilt']) - E0['axis_tilt'][ok]
        R['dtilt_deg_median'] = float(np.median(dtilt)); R['dtilt_deg_p90abs'] = float(np.percentile(np.abs(dtilt), 90))
        res[p] = R
        print(p, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in R.items()}), flush=True)
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        ax[0].plot(E0['t_abs'], E0['Q_o'] * 1e6, lw=1, label='current (tip.npz, TIP_KONA)')
        ax[0].plot(E1['t_abs'], E1['Q_o'] * 1e6, lw=1, label='v2')
        ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('inlet flow Q [mL/s]'); ax[0].legend(); ax[0].set_title(f'{p}: forward-pitcher flow')
        ax[1].plot(E0['t_abs'], E0['axis_tilt'], lw=1, label='current'); ax[1].plot(E1['t_abs'], E1['axis_tilt'], lw=1, label='v2')
        ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('axis tilt (40 ms smoothed) [deg]'); ax[1].legend()
        for E, lab in ((E0, 'current'), (E1, 'v2')):
            m = E['Q_o'] > 1e-6
            ax[2].plot(E['x_hit'][m, 0] * 1e3, E['x_hit'][m, 1] * 1e3, '.', ms=1, label=lab)
        ax[2].set_aspect('equal'); ax[2].set_xlabel('x [mm] (cup frame)'); ax[2].set_ylabel('y [mm]'); ax[2].legend(); ax[2].set_title('landing point while streaming')
        fig.tight_layout(); fig.savefig(f'{V2}/img/downstream_{p}.jpg', dpi=70); plt.close(fig)
    json.dump(res, open(f'{V2}/out/downstream.json', 'w'), indent=1)
