"""Whole pour (mixing + pattern phase) through the forward pitcher: measured poses from the full-recording tag fit
(<scratch>/pours/<p>/full/ba.pkl), cleaned like tip_track (running-median outlier rejection), 1 ms grid; pitcher
geometry from PITCHER_GEOM (default: the real-pitcher parametric JSON); initial fill V0 = milk mass / foam density.
Reports when the model pitcher pours, how much it pours before the pattern phase, and the fill at the cut.
    PITCHER_GEOM=... python -m latte_imex.rig.full_pour <pattern> [--mass 168] [--rho 0.8]"""
import os, sys, json, math, pickle, argparse
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation, Slerp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("PITCHER_GEOM", os.path.join(os.path.dirname(__file__), "..", "pitcher_geom", "pitcher_real.json"))
import latte_imex.rig.real_program as RP

SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
CUT = dict(heart=(10.4, 18.4), push_heart=(12.6, 24.0), leaf=(11.0, 20.0), swan=(10.4, 22.0), tulip=(98.8, 112.2))
TIP_P = np.array([0.0, 0.053, 0.093])
# Per-pour pitcher-axis correction [deg, about the pitcher x axis; negative = more towards the spout], chosen so that the
# whole pour (mixing + pattern) empties 165 g of milk (206 mL at 0.8 g/mL) with ~2-3 mL left when the stream breaks.
DTILT = dict(heart=-6.5, push_heart=-8.0, leaf=-11.5, swan=-8.5, tulip=-7.5)


def poses(p):
    ba = pickle.load(open(f"{SCR}/{p}/full/ba.pkl", "rb"))
    o = np.argsort(ba["t"]); t = ba["t"][o]; P = ba["poses"][o]
    k = np.r_[True, np.diff(t) > 1e-4]; t, P = t[k], P[k]
    R = Rotation.from_rotvec(P[:, :3]).as_matrix()
    up = np.array([0, 0, -1.0])
    tilt = np.degrees(np.arccos(np.clip(R[:, :, 2] @ up, -1, 1)))
    g = np.einsum("nji,j->ni", R, -up); az = np.degrees(np.arctan2(g[:, 1], g[:, 0]))
    good = (np.abs(tilt - median_filter(tilt, 7, mode="nearest")) < 4) & (np.abs(az - median_filter(az, 7, mode="nearest")) < 8)
    good |= tilt < 25        # azimuth is meaningless near upright
    return t[good], P[good]


def run(p, mass=165.0, rho=0.8, dt=1e-3, dtilt=None):
    """dtilt [deg]: extra rotation of the pitcher about its own x axis (towards the spout), i.e. a correction of the
    tag-fitted pitcher axis."""
    dtilt = DTILT.get(p, 0.0) if dtilt is None else dtilt
    t, P = poses(p)
    if dtilt:
        P = P.copy(); P[:, :3] = (Rotation.from_rotvec(P[:, :3]) * Rotation.from_euler("x", dtilt, degrees=True)).as_rotvec()
    tip_p = Rotation.from_euler("x", dtilt, degrees=True).inv().apply(TIP_P)
    n = int((t[-1] - t[0]) / dt); tg = t[0] + (np.arange(n) + 0.5) * dt
    q = Slerp(t, Rotation.from_rotvec(P[:, :3]))(tg).as_quat()
    for i in range(1, n):
        if np.dot(q[i], q[i - 1]) < 0: q[i] = -q[i]
    q = RP._smooth(q, RP.POSE_SIGMA, dt); q /= np.linalg.norm(q, axis=1)[:, None]
    Rb = Rotation.from_quat(q).as_matrix()
    pos = RP._smooth(np.stack([np.interp(tg, t, P[:, 3 + j]) for j in range(3)], 1), RP.POSE_SIGMA, dt)
    tip = (np.einsum("nij,j->ni", Rb, tip_p) + pos) @ RP.S_FLIP.T
    Rc = RP.S_FLIP @ Rb @ RP.M_CAD
    eul = Rotation.from_matrix(Rc).as_euler("ZYX")
    axis_tilt = np.degrees(np.arccos(np.clip(-Rb[:, 2, 2], -1, 1)))
    V0 = mass / rho * 1e-6
    E = RP.forward_pitcher().run(tg - tg[0], tip, eul[:, 0], eul[:, 1], eul[:, 2], V0, dt)
    tt = tg - 0.0                                      # seconds since the recording's first frame (same clock as CUT)
    Q = E["Q_o"] * 1e6; V = E["V_P"] * 1e6
    c0, c1 = CUT[p]
    on = Q > 1.0
    first = float(tt[on][0]) if on.any() else None
    before = float(Q[tt < c0].sum() * dt); during = float(Q[(tt >= c0) & (tt <= c1)].sum() * dt)
    V_cut = float(np.interp(c0, tt, V))
    res = dict(pattern=p, V0_mL=V0 * 1e6, first_flow_s=first, poured_before_cut_mL=before, poured_pattern_mL=during,
               fill_at_cut_mL=V_cut, fill_at_pattern_end_mL=float(np.interp(c1, tt, V)), left_end_mL=float(V[-1]), spill_mL=float(E["Q_spill"].sum() * dt * 1e6),
               tilt_at_first_flow=float(np.interp(first, tt, axis_tilt)) if first else None, dtilt_deg=dtilt, geometry=os.environ["PITCHER_GEOM"])
    return res, dict(t=tt, Q=Q, V=V, tilt=axis_tilt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("patterns", nargs="+"); ap.add_argument("--mass", type=float, default=165.0)
    ap.add_argument("--rho", type=float, default=0.8); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(a.patterns), 1, figsize=(13, 2.8 * len(a.patterns)), squeeze=False)
    allr = {}
    for i, p in enumerate(a.patterns):
        r, S = run(p, a.mass, a.rho); allr[p] = r; print(json.dumps(r), flush=True)
        x = ax[i, 0]; x.plot(S["t"], S["tilt"], "r-", lw=0.8, label="axis tilt [deg]")
        x.plot(S["t"], S["Q"], "b-", lw=0.8, label="model flow [mL/s]"); x.plot(S["t"], S["V"] / 3, "g-", lw=0.8, label="fill/3 [mL]")
        x.axvspan(*CUT[p], color="orange", alpha=0.15, label="pattern phase"); x.set_ylim(0, 110); x.set_ylabel(p); x.grid(alpha=0.3)
        if i == 0: x.legend(loc="upper left", fontsize=8, ncol=4)
    ax[-1, 0].set_xlabel("t [s] since the first frame of the recording")
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "..", "raw_data", "rig", "results", f"full_pour_rho{a.rho:g}.png")
    fig.tight_layout(); fig.savefig(out, dpi=80); json.dump(allr, open(os.path.splitext(out)[0] + ".json", "w"), indent=1)
