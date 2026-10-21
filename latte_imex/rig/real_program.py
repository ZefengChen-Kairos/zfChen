"""Real barista pours (three-RealSense rig) -> forward-pitcher inlet -> latte_imex.run.run program (control 'v05').

Inputs per pattern p (heart, push_heart, leaf, swan, tulip):
  raw_data/rig/results/<p>/pitcher_pose.npz   measured pitcher poses (exported once from the session scratch by
                                              `export`): t [s], rotvec (n,3), pos (n,3) [m] of the pitcher frame in the
                                              calibration-board frame (board up = -z), tip_p = spout tip in the pitcher
                                              frame (pitcher z = axis towards the rim, spout at +y)
  raw_data/rig/results/<p>/disc.json          per-frame liquid disc: centre (board x, y), radius, surface z
  raw_data/rig/targets/<p>.npz                finished-pattern target (disc coordinates, row = +y)

Pipeline (no fitted physics; every constant below is an assumption, listed in `ASSUMPTIONS`):
 1. Poses on the pitcher model's 1 ms grid: rotations slerped (scipy Slerp) from the irregular ~13-20 fps samples and
    smoothed (gaussian, sigma POSE_SIGMA, on the sign-aligned quaternion components, renormalised); the pitcher origin
    linearly interpolated and smoothed the same way.  The disc centre and surface height (cup frame) are interpolated
    from disc.json, which since cup_register.py is already the rigid-cup track every consumer shares (centre robustly
    smoothed, R(t) from the cup's flare, plane from multi-view consistency); DISC_SIGMA smooths only the old free
    per-frame fits (disc_fit.py), which also need their fit_ok frames picked.
 2. Cup frame: origin = disc centre on the liquid surface, axes F = (x_b, -y_b, -z_b) (board x / y, up; a proper
    rotation of the board frame, so the pitcher pose is not mirrored).  The spout tip = R tip_p + pos is expressed in
    this frame; it is the pose's anchor: the CAD stand-in pitcher (NX, s1_forward.geometry(0)) is placed with ITS
    spout tip at the measured tip and the measured body orientation (CAD body X = spout = real pitcher +y, CAD Z = real
    pitcher axis), i.e. R_cad = S R_board<-pitcher M, (yaw, tilt, roll) = ZYX Euler angles of R_cad.  Tip height above
    the surface and the pitcher attitude relative to gravity are therefore the measured ones; lengths stay physical
    (metres) inside the pitcher model, so the ballistic jet (exit point, exit velocity incl. the lip's rigid-body
    velocity, fall to z = 0) is consistent with the measured tip and jet height.
 3. Flow: pitcher.ForwardPitcher (V1 mode B) on these poses with an initial fill V0.  V0 is NOT measured: `calibrate_fill`
    takes the smallest V0 for which the stream is on (Q >= Q_ON = 1 mL/s) during >= ON_FRAC of the time the data say it
    is on, or within ON_PLATEAU of the on fraction a 500 mL fill reaches when that is lower (at the very end of a
    pour the stand-in lip can dry whatever the fill: the pitcher keeps only the volume below its lip at that pose).  'Observed on' = smoothed axis tilt within ON_DTILT of its running max over the last ON_WIN s (a pour holds or
    raises the tilt as the pitcher empties; a stop / reposition shows as a tilt drop of 10-25 deg, and the end of the
    pour as the final tilt-back).  The stream is then switched on/off by the pitcher physics, not by the mask.
    Alternative flow law (flow='weir'): Q = c_w [theta - (theta_env - ON_DTILT)]_+^{3/2} (theta = axis tilt, theta_env its
    running max), one scale constant c_w (default: the same poured volume as the forward-pitcher run); jet origin,
    lip velocity and exit direction from the forward pitcher with a full pitcher, exit speed from the weir head
    H = (Q / (K w))^{2/3} on the forward pitcher's wetted width w.
 4. Arrivals (ForwardPitcher.arrivals, emission-time ballistics, 1 ms bins) -> solver units: the simulator cup radius
    (Numerics.cup_radius = 0.49 D) maps to the median measured disc radius R_disc, i.e. the length unit is
    L = R_disc / 0.49; positions and velocities are divided by L, flows by L^3 (geometric similarity; time unscaled).
    The record is rescaled by 0.08 / L and handed to s1_forward.FwdProgram (= v05_pourable.pl_program 'pitcher'
    inlet rules: drvf_record, foot-speed scan bound), with the S1 closure parameters (s1_forward program json) and
    synthetic 0.1 s phases along the tip path (class 'low' where observed on, 'stop' otherwise; only used for the
    scan-speed floor, the off-record and the 1 mL/s floor of drvf_record).  Duration = pour + TAIL s (no inlet).
 5. latte_imex.run.run(control='v05') with the Params defaults (symmetric viscosity, off-cup accounting).

Simulator (x, y) = cup frame F: x = board x, y = -board y.  render.py shows y up, so final.png / movie.mp4 have the same
orientation as the target images (row = +board y).  Disc coordinates of a simulator point: (x / 0.49, -y / 0.49).

usage (from /home/user/zfChen):
  python3 -m latte_imex.rig.real_program export --src <scratch>/pours             (once: poses into the repo)
  python3 -m latte_imex.rig.real_program inlet --pattern heart                     (pitcher only, prints the summary)
  NUMBA_NUM_THREADS=4 python3 -m latte_imex.rig.real_program run --pattern heart --N 256 [--flow weir] [--no-video]
  python3 -m latte_imex.rig.real_program compare                                   -> raw_data/rig/results/sim_vs_real.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from .. import s1_forward as SF
from ..pitcher import ForwardLaw, ForwardPitcher

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "raw_data", "rig", "results")
TGT = os.path.join(REPO, "raw_data", "rig", "targets")
PATTERNS = ("heart", "push_heart", "leaf", "swan", "tulip")
CUP_R = 0.49                 # Numerics.cup_radius [D]
DT = 1e-3                    # pitcher / table step [s]
POSE_SIGMA = 0.04            # [s] gaussian smoothing of the interpolated poses
DISC_SIGMA = 0.15            # [s] gaussian smoothing of the disc centre / surface height (old disc_fit.py tracks only)
ON_WIN = 1.0                 # [s] running-max window of the tilt envelope
ON_DTILT = 6.0               # [deg] tilt below the envelope that counts as 'stream off'
ON_FRAC = 0.98               # fraction of the observed-on time the calibrated pitcher must stream
ON_PLATEAU = 0.01           # ... or within this of what a very full pitcher (500 mL) reaches, if that is less
Q_ON = 1e-6                  # [m^3/s] stream 'on' threshold (= drvf_record's 1 mL/s floor)
TAIL = 1.0                   # [s] after the last pose: jets land, short settle
PHASE_DT = 0.1               # [s] synthetic phase length
ALPHA_DEG = 0.0              # CAD lip slope (S1 used 0 deg)
SIM_SUB = "sim_S1"
ASSUMPTIONS = dict(pitcher="NX CAD stand-in (s1_forward.geometry(0)), spout tip anchored at the measured tip",
                   law="ForwardLaw() defaults (C_d 0.6, C_v 1, H = max head, projected widths)",
                   fill="smallest V0 with Q >= 1 mL/s on >= min(98%, f(500 mL) - 1%) of the observed-on time",
                   observed_on=f"axis tilt >= running max over {ON_WIN} s - {ON_DTILT} deg",
                   scale="cup radius 0.49 D <-> median disc radius; positions, velocities / L, flows / L^3",
                   smoothing=f"poses sigma {POSE_SIGMA} s, disc sigma {DISC_SIGMA} s", tail_s=TAIL,
                   closure="S1 parameters (latte_imex/data/s1_forward), Params defaults (visc symmetric, offcup account)")


# ----------------------------------------------------------------------------- data
def export(src, patterns=PATTERNS):
    """Copy the measured pitcher poses (session scratch pours/<p>/tip.npz) into raw_data/rig/results/<p>/pitcher_pose.npz."""
    for p in patterns:
        z = np.load(os.path.join(src, p, "tip.npz"))
        np.savez_compressed(os.path.join(RES, p, "pitcher_pose.npz"), t=z["t"], rotvec=z["poses"][:, :3],
                            pos=z["poses"][:, 3:], tip_p=z["tip_p"], tip_board=z["tip_board"], tilt_deg=z["tilt"],
                            az_deg=z["az"], note=np.array("board frame (up = -z); pitcher z = axis to the rim, spout +y; "
                                                          "x_board = R(rotvec) x_p + pos"))
        print(p, len(z["t"]), "poses ->", os.path.join(RES, p, "pitcher_pose.npz"))


# pitcher pose track: 'v2' (raw_data/rig/pose_v2; raw_data/rig/pose_v2/README.md: one continuous fit of both cameras'
# tags + CAD-to-depth, no flips, ~0.2 deg / ~1 mm per instant in slow motion) or 'v1' (the per-sample tag track, kept
# as pitcher_pose_v1.npz for reproducing earlier results): environment LATTE_POSE
POSE_FILES = {"v2": "pitcher_pose.npz", "v1": "pitcher_pose_v1.npz"}


POSE_DEFAULT = "v2"         # since 2026-10-21 (fill and stream feet re-derived with v2); "v1" reproduces earlier results


def pose_version():
    v = os.environ.get("LATTE_POSE", POSE_DEFAULT)
    if v not in POSE_FILES:
        raise ValueError(f"LATTE_POSE={v}: one of {tuple(POSE_FILES)}")
    return v


def stream_foot_file(p):
    """The measured stream feet (stream_foot.py) belonging to the active pose track (its search starts at the tip)."""
    return os.path.join(RES, p, "stream_foot_v1.json" if pose_version() == "v1" else "stream_foot.json")


def tip_board_at(p, t):
    """Spout tip in the board frame [m] of the active pose track at take times t (linear between its samples)."""
    z = np.load(os.path.join(RES, p, POSE_FILES[pose_version()]))
    t = np.atleast_1d(np.asarray(t, float))
    return np.stack([np.interp(t, z["t"], z["tip_board"][:, k]) for k in range(3)], 1)


def load_take(p):
    z = np.load(os.path.join(RES, p, POSE_FILES[pose_version()]))
    P = {k: z[k] for k in ("t", "rotvec", "pos", "tip_p")}
    with open(os.path.join(RES, p, "disc.json")) as f:
        D = json.load(f)
    P["registered"] = bool(D and D[0].get("source") == "cup_register.py")
    if not P["registered"]:                               # old free per-frame circles: only the accepted fits
        D = [d for d in D if d.get("fit_ok", True)]
    P["L_m"] = D[0].get("L_m") if P["registered"] else None
    P["disc_t"] = np.array([d["t"] for d in D])
    P["disc_c"] = np.array([d["disc_centre"] for d in D])
    P["disc_z"] = np.array([d["surf_z"] for d in D])
    P["disc_R"] = np.array([d["disc_R"] for d in D])
    return P


def _smooth(y, sigma_s, dt=DT):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(y, sigma_s / dt, axis=0, mode="nearest") if sigma_s > 0 else y


S_FLIP = np.diag([1.0, -1.0, -1.0])                       # board -> cup frame F (x, -y, up)
M_CAD = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # columns: CAD X, Y, Z in the pitcher frame


DTILT = {}                                                 # pattern -> extra pitcher rotation about its x axis [deg]
TIP_OVERRIDE = None                                        # spout tip in the pitcher (tag) frame [m], else pitcher_pose.npz
LAW_OVERRIDE = None                                        # ForwardLaw fields, e.g. EXIT_CAL
# exit of the stream calibrated to the measured stream (E0 trusted frames of tulip + leaf: landing point and path
# slope; never the pattern): exit speed C_v sqrt(2 g H) with C_v 1.5, direction lifted 9.5 deg towards the spout
# heading, origin 3 mm along it.  Landing bias 8-10 mm -> ~2 mm on all five pours, horizontal arrival speed 0.08-0.10 ->
# 0.18-0.22 m/s, stream slope 0.15-0.17 -> 0.29-0.34 (measured 0.25-0.43).  V2_SURFACE_MODEL.md section 8.
EXIT_CAL = dict(C_v=1.5, exit_lift=math.radians(9.5), anchor=3.0e-3)


def cup_poses(p, dt=DT):
    """Measured poses on a uniform grid in the cup frame F (t = 0 at the first sample that has both a pose and a disc).
    Returns dict t (emission times (k + 1/2) dt), tip (n,3) [m], yaw, tilt, roll [rad] (CAD convention), axis_tilt_deg
    (pitcher axis from vertical), L (length unit [m]), R_disc [m], t0_abs, and the raw samples in the same frame."""
    from scipy.spatial.transform import Rotation, Slerp
    P = load_take(p)
    t_abs = P["t"]
    t0 = max(t_abs[0], P["disc_t"][0])
    t1 = min(t_abs[-1], P["disc_t"][-1])
    n = int(math.floor((t1 - t0) / dt))
    tg = t0 + (np.arange(n) + 0.5) * dt
    rots = Rotation.from_rotvec(P["rotvec"])
    tip_p = np.asarray(P["tip_p"] if TIP_OVERRIDE is None else TIP_OVERRIDE, float)
    if DTILT.get(p):                                       # pitcher-axis correction (full_pour.DTILT), tip kept in place
        rx = Rotation.from_euler("x", DTILT[p], degrees=True)
        rots = rots * rx; tip_p = rx.inv().apply(tip_p)
    q = Slerp(t_abs, rots)(tg).as_quat()
    q *= np.sign(np.sum(q * q[:1], 1))[:, None]           # one hemisphere (no flips across the series)
    for i in range(1, n):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q = _smooth(q, POSE_SIGMA, dt)
    q /= np.linalg.norm(q, axis=1)[:, None]
    Rb = Rotation.from_quat(q).as_matrix()                 # pitcher -> board
    pos = _smooth(np.stack([np.interp(tg, t_abs, P["pos"][:, k]) for k in range(3)], 1), POSE_SIGMA, dt)
    tip_b = np.einsum("nij,j->ni", Rb, tip_p) + pos
    ds = 0.0 if P["registered"] else DISC_SIGMA           # the registered track is shared as it is
    cx = _smooth(np.interp(tg, P["disc_t"], P["disc_c"][:, 0]), ds, dt)
    cy = _smooth(np.interp(tg, P["disc_t"], P["disc_c"][:, 1]), ds, dt)
    cz = _smooth(np.interp(tg, P["disc_t"], P["disc_z"]), ds, dt)
    rel = tip_b - np.stack([cx, cy, cz], 1)
    tip = rel @ S_FLIP.T
    Rc = S_FLIP @ Rb @ M_CAD                               # CAD body -> cup frame
    eul = Rotation.from_matrix(Rc).as_euler("ZYX")         # R = Rz(yaw) Ry(tilt) Rx(roll)
    axis = Rb[:, :, 2]
    axis_tilt = np.degrees(np.arccos(np.clip(-axis[:, 2], -1, 1)))
    Rd = float(P["L_m"] if P["L_m"] else np.median(P["disc_R"]))
    # raw samples (for the figures): tip in disc units, image orientation (x, +y board)
    rb_raw = rots.as_matrix()
    tipr = np.einsum("nij,j->ni", rb_raw, P["tip_p"]) + P["pos"]
    m = (t_abs >= t0) & (t_abs <= t1)
    crx = np.interp(t_abs[m], P["disc_t"], P["disc_c"][:, 0]); cry = np.interp(t_abs[m], P["disc_t"], P["disc_c"][:, 1])
    Rr = np.full(int(m.sum()), Rd) if P["registered"] else np.interp(t_abs[m], P["disc_t"], P["disc_R"])
    raw_uv = np.stack([(tipr[m, 0] - crx) / Rr, (tipr[m, 1] - cry) / Rr], 1)
    return dict(t=tg - t0, tip=tip, yaw=eul[:, 0], tilt=eul[:, 1], roll=eul[:, 2], axis_tilt_deg=axis_tilt,
                L=Rd / CUP_R, R_disc=Rd, t0_abs=float(t0), dt=dt, raw_t=t_abs[m] - t0, raw_uv=raw_uv)


def observed_on(axis_tilt_deg, dt=DT):
    """'Stream on' from the data alone: axis tilt within ON_DTILT of its running max over the last ON_WIN s."""
    from scipy.ndimage import maximum_filter1d
    w = int(round(ON_WIN / dt))
    env = maximum_filter1d(axis_tilt_deg, size=w, origin=(w - 1) // 2, mode="nearest")   # trailing window
    return axis_tilt_deg >= env - ON_DTILT, env


# ----------------------------------------------------------------------------- pitcher
_FP = {}


def forward_pitcher():
    if "fp" not in _FP:
        gp = os.environ.get("PITCHER_GEOM")             # e.g. latte_imex/pitcher_geom/pitcher_real.json (real pitcher)
        if gp:
            from latte_imex.pitcher import GridPitcherGeometry
            geom = GridPitcherGeometry.load(gp, lip_slope=math.radians(ALPHA_DEG))
        else:
            geom = SF.geometry(ALPHA_DEG)
        law = json.loads(os.environ.get("PITCHER_LAW", "{}"))  # e.g. {"C_d": 1.2} (flow-law sensitivity runs)
        law.update(LAW_OVERRIDE or {})
        _FP["fp"] = ForwardPitcher(geom, ForwardLaw(**law))
    return _FP["fp"]


def run_pitcher(C, V0, sub=1):
    fp = forward_pitcher()
    s = slice(None, None, sub)
    return fp.run(C["t"][s], C["tip"][s], C["yaw"][s], C["tilt"][s], C["roll"][s], V0, C["dt"] * sub)


def on_fraction(E, on_obs):
    return float(np.mean(E["Q_o"][on_obs] >= Q_ON)) if on_obs.any() else 1.0


def calibrate_fill(C, on_obs, lo=5e-6, hi=500e-6, tol=0.5e-6, sub=2, log=print):
    """Smallest V0 [m^3] with on_fraction >= ON_FRAC (bisection; the on fraction grows with the fill)."""
    oo = on_obs[::sub]
    f_hi = on_fraction(run_pitcher(C, hi, sub), oo)
    target = min(ON_FRAC, f_hi - ON_PLATEAU)
    if f_hi < ON_FRAC:
        log(f"  fill: even {hi * 1e6:.0f} mL streams only {f_hi:.1%} of the observed-on time; target {target:.1%}")
    hist = []
    while hi - lo > tol:
        m = 0.5 * (lo + hi)
        f = on_fraction(run_pitcher(C, m, sub), oo)
        hist.append((m * 1e6, f))
        if f >= target:
            hi = m
        else:
            lo = m
    log("  fill bisection: " + ", ".join(f"{v:.1f} mL->{f:.3f}" for v, f in hist))
    return hi, target


def weir_emission(C, E_ref, on_obs, env, V_target, law=None):
    """Alternative flow: Q = c_w [theta - (env - ON_DTILT)]_+^{3/2} (theta = axis tilt [deg]); jet geometry (origin,
    lip velocity, exit direction, wetted width) from a full-pitcher forward run E_ref; exit speed from the weir head on
    that width.  c_w makes the poured volume V_target."""
    law = law or ForwardLaw()
    dt = C["dt"]
    x = np.maximum(C["axis_tilt_deg"] - (env - ON_DTILT), 0.0) ** 1.5
    c_w = V_target / max(x.sum() * dt, 1e-30)
    Q = c_w * x
    E = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in E_ref.items()}
    K = (2.0 / 3.0) * law.C_d * math.sqrt(2 * law.g)
    w = np.maximum(E_ref["width"], 1e-4)
    H = (Q / (K * w)) ** (2.0 / 3.0)
    U = law.C_v * np.sqrt(2 * law.g * H)
    v_lip = E_ref["v_o"] - E_ref["U"][:, None] * E_ref["d_o"]
    v_o = v_lip + U[:, None] * E_ref["d_o"]
    g = law.g
    vz, z0 = v_o[:, 2], E_ref["origin"][:, 2]
    tf = np.where(z0 > 0, (vz + np.sqrt(vz ** 2 + 2 * g * np.maximum(z0, 0.0))) / g, 0.0)
    E.update(Q_o=Q, U=U, H=H, v_o=v_o, Q_spill=np.zeros_like(Q), t_f=np.where(Q > 0, tf, np.nan),
             x_hit=E_ref["origin"][:, :2] + v_o[:, :2] * tf[:, None], U_perp=np.sqrt(vz ** 2 + 2 * g * np.maximum(z0, 0.0)),
             u_h=v_o[:, :2].copy(), t_arr=E_ref["t"] + tf, V_P=E_ref["V0"] - np.cumsum(Q) * dt)
    return E, c_w


def build_inlet(p, flow="pitcher", V0=None, log=print):
    """Poses -> fill -> emission -> arrivals (SI) for pattern p.  Returns (C, E, R, info)."""
    t0w = time.time()
    C = cup_poses(p)
    on_obs, env = observed_on(C["axis_tilt_deg"], C["dt"])
    target = None
    if V0 is None:
        V0, target = calibrate_fill(C, on_obs, log=log)
    E = run_pitcher(C, V0)
    info = dict(pattern=p, flow=flow, V0_mL=V0 * 1e6, on_target=target, T_pour=float(C["t"][-1] + 0.5 * C["dt"]), L_m=C["L"],
                R_disc_mm=C["R_disc"] * 1e3, observed_on_frac=float(on_obs.mean()),
                stream_on_frac_when_obs_on=on_fraction(E, on_obs),
                stream_on_frac_when_obs_off=float(np.mean(E["Q_o"][~on_obs] >= Q_ON)) if (~on_obs).any() else None,
                poured_mL=float(E["Q_o"].sum() * C["dt"] * 1e6), spill_pitcher_mL=float(E["Q_spill"].sum() * C["dt"] * 1e6),
                assumptions=ASSUMPTIONS)
    if flow == "weir":
        E_ref = run_pitcher(C, 400e-6)
        E, c_w = weir_emission(C, E_ref, on_obs, env, E["Q_o"].sum() * C["dt"])
        info.update(c_w=c_w, poured_mL=float(E["Q_o"].sum() * C["dt"] * 1e6),
                    stream_on_frac_when_obs_on=on_fraction(E, on_obs))
    n_out = int(round((info["T_pour"] + TAIL) / C["dt"]))
    R = ForwardPitcher.arrivals(E, 0.0, n_out, C["dt"])
    q = E["Q_o"] > Q_ON
    hit = E["x_hit"][q]
    info.update(n_out=n_out, V_cup_mL=float(R["V_cup"][-1] * 1e6),
                Q_mean_on_mLs=float(E["Q_o"][q].mean() * 1e6) if q.any() else 0.0,
                Q_max_mLs=float(E["Q_o"].max() * 1e6), U_exit_mean=float(E["U"][q].mean()) if q.any() else 0.0,
                U_perp_mean=float(E["U_perp"][q].mean()) if q.any() else 0.0,
                tip_z_mm=[float(C["tip"][:, 2].min() * 1e3), float(np.median(C["tip"][:, 2]) * 1e3), float(C["tip"][:, 2].max() * 1e3)],
                foot_minus_tip_mm=float(np.median(np.hypot(*(hit - C["tip"][q, :2]).T)) * 1e3) if q.any() else None,
                tilt_cad_deg=[float(np.degrees(C["tilt"].min())), float(np.degrees(C["tilt"].max()))],
                roll_cad_deg=[float(np.degrees(C["roll"].min())), float(np.degrees(C["roll"].max()))],
                wall_s=time.time() - t0w)
    C["on_obs"], C["env"] = on_obs, env
    return C, E, R, info


# ----------------------------------------------------------------------------- program
def phases_meta(p, C, R, T):
    """Synthetic phase list along the tip path (D units) + FwdProgram meta (S1 closure)."""
    S1 = SF.load_meta("heart")
    L = C["L"]
    nph = int(math.ceil(T / PHASE_DT - 1e-9))
    ev = [min(k * PHASE_DT, T) for k in range(nph + 1)]
    ev[-1] = T
    tt = C["t"]
    xy = C["tip"][:, :2] / L
    phases, cls = [], []
    for k in range(nph):
        a, b = ev[k], ev[k + 1]
        pa = [float(np.interp(a, tt, xy[:, 0])), float(np.interp(a, tt, xy[:, 1]))]
        pb = [float(np.interp(b, tt, xy[:, 0])), float(np.interp(b, tt, xy[:, 1]))]
        m = (tt >= a) & (tt < b)
        on = bool(m.any() and C["on_obs"][m].mean() > 0.5)
        phases.append(dict(duration=b - a, start=pa, end=pb, Q=0.02 if on else 0.0, radius=0.02, amplitude=0.0,
                           frequency=0.0, jet=[0.0, 0.0], U_perp=0.4))
        cls.append("low" if on else "stop")
    cfg = {k: v for k, v in S1["config"].items() if k not in ("phases", "replay_output_times", "selected_from",
                                                              "replay_scope", "shared_parameters_sha256")}
    cfg.update(phases=phases, name=f"real_{p}", action=p, source="latte_imex.rig.real_program (measured poses)",
               continuous_actions=False)
    return dict(pattern=p, config=cfg, kQ=S1["kQ"], stop=False, cls=cls, events=ev, duration=T, dt=C["dt"])


def scaled_record(R, L):
    """Arrival record in metres -> the same record as if the cup were 0.08 m across (FwdProgram divides by SF.D)."""
    k = SF.D / L
    S = dict(R)
    S["x_hit"], S["u_h"], S["U_perp"], S["Q"] = R["x_hit"] * k, R["u_h"] * k, R["U_perp"] * k, R["Q"] * k ** 3
    return S


def real_program(p, N=256, flow="pitcher", V0=None, log=print, closure=None):
    """A latte_imex.run.run program (control 'v05') for the measured pour of pattern p.  Returns (prog, C, E, R, info).
    closure: V0.5 parameters (v05_adapter.SPACE keys, nu, D) replacing the S1 values; its switches e0 (0 / 1 jet /
    2 world), exit_cal (the calibrated stream exit, EXIT_CAL), circ, upwell are applied as in the fit."""
    global LAW_OVERRIDE
    closure = {k: v for k, v in (closure or {}).items() if not k.startswith("_")}     # "_note" etc.: comments
    exit_cal = bool(closure.pop("exit_cal", 0))
    prev = LAW_OVERRIDE
    if exit_cal:
        LAW_OVERRIDE = EXIT_CAL
        _FP.clear()
    try:
        C, E, R, info = build_inlet(p, flow, V0, log)
    finally:
        if exit_cal:
            LAW_OVERRIDE = prev
            _FP.clear()
    T = info["T_pour"] + TAIL
    meta = phases_meta(p, C, R, T)
    foot, e0 = (0.0, 0.0), False
    if closure:
        foot = (float(closure.pop("foot_a", 0.0)), float(closure.pop("foot_b", 0.0)))
        e0 = closure.pop("e0", 0)
        if not closure.pop("circ", 1):                 # fitted without the surface current (diff/fit.py --circ off)
            for k in ("kappa_M", "eps_f", "kappa_w"):
                closure[k] = 0.0
        meta["config"]["parameters"].update(closure)
        meta["kQ"] = float(closure.get("kappa_Q", meta["kQ"]))
    prog = SF.FwdProgram(p, scaled_record(R, C["L"]), N, meta=meta)
    from ..diff import schedule as SCH
    e0 = SCH.e0_mode(e0)
    if e0 == "jet":                                    # E0: measured stream foot (rig/stream_foot.py), closure "e0" 1
        SCH.shift_program(prog, *SCH.e0_offsets(p))
    elif e0 == "world":                                # ... in the cup frame (closure "e0" 2)
        SCH.shift_program_world(prog, *SCH.e0_world_offsets(p, prog))
    if any(foot):                                      # landing-point correction of the differentiable fit
        from ..diff.schedule import shift_program
        shift_program(prog, *foot)
    return prog, C, E, R, info


def sim_dir(p, flow="pitcher", N=256):
    return os.path.join(RES, p, SIM_SUB + ("" if flow == "pitcher" else "_" + flow) + ("" if N == 256 else f"_N{N}"))


def solve(p, N=256, flow="pitcher", video=True, V0=None, log=print, closure=None):
    """Pitcher + latte_imex.run.run on the measured pour; writes <results>/<p>/sim_S1[_weir]/."""
    import latte_imex.run as RR
    prog, C, E, R, info = real_program(p, N, flow, V0, log, closure)
    log(f"{p}: V0 {info['V0_mL']:.1f} mL, poured {info['poured_mL']:.1f} mL, stream on {info['stream_on_frac_when_obs_on']:.1%} of "
        f"observed-on, Q_on mean {info['Q_mean_on_mLs']:.1f} mL/s, U_perp {info['U_perp_mean']:.2f} m/s, L {info['L_m'] * 1e3:.1f} mm")
    od = sim_dir(p, flow, N)
    os.makedirs(od, exist_ok=True)
    k = max(1, int(round(0.01 / C["dt"])))
    np.savez_compressed(os.path.join(od, "inlet.npz"), t=C["t"][::k], tip=C["tip"][::k], yaw=C["yaw"][::k], tilt=C["tilt"][::k],
                        roll=C["roll"][::k], axis_tilt_deg=C["axis_tilt_deg"][::k], on_obs=C["on_obs"][::k],
                        Q_o=E["Q_o"][::k], V_P=E["V_P"][::k], U=E["U"][::k], U_perp=E["U_perp"][::k], x_hit=E["x_hit"][::k],
                        a_t=R["t"][::k], a_Q=R["Q"][::k], a_x=R["x_hit"][::k], a_on=R["on"][::k], L=C["L"],
                        raw_t=C["raw_t"], raw_uv=C["raw_uv"], meta=json.dumps(info, default=float))
    orig = RR.v05_program
    RR.v05_program = lambda nm: prog
    t0 = time.time()
    try:
        sol, m = RR.run(p, N=N, out=od, video=video, quiet=True, control="v05")
    finally:
        RR.v05_program = orig
    c = sol.concentration()
    np.savez_compressed(os.path.join(od, "c.npz"), c=c.astype(np.float32), mask=sol.g.mask.copy())
    for fn in ("snapshots_c.npz", "final_state.npz", "trajectory.json"):
        try:
            os.remove(os.path.join(od, fn))
        except OSError:
            pass
    info.update(solver_wall_s=time.time() - t0, T=m["T"], ledger=m["ledger"], N=N, closure=closure)
    with open(os.path.join(od, "info.json"), "w") as f:
        json.dump(info, f, indent=1, default=float)
    log(f"{p}: solver {info['solver_wall_s']:.0f} s -> {od}")
    return od, info


# ----------------------------------------------------------------------------- comparison
def sim_in_disc(c, n, extent):
    """Simulated field (solver grid, cup radius 0.49) resampled onto the target's disc-coordinate grid (row = +y board)."""
    from scipy.ndimage import map_coordinates
    N = c.shape[0]
    s = np.linspace(-extent, extent, n)
    U, V = np.meshgrid(s, s)
    x, y = U * CUP_R, -V * CUP_R                           # simulator frame F: y = -board y
    j = (x + 0.5) * N - 0.5
    i = (y + 0.5) * N - 0.5
    cc = np.nan_to_num(c, nan=0.0)
    out = map_coordinates(cc, [i, j], order=1, mode="nearest")
    out[U ** 2 + V ** 2 > 1.0] = np.nan
    return out


def metrics(p, flow="pitcher", N=256):
    z = np.load(os.path.join(sim_dir(p, flow, N), "c.npz"))
    T = np.load(os.path.join(TGT, f"{p}.npz"))
    cs = sim_in_disc(z["c"].astype(float), T["white"].shape[0], float(T["extent"]))
    m = T["valid"] & np.isfinite(cs)
    a, b = (cs > 0.5) & m, T["white"] & m
    return dict(iou=float((a & b).sum() / max((a | b).sum(), 1)), white_ratio=float(a.sum() / max(b.sum(), 1)),
                white_frac_sim=float(a.sum() / m.sum()), white_frac_target=float(b.sum() / m.sum())), cs, T


def compare(patterns=PATTERNS, flow="pitcher", N=256, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = out or os.path.join(RES, "sim_vs_real.png" if flow == "pitcher" else f"sim_vs_real_{flow}.png")
    pats = [p for p in patterns if os.path.exists(os.path.join(sim_dir(p, flow, N), "c.npz"))]
    fig, ax = plt.subplots(len(pats), 4, figsize=(12.4, 3.2 * len(pats)), dpi=110, squeeze=False)
    res = {}
    th = np.linspace(0, 2 * np.pi, 200)
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("milk", [(0.36, 0.20, 0.09), (0.66, 0.47, 0.30), (1.0, 0.97, 0.92)])
    cm.set_bad((0.15, 0.15, 0.15))
    for r, p in enumerate(pats):
        M, cs, T = metrics(p, flow, N)
        ex = float(T["extent"])
        info = json.load(open(os.path.join(sim_dir(p, flow, N), "info.json")))
        I = np.load(os.path.join(sim_dir(p, flow, N), "inlet.npz"))
        res[p] = dict(M, V0_mL=info["V0_mL"], poured_mL=info["poured_mL"], V_cup_mL=info["V_cup_mL"],
                      Q_mean_on_mLs=info["Q_mean_on_mLs"], U_perp_mean=info["U_perp_mean"])
        L = float(I["L"])
        tip_uv = np.stack([I["tip"][:, 0], -I["tip"][:, 1]], 1) / (L * CUP_R)
        on = I["Q_o"] >= Q_ON
        foot_uv = np.stack([I["x_hit"][:, 0], -I["x_hit"][:, 1]], 1)[on] / (L * CUP_R)
        img = T["img"][..., ::-1]
        kw = dict(extent=(-ex, ex, ex, -ex), interpolation="nearest")
        ax[r, 0].imshow(img, **kw)
        ax[r, 0].plot(I["raw_uv"][:, 0], I["raw_uv"][:, 1], ".", ms=2.5, color="tab:orange", label="spout tip (raw)")
        ax[r, 1].imshow(np.where(T["valid"], T["white"].astype(float), np.nan), cmap="gray", vmin=0, vmax=1, **kw)
        ax[r, 2].imshow(cs, cmap=cm, vmin=0, vmax=0.8, **kw)
        ax[r, 2].plot(foot_uv[:, 0], foot_uv[:, 1], "-", lw=0.6, color="tab:cyan", label="stream foot (sim)")
        ov = np.zeros(cs.shape + (3,))
        m = T["valid"] & np.isfinite(cs)
        a, b = (cs > 0.5) & m, T["white"] & m
        ov[a & b] = (1, 1, 1); ov[a & ~b] = (0.95, 0.35, 0.2); ov[~a & b] = (0.2, 0.55, 0.95)
        ov[~m] = (0.5, 0.5, 0.5)
        ax[r, 3].imshow(ov, **kw)
        ax[r, 3].plot(tip_uv[:, 0], tip_uv[:, 1], "-", lw=0.7, color="yellow", label="spout tip")
        ax[r, 3].plot(foot_uv[:, 0], foot_uv[:, 1], "-", lw=0.7, color="lime", label="stream foot")
        for k in range(4):
            ax[r, k].plot(np.cos(th), np.sin(th), "-", lw=0.6, color="0.6")
            ax[r, k].set_xlim(-ex, ex); ax[r, k].set_ylim(ex, -ex)
            ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
        ax[r, 0].set_ylabel(p, fontsize=11)
        ax[r, 0].set_title("target + spout tip" if r == 0 else "", fontsize=9)
        ax[r, 1].set_title("target white mask" if r == 0 else "", fontsize=9)
        ax[r, 2].set_title(f"sim c (S1), V0 {info['V0_mL']:.0f} mL, poured {info['poured_mL']:.0f} mL", fontsize=8)
        ax[r, 3].set_title(f"IoU {M['iou']:.2f}  white ratio {M['white_ratio']:.2f}", fontsize=9)
        if r == 0:
            ax[r, 3].legend(fontsize=6, loc="lower right")
    fig.text(0.5, 0.003, "overlay: white = both, orange = sim only (c > 0.5), blue = target only; disc coordinates, row = +y "
             "(board); flow: " + ("forward pitcher (NX stand-in), fill calibrated to the observed stream" if flow == "pitcher"
                                  else "weir law in tilt"), ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(res, f, indent=1)
    for p, v in res.items():
        print(f"{p:11s} IoU {v['iou']:.3f} white ratio {v['white_ratio']:.2f} (sim {v['white_frac_sim']:.3f}, target "
              f"{v['white_frac_target']:.3f})  V0 {v['V0_mL']:.0f} mL poured {v['poured_mL']:.0f} mL")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "inlet", "run", "compare"])
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--flow", default="pitcher", choices=["pitcher", "weir"])
    ap.add_argument("--V0", type=float, default=None, help="initial fill [mL] (default: calibrated)")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args(argv)
    pats = [a.pattern] if a.pattern else list(PATTERNS)
    V0 = None if a.V0 is None else a.V0 * 1e-6
    if a.cmd == "export":
        export(a.src, pats)
    elif a.cmd == "inlet":
        for p in pats:
            C, E, R, info = build_inlet(p, a.flow, V0)
            print(json.dumps({k: v for k, v in info.items() if k != "assumptions"}, default=float))
    elif a.cmd == "run":
        for p in pats:
            solve(p, a.N, a.flow, not a.no_video, V0)
    else:
        compare(pats, a.flow, a.N)


if __name__ == "__main__":
    main()
