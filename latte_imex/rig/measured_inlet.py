"""Measured inlet (mi27, SPEC27 revision 2 section 3): the stream kinematics measured on the validated milk column,
replayed onto the wall-exit pour model inside the measured windows.  An input measurement (the hand's pour), never
corrected from the white pattern.

Conventions (SPEC27 section 0 / 1).
  Units m, s, m/s.  Cup frame F = (x_b, -y_b, up) about the registered disc centre (real_program.S_FLIP); the forward
  pitcher, its arrival record and u_F below are all in F.  Clock: real_program.cup_poses t (t0 = first pose + disc
  sample; tulip t0 98.9101, leaf 11.0818), identical under pose v1 and v2.  Solver units: lengths / L, L = R_disc / 0.49.
  Calibration pours tulip and leaf only; heart, push_heart and swan are held out (no measured inlet; their inlet is
  MI-law, mode 'law').

What was measured (M1: inversion of the validated column, col25 rev-2 Method-A foot + line slope over the frame's point
heights, the pose-v2 spout tip + a declared exit PRIOR d_a in [-5, 0] mm along the heading, d_z in [0, 3] mm up, central
-2.5 / +1.5 mm, free-falling locus of parcels from the moving exit):
  u_F   horizontal arrival velocity (2), w_f = U_perp vertical arrival speed, a_f flight time (prior-dominated, +-12 ms
        structural), each with its per-frame statistical sd68 and systematic sd.  Structural term (exit placement and
        slope operator): +-0.04 m/s on u_h, +-0.04 m/s on U_perp, +-12 ms on a_f (SPEC27 1.1).
  Gate (SPEC27 1.1, declared before its effect was computed): F0 = use_kin and spout height < 15 mm; E1 = central exit
        speed <= sqrt(2 g (H_head + 3 mm)); windows = clusters of accepted frames split at gaps > 0.5 s with >= 3 frames
        (tulip 0-1.70 s and 3.38-4.14 s, leaf 0-4.12 s); taper w(t) = 1 inside, cos^2 to 0 over GAP_S = 0.25 s.
Not measured (nothing is built on it): the flow level Q (stays the weir's), any high pour (H >= 15 mm), the exit point and
direction, the held-out pours, the onset.  Unresolved residual R1: the free-fall column model misfits the measured
column shape (~10 sigma pooled); it moves u_h only inside the structural term.

Replay (mode 'replay', SPEC27 3.3).  Emission, Q and origin are MI-law's (ForwardLaw(exit='wall', capillary='round'),
real_program.EXIT_LAW).  On the gated frames r_q = q_meas - q_law(t_frame) for q in (ux, uy, wf, af), q_law the MI-law
arrival record Q-weighted over |t_arr - t| <= 20 ms (the rule-2 pairing); r_hat = posterior mean of a zero-mean GP
(Matern-3/2 s_f, l + white s_n + per-point noise eta * stat sd68) with the hyperparameters FROZEN in stream_kin.json
(spec/work2/resgp.json; no optimisation at run time).  Per parcel i: t_arr = t_em + t_f,law + w r_hat_af(t_arr) (fixed
point, 2 passes from the law's t_f), u_h = u_h,law + w r_hat_u(t_arr), U_perp = U_perp,law + w r_hat_wf(t_arr).  Outside the
taper support the replay IS the law (exactly).  x_hit stays the law's; the landing is moved onto the validated feet by the
E0 'world' estimator with LATTE_FOOT=mi (stream_foot_mi.json, written by foot_file).

    python3 -m latte_imex.rig.measured_inlet export --src SC/mi27        stream_kin.json (allow-list), stream_foot_mi.json
    python3 -m latte_imex.rig.measured_inlet foot                         stream_foot_mi.json only (from stream_kin.json)
    python3 -m latte_imex.rig.measured_inlet show <p>                     GP hyperparameters, windows, LOO z-sd
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import numpy as np

from .real_program import MI_POURS, RES, TAIL

GAP_S = 0.25                       # [s] taper half-width outside a window (cos^2 to 0)
PAIR_S = 0.020                     # [s] rule-2 pairing half-window of the law arrival record
Q_PAIR = 1e-6                      # [m^3/s] law arrival bins taken into the pairing
QS = ("ux", "uy", "wf", "af")      # residual quantities: u_F,x, u_F,y, w_f (U_perp), a_f
STRUCT = dict(ux=0.04, uy=0.04, wf=0.04, af=0.012)    # structural term (SPEC27 1.1), sd units of each q
G = 9.81
KIN_FILE = "stream_kin.json"
FOOT_FILE = "stream_foot_mi.json"
SRC_DEFAULT = os.environ.get("MI27_SRC") or None    # the mi27 campaign directory (export only): --src or env MI27_SRC

# ----------------------------------------------------------------------------- allow-list (T9)
FRAME_KEYS = ("fi", "t", "quality", "gated", "gp", "u_F", "w_f", "a_f", "stat_sd68", "sys_sd", "rhat")
SD_KEYS = ("u_F_x", "u_F_y", "w_f", "a_f")
TOP_KEYS = ("format", "pattern", "note", "conventions", "L_solver_m", "L_m", "t0_take", "windows", "gap_s", "gate",
            "gp", "provenance", "frames")
GP_KEYS = ("model", "pairing", "hyper", "q")
HYPER_KEYS = ("s_f", "l", "s_n", "eta", "loo_z_sd", "bound_hits", "n", "resid_med")
CONV_KEYS = ("board", "cup_F", "solver", "clock")
PROV_KEYS = ("sources", "m1_code", "col25", "pose", "gate_definition", "spec", "exported_by")
SRC_KEYS = ("path", "sha256")
FORBIDDEN = ("diag", "record_diag_v1", "record_prod_v2", "reference")      # and every key starting with 'oracle'


def allowed_keys():
    """Every key stream_kin.json may contain (T9), at any depth."""
    return set(FRAME_KEYS + SD_KEYS + TOP_KEYS + GP_KEYS + HYPER_KEYS + CONV_KEYS + PROV_KEYS + SRC_KEYS + QS) | {
        "stream_kin_m1", "gate_rule2", "resgp", "pred_m5", "replay_proto", "stream_foot_col", "pitcher_pose", "spec27",
        "prereg27_2"}


def check_keys(obj, path="", bad=None):
    """Walk a JSON object; returns the list of offending key paths (not allow-listed, forbidden or oracle*)."""
    bad = [] if bad is None else bad
    allow = allowed_keys()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in allow or k in FORBIDDEN or k.lower().startswith("oracle"):
                bad.append(path + "/" + k)
            check_keys(v, path + "/" + k, bad)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_keys(v, path + f"[{i}]", bad)
    return bad


# ----------------------------------------------------------------------------- windows and taper
def taper(t, windows, gap=GAP_S):
    """w(t) = 1 inside any window [a, b], cos^2 to 0 over gap outside (max over windows)."""
    tg = np.asarray(t, float)
    w = np.zeros_like(tg)
    for a, b in windows:
        d = np.where(tg < a, a - tg, np.where(tg > b, tg - b, 0.0))
        v = np.where(d <= 0, 1.0, np.where(d < gap, np.cos(0.5 * math.pi * d / gap) ** 2, 0.0))
        w = np.maximum(w, v)
    return w


# ----------------------------------------------------------------------------- residual GP (frozen hyperparameters)
def matern32(t1, t2, sf, l):
    r = np.abs(t1[:, None] - t2[None, :]) / l
    return sf * sf * (1 + math.sqrt(3) * r) * np.exp(-math.sqrt(3) * r)


class ResidualGP:
    """Zero-mean GP of one residual series with frozen hyperparameters (same algebra as the prototype
    spec/code2/resgp.py: K = Matern32 + diag((eta s)^2 + s_n^2), Ki = inv(K), alpha = Ki y)."""

    def __init__(self, t, y, s, s_f, l, s_n, eta):
        self.t, self.y, self.s = (np.asarray(a, float) for a in (t, y, s))
        self.sf, self.l, self.sn, self.eta = float(s_f), float(l), float(s_n), float(eta)
        K = matern32(self.t, self.t, self.sf, self.l) + np.diag((self.eta * self.s) ** 2 + self.sn ** 2)
        self.Ki = np.linalg.inv(K)
        self.alpha = self.Ki @ self.y

    def mean(self, tg):
        tg = np.atleast_1d(np.asarray(tg, float))
        return matern32(tg, self.t, self.sf, self.l) @ self.alpha

    def sd(self, tg):
        tg = np.atleast_1d(np.asarray(tg, float))
        Ks = matern32(tg, self.t, self.sf, self.l)
        return np.sqrt(np.maximum(self.sf * self.sf - np.einsum("ij,jk,ik->i", Ks, self.Ki, Ks), 0.0))

    def loo_z_sd(self):
        d = np.diag(self.Ki)
        return float(np.std((self.alpha / d) * np.sqrt(d)))


# ----------------------------------------------------------------------------- data
_KIN = {}


def kin_path(p, res=RES):
    return os.path.join(res, p, KIN_FILE)


def load(p, res=RES):
    """stream_kin.json of pour p (tulip, leaf), cached."""
    if p not in MI_POURS:
        raise ValueError(f"{p}: no measured inlet (calibration pours {MI_POURS} only)")
    fn = kin_path(p, res)
    if fn not in _KIN:
        if not os.path.exists(fn):
            raise FileNotFoundError(f"{fn}: python3 -m latte_imex.rig.measured_inlet export")
        with open(fn) as f:
            _KIN[fn] = json.load(f)
    return _KIN[fn]


def bin_parcels(E, fields, n_out, t0=0.0, dt_out=None):
    """Q-weighted arrival-bin means of per-parcel fields (n,) / (n, k), with exactly the parcel windows and overlaps of
    ForwardPitcher.arrivals (and of M5's exitlaw.bin_arrivals); nan in empty bins.  Returns {name: (n_out, ...)}."""
    dt = E["dt"]
    dto = dt if dt_out is None else dt_out
    Q = E["Q_o"]
    on = Q > 0
    ta = E["t_arr"]
    lo = ta - 0.5 * dt
    hi = ta + 0.5 * dt
    prev_on = np.r_[False, on[:-1]]
    next_on = np.r_[on[1:], False]
    lo = np.where(prev_on, 0.5 * (np.r_[ta[0], ta[:-1]] + ta), lo)
    hi = np.where(next_on, 0.5 * (ta + np.r_[ta[1:], ta[-1]]), hi)
    a, b = np.minimum(lo, hi), np.maximum(lo, hi)
    b = np.maximum(b, a + 1e-9)
    vol = Q * dt
    acc = np.zeros(n_out)
    out = {k: np.zeros((n_out,) + np.shape(v)[1:]) for k, v in fields.items()}
    for i in np.nonzero(on)[0]:
        j0 = int(math.floor((a[i] - t0) / dto)); j1 = int(math.floor((b[i] - t0) / dto))
        for j in range(max(j0, 0), min(j1, n_out - 1) + 1):
            s0, s1 = t0 + j * dto, t0 + (j + 1) * dto
            ov = min(b[i], s1) - max(a[i], s0)
            if ov <= 0:
                continue
            v = vol[i] * ov / (b[i] - a[i])
            acc[j] += v
            for k, f in fields.items():
                out[k][j] += v * f[i]
    m = acc > 0
    for k in out:
        out[k][m] /= acc[m].reshape((-1,) + (1,) * (out[k].ndim - 1))
        out[k][~m] = np.nan
    return out


def law_record(E, n_out):
    """The MI-law arrival record used by the rule-2 pairing: t (bin ends), Q, u_h (2), U_perp, t_f (Q-weighted)."""
    from ..pitcher import ForwardPitcher
    R = ForwardPitcher.arrivals(E, 0.0, n_out, E["dt"])
    tf = bin_parcels(E, dict(t_f=np.nan_to_num(E["t_f"])), n_out)["t_f"]
    return dict(t=R["t"], Q=R["Q"], uv=R["u_h"], U=R["U_perp"], tf=tf)


def pair_residuals(K, rec):
    """Residuals r_q = q_meas - q_law on the gated frames that have law arrivals within +-PAIR_S (rule-2 pairing).
    Returns dict(t, fi, y {q: (n,)}, s {q: (n,)})."""
    ta = rec["t"]
    rows = []
    for f in K["frames"]:
        if not f["gated"]:
            continue
        m = (np.abs(ta - f["t"]) <= PAIR_S) & (rec["Q"] > Q_PAIR)
        if not m.any():
            continue
        w = rec["Q"][m] / rec["Q"][m].sum()
        lu = (w[:, None] * rec["uv"][m]).sum(0); lU = (w * rec["U"][m]).sum(); ltf = (w * rec["tf"][m]).sum()
        sd = f["stat_sd68"]
        rows.append(dict(t=f["t"], fi=f["fi"], ux=f["u_F"][0] - lu[0], uy=f["u_F"][1] - lu[1], wf=f["w_f"] - lU,
                         af=f["a_f"] - ltf, sx=sd["u_F_x"], sy=sd["u_F_y"], sw=sd["w_f"], sa=sd["a_f"]))
    sk = dict(ux="sx", uy="sy", wf="sw", af="sa")
    return dict(t=np.array([r["t"] for r in rows]), fi=[r["fi"] for r in rows],
                y={q: np.array([r[q] for r in rows]) for q in QS}, s={q: np.array([r[sk[q]] for r in rows]) for q in QS})


def residual_gps(K, pr):
    """One frozen-hyperparameter GP per q from the residuals pr (pair_residuals)."""
    H = K["gp"]["hyper"]
    return {q: ResidualGP(pr["t"], pr["y"][q], pr["s"][q], H[q]["s_f"], H[q]["l"], H[q]["s_n"], H[q]["eta"]) for q in QS}


RHAT_TOL = 1e-9                    # GP mean at the frame times vs the frozen r_hat of stream_kin.json (the T6(b) tolerance)


def checked_gps(K, pr, p="?"):
    """residual_gps, refusing to run on anything but the frozen replay: the paired frame set must equal the export's
    (frames with 'gp' true in stream_kin.json) and the GP mean at those frame times must equal the frozen r_hat to
    RHAT_TOL.  A mismatch (no paired frame, a clock offset, a different stream_kin.json, a different MI-law record) raises
    instead of silently replaying a different inlet (with no paired frame the GP mean is 0 and the replay would be the
    law)."""
    fr = [f for f in K["frames"] if f.get("gp")]
    want = [int(f["fi"]) for f in fr]
    got = [int(x) for x in pr["fi"]]
    if not want or got != want:
        raise RuntimeError(f"{p}: measured-inlet replay: the frames paired with the MI-law record ({len(got)}: "
                           f"{got[:6]}...) differ from the frozen GP frame set of stream_kin.json ({len(want)}: "
                           f"{want[:6]}...); refusing to replay a different inlet")
    gps = residual_gps(K, pr)
    tfr = np.array([f["t"] for f in fr], float)
    d = max(float(np.max(np.abs(gps[q].mean(tfr) - np.array([f["rhat"][q] for f in fr], float)))) for q in QS)
    if not d <= RHAT_TOL:
        raise RuntimeError(f"{p}: measured-inlet replay: GP mean at the frame times differs from the frozen r_hat of "
                           f"stream_kin.json by {d:.3e} > {RHAT_TOL:g}; refusing to replay a different inlet")
    return gps


# ----------------------------------------------------------------------------- the replay
def replay(E, p, C=None, K=None, passes=2):
    """MI-replay of an MI-law emission history E (ForwardPitcher.run with exit 'wall') for pour p (tulip, leaf).
    Returns (E', info): E' a copy of E whose flowing parcels carry the replayed u_h, U_perp, t_f, t_arr (inside the taper
    support; exactly the law's outside), plus E'['replay_w'] (taper at the replayed arrival), E'['replay_rhat'] (n, 4:
    r_hat ux, uy, wf, af there), E'['law_t_arr'], E'['law_u_h'], E'['law_U_perp'], E'['law_t_f'] (the law parcels)."""
    if p not in MI_POURS:
        raise ValueError(f"{p}: inlet 'replay' exists for {MI_POURS} only")
    if "kM" not in E:
        raise ValueError("replay needs an MI-law emission (ForwardLaw(exit='wall'))")
    K = K or load(p)
    t = np.asarray(E["t"], float)
    dt = float(E["dt"])
    n_out = int(round((t[-1] + 0.5 * dt + TAIL) / dt))
    rec = law_record(E, n_out)
    pr = pair_residuals(K, rec)
    gps = checked_gps(K, pr, p)
    wins = K["windows"]
    gap = float(K.get("gap_s", GAP_S))
    on = E["Q_o"] > 0
    idx = np.nonzero(on)[0]
    tf0 = E["t_f"][idx]
    te = t[idx]
    ta = E["t_arr"][idx].copy()                         # = te + tf0 (the law's arrival)

    def corr(q, tt, w):
        return np.where(w > 0, w * gps[q].mean(tt), 0.0)
    af = tf0.copy()
    for _ in range(passes):                             # t_arr = t_em + a_f,MR(t_arr), from the law's t_f
        w = taper(ta, wins, gap)
        af = tf0 + corr("af", ta, w)
        ta = te + af
    w = taper(ta, wins, gap)
    rh = np.stack([gps[q].mean(ta) for q in QS], 1)
    E2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in E.items()}
    for k in ("t_arr", "u_h", "U_perp", "t_f"):
        E2["law_" + k] = E[k].copy()
    du = np.where((w > 0)[:, None], w[:, None] * rh[:, :2], 0.0)
    E2["u_h"][idx] = E["u_h"][idx] + du
    E2["U_perp"][idx] = E["U_perp"][idx] + np.where(w > 0, w * rh[:, 2], 0.0)
    E2["t_arr"][idx] = ta
    E2["t_f"][idx] = af
    E2["replay_w"] = np.zeros(len(t)); E2["replay_w"][idx] = w
    E2["replay_rhat"] = np.full((len(t), 4), np.nan); E2["replay_rhat"][idx] = rh
    info = dict(n_frames=int(len(pr["t"])), windows=wins, gap_s=gap, passes=passes,
                loo_z_sd={q: gps[q].loo_z_sd() for q in QS},
                frac_parcels_in_support=float(np.mean(w > 0)) if len(w) else 0.0,
                frac_parcels_w1=float(np.mean(w >= 1.0)) if len(w) else 0.0)
    return E2, info


def law_emission(E):
    """The MI-law emission under an MI-replay one: replay() keeps the law parcels as law_t_arr / law_u_h / law_U_perp /
    law_t_f and changes nothing else, so a shallow copy with those four restored IS the law emission.  E itself when it
    carries no replay."""
    if "law_u_h" not in E:
        return E
    E2 = dict(E)
    for k in ("t_arr", "u_h", "U_perp", "t_f"):
        E2[k] = E["law_" + k]
    return E2


def band(p, t, K=None, E=None, win_s=0.5):
    """Replay band per q at times t (SPEC27 3.3 item 6; for the oracle box only, not used in the forward run): GP
    posterior sd (+) running median of the gated frames' systematic sd within +-win_s (pour median elsewhere) (+) the
    structural term.  Needs the MI emission E for the residuals (the law one; a replay emission is reverted to it by
    law_emission, so the residuals are measurement - law as in replay()).  Returns {q: (n,)}."""
    K = K or load(p)
    t = np.atleast_1d(np.asarray(t, float))
    E = law_emission(E)
    dt = float(E["dt"])
    n_out = int(round((E["t"][-1] + 0.5 * dt + TAIL) / dt))
    gps = checked_gps(K, pair_residuals(K, law_record(E, n_out)), p)
    fr = [f for f in K["frames"] if f["gated"]]
    tf = np.array([f["t"] for f in fr])
    sk = dict(ux="u_F_x", uy="u_F_y", wf="w_f", af="a_f")
    out = {}
    for q in QS:
        ss = np.array([f["sys_sd"][sk[q]] for f in fr])
        med = np.array([np.median(ss[np.abs(tf - x) <= win_s]) if np.any(np.abs(tf - x) <= win_s) else np.median(ss)
                        for x in t])
        out[q] = np.sqrt(gps[q].sd(t) ** 2 + med ** 2 + STRUCT[q] ** 2)
    return out


# ----------------------------------------------------------------------------- export (once, from the campaign)
def _sha(fn):
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def export_m1(p, src=SRC_DEFAULT, res=RES):
    """Write raw_data/rig/results/<p>/stream_kin.json from the mi27 campaign (src = SC/mi27): M1's
    stream_kin_<p>.json through an ALLOW-LIST (per frame fi, t, quality, gated flag, u_F, w_f, a_f and their stat sd68 /
    sys sd), the gate windows (spec/work2/gate_rule2.json), the frozen GP hyperparameters (spec/work2/resgp.json, eta
    free) and r_hat at the frame times (prototype algebra on M5's hashed pred_<p>.npz phys_round record), provenance.
    No oracle-bearing field (M1's 'diag', 'record_diag_v1', 'record_prod_v2') is read into the output."""
    if p not in MI_POURS:
        raise ValueError(p)
    if not src:
        raise SystemExit("measured_inlet export: the mi27 campaign directory is needed (--src <SC/mi27> or env MI27_SRC)")
    f_kin = os.path.join(src, "M1", f"stream_kin_{p}.json")
    f_gate = os.path.join(src, "spec", "work2", "gate_rule2.json")
    f_gp = os.path.join(src, "spec", "work2", "resgp.json")
    f_pred = os.path.join(src, "M5", f"pred_{p}.npz")
    f_proto = os.path.join(src, "spec", "work2", f"replay_proto_{p}.npz")
    f_spec = os.path.join(src, "spec", "SPEC27.md")
    f_pre = os.path.join(src, "spec", "PREREG27_2.md")
    M = json.load(open(f_kin))
    gate = json.load(open(f_gate))[p]
    hyp = json.load(open(f_gp))[p]["hyper"]
    gated = {r["fi"]: bool(r["gated"]) for r in gate["frames"]}
    info = M["info"]
    conv = info.get("conventions", {})
    frames = []
    for f in M["frames"]:
        c, st, sy = f["central"], f["stat"], f["sys_sd"]
        frames.append(dict(fi=int(f["fi"]), t=float(f["t"]), quality=f["quality"], gated=gated.get(f["fi"], False),
                           u_F=[float(c["u_F"][0]), float(c["u_F"][1])], w_f=float(c["w_f"]), a_f=float(c["a_f"]),
                           stat_sd68={k: float(st[k]["sd68"]) for k in SD_KEYS},
                           sys_sd={k: float(sy[k]) for k in SD_KEYS}))
    K = dict(format="mi27 stream_kin v1", pattern=p,
             note=("Measured stream kinematics at the coffee surface (M1 inversion of the validated milk column; "
                   "SPEC27 rev 2 section 1). Allow-listed fields only. Cup frame F, SI. a_f is prior-dominated."),
             conventions=dict(board=conv.get("board"), cup_F=conv.get("cup_F"), solver=conv.get("solver"),
                              clock="real_program.cup_poses t (t0 = t0_take)"),
             L_solver_m=float(info["L_solver_m"]), L_m=float(info["L_m"]), t0_take=float(info["t0_take"]),
             windows=[[float(a), float(b)] for a, b in gate["windows"]], gap_s=GAP_S,
             gate=("F0 = use_kin and H_tip < 15 mm; E1 = central exit speed <= sqrt(2 g (H_head + 3 mm)), H_head the "
                   "production weir head at the emission step nearest t - a_f; windows = accepted frames clustered at "
                   "gaps > 0.5 s, >= 3 frames, [first, last]; gated = F0 and E1 and inside a window (SPEC27 1.1)"),
             gp=dict(model=("zero-mean GP per residual r_q = q_meas - q_law: Matern-3/2 (s_f, l) + white s_n + per-point "
                            "noise eta * stat sd68; hyperparameters by maximum marginal likelihood (eta in [0.3, 3], "
                            "l in [0.04, 8] s) in spec/code2/resgp.py, frozen here"),
                     pairing=f"MI-law arrival record Q-weighted over |t_arr - t| <= {PAIR_S} s and Q > {Q_PAIR} m^3/s",
                     hyper={q: {k: hyp[f"{q}_etafree"][k] for k in ("s_f", "l", "s_n", "eta", "loo_z_sd", "bound_hits", "n")}
                            | {"resid_med": hyp.get(f"{q}_resid_med")} for q in QS}),
             frames=frames)
    # r_hat at the frame times: the prototype's algebra on M5's hashed phys_round arrival record
    Z = np.load(f_pred)
    rec = dict(t=Z["t_arr"], Q=Z["phys_round_arr_Q"], uv=Z["phys_round_arr_u_h"], U=Z["phys_round_arr_U_perp"],
               tf=Z["phys_round_arr_t_f"])
    pr = pair_residuals(K, rec)
    gps = residual_gps(K, pr)
    used = set(pr["fi"])
    for f in frames:
        f["gp"] = f["fi"] in used
        if f["gp"]:
            f["rhat"] = {q: float(gps[q].mean([f["t"]])[0]) for q in QS}
    m1code = os.path.join(src, "M1", "code")
    K["provenance"] = dict(
        sources=dict(stream_kin_m1=dict(path=f_kin, sha256=_sha(f_kin)), gate_rule2=dict(path=f_gate, sha256=_sha(f_gate)),
                     resgp=dict(path=f_gp, sha256=_sha(f_gp)), pred_m5=dict(path=f_pred, sha256=_sha(f_pred)),
                     replay_proto=dict(path=f_proto, sha256=_sha(f_proto) if os.path.exists(f_proto) else None),
                     stream_foot_col=dict(path=os.path.join(res, p, "stream_foot_col.json"),
                                          sha256=_sha(os.path.join(res, p, "stream_foot_col.json"))),
                     pitcher_pose=dict(path=os.path.join(res, p, "pitcher_pose.npz"),
                                       sha256=_sha(os.path.join(res, p, "pitcher_pose.npz"))),
                     spec27=dict(path=f_spec, sha256=_sha(f_spec)), prereg27_2=dict(path=f_pre, sha256=_sha(f_pre))),
        m1_code=[dict(path=os.path.join(m1code, fn), sha256=_sha(os.path.join(m1code, fn)))
                 for fn in sorted(os.listdir(m1code)) if fn.endswith(".py")] if os.path.isdir(m1code) else [],
        col25="col25 rev 2 Method-A foot + line slope (M1 inversion); landing feet stream_foot_col.json (validated A/B)",
        pose="pose v2 (raw_data/rig/pose_v2, pitcher_pose.npz)",
        gate_definition="SPEC27 rev 2 section 1.1 (spec/code2/gate_rule2.py)",
        spec="mi27 SPEC27 rev 2 sections 1, 3.3; PREREG27_2 section 2",
        exported_by="latte_imex.rig.measured_inlet.export_m1")
    bad = check_keys(K)
    if bad:
        raise RuntimeError(f"stream_kin.json allow-list violated: {bad[:10]}")
    fn = kin_path(p, res)
    with open(fn, "w") as f:
        json.dump(K, f, indent=1)
    _KIN.pop(fn, None)
    return fn


def foot_rows_kept(K, J):
    """fi of the stream_foot_col.json rows that stream_foot_mi.json keeps (SPEC27 rev 2 3.3 item 4 with the 1.1 frame gate,
    which 1.1 applies to stream_foot_mi.json too): validated_grade A / B, taper w(t) > 0, and the frame passes the gate
    (stream_kin.json 'gated' = F0 use_kin and H_tip < 15 mm, E1 energy, inside a window).  A col row without an M1 frame
    cannot pass F0."""
    wins, gap = K["windows"], float(K.get("gap_s", GAP_S))
    gated = {int(f["fi"]) for f in K["frames"] if f["gated"]}
    return {int(r["fi"]) for r in J["frames"]
            if r.get("validated_grade") in ("A", "B") and float(taper([r["t"]], wins, gap)[0]) > 0 and int(r["fi"]) in gated}


def foot_file(p, res=RES):
    """Write raw_data/rig/results/<p>/stream_foot_mi.json: the stream_foot_col.json rows with validated_grade A / B,
    taper w(t) > 0 (the measured-inlet windows of stream_kin.json) and a gated frame (SPEC27 1.1: F0, E1, windows; the
    gate applies to this file) unchanged; every other row tier none (quality 0, trusted False, the column_foot 'none'
    tracker values), so that the E0 'world' estimator of schedule falls back to its median kept offset elsewhere."""
    K = load(p, res)
    J = json.load(open(os.path.join(res, p, "stream_foot_col.json")))
    wins, gap = K["windows"], float(K.get("gap_s", GAP_S))
    kept = foot_rows_kept(K, J)
    rows, tiers = [], {"cand": 0, "plaus": 0, "none": 0}
    for r in J["frames"]:
        r = dict(r)
        keep = int(r["fi"]) in kept
        if not keep:
            r.update(quality=0.0, rms_mm=99.0, slope=9.9, top_mm=99.0, trusted=False)
        r["mi_window"] = bool(keep)
        tiers["cand" if r.get("trusted") else ("plaus" if r.get("quality", 0.0) > 0.2 else "none")] += 1
        rows.append(r)
    out = dict(J, source=("latte_imex/rig/measured_inlet.py foot_file: stream_foot_col.json rows of grade A / B inside the "
                          "measured-inlet windows (taper w > 0) whose frame passes the SPEC27 1.1 gate (stream_kin.json "
                          "'gated': F0, E1, windows); every other row tier none"),
               windows=wins, gap_s=gap, tiers=tiers, frames=rows,
               col_sha256=_sha(os.path.join(res, p, "stream_foot_col.json")),
               kin_sha256=_sha(kin_path(p, res)))
    fn = os.path.join(res, p, FOOT_FILE)
    with open(fn, "w") as f:
        json.dump(out, f, indent=0)
    return fn, tiers


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "foot", "show"])
    ap.add_argument("pours", nargs="*")
    ap.add_argument("--src", default=SRC_DEFAULT)
    a = ap.parse_args(argv)
    for p in a.pours or MI_POURS:
        if a.cmd == "export":
            fn = export_m1(p, a.src)
            fn2, tiers = foot_file(p)
            print(p, "->", fn, fn2, tiers)
        elif a.cmd == "foot":
            fn2, tiers = foot_file(p)
            print(p, "->", fn2, tiers)
        else:
            K = load(p)
            print(p, "windows", K["windows"], "gated", sum(f["gated"] for f in K["frames"]),
                  "gp frames", sum(f.get("gp", False) for f in K["frames"]))
            for q, h in K["gp"]["hyper"].items():
                print("  ", q, h)


if __name__ == "__main__":
    main()
