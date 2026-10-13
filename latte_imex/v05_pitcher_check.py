"""Pitcher-in-the-loop check of the V0.5 physical inlets: can the pitcher model actually pour them?

The all-physical guessed action set FS (v05allphys/final/drvf.py, class PhysF, flags --exp P2 --pulse 0.41
--lat exitmin --lag 0.08 --ztr, flow phase per pattern) and the physical inlet of the barista's real actions
(v05_adapter.PhysInlet on calib.physical_script(name, preroll=0)) were checked so far only quantity by quantity
against an envelope.  Here the whole time series is driven through latte_imex.pitcher (unmodified) with the CAD NX
pitcher, and what arrives on the coffee is compared with what the solver was fed.

WANTED (1 ms tables)
  FS    sampled from PhysF exactly as chk_full.py does: stream-foot path x_hit(t) [D], spout height z(t) (incl. the
        --ztr transitions), received flow Q(t) at the surface (taper, 0.1 s ramps, tau = 0.08 s lag, 5 Hz pulsation
        0.41 after the lag, 1-32 mL/s clip), U_perp = sqrt(0.35^2 + 2 g z), cup-frame u_h = v_hand + exit part,
        continuity footprint r = sqrt(Q/(pi U)), d; also the pre-lag command Q_cmd(t) (envelope x pulsation) and the
        lagged envelope without pulsation Q_env(t).
  real  v3min_adapter.Inlet (via PhysInlet.foot): foot path, z, wanted flow Q_w = flow x 15 mL/s; the solver inlet
        PhysInlet.state: lagged Q, U, u_h = D v_h + 0.15 m/s e_yaw, continuity footprint.
  yaw   FS: direction of the V0.5 control's jet b of the stroke (the FS exit part is along b); real: script yaw.  Between strokes
        the yaw turns with a cosine ease over the bridge (or +-0.05 s around a direct stroke boundary).

DRIVE
  Pitcher(GridPitcherGeometry 'pitcher_nx', nr = 6, OutflowLaw defaults: C_d 0.60, h_k 0.8 mm, h_on 2.5 mm,
  h_off 0.5 mm, tau 0.08 s, tau_stop 0.15 s, slosh zeta 0.3, carry velocity), fill V0 (350 mL; 250 / 500 mL for two
  patterns per set, 250 mL also for FS heart and swan), dt = 1 ms.  FastPitcher only replaces the free-surface
  evaluation by an algebraically identical one (heights along the surface normal as one mat-vec instead of transforming
  all points to the world frame; the weighted quantile from a 256-bin bucketing, sorting only the bin that holds V; same
  interpolation); selftest() checks it against Pitcher.free_surface (round-off, ~1e-14 m).
  TableBarista = pitcher.Barista with its controller unchanged (feed-forward tilt from tilt_for_flow + PI on the
  low-passed flow, 60 deg/s rate limit, swing / hold logic, spout-height rate limit 0.25 m/s, the 0.3 s stream-offset
  correction, collision lift Pitcher.required_lift against the default Cup) whose intent() returns the wanted table
  (foot path, z, Q) and whose pose carries the wanted yaw.  tilt0 (arrival tilt) = flow-onset tilt of the fill - 2 deg.
  Variants
    stock  pitcher.Barista exactly (feed-forward cache 3 %, collision lift applied raw every step).
    main   intent Q = the wanted flow at the surface (FS Q(t) incl. pulsation / real Q_w), with two numerical fixes that
           the 1 ms step needs: the hand follows required_lift through a 9-step median + 40 ms critically damped filter
           (see _HandProxy), and the feed-forward cache tolerance is 0.5 % (FF_TOL).
    cmd    (FS, main settings) intent Q = the FS pre-lag command (envelope x pulsation): FS's own tau = 0.08 s lag is then supplied
           by the pitcher's spout instead of being applied twice.
    pulse  (FS, pulsed patterns, main settings) the controller tracks the lagged envelope Q_env and an open-loop 5 Hz tilt oscillation
           about the spout tip is added (via a proxy pitcher, the barista's tilt state is untouched); its complex
           amplitude starts from the linearised weir + lag + flight transfer function and is adapted once per 5 Hz
           cycle by a lock-in on the arriving flow towards the wanted depth and phase (best case for the hand).
METRICS (per stroke for FS, per stage for real; arrival quantities from the pitcher's arrival record 'hit', exit
quantities from the release record 'jet')
  flow: volume ratio, flow-weighted mean |error| raw and after the best global delay, max |error|, delay, start delay,
  breaks; pulsation depth m1 and phase at 5 Hz (lock-in on whole cycles, first 0.4 s skipped); U_perp (also the FS
  formula at the pitcher's own drop height); exit speed u0 = sqrt(2 g h_bar); exit horizontal speed (relative to the
  pitcher) and tan alpha; cup-frame |u_h|; footprint r (continuity from the pitcher's Q, U; and the pitcher's physical
  sqrt(r1 r2)); foot position error [mm] raw and after the best per-stroke delay; spout height reached (tip z incl. the
  collision lift); tilt range / rate; yaw rate; milk used vs fill; overflow; Grant-Middleman coherence.
  Classes: DELIVERED (|rel| <= 15 %, positions / heights <= 1 mm, phase <= 10 deg), DEVIATES (by how much),
  IMPOSSIBLE (the pitcher cannot do it: collision lift > 1 mm, i.e. the wanted height is below the body / lip
  clearance; or no stream at all).
SOLVER TABLES  tables/<run>.npz: for 'pitcher' (delivered) and 'wanted' (FS / PhysInlet as fed to the solver) on the
  same 1 ms grid: active, x, y [D], S_eff = kappa_Q Q / D^3, U_perp [D/s], u_in [D/s], d_jet, r1, r2 [D], phi,
  scan_speed [D/s] (+ Q [mL/s], z [m], physical r1 / r2 for the pitcher).  TableProgram (drop-in for run.py's
  v05_program, like drvf) and TableInlet / TableV05Model (drop-in for v05_adapter.PhysInlet / V05Model) read them.

usage (from /home/user/zfChen):
  python3 -m latte_imex.v05_pitcher_check selftest
  python3 -m latte_imex.v05_pitcher_check sim fs tulip --variant main --fill 350
  python3 -m latte_imex.v05_pitcher_check batch --jobs 2        # all runs, then 'report'
  python3 -m latte_imex.v05_pitcher_check clearance              # lowest reachable spout height map
  python3 -m latte_imex.v05_pitcher_check report                 # summary_tables.md, summary.json, clearance.png
BEST CASE (ideal hand, section 'best case' below; output v05pitcher/best/):
  python3 -m latte_imex.v05_pitcher_check best_static             # reachable heights / exit speeds over yaw, roll, fill
  python3 -m latte_imex.v05_pitcher_check best_sim swan --variant ideal --fill 350   # variants: ideal, idealslosh,
                                                                  # Cd055, Cd065, tau004, tau015, hon125, hon375
  python3 -m latte_imex.v05_pitcher_check best_batch --which all --jobs 2
  python3 -m latte_imex.v05_pitcher_check best_report             # best/summary_tables.md, best/summary.json
SOLVE (V0.5 solver, FS closure, with the pitcher-delivered inlet; section 'SOLVE' below; output v05pitcher/solve/):
  python3 -m latte_imex.v05_pitcher_check solve_table             # raw 1 ms tables (delivered 350 / 250 mL + FS-wanted)
  python3 -m latte_imex.v05_pitcher_check solve_check             # records bitwise = drvf (exact times and table times)
  python3 -m latte_imex.v05_pitcher_check solve_run tulip --source pitcher   # sources: pitcher, pitcher250, wanted,
                                                                  # fsexact, pitcher_zfs (diagnostic)
  python3 -m latte_imex.v05_pitcher_check solve_batch --jobs 2 --list heart:fsexact,heart:pitcher
  python3 -m latte_imex.v05_pitcher_check solve_report            # solve/ana.json, FS_vs_pitcher(_diag).png
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

from .pitcher import Barista, Coupling, Cup, G, GridPitcherGeometry, Move, OutflowLaw, Pitcher, Pose, rotation
from .solver import Inlet

# --------------------------------------------------------------------------- paths and constants
def _review_dir():
    """The review scratch directory that holds v05allphys/final/drvf.py (env V05_REVIEW_DIR overrides)."""
    if os.environ.get("V05_REVIEW_DIR"):
        return os.environ["V05_REVIEW_DIR"]
    import glob
    hits = sorted(glob.glob("/tmp/*/-home-user/*/scratchpad/review/v05allphys/final/drvf.py"))
    return os.path.dirname(os.path.dirname(os.path.dirname(hits[-1]))) if hits else os.path.join(os.getcwd(), "review")


REVIEW = _review_dir()
FINAL = os.path.join(REVIEW, "v05allphys", "final")
EXPS = os.path.join(REVIEW, "v05allphys", "allphys_experiments.json")
OUT = os.path.join(REVIEW, "v05pitcher", "track")
PATTERNS = ("heart", "push_heart", "layered_heart", "tulip", "leaf", "swan")
CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="郁金香", leaf="树叶", swan="天鹅")
FS_FLOW_PHASE = dict(heart=None, push_heart=None, layered_heart=math.pi / 2, tulip=0.0, leaf=0.0, swan=0.0)

D = 0.08                 # cup diameter [m]
D3 = D ** 3              # 1 D^3 = 512 mL
ML = 1e-6
DT = 1e-3                # pitcher / table step [s]
V0_EXIT = 0.35           # FS exit speed [m/s]
KAPPA_Q = 2.0            # FS kappa_Q [1/D]
TOL_REL, TOL_MM, TOL_DEG = 0.15, 1.0, 10.0
FILLS_EXTRA = dict(fs=("tulip", "push_heart"), real=("push_heart", "tulip"))
FF_TOL = 0.005           # feed-forward cache tolerance of the 'main' barista (stock: 0.03)
PULSED = ("layered_heart", "tulip", "leaf", "swan")


# --------------------------------------------------------------------------- pitcher with a faster free surface
class FastPitcher(Pitcher):
    """Pitcher with an algebraically identical free_surface (differences are round-off, selftest: ~1e-14 m):
    heights along the surface normal z_w = (P - tip_b) R^T n + tip_w.n = P (R^T n) + const as one mat-vec; the
    weighted quantile is found by bucketing the heights into 256 bins (bincount of the volumes), then sorting only
    the bin that holds V; the interpolation between consecutive sorted points is the same as Pitcher.free_surface."""
    NB = 256

    def free_surface(self, pose, V):
        R = rotation(pose.yaw, pose.tilt, pose.roll)
        w = R.T @ self.n
        dv = self.geom.dv
        zw = self.geom.points @ w + (float(np.asarray(pose.tip, float) @ self.n) - float(self.geom.tip @ w))
        lo, hi = float(zw.min()), float(zw.max())
        s = self.NB / ((hi - lo) * (1 + 1e-9) + 1e-300)
        idx = ((zw - lo) * s).astype(np.intp)
        cb = np.cumsum(np.bincount(idx, weights=dv, minlength=self.NB))
        if V <= 0:
            return lo, 0.0
        if V >= cb[-1]:
            return hi, V - cb[-1]
        j = int(np.searchsorted(cb, V))
        below = cb[j - 1] if j > 0 else 0.0
        sel = np.flatnonzero(idx == j)
        zb = zw[sel]
        o = np.argsort(zb)
        zs = zb[o]
        cum = below + np.cumsum(dv[sel][o])
        k = int(np.searchsorted(cum, V))
        if k > 0:
            c0, v0 = zs[k - 1], cum[k - 1]
        else:                                   # V falls between the last point below this bin and its first point
            m = idx < j
            c0, v0 = (float(zw[m].max()), below) if m.any() else (zs[0], 0.0)
        f = (V - v0) / max(cum[k] - v0, 1e-18)
        return c0 + f * (zs[k] - c0), 0.0


_GEOM = {}


def nx_geometry(nr=6):
    if nr not in _GEOM:
        _GEOM[nr] = GridPitcherGeometry.load(nr=nr)
    return _GEOM[nr]


def onset_tilt(P, pose, V):
    """Tilt at which the head at the lip reaches h_on (the stream starts) for fill V at this tip / yaw."""
    p = Pose(pose.tip, pose.yaw, 0.0, 0.0)

    def hmax(t):
        p.tilt = t
        c, _ = P.free_surface(p, V)
        return P.weir(p, c)["h_max"]
    lo, hi = 0.0, math.radians(150)
    for _ in range(30):
        m = 0.5 * (lo + hi)
        if hmax(m) < P.law.h_on:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- wanted tables
def _drvf():
    if FINAL not in sys.path:
        sys.path.insert(0, FINAL)
    import drvf                                      # noqa: E402  (the FS driver, scratch copy, not edited)
    return drvf


def fs_program(pattern):
    """The FS inlet program exactly as runs/FS_<pattern> used it (drvf main with --exp P2 --pulse 0.41 --lat exitmin
    --lag 0.08 --ztr and the pattern's flow phase)."""
    drvf = _drvf()
    with open(EXPS) as f:
        e = json.load(f)["P2"]
    params = dict(kappa_Q=2.0, cp=1.36, beta=3.0, nu=1e-3)
    for key in ("B_dep", "p_dep", "kappa_r"):
        params[key] = e["params"][key]
    sp = e["inlet"][pattern]
    return drvf.PhysF(pattern, sp["U"], sp["kappa_Q"], params, 0.41, FS_FLOW_PHASE[pattern], ramp=0.1, lat="exitmin",
                      lag=0.08, stop=False, ztr=True)


def _yaw_schedule(t, segs, T):
    """segs: list of (t0, t1, yaw) of the pouring strokes in time order.  Returns yaw(t) (unwrapped, shortest turns):
    the stroke's yaw inside a stroke; across a gap a cosine ease over the gap; between two adjacent strokes with
    different yaw a cosine ease over +-0.05 s around the boundary."""
    if not segs:
        return np.full_like(t, -math.pi / 2)
    ys = [segs[0][2]]
    for s in segs[1:]:
        d = (s[2] - ys[-1] + math.pi) % (2 * math.pi) - math.pi
        ys.append(ys[-1] + d)
    yaw = np.full_like(t, ys[0])
    for i, (a, b, _) in enumerate(segs):
        yaw[(t >= a) & (t < b)] = ys[i]
    yaw[t >= segs[-1][1]] = ys[-1]
    for i in range(1, len(segs)):
        a, b = segs[i - 1][1], segs[i][0]
        y0, y1 = ys[i - 1], ys[i]
        if abs(y1 - y0) < 1e-9:
            if b > a:
                yaw[(t >= a) & (t < b)] = y0
            continue
        if b - a < 0.1:                                  # direct boundary (or a very short gap)
            c = 0.5 * (a + b)
            a, b = c - 0.05, c + 0.05
        m = (t >= a) & (t < b)
        s = (t[m] - a) / (b - a)
        yaw[m] = y0 + (y1 - y0) * (0.5 - 0.5 * np.cos(math.pi * s))
    return yaw


def fs_wanted(pattern, dt=DT):
    """1 ms table of the FS inlet (see module docstring)."""
    prog = fs_program(pattern)
    from .v05_program import Program
    T = prog.duration
    n = int(math.floor(T / dt + 1e-9))
    t = np.arange(n) * dt
    keys = ("x", "y", "z", "Q", "Q_cmd", "Q_env", "U", "uhx", "uhy", "vhx", "vhy", "exit_h", "r", "d", "fac", "k", "ks",
            "active", "S_eff", "U_D", "ux_D", "uy_D", "d_D", "r1_D", "r2_D", "phi", "scan")
    W = {k: np.full(n, np.nan) for k in keys}
    kQ = prog.kQ
    for i, ti in enumerate(t):
        n0 = len(prog.log)
        inl = prog(ti)
        kph = prog.phase_index(ti)
        p = prog.phases[kph]
        tt = max(0.0, min(p["duration"], ti - prog.events[kph]))
        x, y = inl.x_hit
        W["x"][i], W["y"][i] = x, y
        W["z"][i] = prog.z_at(ti, kph, tt)
        vh = prog.hand_v(ti)
        W["vhx"][i], W["vhy"][i] = vh[0] * D, vh[1] * D
        W["k"][i] = kph
        fac = 1.0
        if prog.cls[kph] in ("low", "cut") and p.get("flow_modulation", 0.0) > 0:
            fac = 1 + p["flow_modulation"] * math.cos(2 * math.pi * p.get("flow_frequency", p["frequency"]) * tt
                                                       + p.get("flow_phase", 0.0))
        W["fac"][i] = fac
        E = float(np.interp(ti, prog.tg, prog.Ef))
        Ec = float(np.interp(ti, prog.tg, prog.Ecmd))
        W["Q_cmd"][i] = min(Ec * fac / kQ * D3, 32 * ML)
        W["scan"][i] = inl.scan_speed
        if len(prog.log) > n0:
            row = prog.log[-1]
            Q = row[5] * ML
            W["Q"][i] = Q
            W["Q_env"][i] = min(max(E / kQ * D3, 1 * ML), 32 * ML)
            W["U"][i] = row[6]
            W["uhx"][i], W["uhy"][i] = row[9], row[10]
            W["exit_h"][i] = row[13]
            W["r"][i], W["d"][i] = row[14] * 1e-3, row[15] * 1e-3
            W["ks"][i] = row[3]
            W["active"][i] = 1.0
            W["S_eff"][i] = inl.S_eff
            W["U_D"][i] = inl.U_perp
            W["ux_D"][i], W["uy_D"][i] = inl.u_in
            W["d_D"][i], W["r1_D"][i], W["r2_D"][i], W["phi"][i] = inl.d_jet, inl.r1, inl.r2, inl.phi
        else:
            W["Q"][i] = 0.0
            W["Q_env"][i] = 0.0
            W["active"][i] = 0.0
            W["S_eff"][i] = 0.0
            ks = kph if prog.cls[kph] in ("low", "cut") else (prog.prev_fl[kph] if prog.prev_fl[kph] is not None
                                                                else prog.next_fl[kph])
            W["ks"][i] = -1 if ks is None else ks
    strokes = []
    for k, c in enumerate(prog.cls):
        p = prog.phases[k]
        st = dict(k=k, n=k + 1, t0=prog.events[k], t1=prog.events[k + 1], cls=c, name=f"#{k + 1} {c}")
        if c in ("low", "cut"):
            bx, by = p["jet"]
            st["yaw"] = math.atan2(by, bx) if math.hypot(bx, by) > 0 else -math.pi / 2
            if p.get("flow_modulation", 0.0) > 0:
                st["pulse"] = dict(m=p["flow_modulation"], f=p.get("flow_frequency", p["frequency"]),
                                   phi=p.get("flow_phase", 0.0))
            if p.get("frequency", 0.0) > 0 and max(abs(p.get("amplitude", 0.0)), abs(p.get("amplitude_end", 0.0))) > 0:
                st["wiggle"] = dict(f=p["frequency"], a0=p["amplitude"], a1=p.get("amplitude_end", p["amplitude"]))
            st["z_design"] = prog.z[k]
        strokes.append(st)
    segs = [(s["t0"], s["t1"], s["yaw"]) for s in strokes if "yaw" in s]
    W["yaw"] = _yaw_schedule(t, segs, T)
    W["t"] = t
    W["Q_lag"] = W["Q"]               # the flow the solver received
    meta = dict(set="fs", pattern=pattern, T=T, dt=dt, strokes=strokes, kappa_Q=kQ,
                source="drvf.PhysF P2 pulse 0.41 exitmin lag 0.08 ztr flow_phase %s" % FS_FLOW_PHASE[pattern])
    return W, meta


def real_wanted(pattern, dt=DT):
    """1 ms table of the real-action wanted path and of its physical inlet (v05_adapter.PhysInlet)."""
    from . import calib as CB
    from .v05_adapter import FS as FSC, PhysInlet
    sc = CB.physical_script(pattern, preroll=0)
    PI = PhysInlet(sc, kappa_Q=FSC["kappa_Q"])
    T = PI.T
    n = int(math.floor(T / dt + 1e-9))
    t = np.arange(n) * dt
    keys = ("x", "y", "z", "Q", "Q_w", "U", "uhx", "uhy", "vhx", "vhy", "exit_h", "r", "d", "k", "active", "S_eff",
            "U_D", "ux_D", "uy_D", "d_D", "r1_D", "r2_D", "phi", "scan")
    W = {k: np.full(n, np.nan) for k in keys}
    for i, ti in enumerate(t):
        x, y, z, Qw, vx, vy = PI.foot(ti)
        st = PI.state(ti)
        W["x"][i], W["y"][i], W["z"][i], W["Q_w"][i] = x, y, z, Qw
        W["vhx"][i], W["vhy"][i] = vx * D, vy * D
        k = PI.move_index(ti)
        W["k"][i] = -1 if k is None else k
        W["scan"][i] = st["scan"]
        W["Q"][i] = st["Q"]          # solver inlet flow (lagged); the barista's intent is Q_w (see simulate)
        if st["active"]:
            I = st["inlet"]
            W["active"][i] = 1.0
            W["U"][i] = st["U"]
            W["uhx"][i], W["uhy"][i] = st["uh"]
            W["exit_h"][i] = PI.u_exit
            W["r"][i], W["d"][i] = st["r"], st["d"]
            W["S_eff"][i] = I.S_eff
            W["U_D"][i] = I.U_perp
            W["ux_D"][i], W["uy_D"][i] = I.u_in
            W["d_D"][i], W["r1_D"][i], W["r2_D"][i], W["phi"][i] = I.d_jet, I.r1, I.r2, I.phi
        else:
            W["active"][i] = 0.0
            W["S_eff"][i] = 0.0
    strokes = []
    for k, m in enumerate(PI.moves):
        pour = max(m.Q0, m.Q1) > 0
        c = m.name.split()[0]
        st = dict(k=k, n=k + 1, t0=float(PI.starts[k]), t1=float(PI.starts[k + 1]), cls=c if pour else "gap",
                  name=f"#{k + 1} {c}")
        if pour:
            st["yaw"] = PI.yaw
            if m.wfreq > 0 and max(abs(m.wobble), abs(m.wobble1)) > 0:
                st["wiggle"] = dict(f=m.wfreq, a0=m.wobble, a1=m.wobble1)
            st["z_design"] = [m.z0, m.z1]
            st["Q_design_mLs"] = [m.Q0 / ML, m.Q1 / ML]
        strokes.append(st)
    W["yaw"] = np.full(n, PI.yaw)
    W["t"] = t
    W["Q_lag"] = W["Q"]
    meta = dict(set="real", pattern=pattern, T=T, dt=dt, strokes=strokes, kappa_Q=FSC["kappa_Q"], yaw=PI.yaw,
                source="calib.physical_script(preroll=0) -> v3min_adapter.Inlet / v05_adapter.PhysInlet (u_exit 0.15, "
                       "tau 0.08)")
    return W, meta


# --------------------------------------------------------------------------- the barista
class _HandProxy:
    """Proxy pitcher seen by the barista (the barista's own tilt / height state is untouched):
      * adds the owner's open-loop tilt oscillation `osc_val` to every pose that reaches required_lift / step;
      * lift_smooth: the hand follows Pitcher.required_lift through a 9-step running median (drops single-step
        glitches) and a critically damped second-order filter (time constant lift_tau).  required_lift bisects a
        clearance test that is not monotone in the lift (a wall point beside the tilted cup can go from 'outside' to
        'over the rim' when raised) on a coarse wall sampling (40 x 36 points), so its answer jumps by 0.5-2.8 mm
        from one 1 ms step to the next; applied raw, that is a 1.5 m/s / 30 m/s^2 hand jerk (the kinematics clip)
        that sloshes the milk, changes g_eff and the flight time, and makes 20-75 mL/s arrival bursts.  The price is
        a short clearance violation (raw requirement above the applied lift), which is recorded ('lift_raw').
    Everything else is forwarded to the real pitcher."""

    def __init__(self, P, owner, lift_smooth=False, lift_tau=0.04, nmed=9):
        self._P, self._owner = P, owner
        self.lift_smooth, self.lift_tau, self.nmed = lift_smooth, lift_tau, nmed
        self.buf = []
        self.L, self.vL = None, 0.0
        self.lift_raw = 0.0

    def __getattr__(self, k):
        return getattr(self._P, k)

    def _p(self, pose):
        return Pose(pose.tip, pose.yaw, max(0.0, pose.tilt + self._owner.osc_val), pose.roll)

    def required_lift(self, pose, cup, iters=14):
        L = self._P.required_lift(self._p(pose), cup, iters)
        self.lift_raw = L
        if not self.lift_smooth:
            return L
        self.buf.append(L)
        if len(self.buf) > self.nmed:
            self.buf.pop(0)
        tgt = float(np.median(self.buf))
        if self.L is None:
            self.L = tgt
            return tgt
        dt, tau = self._owner.dt_step, self.lift_tau
        self.vL += dt * ((tgt - self.L) / (tau * tau) - 2.0 * self.vL / tau)
        self.L = max(0.0, self.L + dt * self.vL)
        return self.L

    def step(self, pose, dt, z_surface=0.0):
        return self._P.step(self._p(pose), dt, z_surface)


class TableBarista(Barista):
    """pitcher.Barista with its controller; intent() returns the wanted table (foot path [D], z [m], flow [m^3/s] from
    column qkey) and the pose carries the wanted yaw.
      ff_tol    None: the stock feed-forward cache (re-solve tilt_for_flow when Q_want or the fill changed by > 3 %);
                a number: the same with this tolerance (0.005 in 'main': with 3 % the cached tilt is held while the fill
                drops ~10 mL, and at dlnQ/dtilt ~ 0.37 /deg the flow sags by up to ~40 % before the next re-solve)
      lift_smooth see _HandProxy."""

    def __init__(self, pitcher, W, meta, coupling, qkey="Q", ff_tol=None, lift_smooth=False, **kw):
        self.W, self.meta, self.qkey = W, meta, qkey
        self.ff_tol = ff_tol
        self.osc_val = 0.0
        self.dt_step = DT
        self.real = pitcher
        self.dt_tab = meta["dt"]
        self.n_tab = len(W["t"])
        self.T_tab = self.n_tab * self.dt_tab
        self._x, self._y, self._z, self._q = (np.asarray(W[k], float) for k in ("x", "y", "z", qkey))
        self._q = np.nan_to_num(self._q)
        self._yaw = np.asarray(W["yaw"], float)
        self._k = np.asarray(W["k"], float)
        self.n_ff = 0
        mv = [Move(self.T_tab, (self._x[0], self._y[0]), (self._x[-1], self._y[-1]), z0=float(self._z[0]))]
        super().__init__(_HandProxy(pitcher, self, lift_smooth=lift_smooth), mv, coupling, yaw=float(self._yaw[0]), **kw)

    def _at(self, a, t):
        u = t / self.dt_tab
        i = int(math.floor(u))
        if i >= self.n_tab - 1:
            return float(a[-1])
        if i < 0:
            return float(a[0])
        f = u - i
        return float(a[i] * (1 - f) + a[i + 1] * f)

    def intent(self, t):
        if t >= self.T_tab:
            return (float(self._x[-1]), float(self._y[-1])), float(self._z[-1]), 0.0, "settle"
        k = int(self._k[min(int(t / self.dt_tab), self.n_tab - 1)])
        return (self._at(self._x, t), self._at(self._y, t)), self._at(self._z, t), max(self._at(self._q, t), 0.0), \
            f"#{k + 1}"

    def pose_at(self, t, dt=None):
        pose, Q, name = super().pose_at(t, dt)
        yaw = self._at(self._yaw, t)
        pose.yaw = yaw
        self.yaw = yaw
        return pose, Q, name

    def _tilt_ff(self, pose, Q_want):
        if self.ff_tol is None:
            q0, v0, _ = self._ff
            V = self.pitcher.state.V
            if not (q0 is not None and abs(Q_want - q0) <= 0.03 * q0 and abs(V - v0) <= 0.03 * v0):
                self.n_ff += 1
            return super()._tilt_ff(pose, Q_want)
        q0, v0, t0 = self._ff
        V = self.pitcher.state.V
        tol = self.ff_tol
        if q0 is not None and abs(Q_want - q0) <= tol * q0 and abs(V - v0) <= tol * v0:
            return t0
        lo, hi = max(0.0, self.tilt - math.radians(12)), min(math.radians(150), self.tilt + math.radians(12))
        t = self.pitcher.tilt_for_flow(pose, Q_want, V=V, lo=lo, hi=hi, iters=10)
        self._ff = (Q_want, V, t)
        self.n_ff += 1
        return t

    def step(self, dt):
        self.dt_step = dt
        rec, name = super().step(dt)
        rec["lift_raw"] = self.pitcher.lift_raw
        return rec, name


class PulseBarista(TableBarista):
    """Best-case pulsation: the stock controller tracks the lagged envelope (qkey='Q_env'); a 5 Hz tilt oscillation
    about the spout tip, Re(C e^{i w tt}) with tt the stroke clock, is added open loop.  C starts from the linear
    transfer function (weir sensitivity s = dlnQ_ss/dtilt, spout lag 1/(1 + i w tau), flight delay e^{-i w t_hit}) and
    is corrected once per cycle by a lock-in on the arriving flow: C <- C (c_want / c_meas)^0.5, |C| <= amp_max.
    The oscillation rectifies through the weir law (Q ~ h^2.5), so a slow tilt offset d0 corrects the cycle-mean flow:
    d0 <- d0 - 0.5 ln(mean_meas / mean_want) / s.  The applied (C, d0) follow their targets with a 0.1 s low-pass,
    so a cycle update never steps the tilt."""

    def __init__(self, pitcher, W, meta, coupling, amp_max=math.radians(12.0), gain=0.5, **kw):
        super().__init__(pitcher, W, meta, coupling, qkey="Q_env", **kw)
        self.real = pitcher
        self.pulsed = [s for s in meta["strokes"] if s.get("pulse")]
        self.amp_max, self.gain = amp_max, gain
        self.C = None
        self.C_app = None
        self.d0 = 0.0
        self.d0_app = 0.0
        self.sens = None
        self.cur = None
        self.buf_t, self.buf_q, self.buf_w = [], [], []
        self.cycle_end = None
        self.history = []
        self._qw = np.nan_to_num(np.asarray(W["Q"], float))

    def _stroke(self, t):
        for s in self.pulsed:
            if s["t0"] <= t < s["t1"]:
                return s
        return None

    def _init_C(self, s, t):
        P = self.real
        pose, _, _ = TableBarista.pose_at(self, t)
        pose.tilt = self.tilt
        V = P.state.V
        q = []
        for d in (-0.3, 0.3):
            p = Pose(pose.tip, pose.yaw, self.tilt + math.radians(d), 0.0)
            c, _ = P.free_surface(p, V)
            q.append(P.weir(p, c)["Q_ss"])
        p = Pose(pose.tip, pose.yaw, self.tilt, 0.0)
        c, _ = P.free_surface(p, V)
        q0 = P.weir(p, c)["Q_ss"]
        sens = (q[1] - q[0]) / math.radians(0.6) / max(q0, 1e-9)
        self.sens = sens if sens > 0 else 20.0
        w = 2 * math.pi * s["pulse"]["f"]
        cw = s["pulse"]["m"] * complex(math.cos(s["pulse"]["phi"]), math.sin(s["pulse"]["phi"]))
        th = 0.04
        if q0 <= 0 or sens <= 0:
            C = cw / abs(cw) * math.radians(3.0)
        else:
            C = cw * complex(1.0, w * P.law.tau) * complex(math.cos(w * th), math.sin(w * th)) / sens
        if abs(C) > self.amp_max:
            C *= self.amp_max / abs(C)
        return C

    def step(self, dt):
        t = self.real.state.t
        s = self._stroke(t)
        if s is not None:
            if s is not self.cur:
                if self.C is None:
                    self.C = self._init_C(s, t)
                    self.C_app = self.C
                self.cur = s
                self.buf_t, self.buf_q, self.buf_w = [], [], []
                self.cycle_end = None
            a = 1.0 - math.exp(-dt / 0.1)
            self.C_app += (self.C - self.C_app) * a
            self.d0_app += (self.d0 - self.d0_app) * a
            tt = t - s["t0"]
            w = 2 * math.pi * s["pulse"]["f"]
            env = min(1.0, tt / 0.05, (s["t1"] - t) / 0.05)
            self.osc_val = env * ((self.C_app * complex(math.cos(w * tt), math.sin(w * tt))).real + self.d0_app)
        else:
            self.osc_val = 0.0
        rec, name = TableBarista.step(self, dt)
        rec["osc"] = self.osc_val
        # lock-in on the arriving flow (stroke clock), one update per full cycle, after 0.25 s into the stroke
        if self.cur is not None:
            s = self.cur
            ta = rec["t"] - 0.5 * dt
            tt = ta - s["t0"]
            per = 1.0 / s["pulse"]["f"]
            if 0.25 <= tt < s["t1"] - s["t0"]:
                if self.cycle_end is None:
                    self.cycle_end = tt + per
                self.buf_t.append(tt)
                self.buf_q.append(rec["Q_hit"])
                self.buf_w.append(self._at(self._qw, ta))
                if tt >= self.cycle_end:
                    w = 2 * math.pi * s["pulse"]["f"]
                    tb = np.array(self.buf_t)
                    e = np.exp(-1j * w * tb)
                    qd, qw = np.array(self.buf_q), np.array(self.buf_w)
                    if qd.mean() > 0 and qw.mean() > 0:
                        cm = 2 * np.mean(qd * e) / qd.mean()
                        cw = 2 * np.mean(qw * e) / qw.mean()
                        if abs(cm) < 1e-3:
                            r = 1.5
                        else:
                            r = cw / cm
                            if abs(r) > 2.0:
                                r *= 2.0 / abs(r)
                            if abs(r) < 0.5:
                                r *= 0.5 / abs(r)
                        r = abs(r) ** self.gain * complex(math.cos(self.gain * np.angle(r)),
                                                          math.sin(self.gain * np.angle(r)))
                        self.C = self.C * r
                        if abs(self.C) > self.amp_max:
                            self.C *= self.amp_max / abs(self.C)
                        self.d0 -= 0.5 * math.log(max(qd.mean(), 1e-12) / qw.mean()) / self.sens
                        self.d0 = max(-math.radians(8.0), min(math.radians(8.0), self.d0))
                        self.history.append((float(ta), abs(cm), float(np.angle(cm)), abs(cw), float(np.angle(cw)),
                                             abs(self.C), float(np.angle(self.C)), float(qd.mean() / qw.mean()),
                                             float(self.d0)))
                    self.buf_t, self.buf_q, self.buf_w = [], [], []
                    self.cycle_end = tt + per
            if rec["t"] >= s["t1"] + 0.3:
                self.cur = None
        return rec, name


# --------------------------------------------------------------------------- simulation
REC_KEYS = ("t", "V", "Q", "Q_feed", "Q_hit", "Q_ss", "h_max", "h_bar", "c", "width", "flowing", "overflow", "tilt",
            "tip_x", "tip_y", "tip_z", "lift", "lift_raw", "yaw", "z_cmd", "off_x", "off_y", "Q_want", "osc",
            "j_on", "j_u0", "j_d0", "j_v0x", "j_v0y", "j_v0z", "j_ox", "j_oy", "j_oz", "j_thit", "j_xh", "j_yh", "j_U",
            "j_uhx", "j_uhy", "j_dhit", "j_r1", "j_r2", "j_coh", "j_We", "j_exh", "j_beta",
            "h_on", "h_xh", "h_yh", "h_U", "h_uhx", "h_uhy", "h_dhit", "h_r1", "h_r2", "h_phi", "h_coh", "h_speed",
            "vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z", "slope_x", "slope_y", "g_eff", "V_lip", "V_jet",
            "V_cup")


def _flatten(rec, B, A, i):
    A["t"][i] = rec["t"]
    for k in ("V", "Q", "Q_feed", "Q_hit", "Q_ss", "h_max", "h_bar", "c", "width", "overflow", "tilt", "lift", "Q_want",
              "g_eff", "V_lip", "V_jet", "V_cup"):
        A[k][i] = float(rec[k])
    A["flowing"][i] = float(rec["flowing"])
    A["tip_x"][i], A["tip_y"][i], A["tip_z"][i] = rec["tip"]
    A["yaw"][i] = B.yaw
    A["z_cmd"][i] = B.z_cmd if B.z_cmd is not None else np.nan
    A["off_x"][i], A["off_y"][i] = B.offset
    A["osc"][i] = rec.get("osc", 0.0)
    A["lift_raw"][i] = rec.get("lift_raw", rec["lift"])
    A["vel_x"][i], A["vel_y"][i], A["vel_z"][i] = rec["vel"]
    A["acc_x"][i], A["acc_y"][i], A["acc_z"][i] = rec["acc"]
    A["slope_x"][i], A["slope_y"][i] = rec["slope"]
    J = rec["jet"]
    if J is not None:
        A["j_on"][i] = 1.0
        A["j_u0"][i], A["j_d0"][i] = J["u0"], J["d0"]
        A["j_v0x"][i], A["j_v0y"][i], A["j_v0z"][i] = J["v0"]
        A["j_ox"][i], A["j_oy"][i], A["j_oz"][i] = J["origin"]
        A["j_thit"][i] = J["t_hit"]
        A["j_xh"][i], A["j_yh"][i] = J["x_hit"]
        A["j_U"][i] = J["U_perp"]
        A["j_uhx"][i], A["j_uhy"][i] = J["u_h"]
        A["j_dhit"][i], A["j_r1"][i], A["j_r2"][i] = J["d_hit"], J["r1"], J["r2"]
        A["j_coh"][i], A["j_We"][i] = float(J["coherent"]), J["We"]
        rel = np.asarray(J["v0"], float) - (np.asarray(rec["vel"], float) if B.pitcher.law.carry_velocity else 0.0)
        A["j_exh"][i] = math.hypot(rel[0], rel[1])
        A["j_beta"][i] = math.degrees(math.atan2(-rel[2], math.hypot(rel[0], rel[1])))
    else:
        A["j_on"][i] = 0.0
    H = rec["hit"]
    if H is not None:
        A["h_on"][i] = 1.0
        A["h_xh"][i], A["h_yh"][i] = H["x_hit"]
        A["h_U"][i] = H["U_perp"]
        A["h_uhx"][i], A["h_uhy"][i] = H["u_h"]
        A["h_dhit"][i], A["h_r1"][i], A["h_r2"][i], A["h_phi"][i] = H["d_hit"], H["r1"], H["r2"], H["phi"]
        A["h_coh"][i], A["h_speed"][i] = float(H["coherent"]), H["speed"]
    else:
        A["h_on"][i] = 0.0


def run_name(sset, pattern, variant="main", fill=350):
    return f"{sset}_{pattern}_{variant}_f{int(fill)}"


def wanted(sset, pattern):
    return fs_wanted(pattern) if sset == "fs" else real_wanted(pattern)


def simulate(sset, pattern, variant="main", fill=350, dt=DT, tail=0.6, nr=6, verbose=True):
    """Drive the NX pitcher with the wanted table; returns (W, meta, A, info)."""
    t0w = time.time()
    W, meta = wanted(sset, pattern)
    t_want = time.time() - t0w
    geom = nx_geometry(nr)
    P = FastPitcher(geom=geom, V0=fill * ML, law=OutflowLaw())
    pose0 = Pose((W["x"][0] * D, W["y"][0] * D, float(W["z"][0])), float(W["yaw"][0]), 0.0, 0.0)
    th_on = onset_tilt(P, pose0, fill * ML)
    tilt0 = max(0.0, th_on - math.radians(2.0))
    cp = Coupling()
    kw = dict(tilt0=tilt0) if variant == "stock" else dict(tilt0=tilt0, ff_tol=FF_TOL, lift_smooth=True)
    # FS: W['Q'] is the wanted flow at the surface.  Real actions: W['Q'] is PhysInlet's flow, already lagged by
    # tau = 0.08 s, so the barista's intent is the script flow Q_w (before the lag) and the pitcher's spout supplies
    # the lag once (feeding W['Q'] applied the 0.08 s lag twice).
    qk = "Q_w" if sset == "real" else "Q"
    if variant in ("main", "stock"):
        B = TableBarista(P, W, meta, cp, qkey=qk, **kw)
    elif variant == "cmd":
        assert sset == "fs"
        B = TableBarista(P, W, meta, cp, qkey="Q_cmd", **kw)
    elif variant == "pulse":
        assert sset == "fs"
        B = PulseBarista(P, W, meta, cp, **kw)
    else:
        raise ValueError(variant)
    n = int(round((len(W["t"]) * meta["dt"] + tail) / dt))
    A = {k: np.full(n, np.nan) for k in REC_KEYS}
    t1w = time.time()
    for i in range(n):
        rec, _ = B.step(dt)
        _flatten(rec, B, A, i)
        if verbose and i % 2000 == 0:
            print(f"  {run_name(sset, pattern, variant, fill)} t={rec['t']:.2f}/{n * dt:.2f} V={rec['V'] / ML:.0f} mL "
                  f"Q_hit={rec['Q_hit'] / ML:.1f} tilt={math.degrees(rec['tilt']):.1f} [{time.time() - t1w:.0f}s]",
                  flush=True)
    info = dict(run=run_name(sset, pattern, variant, fill), set=sset, pattern=pattern, variant=variant, fill_mL=fill,
                dt=dt, nr=nr, steps=n, wall_s=time.time() - t0w, n_ff_solves=B.n_ff,
                ff_tol=(0.03 if variant == "stock" else FF_TOL), lift_smooth=(variant != "stock"), wall_want_s=t_want, wall_sim_s=time.time() - t1w,
                onset_tilt_deg=math.degrees(th_on), tilt0_deg=math.degrees(tilt0),
                law={k: getattr(P.law, k) for k in ("C_d", "h_k", "h_on", "h_off", "tau", "tau_stop", "zeta", "slosh",
                                                    "carry_velocity")},
                cup={k: getattr(B.cup, k) for k in ("r_in", "r_out", "height", "rim_z", "tilt", "margin")},
                barista=dict(max_rate_deg_s=math.degrees(B.max_rate), swing_rate_deg_s=math.degrees(B.swing_rate),
                             tau_ff=B.tau_ff, tau_see=B.tau_see, z_rate=B.z_rate, tau_z=B.tau_z))
    if variant == "pulse":
        info["lockin_history"] = B.history
        info["C_final_deg"] = [math.degrees(abs(B.C)), math.degrees(np.angle(B.C))] if B.C is not None else None
    return W, meta, A, info


def save_run(W, meta, A, info, out=OUT):
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    name = info["run"]
    np.savez_compressed(os.path.join(out, "runs", name + ".npz"), **{"w_" + k: v for k, v in W.items()},
                        **{"p_" + k: v for k, v in A.items()})
    with open(os.path.join(out, "runs", name + ".json"), "w") as f:
        json.dump(dict(meta=meta, info=info), f, indent=1, default=float)


def load_run(name, out=OUT):
    z = np.load(os.path.join(out, "runs", name + ".npz"))
    W = {k[2:]: z[k] for k in z.files if k.startswith("w_")}
    A = {k[2:]: z[k] for k in z.files if k.startswith("p_")}
    with open(os.path.join(out, "runs", name + ".json")) as f:
        j = json.load(f)
    return W, j["meta"], A, j["info"]


# --------------------------------------------------------------------------- analysis helpers
def _wint(W, key, t, dt):
    """Wanted column interpolated at times t (0 beyond the table for flows, last value otherwise)."""
    tw = W["t"]
    a = np.asarray(W[key], float)
    v = np.interp(t, tw, np.nan_to_num(a) if key.startswith("Q") or key in ("active", "S_eff") else a)
    if key.startswith("Q") or key in ("active", "S_eff"):
        v = np.where(t > tw[-1] + dt, 0.0, v)
    return v


def lockin(t, q, f, t_ref):
    """Fundamental of q at frequency f on whole cycles: (m1 = |c|/mean, phase of c) with c = 2 <q e^{-i w (t - t_ref)}>."""
    if len(t) < 10 or np.nanmean(q) <= 0:
        return float("nan"), float("nan")
    w = 2 * math.pi * f
    c = 2 * np.mean(q * np.exp(-1j * w * (t - t_ref)))
    return float(abs(c) / np.mean(q)), float(np.angle(c))


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def best_delay(td, d, tw, w, max_lag=0.3, step=0.001):
    """Delay L >= 0 that minimises sum |d(t) - w(t - L)| (w sampled on tw)."""
    best, bl = None, 0.0
    for L in np.arange(0.0, max_lag + 1e-12, step):
        e = np.abs(d - np.interp(td - L, tw, w, left=0.0, right=0.0)).sum()
        if best is None or e < best:
            best, bl = e, L
    return float(bl)


def _fw(x, w):
    m = np.isfinite(x) & (w > 0)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.any() else float("nan")


def _rng(x):
    x = x[np.isfinite(x)]
    return [float(x.min()), float(x.max())] if len(x) else None


def cls_rel(rel, tol=TOL_REL):
    if rel is None or not np.isfinite(rel):
        return "n/a"
    return "DELIVERED" if abs(rel) <= tol else f"DEVIATES {rel * 100:+.0f}%"


def analyze(name, out=OUT, figure=True):
    W, meta, A, info = load_run(name, out)
    dt = info["dt"]
    t = A["t"] - 0.5 * dt                         # step mid-times (the arrival window of step i is [t-dt, t])
    Tt = meta["T"]
    sset = meta["set"]
    hit = A["h_on"] > 0
    Qd = A["Q_hit"]                              # delivered (arriving) flow [m^3/s]
    Qw = _wint(W, "Q", t, dt)                     # wanted at the surface (FS) / solver inlet (real: lagged)
    Qraw = _wint(W, "Q_w", t, dt) if sset == "real" else _wint(W, "Q_cmd", t, dt)
    Qint = _wint(W, "Q_cmd", t, dt) if (info["variant"] == "cmd") else (
        _wint(W, "Q_env", t, dt) if info["variant"] == "pulse" else (Qraw if sset == "real" else Qw))
    xw, yw, zw = _wint(W, "x", t, dt) * D, _wint(W, "y", t, dt) * D, _wint(W, "z", t, dt)
    Uw = _wint(W, "U", t, dt)
    uhw = np.hypot(_wint(W, "uhx", t, dt), _wint(W, "uhy", t, dt))
    exw = _wint(W, "exit_h", t, dt)
    rw = _wint(W, "r", t, dt)
    # delivered arrival quantities
    Ud = np.where(hit, A["h_U"], np.nan)
    uhd = np.where(hit, np.hypot(A["h_uhx"], A["h_uhy"]), np.nan)
    rd = np.where(hit & (Ud > 0), np.sqrt(np.maximum(Qd, 0) / (math.pi * np.maximum(Ud, 1e-9))), np.nan)
    rphys = np.where(hit, np.sqrt(A["h_r1"] * A["h_r2"]), np.nan)
    ex = (A["h_xh"] - xw) * 1e3
    ey = (A["h_yh"] - yw) * 1e3
    efoot = np.where(hit, np.hypot(ex, ey), np.nan)
    # release quantities, shifted to the arrival axis by the flight time
    jon = A["j_on"] > 0
    t_arr_rel = t + np.where(jon, A["j_thit"], 0.0)
    u0 = np.where(jon, A["j_u0"], np.nan)
    exh = np.where(jon, A["j_exh"], np.nan)
    beta = np.where(jon, A["j_beta"], np.nan)
    drop = np.where(jon, A["j_oz"], np.nan)
    U_formula_drop = np.sqrt(V0_EXIT ** 2 + 2 * G * np.maximum(drop, 0.0))
    z_tip = A["tip_z"]
    lift = A["lift"]
    tilt = np.degrees(A["tilt"])
    tilt_rate = np.gradient(tilt, dt)
    yaw = np.degrees(A["yaw"])
    yaw_rate = np.gradient(yaw, dt)

    # global flow delay (delivered vs wanted at the surface)
    m_in = t <= Tt + 0.3
    L_Q = best_delay(t[m_in], Qd[m_in], t[m_in], Qw[m_in], max_lag=0.5)
    Qw_L = np.interp(t - L_Q, t, Qw, left=0.0, right=0.0)
    L_int = best_delay(t[m_in], Qd[m_in], t[m_in], Qint[m_in], max_lag=0.5)

    strokes = []
    for s in meta["strokes"]:
        if "yaw" not in s:
            continue
        a, b = s["t0"], s["t1"]
        m = (t >= a) & (t < b)
        if not m.any():
            continue
        o = dict(n=s["n"], name=s["name"], cls=s["cls"], t0=a, t1=b, dur=b - a)
        Vw_ = float(np.sum(Qw[m]) * dt / ML)
        Vd_ = float(np.sum(Qd[m]) * dt / ML)
        o["V_w_mL"], o["V_d_mL"] = Vw_, Vd_
        o["Q_w_mean"] = float(np.mean(Qw[m]) / ML)
        o["Q_d_mean"] = float(np.mean(Qd[m]) / ML)
        o["Q_ratio"] = Vd_ / Vw_ if Vw_ > 0 else float("nan")
        o["Q_err_rel"] = float(np.sum(np.abs(Qd[m] - Qw[m])) / max(np.sum(Qw[m]), 1e-30))
        o["Q_err_rel_delayed"] = float(np.sum(np.abs(Qd[m] - Qw_L[m])) / max(np.sum(Qw_L[m]), 1e-30))
        o["Q_maxerr_mLs"] = float(np.max(np.abs(Qd[m] - Qw[m])) / ML)
        o["Q_maxerr_delayed_mLs"] = float(np.max(np.abs(Qd[m] - Qw_L[m])) / ML)
        if sset == "real":
            o["Qraw_V_mL"] = float(np.sum(Qraw[m]) * dt / ML)
            o["Qraw_err_rel"] = float(np.sum(np.abs(Qd[m] - Qraw[m])) / max(np.sum(Qraw[m]), 1e-30))
        # start delay (first time Q_w >= 1 mL/s within the stroke window -> first time Q_hit >= 1 mL/s)
        mw = m & (Qw >= 1 * ML)
        if mw.any():
            tw0 = t[mw][0]
            md = (t >= tw0) & (Qd >= 1 * ML) & (t < b + 0.5)
            o["start_delay_s"] = float(t[md][0] - tw0) if md.any() else float("inf")
            first = t[md][0] if md.any() else b
            mb = m & (t > first) & (Qw >= 1 * ML) & (Qd < 0.5 * ML)
            o["break_s"] = float(mb.sum() * dt)
        w8 = np.where(m, Qd, 0.0)
        wW = np.where(m, Qw, 0.0)
        o["U_w"], o["U_d"] = _fw(Uw, wW), _fw(Ud, w8)
        o["U_rel"] = o["U_d"] / o["U_w"] - 1 if o["U_w"] > 0 else float("nan")
        mr = (t_arr_rel >= a) & (t_arr_rel < b) & jon
        wr = np.where(mr, A["Q"], 0.0)
        o["U_formula_at_drop"] = _fw(U_formula_drop, wr)
        o["drop_mm"] = _fw(drop * 1e3, wr)
        o["u0"] = _fw(u0, wr)
        o["u0_rel"] = o["u0"] / V0_EXIT - 1
        o["u0_range"] = _rng(u0[mr])
        o["exit_h_w"], o["exit_h_d"] = _fw(exw, wW), _fw(exh, wr)
        o["exit_h_rel"] = o["exit_h_d"] / o["exit_h_w"] - 1 if o["exit_h_w"] > 0 else float("nan")
        o["exit_h_range"] = _rng(exh[mr])
        o["beta_deg"] = _fw(beta, wr)
        o["uh_w"], o["uh_d"] = _fw(uhw, wW), _fw(uhd, w8)
        o["uh_rel"] = o["uh_d"] / o["uh_w"] - 1 if o["uh_w"] > 0 else float("nan")
        o["tan_w"] = _fw(uhw / Uw, wW)
        o["tan_d"] = _fw(uhd / Ud, w8)
        o["tan_rel"] = o["tan_d"] / o["tan_w"] - 1 if o["tan_w"] > 0 else float("nan")
        o["r_w_mm"], o["r_d_mm"] = _fw(rw * 1e3, wW), _fw(rd * 1e3, w8)
        o["r_rel"] = o["r_d_mm"] / o["r_w_mm"] - 1 if o["r_w_mm"] > 0 else float("nan")
        o["r_phys_mm"] = _fw(rphys * 1e3, w8)
        # foot position
        mf = m & hit & (Qd > 0)
        if mf.any():
            wf = Qd[mf]
            o["foot_rms_mm"] = float(np.sqrt(np.sum(wf * efoot[mf] ** 2) / wf.sum()))
            o["foot_max_mm"] = float(np.nanmax(efoot[mf]))
            best = None
            for L in np.arange(0.0, 0.2001, 0.002):
                xs = np.interp(t[mf] - L, W["t"], W["x"]) * D
                ys = np.interp(t[mf] - L, W["t"], W["y"]) * D
                e = np.sqrt(np.sum(wf * ((A["h_xh"][mf] - xs) ** 2 + (A["h_yh"][mf] - ys) ** 2)) / wf.sum()) * 1e3
                if best is None or e < best[0]:
                    best = (e, L)
            o["foot_rms_delayed_mm"], o["foot_delay_s"] = float(best[0]), float(best[1])
            o["foot_mean_dx_mm"] = float(np.sum(wf * ex[mf]) / wf.sum())
            o["foot_mean_dy_mm"] = float(np.sum(wf * ey[mf]) / wf.sum())
        # height
        o["z_w_cm"] = _fw(zw * 100, wW)
        o["z_tip_cm"] = _fw(z_tip * 100, wW)
        o["z_err_mm"] = _fw(np.abs(z_tip - zw) * 1e3, wW)
        o["lift_max_mm"] = float(np.nanmax(lift[m]) * 1e3)
        o["lift_fw_mm"] = _fw(lift * 1e3, wW)
        mp_ = m & (Qw >= 1 * ML)
        o["lift_p90_mm"] = float(np.percentile(lift[mp_], 90) * 1e3) if mp_.any() else 0.0
        o["lift_raw_fw_mm"] = _fw(A["lift_raw"] * 1e3, wW)
        o["lift_time_s"] = float(np.sum(lift[m] > 1e-3) * dt)
        o["z_reachable_min_cm"] = float(np.nanmax((zw + lift)[m & (lift > 1e-3)]) * 100) if (m & (lift > 1e-3)).any() \
            else None
        o["tilt_range_deg"] = _rng(tilt[m])
        o["tilt_fw_deg"] = _fw(tilt, wW)
        s_tip = -(A["tip_x"] * np.cos(A["yaw"]) + A["tip_y"] * np.sin(A["yaw"]))
        o["tip_s_mm"] = _fw(s_tip * 1e3, wW)
        s_foot = -(xw * np.cos(A["yaw"]) + yw * np.sin(A["yaw"]))
        o["foot_s_mm"] = _fw(s_foot * 1e3, wW)
        o["tilt_rate_max"] = float(np.nanmax(np.abs(tilt_rate[m])))
        o["coherent_frac"] = float(np.mean(A["h_coh"][m & hit])) if (m & hit).any() else None
        # wiggle amplitude (lateral lock-in on x at the wiggle frequency)
        if s.get("wiggle"):
            f = s["wiggle"]["f"]
            ma = m & (t >= a + 0.3) & (t < b - 0.3)
            if ma.sum() > 50:
                nper = int((b - a - 0.6) * f)
                mm = ma & (t < a + 0.3 + nper / f)
                xd = np.where(hit, A["h_xh"], np.nan)[mm]
                ok = np.isfinite(xd)
                cw_ = 2 * np.mean(xw[mm] * np.exp(-2j * math.pi * f * (t[mm] - a)))
                cd_ = 2 * np.mean(np.where(ok, xd, xw[mm]) * np.exp(-2j * math.pi * f * (t[mm] - a)))
                o["wiggle_amp_w_mm"] = float(abs(cw_) * 1e3)
                o["wiggle_amp_d_mm"] = float(abs(cd_) * 1e3)
                o["wiggle_phase_lag_deg"] = float(_wrap_deg(math.degrees(np.angle(cw_) - np.angle(cd_))))
        # pulsation
        if s.get("pulse"):
            f = s["pulse"]["f"]
            a0 = a + 0.4
            nper = int(math.floor((b - 0.05 - a0) * f))
            if nper >= 2:
                mp = (t >= a0) & (t < a0 + nper / f)
                mw_, pw_ = lockin(t[mp], Qw[mp], f, a)
                md_, pd_ = lockin(t[mp], Qd[mp], f, a)
                o["pulse_m_w"], o["pulse_m_d"] = mw_, md_
                o["pulse_phase_w_deg"], o["pulse_phase_d_deg"] = math.degrees(pw_), math.degrees(pd_)
                o["pulse_dphase_deg"] = float(_wrap_deg(math.degrees(pd_ - pw_)))
                o["pulse_dphase_ms"] = float(-o["pulse_dphase_deg"] / 360.0 / f * 1e3)
                o["pulse_m_rel"] = md_ / mw_ - 1 if mw_ > 0 else float("nan")
                qq = Qd[mp]
                o["pulse_min_rel"] = float(qq.min() / qq.mean()) if qq.mean() > 0 else None
                o["pulse_noflow_frac"] = float(np.mean(qq < 0.05 * qq.mean())) if qq.mean() > 0 else None
                o["pulse_tilt_amp_deg"] = float(0.5 * (np.percentile(tilt[mp], 98) - np.percentile(tilt[mp], 2)))
                osc = np.degrees(A["osc"][mp])
                o["pulse_osc_amp_deg"] = float(0.5 * (np.percentile(osc, 98) - np.percentile(osc, 2)))
                o["pulse_tilt_rate_max"] = float(np.nanmax(np.abs(tilt_rate[mp])))
        # classes
        c = {}
        rq = o["Q_err_rel"]
        late = o.get("start_delay_s") is not None and o["start_delay_s"] >= 0.8 * o["dur"]
        c["Q"] = "DELIVERED" if rq <= TOL_REL else (
            f"DEVIATES late start {o['start_delay_s'] * 1e3:.0f} ms > stroke {o['dur'] * 1e3:.0f} ms, V {o['Q_ratio']:.2f}x"
            if late else ("IMPOSSIBLE no stream" if o["V_d_mL"] < 0.05 * max(o["V_w_mL"], 1e-9) else
                          f"DEVIATES |dQ| {rq * 100:.0f}% (delay-corr. {o['Q_err_rel_delayed'] * 100:.0f}%), "
                          f"V {o['Q_ratio']:.2f}x"))
        if s.get("pulse") and "pulse_m_d" in o:
            okm = abs(o["pulse_m_rel"]) <= TOL_REL
            okp = abs(o["pulse_dphase_deg"]) <= TOL_DEG
            c["pulse"] = "DELIVERED" if okm and okp else \
                f"DEVIATES m {o['pulse_m_d']:.2f}/{o['pulse_m_w']:.2f}, phase {o['pulse_dphase_deg']:+.0f} deg"
        c["U_perp"] = cls_rel(o["U_rel"])
        c["u0"] = cls_rel(o["u0_rel"])
        c["exit_h"] = cls_rel(o["exit_h_rel"])
        c["uh"] = cls_rel(o["uh_rel"])
        c["tan"] = cls_rel(o["tan_rel"])
        c["r"] = cls_rel(o["r_rel"])
        if "foot_rms_mm" in o:
            c["foot"] = "DELIVERED" if o["foot_rms_mm"] <= TOL_MM else \
                f"DEVIATES rms {o['foot_rms_mm']:.1f} mm ({o['foot_rms_delayed_mm']:.1f} after {o['foot_delay_s'] * 1e3:.0f} ms)"
        else:
            c["foot"] = "n/a"
        if o["lift_fw_mm"] > TOL_MM:
            c["z"] = f"IMPOSSIBLE lift {o['lift_fw_mm']:.1f} mm (max {o['lift_max_mm']:.1f})"
        else:
            c["z"] = "DELIVERED" if o["z_err_mm"] <= TOL_MM else f"DEVIATES {o['z_err_mm']:.1f} mm"
        o["class"] = c
        strokes.append(o)

    # gaps / bridges: leak delivered vs wanted
    gaps = []
    for s in meta["strokes"]:
        if "yaw" in s:
            continue
        m = (t >= s["t0"]) & (t < s["t1"])
        gaps.append(dict(n=s["n"], name=s["name"], t0=s["t0"], t1=s["t1"], V_w_mL=float(np.sum(Qw[m]) * dt / ML),
                         V_d_mL=float(np.sum(Qd[m]) * dt / ML),
                         zero_w_s=float(np.sum(Qw[m] < 1 * ML) * dt), zero_d_s=float(np.sum(Qd[m] < 1 * ML) * dt),
                         lift_max_mm=float(np.nanmax(lift[m]) * 1e3) if m.any() else 0.0))
    mT = t <= Tt
    tot = dict(V_w_mL=float(np.sum(Qw) * dt / ML), V_d_mL=float(np.sum(Qd) * dt / ML),
               V_d_in_T_mL=float(np.sum(Qd[mT]) * dt / ML), V_after_T_mL=float(np.sum(Qd[~mT]) * dt / ML),
               milk_used_mL=float((info["fill_mL"] * ML - A["V"][-1]) / ML), fill_mL=info["fill_mL"],
               V_end_mL=float(A["V"][-1] / ML), overflow_mL=float(np.sum(A["overflow"]) / ML),
               V_cup_mL=float(A["V_cup"][-1] / ML), Q_delay_s=L_Q, Q_delay_vs_intent_s=L_int,
               Q_err_rel=float(np.sum(np.abs(Qd - Qw)[m_in]) / max(np.sum(Qw[m_in]), 1e-30)),
               Q_err_rel_delayed=float(np.sum(np.abs(Qd - Qw_L)[m_in]) / max(np.sum(Qw_L[m_in]), 1e-30)),
               tilt_range_deg=_rng(tilt[mT]), tilt_rate_max=float(np.nanmax(np.abs(tilt_rate[mT]))),
               yaw_rate_max=float(np.nanmax(np.abs(yaw_rate[mT]))), yaw_range_deg=_rng(yaw[mT]),
               lift_max_mm=float(np.nanmax(lift[mT]) * 1e3), lift_time_s=float(np.sum(lift[mT] > 1e-3) * dt),
               tip_speed_max=float(np.nanmax(np.hypot(A["vel_x"], np.hypot(A["vel_y"], A["vel_z"]))[mT])),
               acc_max=float(np.nanmax(np.hypot(A["acc_x"], np.hypot(A["acc_y"], A["acc_z"]))[mT])),
               slope_max_deg=float(np.degrees(np.arctan(np.nanmax(np.hypot(A["slope_x"], A["slope_y"])[mT])))),
               incoherent_frac=float(np.mean(A["h_coh"][hit & mT] < 0.5)) if (hit & mT).any() else None,
               Q_hit_max_mLs=float(np.nanmax(Qd) / ML), Q_w_max_mLs=float(np.nanmax(Qw) / ML))
    res = dict(run=name, info=info, total=tot, strokes=strokes, gaps=gaps)
    with open(os.path.join(out, "runs", name + "_metrics.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    save_tables(name, W, meta, A, info, out)
    if figure:
        plot_run(name, W, meta, A, info, res, out)
    return res


# --------------------------------------------------------------------------- solver tables
def _continuity(Q, U, ux, uy):
    """FS / PhysInlet continuity footprint from SI Q [m^3/s], U [m/s], u_h [m/s] -> (r1, r2, d, phi) [m]."""
    v = np.hypot(ux, uy)
    spd = np.hypot(U, v)
    aspect = spd / np.maximum(U, 1e-12)
    r = np.sqrt(Q / (math.pi * np.maximum(U, 1e-12)))
    d = 2 * np.sqrt(Q / (math.pi * np.maximum(spd, 1e-12)))
    return r * np.sqrt(aspect), r / np.sqrt(aspect), d, np.arctan2(uy, ux)


def save_tables(name, W, meta, A, info, out=OUT, qmin=1.0):
    """Both inlets on one 1 ms grid t_k = k dt: delivered record k covers [t_k, t_k + dt); wanted sampled at t_k.
    'pitcher_active' uses the FS convention: a stream below qmin mL/s is no inlet."""
    dt = info["dt"]
    n = len(A["t"])
    tk = np.arange(n) * dt
    T = {"t": tk}
    kQ = meta["kappa_Q"]
    Q = np.nan_to_num(A["Q_hit"])
    act = (A["h_on"] > 0) & (Q >= qmin * ML)
    U = np.nan_to_num(A["h_U"])
    ux, uy = np.nan_to_num(A["h_uhx"]), np.nan_to_num(A["h_uhy"])
    r1, r2, d, phi = _continuity(Q, U, ux, uy)
    scan = np.hypot(np.gradient(np.nan_to_num(A["h_xh"]), dt), np.gradient(np.nan_to_num(A["h_yh"]), dt)) / D
    scan_w = np.interp(tk, W["t"], np.nan_to_num(W["scan"]), right=0.0)
    T.update(pitcher_active=act.astype(np.int8), pitcher_x=np.nan_to_num(A["h_xh"]) / D,
             pitcher_y=np.nan_to_num(A["h_yh"]) / D, pitcher_Q_mLs=Q / ML,
             pitcher_S_eff=np.where(act, kQ * Q / D3, 0.0), pitcher_U_perp=np.where(act, U / D, 0.0),
             pitcher_ux=np.where(act, ux / D, 0.0), pitcher_uy=np.where(act, uy / D, 0.0),
             pitcher_d_jet=np.where(act, d / D, 0.0), pitcher_r1=np.where(act, r1 / D, 0.0),
             pitcher_r2=np.where(act, r2 / D, 0.0), pitcher_phi=np.where(act, phi, 0.0),
             pitcher_r1_phys=np.nan_to_num(A["h_r1"]) / D, pitcher_r2_phys=np.nan_to_num(A["h_r2"]) / D,
             pitcher_phi_phys=np.nan_to_num(A["h_phi"]), pitcher_z=A["tip_z"],
             pitcher_scan=np.maximum(np.minimum(scan, 5.0), scan_w))
    def wcol(k, fill=0.0):
        a = np.asarray(W[k], float)
        return np.where(tk <= W["t"][-1], np.interp(tk, W["t"], np.nan_to_num(a, nan=fill)), fill)
    wa = wcol("active") > 0.5
    T.update(wanted_active=wa.astype(np.int8), wanted_x=wcol("x"), wanted_y=wcol("y"), wanted_Q_mLs=wcol("Q") / ML,
             wanted_S_eff=np.where(wa, wcol("S_eff"), 0.0), wanted_U_perp=np.where(wa, wcol("U_D"), 0.0),
             wanted_ux=np.where(wa, wcol("ux_D"), 0.0), wanted_uy=np.where(wa, wcol("uy_D"), 0.0),
             wanted_d_jet=np.where(wa, wcol("d_D"), 0.0), wanted_r1=np.where(wa, wcol("r1_D"), 0.0),
             wanted_r2=np.where(wa, wcol("r2_D"), 0.0), wanted_phi=np.where(wa, wcol("phi"), 0.0),
             wanted_z=wcol("z"), wanted_scan=scan_w)
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    np.savez_compressed(os.path.join(out, "tables", name + ".npz"), **T,
                        meta=json.dumps(dict(run=name, set=meta["set"], pattern=meta["pattern"], dt=dt, kappa_Q=kQ,
                                             qmin_mLs=qmin, T=meta["T"],
                                             units="positions D, speeds D/s, S_eff kappa_Q Q / D^3, Q mL/s, z m")))


class TableInlet:
    """Solver inlet from a table npz (kind 'pitcher' or 'wanted'); call(t) -> solver.Inlet (sample-and-hold on the
    1 ms grid).  state(t) mimics v05_adapter.PhysInlet.state for V05Model."""

    def __init__(self, path, kind="pitcher"):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.kind = kind
        self.t = z["t"]
        self.dt = float(self.meta["dt"])
        self.T = float(self.t[-1] + self.dt)
        self.c = {k: z[f"{kind}_{k}"] for k in ("active", "x", "y", "S_eff", "U_perp", "ux", "uy", "d_jet", "r1", "r2",
                                                 "phi", "scan", "Q_mLs", "z")}
        self.u_exit, self.tau = float("nan"), float("nan")     # V05Model.run reports these; not defined for a table

    def _i(self, t):
        return min(max(int(math.floor(t / self.dt + 1e-9)), 0), len(self.t) - 1)

    def __call__(self, t):
        i = self._i(t)
        c = self.c
        x = (float(c["x"][i]), float(c["y"][i]))
        if t >= self.T or not c["active"][i]:
            return Inlet(active=False, x_hit=x, S_eff=0.0, scan_speed=float(c["scan"][i]))
        return Inlet(active=True, x_hit=x, S_eff=float(c["S_eff"][i]), u_in=(float(c["ux"][i]), float(c["uy"][i])),
                     U_perp=float(c["U_perp"][i]), d_jet=float(c["d_jet"][i]), r1=float(c["r1"][i]),
                     r2=float(c["r2"][i]), phi=float(c["phi"][i]), scan_speed=float(c["scan"][i]))

    def state(self, t):
        I = self(t)
        i = self._i(t)
        Q = float(self.c["Q_mLs"][i]) * ML if I.active else 0.0
        out = dict(t=t, x=I.x_hit[0], y=I.x_hit[1], z=float(self.c["z"][i]), Q_w=Q, Q=Q, vh=(0.0, 0.0),
                   active=I.active, off_cup=False, inlet=I, scan=I.scan_speed)
        if I.active:
            out.update(U=I.U_perp * D, uh=(I.u_in[0] * D, I.u_in[1] * D), r=math.sqrt(I.r1 * I.r2) * D,
                       d=I.d_jet * D, r1=I.r1 * D, r2=I.r2 * D)
        return out


def TableProgram(path, kind="pitcher", closure=None):
    """Drop-in for run.py's v05_program (as drvf does): `R.v05_program = lambda name: TableProgram(...)`.  The closure
    defaults to FS (v05_adapter.FS)."""
    from .v05_adapter import FS as FSC
    from .v05_program import Program, load_control
    inl = TableInlet(path, kind)
    cfg = copy.deepcopy(load_control(inl.meta["pattern"]))
    cfg["parameters"].update(dict(FSC if closure is None else closure))

    class _Prog(Program):
        def __call__(self, t):
            return inl(t)

        def phase_label(self, t):
            return f"{kind} table"
    p = _Prog(cfg)
    p.duration = inl.T
    p.table = inl
    return p


def TableV05Model(path, kind="pitcher", **kw):
    """v05_adapter.V05Model whose inlet is the table (model time 0 = table time 0)."""
    from .v05_adapter import V05Model

    class _M(V05Model):
        def inlet(self, script, params=None, cup_radius=0.49, t_end=None):
            return TableInlet(path, kind), 0.0
    return _M(**kw)


# --------------------------------------------------------------------------- figures
C_W, C_D, C_3, C_4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_run(name, W, meta, A, info, res, out=OUT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dt = info["dt"]
    t = A["t"] - 0.5 * dt
    hit = A["h_on"] > 0
    sset = meta["set"]
    Qw = _wint(W, "Q", t, dt)
    fig, ax = plt.subplots(6, 1, figsize=(12, 13.5), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    lab_w = "FS inlet (solver)" if sset == "fs" else "PhysInlet (solver, lagged)"
    a = ax[0]
    a.plot(t, Qw / ML, color=C_W, lw=1.2, label=lab_w)
    if sset == "real":
        a.plot(t, _wint(W, "Q_w", t, dt) / ML, color=C_3, lw=1.0, ls="--", label="wanted flow (script)")
    elif info["variant"] == "cmd":
        a.plot(t, _wint(W, "Q_cmd", t, dt) / ML, color=C_3, lw=1.0, ls="--", label="barista intent = FS command")
    elif info["variant"] == "pulse":
        a.plot(t, _wint(W, "Q_env", t, dt) / ML, color=C_3, lw=1.0, ls="--", label="barista intent = envelope")
    a.plot(t, A["Q_hit"] / ML, color=C_D, lw=1.0, label="pitcher, arriving")
    a.set_ylabel("Q [mL/s]")
    a = ax[1]
    a.plot(t, _wint(W, "U", t, dt), color=C_W, lw=1.2, label=lab_w)
    a.plot(t, np.where(hit, A["h_U"], np.nan), color=C_D, lw=1.0, label="pitcher")
    a.set_ylabel("U_perp [m/s]")
    a = ax[2]
    a.plot(t, np.hypot(_wint(W, "uhx", t, dt), _wint(W, "uhy", t, dt)), color=C_W, lw=1.2, label=lab_w + " |u_h|")
    a.plot(t, np.where(hit, np.hypot(A["h_uhx"], A["h_uhy"]), np.nan), color=C_D, lw=1.0, label="pitcher |u_h|")
    a.plot(t, _wint(W, "exit_h", t, dt), color=C_W, lw=0.8, ls=":", label="wanted exit part")
    a.plot(t, np.where(A["j_on"] > 0, A["j_exh"], np.nan), color=C_D, lw=0.8, ls=":", label="pitcher exit horizontal")
    a.set_ylabel("speed [m/s]")
    a = ax[3]
    a.plot(t, _wint(W, "z", t, dt) * 100, color=C_W, lw=1.2, label="wanted spout height")
    a.plot(t, A["tip_z"] * 100, color=C_D, lw=1.0, label="pitcher tip height (incl. lift)")
    a.plot(t, A["lift"] * 100, color=C_4, lw=1.0, label="collision lift")
    a.set_ylabel("z [cm]")
    a = ax[4]
    xw, yw = _wint(W, "x", t, dt) * D, _wint(W, "y", t, dt) * D
    a.plot(t, np.where(hit, np.hypot(A["h_xh"] - xw, A["h_yh"] - yw) * 1e3, np.nan), color=C_D, lw=1.0,
           label="|foot - wanted foot|")
    a.axhline(TOL_MM, color=INK2, lw=0.6, ls="--")
    a.set_ylabel("foot error [mm]")
    a = ax[5]
    a.plot(t, np.degrees(A["tilt"]), color=C_D, lw=1.0, label="tilt")
    a.set_ylabel("tilt [deg]")
    a.set_xlabel("t [s]")
    for i, a in enumerate(ax):
        a.grid(True, color=GRID, lw=0.6)
        a.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for s in meta["strokes"]:
            if "yaw" in s:
                a.axvspan(s["t0"], s["t1"], color="#000000", alpha=0.04, lw=0)
                if i == 0:
                    a.text(0.5 * (s["t0"] + s["t1"]), 1.0, f"#{s['n']}", transform=a.get_xaxis_transform(),
                           ha="center", va="bottom", fontsize=8, color=INK2)
        a.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    tot = res["total"]
    fig.suptitle(f"{name}: {meta['pattern']}, fill {info['fill_mL']:.0f} mL, "
                 f"milk used {tot['milk_used_mL']:.0f} mL, flow delay {tot['Q_delay_s'] * 1e3:.0f} ms, "
                 f"max lift {tot['lift_max_mm']:.1f} mm", color=INK, fontsize=11)
    fn = f"{sset}_{meta['pattern']}.png" if (info["variant"] == "main" and info["fill_mL"] == 350) else f"{name}.png"
    fig.savefig(os.path.join(out, fn), dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------- report
QUANT = ("Q", "pulse", "U_perp", "u0", "exit_h", "tan", "uh", "r", "foot", "z")
QLABEL = dict(Q="流量 Q", pulse="5 Hz 脉动", U_perp="U⊥", u0="出口速度 u0", exit_h="出口水平速度", tan="tanα",
              uh="杯面 \\|u_h\\|", r="足迹 r", foot="落点位置", z="壶嘴高度")


def _worst(strokes, q):
    """Worst class over strokes: IMPOSSIBLE > DEVIATES (largest number) > DELIVERED; returns (label, stroke n)."""
    best = None
    for o in strokes:
        c = o["class"].get(q)
        if c is None or c == "n/a":
            continue
        rank = 2 if c.startswith("IMPOSSIBLE") else (1 if c.startswith("DEVIATES") else 0)
        key = (rank, _severity(o, q))
        if best is None or key > best[0]:
            best = (key, c, o["n"])
    return (best[1], best[2]) if best else ("n/a", None)


def _severity(o, q):
    return dict(Q=o.get("Q_err_rel", 0), pulse=abs(o.get("pulse_m_rel", 0)) + abs(o.get("pulse_dphase_deg", 0)) / 100,
                U_perp=abs(o.get("U_rel", 0)), u0=abs(o.get("u0_rel", 0)), exit_h=abs(o.get("exit_h_rel", 0)),
                tan=abs(o.get("tan_rel", 0)), uh=abs(o.get("uh_rel", 0)), r=abs(o.get("r_rel", 0)),
                foot=o.get("foot_rms_mm", 0), z=o.get("lift_fw_mm", 0) + o.get("z_err_mm", 0))[q]


def _short(c):
    if c.startswith("DELIVERED"):
        return "✓"
    if c.startswith("IMPOSSIBLE"):
        return "✗ " + c[len("IMPOSSIBLE "):]
    if c.startswith("DEVIATES"):
        return "≠ " + c[len("DEVIATES "):]
    return c


def _f(x, fmt="{:.2f}"):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def plot_clearance(R, out=OUT):
    """Lowest reachable spout height vs tip position along the spout axis (clearance.json), with the strokes of the
    FS and real 'main' runs at the tip position the pitcher actually took (marker = wanted height, line to the height
    the pitcher reached)."""
    path = os.path.join(out, "clearance.json")
    if not os.path.exists(path):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(path) as f:
        cm = json.load(f)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    cols = {"40": "#c3c2b7", "45": "#8f8e88", "50": "#52514e", "55": "#0b0b0b"}
    for ax, sset, title in ((axs[0], "fs", "FS strokes"), (axs[1], "real", "real-action stages")):
        ax.set_facecolor("#fcfcfb")
        for tl, c in cols.items():
            ax.plot(cm["s_mm"], cm["z_min_cm"][tl], color=c, lw=1.4, label=f"lowest tip height, tilt {tl} deg")
        first, first_ok = True, True
        for r, x in R.items():
            if x["info"]["set"] != sset or x["info"]["variant"] != "main" or x["info"]["fill_mL"] != 350:
                continue
            for o in x["strokes"]:
                if o.get("tip_s_mm") is None or not np.isfinite(o["tip_s_mm"]):
                    continue
                imp = o["class"]["z"].startswith("IMPOSSIBLE")
                ax.plot([o["tip_s_mm"]] * 2, [o["z_w_cm"], o["z_tip_cm"]], color=C_D if imp else C_W, lw=1.0, alpha=0.8)
                lab = None
                if imp and first:
                    lab, first = "wanted height, lift > 1 mm needed (line: height reached)", False
                elif not imp and first_ok:
                    lab, first_ok = "wanted height, reachable (lift <= 1 mm)", False
                ax.plot(o["tip_s_mm"], o["z_w_cm"], "o", ms=5, color=C_D if imp else C_W, label=lab)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel("spout-tip position along the spout axis from the cup centre [mm]\n"
                      "(the tip, not the stream foot; +: towards the pitcher body)", fontsize=9)
        ax.grid(True, color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("spout tip height above the coffee [cm]")
    axs[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("NX pitcher clearance (Pitcher.required_lift, default Cup): wanted heights vs what the pitcher can reach\n"
                 "strokes / stages: original barista ('main' controller), 350 mL initial fill, flow-weighted",
                 color=INK, fontsize=11)
    fig.savefig(os.path.join(out, "clearance.png"), dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def report(out=OUT):
    runs = sorted(f[:-13] for f in os.listdir(os.path.join(out, "runs")) if f.endswith("_metrics.json"))
    R = {}
    for r in runs:
        with open(os.path.join(out, "runs", r + "_metrics.json")) as f:
            R[r] = json.load(f)
    order = [run_name(s, p, v, fl) for s in ("fs", "real") for v, fl in (("main", 350), ("stock", 350), ("cmd", 350),
                                                                         ("pulse", 350), ("main", 250), ("main", 500))
             for p in PATTERNS]
    order = [r for r in order if r in R]
    js = dict(runs={}, criteria=dict(rel=TOL_REL, mm=TOL_MM, deg=TOL_DEG))
    L = []
    L.append("# 壶模型闭环核对：FS 和真实动作的入口量，奶壶倒不倒得出来\n")
    L.append("> 由 `latte_imex/v05_pitcher_check.py report` 生成。数据：`track/runs/*_metrics.json`；图：`track/<组>_<图案>.png`；"
             "求解器可用的 1 ms 入口表：`track/tables/*.npz`（pitcher = 壶实际送到杯面的，wanted = 原来喂给求解器的）。\n")
    L.append("> 判定：DELIVERED（✓）= 相对差 ≤ 15 %、位置/高度 ≤ 1 mm、相位 ≤ 10°；DEVIATES（≠）给出差多少；"
             "IMPOSSIBLE（✗）= 壶做不到（碰撞抬升 > 1 mm，即要的高度低于壶身/唇沿与杯子的间隙；或者根本没有奶柱）。"
             "逐笔取最差的一笔（括号里是笔号）。流量判据是逐笔的 ∫|Q_壶 − Q_要|dt / ∫Q_要 dt（不扣时延）。\n")
    blocks = [("fs", "main", 350, "FS（猜的全物理动作）：装奶 350 mL，main 咖啡师（原控制器 + 两处数值修正）"),
              ("real", "main", 350, "真实动作（stage scripts + PhysInlet）：装奶 350 mL，main 咖啡师"),
              ("fs", "stock", 350, "FS：装奶 350 mL，stock 咖啡师（pitcher.Barista 原样）"),
              ("real", "stock", 350, "真实动作：装奶 350 mL，stock 咖啡师"),
              ("fs", "pulse", 350, "FS：pulse 变体（理想 5 Hz 绕壶嘴尖倾转 + 锁相自适应）"),
              ("fs", "cmd", 350, "FS：cmd 变体（咖啡师照 FS 的滞后前指令倒，壶嘴滞后由壶自己给）"),
              ("fs", "main", 250, "FS：装奶 250 mL"), ("fs", "main", 500, "FS：装奶 500 mL"),
              ("real", "main", 250, "真实动作：装奶 250 mL"), ("real", "main", 500, "真实动作：装奶 500 mL")]
    for sset, var, fill, title in blocks:
        rows = [r for r in order if R[r]["info"]["set"] == sset and R[r]["info"]["variant"] == var
                and R[r]["info"]["fill_mL"] == fill]
        if not rows:
            continue
        L.append(f"\n## {title}\n")
        L.append("格内：最差一笔的判定（笔号）[没送到的笔数 / 总笔数]。\n")
        L.append("| 图案 | " + " | ".join(QLABEL[q] for q in QUANT) + " |")
        L.append("|---|" + "---|" * len(QUANT))
        for r in rows:
            st = R[r]["strokes"]
            cells = []
            for q in QUANT:
                c, n = _worst(st, q)
                cl = [o["class"].get(q, "n/a") for o in st]
                cl = [x for x in cl if x != "n/a"]
                bad = sum(1 for x in cl if not x.startswith("DELIVERED"))
                cell = "–" if c == "n/a" else (_short(c) + (f"（#{n}）" if n is not None and not c.startswith("DELIVERED")
                                                             else "") + f" [{bad}/{len(cl)}]")
                cells.append(cell.replace("|", "\\|"))
            L.append(f"| {CN[R[r]['info']['pattern']]} | " + " | ".join(cells) + " |")
    # overview
    L.append("\n## 运行总览\n")
    L.append("| 运行 | 墙钟 s | 步数 | 要的奶量 / 壶送到 mL | 用奶 / 装奶 mL | 流量时延 ms | 全程 ∫\\|ΔQ\\| / ∫Q（扣时延后） | 倾角范围 ° | 最大倾角速率 °/s | 最大偏航速率 °/s | 最大碰撞抬升 mm（时长 s） | 溢出 mL |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in order:
        x = R[r]
        tt, inf = x["total"], x["info"]
        L.append(f"| {r} | {inf['wall_s']:.0f} | {inf['steps']} | {tt['V_w_mL']:.1f} / {tt['V_d_mL']:.1f} | "
                 f"{tt['milk_used_mL']:.0f} / {tt['fill_mL']:.0f} | {tt['Q_delay_s'] * 1e3:.0f} | "
                 f"{tt['Q_err_rel'] * 100:.0f} % ({tt['Q_err_rel_delayed'] * 100:.0f} %) | "
                 f"{tt['tilt_range_deg'][0]:.1f}–{tt['tilt_range_deg'][1]:.1f} | {tt['tilt_rate_max']:.0f} | "
                 f"{tt['yaw_rate_max']:.0f} | {tt['lift_max_mm']:.1f} ({tt['lift_time_s']:.2f}) | {tt['overflow_mL']:.1f} |")
    # per-stroke detail
    L.append("\n## 逐笔数字\n")
    L.append("列：Q 要/壶（mL/s，笔内平均）、奶量比、∫|ΔQ|/∫Q（扣全程时延后）、起流迟到 ms、断流 s；脉动 m 要/壶、相位差 °；"
             "U⊥ 要/壶（m/s，壶的 U⊥ 若按 FS 公式用壶的实际落差算）；u0 壶（FS 取 0.35）；出口水平速度 要/壶；tanα 要/壶；|u_h| 要/壶；"
             "足迹 r 要/壶（连续性，mm；括号是壶的物理液片足迹 √(r1 r2)）；落点误差 rms（扣最佳时延后，时延 ms）；壶嘴高 要/实际 cm、最大抬升 mm；倾角范围。\n")
    for r in order:
        x = R[r]
        L.append(f"\n### {r}\n")
        L.append("| 笔 | Q 要/壶 | 奶量比 | ∫\\|ΔQ\\| | 起流迟 ms / 断流 s | 脉动 m 要/壶, Δφ | U⊥ 要/壶 (公式@壶落差) | u0 | 出口水平 要/壶 | tanα 要/壶 | \\|u_h\\| 要/壶 | r 要/壶 (物理) mm | 落点 rms mm | z 要/实 cm, 抬升 mm | 倾角 ° |")
        L.append("|---|" + "---|" * 14)
        for o in x["strokes"]:
            pul = "–"
            if "pulse_m_d" in o:
                pul = f"{o['pulse_m_w']:.2f}/{o['pulse_m_d']:.2f}, {o['pulse_dphase_deg']:+.0f}°"
            sd = o.get("start_delay_s")
            L.append(
                f"| {o['name']} | {o['Q_w_mean']:.1f}/{o['Q_d_mean']:.1f} | {o['Q_ratio']:.2f} | "
                f"{o['Q_err_rel'] * 100:.0f} % ({o['Q_err_rel_delayed'] * 100:.0f} %) | "
                f"{_f(sd * 1e3 if sd is not None and np.isfinite(sd) else sd, '{:.0f}')} / {_f(o.get('break_s'), '{:.3f}')} | {pul} | "
                f"{_f(o['U_w'])}/{_f(o['U_d'])} ({_f(o['U_formula_at_drop'])}) | {_f(o['u0'])} | "
                f"{_f(o['exit_h_w'], '{:.3f}')}/{_f(o['exit_h_d'], '{:.3f}')} | {_f(o['tan_w'])}/{_f(o['tan_d'])} | "
                f"{_f(o['uh_w'], '{:.3f}')}/{_f(o['uh_d'], '{:.3f}')} | {_f(o['r_w_mm'])}/{_f(o['r_d_mm'])} ({_f(o['r_phys_mm'])}) | "
                f"{_f(o.get('foot_rms_mm'), '{:.1f}')} ({_f(o.get('foot_rms_delayed_mm'), '{:.1f}')}, {_f((o.get('foot_delay_s') or 0) * 1e3, '{:.0f}')}) | "
                f"{o['z_w_cm']:.2f}/{o['z_tip_cm']:.2f}, {o['lift_fw_mm']:.1f} (max {o['lift_max_mm']:.1f}) | "
                f"{o['tilt_range_deg'][0]:.0f}–{o['tilt_range_deg'][1]:.0f} |")
        if x["gaps"]:
            gl = "; ".join(f"{g['name']} 要漏 {g['V_w_mL']:.2f} mL / 壶漏 {g['V_d_mL']:.2f} mL，<1 mL/s 时长 要 {g['zero_w_s']:.2f} / 壶 {g['zero_d_s']:.2f} s"
                           for g in x["gaps"])
            L.append(f"\n桥/停：{gl}\n")
        js["runs"][r] = dict(info=x["info"], total=x["total"],
                             classes={str(o["n"]): o["class"] for o in x["strokes"]},
                             worst={q: _worst(x["strokes"], q) for q in QUANT})
    plot_clearance(R, out)
    with open(os.path.join(out, "summary_tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(js, f, indent=1, default=float)
    return R


# --------------------------------------------------------------------------- clearance map
def clearance_map(out=OUT):
    """Lowest spout-tip height the NX pitcher can take above the coffee (Pitcher.required_lift against the default
    Cup) vs the tip position along the spout axis (s > 0: towards the pitcher body / the near rim) and the tilt; and the
    steady tilt for a flow at several fills (tilt_for_flow, tip 1.5 cm above the surface at the centre)."""
    g = nx_geometry(6)
    P = FastPitcher(geom=g, V0=350 * ML)
    cup = Cup()
    S = list(range(-30, 35, 5))
    TL = list(range(30, 70, 5))
    zmin = {}
    for tl in TL:
        row = []
        for s_mm in S:
            pose = Pose((0.0, s_mm * 1e-3, 0.002), -math.pi / 2, math.radians(tl), 0.0)
            row.append(round((0.002 + P.required_lift(pose, cup)) * 100, 2))
        zmin[tl] = row
    tilt = {}
    for fill in (250, 300, 350, 400, 450, 500):
        P2 = FastPitcher(geom=g, V0=fill * ML)
        pose = Pose((0.0, 0.008, 0.015), -math.pi / 2, 0.0, 0.0)
        tilt[fill] = dict(onset=round(math.degrees(onset_tilt(P2, pose, fill * ML)), 1))
        for q in (2, 5, 8, 12, 16, 22):
            tilt[fill][q] = round(math.degrees(P2.tilt_for_flow(pose, q * ML, V=fill * ML, iters=26)), 1)
    res = dict(s_mm=S, tilts_deg=TL, z_min_cm=zmin, tilt_for_flow_deg=tilt,
               note="tip position s along the spout axis from the cup centre, s > 0 towards the pitcher body; the stream"
                    " foot is 5-10 mm ahead of the tip (s smaller)")
    with open(os.path.join(out, "clearance.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


# --------------------------------------------------------------------------- self test
def selftest():
    """FastPitcher.free_surface against Pitcher.free_surface on random poses / fills; table round trip."""
    g = nx_geometry(6)
    rng = np.random.default_rng(1)
    P0, P1 = Pitcher(geom=g, V0=350 * ML), FastPitcher(geom=g, V0=350 * ML)
    err = err_q = 0.0
    for _ in range(400):
        pose = Pose((rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(0.0, 0.08)),
                    rng.uniform(-math.pi, math.pi), math.radians(rng.uniform(0, 110)), math.radians(rng.uniform(-5, 5)))
        sl = rng.uniform(-0.2, 0.2, 2)
        n = np.array([-sl[0], -sl[1], 1.0])
        P0.n = P1.n = n / np.linalg.norm(n)
        V = rng.uniform(-10, 560) * ML
        c0, o0 = P0.free_surface(pose, V)
        c1, o1 = P1.free_surface(pose, V)
        err = max(err, abs(c0 - c1), abs(o0 - o1))
        W0, W1 = P0.weir(pose, c0), P1.weir(pose, c1)
        err_q = max(err_q, abs(W0["Q_ss"] - W1["Q_ss"]))
    print(f"free_surface max |diff| over 400 random poses / fills: {err:.3e} m; weir Q_ss {err_q:.3e} m^3/s")
    assert err < 1e-12
    return err


# =========================================================================== best case: an ideal hand
# Is the virtual barista (pitcher.Barista: feed-forward tilt + PI, 60 deg/s, no preview) the bottleneck, or the
# pitcher?  For the FS inlets the pitcher model is driven by an IDEAL HAND that knows the model and the plan:
#   * flow: the wanted arrival flow Q_ref(t) is shifted by the jet flight time (preview), the spout lag is inverted
#     exactly on the pitcher's own lip buffer (Q_feed = Q_out + tau dQ_out/dt + (tau Q_out - V_lip)/Tc), and the steady
#     weir law is inverted for the tilt every 1 ms (secant on the lip head with the local law Q = K (h - h_k)^2.5, using
#     the current fill, free-surface slope and g_eff).  Zero / tiny feed (< 0.2 mL/s) is never faked by a trickle: the
#     hand then holds the head between h_off and h_k (no feed, the lip drains on tau_stop, but the meniscus stays on the
#     lip so the next stroke restarts without the h_on jump).  Before the first stroke the head waits 0.3 mm below
#     h_on; after the last one the hand tilts back below h_off.
#   * iterative learning (a barista who has practised the pattern): after each pass Q_ref <- Q_ref + gamma (Q_FS - Q_hit)
#     on the arrival time axis (gamma 0.8, 5 ms smoothing), 4 passes.  Whatever the limits below allow, it finds.
#   * tilt actuator: second-order tracker (20 Hz) bounded by the human limits HAND: tilt rate <= 300 deg/s and tilt
#     acceleration <= acc / pivot_L (the model tilts about the spout tip; a real hand holds the handle ~15 cm away, so a
#     tip-fixed tilt makes the hand translate with L theta''; <= 10 m/s^2).
#   * foot: tip_xy(t) = foot_FS(t + t_hit) - (x_hit - tip_xy) of the last released parcel (flight time and jet offset,
#     including the carried pitcher velocity, compensated with preview).
#   * height: tip z = z_FS + L(t); L is planned from the collision requirement R(t) = Pitcher.required_lift at the
#     wanted height recorded in the previous passes (max over passes): morphological closing of R with the hand's
#     parabola (|L'| <= 0.5 m/s, |L''| <= 10 m/s^2), so L >= R everywhere and the hand starts lifting early.  Pass 1
#     follows R online (median + 40 ms filter, as the Track 'main' hand).
#   * yaw: the wanted yaw schedule smoothed with a zero-phase Gaussian until |yaw'| <= 300 deg/s.
# best_static() is the complementary static search: for samples of every FS stroke, the lowest reachable spout height
# (required_lift) over yaw (every 30 deg + the FS yaw), roll (-20..20 deg) and fill (350..200 mL) at the tilt that gives
# the wanted flow, and the exit speed / exit horizontal speed that the pitcher gives there.
BEST = os.path.join(REVIEW, "v05pitcher", "best")
HAND = dict(tilt_rate=math.radians(300.0), pivot_L=0.15, acc=10.0, speed=0.5, yaw_rate=math.radians(300.0), f_max=6.0)
HAND_SRC = dict(
    tilt_rate="300 deg/s: daily wrist flexion/extension peak (envelope.md section 7, arXiv:2408.14361)",
    pivot_L="0.15 m wrist-to-spout lever (envelope.md section 6.2 uses a 15 cm pivot)",
    acc="10 m/s^2: upper end of controllable hand motion (envelope.md section 7 'reasonable range'; video wiggle peaks "
        "1.0-1.9 m/s^2, video_ranges.md section 2)",
    speed="0.5 m/s tip speed (video: path <= 0.071 m/s, lifts <= 0.11 m/s, envelope.md section 7; pitcher.kinematics "
          "clips at 1.5 m/s)",
    yaw_rate="300 deg/s about the vertical (no measurement; the wrist number is used for the forearm)",
    f_max="6 Hz voluntary oscillation bandwidth (envelope.md section 7: tapping 5-7 Hz, joystick ~6 Hz)")
LAW_VARIANTS = dict(ideal={}, idealslosh={}, Cd055=dict(C_d=0.55), Cd065=dict(C_d=0.65),
                    tau004=dict(tau=0.04, tau_stop=0.075), tau015=dict(tau=0.15, tau_stop=0.28), hon125=dict(h_on=0.00125),
                    hon375=dict(h_on=0.00375))
PULSE_MODE = dict(idealslosh="slosh")     # every other variant makes the 5 Hz with the tilt
PULSE_GAIN, PULSE_RCLIP = 0.6, 1.6      # between-pass update of the 5 Hz amplitude: C <- C r^0.6, |r| in [1/1.6, 1.6]
PULSE_HREF = 5e-3                        # [m] lip head above h_k at which the oscillation amplitude C is quoted
SLOSH_ACC = 5.0                          # [m/s^2] share of the hand's 10 m/s^2 given to the 5 Hz fore-aft slosh drive
XKEYS = ("x_tilt_tgt", "x_rate", "x_F", "x_mode", "x_Qo", "x_R", "x_th", "x_oscy")


def _gauss_smooth(a, sigma_n):
    m = int(math.ceil(4 * sigma_n))
    k = np.exp(-0.5 * (np.arange(-m, m + 1) / sigma_n) ** 2)
    k /= k.sum()
    ap = np.concatenate([np.full(m, a[0]), a, np.full(m, a[-1])])
    return np.convolve(ap, k, mode="valid")


def smooth_yaw(yaw, dt, rate):
    """Fastest yaw schedule within |yaw'| <= rate that keeps every turn of the wanted (unwrapped) schedule centred where
    it was: each turn (maximal run of changing samples) whose peak rate exceeds the limit is replaced by a constant-rate
    ramp of duration |turn| / (0.95 rate) centred on the turn's midpoint; corners rounded by a 10 ms Gaussian (which
    does not raise the peak rate)."""
    yaw = np.asarray(yaw, float)
    n = len(yaw)
    d = np.diff(yaw)
    if n < 3 or np.max(np.abs(d)) / dt <= rate:
        return yaw.copy()
    mv = np.nonzero(np.abs(d) > 1e-12)[0]
    runs, a = [], mv[0]
    for i0, i1 in zip(mv[:-1], mv[1:]):
        if i1 > i0 + 1:
            runs.append((a, i0 + 1)); a = i1
    runs.append((a, mv[-1] + 1))
    y = yaw.copy()
    tt = np.arange(n) * dt
    for a, b in runs:
        if np.max(np.abs(d[a:b])) / dt <= rate:
            continue
        y0, y1 = yaw[a], yaw[b]
        c = 0.5 * (a + b) * dt
        T = abs(y1 - y0) / (0.95 * rate)
        lo, hi = c - 0.5 * T, c + 0.5 * T
        m = (tt >= min(lo, a * dt)) & (tt <= max(hi, b * dt))
        y[m] = y0 + (y1 - y0) * np.clip((tt[m] - lo) / T, 0.0, 1.0)
    return _gauss_smooth(y, 0.01 / dt)


def max_filter(a, n):
    """Running maximum over a centred window of n samples (edge-padded)."""
    a = np.asarray(a, float)
    h = n // 2
    ap = np.concatenate([np.full(h, a[0]), a, np.full(n - 1 - h, a[-1])])
    out = np.full(len(a), -np.inf)
    for j in range(n):
        out = np.maximum(out, ap[j:j + len(a)])
    return out


def lift_plan(R, dt, speed):
    """Lift the hand follows: running maximum over LIFT_MAXWIN (no lift oscillation at the 5 Hz of a pulsed stroke, whose
    tilt oscillation makes the requirement oscillate; an oscillating lift is a vertical acceleration that modulates
    g_eff and the flow), then the closing with the hand parabola, plus 0.2 mm where anything is needed."""
    Rf = max_filter(np.where(R > 0, R + 2e-4, 0.0), max(1, int(round(LIFT_MAXWIN / dt))))
    return lift_closing(Rf, dt, LIFT_ACC, speed)


def lift_closing(R, dt, acc, speed, win=0.4):
    """Smallest smooth upper envelope of R a hand can follow: morphological closing (dilation, then erosion) with the
    structuring function phi(s) = acc s^2 / 2 (|s| < speed/acc), speed |s| - speed^2/(2 acc) beyond.  Result >= R,
    slope <= speed, curvature <= acc on every parabolic piece."""
    R = np.asarray(R, float)
    n = len(R)
    m = int(round(win / dt))
    s = np.abs(np.arange(-m, m + 1) * dt)
    phi = np.where(s < speed / acc, 0.5 * acc * s ** 2, speed * s - speed ** 2 / (2 * acc))
    Rp = np.concatenate([np.full(m, R[0]), R, np.full(m, R[-1])])
    Dl = np.full(n, -np.inf)
    for j in range(2 * m + 1):
        Dl = np.maximum(Dl, Rp[j:j + n] - phi[j])
    Dp = np.concatenate([np.full(m, Dl[0]), Dl, np.full(m, Dl[-1])])
    E = np.full(n, np.inf)
    for j in range(2 * m + 1):
        E = np.minimum(E, Dp[j:j + n] + phi[j])
    return np.maximum(E, R)


def _movavg(a, k):
    if k <= 1:
        return a
    ker = np.ones(k) / k
    ap = np.concatenate([np.full(k // 2, a[0]), a, np.full(k - 1 - k // 2, a[-1])])
    return np.convolve(ap, ker, mode="valid")


class IdealHand:
    """Best-case barista for one pass (see the section comment).  Interface used by _flatten: yaw, z_cmd, offset,
    pitcher."""

    def __init__(self, P, W, meta, Q_ref, n_ext, hand=None, lift_plan=None, yaw_plan=None, Tc=0.08, Fmin=0.2 * ML,
                 wn=2 * math.pi * 20.0, cup=None, pulse=None, Q_env=None):
        self.P = self.pitcher = P
        # pulsed strokes: list of dict(t0, t1 [arrival axis], w, C, m, mode, c_max): the flow inversion targets the
        # envelope x m (no lip-volume correction, which would fight the 5 Hz), and the 5 Hz is made open loop by a tilt
        # oscillation about the spout tip (mode 'tilt', C in rad) or a fore-aft tip translation along the spout direction
        # (mode 'slosh', C in m), Re(C e^{i w tt}) x (lip head above h_k for this feed) / 5 mm, at most c_max;
        # tt = stroke clock on the arrival axis
        self.pulse = list(pulse or [])
        self.Qenv = None if Q_env is None else np.asarray(Q_env, float)

        self.hand = dict(HAND if hand is None else hand)
        self.dt_tab = meta["dt"]
        n_tab = len(W["t"])

        def ext(a):
            a = np.asarray(a, float)
            return np.concatenate([a, np.full(n_ext - n_tab, a[-1])])
        self.xw, self.yw, self.zw = ext(W["x"]) * D, ext(W["y"]) * D, ext(W["z"])
        self.Qref = np.asarray(Q_ref, float)
        self.yaw_plan = ext(W["yaw"]) if yaw_plan is None else np.asarray(yaw_plan, float)
        self.L_plan = None if lift_plan is None else np.asarray(lift_plan, float)
        self.n = n_ext
        pos = np.nonzero(self.Qref > 1e-9)[0]
        self.t_last = (pos[-1] + 1) * self.dt_tab if len(pos) else 0.0
        self.Tc, self.Fmin, self.wn = Tc, Fmin, wn
        self.amax = self.hand["acc"] / self.hand["pivot_L"]
        self.cup = Cup() if cup is None else cup
        self.n_eval = 0
        law = P.law
        # initial pose: head 0.3 mm below the onset, foot on the first wanted point
        yaw0, z0 = float(self.yaw_plan[0]), float(self.zw[0])
        tip = np.array([self.xw[0], self.yw[0]])
        th_wait = self._tilt_for_head(Pose((tip[0], tip[1], z0), yaw0, 0.0, 0.0), law.h_on - 3e-4)
        th_run = self._tilt_for_head(Pose((tip[0], tip[1], z0), yaw0, 0.0, 0.0), law.h_on + 1e-3)
        for _ in range(3):                                   # jet offset at a running head
            pose = Pose((tip[0], tip[1], z0), yaw0, th_run, 0.0)
            c, _ = P.free_surface(pose, P.state.V)
            Wr = P.weir(pose, c)
            J = P.jet(pose, max(Wr["Q_ss"], 1e-7), Wr["h_bar"], Wr["origin"], 0.0, Wr["width"], Wr["tangent"])
            off = J["x_hit"] - tip
            tip = np.array([self.xw[0], self.yw[0]]) - off
        self.offset = off
        self.t_hit = J["t_hit"]
        self.K = Wr["Q_ss"] / max(Wr["h_max"] - law.h_k, 1e-5) ** 2.5
        h1, _ = self._hq(pose, th_run + math.radians(0.2))
        h0, _ = self._hq(pose, th_run - math.radians(0.2))
        self.sh = max((h1 - h0) / math.radians(0.4), 1e-4)  # d h_max / d tilt [m/rad]
        self.tilt, self.rate = th_wait, 0.0
        self.tgt_prev, self.vt, self.base_prev = None, 0.0, None
        self.tilt0 = th_wait
        self.yaw, self.z_cmd = yaw0, z0
        self.buf, self.L_on, self.vL = [], None, 0.0
        self.tip_prev = None
        self.off_init = False
        self.th_last, self.dth = None, 0.0

    # ---- helpers
    def _at(self, a, t):
        u = t / self.dt_tab
        i = int(math.floor(u))
        if i >= self.n - 1:
            return float(a[-1])
        if i < 0:
            return float(a[0])
        f = u - i
        return float(a[i] * (1 - f) + a[i + 1] * f)

    def _hq(self, pose, tilt):
        p = Pose(pose.tip, pose.yaw, tilt, pose.roll)
        c, _ = self.P.free_surface(p, self.P.state.V)
        Wr = self.P.weir(p, c)
        self.n_eval += 1
        return Wr["h_max"], Wr["Q_ss"]

    def _tilt_for_head(self, pose, h):
        lo, hi = 0.0, math.radians(150)
        for _ in range(32):
            m = 0.5 * (lo + hi)
            if self._hq(pose, m)[0] < h:
                lo = m
            else:
                hi = m
        return 0.5 * (lo + hi)

    def _solve(self, pose, kind, target):
        """Tilt for a lip head (kind 'h') or a steady weir flow (kind 'Q'): two secant steps on h_max."""
        hk = self.P.law.h_k
        lim = (0.0, math.radians(150))
        th0 = self.base_prev if self.base_prev is not None else self.tilt
        h0, q0 = self._hq(pose, th0)
        if kind == "Q":
            if q0 > 0 and h0 > hk + 5e-5:
                self.K = q0 / (h0 - hk) ** 2.5
            ht = hk + (target / self.K) ** 0.4
        else:
            ht = target
        th1 = min(max(th0 + max(-0.2, min(0.2, (ht - h0) / self.sh)), lim[0]), lim[1])
        h1, q1 = self._hq(pose, th1)
        if abs(th1 - th0) > math.radians(0.02):
            s = (h1 - h0) / (th1 - th0)
            if SH_MIN < s < SH_MAX:                    # local slope of the lip head [m/rad]
                self.sh = s
        if kind == "Q" and q1 > 0 and h1 > hk + 5e-5:
            self.K = q1 / (h1 - hk) ** 2.5
            ht = hk + (target / self.K) ** 0.4
        return min(max(th1 + max(-0.2, min(0.2, (ht - h1) / self.sh)), lim[0]), lim[1])

    def _lift(self, t, R, dt):
        if self.L_plan is not None:
            return max(self._at(self.L_plan, t), 0.0)
        self.buf.append(R)
        if len(self.buf) > 9:
            self.buf.pop(0)
        tgt = float(np.median(self.buf))
        if self.L_on is None:
            self.L_on = tgt
            return tgt
        tau = 0.04
        self.vL += dt * ((tgt - self.L_on) / (tau * tau) - 2.0 * self.vL / tau)
        self.L_on = max(0.0, self.L_on + dt * self.vL)
        return self.L_on

    # ---- one 1 ms frame
    def step(self, dt):
        P, st, law, Hd = self.P, self.P.state, self.P.law, self.hand
        t = st.t
        ta = t + 0.5 * dt + self.t_hit
        yaw = self._at(self.yaw_plan, t)
        zw = self._at(self.zw, t)
        # foot placement with the carried velocity: x_hit = tip + off_nc + v_tip t_hit with v_tip = (tip - tip_prev)/dt,
        # solved implicitly for this frame's tip (an explicit use of the last offset, carry included, is a loop gain
        # t_hit/dt ~ 30 and diverges)
        fx = self._at(self.xw, ta) - self.offset[0]
        fy = self._at(self.yw, ta) - self.offset[1]
        if self.tip_prev is None:
            tx, ty = fx, fy
        else:
            kk = self.t_hit / dt
            tx = (fx + kk * self.tip_prev[0]) / (1.0 + kk)
            ty = (fy + kk * self.tip_prev[1]) / (1.0 + kk)
        hd = 0.002
        pz = None
        for q in self.pulse:
            if q["t0"] <= ta < q["t1"]:
                pz = q
                break
        if pz is not None:
            tt = ta - pz["t0"]
            env = max(0.0, min(1.0, tt / 0.05, (pz["t1"] - ta) / 0.05))
            ph = complex(math.cos(pz["w"] * tt), math.sin(pz["w"] * tt))
            Qo = max(self._at(self.Qenv, ta), 0.0) * pz["m"]
            dQo = (self._at(self.Qenv, ta + hd) - self._at(self.Qenv, ta - hd)) / (2 * hd) * pz["m"]
            F = Qo + law.tau * dQo           # the cycle-mean factor m (learned between passes) absorbs the rectification
            # oscillation scaled with the lip head above h_k for this feed (the same relative modulation needs a
            # proportionally smaller tilt / slosh at a small head), bounded by the hand
            sc = min(2.0, max(0.3, (max(F, 0.0) / max(self.K, 1e-12)) ** 0.4 / PULSE_HREF))
            amp = abs(pz["C"]) * sc
            cl = min(1.0, pz["c_max"] / amp) if amp > 0 else 1.0
            o = env * sc * cl * (pz["C"] * ph).real
            osc_t, osc_y = (o, 0.0) if pz["mode"] == "tilt" else (0.0, o)
        else:
            osc_t = osc_y = 0.0
            # arrival Jacobian: a growing flight time (lifting) stretches the arrivals, Q_hit = Q_out / (1 + dt_hit/dt)
            jac = max(0.5, min(2.0, 1.0 + self.dth))
            Qo = max(self._at(self.Qref, ta), 0.0) * jac
            dQo = (self._at(self.Qref, ta + hd) - self._at(self.Qref, ta - hd)) / (2 * hd) * jac
            F = Qo + law.tau * dQo + (law.tau * Qo - st.V_lip) / self.Tc
        pose = Pose((tx, ty, zw), yaw, self.tilt, 0.0)
        if ta > self.t_last + 0.05:
            mode, kind, tgt = 3, "h", 0.4 * law.h_off               # done: tilt back, the stream stops
        elif F <= self.Fmin:
            mode, kind = 0, "h"                                       # no feed: hold the meniscus on the lip
            tgt = 0.5 * (law.h_off + law.h_k) if st.flowing else law.h_on - 3e-4
        elif not st.flowing:
            mode, kind, tgt = 1, "h", law.h_on + 1.5e-4              # start: just over the onset head
        else:
            mode, kind, tgt = 2, "Q", F
        base = self._solve(pose, kind, tgt)
        self.base_prev = base
        th_t = base + osc_t
        # tilt actuator: 20 Hz second-order tracker, rate and acceleration bounded by the hand
        if self.tgt_prev is not None:
            d_t = th_t - self.tgt_prev
            if abs(d_t) > math.radians(0.3):          # a jump of the target (mode switch) is not a target velocity
                d_t = 0.0
            self.vt += (d_t / dt - self.vt) * (1.0 - math.exp(-dt / 0.005))
            self.vt = max(-Hd["tilt_rate"], min(Hd["tilt_rate"], self.vt))
        self.tgt_prev = th_t
        e = th_t - self.tilt
        # braking distance with 60 % of the hand's angular deceleration: no overshoot on target steps
        r_des = self.vt + math.copysign(min(math.sqrt(1.2 * self.amax * abs(e)), self.wn * abs(e)), e)
        r_des = max(-Hd["tilt_rate"], min(Hd["tilt_rate"], r_des))
        a = max(-self.amax, min(self.amax, (r_des - self.rate) / dt))
        self.rate = max(-Hd["tilt_rate"], min(Hd["tilt_rate"], self.rate + a * dt))
        self.tilt = max(0.0, min(math.radians(150), self.tilt + self.rate * dt))
        ax, ay = tx + osc_y * math.cos(yaw), ty + osc_y * math.sin(yaw)        # actual tip incl. the slosh drive
        R = P.required_lift(Pose((ax, ay, zw), yaw, self.tilt, 0.0), self.cup)
        L = self._lift(t, R, dt)
        z = zw + L
        pose = Pose((ax, ay, z), yaw, self.tilt, 0.0)
        self.yaw, self.z_cmd = yaw, z
        rec = P.step(pose, dt)
        J = rec["jet"]
        if J is not None:
            if self.th_last is not None:
                self.dth += ((J["t_hit"] - self.th_last) / dt - self.dth) * (1.0 - math.exp(-dt / 0.02))
            self.th_last = J["t_hit"]
            self.t_hit = J["t_hit"]
            vel = np.asarray(rec["vel"], float)[:2] if P.law.carry_velocity else np.zeros(2)
            off = J["x_hit"] - np.array([ax, ay]) - vel * J["t_hit"]       # jet offset without the carry
            # low-passed (OFF_TAU): the slow drift of the offset (height, flow, fill) is compensated with preview, its
            # 5 Hz jitter (crest point and exit direction move with a 5 Hz tilt) is not chased: chasing it shakes the
            # tip at 5 Hz, which sloshes the milk
            if self.off_init:
                self.offset = self.offset + (off - self.offset) * (1.0 - math.exp(-dt / OFF_TAU))
            else:
                self.offset, self.off_init = off, True
        self.tip_prev = (tx, ty)                       # the foot placement works on the tip without the slosh drive
        rec.update(Q_want=Qo, lift=L, lift_raw=R, osc=osc_t, x_tilt_tgt=th_t, x_rate=self.rate, x_F=F, x_mode=mode,
                   x_Qo=Qo, x_R=R, x_th=self.t_hit, x_oscy=osc_y)
        return rec


PREROLL = 0.3                            # [s] the ideal hand may start this long before the FS clock
OFF_TAU = 0.05                           # [s] low-pass of the jet offset used for the foot placement
LIFT_MAXWIN = 0.25                       # [s] running-max window on the collision requirement before the closing
SH_MIN, SH_MAX = 0.1e-3 / math.radians(1.0), 5e-3 / math.radians(1.0)    # accepted d h_max / d tilt [m/rad]
LIFT_ACC = 3.0      # [m/s^2] the ideal hand plans its lifts gentler than the 10 m/s^2 limit (vertical acceleration changes
                    # g_eff and sloshes the milk; 3 m/s^2 still lifts 10 mm in 0.12 s)


def lift_preplan(W, meta, fill, law_kw, n_ext, every=0.01):
    """Collision requirement R(t) for pass 1: along the wanted path, at the steady tilt for the wanted envelope flow
    (at least 0.5 mL/s, i.e. about the hold tilt in bridges) and the fill drawn down by the wanted volume; the tip is
    placed from a static jet so that the foot is on the path; sampled every 10 ms, interpolated to 1 ms."""
    dt = meta["dt"]
    P = FastPitcher(geom=nx_geometry(6), V0=fill * ML, law=OutflowLaw(**law_kw))
    cup = Cup()
    Qw = np.nan_to_num(np.asarray(W["Q"], float))
    used = np.cumsum(Qw) * dt
    Qe = np.nan_to_num(np.asarray(W["Q_env"], float)) if "Q_env" in W else Qw
    n_tab = len(Qw)
    idx = np.arange(0, n_tab, max(1, int(round(every / dt))))
    R = []
    for i in idx:
        V = fill * ML - used[i]
        Q = max(Qe[i], 0.5 * ML)
        foot = np.array([W["x"][i] * D, W["y"][i] * D])
        z, yaw = float(W["z"][i]), float(W["yaw"][i])
        tilt = P.tilt_for_flow(Pose((foot[0], foot[1], z), yaw, 0.0, 0.0), Q, V=V, iters=22)
        pose = Pose((foot[0], foot[1], z), yaw, tilt, 0.0)
        c, _ = P.free_surface(pose, V)
        Wr = P.weir(pose, c)
        J = P.jet(pose, Q, max(Wr["h_bar"], P.law.h_k), Wr["origin"], 0.0, Wr["width"], Wr["tangent"])
        tip = foot - (J["x_hit"] - foot)
        R.append(P.required_lift(Pose((tip[0], tip[1], z), yaw, tilt, 0.0), cup))
    R = np.interp(np.arange(n_ext), idx, np.asarray(R))
    return R


def best_name(pattern, variant="ideal", fill=350):
    return run_name("fs", pattern, variant, fill)


def best_simulate(pattern, variant="ideal", fill=350, npass=None, hand=None, dt=DT, tail=0.6, verbose=True):
    """Ideal-hand runs (see the section comment); returns (W, meta, A, info) of the last pass.
    Passes: pass 0 lifts on the pre-planned requirement; every pass adds its recorded requirement to the lift plan
    (max over passes).  Pulsed strokes: the hand pours the envelope x m and adds a 5 Hz oscillation (tilt about the tip, or with
    variant 'idealslosh' a fore-aft tip translation that drives the slosh mode), scaled with the lip head above h_k and
    bounded by the hand (tilt: angular acceleration acc / pivot_L and rate; slosh: 5 m/s^2 of the 10 m/s^2 budget).  Its
    complex amplitude C and the cycle-mean factor m (the weir law rectifies the oscillation) are learned between passes
    from the lock-in of the arriving flow against the FS flow: C <- C (c_FS / c_hit)^0.6 (ratio clipped to [1/1.6, 1.6]),
    m <- m (mean_FS / mean_hit)^0.6 (a practising barista; a per-cycle update inside the pass was less stable).  The envelope
    itself is not learned (a time-domain learning update on the arrival axis made the non-pulsed strokes worse: the lip
    tail and the changing lift plan are not a pure delay)."""
    t0w = time.time()
    hand = dict(HAND if hand is None else hand)
    law_kw = LAW_VARIANTS[variant]
    W0, meta0 = fs_wanted(pattern)
    # pre-roll: the FS clock starts with the stream already on its way (Q ramps from t = 0 at the surface); the ideal hand
    # may start tilting before that, so the run starts PREROLL earlier on a padded copy and is cut back afterwards
    n_pre = int(round(PREROLL / dt))
    W = {k: np.concatenate([np.full(n_pre, 0.0 if k.startswith("Q") else np.asarray(W0[k], float)[0]),
                            np.asarray(W0[k], float)]) for k in ("x", "y", "z", "yaw", "Q", "Q_env")}
    W["t"] = np.arange(len(W["x"])) * dt
    meta = copy.deepcopy(meta0)
    meta["T"] += PREROLL
    for s_ in meta["strokes"]:
        s_["t0"] += PREROLL
        s_["t1"] += PREROLL
    n_tab = len(W["t"])
    n_ext = n_tab + int(round(tail / dt)) + 2
    tg = np.arange(n_ext) * dt
    Qw = np.concatenate([np.nan_to_num(np.asarray(W["Q"], float)), np.zeros(n_ext - n_tab)])
    Qenv = np.concatenate([np.nan_to_num(np.asarray(W["Q_env"], float)), np.zeros(n_ext - n_tab)])
    pulsed = [s for s in meta["strokes"] if s.get("pulse")]
    if npass is None:
        npass = 6 if pulsed else 3
    mode = PULSE_MODE.get(variant, "tilt")
    amax = hand["acc"] / hand["pivot_L"]
    PZ = []
    for s_ in pulsed:
        w = 2 * math.pi * s_["pulse"]["f"]
        cw = s_["pulse"]["m"] * complex(math.cos(s_["pulse"]["phi"]), math.sin(s_["pulse"]["phi"]))
        lead = complex(1.0, w * OutflowLaw(**law_kw).tau)          # the spout lag to be pre-compensated
        ct_max = 0.95 * min(amax / w ** 2, hand["tilt_rate"] / w)
        cy_max = SLOSH_ACC / w ** 2
        if mode == "tilt":
            C = cw * lead / 21.0                                       # dlnQ/dtilt ~ 0.37 /deg = 21 /rad
            C *= min(1.0, ct_max / abs(C))
            cmax = ct_max
        else:
            C = -1.2e-3 * lead / abs(lead) * cw / abs(cw)              # head at the lip ~ -displacement above resonance
            cmax = cy_max
        PZ.append(dict(n=s_["n"], t0=s_["t0"], t1=s_["t1"], w=w, cw=cw, C=C, m=1.0, mode=mode, c_max=cmax))
    yaw_ext = np.concatenate([np.asarray(W["yaw"], float), np.full(n_ext - n_tab, W["yaw"][-1])])
    yaw_plan = smooth_yaw(yaw_ext, dt, hand["yaw_rate"])
    nsteps = n_ext - 2
    name = best_name(pattern, variant, fill)
    tp = time.time()
    Rmax = lift_preplan(W, meta, fill, law_kw, n_ext)
    Lplan = lift_plan(Rmax, dt, hand["speed"])
    hist = []
    if verbose:
        print(f"  {name} lift pre-plan: max {np.max(Rmax) * 1e3:.1f} mm [{time.time() - tp:.0f}s]", flush=True)
    for k in range(npass):
        tp = time.time()
        Qref = Qw
        P = FastPitcher(geom=nx_geometry(6), V0=fill * ML, law=OutflowLaw(**law_kw))
        H = IdealHand(P, W, meta, Qref, n_ext, hand=hand, lift_plan=Lplan, yaw_plan=yaw_plan, pulse=PZ, Q_env=Qenv)
        A = {key: np.full(nsteps, np.nan) for key in REC_KEYS + XKEYS}
        for i in range(nsteps):
            rec = H.step(dt)
            _flatten(rec, H, A, i)
            for xk in XKEYS:
                A[xk][i] = float(rec[xk])
        Qd = np.interp(tg, A["t"] - 0.5 * dt, A["Q_hit"], left=0.0, right=0.0)
        e = Qw - Qd
        R = np.concatenate([A["x_R"], np.full(n_ext - nsteps, A["x_R"][-1])])
        viol = float(np.sum(A["x_R"] > A["lift"] + 5e-4) * dt)
        strk = []
        for s_ in meta["strokes"]:
            if "yaw" in s_:
                m_ = (tg >= s_["t0"]) & (tg < s_["t1"])
                strk.append([s_["n"], float(np.sum(Qd[m_]) / max(np.sum(Qw[m_]), 1e-30)),
                             float(np.sum(np.abs(e[m_])) / max(np.sum(Qw[m_]), 1e-30))])
        pl = []
        for q in PZ:
            f = q["w"] / (2 * math.pi)
            a0 = q["t0"] + 0.4
            nper = int(math.floor((q["t1"] - 0.05 - a0) * f))
            if nper < 2:
                continue
            mp = (tg >= a0) & (tg < a0 + nper / f)
            ph = np.exp(-1j * q["w"] * (tg[mp] - q["t0"]))
            cd = 2 * np.mean(Qd[mp] * ph) / max(np.mean(Qd[mp]), 1e-30)
            cw = 2 * np.mean(Qw[mp] * ph) / max(np.mean(Qw[mp]), 1e-30)
            rm = np.mean(Qw[mp]) / max(np.mean(Qd[mp]), 1e-30)
            osc = A["osc"] if mode == "tilt" else A["x_oscy"]
            mo = (A["t"] - 0.5 * dt >= a0 - 0.1) & (A["t"] - 0.5 * dt < a0 + nper / f - 0.1)
            oa = float(0.5 * (np.percentile(osc[mo], 98) - np.percentile(osc[mo], 2))) if mo.any() else 0.0
            pl.append(dict(n=q["n"], m_d=float(abs(cd)), m_w=float(abs(cw)),
                           dphi_deg=float(_wrap_deg(math.degrees(np.angle(cd) - np.angle(cw)))),
                           mean_ratio=float(1.0 / rm),
                           osc_amp=(math.degrees(oa) if mode == "tilt" else oa * 1e3), osc_unit=("deg" if mode == "tilt" else "mm"),
                           C=[float(abs(q["C"])), float(math.degrees(np.angle(q["C"])))], m_used=float(q["m"]),
                           osc_max=(math.degrees(q["c_max"]) if mode == "tilt" else q["c_max"] * 1e3)))
            if k < npass - 1 and abs(cd) > 1e-3:
                r = cw / cd
                r *= min(1.0, PULSE_RCLIP / abs(r)) * max(1.0, 1.0 / (PULSE_RCLIP * abs(r)))
                r = abs(r) ** PULSE_GAIN * complex(math.cos(PULSE_GAIN * np.angle(r)), math.sin(PULSE_GAIN * np.angle(r)))
                q["C"] = q["C"] * r
                q["m"] = float(np.clip(q["m"] * rm ** PULSE_GAIN, 0.5, 2.0))
        hist.append(dict(k=k, L1=float(np.sum(np.abs(e)) / max(np.sum(Qw), 1e-30)), strokes=strk, pulse=pl,
                         V_w_mL=float(np.sum(Qw) * dt / ML), V_d_mL=float(np.sum(Qd) * dt / ML),
                         lift_max_mm=float(np.nanmax(A["lift"]) * 1e3), lift_violation_s=viol,
                         tilt_rate_max_deg_s=float(np.degrees(np.nanmax(np.abs(A["x_rate"])))),
                         evals=H.n_eval, wall_s=time.time() - tp))
        if verbose:
            print(f"  {name} pass {k}: L1 {hist[-1]['L1'] * 100:.1f} %, V {hist[-1]['V_d_mL']:.1f}/{hist[-1]['V_w_mL']:.1f} mL,"
                  f" lift max {hist[-1]['lift_max_mm']:.1f} mm, viol {viol:.3f} s [{time.time() - tp:.0f}s]  strokes "
                  + " ".join(f"#{a_}:{b_:.2f}/{c_ * 100:.0f}%" for a_, b_, c_ in strk)
                  + ("  pulse " + " ".join(f"#{x['n']}:{x['m_d']:.2f}/{x['m_w']:.2f},{x['dphi_deg']:+.0f}deg,mean{x['mean_ratio']:.2f},"
                                           f"osc{x['osc_amp']:.1f}/{x['osc_max']:.1f}{x['osc_unit']}"
                                           for x in pl) if pl else ""), flush=True)
        if k < npass - 1:
            Rmax = np.maximum(Rmax, R)
            Lplan = lift_plan(Rmax, dt, hand["speed"])
    V_pre = float(np.sum(A["Q_hit"][:n_pre]) * dt / ML)            # arrived before the FS clock starts (dropped)
    A = {k: v[n_pre:].copy() for k, v in A.items()}
    A["t"] = A["t"] - PREROLL
    W, meta = W0, meta0
    info = dict(run=name, set="fs", pattern=pattern, variant=variant, fill_mL=fill, dt=dt, nr=6, steps=nsteps - n_pre,
                preroll_s=PREROLL, V_preroll_mL=V_pre,
                wall_s=time.time() - t0w, n_pass=npass, ilc=hist, pass_used=npass - 1, hand=hand,
                hand_src=HAND_SRC, lift_acc=LIFT_ACC,
                pulse_mode=mode, pulse_final={str(q["n"]): dict(C=[abs(q["C"]), math.degrees(np.angle(q["C"]))],
                                                               C_unit=("rad at 5 mm head" if mode == "tilt" else "m at 5 mm head"),
                                                               m=q["m"], c_max=q["c_max"]) for q in PZ},
                law={k: getattr(P.law, k) for k in ("C_d", "h_k", "h_on", "h_off", "tau", "tau_stop", "zeta", "slosh",
                                                    "carry_velocity")},
                cup={k: getattr(H.cup, k) for k in ("r_in", "r_out", "height", "rim_z", "tilt", "margin")},
                tilt0_deg=math.degrees(H.tilt0), yaw_plan_rate_max_deg_s=float(np.degrees(np.max(np.abs(np.diff(yaw_plan))) / dt)),
                yaw_want_rate_max_deg_s=float(np.degrees(np.max(np.abs(np.diff(yaw_ext))) / dt)))
    return W, meta, A, info


def flow_err_smoothed(W, meta, A, info, ms=20):
    """Per stroke: integral |Q_hit - Q_FS| / integral Q_FS after a `ms` moving average of both (separates timing and
    volume from 1 ms arrival jitter and 5 Hz waveform detail)."""
    dt = info["dt"]
    t = A["t"] - 0.5 * dt
    k = max(1, int(round(ms * 1e-3 / dt)))
    Qd = _movavg(np.nan_to_num(A["Q_hit"]), k)
    Qw = _movavg(_wint(W, "Q", t, dt), k)
    out = {}
    for s in meta["strokes"]:
        if "yaw" not in s:
            continue
        m = (t >= s["t0"]) & (t < s["t1"])
        out[s["n"]] = float(np.sum(np.abs(Qd[m] - Qw[m])) / max(np.sum(Qw[m]), 1e-30))
    return out


def best_extra(name, out=BEST):
    """Metrics the Track analysis does not have: cup-frame u_h direction / vector error, pulse waveform error, hand
    kinematics against HAND, clearance violations, ILC history."""
    W, meta, A, info = load_run(name, out)
    dt = info["dt"]
    t = A["t"] - 0.5 * dt
    hit = A["h_on"] > 0
    Qd = np.nan_to_num(A["Q_hit"])
    Qw = _wint(W, "Q", t, dt)
    uxw, uyw = _wint(W, "uhx", t, dt), _wint(W, "uhy", t, dt)
    uxd, uyd = np.where(hit, A["h_uhx"], np.nan), np.where(hit, A["h_uhy"], np.nan)
    ang = np.degrees(np.abs((np.arctan2(uyd, uxd) - np.arctan2(uyw, uxw) + np.pi) % (2 * np.pi) - np.pi))
    dvec = np.hypot(uxd - uxw, uyd - uyw)
    hand = info.get("hand", HAND)
    tilt = A["tilt"]
    rate = np.gradient(tilt, dt)
    acc_t = np.gradient(_movavg(rate, 5), dt)
    yaw = A["yaw"]
    yr = np.gradient(yaw, dt)
    strokes = []
    for s in meta["strokes"]:
        if "yaw" not in s:
            continue
        m = (t >= s["t0"]) & (t < s["t1"])
        w = np.where(m, Qd, 0.0)
        o = dict(n=s["n"], cls=s["cls"], uh_dir_err_deg=_fw(ang, w), uh_dir_err_max_deg=float(np.nanmax(np.where(m & hit & (Qd > 1 * ML), ang, np.nan))) if (m & hit & (Qd > 1 * ML)).any() else None,
                 uh_vec_err=_fw(dvec, w), uh_w=_fw(np.hypot(uxw, uyw), w),
                 tilt_rate_max=float(np.degrees(np.max(np.abs(rate[m])))),
                 tilt_acc_max=float(np.degrees(np.max(np.abs(acc_t[m])))),
                 pivot_acc_max=float(hand["pivot_L"] * np.max(np.abs(acc_t[m]))),
                 yaw_rate_max=float(np.degrees(np.max(np.abs(yr[m])))))
        if s.get("pulse"):
            a0 = s["t0"] + 0.4
            f = s["pulse"]["f"]
            nper = int(math.floor((s["t1"] - 0.05 - a0) * f))
            if nper >= 2:
                mp = (t >= a0) & (t < a0 + nper / f)
                # waveform error after removing the cycle mean ratio (shape only), and the 2nd harmonic
                qd, qw = Qd[mp], Qw[mp]
                o["pulse_wave_L1"] = float(np.sum(np.abs(qd / qd.mean() - qw / qw.mean())) / len(qd)) if qd.mean() > 0 else None
                o["pulse_m2_d"] = lockin(t[mp], qd, 2 * f, s["t0"])[0]
                o["pulse_m2_w"] = lockin(t[mp], qw, 2 * f, s["t0"])[0]
                o["pulse_min_rel"] = float(qd.min() / qd.mean()) if qd.mean() > 0 else None
                th = np.degrees(tilt[mp])
                o["pulse_tilt_amp_deg"] = float(0.5 * (np.percentile(th, 98) - np.percentile(th, 2)))
                o["pulse_hold_frac"] = float(np.mean(A["x_mode"][mp] == 0))
        strokes.append(o)
    fe20 = flow_err_smoothed(W, meta, A, info, 20)
    for o in strokes:
        o["Q_err_rel_s20"] = fe20.get(o["n"])
    mT = t <= meta["T"]
    tot = dict(tilt_rate_max=float(np.degrees(np.max(np.abs(rate[mT])))),
               tilt_acc_max=float(np.degrees(np.max(np.abs(acc_t[mT])))),
               pivot_acc_max=float(hand["pivot_L"] * np.max(np.abs(acc_t[mT]))),
               tip_speed_max=float(np.nanmax(np.hypot(A["vel_x"], np.hypot(A["vel_y"], A["vel_z"]))[mT])),
               tip_acc_max=float(np.nanmax(np.hypot(A["acc_x"], np.hypot(A["acc_y"], A["acc_z"]))[mT])),
               yaw_rate_max=float(np.degrees(np.max(np.abs(yr[mT])))),
               lift_violation_s=float(np.sum((A["x_R"] > A["lift"] + 5e-4)[mT]) * dt),
               lift_violation_max_mm=float(np.max((A["x_R"] - A["lift"])[mT]) * 1e3),
               hold_s=float(np.sum(A["x_mode"][mT] == 0) * dt), ilc=info.get("ilc"))
    res = dict(run=name, strokes=strokes, total=tot)
    with open(os.path.join(out, "runs", name + "_extra.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


SENS_PATTERNS = ("swan", "heart")
SENS_VARIANTS = ("Cd055", "Cd065", "tau004", "tau015", "hon125", "hon375")
BEST_FILLS = (350, 250)


def best_job_list(which="main"):
    J = []
    if which in ("main", "all"):
        for fl in BEST_FILLS:
            for p in PATTERNS:
                J.append((p, "ideal", fl))
        for p in ("layered_heart", "tulip"):
            J.append((p, "idealslosh", 350))
    if which in ("sens", "all"):
        for p in SENS_PATTERNS:
            for v in SENS_VARIANTS:
                J.append((p, v, 350))
    return J


def _cls_cell(R, q):
    c, n = _worst(R["strokes"], q)
    cl = [o["class"].get(q, "n/a") for o in R["strokes"]]
    cl = [x for x in cl if x != "n/a"]
    bad = sum(1 for x in cl if not x.startswith("DELIVERED"))
    if c == "n/a":
        return "–"
    return (_short(c) + (f"（#{n}）" if n is not None and not c.startswith("DELIVERED") else "") +
            f" [{bad}/{len(cl)}]").replace("|", "\\|")


def _load_metrics(name, out):
    path = os.path.join(out, "runs", name + "_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        R = json.load(f)
    px = os.path.join(out, "runs", name + "_extra.json")
    if os.path.exists(px):
        with open(px) as f:
            R["extra"] = json.load(f)
    return R


def best_report(out=BEST):
    """best/summary_tables.md and best/summary.json: Track (stock / 'main' barista) vs the ideal hand, per pattern and
    per stroke; bridges; pulsation; hand kinematics; static reachability; sensitivity to the pitcher constants."""
    L = []
    js = dict(hand=HAND, hand_src=HAND_SRC, lift_acc=LIFT_ACC, criteria=dict(rel=TOL_REL, mm=TOL_MM, deg=TOL_DEG),
              runs={})
    L.append("# 理想手核对：FS 入口量，是咖啡师控制器做不到，还是壶本身做不到\n")
    L.append("> 由 `latte_imex/v05_pitcher_check.py best_report` 生成。Track 的结果来自 `v05pitcher/track/`（main = 原控制器 + 两处数值修正）；"
             "理想手的运行在 `v05pitcher/best/runs/`，1 ms 入口表在 `best/tables/`，图 `best/<运行名>.png`。\n")
    L.append("> 判定与 Track 相同：✓ = 相对差 ≤ 15 %、位置/高度 ≤ 1 mm、相位 ≤ 10°；≠ 给出差多少；✗ = 壶够不到（流量加权碰撞抬升 > 1 mm）或没有奶柱。"
             "格内是最差一笔（笔号）[没送到的笔数/总笔数]。\n")
    L.append("## 理想手的人手上限\n")
    for k, v in HAND_SRC.items():
        L.append(f"- `{k}` = {HAND[k] if k in ('pivot_L', 'acc', 'speed', 'f_max') else round(math.degrees(HAND[k]))}：{v}")
    L.append(f"- 抬壶规划加速度 {LIFT_ACC} m/s²（低于 10 m/s² 上限；理想手选择更柔的抬壶以免晃奶）\n")
    blocks = [("main", 350, OUT, "Track main（原控制器），装奶 350 mL"),
              ("ideal", 350, out, "理想手，装奶 350 mL"),
              ("ideal", 250, out, "理想手，装奶 250 mL")]
    allR = {}
    for var, fill, od, title in blocks:
        rows = []
        for p_ in PATTERNS:
            nm = run_name("fs", p_, var, fill)
            R = _load_metrics(nm, od)
            if R is None:
                continue
            allR[(var, fill, p_)] = R
            rows.append((p_, R))
        if not rows:
            continue
        L.append(f"\n## {title}\n")
        L.append("| 图案 | " + " | ".join(QLABEL[q] for q in QUANT) + " |")
        L.append("|---|" + "---|" * len(QUANT))
        for p_, R in rows:
            L.append(f"| {CN[p_]} | " + " | ".join(_cls_cell(R, q) for q in QUANT) + " |")
    # item counts over all FS strokes
    L.append("\n## 逐项计数：送到的笔数 / 有这一项的笔数（六个图案合计）\n")
    L.append("| 项目 | Track main 350 | 理想手 350 | 理想手 250 |")
    L.append("|---|---|---|---|")
    cnt = {}
    for q in QUANT:
        cells = []
        for var, fill in (("main", 350), ("ideal", 350), ("ideal", 250)):
            ok = tot = 0
            for p_ in PATTERNS:
                R = allR.get((var, fill, p_))
                if R is None:
                    continue
                for o in R["strokes"]:
                    c = o["class"].get(q, "n/a")
                    if c == "n/a":
                        continue
                    tot += 1
                    ok += c.startswith("DELIVERED")
            cells.append(f"{ok}/{tot}" if tot else "–")
            cnt[f"{q}/{var}{fill}"] = [ok, tot]
        L.append(f"| {QLABEL[q]} | " + " | ".join(cells) + " |")
    js["item_counts"] = cnt
    # per-stroke comparison of the key numbers
    s20 = {}
    L.append("\n## 逐笔：Track main → 理想手 350 mL → 理想手 250 mL\n")
    L.append("列：笔内 ∫|ΔQ|/∫Q；起流迟到 ms；5 Hz 脉动 m（要 0.41）与相位差；u0 m/s（FS 0.35）；出口水平速度 m/s（FS 要的在括号里）；"
             "杯面 |u_h| 相对差；u_h 方向误差 °（流量加权）；U⊥ 相对差；足迹 r 相对差；落点 rms mm；流量加权抬升 mm（最大）。\n")
    for p_ in PATTERNS:
        keys = [(v, f) for v, f in (("main", 350), ("ideal", 350), ("ideal", 250)) if (v, f, p_) in allR]
        if not keys:
            continue
        L.append(f"\n### {CN[p_]}（{p_}）\n")
        L.append("| 笔 | 运行 | ∫\\|ΔQ\\|（20 ms 平滑后） | 奶量比 | 起流迟 ms | 脉动 m, Δφ | u0 | 出口水平 (要) | \\|u_h\\| | u_h 方向 ° | U⊥ | r | 落点 rms mm | 抬升 mm |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        base = allR[keys[0] + (p_,)]
        for o0 in base["strokes"]:
            for v, f in keys:
                R = allR[(v, f, p_)]
                o = next((x for x in R["strokes"] if x["n"] == o0["n"]), None)
                if o is None:
                    continue
                ex = next((x for x in R.get("extra", {}).get("strokes", []) if x["n"] == o0["n"]), {})
                if v == "main":
                    if (v, f, p_) not in s20:
                        W_, me_, A_, in_ = load_run(run_name("fs", p_, v, f), OUT)
                        s20[(v, f, p_)] = flow_err_smoothed(W_, me_, A_, in_, 20)
                    e20 = s20[(v, f, p_)].get(o["n"])
                else:
                    e20 = ex.get("Q_err_rel_s20")
                pul = "–" if "pulse_m_d" not in o else f"{o['pulse_m_d']:.2f}, {o['pulse_dphase_deg']:+.0f}°"
                sd = o.get("start_delay_s")
                L.append(f"| {o['name']} | {v} {f} | {o['Q_err_rel'] * 100:.0f} %（{_f(e20 * 100 if e20 is not None else None, '{:.0f}')} %） | {o['Q_ratio']:.2f} | "
                         f"{_f(sd * 1e3 if sd is not None and np.isfinite(sd) else sd, '{:.0f}')} | {pul} | {_f(o['u0'])} | "
                         f"{_f(o['exit_h_d'], '{:.3f}')} ({_f(o['exit_h_w'], '{:.3f}')}) | {o['uh_rel'] * 100:+.0f} % | "
                         f"{_f(ex.get('uh_dir_err_deg'), '{:.0f}')} | {o['U_rel'] * 100:+.0f} % | {o['r_rel'] * 100:+.0f} % | "
                         f"{_f(o.get('foot_rms_mm'), '{:.1f}')} | {o['lift_fw_mm']:.1f} ({o['lift_max_mm']:.1f}) |")
    # bridges
    L.append("\n## 桥与停：漏奶 mL / 杯面 < 1 mL/s 的时长 s（FS 要的 → Track main → 理想手 350 → 理想手 250）\n")
    L.append("| 图案 | 桥 | FS | Track main | 理想手 350 | 理想手 250 |")
    L.append("|---|---|---|---|---|---|")
    for p_ in PATTERNS:
        keys = [(v, f) for v, f in (("main", 350), ("ideal", 350), ("ideal", 250)) if (v, f, p_) in allR]
        if not keys:
            continue
        for g0 in allR[keys[0] + (p_,)]["gaps"]:
            cells = [f"{g0['V_w_mL']:.2f} / {g0['zero_w_s']:.2f}"]
            for v, f in (("main", 350), ("ideal", 350), ("ideal", 250)):
                R = allR.get((v, f, p_))
                g = next((x for x in R["gaps"] if x["n"] == g0["n"]), None) if R else None
                cells.append("–" if g is None else f"{g['V_d_mL']:.2f} / {g['zero_d_s']:.2f}")
            L.append(f"| {CN[p_]} | {g0['name']}（{g0['t1'] - g0['t0']:.2f} s） | " + " | ".join(cells) + " |")
    # hand kinematics
    L.append("\n## 理想手的动作量（全程最大）\n")
    L.append("| 运行 | 倾角速率 °/s | 倾角角加速度 °/s² | 手（15 cm 力臂）加速度 m/s² | 壶嘴尖速度 m/s | 壶嘴尖加速度 m/s²（模型 20 ms 低通） | 偏航速率 °/s（要的 → 做的） | 抬升不足时长 s（最大 mm） | 用奶 mL | 墙钟 s | 用的 pass |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    names = sorted(f[:-13] for f in os.listdir(os.path.join(out, "runs")) if f.endswith("_metrics.json"))
    for nm in names:
        R = _load_metrics(nm, out)
        x = R.get("extra", {}).get("total", {})
        inf = R["info"]
        js["runs"][nm] = dict(info=inf, total=R["total"], extra=x, classes={str(o["n"]): o["class"] for o in R["strokes"]},
                              worst={q: _worst(R["strokes"], q) for q in QUANT})
        if not x:
            continue
        L.append(f"| {nm} | {x['tilt_rate_max']:.0f} | {x['tilt_acc_max']:.0f} | {x['pivot_acc_max']:.1f} | {x['tip_speed_max']:.2f} | "
                 f"{x['tip_acc_max']:.1f} | {inf.get('yaw_want_rate_max_deg_s', 0):.0f} → {x['yaw_rate_max']:.0f} | "
                 f"{x['lift_violation_s']:.3f} ({x['lift_violation_max_mm']:.1f}) | {R['total']['milk_used_mL']:.0f} | "
                 f"{inf['wall_s']:.0f} | {inf.get('pass_used')} |")
    # static reachability
    sp = os.path.join(out, "static.json")
    if os.path.exists(sp):
        with open(sp) as f:
            st = json.load(f)
        fills = st["fills"]
        L.append("\n## 静态可达：每笔要的壶嘴高度能不能到（`best_static`，每笔 6 个采样点的平均抬升 mm）\n")
        L.append("FS 偏航、滚转 0 / FS 偏航、最佳滚转（±20°）/ 任意偏航 + 最佳滚转；装奶 " + " / ".join(str(x) for x in fills) +
                 " mL（按要的奶量逐点扣减）。出口速度 u0 与出口水平速度取 FS 偏航、滚转 0（u0 括号内是所有滚转的范围）。\n")
        L.append("| 图案 | 笔 | 要的 z cm | Q mL/s | 抬升：FS 偏航 | 抬升：FS 偏航+滚转 | 抬升：任意偏航+滚转 | 350 mL 最佳偏航相对 FS | 倾角 °（350） | u0 m/s（350；所有滚转） | 出口水平 m/s：" + " / ".join(str(x) for x in fills) + " | 要的出口水平 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p_, rows in st["res"].items():
            for r in rows:
                F = r["fills"]
                a = " / ".join(f"{F[str(x)]['lift_fsyaw_roll0_mm'][0]:.1f}" for x in fills)
                b = " / ".join(f"{F[str(x)]['lift_fsyaw_bestroll_mm'][0]:.1f}" for x in fills)
                c = " / ".join(f"{F[str(x)]['lift_bestyaw_bestroll_mm'][0]:.1f}" for x in fills)
                dy = float(np.median([y["dyaw"] for y in F[str(fills[0])]["best"]]))
                ex = " / ".join(f"{F[str(x)]['exit_h_roll0'][0]:.2f}–{F[str(x)]['exit_h_roll0'][1]:.2f}" for x in fills)
                f0 = F[str(fills[0])]
                L.append(f"| {CN[p_]} | #{r['n']} {r['cls']} | {r['z_w_cm']:.2f} | {r['Q_mLs']:.1f} | {a} | {b} | {c} | {dy:+.0f}° | "
                         f"{f0['tilt_deg'][0]:.0f}–{f0['tilt_deg'][1]:.0f} | {f0['u0_roll0'][0]:.2f}–{f0['u0_roll0'][1]:.2f}"
                         f"（{f0['u0_allroll'][0]:.2f}–{f0['u0_allroll'][1]:.2f}） | {ex} | {r['exit_h_w']:.3f} |")
        js["static"] = {p_: [{k: v for k, v in r.items() if k != "samples"} for r in rows] for p_, rows in st["res"].items()}
    # sensitivity
    sens = [(p_, v) for p_ in SENS_PATTERNS for v in SENS_VARIANTS]
    have = [(p_, v) for p_, v in sens if _load_metrics(run_name("fs", p_, v, 350), out) is not None]
    if have:
        L.append("\n## 敏感性：未标定的壶常数（理想手，装奶 350 mL）\n")
        L.append("| 图案 | 常数 | 全程 ∫\\|ΔQ\\|/∫Q | 最差笔 ∫\\|ΔQ\\| | 脉动 m（相位差） | 起流迟到 ms（最大） | 桥漏奶 mL（合计）/ 断流 s（最短） | u0 范围 | 出口水平 范围 | 流量加权抬升 mm（最大笔） | 倾角范围 ° |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for p_ in SENS_PATTERNS:
            for v in ("ideal",) + SENS_VARIANTS:
                R = _load_metrics(run_name("fs", p_, v, 350), out)
                if R is None:
                    continue
                st_ = R["strokes"]
                worstQ = max(st_, key=lambda o: o["Q_err_rel"])
                pul = [o for o in st_ if "pulse_m_d" in o]
                pc = ", ".join(f"#{o['n']} {o['pulse_m_d']:.2f}（{o['pulse_dphase_deg']:+.0f}°）" for o in pul) or "–"
                sds = [o.get("start_delay_s") for o in st_ if o.get("start_delay_s") is not None and np.isfinite(o.get("start_delay_s"))]
                gl = R["gaps"]
                br = [g for g in gl if "bridge" in g["name"]]
                brs = f"{sum(g['V_d_mL'] for g in br):.2f}（要 {sum(g['V_w_mL'] for g in br):.2f}）/ {min((g['zero_d_s'] for g in br), default=float('nan')):.2f}" if br else "–"
                u0s = [o["u0"] for o in st_ if np.isfinite(o["u0"])]
                exs = [o["exit_h_d"] for o in st_ if np.isfinite(o["exit_h_d"])]
                lw = max(st_, key=lambda o: o["lift_fw_mm"])
                trs = R["total"]["tilt_range_deg"]
                lab = dict(ideal="默认（C_d 0.60, τ 0.08/0.15, h_on 2.5 mm）", Cd055="C_d 0.55", Cd065="C_d 0.65",
                           tau004="τ 0.04（τ_stop 0.075）", tau015="τ 0.15（τ_stop 0.28）", hon125="h_on 1.25 mm",
                           hon375="h_on 3.75 mm")[v]
                L.append(f"| {CN[p_]} | {lab} | {R['total']['Q_err_rel'] * 100:.0f} % | {worstQ['name']} {worstQ['Q_err_rel'] * 100:.0f} % | {pc} | "
                         f"{max(sds) * 1e3 if sds else float('nan'):.0f} | {brs} | {min(u0s):.2f}–{max(u0s):.2f} | "
                         f"{min(exs):.3f}–{max(exs):.3f} | {lw['name']} {lw['lift_fw_mm']:.1f} | {trs[0]:.0f}–{trs[1]:.0f} |")
                js.setdefault("sensitivity", {})[f"{p_}/{v}"] = dict(total=R["total"], strokes=st_, gaps=gl)
    with open(os.path.join(out, "summary_tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(js, f, indent=1, default=float)
    return js


# --------------------------------------------------------------------------- best case: static reachability
def _reach(P, cup, foot, z, yaw, tilt, roll, V, Q):
    """Lowest spout height for this pose with the stream foot at `foot`: required_lift at the wanted height z with the
    tip placed so that the jet lands on the foot; returns (z_min, jet, tip)."""
    foot = np.asarray(foot, float)
    tip = foot.copy()
    J = None
    zz = z
    L = 0.0
    for it in range(4):
        pose = Pose((tip[0], tip[1], zz), yaw, tilt, roll)
        c, _ = P.free_surface(pose, V)
        Wr = P.weir(pose, c)
        J = P.jet(pose, Q, max(Wr["h_bar"], P.law.h_k), Wr["origin"], 0.0, Wr["width"], Wr["tangent"])
        tip = tip + (foot - J["x_hit"])
        L = P.required_lift(Pose((tip[0], tip[1], zz), yaw, tilt, roll), cup)
        if L <= 0 or it == 3:
            break
        zz = zz + L
    return zz + L, J, tip


def best_static(patterns=PATTERNS, fills=(350, 300, 250, 200), rolls=(-20, -10, 0, 10, 20),
                yaws=tuple(range(0, 360, 30)), nsamp=6, out=BEST, verbose=True):
    g = nx_geometry(6)
    cup = Cup()
    res = {}
    os.makedirs(out, exist_ok=True)
    for pat in patterns:
        t0 = time.time()
        W, meta = fs_wanted(pat)
        dt = meta["dt"]
        Qw = np.nan_to_num(np.asarray(W["Q"], float))
        used = np.cumsum(Qw) * dt
        Qs = np.nan_to_num(np.asarray(W["Q_env"], float))
        rows = []
        for s in meta["strokes"]:
            if "yaw" not in s:
                continue
            smps = []
            for j in range(nsamp):
                ti = s["t0"] + (j + 0.5) / nsamp * (s["t1"] - s["t0"])
                i = min(int(round(ti / dt)), len(Qw) - 1)
                Q = max(Qs[i], 1.0 * ML)
                foot = (float(W["x"][i]) * D, float(W["y"][i]) * D)
                z = float(W["z"][i])
                yfs = float(W["yaw"][i])
                smp = dict(t=ti, Q_mLs=Q / ML, z_w_cm=z * 100, foot_mm=[foot[0] * 1e3, foot[1] * 1e3],
                           yaw_fs_deg=math.degrees(yfs), exit_h_w=float(W["exit_h"][i]), fills={})
                for F in fills:
                    V = F * ML - used[i]
                    P = FastPitcher(geom=g, V0=V)
                    fr = {}
                    for ro in rolls:
                        rr = math.radians(ro)
                        tilt = P.tilt_for_flow(Pose((0.0, 0.0, z), 0.0, 0.0, rr), Q, V=V, iters=26)
                        zmin_fs, J, _ = _reach(P, cup, foot, z, yfs, tilt, rr, V, Q)
                        v0 = np.asarray(J["v0"], float)
                        head = math.degrees((math.atan2(v0[1], v0[0]) - yfs + math.pi) % (2 * math.pi) - math.pi)
                        ent = dict(tilt_deg=math.degrees(tilt), u0=J["u0"], exit_h=float(math.hypot(v0[0], v0[1])),
                                   exit_head_dev_deg=head, U_perp=J["U_perp"], z_min_fs_cm=zmin_fs * 100, yaw={})
                        for yd in yaws:
                            zm, _, _ = _reach(P, cup, foot, z, math.radians(yd), tilt, rr, V, Q)
                            ent["yaw"][yd] = zm * 100
                        fr[ro] = ent
                    smp["fills"][F] = fr
                smps.append(smp)
            # stroke summary
            sm = dict(n=s["n"], cls=s["cls"], t0=s["t0"], t1=s["t1"], z_w_cm=float(np.mean([x["z_w_cm"] for x in smps])),
                      Q_mLs=float(np.mean([x["Q_mLs"] for x in smps])), yaw_fs_deg=smps[0]["yaw_fs_deg"], fills={})
            for F in fills:
                def need(sel):
                    return [max(0.0, sel(x) - x["z_w_cm"]) * 10 for x in smps]       # mm
                a = need(lambda x: x["fills"][F][0]["z_min_fs_cm"])
                b = need(lambda x: min(x["fills"][F][r]["z_min_fs_cm"] for r in rolls))
                # any yaw = the 30 deg grid plus the FS yaw itself (z_min_fs_cm); recomputed from the stored samples of
                # the published static.json this adds nothing (no stroke / fill changes by > 0.05 mm)
                def anyyaw(e):
                    return min(min(e["yaw"].values()), e["z_min_fs_cm"])
                c = need(lambda x: min(anyyaw(x["fills"][F][r]) for r in rolls))
                c2 = need(lambda x: min(anyyaw(x["fills"][F][r]) for r in rolls if abs(r) <= 10))
                best_yaw = []
                for x in smps:
                    bb = min(((x["fills"][F][r]["yaw"][y], y, r) for r in rolls for y in yaws))
                    best_yaw.append(dict(z_cm=bb[0], yaw=bb[1], roll=bb[2], dyaw=float((bb[1] - x["yaw_fs_deg"] + 180) % 360 - 180)))
                ex = [x["fills"][F][r] for x in smps for r in rolls]
                ex0 = [x["fills"][F][0] for x in smps]
                sm["fills"][F] = dict(
                    lift_fsyaw_roll0_mm=[float(np.mean(a)), float(np.max(a))],
                    lift_fsyaw_bestroll_mm=[float(np.mean(b)), float(np.max(b))],
                    lift_bestyaw_bestroll_mm=[float(np.mean(c)), float(np.max(c))],
                    lift_bestyaw_roll10_mm=[float(np.mean(c2)), float(np.max(c2))],
                    best=best_yaw,
                    z_min_fs_cm=[float(min(x["fills"][F][0]["z_min_fs_cm"] for x in smps)),
                                 float(max(x["fills"][F][0]["z_min_fs_cm"] for x in smps))],
                    tilt_deg=[float(min(e["tilt_deg"] for e in ex0)), float(max(e["tilt_deg"] for e in ex0))],
                    u0_roll0=[float(min(e["u0"] for e in ex0)), float(max(e["u0"] for e in ex0))],
                    u0_allroll=[float(min(e["u0"] for e in ex)), float(max(e["u0"] for e in ex))],
                    exit_h_roll0=[float(min(e["exit_h"] for e in ex0)), float(max(e["exit_h"] for e in ex0))],
                    exit_h_allroll=[float(min(e["exit_h"] for e in ex)), float(max(e["exit_h"] for e in ex))],
                    head_dev_allroll=[float(min(e["exit_head_dev_deg"] for e in ex)),
                                      float(max(e["exit_head_dev_deg"] for e in ex))])
            sm["exit_h_w"] = float(np.mean([x["exit_h_w"] for x in smps]))
            sm["samples"] = smps
            rows.append(sm)
            if verbose:
                f3 = sm["fills"][fills[0]]
                print(f"  {pat} #{s['n']} {s['cls']}: z_w {sm['z_w_cm']:.2f} cm, Q {sm['Q_mLs']:.1f}: lift FS yaw "
                      f"{f3['lift_fsyaw_roll0_mm'][0]:.1f} mm, best roll {f3['lift_fsyaw_bestroll_mm'][0]:.1f}, best yaw+roll "
                      f"{f3['lift_bestyaw_bestroll_mm'][0]:.1f}; " + ", ".join(
                          f"{F}: {sm['fills'][F]['lift_bestyaw_bestroll_mm'][0]:.1f}" for F in fills[1:]) +
                      f" [{time.time() - t0:.0f}s]", flush=True)
        res[pat] = rows
        with open(os.path.join(out, "static.json"), "w") as f:
            json.dump(dict(fills=fills, rolls=rolls, yaws=yaws, nsamp=nsamp, cup=Cup().__dict__, res=res), f, indent=1,
                      default=float)
    return res


# --------------------------------------------------------------------------- CLI
# --------------------------------------------------------------------------- SOLVE: V0.5 with the pitcher-delivered inlet
# The FS run (v05allphys/final/runs/FS_<pattern>, drvf.py flags --exp P2 --pulse 0.41 --lat exitmin --lag 0.08 --ztr,
# FS closure c_p 1.36, beta 3, nu 1e-3, D 1e-7, kappa_Q 2, kappa_c 1, kappa_t 0.7, kappa_r 0.037255, B_dep 1550.415,
# p_dep 8) is repeated with only the SOURCE of the inlet quantities changed.  SolveProgram is drvf.PhysF built with the
# FS flags (same phases, bridges, timing, phase labels, closure, run length T); its __call__ takes per call
#   Qraw [D^3/s], U_perp [D/s], cup-frame u_h [D/s], foot x_hit [D]
# from a source and turns them into the solver Inlet with drvf's own arithmetic (drvf_record: off if no stream; off below
# 1 mL/s in a bridge / stop; Q = clip(Qraw, 1, 32 mL/s); |u_h| <= min(0.40 m/s, U); continuity ellipse r = sqrt(Q/(pi U)),
# aspect sqrt(U^2 + v^2)/U, d = 2 sqrt(Q/(pi |v|)); S_eff = kappa_Q Q; phi = atan2(u_h)).  Sources:
#   fsexact  drvf's own quantities at the exact call time (PhysF.__call__ itself; every record is compared with drvf's)
#   wanted   the FS-wanted 1 ms table (fs_wanted: drvf sampled at t_k), held over [t_k, t_k + dt)
#   pitcher  the pitcher-delivered 1 ms arrival record (best/runs/<run>.npz, ideal hand at 350 mL; Track 'main' if no
#            ideal-hand run exists): record k = what arrives during [t_k, t_k + dt), held over that interval
#   pitcher250  as 'pitcher' with the ideal-hand run at 250 mL fill (steeper tilt: more strokes reach their height)
#   pitcher_zfs  (diagnostic, not pourable) 'pitcher' with U_perp replaced by FS's sqrt(0.35^2 + 2 g z_FS(t)) at the same t
#            (z_FS = drvf.z_at, the FS height schedule incl. --ztr transitions and bridges): what the delivered flow, foot
#            and u_h give if the spout could reach FS's heights with FS's 0.35 m/s exit speed
# Scan speed (a numerical bound on the foot speed that shortens solver frames): drvf's (the V0.5 control's) at t, raised to
# (pitcher source only; the wanted table's foot is the FS foot, bounded by drvf's scan) the table's own foot speed where
# that is faster (the delivered foot jumps at wiggle reversals and with the lift / yaw of
# the hand), the latter capped at 0.5 h / dt_table (h = 1/N) so that no frame is asked to be shorter than the 1 ms table.
SOLVE = os.path.join(REVIEW, "v05pitcher", "solve")
SOLVE_TAGS = dict(pitcher="PD", wanted="PW", fsexact="FX", pitcher_zfs="PDz", pitcher250="PD250")
_D_M, _ML_D3 = 0.08, 512.0           # drvf's constants (D_M, ML)


def solve_source_run(pattern, fill=350):
    """(directory, run name) of the pitcher run that supplies the delivered inlet."""
    nm = best_name(pattern, "ideal", fill)
    if os.path.exists(os.path.join(BEST, "runs", nm + ".npz")):
        return BEST, nm
    return OUT, run_name("fs", pattern, "main", fill)


def _foot_speed(on, x, y, dt, win=11, vmax=12.5):
    """Foot speed bound [D/s]: centred difference between streaming neighbours, running max over win samples."""
    from scipy.ndimage import maximum_filter1d
    v = np.zeros(len(x))
    ok = on[:-2] & on[1:-1] & on[2:]
    v[1:-1] = np.where(ok, np.hypot(x[2:] - x[:-2], y[2:] - y[:-2]) / (2 * dt), 0.0)
    return np.minimum(maximum_filter1d(v, win), vmax)


def solve_table(pattern, rebuild=False, fill=350):
    """solve/tables/<run>.npz: raw (no rule applied) 1 ms inlet quantities of the delivered ('p_') and the FS-wanted ('w_')
    inlet: on, x, y [D], Q [D^3/s], U [D/s], ux, uy [D/s], vfoot [D/s]."""
    d, nm = solve_source_run(pattern, fill)
    path = os.path.join(SOLVE, "tables", nm + ".npz")
    if os.path.exists(path) and not rebuild:
        return path
    z = np.load(os.path.join(d, "runs", nm + ".npz"))
    pt = z["p_t"]
    assert abs(pt[0] - DT) < 1e-9 and np.allclose(np.diff(pt), DT, atol=1e-9), "arrival record not on the 1 ms grid"
    Q = np.nan_to_num(z["p_Q_hit"])
    on = (np.nan_to_num(z["p_h_on"]) > 0) & (Q > 0)
    T = {}
    T["p_on"] = on.astype(np.int8)
    T["p_x"] = np.where(on, np.nan_to_num(z["p_h_xh"]), 0.0) / D
    T["p_y"] = np.where(on, np.nan_to_num(z["p_h_yh"]), 0.0) / D
    T["p_Q"] = np.where(on, Q, 0.0) / D3
    T["p_U"] = np.where(on, np.nan_to_num(z["p_h_U"]), 0.0) / D
    T["p_ux"] = np.where(on, np.nan_to_num(z["p_h_uhx"]), 0.0) / D
    T["p_uy"] = np.where(on, np.nan_to_num(z["p_h_uhy"]), 0.0) / D
    T["p_vfoot"] = _foot_speed(on, T["p_x"], T["p_y"], DT)
    W, meta = fs_wanted(pattern)
    kQ = meta["kappa_Q"]
    wa = np.nan_to_num(W["active"]) > 0.5
    T["w_on"] = wa.astype(np.int8)
    T["w_x"], T["w_y"] = np.asarray(W["x"], float), np.asarray(W["y"], float)
    T["w_Q"] = np.where(wa, np.nan_to_num(W["S_eff"]) / kQ, 0.0)      # = drvf's Q exactly (kappa_Q = 2)
    T["w_U"] = np.where(wa, np.nan_to_num(W["U_D"]), 0.0)
    T["w_ux"] = np.where(wa, np.nan_to_num(W["ux_D"]), 0.0)
    T["w_uy"] = np.where(wa, np.nan_to_num(W["uy_D"]), 0.0)
    T["w_vfoot"] = _foot_speed(wa, T["w_x"], T["w_y"], DT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, dt=DT, **T, meta=json.dumps(dict(
        pattern=pattern, source_run=os.path.join(d, "runs", nm + ".npz"), T=meta["T"], kappa_Q=kQ,
        units="x, y D; Q D^3/s (1 D^3 = 512 mL); U, ux, uy, vfoot D/s; p_ record k = arrivals in [k dt, (k+1) dt); "
              "w_ row k = drvf at t = k dt")))
    return path


class _SolveTable:
    def __init__(self, path, kind):
        z = np.load(path)
        pre = dict(pitcher="p_", wanted="w_", pitcher_zfs="p_", pitcher250="p_")[kind]
        self.dt = float(z["dt"])
        self.c = {k: np.asarray(z[pre + k]) for k in ("on", "x", "y", "Q", "U", "ux", "uy", "vfoot")}
        self.n = len(self.c["on"])
        self.meta = json.loads(str(z["meta"]))

    def index(self, t):
        return min(max(int(math.floor(t / self.dt + 1e-9)), 0), self.n - 1)


def drvf_record(prog, k, inl0, on, x_hit, Qraw, U, jx, jy, scan):
    """The tail of drvf.PhysF.__call__ (from 'Qraw = F / kQ'; FS flags: no --stop, no diag_*) with the quantities given
    as arguments (D, s).  Returns (Inlet, Q used or None)."""
    off = Inlet(active=False, x_hit=inl0.x_hit, S_eff=0.0, scan_speed=inl0.scan_speed, r1=inl0.r1, r2=inl0.r2)
    if not on or Qraw <= 0:
        return off, None
    if Qraw < 1.0 / _ML_D3 and (prog.stop or prog.cls[k] not in ("low", "cut")):
        return off, None
    Q = min(max(Qraw, 1.0 / _ML_D3), 32.0 / _ML_D3)
    hi = min(0.40 / _D_M, 1.0 * U)
    v = math.hypot(jx, jy)
    if v > hi:
        jx, jy, v = jx * hi / v, jy * hi / v, hi
    v = math.hypot(jx, jy)
    spd = math.hypot(U, v)
    aspect = spd / U
    r = math.sqrt(Q / (math.pi * U))
    d = 2 * math.sqrt(Q / spd / math.pi)
    return Inlet(active=True, x_hit=x_hit, S_eff=prog.kQ * Q, u_in=(jx, jy), U_perp=U, d_jet=d,
                 r1=r * math.sqrt(aspect), r2=r / math.sqrt(aspect), phi=math.atan2(jy, jx), scan_speed=scan), Q


def _inlet_same(a, b):
    if a.active != b.active:
        return False
    keys = ("x_hit", "S_eff", "scan_speed", "r1", "r2") + (("u_in", "U_perp", "d_jet", "phi") if a.active else ())
    return all(getattr(a, k) == getattr(b, k) for k in keys)


def solve_program(pattern, source, N=256):
    """drvf.PhysF with the FS flags whose inlet quantities come from `source` (fsexact | wanted | pitcher)."""
    drvf = _drvf()
    from .v05_program import Program

    class SolveProgram(drvf.PhysF):
        def __call__(self, t):
            k = self.phase_index(t)
            inl0 = Program.__call__(self, t)
            if self.source == "fsexact":
                I = drvf.PhysF.__call__(self, t)
                if I.active:
                    args = (True, I.x_hit, I.S_eff / self.kQ, I.U_perp, I.u_in[0], I.u_in[1], I.scan_speed)
                else:
                    args = (False, inl0.x_hit, 0.0, 0.0, 0.0, 0.0, inl0.scan_speed)
            else:
                c = self.tab.c
                i = self.tab.index(t)
                on = bool(c["on"][i])
                U = float(c["U"][i])
                if self.source == "pitcher_zfs":
                    p_ = self.phases[k]
                    U = drvf.U_of_z(self.z_at(t, k, max(0.0, min(p_["duration"], t - self.events[k])))) / _D_M
                args = (on, (float(c["x"][i]), float(c["y"][i])), float(c["Q"][i]), U,
                        float(c["ux"][i]), float(c["uy"][i]),
                        inl0.scan_speed if self.source == "wanted" else
                        max(inl0.scan_speed, min(float(c["vfoot"][i]), self.vcap)))
            out, Q = drvf_record(self, k, inl0, *args)
            if self.source == "fsexact" and not _inlet_same(out, I):
                self.n_mismatch += 1
            self.slog.append((t, k, float(args[0]), args[2] * _ML_D3, (Q or 0.0) * _ML_D3, args[3] * _D_M,
                              (out.u_in[0] if out.active else 0.0) * _D_M, (out.u_in[1] if out.active else 0.0) * _D_M,
                              (math.sqrt(out.r1 * out.r2) if out.active else 0.0) * _D_M * 1e3,
                              (out.d_jet if out.active else 0.0) * _D_M * 1e3, out.scan_speed, out.x_hit[0], out.x_hit[1],
                              float(out.active)))
            return out

    with open(EXPS) as f:
        e = json.load(f)["P2"]
    params = dict(kappa_Q=2.0, cp=1.36, beta=3.0, nu=1e-3)
    for key in ("B_dep", "p_dep", "kappa_r"):
        params[key] = e["params"][key]
    sp = e["inlet"][pattern]
    prog = SolveProgram(pattern, sp["U"], sp["kappa_Q"], params, 0.41, FS_FLOW_PHASE[pattern], ramp=0.1, lat="exitmin",
                        lag=0.08, stop=False, ztr=True)
    prog.source = source
    prog.tab = None if source == "fsexact" else _SolveTable(solve_table(pattern, fill=250 if source == "pitcher250" else 350),
                                                          source)
    prog.n_mismatch = 0
    prog.slog = []
    prog.vcap = 0.5 / (N * DT)
    return prog


SOLVE_LOGCOLS = "t k on Qraw_mLs Q_mLs U_ms ux_ms uy_ms r_mm d_mm scan x y active".split()


def _fs_c(pattern):
    z = np.load(os.path.join(FINAL, "runs", f"FS_{pattern}", "c.npz"))
    return z["c"].astype(float), z["mask"]


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def solve_run(pattern, source, tag=None, N=256):
    """One V0.5 solver run (FS closure and settings) with the inlet from `source`; writes solve/runs/<tag>_<pattern>/."""
    from . import run as R
    tag = tag or SOLVE_TAGS[source]
    prog = solve_program(pattern, source, N)
    R.v05_program = lambda name: prog
    out = os.path.join(SOLVE, "runs", f"{tag}_{pattern}")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sol, m = R.run(pattern, N=N, out=out, video=False, quiet=True, control="v05")
    wall = time.time() - t0
    c = sol.concentration()
    mask = sol.g.mask.copy()
    np.savez_compressed(os.path.join(out, "c.npz"), c=c.astype(np.float32), mask=mask)
    np.save(os.path.join(out, "inlet_log.npy"), np.array(prog.slog))
    try:
        os.remove(os.path.join(out, "final_state.npz"))
    except OSError:
        pass
    drvf = _drvf()
    cr, mr = drvf.load_ref(pattern)
    cf, mf = _fs_c(pattern)
    c32 = c.astype(np.float32).astype(float)
    res = dict(tag=tag, pattern=pattern, source=source, table=None if prog.tab is None else prog.tab.meta,
               params={k: prog.config["parameters"][k] for k in ("cp", "beta", "nu", "D", "kappa_Q", "kappa_c", "kappa_t",
                                                                 "kappa_r", "B_dep", "p_dep")},
               scan_cap=prog.vcap, wall_s=wall, stats=m["stats"], ledger=m["ledger"], T=m["T"], n_calls=len(prog.slog),
               n_mismatch_vs_drvf=prog.n_mismatch if source == "fsexact" else None,
               iou04=_iou((c > 0.4) & mask, (cr > 0.4) & mr), iou05=_iou((c > 0.5) & mask, (cr > 0.5) & mr),
               iou04_FS=_iou((c32 > 0.4) & mask, (cf > 0.4) & mf), maxabs_dc_FS=float(np.max(np.abs(c32 - cf)[mask])),
               white=float(((c > 0.4) & mask).sum() / mask.sum()), white_ref=float(((cr > 0.4) & mr).sum() / mr.sum()))
    with open(os.path.join(out, "res.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("params", "stats", "ledger", "table")}, default=float),
          flush=True)
    return res


def solve_check(patterns=PATTERNS, dt=2.5e-4):
    """Offline check of drvf_record: on a dense grid and at every logged call time of the stored FS run, the fsexact
    program's records must be bitwise drvf's; the 'wanted' table program's records must be bitwise drvf's at t_k."""
    out = {}
    for p in patterns:
        pe = solve_program(p, "fsexact")
        pw = solve_program(p, "wanted")
        ref = fs_program(p)
        L = np.load(os.path.join(FINAL, "runs", f"FS_{p}", "inlet_log.npy"))
        ts = np.unique(np.concatenate([np.arange(0.0, pe.duration, dt), L[:, 0]]))
        n_bad_e = 0
        for t in ts:
            a = pe(t)
            b = ref(t)
            n_bad_e += 0 if _inlet_same(a, b) else 1
        n_bad_w = 0
        tk = np.arange(pw.tab.n) * pw.tab.dt
        tk = tk[tk < pw.duration]
        for t in tk:
            a = pw(t)
            b = ref(t)
            if not _inlet_same(a, b):
                # the scan bound may be raised by the table's foot speed; everything else must agree
                if a.active and b.active and all(getattr(a, k) == getattr(b, k) for k in
                                                 ("x_hit", "S_eff", "r1", "r2", "u_in", "U_perp", "d_jet", "phi")):
                    continue
                n_bad_w += 1
        n_scan = sum(1 for t in tk if pw(t).scan_speed != ref(t).scan_speed)
        out[p] = dict(n_exact=len(ts), mismatch_exact=n_bad_e + pe.n_mismatch, n_table=len(tk), mismatch_table=n_bad_w,
                      table_scan_raised=n_scan)
        print(p, out[p], flush=True)
    return out

SOLVE_BOXZ = dict(heart=(-0.3, 0.3, -0.35, 0.35), push_heart=(-0.35, 0.35, -0.35, 0.35), layered_heart=(-0.25, 0.25, -0.3, 0.3),
                  tulip=(-0.3, 0.3, -0.32, 0.38), leaf=(-0.3, 0.3, -0.38, 0.38), swan=(-0.32, 0.3, -0.2, 0.36))   # final/figs.py


def _ana():
    """final/ana.py with 's:<tag>' sources mapped to solve/runs/<tag>_<pattern> (ana itself unchanged)."""
    if FINAL not in sys.path:
        sys.path.insert(0, FINAL)
    import ana                                       # noqa: E402
    if not getattr(ana, "_solve_patched", False):
        orig = ana.path_of

        def path_of(src, p):
            if src.startswith("s:"):
                return os.path.join(SOLVE, "runs", f"{src[2:]}_{p}")
            return orig(src, p)
        ana.path_of = path_of
        ana._solve_patched = True
    return ana


def _phase_volumes(run_dir, kQ=KAPPA_Q):
    """Injected milk per phase label [mL] from trajectory.json (S_eff x frame length / kappa_Q x 512 mL)."""
    tj = os.path.join(run_dir, "trajectory.json")
    if not os.path.exists(tj):
        return {}
    out, tprev = {}, 0.0
    for q in json.load(open(tj)):
        dtf = q["t"] - tprev
        tprev = q["t"]
        if q["active"]:
            k = q["phase"].split(":")[0].replace("phase ", "")
            out[k] = out.get(k, 0.0) + q["S"] * dtf / kQ * 512.0
    return out


def solve_figure(name="FS_vs_pitcher", cols=(("V05", "V0.5"), ("f:FS", "FS (wanted inlet)"),
                                              ("s:PD", "pitcher-delivered inlet\n(ideal hand, 350 mL fill)")),
                 patterns=PATTERNS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ana = _ana()
    CM = LinearSegmentedColormap.from_list("s", [(0.25, 0.14, 0.06), (0.62, 0.45, 0.30), (1.0, 0.97, 0.90)])
    CM.set_bad((0.86, 0.86, 0.86))
    nc = len(cols)
    fig, ax = plt.subplots(len(patterns), nc, figsize=(2.4 * nc, 2.5 * len(patterns) + 0.4), dpi=120, squeeze=False)
    for i, p in enumerate(patterns):
        x0, x1, y0, y1 = SOLVE_BOXZ[p]
        for j, (src, lab) in enumerate(cols):
            a = ax[i, j]
            a.set_xticks([])
            a.set_yticks([])
            try:
                c, m = ana.load(src, p)
            except (FileNotFoundError, OSError):
                a.text(0.5, 0.5, "missing", ha="center", transform=a.transAxes)
                continue
            a.imshow(np.clip((c - 0.35) / 0.65, 0, 1), cmap=CM, vmin=0, vmax=1, origin="upper",
                     extent=(-0.5, 0.5, 0.5, -0.5), interpolation="nearest")
            a.set_xlim(x0, x1)
            a.set_ylim(y1, y0)
            if i == 0:
                a.set_title(lab, fontsize=7)
            if j == 0:
                a.set_ylabel(p, fontsize=7)
    fig.suptitle(f"{name}: c stretched 0.35..1 (x right, y down)", fontsize=8)
    fig.tight_layout()
    path = os.path.join(SOLVE, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def solve_report(srcs=("f:FS", "s:PD", "s:PW", "s:FX", "s:PDz", "s:PD250")):
    """ana.py metrics (section 3.1 criteria: struct measures, IoU vs V0.5, r_hp, high-pass box) for FS and the solve runs,
    IoU vs FS, per-stroke chi and injected milk (wanted = FS run, delivered = PD, table floor = PW); FS_vs_pitcher.png."""
    ana = _ana()
    out_json = os.path.join(SOLVE, "ana.json")
    argv0 = sys.argv
    try:
        sys.argv = ["ana.py", out_json, ",".join(srcs)]
        ana.main()
    finally:
        sys.argv = argv0
    A = json.load(open(out_json))
    fsst = json.load(open(os.path.join(FINAL, "fs_strokes.json")))["strokes"]
    tgt = {(s["pattern"], s["phase"]): s["chi_t"] for s in fsst}
    rows = {}
    for p in PATTERNS:
        cf, mf = ana.load("f:FS", p)
        R = A.get(p, {})
        for s in srcs:
            if s not in R:
                continue
            c, m = ana.load(s, p)
            R[s]["iou04_FS"] = round(_iou((np.nan_to_num(c) > 0.4) & m, (np.nan_to_num(cf) > 0.4) & mf), 3)
            R[s]["iou05_FS"] = round(_iou((np.nan_to_num(c) > 0.5) & m, (np.nan_to_num(cf) > 0.5) & mf), 3)
            d = ana.path_of(s, p)
            R[s]["vol_mL"] = {k: round(v, 3) for k, v in _phase_volumes(d).items()}
            R[s]["vol_total_mL"] = round(sum(R[s]["vol_mL"].values()), 2)
            if s.startswith("s:"):
                rj = json.load(open(os.path.join(d, "res.json")))
                R[s]["maxabs_dc_FS"] = rj.get("maxabs_dc_FS")
                R[s]["n_mismatch_vs_drvf"] = rj.get("n_mismatch_vs_drvf")
                R[s]["wall_s"] = rj.get("wall_s")
        prog = fs_program(p)
        st = []
        for k, cl in enumerate(prog.cls):
            key = str(k + 1)
            row = dict(n=k + 1, cls=cl, chi_t=tgt.get((p, k + 1)))
            for s in srcs:
                if s in R:
                    row["chi_" + s] = (R[s].get("chi") or {}).get(key)
                    row["vol_" + s] = R[s]["vol_mL"].get(key, 0.0)
            st.append(row)
        rows[p] = st
        A[p] = R
    A["_strokes"] = rows
    with open(out_json, "w") as f:
        json.dump(A, f, indent=1, default=float)
    solve_figure()
    solve_figure(name="FS_vs_pitcher_diag", cols=(("V05", "V0.5"), ("f:FS", "FS (wanted inlet)"),
                                                  ("s:PW", "FS 1 ms table (sampling check)"),
                                                  ("s:PD", "delivered, ideal hand,\n350 mL fill"),
                                                  ("s:PD250", "delivered, ideal hand,\n250 mL fill"),
                                                  ("s:PDz", "diag: delivered + FS heights")))
    return A


def job_list():
    J = []
    for sset in ("fs", "real"):
        for p in PATTERNS:
            J.append((sset, p, "main", 350))
    for p in PULSED:
        J.append(("fs", p, "pulse", 350))
    for sset in ("fs", "real"):
        for p in FILLS_EXTRA[sset]:
            for fl in (250, 500):
                J.append((sset, p, "main", fl))
    J += [("fs", "heart", "main", 250), ("fs", "swan", "main", 250)]
    for sset in ("fs", "real"):
        for p in PATTERNS:
            J.append((sset, p, "stock", 350))
    for p in PATTERNS:
        J.append(("fs", p, "cmd", 350))
    return J


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("sim")
    s.add_argument("set", choices=["fs", "real"])
    s.add_argument("pattern")
    s.add_argument("--variant", default="main", choices=["main", "stock", "cmd", "pulse"])
    s.add_argument("--fill", type=float, default=350)
    s.add_argument("--no_fig", action="store_true")
    a2 = sub.add_parser("analyze")
    a2.add_argument("name")
    b = sub.add_parser("batch")
    b.add_argument("--jobs", type=int, default=2)
    b.add_argument("--only", default="")
    b.add_argument("--skip_done", action="store_true")
    sub.add_parser("report")
    sub.add_parser("selftest")
    sub.add_parser("clearance")
    bs = sub.add_parser("best_sim")
    bs.add_argument("pattern")
    bs.add_argument("--variant", default="ideal", choices=list(LAW_VARIANTS))
    bs.add_argument("--fill", type=float, default=350)
    bs.add_argument("--npass", type=int, default=None)
    bs.add_argument("--no_fig", action="store_true")
    bb = sub.add_parser("best_batch")
    bb.add_argument("--jobs", type=int, default=2)
    bb.add_argument("--which", default="main", choices=["main", "sens", "all"])
    bb.add_argument("--only", default="")
    bb.add_argument("--skip_done", action="store_true")
    bst = sub.add_parser("best_static")
    bst.add_argument("--patterns", default=",".join(PATTERNS))
    sub.add_parser("best_report")
    st = sub.add_parser("solve_table")
    st.add_argument("--patterns", default=",".join(PATTERNS))
    sc = sub.add_parser("solve_check")
    sc.add_argument("--patterns", default=",".join(PATTERNS))
    sr = sub.add_parser("solve_run")
    sr.add_argument("pattern")
    sr.add_argument("--source", default="pitcher", choices=list(SOLVE_TAGS))
    sr.add_argument("--tag", default=None)
    sr.add_argument("--N", type=int, default=256)
    sb = sub.add_parser("solve_batch")
    sb.add_argument("--jobs", type=int, default=2)
    sb.add_argument("--list", default="", help="pattern:source,... (default: pitcher x 6, then wanted x 6)")
    sb.add_argument("--N", type=int, default=256)
    sub.add_parser("solve_report")
    a = ap.parse_args(argv)
    if a.cmd == "sim":
        W, meta, A, info = simulate(a.set, a.pattern, a.variant, a.fill)
        save_run(W, meta, A, info)
        r = analyze(info["run"], figure=not a.no_fig)
        print(json.dumps(dict(run=info["run"], wall_s=info["wall_s"], total=r["total"]), default=float))
    elif a.cmd == "analyze":
        analyze(a.name)
    elif a.cmd == "batch":
        J = job_list()
        if a.only:
            keep = set(a.only.split(","))
            J = [j for j in J if run_name(*j) in keep]
        if a.skip_done:
            J = [j for j in J if not os.path.exists(os.path.join(OUT, "runs", run_name(*j) + "_metrics.json"))]
        os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
        env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        running = []
        t0 = time.time()
        while J or running:
            while J and len(running) < a.jobs:
                j = J.pop(0)
                nm = run_name(*j)
                log = open(os.path.join(OUT, "logs", nm + ".log"), "w")
                cmd = [sys.executable, "-m", "latte_imex.v05_pitcher_check", "sim", j[0], j[1], "--variant", j[2],
                       "--fill", str(j[3])]
                pr = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      stdout=log, stderr=subprocess.STDOUT, env=env)
                running.append((pr, nm, time.time(), log))
                print(f"[{time.time() - t0:6.0f}s] start {nm}", flush=True)
            time.sleep(2.0)
            for r in list(running):
                if r[0].poll() is not None:
                    r[3].close()
                    running.remove(r)
                    print(f"[{time.time() - t0:6.0f}s] done  {r[1]} rc={r[0].returncode} ({time.time() - r[2]:.0f}s)",
                          flush=True)
        report()
    elif a.cmd == "report":
        report()
    elif a.cmd == "selftest":
        selftest()
    elif a.cmd == "clearance":
        print(json.dumps(clearance_map(), indent=1))
    elif a.cmd == "best_sim":
        W, meta, A, info = best_simulate(a.pattern, a.variant, a.fill, npass=a.npass)
        save_run(W, meta, A, info, out=BEST)
        r = analyze(info["run"], out=BEST, figure=not a.no_fig)
        x = best_extra(info["run"])
        print(json.dumps(dict(run=info["run"], wall_s=info["wall_s"], total=r["total"], extra=x["total"]), default=float))
    elif a.cmd == "best_batch":
        J = best_job_list(a.which)
        if a.only:
            keep = set(a.only.split(","))
            J = [j for j in J if best_name(j[0], j[1], j[2]) in keep]
        if a.skip_done:
            J = [j for j in J if not os.path.exists(os.path.join(BEST, "runs", best_name(*j) + "_extra.json"))]
        os.makedirs(os.path.join(BEST, "logs"), exist_ok=True)
        env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        running = []
        t0 = time.time()
        while J or running:
            while J and len(running) < a.jobs:
                j = J.pop(0)
                nm = best_name(*j)
                log = open(os.path.join(BEST, "logs", nm + ".log"), "w")
                cmd = [sys.executable, "-m", "latte_imex.v05_pitcher_check", "best_sim", j[0], "--variant", j[1],
                       "--fill", str(j[2])]
                pr = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      stdout=log, stderr=subprocess.STDOUT, env=env)
                running.append((pr, nm, time.time(), log))
                print(f"[{time.time() - t0:6.0f}s] start {nm}", flush=True)
            time.sleep(2.0)
            for r in list(running):
                if r[0].poll() is not None:
                    r[3].close()
                    running.remove(r)
                    print(f"[{time.time() - t0:6.0f}s] done  {r[1]} rc={r[0].returncode} ({time.time() - r[2]:.0f}s)",
                          flush=True)
    elif a.cmd == "best_static":
        best_static(patterns=tuple(a.patterns.split(",")))
    elif a.cmd == "best_report":
        best_report()
    elif a.cmd == "solve_table":
        for p in a.patterns.split(","):
            for fl in (350, 250):
                print(solve_table(p, rebuild=True, fill=fl), flush=True)
    elif a.cmd == "solve_check":
        r = solve_check(tuple(a.patterns.split(",")))
        os.makedirs(SOLVE, exist_ok=True)
        with open(os.path.join(SOLVE, "check_records.json"), "w") as f:
            json.dump(r, f, indent=1)
    elif a.cmd == "solve_run":
        solve_run(a.pattern, a.source, a.tag, a.N)
    elif a.cmd == "solve_batch":
        J = [tuple(x.split(":")) for x in a.list.split(",") if x] or \
            [(p, "pitcher") for p in PATTERNS] + [(p, "wanted") for p in PATTERNS]
        os.makedirs(os.path.join(SOLVE, "logs"), exist_ok=True)
        env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        running = []
        t0 = time.time()
        while J or running:
            while J and len(running) < a.jobs:
                p, s_ = J.pop(0)
                nm = f"{SOLVE_TAGS[s_]}_{p}"
                log = open(os.path.join(SOLVE, "logs", nm + ".log"), "w")
                cmd = [sys.executable, "-m", "latte_imex.v05_pitcher_check", "solve_run", p, "--source", s_,
                       "--N", str(a.N)]
                pr = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      stdout=log, stderr=subprocess.STDOUT, env=env)
                running.append((pr, nm, time.time(), log))
                print(f"[{time.time() - t0:6.0f}s] start {nm}", flush=True)
            time.sleep(2.0)
            for r in list(running):
                if r[0].poll() is not None:
                    r[3].close()
                    running.remove(r)
                    print(f"[{time.time() - t0:6.0f}s] done  {r[1]} rc={r[0].returncode} ({time.time() - r[2]:.0f}s)",
                          flush=True)
    elif a.cmd == "solve_report":
        solve_report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
