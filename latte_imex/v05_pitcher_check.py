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
  yaw   FS: direction of Codex's jet b of the stroke (the FS exit part is along b); real: script yaw.  Between strokes
        the yaw turns with a cosine ease over the bridge (or +-0.05 s around a direct stroke boundary).

DRIVE
  Pitcher(GridPitcherGeometry 'pitcher_nx', nr = 6, OutflowLaw defaults: C_d 0.60, h_k 0.8 mm, h_on 2.5 mm,
  h_off 0.5 mm, tau 0.08 s, tau_stop 0.15 s, slosh zeta 0.3, carry velocity), fill V0 (350 mL; 250 / 500 mL for two
  patterns per set), dt = 1 ms.  FastPitcher only replaces the free-surface evaluation by an algebraically identical
  one (heights of the sample points along the surface normal as a single mat-vec instead of transforming all points to
  the world frame; same sort / cumulative-volume interpolation); selftest() checks it against Pitcher.free_surface.
  TableBarista = pitcher.Barista with its controller unchanged (feed-forward tilt from tilt_for_flow + PI on the
  low-passed flow, 60 deg/s rate limit, swing / hold logic, spout-height rate limit 0.25 m/s, the 0.3 s stream-offset
  correction, collision lift Pitcher.required_lift against the default Cup) whose intent() returns the wanted table
  (foot path, z, Q) and whose pose carries the wanted yaw.  tilt0 (arrival tilt) = flow-onset tilt of the fill - 2 deg.
  Variants
    main   intent Q = the wanted flow at the surface (FS Q(t) incl. pulsation / real Q_w).
    cmd    (FS) intent Q = the FS pre-lag command (envelope x pulsation): FS's own tau = 0.08 s lag is then supplied
           by the pitcher's spout instead of being applied twice.
    pulse  (FS, pulsed patterns) the controller tracks the lagged envelope Q_env and an open-loop 5 Hz tilt oscillation
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
  python3 -m latte_imex.v05_pitcher_check report
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
REVIEW = os.environ.get("V05_REVIEW_DIR",
                        "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review")
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
        W["Q"][i] = st["Q"]
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
class TableBarista(Barista):
    """pitcher.Barista with the controller unchanged; intent() returns the wanted table (foot path [D], z [m], flow
    [m^3/s] from column qkey) and the pose carries the wanted yaw."""

    def __init__(self, pitcher, W, meta, coupling, qkey="Q", **kw):
        self.W, self.meta, self.qkey = W, meta, qkey
        self.dt_tab = meta["dt"]
        self.n_tab = len(W["t"])
        self.T_tab = self.n_tab * self.dt_tab
        self._x, self._y, self._z, self._q = (np.asarray(W[k], float) for k in ("x", "y", "z", qkey))
        self._q = np.nan_to_num(self._q)
        self._yaw = np.asarray(W["yaw"], float)
        self._k = np.asarray(W["k"], float)
        mv = [Move(self.T_tab, (self._x[0], self._y[0]), (self._x[-1], self._y[-1]), z0=float(self._z[0]))]
        super().__init__(pitcher, mv, coupling, yaw=float(self._yaw[0]), **kw)

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


class _OscPitcher:
    """Proxy pitcher seen by the barista: adds the barista's open-loop tilt oscillation `osc_val` to every pose that
    reaches required_lift / step; everything else is forwarded to the real pitcher."""

    def __init__(self, P, owner):
        self._P, self._owner = P, owner

    def __getattr__(self, k):
        return getattr(self._P, k)

    def _p(self, pose):
        return Pose(pose.tip, pose.yaw, max(0.0, pose.tilt + self._owner.osc_val), pose.roll)

    def required_lift(self, pose, cup, iters=14):
        return self._P.required_lift(self._p(pose), cup, iters)

    def step(self, pose, dt, z_surface=0.0):
        return self._P.step(self._p(pose), dt, z_surface)


class PulseBarista(TableBarista):
    """Best-case pulsation: the stock controller tracks the lagged envelope (qkey='Q_env'); a 5 Hz tilt oscillation
    about the spout tip, Re(C e^{i w tt}) with tt the stroke clock, is added open loop.  C starts from the linear
    transfer function (weir sensitivity s = dlnQ_ss/dtilt, spout lag 1/(1 + i w tau), flight delay e^{-i w t_hit}) and
    is corrected once per cycle by a lock-in on the arriving flow: C <- C (c_want / c_meas)^0.5, |C| <= amp_max."""

    def __init__(self, pitcher, W, meta, coupling, amp_max=math.radians(12.0), gain=0.5, **kw):
        self.real = pitcher
        self.osc_val = 0.0
        super().__init__(_OscPitcher(pitcher, self), W, meta, coupling, qkey="Q_env", **kw)
        self.pulsed = [s for s in meta["strokes"] if s.get("pulse")]
        self.amp_max, self.gain = amp_max, gain
        self.C = None
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
                self.cur = s
                self.buf_t, self.buf_q, self.buf_w = [], [], []
                self.cycle_end = None
            tt = t - s["t0"]
            w = 2 * math.pi * s["pulse"]["f"]
            env = min(1.0, tt / 0.05, (s["t1"] - t) / 0.05)
            self.osc_val = env * (self.C * complex(math.cos(w * tt), math.sin(w * tt))).real
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
                        self.history.append((float(ta), abs(cm), float(np.angle(cm)), abs(cw), float(np.angle(cw)),
                                             abs(self.C), float(np.angle(self.C))))
                    self.buf_t, self.buf_q, self.buf_w = [], [], []
                    self.cycle_end = tt + per
            if rec["t"] >= s["t1"] + 0.3:
                self.cur = None
        return rec, name


# --------------------------------------------------------------------------- simulation
REC_KEYS = ("t", "V", "Q", "Q_feed", "Q_hit", "Q_ss", "h_max", "h_bar", "c", "width", "flowing", "overflow", "tilt",
            "tip_x", "tip_y", "tip_z", "lift", "yaw", "z_cmd", "off_x", "off_y", "Q_want", "osc",
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
    kw = dict(tilt0=tilt0)
    if variant == "main":
        B = TableBarista(P, W, meta, cp, qkey="Q", **kw)
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
                dt=dt, nr=nr, steps=n, wall_s=time.time() - t0w, wall_want_s=t_want, wall_sim_s=time.time() - t1w,
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
    L_Q = best_delay(t[m_in], Qd[m_in], t[m_in], Qw[m_in])
    Qw_L = np.interp(t - L_Q, t, Qw, left=0.0, right=0.0)
    L_int = best_delay(t[m_in], Qd[m_in], t[m_in], Qint[m_in])

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
        o["lift_time_s"] = float(np.sum(lift[m] > 1e-3) * dt)
        o["z_reachable_min_cm"] = float(np.nanmax((zw + lift)[m & (lift > 1e-3)]) * 100) if (m & (lift > 1e-3)).any() \
            else None
        o["tilt_range_deg"] = _rng(tilt[m])
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
        # classes
        c = {}
        rq = o["Q_err_rel"]
        c["Q"] = "DELIVERED" if rq <= TOL_REL else (
            "IMPOSSIBLE no stream" if o["V_d_mL"] < 0.05 * max(o["V_w_mL"], 1e-9) else
            f"DEVIATES |dQ| {rq * 100:.0f}% (delay-corr. {o['Q_err_rel_delayed'] * 100:.0f}%), V {o['Q_ratio']:.2f}x")
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
        if o["lift_max_mm"] > TOL_MM:
            c["z"] = f"IMPOSSIBLE lift {o['lift_max_mm']:.1f} mm ({o['lift_time_s']:.2f} s)"
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
    fig.suptitle(f"{name}: {CN.get(meta['pattern'], meta['pattern'])}  fill {info['fill_mL']:.0f} mL, "
                 f"milk used {tot['milk_used_mL']:.0f} mL, flow delay {tot['Q_delay_s'] * 1e3:.0f} ms, "
                 f"max lift {tot['lift_max_mm']:.1f} mm", color=INK, fontsize=11)
    fn = f"{sset}_{meta['pattern']}.png" if (info["variant"] == "main" and info["fill_mL"] == 350) else f"{name}.png"
    fig.savefig(os.path.join(out, fn), dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------- report
QUANT = ("Q", "pulse", "U_perp", "u0", "exit_h", "tan", "uh", "r", "foot", "z")
QLABEL = dict(Q="流量 Q", pulse="5 Hz 脉动", U_perp="U⊥", u0="出口速度 u0", exit_h="出口水平速度", tan="tanα",
              uh="杯面 |u_h|", r="足迹 r", foot="落点位置", z="壶嘴高度")


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
                foot=o.get("foot_rms_mm", 0), z=o.get("lift_max_mm", 0) + o.get("z_err_mm", 0))[q]


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


def report(out=OUT):
    runs = sorted(f[:-13] for f in os.listdir(os.path.join(out, "runs")) if f.endswith("_metrics.json"))
    R = {}
    for r in runs:
        with open(os.path.join(out, "runs", r + "_metrics.json")) as f:
            R[r] = json.load(f)
    order = [run_name(s, p, v, fl) for s in ("fs", "real") for v, fl in (("main", 350), ("cmd", 350), ("pulse", 350),
                                                                         ("main", 250), ("main", 500))
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
    for sset, title in (("fs", "FS（猜的全物理动作）"), ("real", "真实动作（stage scripts + PhysInlet）")):
        rows = [r for r in order if R[r]["info"]["set"] == sset and R[r]["info"]["variant"] == "main"
                and R[r]["info"]["fill_mL"] == 350]
        if not rows:
            continue
        L.append(f"\n## {title}：装奶 350 mL，标准咖啡师控制器\n")
        L.append("| 图案 | " + " | ".join(QLABEL[q] for q in QUANT) + " |")
        L.append("|---|" + "---|" * len(QUANT))
        for r in rows:
            st = R[r]["strokes"]
            cells = []
            for q in QUANT:
                c, n = _worst(st, q)
                cells.append("–" if c == "n/a" else (_short(c) + (f"（#{n}）" if n is not None and not c.startswith("DELIVERED") else "")))
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
                f"{o['z_w_cm']:.2f}/{o['z_tip_cm']:.2f}, {o['lift_max_mm']:.1f} | "
                f"{o['tilt_range_deg'][0]:.0f}–{o['tilt_range_deg'][1]:.0f} |")
        if x["gaps"]:
            gl = "; ".join(f"{g['name']} 要漏 {g['V_w_mL']:.2f} mL / 壶漏 {g['V_d_mL']:.2f} mL，<1 mL/s 时长 要 {g['zero_w_s']:.2f} / 壶 {g['zero_d_s']:.2f} s"
                           for g in x["gaps"])
            L.append(f"\n桥/停：{gl}\n")
        js["runs"][r] = dict(info=x["info"], total=x["total"],
                             classes={str(o["n"]): o["class"] for o in x["strokes"]},
                             worst={q: _worst(x["strokes"], q) for q in QUANT})
    with open(os.path.join(out, "summary_tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(js, f, indent=1, default=float)
    return R


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


# --------------------------------------------------------------------------- CLI
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
    for p in PATTERNS:
        J.append(("fs", p, "cmd", 350))
    return J


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("sim")
    s.add_argument("set", choices=["fs", "real"])
    s.add_argument("pattern")
    s.add_argument("--variant", default="main", choices=["main", "cmd", "pulse"])
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
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
