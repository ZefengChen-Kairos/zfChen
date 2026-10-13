"""calib.py model adapter for the V0.5 single-layer model (IMEX solver `solver.py`, closure 'chi', return law 'v05')
driven by the barista's REAL actions (the physical-time stage scripts read off the six clips) through a PHYSICAL inlet.

    from latte_imex import calib as CB
    from latte_imex.v05_adapter import V05Model, FS, FSP2
    m = V05Model(base=FS)                                     # closure set FS (final all-physical set, p_dep = 8)
    clip = CB.load_clip("tulip")
    out = m.run(clip["script"], {}, 128, clip["res"]["t"])    # dict(c=[n, N, N] M frame, ledger=..., inlet_log=...)
    wT, led = CB.simulate_clip(m, {}, clip, 128)              # the same through calib
    python -m latte_imex.v05_adapter test                     # unit tests (inlet formulas, zero flow, ledgers)

Nothing in the solver is changed; this module only builds the solver's `Inlet` record at each control frame and runs
the solver loop so that the requested output times are hit exactly.

UNITS.  D = 0.08 m (cup diameter, `video_extract.D_CUP`).  Positions in D, solver speeds in D/s, flows Q in m^3/s
(reported in mL/s), spout height z in m.  Solver arrays are in the M frame, c_M[iy, ix] with x = (ix + 0.5)/N - 0.5,
y = (iy + 0.5)/N - 0.5 (`calib.observe_model` converts to the T frame of the clips).

TIME BASE.  `script` is a physical-time stage script (`calib.physical_script`: freeze frames removed).  V0.5 deposits
milk instantly (no first-white delay), so the action starts WITHOUT a pre-roll: if the script carries one
(script['preroll'] > 0, e.g. `calib.load_clip('tulip')['script']` with PREROLL = 1.5 s), the inserted pre-roll stage is
dropped (preroll='drop', default) and the model clock starts at the first real stage, t_start = its t0 on the script's
axis; t_out are on the script's axis, model time tau = t - t_start, and every t_out <= t_start returns the initial
state (all brown).  This equals running `calib.physical_script(name, preroll=0)` from tau = 0 and placing the result on
the clip's (pre-rolled) axis.  preroll='keep' pours the pre-roll stage as well.

WANTED ACTION (`v3min_adapter.Inlet`, the barista's intended foot path as `pitcher.Barista.intent` defines it):
  x_f(tau), y_f(tau) [D]   stream-foot path of the stage script (cosine easing p0 -> p1, optional Bezier, wiggle
                           amp(t) sin(2 pi f t) along the wiggle axis with a 0.25 s ease-in/out of the amplitude);
                           gaps between stages are dry repositions
  z(tau) [m]               spout height above the surface (z0 -> z1 with the same easing; dry gaps at max(z1, 3 cm))
  Q_w(tau) [m^3/s]         wanted flow = flow(tau) x Q_ref, Q_ref = 15 mL/s (flow0 -> flow1 linear in time) x Q_scale
  v_h(tau) [D/s]           hand (= foot) velocity d(x_f, y_f)/dtau, analytic derivative of the same formulas
  e = (cos yaw, sin yaw)   spout direction, yaw = script['barista_yaw_rad'] (M frame; pitcher body at yaw + pi)

PHYSICAL INLET (same physics as the all-physical set FS, `v05allphys/final/drvf.py --lat exitmin --lag 0.08`, with
the spout direction from the script instead of Codex's jet direction and no flow pulsation):
  spout lag        tau_s dQ_lag/dtau = Q_w - Q_lag,  tau_s = 0.08 s, Q_lag(0) = 0  (exact first-order-hold recursion on a
                   0.5 ms grid with the stage boundaries inserted; Q_w is piecewise linear in time, so it is exact)
  received flow    pouring (Q_w > 0):  Q = clip(Q_lag, 1 mL/s, 32 mL/s)
                   not pouring (Q_w = 0, the lag leaks into a gap / stop):  Q = Q_lag if Q_lag >= 1 mL/s, else no flow
  impact speed     U_perp = sqrt(V0^2 + 2 g z),  V0 = 0.35 m/s (spout exit speed), g = 9.81 m/s^2
  horizontal       u_h = D v_h + u_exit e  [m/s],  u_exit = 0.15 m/s (envelope 0.10-0.27); the cross-spout component
                   of u_h is exactly the hand speed; if |u_h| > min(0.40 m/s, U_perp) it is scaled down to that bound
                   (tan alpha <= 1, as in FS)
  footprint        continuity ellipse as `v05_program` 'continuity_ellipse':  r = sqrt(Q / (pi U_perp)),
                   spd = sqrt(U_perp^2 + |u_h|^2), aspect = spd / U_perp, semi-axes r1 = r sqrt(aspect) (along u_h),
                   r2 = r / sqrt(aspect), phi = atan2(u_h,y, u_h,x), jet diameter d = 2 sqrt(Q / (pi spd))
  source           S_eff = kappa_Q Q / D^3   (Q_rec in D^3/s; 1 D^3 = 512 mL)
  solver record    Inlet(active, x_hit = (x_f, y_f), S_eff, u_in = u_h / D, U_perp / D, d_jet = d / D, r1 / D, r2 / D,
                   phi, scan_speed = max |v_h| over the current move (frame-length bound of the solver))
  off the cup      a foot farther than cup_radius + max(r1, r2) from the centre gives no inlet (counted in the ledger)
  no pulsation     the clips do not measure a 5 Hz flow modulation, so none is added.
  The solver then applies the unchanged V0.5 closure: chi = 1 / (1 + (U_perp^2 / (B_dep d))^p_dep), deposition
  s = chi S_eff K, traction Lambda = kappa_c (1 - chi) S_eff K, target velocity v* = kappa_t u_in + kappa_r V_R e_r
  with V_R = min(U_perp chi^(1/(2 p_dep)), U_perp, sqrt(B_dep d)) (return law 'v05').

CLOSURE SETS (zero tuning on the videos; derived for Codex's strokes in v05allphys/ALLPHYS_RESULT.md):
  FS    c_p 1.36 D/s, beta 3 /s, nu 1e-3 D^2/s, D 1e-7 D^2/s, kappa_Q 2.0 /D, kappa_c 1.0, kappa_t 0.7,
        kappa_r 0.037255, B_dep 1550.415 D/s^2, p_dep 8   (fs.json / runs/FS_*/res.json)
  FSP2  FS with p_dep 2, B_dep 2149.443 D/s^2, kappa_r 0.042312   (fits.json F_p2)

SPACE (closure parameters a later calibration would fit; bounds = physical ranges of v05audit/closure.md and
ALLPHYS_RESULT.md):
  B_dep    300-6000 D/s^2, log   g' Fr_c^2 with g' = 73.6 D/s^2 (rho_f/rho = 0.4) and Fr_c 2-9 (nominal 1400-2200)
  p_dep    1-8, lin              switch steepness, prior range of the all-physical inverse design
  kappa_r  0.005-0.2, log        per-stroke need 0.007-0.17 with physical U_perp
  kappa_Q  0.7-5 /D, log         ell_w D / Gamma, Gamma = poured volume per white area 16-120 mm
  cp       0.68-2.07 D/s, log    sqrt(g' h_f) / D of the foam raft, rho_f/rho 0.3-0.7, h_f 1-4 mm ('c_p' accepted)
  beta     0.12-7.5 /s, log      Stokes-layer drag (rho/rho_f) sqrt(nu_l / t) / h_f
  kappa_t  0-1, lin              share of the stream's horizontal momentum given to the surface
  kappa_c  0-1, lin              traction efficiency of the non-deposited part
Inlet options that are not closure parameters may also be passed in `params`: u_exit [m/s], tau_spout [s], Q_scale.

LEDGER (returned with every run): the solver's own layer and milk ledgers (layer_error = layer - (initial area +
deposited), milk_error = milk - deposited; both close to round-off), wanted / received milk volumes [mL], deposited
layer volume, flow-weighted chi, inlet ranges, wall time, steps, retries, ok / error.

OUTPUT FIELD.  c = m / l (raft coverage = whiteness, the V0.5 'chi' closure); cells outside the cup (r > 0.49 D) get
the value of the nearest cup cell so that the bilinear resampling of `calib.observe_model` sees no NaN at the rim.
"""
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

from . import video_extract as VE
from .solver import Inlet, Numerics, Params, Solver
from .v3min_adapter import Inlet as WantedPath

D_CUP = VE.D_CUP          # 0.08 m
G = 9.81                  # m/s^2
V0_EXIT = 0.35            # spout exit speed in U_perp = sqrt(V0^2 + 2 g z) [m/s]
U_EXIT = 0.15             # exit part of the horizontal material velocity along the spout [m/s] (envelope 0.10-0.27)
TAU_SPOUT = 0.08          # first-order spout lag [s]
Q_MIN = 1e-6              # coherent-stream floor [m^3/s] (1 mL/s)
Q_MAX = 32e-6             # flow cap [m^3/s]
UH_MAX = 0.40             # cap of |u_h| [m/s] (and |u_h| <= U_perp)
LAG_DT = 5e-4             # lag integration grid [s]
EASE = 0.25               # wiggle amplitude ease-in/out of v3min_adapter.Inlet [s]
ML = 1e-6

FS = dict(cp=1.36, beta=3.0, nu=1e-3, D=1e-7, kappa_Q=2.0, kappa_c=1.0, kappa_t=0.7,
          kappa_r=0.03725531625168605, B_dep=1550.4150287146037, p_dep=8.0)
FSP2 = dict(FS, B_dep=2149.4432509690528, p_dep=2.0, kappa_r=0.04231228241693512)
CLOSURE_SETS = dict(FS=FS, FSp2=FSP2)

SPACE = dict(B_dep=(300.0, 6000.0, "log"), p_dep=(1.0, 8.0, "lin"), kappa_r=(0.005, 0.2, "log"),
             kappa_Q=(0.7, 5.0, "log"), cp=(0.68, 2.07, "log"), beta=(0.12, 7.5, "log"),
             kappa_t=(0.0, 1.0, "lin"), kappa_c=(0.0, 1.0, "lin"))
CLOSURE_KEYS = ("cp", "beta", "nu", "D", "kappa_Q", "kappa_c", "kappa_t", "kappa_r", "B_dep", "p_dep")
INLET_KEYS = ("u_exit", "tau_spout", "Q_scale")

LOGCOLS = ("t", "active", "Q_w_mLs", "Q_mLs", "z_cm", "U_ms", "uhx_ms", "uhy_ms", "vhx_ms", "vhy_ms", "r_mm", "d_mm",
           "r1_mm", "r2_mm", "chi", "fdt")


# =========================================================================== wanted action + physical inlet
class PhysInlet:
    """Physical inlet of a physical-time stage script (model time tau = 0 at the script's first stage)."""

    def __init__(self, script, kappa_Q, u_exit=U_EXIT, tau_spout=TAU_SPOUT, Q_scale=1.0, cup_radius=0.49,
                 t_end=None, lag_dt=LAG_DT):
        self.path = WantedPath(script, Q_scale=Q_scale)
        self.moves = self.path.moves
        self.T = self.path.T
        self.yaw = self.path.yaw
        self.e = (math.cos(self.yaw), math.sin(self.yaw))
        self.kappa_Q = float(kappa_Q)
        self.u_exit = float(u_exit)
        self.tau = float(tau_spout)
        self.cup_radius = cup_radius
        self.starts = np.concatenate([[0.0], np.cumsum([m.dur for m in self.moves])])
        # scan-speed bound per move (max |v_h| on a fine sample, +10 %)
        self.vmax = []
        for k, m in enumerate(self.moves):
            ts = self.starts[k] + np.linspace(0.0, m.dur, 401)[:-1]
            v = max(math.hypot(*self.foot(t)[4:6]) for t in ts)
            self.vmax.append(1.1 * v + 1e-3)
        # spout lag on a fine grid with the move boundaries inserted (left / right limits)
        t_end = max(self.T, t_end or 0.0) + 0.5
        grid = np.arange(0.0, t_end + lag_dt, lag_dt)
        eps = 1e-9
        bnd = np.concatenate([self.starts[1:] - eps, self.starts[1:] + eps])
        tg = np.unique(np.concatenate([grid, bnd[(bnd > 0) & (bnd < t_end)]]))
        qw = np.array([self.path.at(t)[3] for t in tg])
        y = np.zeros_like(qw)
        for i in range(1, len(tg)):
            h = tg[i] - tg[i - 1]
            if self.tau <= 0:
                y[i] = qw[i]
                continue
            a = math.exp(-h / self.tau)
            s = (qw[i] - qw[i - 1]) / h if h > 0 else 0.0
            # exact for an input linear on [t_{i-1}, t_i]:  y = u - s tau + (y0 - u0 + s tau) e^{-h/tau}
            y[i] = qw[i] - s * self.tau + (y[i - 1] - qw[i - 1] + s * self.tau) * a
        self.tg, self.qw_grid, self.qlag_grid = tg, qw, y

    # ----------------------------------------------------------------------- wanted action
    def move_index(self, t):
        k = int(np.searchsorted(self.starts, t, side="right")) - 1
        return k if 0 <= k < len(self.moves) else None

    def foot(self, t):
        """(x_f [D], y_f [D], z [m], Q_w [m^3/s], v_hx [D/s], v_hy [D/s]) at model time t (v3min_adapter.Inlet.at +
        its analytic time derivative)."""
        x, y, z, Q = self.path.at(t)
        k = self.move_index(t)
        if k is None:
            return x, y, z, Q, 0.0, 0.0
        m = self.moves[k]
        s = t - self.starts[k]
        f = s / m.dur
        de = 0.5 * math.pi * math.sin(math.pi * f) / m.dur
        e = 0.5 - 0.5 * math.cos(math.pi * f)
        if m.bezier is not None:
            (ax, ay), (bx, by) = m.bezier
            p0, p1 = m.p0, m.p1
            dxe = 3 * (1 - e) ** 2 * (ax - p0[0]) + 6 * (1 - e) * e * (bx - ax) + 3 * e * e * (p1[0] - bx)
            dye = 3 * (1 - e) ** 2 * (ay - p0[1]) + 6 * (1 - e) * e * (by - ay) + 3 * e * e * (p1[1] - by)
        else:
            dxe, dye = m.p1[0] - m.p0[0], m.p1[1] - m.p0[1]
        vx, vy = dxe * de, dye * de
        if m.wfreq > 0 and (m.wobble != 0 or m.wobble1 != 0):
            A0 = m.wobble + (m.wobble1 - m.wobble) * f
            dA0 = (m.wobble1 - m.wobble) / m.dur
            c = [1.0, s / EASE, (m.dur - s) / EASE]
            i = int(np.argmin(c))
            env, denv = c[i], (0.0, 1.0 / EASE, -1.0 / EASE)[i]
            w = 2 * math.pi * m.wfreq
            dwob = (dA0 * env + A0 * denv) * math.sin(w * s) + A0 * env * w * math.cos(w * s)
            if m.wobble_dir is not None:
                vx += dwob * math.cos(m.wobble_dir); vy += dwob * math.sin(m.wobble_dir)
            elif m.wobble_axis == "x":
                vx += dwob
            elif m.wobble_axis == "y":
                vy += dwob
        return x, y, z, Q, vx, vy

    def q_lag(self, t):
        if t <= 0:
            return 0.0
        if t <= self.tg[-1]:
            return float(np.interp(t, self.tg, self.qlag_grid))
        return float(self.qlag_grid[-1] * math.exp(-(t - self.tg[-1]) / self.tau)) if self.tau > 0 else 0.0

    # ----------------------------------------------------------------------- physical inlet
    def state(self, t):
        """All inlet quantities at model time t (SI + solver units); 'inlet' is the solver record."""
        x, y, z, Qw, vx, vy = self.foot(t)
        ql = self.q_lag(t)
        if Qw > 0:
            Q = min(max(ql, Q_MIN), Q_MAX)
        else:
            Q = min(ql, Q_MAX) if ql >= Q_MIN else 0.0
        k = self.move_index(t)
        scan = self.vmax[k] if k is not None else 0.0
        out = dict(t=t, x=x, y=y, z=z, Q_w=Qw, Q_lag=ql, Q=Q, vh=(vx, vy), scan=scan, active=False, off_cup=False)
        if Q <= 0:
            out["inlet"] = Inlet(active=False, x_hit=(x, y), S_eff=0.0, scan_speed=scan)
            return out
        U = math.sqrt(V0_EXIT ** 2 + 2.0 * G * max(z, 0.0))                       # m/s
        ux = vx * D_CUP + self.u_exit * self.e[0]
        uy = vy * D_CUP + self.u_exit * self.e[1]                                    # m/s
        hi = min(UH_MAX, U)
        uh = math.hypot(ux, uy)
        if uh > hi:
            ux, uy, uh = ux * hi / uh, uy * hi / uh, hi
        spd = math.hypot(U, uh)
        aspect = spd / U
        r = math.sqrt(Q / (math.pi * U))                                             # m
        d = 2.0 * math.sqrt(Q / (math.pi * spd))                                     # m
        r1, r2 = r * math.sqrt(aspect), r / math.sqrt(aspect)
        phi = math.atan2(uy, ux)
        S = self.kappa_Q * Q / D_CUP ** 3
        out.update(U=U, uh=(ux, uy), r=r, d=d, r1=r1, r2=r2, phi=phi, S_eff=S)
        if math.hypot(x, y) > self.cup_radius + max(r1, r2) / D_CUP:
            out["off_cup"] = True
            out["inlet"] = Inlet(active=False, x_hit=(x, y), S_eff=0.0, scan_speed=scan)
            return out
        out["active"] = True
        out["inlet"] = Inlet(active=True, x_hit=(x, y), S_eff=S, u_in=(ux / D_CUP, uy / D_CUP), U_perp=U / D_CUP,
                             d_jet=d / D_CUP, r1=r1 / D_CUP, r2=r2 / D_CUP, phi=phi, scan_speed=scan)
        return out


# =========================================================================== model
def strip_preroll(script):
    """(script without the pre-roll stage, t_start on the script's axis)."""
    st = list(script["stages"])
    pre = float(script.get("preroll", 0.0) or 0.0)
    if pre > 0 and st and st[0].get("t_clip") is None and "pre-roll" in str(st[0].get("note", "")):
        st = st[1:]
    out = dict(script)
    out["stages"] = st
    out["t_clip0"] = st[0]["t0"]
    return VE.load_stages(out), float(st[0]["t0"])


class V05Model:
    """V0.5 single-layer model + physical inlet from the stage script (see module docstring).

    base     closure dict (FS or FSP2, or any dict with CLOSURE_KEYS); params passed to run() override it
    u_exit   exit part of the horizontal material velocity [m/s];  tau_spout  spout lag [s]
    preroll  'drop' (default: V0.5 has no first-white delay) or 'keep'
    """

    def __init__(self, base=None, u_exit=U_EXIT, tau_spout=TAU_SPOUT, preroll="drop", numerics=None):
        self.base = dict(FS if base is None else base)
        self.u_exit, self.tau_spout, self.preroll = u_exit, tau_spout, preroll
        self.numerics = dict(numerics or {})
        self.space = dict(SPACE)
        self._fill = {}

    def closure(self, params):
        c = dict(self.base)
        for k, v in (params or {}).items():
            if k == "c_p":
                k = "cp"
            if k in CLOSURE_KEYS:
                c[k] = float(v)
        return c

    def solver_params(self, params):
        c = self.closure(params)
        return Params(closure="chi", return_law="v05", **c)

    def inlet(self, script, params=None, cup_radius=0.49, t_end=None):
        p = params or {}
        sc, t_start = (strip_preroll(script) if self.preroll == "drop" else (VE.load_stages(script),
                                                                              float(script["stages"][0]["t0"])))
        inl = PhysInlet(sc, kappa_Q=self.closure(p)["kappa_Q"], u_exit=float(p.get("u_exit", self.u_exit)),
                        tau_spout=float(p.get("tau_spout", self.tau_spout)), Q_scale=float(p.get("Q_scale", 1.0)),
                        cup_radius=cup_radius, t_end=t_end)
        return inl, t_start

    def _nearest(self, mask):
        key = mask.shape[0]
        if key not in self._fill:
            self._fill[key] = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        return self._fill[key]

    def run(self, script, params, N, t_out, keep_log=True):
        t_out = np.asarray(t_out, float)
        P = self.solver_params(params)
        num = Numerics(N=N, **self.numerics)
        sol = Solver(P, num)
        g = sol.g
        t0w = time.time()
        inl, t_start = self.inlet(script, params, num.cup_radius, t_end=float(t_out.max()) - 0.0)
        tm = t_out - t_start
        targets = np.unique(tm[tm > 1e-9])
        iy, ix = self._nearest(g.mask)

        def snap():
            c = sol.concentration()
            return np.where(g.mask, c, c[iy, ix]).astype(np.float32)

        c0 = snap()
        snaps = np.empty((len(t_out), N, N), np.float32)
        filled = np.zeros(len(t_out), bool)
        for i in np.flatnonzero(tm <= 1e-9):
            snaps[i] = c0; filled[i] = True
        log = []
        Vw = Vr = 0.0
        n_off = 0
        ok, err = True, ""
        chi_w = chi_q = 0.0
        try:
            j = 0
            while j < len(targets):
                tgt = targets[j]
                if sol.t >= tgt - 1e-9:
                    for i in np.flatnonzero(np.abs(tm - tgt) < 1e-9):
                        snaps[i] = snap(); filled[i] = True
                    j += 1
                    continue
                fdt = num.frame_dt
                probe = inl.state(sol.t)["inlet"]
                if probe.active and probe.scan_speed > 0:
                    fdt = min(fdt, max(num.scan_safety * min(g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
                fdt = min(fdt, tgt - sol.t)
                st = inl.state(sol.t + 0.5 * fdt)
                tb = sol.t
                chi, nsub = sol.advance_frame(st["inlet"], frame_dt=fdt)
                dt_taken = sol.t - tb
                Vw += st["Q_w"] * dt_taken
                if st["active"]:
                    Vr += st["Q"] * dt_taken
                    chi_w += chi * st["Q"] * dt_taken
                    chi_q += st["Q"] * dt_taken
                n_off += int(st["off_cup"])
                if keep_log and (st["active"] or st["Q_w"] > 0):
                    a = st["active"]
                    log.append((st["t"], float(a), st["Q_w"] / ML, st["Q"] / ML, st["z"] * 100,
                                st.get("U", np.nan), st.get("uh", (np.nan, np.nan))[0], st.get("uh", (np.nan, np.nan))[1],
                                st["vh"][0] * D_CUP, st["vh"][1] * D_CUP, st.get("r", np.nan) * 1e3,
                                st.get("d", np.nan) * 1e3, st.get("r1", np.nan) * 1e3, st.get("r2", np.nan) * 1e3,
                                chi if a else np.nan, dt_taken))
        except Exception as e:                       # recorded, never silently scored
            ok, err = False, f"{type(e).__name__}: {e}"
        if not filled.all():
            last = snap()
            snaps[~filled] = last
        L = sol.ledger()
        lg = np.array(log, float) if log else np.zeros((0, len(LOGCOLS)))
        act = lg[:, 1] > 0 if len(lg) else np.zeros(0, bool)

        def rng(col):
            if not act.any():
                return None
            v = lg[act, LOGCOLS.index(col)]
            return [float(np.nanmin(v)), float(np.nanmax(v))]

        led = dict(ok=ok and bool(filled.all()), error=err, t_start=t_start, T_model=float(sol.t),
                   layer_error=float(L["layer_error"]), milk_error=float(L["milk_error"]), layer=float(L["layer"]),
                   milk=float(L["milk"]), deposited=float(L["deposited"]),
                   V_wanted_mL=Vw / ML, V_received_mL=Vr / ML, chi_flow_weighted=chi_w / chi_q if chi_q > 0 else None,
                   frames_off_cup=n_off, ranges=dict(Q_mLs=rng("Q_mLs"), z_cm=rng("z_cm"), U_ms=rng("U_ms"),
                                                     r_mm=rng("r_mm"), d_mm=rng("d_mm"), chi=rng("chi")),
                   steps=int(sol.stats["steps"]), frames=int(sol.stats["frames"]), retries=int(sol.stats["retries"]),
                   max_substeps=int(sol.stats["max_substeps"]), wall_s=time.time() - t0w,
                   closure=self.closure(params), u_exit=inl.u_exit, tau_spout=inl.tau, N=N)
        out = dict(c=snaps, ledger=led)
        if keep_log:
            out["inlet_log"] = lg
            out["inlet_cols"] = LOGCOLS
        return out


# =========================================================================== unit tests
def _script_from_name(name):
    from . import calib as CB
    return CB.physical_script(name, preroll=0)


def test_inlet(names=VE.PATTERNS, dt=1e-3, verbose=True):
    """Inlet values on a fine grid against the formulas computed independently here, the hand velocity against a
    central difference of v3min_adapter.Inlet.at, and the spout lag against an independent ODE integration."""
    from scipy.integrate import solve_ivp
    rep = {}
    for name in names:
        sc = _script_from_name(name)
        inl = PhysInlet(sc, kappa_Q=FS["kappa_Q"])
        ts = np.arange(0.0, inl.T + 0.3, dt)
        e = np.array([math.cos(VE.script_yaw(sc)), math.sin(VE.script_yaw(sc))])
        err = dict(U=0.0, r=0.0, d=0.0, r1r2=0.0, S=0.0, uh=0.0, aspect=0.0, vh=0.0, cap=0.0, floor=0.0)
        n_act = 0
        kinks = np.concatenate([inl.starts, inl.starts[:-1] + EASE, inl.starts[1:] - EASE])
        for t in ts:
            st = inl.state(t)
            x, y, z, Qw = inl.path.at(t)
            # hand velocity vs a central difference of the wanted path (skip 2 ms around move boundaries and the
            # kinks of the 0.25 s wiggle ease-in/out, where the derivative jumps)
            if np.min(np.abs(kinks - t)) > 2e-3:
                h = 1e-6
                xa, ya = inl.path.at(t - h)[:2]; xb, yb = inl.path.at(t + h)[:2]
                err["vh"] = max(err["vh"], abs((xb - xa) / (2 * h) - st["vh"][0]), abs((yb - ya) / (2 * h) - st["vh"][1]))
            if not st["active"]:
                if Qw > 0:
                    assert st["off_cup"] or st["Q"] > 0, (name, t)
                continue
            n_act += 1
            Q = st["Q"]
            if Qw > 0:
                err["floor"] = max(err["floor"], max(0.0, Q_MIN - Q))
            else:
                assert Q >= Q_MIN - 1e-15
            U = math.sqrt(0.35 ** 2 + 2 * 9.81 * max(z, 0.0))
            u = np.array(st["vh"]) * 0.08 + 0.15 * e
            if np.hypot(*u) > min(0.40, U):
                u *= min(0.40, U) / np.hypot(*u)
            spd = math.hypot(U, np.hypot(*u))
            r = math.sqrt(Q / (math.pi * U))
            d = 2 * math.sqrt(Q / (math.pi * spd))
            I = st["inlet"]
            err["U"] = max(err["U"], abs(I.U_perp * 0.08 - U))
            err["r"] = max(err["r"], abs(math.sqrt(I.r1 * I.r2) * 0.08 - r))
            err["d"] = max(err["d"], abs(I.d_jet * 0.08 - d))
            err["aspect"] = max(err["aspect"], abs(I.r1 / I.r2 - spd / U))
            err["r1r2"] = max(err["r1r2"], abs(math.pi * I.r1 * I.r2 * 0.08 ** 2 - Q / U))
            err["S"] = max(err["S"], abs(I.S_eff - 2.0 * Q / 0.08 ** 3) / max(I.S_eff, 1e-30))
            err["uh"] = max(err["uh"], float(np.max(np.abs(np.array(I.u_in) * 0.08 - u))))
            err["cap"] = max(err["cap"], max(0.0, math.hypot(*I.u_in) * 0.08 - min(0.40, U)))
        # spout lag against an independent ODE integration, move by move, with the wanted flow written out from the
        # move records (Q0 -> Q1 linear in time), not taken from the inlet code
        y0, lag_err = 0.0, 0.0
        for k, mv in enumerate(inl.moves):
            a, b = inl.starts[k], inl.starts[k + 1]

            def rhs(t, yv, a=a, mv=mv):
                q = mv.Q0 + (mv.Q1 - mv.Q0) * (t - a) / mv.dur
                return [(q - yv[0]) / inl.tau]
            sol = solve_ivp(rhs, (a, b), [y0], dense_output=True, rtol=1e-10, atol=1e-14, max_step=0.002)
            for t in np.linspace(a, b, max(3, int((b - a) / 0.01)))[1:-1]:
                lag_err = max(lag_err, abs(inl.q_lag(t) - sol.sol(t)[0]))
            y0 = float(sol.y[0][-1])
        for t in np.linspace(inl.T, inl.T + 0.3, 31)[1:]:                    # free decay after the script
            lag_err = max(lag_err, abs(inl.q_lag(t) - y0 * math.exp(-(t - inl.T) / inl.tau)))
        lag_err /= ML
        rep[name] = dict(n_active=n_act, lag_err_mLs=lag_err, **{k: float(v) for k, v in err.items()})
        if verbose:
            print(name, {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in rep[name].items()}, flush=True)
        assert n_act > 0
        assert err["U"] < 1e-12 and err["r"] < 1e-12 and err["d"] < 1e-12 and err["r1r2"] < 1e-15, err
        assert err["aspect"] < 1e-9 and err["S"] < 1e-12 and err["uh"] < 1e-12 and err["cap"] < 1e-12, err
        assert err["vh"] < 1e-4 and err["floor"] == 0.0 and lag_err < 1e-3, (err, lag_err)
    return rep


def test_zero_flow(N=48, verbose=True):
    """A script with zero flow everywhere (the heart's path, flows 0) leaves the surface unchanged."""
    sc = _script_from_name("heart")
    for s in sc["stages"]:
        s["flow0"] = s["flow1"] = 0.0
    m = V05Model(FS)
    sol0 = Solver(m.solver_params({}), Numerics(N=N))
    l0, m0 = sol0.l.copy(), sol0.m.copy()
    t_out = np.array([0.0, 2.0, 5.0, 9.0])
    out = m.run(sc, {}, N, t_out)
    led = out["ledger"]
    dc = float(np.abs(out["c"]).max())
    if verbose:
        print("zero-flow", dict(max_c=dc, V_received_mL=led["V_received_mL"], layer_error=led["layer_error"],
                                milk_error=led["milk_error"], ok=led["ok"]), flush=True)
    assert led["ok"] and dc == 0.0 and led["V_received_mL"] == 0.0 and led["deposited"] == 0.0
    assert abs(led["layer"] - float(l0[sol0.g.mask].sum() * sol0.g.area)) < 1e-13 and led["milk"] == 0.0
    return dict(max_c=dc, **{k: led[k] for k in ("layer_error", "milk_error", "V_received_mL")})


def test_ledger(name="heart", N=64, T=4.0, verbose=True):
    """A real script (first T seconds, both closure sets): layer and milk ledgers close to round-off, output times hit
    exactly, c in [0, 1]."""
    sc = _script_from_name(name)
    rep = {}
    for tag, base in CLOSURE_SETS.items():
        m = V05Model(base)
        t_out = np.array([0.0, 0.37, 1.0, 1.013, 2.5, T])
        out = m.run(sc, {}, N, t_out)
        led = out["ledger"]
        rep[tag] = dict(layer_error=led["layer_error"], milk_error=led["milk_error"], T_model=led["T_model"],
                        V_received_mL=led["V_received_mL"], deposited=led["deposited"], chi=led["chi_flow_weighted"])
        if verbose:
            print("ledger", name, tag, rep[tag], flush=True)
        assert led["ok"], led["error"]
        assert abs(led["layer_error"]) < 1e-12 and abs(led["milk_error"]) < 1e-12
        assert abs(led["T_model"] - T) < 1e-9
        assert out["c"].min() >= -1e-9 and out["c"].max() <= 1 + 1e-9 and led["deposited"] > 0
    return rep


def test_preroll(verbose=True):
    """The tulip clip script (pre-roll 1.5 s) is run without its pre-roll: the inlet at axis time t equals the inlet of
    physical_script('tulip', preroll=0) at t - 1.5 s."""
    from . import calib as CB
    m = V05Model(FS)
    a, ta = m.inlet(CB.physical_script("tulip"))
    b, tb = m.inlet(CB.physical_script("tulip", preroll=0))
    assert abs(ta - 1.5) < 1e-9 and abs(tb) < 1e-12
    worst = 0.0
    for t in np.arange(0.0, b.T, 0.01):
        sa, sb = a.state(t), b.state(t)
        worst = max(worst, abs(sa["Q"] - sb["Q"]), abs(sa["x"] - sb["x"]), abs(sa["y"] - sb["y"]))
    if verbose:
        print("preroll", dict(t_start=ta, max_diff=worst), flush=True)
    assert worst < 1e-12
    return dict(t_start=ta, max_diff=worst)


def selftest():
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    return dict(inlet=test_inlet(), zero_flow=test_zero_flow(), ledger=test_ledger(), preroll=test_preroll())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        import json
        r = selftest()
        print(json.dumps(r, default=float))
        print("ALL TESTS PASSED")
