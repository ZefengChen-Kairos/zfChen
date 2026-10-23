"""Three-way diagnosis at a fine grid (default N 256): when the simulated tulip does not open, is it the surface
model, its parameters, or the pour model (the inlet the measured pitcher delivers)?

Gradient fits on the calibration pours (tulip, leaf) from one start closure, with fit.py's whole-video loss (every
observed frame, multi-scale residual, sigmoid temperature annealed 0.10 -> 0.03) and the exact adjoint (run.DiffPour):
  params      the closure's 9 numbers free in a WIDE box (PARAM_WIDE: physical or not), inlet as measured
  inlet       closure fixed; the inlet free within its measurement error (INLET_BOX 'err'): per pour a smooth
              correction of the landing point (dx, dy), the flow (Q e^aQ), the arrival speed (U_perp e^aU) and the
              horizontal arrival velocity (u_h + du), piecewise linear between knots -- the pitcher is taken out
  both        both of the above
  both_wide   closure wide + the inlet in a box 2.5-3x the measurement error (INLET_BOX 'wide'): can the surface
              model draw the tulip at all
The footprint follows the corrected Q, U_perp, u_h exactly as the arrival record defines it
(v05_pitcher_check.drvf_record: r = sqrt(Q / (pi U)), r1 = r sqrt(s / U), r2 = r sqrt(U / s), d = 2 sqrt(Q / (pi s)),
s = |(u_h, U)|, phi = direction of u_h), applied as ratios so that a zero correction reproduces the record bitwise;
the landing-point velocity V_j is recomputed (schedule.foot_velocity, no gradient through it: it only acts with the
closure switch vj).  The frame grid (frame lengths, quadrature, stream on / off) stays the measured one; the knot
spacing bounds the extra landing-point speed (reported as 'scan' = largest frame step over the solver's own bound).

Every run writes records.jsonl, state.npz (resumable), best.json, and at the end (render) per pour: eval.json (J of
render512.score48, the calibration's scale, and the petal separation of validate.part_separation over the last 30 %
of the observed frames), <p>_stills.jpg (real | target | start | best at key times), <p>_compare.mp4,
<p>_sim<N>.mp4, <p>_c128.npz, <p>_inlet.png / .npz (measured vs fitted inlet).  'summary' compares the runs and
applies the decision rule of README_diag256.md.
    python -m latte_imex.diff.diag run --mode both --init closures/bp256_trained.json --out DIR --N 256 --time-h 1.8
    python -m latte_imex.diff.diag summary --out ROOT                        # ROOT/<mode>/ for every mode run
    python -m latte_imex.diff.diag gradcheck --N 48 --T 1.0                  # adjoint vs finite differences, inlet chain

Measured inlet (mi27, SPEC27 rev 2 sections 3.5 item 7 and 6; every switch below defaults to the behaviour above):
  inlet       the start closure's key 'inlet' (0 record, 1 law, 2 replay; render512.load_closure -> sw['inlet']) or the
              environment LATTE_INLET (real_program.resolve_inlet).  Under an MI inlet (law / replay) the oracle boxes are
              'meas' / 'meas_wide' in place of 'err' / 'wide' (--box-family auto; SPEC27 6.2): per-knot 2-sigma bounds,
              the replay band inside the measured windows (measured_inlet.band), the law's calibration spread outside.
  --pset      the closure numbers freed in the params / both modes: PSETS['v05n'] = PARAM_WIDE (default; PKEYS and
              done.json unchanged) or PSETS['S1'] (the five F entries of the physics model S1kq, SPEC27 4.2).
  held-out    whenever the calibration pours' inlet is not 'record', the held-out pours run HELDOUT_INLET ('law', e0 0)
              through the per-pour override (real_program.INLET_OVERRIDE), and a held-out failure is fatal.
  oracle      best_closure.json of the inlet / both modes carries no 'inlet' field (SPEC27 6.3, PREREG27_2 D11: the
              oracle's corrections are diagnostic only, never an inlet or a closure input).
  V_j         with the closure switch vj (or kappa_tc) the landing-point velocity V_j = schedule.foot_velocity(x, y) is
              recomputed from the corrected landing points and its adjoint (InletCorr.vj_T, the transpose of the linear
              EMA) enters dL/d(dx, dy); the surface current (circ) stays refused with an inlet correction.
    python -m latte_imex.diff.diag run --mode params --pset S1 --init closures/mcd_S1kq_mi_N96.json --out DIR
    python -m latte_imex.diff.diag gradcheck --N 32 --T 1.0 --eps 1e-7 --inlet replay --box meas \\
        --init closures/mcd_S1kq_N96.json --keys tau_s,p_dep,kappa_c,B_dep
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from . import schedule as SC
from . import wsolver as W

CAL = ("tulip", "leaf")
HELDOUT = ("heart", "push_heart", "swan")
MODES = ("params", "inlet", "both", "both_wide")
# the closure's numbers, free whether physical or not (a superset of v05_adapter.SPACE, which is the physical box)
PARAM_WIDE = dict(cp=(0.1, 6.0, "log"), beta=(0.02, 30.0, "log"), nu=(1e-4, 0.05, "log"), kappa_Q=(0.2, 15.0, "log"),
                  kappa_c=(0.0, 3.0, "lin"), kappa_t=(0.0, 2.0, "lin"), kappa_r=(1e-3, 1.0, "log"),
                  B_dep=(30.0, 3e4, "log"), p_dep=(0.5, 12.0, "lin"))
PKEYS = tuple(PARAM_WIDE)
# mi27 (SPEC27 rev 2 sections 4.2, 6.1): the closure numbers a params / both run frees, selected by --pset.  'v05n' (the
# default) is PARAM_WIDE, so PKEYS and done.json are unchanged by default.  'S1': the five entries of the physics model
# S1kq that are F (fitted) under the measured inlet (kappa_c, p_dep, tau_s, kappa_Q, B_dep); kappa_t, kappa_r, mu_s,
# kappa_tc (0) and the crema numbers (cp, beta, nu, xi_b, lam_b) stay at the start closure's values.
PSETS = {"v05n": PARAM_WIDE,
         "S1": dict(kappa_c=(0.0, 3.0, "lin"), p_dep=(0.5, 12.0, "lin"), tau_s=(0.0, 0.12, "lin"),
                    kappa_Q=(0.5, 6.0, "log"), B_dep=(30.0, 3e4, "log"))}
PSET_DEFAULT = "v05n"
# inlet correction boxes: 'err' = the measured inlet's own uncertainty (pose 5-10 mm at the spout tip and ~1 deg of
# tilt = 30-50 % of the flow per degree; fall height +-5-10 mm of 15-55 mm = +-15-25 % of U_perp; arrival direction
# from the same pose); 'wide' = 2.5-3x that
# mi27 (SPEC27 rev 2 section 6.2), the boxes of the measured inlet (MI runs): every limit a 2-sigma per-knot bound
# (InletCorr.lim (6, K)).  The values below are the law-only frames' (high pours, gaps: the law's calibration spread, not a
# measurement sigma): landing +-4 mm (col25 sigma_V 2.1-2.4 mm with SYS 1 mm; law along / across sd 2.0-2.1 mm), Q x2
# (the weir's; measured early ratio 0.4-0.74), U_perp x1.19 (2 (0.035 dU p16-84 (+) 0.04 struct) / 0.52), u_h +-0.10 m/s
# (per-frame vector miss p84 0.087-0.094).  Inside the measured windows U_perp and u_h take 2 x the replay band
# (measured_inlet.band: GP sd (+) running systematic sd (+) structural), blended with the law values by the taper w of the
# knot (InletCorr._band_box); landing and Q are the same in both regions.  'meas_wide' multiplies every half-width by
# 'wide' = 2.5 (landing 10 mm, U_perp x1.19^2.5, u_h 0.25 m/s, the band ones likewise), except Q: x5 as SPEC27 6.2 states
# (VOL_TOL 2.0).  'meas' does not cover R1's straight-column reading (+0.15 to +0.3 m/s on u_h) by construction.
INLET_BOX = {"err": dict(dx_mm=8.0, Q_x=2.0, U_x=1.4, du_ms=0.10, knot_s=0.2),
             "wide": dict(dx_mm=20.0, Q_x=5.0, U_x=2.5, du_ms=0.30, knot_s=0.3),
             "meas": dict(dx_mm=4.0, Q_x=2.0, U_x=1.19, du_ms=0.10, knot_s=0.2, band=True, wide=1.0),
             "meas_wide": dict(dx_mm=10.0, Q_x=5.0, U_x=1.19 ** 2.5, du_ms=0.25, knot_s=0.2, band=True, wide=2.5)}
MEAS_BOX = "meas"                    # the 1x measurement box (the 2-sigma reference of 'meas_wide')
# --box-family: which boxes the inlet / both / both_wide modes use ('auto': 'meas' under an MI inlet, else 'err')
BOX_FAMILY = {"err": {"err": "err", "wide": "wide"}, "meas": {"err": "meas", "wide": "meas_wide"}}
HELDOUT_INLET = "law"                # held-out pours of an MI run (SPEC27 6.1, PREREG27_2 D2): MI-law, e0 0
IV = ("dx", "dy", "aQ", "aU", "dux", "duy")
# the pattern-phase milk volume of the corrected inlet stays within this factor of the measured pour's (the whole pour's
# 165 g is weighed; its pattern-phase share comes from the pour model): quadratic penalty W_VOL (ln excess)^2 beyond it
VOL_TOL = {"err": 1.25, "wide": 2.0, "meas": 1.25, "meas_wide": 2.0}
W_VOL = 1.0
PRIOR = 2e-6             # weak prior PRIOR sum (2u - 1)^2 on the inlet knots (see InletCorr.penalty)
MODE_SET = {"params": (True, None), "inlet": (False, "err"), "both": (True, "err"), "both_wide": (True, "wide")}
KEY_T = {"tulip": (1.92, 2.78, 4.02, 4.87, 13.21), "leaf": (2.21, 3.03, 4.41, 8.79)}
SEP_FRAC = 0.3           # petal separation over the last 30 % of the observed frames


# ----------------------------------------------------------------------------------------------- parameter box
def p_to_unit(P, ps=PARAM_WIDE):
    u = []
    for k in ps:
        lo, hi, sc = ps[k]
        v = min(max(float(P[k]), lo), hi)
        u.append(math.log(v / lo) / math.log(hi / lo) if sc == "log" else (v - lo) / (hi - lo))
    return np.array(u)


def p_from_unit(u, base, ps=PARAM_WIDE):
    P = dict(base)
    for k, x in zip(tuple(ps), np.clip(u, 0.0, 1.0)):
        lo, hi, sc = ps[k]
        P[k] = lo * (hi / lo) ** x if sc == "log" else lo + (hi - lo) * x
    return P


def p_dunit(P, ps=PARAM_WIDE):
    return np.array([float(P[k]) * math.log(ps[k][1] / ps[k][0]) if ps[k][2] == "log"
                     else ps[k][1] - ps[k][0] for k in ps])


# ----------------------------------------------------------------------------------------------- inlet correction
class InletCorr:
    """Smooth correction of one pour's arrival record (schedule arrays) on knots; unit box u (6, K), 0.5 = none."""

    def __init__(self, D, box):
        S = D.S
        self.S0 = {k: np.array(S[k], float).copy() for k in W.SCHED}
        self.active = np.asarray(S["active"], bool)
        self.act = self.active & (self.S0["Q"] > 0) & (self.S0["U"] > 0)
        self.fdt = np.asarray(S["fdt"], float)
        self.t = np.cumsum(self.fdt) - 0.5 * self.fdt
        b = INLET_BOX[box]
        self.box, self.b = box, b
        self.L_m = float(D.pour.info["L_m"])                 # metres per solver length unit (cup radius 0.49)
        kn = np.arange(0.0, self.t[-1] + b["knot_s"], b["knot_s"])
        self.knots, self.K = kn, len(kn)
        self.j = np.clip(np.searchsorted(kn, self.t, side="right") - 1, 0, self.K - 2)
        self.w = np.clip((self.t - kn[self.j]) / (kn[self.j + 1] - kn[self.j]), 0.0, 1.0)
        dx = b["dx_mm"] * 1e-3 / self.L_m
        du = b["du_ms"] / self.L_m
        # half-widths per knot (6, K); for 'err' / 'wide' every knot has the same six numbers (the products below are
        # then the same IEEE operations as with a (6,) array broadcast, bit for bit)
        self.lim = np.repeat(np.array([dx, dx, math.log(b["Q_x"]), math.log(b["U_x"]), du, du])[:, None], self.K, 1)
        self.lim1, self.kw, self.windows = None, None, None
        if b.get("band"):
            self._band_box(D)
        self.h, self.safety = 1.0 / D.N, 0.5                 # solver.Numerics.scan_safety
        self.cur = None

    def _band_box(self, D):
        """Per-knot 2-sigma box of the measured inlet (SPEC27 rev 2 section 6.2).  Inside the measured windows the
        U_perp and u_h half-widths are 2 x the replay band (measured_inlet.band at the knot: GP sd (+) running systematic
        sd (+) structural; U_perp relative to the schedule's own U_perp near the knot, ln(1 + 2 sigma_w / U)); outside
        (law-only frames) the law's calibration spread of INLET_BOX['meas']; at a knot inside the taper the two are
        blended with its weight w (w = 1 inside a window, 0 beyond the taper: exactly the two regions).  Landing and Q:
        the 'meas' values everywhere.  self.lim1 holds the 1x 'meas' box (the measurement's 2 sigma), self.lim that x
        INLET_BOX[box]['wide'] (Q: ln Q_x of the box)."""
        from ..rig import measured_inlet as MI
        p, mode = D.p, getattr(D, "inlet", "record")
        if p not in MI.MI_POURS or mode not in ("law", "replay"):
            raise ValueError(f"inlet box {self.box!r}: the measured windows and the replay band exist for {MI.MI_POURS} "
                             f"under inlet 'law' / 'replay' only (pour {p}, inlet {mode})")
        E = getattr(D.pour, "E", None)
        if E is None:
            raise ValueError(f"inlet box {self.box!r}: the pour carries no MI emission (tulip_feas.Pour.E)")
        K = MI.load(p)
        L, kn, b1 = self.L_m, self.knots, INLET_BOX[MEAS_BOX]
        self.windows, self.gap = [tuple(w) for w in K["windows"]], float(K.get("gap_s", MI.GAP_S))
        wk = MI.taper(kn, self.windows, self.gap)
        bd = MI.band(p, kn, K=K, E=E)
        half = 0.5 * b1["knot_s"]
        ta, Ua = self.t[self.act], self.S0["U"][self.act] * L
        Uref = np.array([float(np.median(Ua[np.abs(ta - x) <= half])) if np.any(np.abs(ta - x) <= half) else np.nan
                         for x in kn])
        okU = np.isfinite(Uref) & (Uref > 0)
        lawU, lawdu = math.log(b1["U_x"]), b1["du_ms"]
        measU = np.where(okU, np.log1p(2.0 * bd["wf"] / np.where(okU, Uref, 1.0)), lawU)
        lim1 = np.repeat(np.array([b1["dx_mm"] * 1e-3 / L, b1["dx_mm"] * 1e-3 / L, math.log(b1["Q_x"]), lawU,
                                   lawdu / L, lawdu / L])[:, None], self.K, 1)
        lim1[3] = wk * measU + (1.0 - wk) * lawU
        lim1[4] = (wk * 2.0 * bd["ux"] + (1.0 - wk) * lawdu) / L
        lim1[5] = (wk * 2.0 * bd["uy"] + (1.0 - wk) * lawdu) / L
        s = float(self.b.get("wide", 1.0))
        lim = lim1.copy()
        lim[[0, 1, 3, 4, 5]] *= s
        lim[2] = math.log(self.b["Q_x"])
        self.lim1, self.lim, self.kw, self.Uref, self.band_q = lim1, lim, wk, Uref, bd

    def box_info(self):
        """The per-knot box in physical units (band boxes; for the run directory and the report)."""
        if self.kw is None:
            return None
        L = self.L_m

        def phys(lim):
            return dict(dx_mm=(lim[0] * L * 1e3).tolist(), Q_x=np.exp(lim[2]).tolist(), U_x=np.exp(lim[3]).tolist(),
                        dux_ms=(lim[4] * L).tolist(), duy_ms=(lim[5] * L).tolist())
        inw, law = self.kw >= 1.0, self.kw <= 0.0
        med = lambda a, m: float(np.median(a[m])) if m.any() else None
        return dict(box=self.box, knots=self.knots.tolist(), w=self.kw.tolist(), windows=self.windows, gap_s=self.gap,
                    Uref_ms=[None if not np.isfinite(x) else float(x) for x in self.Uref],
                    band={q: v.tolist() for q, v in self.band_q.items()}, lim=phys(self.lim), lim_meas=phys(self.lim1),
                    median_windows=dict(U_x=med(np.exp(self.lim[3]), inw), dux_ms=med(self.lim[4] * L, inw),
                                        duy_ms=med(self.lim[5] * L, inw)),
                    median_law_only=dict(U_x=med(np.exp(self.lim[3]), law), dux_ms=med(self.lim[4] * L, law),
                                         duy_ms=med(self.lim[5] * L, law)),
                    knots_in_windows=int(inw.sum()), knots_law_only=int(law.sum()))

    def zero(self):
        return np.full((6, self.K), 0.5)

    def theta(self, u):
        return (2.0 * np.asarray(u, float) - 1.0) * self.lim

    def frames(self, th):
        return th[:, self.j] * (1.0 - self.w) + th[:, self.j + 1] * self.w

    def frames_T(self, g):
        out = np.zeros((6, self.K))
        for v in range(6):
            np.add.at(out[v], self.j, g[v] * (1.0 - self.w))
            np.add.at(out[v], self.j + 1, g[v] * self.w)
        return out

    @staticmethod
    def _logs(Q, U, ux, uy):
        v2 = ux * ux + uy * uy
        s2 = U * U + v2
        return dict(r1=0.5 * np.log(Q) + 0.25 * np.log(s2) - np.log(U), r2=0.5 * np.log(Q) - 0.25 * np.log(s2),
                    d=0.5 * np.log(Q) - 0.25 * np.log(s2), v2=v2, s2=s2)

    def apply(self, u):
        """Corrected schedule (dict of numpy).  u = zero() reproduces the measured record bitwise."""
        c = self.frames(self.theta(u))
        a, S0 = self.act, self.S0
        out = {k: v.copy() for k, v in S0.items()}
        Q0, U0, ux0, uy0 = S0["Q"][a], S0["U"][a], S0["ux"][a], S0["uy"][a]
        Q, U = Q0 * np.exp(c[2][a]), U0 * np.exp(c[3][a])
        ux, uy = ux0 + c[4][a], uy0 + c[5][a]
        old, new = self._logs(Q0, U0, ux0, uy0), self._logs(Q, U, ux, uy)
        out["x"][a] = S0["x"][a] + c[0][a]
        out["y"][a] = S0["y"][a] + c[1][a]
        out["Q"][a], out["U"][a], out["ux"][a], out["uy"][a] = Q, U, ux, uy
        for k in ("r1", "r2", "d"):
            out[k][a] = S0[k][a] * np.exp(new[k] - old[k])
        out["phi"][a] = S0["phi"][a] + (np.arctan2(uy, ux) - np.arctan2(uy0, ux0))
        out["vjx"], out["vjy"] = SC.foot_velocity(out["x"], out["y"], self.active.astype(float), self.fdt)
        r = dict(Q=Q, U=U, ux=ux, uy=uy, v2=new["v2"], s2=new["s2"], r1=out["r1"][a], r2=out["r2"][a], d=out["d"][a])
        self.cur = r
        return out

    def vj_T(self, Gvx, Gvy, tau=None):
        """Adjoint of schedule.foot_velocity (V_j from the landing points; linear in x, y): dL/dx, dL/dy (per frame) from
        dL/dV_j (Gvx, Gvy).  Forward, for k >= 1: V_k = (1 - a_k) V_(k-1) + a_k (x_k - x_(k-1)) / max(fdt_(k-1), 1e-6),
        a_k = min(1, fdt_(k-1) / tau), when frames k and k - 1 are both active, else V_k = 0 (V_0 = 0).  Backward: the
        adjoint of V_k is Gv_k plus (1 - a_(k+1)) times that of V_(k+1) when V_(k+1) continued from V_k; it adds
        +a_k / fdt_(k-1) to dL/dx_k and -a_k / fdt_(k-1) to dL/dx_(k-1)."""
        from .schedule import V_FOOT_TAU
        tau = V_FOOT_TAU if tau is None else tau
        act, fdt = self.active, self.fdt
        n = len(fdt)
        gx, gy = np.zeros(n), np.zeros(n)
        lx = ly = 0.0
        for k in range(n - 1, 0, -1):
            lx += float(Gvx[k])
            ly += float(Gvy[k])
            if act[k] and act[k - 1]:
                a = min(1.0, fdt[k - 1] / tau)
                c = a / max(fdt[k - 1], 1e-6)
                gx[k] += c * lx; gx[k - 1] -= c * lx
                gy[k] += c * ly; gy[k - 1] -= c * ly
                lx *= 1.0 - a
                ly *= 1.0 - a
            else:
                lx = ly = 0.0
        return gx, gy

    def chain(self, G, vj=True):
        """dL/du (6, K) from the per-frame schedule gradient G (dict of numpy) at the last apply().  The landing-point
        velocity V_j (schedule arrays vjx, vjy, recomputed by apply() from the corrected x, y) enters dL/d(dx, dy)
        through vj_T whenever the solver read it (closure switch vj or kappa_tc: G['vjx'], G['vjy'] not all zero; with
        vj = kappa_tc = 0 they are exactly 0 and the term is skipped, so that path is unchanged bit for bit).  vj=False
        leaves it out (gradcheck reports its share)."""
        a, c = self.act, self.cur
        n = len(self.t)
        g = np.zeros((6, n))
        Gx = {k: np.asarray(G[k], float)[a] for k in W.SCHED}
        R1, R2, Dd = Gx["r1"] * c["r1"], Gx["r2"] * c["r2"], Gx["d"] * c["d"]
        g[0][a], g[1][a] = Gx["x"], Gx["y"]
        Gvx, Gvy = np.asarray(G["vjx"], float), np.asarray(G["vjy"], float)
        if vj and (np.any(Gvx != 0.0) or np.any(Gvy != 0.0)):
            vx, vy = self.vj_T(Gvx, Gvy)
            g[0][a] += vx[a]
            g[1][a] += vy[a]
        g[2][a] = Gx["Q"] * c["Q"] + 0.5 * (R1 + R2 + Dd)
        s = c["U"] ** 2 / c["s2"]
        g[3][a] = Gx["U"] * c["U"] + (0.5 * s - 1.0) * R1 - 0.5 * s * (R2 + Dd)
        okv = c["v2"] > 1e-24
        iv2 = np.where(okv, 1.0 / np.where(okv, c["v2"], 1.0), 0.0)
        com = 0.5 * (R1 - R2 - Dd) / c["s2"]
        g[4][a] = Gx["ux"] + c["ux"] * com - c["uy"] * iv2 * Gx["phi"]
        g[5][a] = Gx["uy"] + c["uy"] * com + c["ux"] * iv2 * Gx["phi"]
        return self.frames_T(g) * (2.0 * self.lim)

    def penalty(self, u, prior=PRIOR):
        """Host-side terms of the inlet correction: (i) the pattern-phase volume within VOL_TOL of the measured one,
        W_VOL (|ln V/V0| - ln VOL_TOL)^2 beyond it; (ii) a weak prior `prior` sum (2u - 1)^2 that keeps the knots the
        video does not constrain at the measured inlet (Adam moves every coordinate by ~lr per step whatever the size of
        its gradient, so without it those knots drift to the box edges and inflate the correction statistics).
        Returns (value, dL/du (6, K), V / V0)."""
        c = self.frames(self.theta(u))
        a = self.act
        w = self.S0["Q"][a] * self.fdt[a]
        e = np.exp(c[2][a])
        V0, V = float(w.sum()), float((w * e).sum())
        gu = np.zeros((6, self.K))
        pen = 0.0
        if V0 > 0:
            lv = math.log(V / V0)
            ex = abs(lv) - math.log(VOL_TOL[self.box])
            if ex > 0:
                pen = W_VOL * ex * ex
                g = np.zeros((6, len(self.t)))
                g[2][a] = W_VOL * 2.0 * ex * math.copysign(1.0, lv) * (w * e) / V
                gu = self.frames_T(g) * (2.0 * self.lim)
        z = 2.0 * np.asarray(u, float) - 1.0
        pen += prior * float((z * z).sum())
        gu = gu + 4.0 * prior * z
        return pen, gu, (V / V0 if V0 > 0 else 1.0)

    def scan_ratio(self, S):
        """Largest landing-point step per frame over the solver's own bound scan_safety min(h, r1, r2).  Not ~1 at fine
        grids: the uncorrected N 256 schedules reach about 5.6-5.8 (MI-R) and 7.6-10.7 (record R1) on their largest
        step (99 % of the steps below 3); a rise above the run's own iteration-0 value is what matters."""
        a = self.active
        both = a[1:] & a[:-1]
        if not both.any():
            return 0.0
        step = np.hypot(np.diff(S["x"]), np.diff(S["y"]))[both]
        lim = self.safety * np.minimum(self.h, np.minimum(S["r1"], S["r2"]))[1:][both]       # schedule.build's bound
        return float(np.max(step / lim))

    def series(self, S):
        """Measured vs corrected inlet on the active frames, physical units (for the figure)."""
        a, L = self.act, self.L_m
        S0 = self.S0
        ser = dict(t=self.t[a], Q0=S0["Q"][a] * L ** 3 * 1e6, Q=S["Q"][a] * L ** 3 * 1e6,
                   dx=(S["x"][a] - S0["x"][a]) * L * 1e3, dy=(S["y"][a] - S0["y"][a]) * L * 1e3,
                   U0=S0["U"][a] * L, U=S["U"][a] * L,
                   uh0=np.hypot(S0["ux"][a], S0["uy"][a]) * L, uh=np.hypot(S["ux"][a], S["uy"][a]) * L,
                   du=np.hypot(S["ux"][a] - S0["ux"][a], S["uy"][a] - S0["uy"][a]) * L)
        if self.kw is not None:                  # band boxes: region (taper w at the frame) and |correction| / 'meas' 2 sigma
            from ..rig.measured_inlet import taper
            ser["w"] = taper(self.t[a], self.windows, self.gap)
            l1 = self.frames(self.lim1)[:, a]
            c = np.array([S["x"][a] - S0["x"][a], S["y"][a] - S0["y"][a], np.log(S["Q"][a] / S0["Q"][a]),
                          np.log(S["U"][a] / S0["U"][a]), S["ux"][a] - S0["ux"][a], S["uy"][a] - S0["uy"][a]])
            ser["r_meas"] = np.abs(c) / l1
        return ser


def assign(D, S):
    for k in W.SCHED:
        D.Sw[k].assign(np.asarray(S[k], float))
        D.S[k][:] = S[k]


# ----------------------------------------------------------------------------------------------- one pour (worker side)
def make_pour(p, sw, kw):
    """DiffPour of pour p with the closure switches sw (inlet: the closure level of real_program.resolve_inlet) and
    kw['box'] (inlet correction box or None).  kw['inlet_override'] (held-out pours of an MI run, HELDOUT_INLET) is set
    as real_program.INLET_OVERRIDE[p] and stays set for the pour's lifetime (render's N 48 scoring build of the same
    pour reads it; Pour.close() removes it in-process, a worker process ends with it)."""
    from .run import DiffPour
    ov = kw.get("inlet_override")
    if ov is not None:
        from ..rig import real_program as RP
        RP.INLET_OVERRIDE[p] = RP.inlet_mode(ov)
    D = DiffPour(p, N=kw["N"], device=kw["device"], T_max=kw.get("T"), w_gap=2.0, tau=0.10, loss="video",
                 scales=(0.3, 0.15, 0.0), e0=sw["e0"], exit_cal=sw["exit_cal"], upwell=sw["upwell"], circ=sw["circ"],
                 circ_shape=sw["circ_shape"], milk=sw.get("milk", 0), sink=sw.get("sink", 0),
                 visc_tr=sw.get("visc_tr", 0), inlet=sw.get("inlet"), log=lambda *x: print(*x, flush=True))
    if ov is not None and D.inlet != RP.inlet_mode(ov):
        raise RuntimeError(f"{p}: inlet override {ov!r} not applied (built {D.inlet!r})")
    D.inl = InletCorr(D, kw["box"]) if kw.get("box") else None
    return D


def _pour_info(D):
    info = dict(s1=D.s1, K=(D.inl.K if D.inl else 0), n=int(D.S["n"]))
    if getattr(D, "inlet", "record") != "record":                # MI runs only (the default info is unchanged)
        info["inlet"] = dict(mode=D.inlet, source=D.pour.inlet_source)
        if D.inl is not None and D.inl.kw is not None:
            info["box"] = D.inl.box_info()
    return info


def full_P(D, num, sw):
    P = dict(D.s1)
    if not sw["circ"]:
        P.update(kappa_M=0.0, eps_f=0.0, kappa_w=0.0)
    P.update({k: v for k, v in num.items() if k in W.PARAMS})
    return P


def step(D, P, tau, u, prior=PRIOR):
    scan, pen, vol = 0.0, 0.0, 1.0
    if D.inl is not None and u is not None:
        S = D.inl.apply(u)
        assign(D, S)
        scan = D.inl.scan_ratio(S)
    D.tau = tau
    L, g, _ = D.loss_grad(P)
    gu = None
    if D.inl is not None and u is not None:
        gu = D.inl.chain(D.sched_grad)
        pen, gp, vol = D.inl.penalty(u, prior)
        L, gu = L + pen, gu + gp
    gvj = float(np.abs(D.sched_grad["vjx"]).sum() + np.abs(D.sched_grad["vjy"]).sum())
    viou = float(np.mean([v for _, v in D.vmetrics])) if D.vmetrics else float("nan")
    return dict(L=float(L), g=g, gu=gu, viou=viou, scan=scan, gvj=gvj, pen=pen, vol=vol, timing=D.timing)


def separation(D, ML, frac=SEP_FRAC):
    """validate.part_separation summed over the last `frac` of the observed frames: pairs of video white parts
    (>= topo_metric.BIG_COMP_MM2) kept apart / merged / missing in the simulation, same function for both."""
    from . import validate as V
    from ..rig import topo_metric as TM
    obs = D.obs
    ts = np.array([o["t"] for o in obs])
    t0 = ts[-1] - frac * (ts[-1] - ts[0])
    msk = D.geo.mask_np
    cell_mm = float(D.pour.info["L_m"]) * 1e3 / D.N
    big = TM.BIG_COMP_MM2 / cell_mm ** 2
    tot = dict(pairs=0, kept=0, merged=0, missing=0, frames=0)
    for o in obs:
        if o["t"] < t0 or int(o["k"]) not in ML:
            continue
        m, l = ML[int(o["k"])]
        c = np.where(msk, m / np.where(msk, l, 1.0), 0.0)
        r = V.part_separation(c, np.asarray(o["wo_np"], float), np.asarray(o["vo_np"], bool) & msk, big)
        for k in ("pairs", "kept", "merged", "missing"):
            tot[k] += r[k]
        tot["frames"] += 1
    tot["sep"] = tot["kept"] / tot["pairs"] if tot["pairs"] else float("nan")
    return tot


def stills(p, out, runs, labels, key_t=None):
    """Rows: real colour | video target | each run (C dict, t_end, mask); columns: key times."""
    from PIL import Image, ImageDraw
    from . import render512 as RR
    from ..rig.real_program import REPO
    F, Fs, cjk = RR._fonts()
    Z = np.load(os.path.join(REPO, "raw_data", "rig", "video_obs", f"{p}.npz"))
    tv, img, Wv, Vd = Z["t"], Z["img"], Z["w"].astype(float) / 255.0, Z["valid"]
    t_last = min(tv[-1], min(r[0][max(r[1])] for r in runs))
    key_t = key_t or KEY_T.get(p) or tuple(np.round(np.linspace(0.25, 1.0, 5) * t_last, 2))
    key_t = [min(t, t_last) for t in key_t]
    H, pad, left, top = 256, 8, 170, 30
    s = np.linspace(-RR.EXT, RR.EXT, H)
    Uu, Vv = np.meshgrid(s, s)
    disc = Uu ** 2 + Vv ** 2 <= 1.0
    rows = 2 + len(runs)
    Wd = left + len(key_t) * (H + pad)
    Ht = top + rows * (H + pad)
    canvas = np.full((Ht, Wd, 3), 18, np.uint8)
    for ci, t in enumerate(key_t):
        x = left + ci * (H + pad)
        j = int(np.argmin(np.abs(tv - t)))
        a = np.asarray(Image.fromarray(img[j]).resize((H, H), Image.BICUBIC), float); a[~disc] = RR.CUP
        wv = np.asarray(Image.fromarray((Wv[j] * 255).astype(np.uint8)).resize((H, H), Image.BILINEAR), float) / 255
        vv = np.asarray(Image.fromarray(Vd[j].astype(np.uint8) * 255).resize((H, H), Image.NEAREST)) > 127
        b = RR.latte(wv); b[~vv] = RR.HIDDEN; b[~disc] = RR.CUP
        panels = [a, b]
        for t_end, C, msk in runs:
            ks = np.array(sorted(C))
            k = ks[int(np.argmin(np.abs(t_end[ks] - t)))]
            c = RR.latte(RR.to_H(C[k], msk, H)); c[~disc] = RR.CUP
            panels.append(c)
        for ri, pa in enumerate(panels):
            y = top + ri * (H + pad)
            canvas[y:y + H, x:x + H] = np.clip(pa, 0, 255).astype(np.uint8)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    heads = (["真实视频", "视频白度目标"] if cjk else ["real video", "video target"]) + list(labels)
    for ri, h in enumerate(heads):
        d.text((6, top + ri * (H + pad) + H // 2 - 10), h, fill=(255, 230, 150), font=Fs)
    for ci, t in enumerate(key_t):
        d.text((left + ci * (H + pad) + 4, 6), f"t = {t:.2f} s", fill=(255, 255, 255), font=Fs)
    im.save(os.path.join(out, f"{p}_stills.jpg"), quality=90)


def inlet_figure(p, out, ser, box):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    np.savez_compressed(os.path.join(out, f"{p}_inlet.npz"), **ser)
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    t = ser["t"]
    ax[0].plot(t, ser["Q0"], ".", ms=1.5, label="measured pour model"); ax[0].plot(t, ser["Q"], ".", ms=1.5, label="fitted")
    ax[0].set_ylabel("Q [mL/s]"); ax[0].legend(fontsize=8)
    ax[1].plot(t, ser["dx"], ".", ms=1.5, label="dx"); ax[1].plot(t, ser["dy"], ".", ms=1.5, label="dy")
    ax[1].axhline(INLET_BOX[box]["dx_mm"], color="k", lw=0.6); ax[1].axhline(-INLET_BOX[box]["dx_mm"], color="k", lw=0.6)
    ax[1].set_ylabel("landing point shift [mm]"); ax[1].legend(fontsize=8)
    ax[2].plot(t, ser["U"] / np.maximum(ser["U0"], 1e-9), ".", ms=1.5); ax[2].set_ylabel("U_perp fitted / measured")
    ax[3].plot(t, ser["uh0"], ".", ms=1.5, label="|u_h| measured"); ax[3].plot(t, ser["uh"], ".", ms=1.5, label="|u_h| fitted")
    ax[3].plot(t, ser["du"], ".", ms=1.5, label="|du|"); ax[3].set_ylabel("[m/s]"); ax[3].legend(fontsize=8)
    ax[3].set_xlabel("t [s]")
    for x in ax:
        x.grid(alpha=0.3)
        if "w" in ser:                                           # measured windows (band boxes): shaded
            x.fill_between(t, 0, 1, where=ser["w"] >= 1.0, transform=x.get_xaxis_transform(), color="g", alpha=0.08,
                           lw=0)
    fig.suptitle(f"{p}: inlet correction (box '{box}')")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{p}_inlet.png"), dpi=110)
    plt.close(fig)


def inlet_stats(ser):
    """Correction statistics over the active frames; for the band boxes ('meas', 'meas_wide') also separately over the
    measured windows (w = 1) and the law-only frames (w = 0), SPEC27 6.3 / PREREG27_2 D11, with the fraction of frames
    whose correction stays within the measurement's 2 sigma ('meas' box; read in the windows only)."""
    if ser is None or len(ser["t"]) == 0:
        return None
    out = _istats(ser, np.ones(len(ser["t"]), bool))
    if "w" in ser:
        for name, m in (("windows", ser["w"] >= 1.0), ("law_only", ser["w"] <= 0.0)):
            st = _istats(ser, m)
            if st is not None:
                r = ser["r_meas"][:, m]
                st.update(frames=int(m.sum()), within_meas_2sigma=float(np.mean(np.all(r <= 1.0 + 1e-9, axis=0))),
                          within_meas_2sigma_per_var={v: float(np.mean(r[i] <= 1.0 + 1e-9)) for i, v in enumerate(IV)})
            out[name] = st
    return out


def _istats(ser, m):
    if not m.any():
        return None
    sh = np.hypot(ser["dx"][m], ser["dy"][m])
    qr = ser["Q"][m] / np.maximum(ser["Q0"][m], 1e-12)
    ur = ser["U"][m] / np.maximum(ser["U0"][m], 1e-12)
    du = ser["du"][m]
    return dict(shift_mm_p50=float(np.median(sh)), shift_mm_p90=float(np.percentile(sh, 90)), shift_mm_max=float(sh.max()),
                Q_ratio_p10=float(np.percentile(qr, 10)), Q_ratio_p50=float(np.median(qr)), Q_ratio_p90=float(np.percentile(qr, 90)),
                Q_total_ratio=float(ser["Q"][m].sum() / max(ser["Q0"][m].sum(), 1e-12)),
                U_ratio_p10=float(np.percentile(ur, 10)), U_ratio_p90=float(np.percentile(ur, 90)),
                du_ms_p50=float(np.median(du)), du_ms_p90=float(np.percentile(du, 90)))


def render(D, sw, out, P0, P1, u1, label, every, movie, T):
    """Start (measured inlet, start closure) and result (fitted closure + inlet) forward; scores, stills, movies."""
    from . import render512 as RR
    p = D.p
    D48 = RR.make_pour(p, sw, 48, "cpu", T)
    if D.inl is not None:
        assign(D, D.inl.S0)
    t_end, C0, ML0 = RR.run_pour(D, P0, sw, every)
    r0 = dict(RR.score48(p, sw, D, ML0, T, D48=D48), **separation(D, ML0))
    ser = None
    if D.inl is not None and u1 is not None:
        S = D.inl.apply(u1)
        assign(D, S)
        ser = D.inl.series(S)
        try:
            inlet_figure(p, out, ser, D.inl.box)
        except Exception as e:                                   # noqa: BLE001
            print(f"{p}: inlet figure failed ({e})", flush=True)
    _, C1, ML1 = RR.run_pour(D, P1, sw, every)
    r1 = dict(RR.score48(p, sw, D, ML1, T, D48=D48), **separation(D, ML1))
    msk = D.geo.mask_np
    # the start panel names the inlet it was built with (mi27 update21: 测量入口 = the measured inlet MI)
    lab0 = {"replay": "起点（测量入口 MI-R）", "law": "起点（MI-L，只有出口定律）"}.get(getattr(D, "inlet", None),
                                                                                "起点（旧入口 R1）")
    errs = []                                                    # a figure failing must not lose the scores

    def guard(name, f):
        try:
            f()
        except Exception as e:                                   # noqa: BLE001
            errs.append(f"{name}: {type(e).__name__}: {e}")
    guard("stills", lambda: stills(p, out, [(t_end, C0, msk), (t_end, C1, msk)], [lab0, label]))
    if movie:
        guard("movies", lambda: RR.movies(p, out, (t_end, C1, msk, D.N), (t_end, C0, msk, D.N), label, lab0, every))

    def c128():
        tobs = np.array([o["t"] for o in D.obs])
        c = np.array([np.clip(RR.to_H(C1[int(o["k"])], msk, 128), 0, 1) for o in D.obs])
        np.savez_compressed(os.path.join(out, f"{p}_c128.npz"), t=tobs, c=(c * 255 + 0.5).astype(np.uint8))
    guard("c128", c128)
    if D.inl is not None:
        assign(D, D.inl.S0)
    return dict(start=r0, best=r1, inlet=inlet_stats(ser), errors=errs)


def _worker(conn, p, sw, kw):
    import traceback
    try:
        D = make_pour(p, sw, kw)
        conn.send(("ok", _pour_info(D)))
    except Exception:
        conn.send(("err", traceback.format_exc()))
        return
    while True:
        msg = conn.recv()
        if msg is None:
            return
        try:
            if msg[0] == "step":
                conn.send(("ok", step(D, *msg[1:])))
            elif msg[0] == "render":
                conn.send(("ok", render(D, sw, *msg[1:])))
        except Exception as e:
            conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


class Pour:
    """One pour in a worker process (CUDA: own context, the backward replay is launch-bound) or in-process (CPU)."""

    def __init__(self, p, sw, kw, process):
        self.p, self.process = p, process
        if process:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self.conn, child = ctx.Pipe()
            self.proc = ctx.Process(target=_worker, args=(child, p, sw, kw), daemon=True)
            self.proc.start()
            st, info = self.conn.recv()
            if st != "ok":
                raise RuntimeError(f"{p} worker failed to start:\n{info}")
        else:
            self.D = make_pour(p, sw, kw)
            info = _pour_info(self.D)
        self.s1, self.K, self.n = info["s1"], info["K"], info["n"]
        self.sw, self.kw, self.info = sw, kw, info

    def send(self, *msg):
        if self.process:
            try:
                self.conn.send(msg)
            except (BrokenPipeError, EOFError, OSError) as e:
                self._dead = f"{type(e).__name__}: {e}"
        else:
            try:
                fn = step if msg[0] == "step" else (lambda D, *a: render(D, self.sw, *a))
                self._res = ("ok", fn(self.D, *msg[1:]))
            except RuntimeError as e:
                self._res = ("err", str(e))

    def recv(self):
        if self.process:
            if getattr(self, "_dead", None):
                raise RuntimeError(f"{self.p}: worker gone ({self._dead})")
            try:
                st, r = self.conn.recv()
            except (EOFError, OSError) as e:
                self._dead = f"{type(e).__name__}: {e}"
                raise RuntimeError(f"{self.p}: worker gone ({self._dead})")
        else:
            st, r = self._res
        if st != "ok":
            raise RuntimeError(f"{self.p}: {r}")
        return r

    def close(self):
        if self.process:
            try:
                self.conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            self.proc.join(timeout=60)
        elif self.kw.get("inlet_override") is not None:            # in-process: the held-out override ends with the pour
            from ..rig import real_program as RP
            RP.INLET_OVERRIDE.pop(self.p, None)


# ----------------------------------------------------------------------------------------------- driver
def _log_to(out):
    fn = os.path.join(out, "run.log")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        with open(fn, "a") as f:
            f.write(s + "\n")
    return log


def progress_png(out, recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    it = [r["it"] for r in recs]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(it, [r["loss"] for r in recs], "k.-"); ax[0].set_title("video loss (tau annealed)")
    for p in recs[0]["per"]:
        ax[1].plot(it, [r["per"][p]["viou"] for r in recs], ".-", label=p)
    ax[1].plot(it, [r["viou"] for r in recs], "k.-", label="mean"); ax[1].legend(fontsize=8); ax[1].set_title("video IoU")
    for x in ax:
        x.grid(alpha=0.3); x.set_xlabel("iteration")
    fig.tight_layout(); fig.savefig(os.path.join(out, "progress.png"), dpi=100); plt.close(fig)


def _atomic_json(obj, fn, **kw):
    tmp = fn + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=float, **kw)
    os.replace(tmp, fn)


def run(a):
    from .render512 import load_closure
    mode = a.mode
    free_p, box = MODE_SET[mode]
    out = a.out
    os.makedirs(out, exist_ok=True)
    log = _log_to(out)
    if os.path.exists(os.path.join(out, "done.json")):
        log(f"{mode}: done before ({out}/done.json), skipped")
        return
    num, sw = load_closure(a.init)
    from ..rig import real_program as RP
    from .fit import MILK_GRAD
    cal = tuple(a.cal.split(","))
    pset = getattr(a, "pset", PSET_DEFAULT)
    ps = PSETS[pset]
    keys = tuple(ps)
    # inlet of the calibration pours (override > closure key > env LATTE_INLET > 'record'); MI = law or replay
    cal_inlet = {p: RP.resolve_inlet(p, num) for p in cal}
    mi = any(m != "record" for m, _ in cal_inlet.values())
    fam = getattr(a, "box_family", "auto")
    fam = ("meas" if mi else "err") if fam == "auto" else fam
    box = BOX_FAMILY[fam][box] if box else None
    if box and sw["circ"]:
        raise SystemExit("diag.py: a start closure with the surface current (circ) is not supported with an inlet "
                         "correction: its fields are not re-derived from the corrected inlet")
    if W.MILK_SCHEMES[sw.get("milk", 0)] not in MILK_GRAD:
        raise SystemExit(f"diag.py: milk scheme {W.MILK_SCHEMES[sw.get('milk', 0)]} has no checked gradient "
                         f"(fit.MILK_GRAD {MILK_GRAD})")
    kw = dict(N=a.N, device=a.device, T=a.T, box=box)
    workers = (a.device or "cuda").startswith("cuda") if a.workers < 0 else a.workers > 0
    t_start = time.time()
    log(f"== {mode}: N {a.N}, pours {cal}, start closure {a.init}, closure {'free (wide box)' if free_p else 'fixed'}, "
        f"inlet {'box ' + box + ' ' + json.dumps(INLET_BOX[box]) if box else 'as measured'}; slot {a.time_h:.2f} h")
    if mi or pset != PSET_DEFAULT:
        log(f"   inlet {', '.join(f'{p} {m} ({src})' for p, (m, src) in cal_inlet.items())}; parameter set {pset} "
            f"{list(keys) if free_p else '(fixed)'}" + (f"; held-out pours: inlet {HELDOUT_INLET}, e0 0 (fatal on failure)"
                                                        if mi and mode == "params" and a.heldout else ""))
    pours = {p: Pour(p, sw, kw, workers) for p in cal}
    for p in cal:
        if pours[p].info.get("box"):
            bi = pours[p].info["box"]
            _atomic_json(bi, os.path.join(out, f"box_{p}.json"), indent=1)
            log(f"   {p} box {box}: {bi['knots_in_windows']} knots in the measured windows (median U x"
                f"{bi['median_windows']['U_x'] or float('nan'):.3f}, du {bi['median_windows']['dux_ms'] or float('nan'):.3f}"
                f" / {bi['median_windows']['duy_ms'] or float('nan'):.3f} m/s), {bi['knots_law_only']} law-only "
                f"(U x{bi['median_law_only']['U_x'] or float('nan'):.3f}, du {bi['median_law_only']['dux_ms'] or float('nan'):.3f} m/s)")
    base = full_P_host(pours[cal[0]].s1, num, sw)
    up = p_to_unit(base, ps)
    ui = {p: np.full((6, pours[p].K), 0.5) for p in cal} if box else {}
    nP = len(keys)
    # Adam state over [up (if free)] + [ui per pour (if any)]
    def pack(up_, ui_):
        parts = ([up_] if free_p else []) + [ui_[p].ravel() for p in cal if box]
        return np.concatenate(parts) if parts else np.zeros(0)

    def unpack(v):
        i, up_, ui_ = 0, up.copy(), {}
        if free_p:
            up_ = v[:nP]; i = nP
        for p in cal:
            if box:
                k = 6 * pours[p].K
                ui_[p] = v[i:i + k].reshape(6, pours[p].K); i += k
        return up_, ui_

    lr = np.concatenate(([np.full(nP, a.lr)] if free_p else []) + [np.full(6 * pours[p].K, a.lr_in) for p in cal if box]) \
        if (free_p or box) else np.zeros(0)
    x = pack(up, ui)
    m1, m2 = np.zeros_like(x), np.zeros_like(x)
    it, best, recs = 0, None, []
    stf = os.path.join(out, "state.npz")
    if os.path.exists(stf):                                      # resume
        Z = np.load(stf, allow_pickle=False)
        if Z["x"].shape == x.shape:
            x, m1, m2, it = Z["x"], Z["m1"], Z["m2"], int(Z["it"])
            if "lr" in Z.files and Z["lr"].shape == lr.shape:
                lr = Z["lr"]
            recs = [json.loads(l) for l in open(os.path.join(out, "records.jsonl"))] if os.path.exists(
                os.path.join(out, "records.jsonl")) else []
            recs = [r for r in recs if r["it"] < it]
            if os.path.exists(os.path.join(out, "best.json")):
                best = json.load(open(os.path.join(out, "best.json")))
            log(f"resumed at iteration {it}")
    fn = os.path.join(out, "records.jsonl")
    with open(fn, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    b1, b2, eps = 0.9, 0.999, 1e-8
    x_prev, nfail, t_it = None, 0, None
    while it < a.iters and (free_p or box):
        if t_it is not None and time.time() - t_start + 1.15 * t_it > a.time_h * 3600:
            log(f"time slot reached after {it} iterations")
            break
        tau = a.tau0 * (a.tau1 / a.tau0) ** (it / max(a.iters - 1, 1))
        up_, ui_ = unpack(x)
        P = p_from_unit(up_, base, ps) if free_p else dict(base)
        t0 = time.time()
        try:
            for p in cal:
                pours[p].send("step", P, tau, ui_.get(p), a.prior)
            res, errs = {}, []
            for p in cal:
                try:
                    res[p] = pours[p].recv()
                except RuntimeError as e:
                    errs.append(str(e))
            if errs:
                raise RuntimeError("; ".join(errs))
        except RuntimeError as e:
            nfail += 1
            if x_prev is None or nfail > 6:
                for po in pours.values():
                    po.close()
                raise SystemExit(f"{mode} it {it}: forward failed ({e}); "
                                 + ("the start fails, nothing to back off to" if x_prev is None else "6 halvings did not help"))
            log(f"it {it}: forward failed ({str(e)[:300]}); halving the step")
            lr *= 0.5
            x = x_prev.copy()
            continue
        t_it = time.time() - t0
        L = float(np.mean([r["L"] for r in res.values()]))
        vI = float(np.mean([r["viou"] for r in res.values()]))
        gparts = []
        if free_p:
            gth = np.mean([[r["g"][k] for k in keys] for r in res.values()], axis=0)
            gparts.append(gth * p_dunit(P, ps))
        for p in cal:
            if box:
                gparts.append(res[p]["gu"].ravel() / len(cal))
        gx = np.concatenate(gparts)
        if not (np.isfinite(L) and np.all(np.isfinite(gx))):         # treat like a failed forward: keep x, no Adam update
            nfail += 1
            if x_prev is None or nfail > 6:
                for po in pours.values():
                    po.close()
                raise SystemExit(f"{mode} it {it}: non-finite loss or gradient"
                                 + ("" if x_prev is not None else " at the start"))
            log(f"it {it}: non-finite loss or gradient; halving the step")
            lr *= 0.5
            x = x_prev.copy()
            continue
        nfail = 0
        rec = dict(it=it, loss=L, viou=vI, tau=tau, wall_s=t_it, params={k: float(P[k]) for k in keys},
                   per={p: dict(loss=r["L"], viou=r["viou"], scan=r["scan"], gvj=r["gvj"], pen=r.get("pen", 0.0),
                                vol=r.get("vol", 1.0), fwd_s=r["timing"]["forward_s"], bwd_s=r["timing"]["backward_s"])
                        for p, r in res.items()})
        recs.append(rec)
        with open(fn, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"{mode} it {it:3d}  loss {L:.4f}  vIoU {vI:.3f}  " + "  ".join(
            f"{p} v{r['viou']:.3f}" + (f" scan {r['scan']:.2f} vol x{r['vol']:.2f}" if box else "") for p, r in res.items())
            + f"  |g| {np.linalg.norm(gx):.2e}  {t_it:.0f} s")
        if best is None or vI > best["viou"]:
            best = dict(it=it, viou=vI, loss=L, params=dict(P), u_in={p: ui_[p].tolist() for p in ui_})
            _atomic_json(best, os.path.join(out, "best.json"))
            # with the start closure's switches (e0, milk, sink, ...), so load_closure reproduces the same model
            _atomic_json(best_closure(num, P, mode, it, vI, box), os.path.join(out, "best_closure.json"), indent=1)
        m1 = b1 * m1 + (1 - b1) * gx
        m2 = b2 * m2 + (1 - b2) * gx ** 2
        mh, vh = m1 / (1 - b1 ** (it + 1)), m2 / (1 - b2 ** (it + 1))
        x_prev = x.copy()
        x = np.clip(x - lr * mh / (np.sqrt(vh) + eps), 0.0, 1.0)
        it += 1
        with open(stf + ".tmp", "wb") as f:
            np.savez(f, x=x, m1=m1, m2=m2, it=it, lr=lr)
        os.replace(stf + ".tmp", stf)
        progress_png(out, recs)
    # ---------------------------------------------------------------- render: start vs best
    if best is None:                                             # nothing free: the start itself
        best = dict(it=-1, viou=float("nan"), params=dict(base), u_in={})
    P1 = dict(base, **{k: float(v) for k, v in best["params"].items() if k in W.PARAMS})
    label = {"params": "只放开参数", "inlet": "只修正入口（误差内）", "both": "参数+入口（误差内）",
             "both_wide": "参数+入口（放宽）"}[mode]
    log(f"render: best iteration {best['it']} (video IoU {best['viou']:.4f})")
    ev = {}
    for p in cal:
        u1 = np.array(best["u_in"][p]) if best["u_in"].get(p) is not None else None
        pours[p].send("render", out, base, P1, u1, label, a.every, a.movie, a.T)
    for p in cal:
        try:
            ev[p] = pours[p].recv()
        except RuntimeError as e:
            ev[p] = dict(error=str(e))
            log(f"{p}: render failed ({str(e)[:400]})")
            continue
        for er in ev[p].get("errors", []):
            log(f"{p}: figure failed: {er[:300]}")
        b, s = ev[p]["best"], ev[p]["start"]
        log(f"{p}: J {s['J']:.3f} -> {b['J']:.3f}   separation {s['sep']:.2f} -> {b['sep']:.2f} "
            f"(kept {b['kept']}/{b['pairs']}, merged {b['merged']}, missing {b['missing']})"
            + (f"   inlet: shift p50 {ev[p]['inlet']['shift_mm_p50']:.1f} mm, Q x{ev[p]['inlet']['Q_ratio_p10']:.2f}-"
               f"{ev[p]['inlet']['Q_ratio_p90']:.2f}" if ev[p].get("inlet") else ""))
    for po in pours.values():
        po.close()
    ecf = os.path.join(out, "eval_cal.json")
    if mi and mode == "params" and a.heldout:
        # the calibration answer is on disk before the held-out pours run: a held-out failure under an MI inlet is fatal
        # (no done.json), but it must not hide the tulip / leaf evaluation (REV_code finding 2); summary tools read it
        ec = dict(mode=mode, init=os.path.abspath(a.init), N=a.N, best_it=best["it"], best_viou=best["viou"],
                  params={k: float(P1[k]) for k in keys}, eval=dict(ev), iterations=it,
                  hours=(time.time() - t_start) / 3600, pset=pset, heldout="running")
        _atomic_json(ec, ecf, indent=1)
    if mode == "params" and a.heldout:                          # frozen-parameter test on the held-out pours
        kwh, swh = dict(kw, box=None), sw
        if mi:                          # SPEC27 6.1: MI-law with e0 0 (precedence: the per-pour override wins)
            kwh["inlet_override"] = HELDOUT_INLET
            swh = dict(sw, e0=False)
        for p in HELDOUT:
            po = None
            try:
                po = Pour(p, swh, kwh, workers)
                P0h = full_P_host(po.s1, num, sw)
                P1h = dict(P0h, **{k: float(P1[k]) for k in keys})
                po.send("render", out, P0h, P1h, None, label, a.every, a.movie, a.T)
                ev[p] = po.recv()
                po.close()
                log(f"held-out {p}: J {ev[p]['start']['J']:.3f} -> {ev[p]['best']['J']:.3f}  separation "
                    f"{ev[p]['start']['sep']:.2f} -> {ev[p]['best']['sep']:.2f}"
                    + (f"  (inlet {po.info['inlet']['mode']}, {po.info['inlet']['source']})" if mi else ""))
            except Exception as e:                               # noqa: BLE001 -- report and go on (record inlet)
                log(f"held-out {p}: failed ({e})")
                if po is None and not workers:
                    RP.INLET_OVERRIDE.pop(p, None)               # (make_pour set it before the failure)
                if mi:                  # SPEC27 3.5 item 7: under a measured inlet a held-out failure is fatal
                    if po is not None:
                        po.close()
                    try:
                        ec = json.load(open(ecf))
                        ec.update(heldout="fatal failure", heldout_failed=dict(pour=p, error=str(e)[:2000]))
                        _atomic_json(ec, ecf, indent=1)
                    except Exception as e2:                      # noqa: BLE001
                        log(f"eval_cal.json not updated ({e2})")
                    raise RuntimeError(f"held-out {p} failed under the measured inlet (HELDOUT_INLET "
                                       f"{HELDOUT_INLET!r}); fatal, done.json not written") from e
    if mi and mode == "params" and a.heldout and os.path.exists(ecf):     # the held-out pours ran
        ec = json.load(open(ecf))
        ec["heldout"] = "ok (done.json)"
        _atomic_json(ec, ecf, indent=1)
    done = dict(mode=mode, init=os.path.abspath(a.init), N=a.N, best_it=best["it"], best_viou=best["viou"],
                params={k: float(P1[k]) for k in keys}, eval=ev, iterations=it, hours=(time.time() - t_start) / 3600)
    if pset != PSET_DEFAULT:                                     # non-default runs only (done.json unchanged by default)
        done["pset"] = pset
    if mi:
        done["inlet"] = dict(cal={p: dict(mode=m, source=src) for p, (m, src) in cal_inlet.items()}, box=box,
                             heldout=(dict(mode=HELDOUT_INLET, e0=0) if mode == "params" and a.heldout else None),
                             status="provisional (marginal pass), SPEC27 rev 2 section 2.4")
    _atomic_json(done, os.path.join(out, "done.json"), indent=1)
    log(f"== {mode} done in {(time.time() - t_start) / 3600:.2f} h -> {out}")


def best_closure(num, P, mode, it, vI, box):
    """best_closure.json: the start closure's numbers and switches with the fitted numbers.  With an inlet correction
    (inlet / both modes) it carries no 'inlet' field (SPEC27 6.3, PREREG27_2 D11: the oracle's corrections, kept in
    best.json, are diagnostic only and never become an inlet or a closure input)."""
    bc = dict(num, **{k: float(P[k]) for k in W.PARAMS if k in P},
              _note=f"diag {mode} best (it {it}, video IoU {vI:.4f}); inlet corrections in best.json")
    if box and "inlet" in bc:
        del bc["inlet"]
        bc["_note"] += (f"; fitted under the start closure's inlet with the oracle box {box!r}: the 'inlet' key is removed "
                        "(SPEC27 6.3 / D11, the oracle is diagnostic only)")
    if box:
        assert "inlet" not in bc and not any(k.startswith("oracle") for k in bc), "best_closure: inlet guard"
    return bc


def full_P_host(s1, num, sw):
    P = dict(s1)
    if not sw["circ"]:
        P.update(kappa_M=0.0, eps_f=0.0, kappa_w=0.0)
    P.update({k: v for k, v in num.items() if k in W.PARAMS})
    return P


# ----------------------------------------------------------------------------------------------- summary + decision
OPEN_J, OPEN_SEP = 0.80, 0.50      # pre-registered: a run 'opens the tulip' if tulip J >= 0.80 and separation >= 0.50


def opens(ev):
    b = ev.get("tulip", {}).get("best") if ev else None
    return bool(b and np.isfinite(b.get("J", np.nan)) and b["J"] >= OPEN_J and b.get("sep", 0.0) >= OPEN_SEP)


def decide(o):
    """o: {mode: opens?} over the finished modes.  The pre-registered rule of README_diag256.md; a mode that did not
    finish is unknown, never 'does not open'."""
    done, op = set(o), {m for m in o if o[m]}
    notes = []
    if "params" in op and "both_wide" in done and "both_wide" not in op:
        notes.append("注意：只放开参数能打开，可搜索范围更大的 D 反而没打开——D 的优化没收敛，D 的结论不可信。")
    if "inlet" in op and "both" in done and "both" not in op:
        notes.append("注意：只修正入口能打开，范围更大的 C 反而没打开——C 的优化没收敛。")
    if not op and {"both_wide", "both"} <= done:
        v = "1 物理模型的问题：参数随便放、入口放宽到测量误差的 2.5–3 倍，郁金香仍打不开。表层模型本身缺机制。"
    elif "params" in op:
        v = "2 物理参数的问题：入口按测量值，只放开参数就能打开。倒奶模型在误差内够用。"
    elif "inlet" in op and "params" in done:
        v = "3 倒奶模型的问题：参数固定、只在测量误差内修正入口就能打开；只放开参数打不开。"
    elif "both" in op and {"params", "inlet"} <= done:
        v = "2 + 3：参数和倒奶模型都有问题——两者一起放开（入口在误差内）才打开，单放一个都不够。"
    elif op == {"both_wide"} and {"both", "params", "inlet"} <= done:
        v = ("1（为主）+ 3：只有把入口改到超出测量误差才打开。要么倒奶误差比估计的大得多，要么表层模型缺机制、"
             "靠扭曲入口来凑；看 D 的入口改动图判断改动像不像真实的手。")
    else:
        v = "结果不全，还不能下结论（未完成的试验见上表），或结果互相矛盾；见各试验的 run.log。"
    return v, notes


def _row(name, R):
    if R is None:
        return f"| {name} | 未完成 | | | | | |"
    ev = R["eval"]

    def js(p):
        e = ev.get(p, {})
        if "best" not in e:
            return "失败", "失败"
        return (f"{e['start']['J']:.3f} → {e['best']['J']:.3f}", f"{e['start']['sep']:.2f} → {e['best']['sep']:.2f}")
    (tj, ts), (lj, ls) = js("tulip"), js("leaf")
    inl = ev.get("tulip", {}).get("inlet")
    ic = (f"{inl['shift_mm_p50']:.1f} mm / ×{inl['Q_ratio_p10']:.2f}–{inl['Q_ratio_p90']:.2f} / 总量 ×{inl['Q_total_ratio']:.2f}"
          if inl else "—")
    return f"| {name} | {tj} | {ts} | {lj} | {ls} | {ic} | {'是' if opens(ev) else '否'} |"


def summary(root):
    R = {}
    for m in MODES:
        fn = os.path.join(root, m, "done.json")
        if os.path.exists(fn):
            R[m] = json.load(open(fn))
    lines = ["# 三方诊断结果（N = %s）" % (next(iter(R.values()))["N"] if R else "?"), "",
             f"判据（事先定）：郁金香 J ≥ {OPEN_J} 且 末段花瓣分离率 ≥ {OPEN_SEP} 记为“打开”。",
             "J 与标定同一把尺（0.25 IoU + 0.75 结构，按视频自身 0.25 s 一致性归一，1 = 像视频自己）；"
             "分离率 = 视频里成对分开的白色部件在模拟里也分开的比例（validate.part_separation，末 30 % 的观测帧）。", "",
             "| 试验 | tulip J 起点→结果 | tulip 分离率 | leaf J | leaf 分离率 | 入口改动（tulip：落点中位 / Q 倍率 p10–p90 / 总量） | 打开？ |",
             "|---|---|---|---|---|---|---|"]
    name = {"params": "A 只放开参数（宽）", "inlet": "B 只修正入口（误差内）", "both": "C 参数+入口（误差内）",
            "both_wide": "D 参数+入口（放宽）"}
    for m in MODES:
        lines.append(_row(name[m], R.get(m)))
    if "params" in R:
        ho = [p for p in HELDOUT if "best" in R["params"]["eval"].get(p, {})]
        if ho:
            lines += ["", "留出三杯（只放开参数的结果冻结后直接跑，不参与任何选择）：" + "，".join(
                f"{p} J {R['params']['eval'][p]['start']['J']:.3f} → {R['params']['eval'][p]['best']['J']:.3f}" for p in ho)]
    v, notes = decide({m: opens(R[m]["eval"]) for m in R})
    lines += ["", "## 判定", v] + notes
    txt = "\n".join(lines) + "\n"
    open(os.path.join(root, "summary.md"), "w").write(txt)
    print(txt)


# ----------------------------------------------------------------------------------------------- gradient check
def gradcheck(a):
    """The gradient check (_gradcheck); SystemExit(1) on a failure, else its result dict."""
    res = _gradcheck(a)
    if not res["ok"]:
        raise SystemExit(1)
    return res


def _gradcheck(a):
    """Adjoint vs central finite differences of the loss w.r.t. inlet knot variables (two random active knots per
    variable) and closure numbers (--keys), and the bitwise identity of a zero correction.  --box: the inlet box
    ('meas' / 'meas_wide' need an MI inlet: --inlet law / replay, the closure key or LATTE_INLET); --eps: the knot step in
    the unit box and the closure numbers' relative step.  With the closure switch vj the dx / dy knots carry the V_j
    adjoint (InletCorr.vj_T); its share is printed.  Returns the result dict (ok, worst, rows, ...)."""
    from .render512 import load_closure
    from ..rig.real_program import inlet_mode
    num, sw = load_closure(a.init)
    if getattr(a, "inlet", None):
        sw = dict(sw, inlet=inlet_mode(a.inlet))            # closure level of real_program.resolve_inlet
    box = getattr(a, "box", "err")
    D = make_pour(a.pour, sw, dict(N=a.N, device=a.device, T=a.T, box=box))
    P = full_P_host(D.s1, num, sw)
    print(f"gradcheck {a.pour}: init {os.path.basename(a.init)}, inlet {D.inlet} ({D.pour.inlet_source}), box {box}, "
          f"N {a.N}, T {a.T}, eps {a.eps:g}, milk {W.MILK_SCHEMES[sw.get('milk', 0)]}, vj {P.get('vj', 0.0):g}, "
          f"kappa_tc {P.get('kappa_tc', 0.0):g}, K {D.inl.K}", flush=True)
    S0 = {k: D.Sw[k].numpy().copy() for k in W.SCHED}
    Sz = D.inl.apply(D.inl.zero())
    same = all(np.array_equal(Sz[k], S0[k]) for k in W.SCHED)
    print(f"zero correction reproduces the record bitwise: {same}")
    rng = np.random.default_rng(3)
    u = np.clip(0.5 + 0.15 * rng.standard_normal((6, D.inl.K)), 0.05, 0.95)
    r = step(D, P, 0.05, u, 0.0)
    pen0, gpen, _ = D.inl.penalty(u, 0.0)
    gu = r["gu"] - gpen                                      # the solver chain alone (the penalty is host-side, exact)
    gnv = D.inl.chain(D.sched_grad, vj=False)                # without the V_j adjoint (its share, reported)
    act_k = np.unique(np.r_[D.inl.j[D.inl.act], D.inl.j[D.inl.act] + 1])
    worst, ok = 0.0, same
    rows = []
    nr = int(getattr(a, "rand_knots", 2))
    pick = [(v, k) for v in range(6) for k in (rng.choice(act_k, size=min(nr, len(act_k)), replace=False) if nr > 0
                                               else [])]
    nvjk = int(getattr(a, "vj_knots", 0))
    if nvjk > 0:            # the dx / dy knots where the V_j adjoint carries the largest share (discriminating check)
        vjp = np.abs(gu[:2] - gnv[:2])
        big = np.abs(gu[:2]) > 1e-3 * np.abs(gu[:2]).max()
        cand = sorted(((v, k) for v in range(2) for k in range(D.inl.K) if big[v, k] and vjp[v, k] > 0),
                      key=lambda vk: -vjp[vk] / abs(gu[vk]))
        pick += [vk for vk in cand[:nvjk] if vk not in pick]
    for v, k in pick:
        h = a.eps
        up, dn = u.copy(), u.copy()
        up[v, k] += h; dn[v, k] -= h
        Lp = step_loss(D, P, up)
        Lm = step_loss(D, P, dn)
        fd = (Lp - Lm) / (2 * h)
        ad = gu[v, k]
        rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-12)
        if not (np.isfinite(fd) and np.isfinite(ad)):
            worst = float("inf")
        else:
            worst = max(worst, rel if max(abs(fd), abs(ad)) > 1e-9 else 0.0)
        vjs = ad - gnv[v, k] if v < 2 else 0.0             # (the V_j adjoint enters dx, dy only)
        rel_nv = abs(fd - gnv[v, k]) / max(abs(fd), abs(gnv[v, k]), 1e-12)
        rows.append(dict(var=IV[v], knot=int(k), t_knot=float(D.inl.knots[k]), adjoint=float(ad), fd=float(fd),
                         rel=float(rel), vj_part=float(vjs), vj_share=float(abs(vjs) / max(abs(ad), 1e-300)),
                         rel_without_vj=float(rel_nv)))
        print(f"  {IV[v]:4s} knot {k:3d}: adjoint {ad:+.6e}  FD {fd:+.6e}  rel {rel:.2e}"
              + (f"   (V_j part {vjs:+.3e}; without it rel {rel_nv:.2e})" if vjs != 0.0 else ""))
    for key in [k for k in getattr(a, "keys", "kappa_Q,beta").split(",") if k]:
        h = a.eps * abs(float(P[key])) if float(P[key]) != 0.0 else a.eps
        Pp, Pm = dict(P), dict(P)
        Pp[key] += h; Pm[key] -= h
        fd = (step_loss(D, Pp, u) - step_loss(D, Pm, u)) / (2 * h)
        ad = r["g"][key]
        rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-12)
        worst = max(worst, rel) if (np.isfinite(fd) and np.isfinite(ad)) else float("inf")
        rows.append(dict(var=key, adjoint=float(ad), fd=float(fd), rel=float(rel)))
        print(f"  {key:8s}: adjoint {ad:+.6e}  FD {fd:+.6e}  rel {rel:.2e}")
    ok = ok and worst < a.tol
    nvj = float(np.abs(gu[:2] - gnv[:2]).sum())
    print(f"gradcheck {'PASS' if ok else 'FAIL'} (worst relative error {worst:.2e}, tolerance {a.tol:g}; scan {r['scan']:.2f}"
          f"; sum |dL/dV_j| {r['gvj']:.3e}, V_j share of |dL/d(dx, dy)| {nvj / max(float(np.abs(gu[:2]).sum()), 1e-300):.3f})")
    res = dict(ok=bool(ok), same=bool(same), worst=float(worst), tol=float(a.tol), rows=rows, inlet=D.inlet, box=box,
               scan=float(r["scan"]), gvj=float(r["gvj"]), vj_share=nvj / max(float(np.abs(gu[:2]).sum()), 1e-300),
               K=int(D.inl.K), N=a.N, T=a.T, eps=a.eps, init=os.path.abspath(a.init), pour=a.pour,
               milk=W.MILK_SCHEMES[sw.get("milk", 0)], vj=float(P.get("vj", 0.0)), box_info=D.inl.box_info())
    return res


def step_loss(D, P, u):
    S = D.inl.apply(u)
    assign(D, S)
    D.tau = 0.05
    L, _, _ = D.loss(P, nsub=D.nsub)                         # the sub-steps of the adjoint's forward
    return float(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", required=True, choices=MODES)
    r.add_argument("--init", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--N", type=int, default=256)
    r.add_argument("--cal", default=",".join(CAL))
    r.add_argument("--iters", type=int, default=80)
    r.add_argument("--time-h", dest="time_h", type=float, default=1.8)
    r.add_argument("--lr", type=float, default=0.03, help="Adam step of the closure numbers (unit box)")
    r.add_argument("--lr-in", dest="lr_in", type=float, default=0.05, help="Adam step of the inlet knots (unit box)")
    r.add_argument("--tau0", type=float, default=0.10)
    r.add_argument("--tau1", type=float, default=0.03)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--workers", type=int, default=-1, help="-1: one process per pour on CUDA, in-process on the CPU")
    r.add_argument("--T", type=float, default=None, help="simulate only the first T seconds (tests)")
    r.add_argument("--every", type=int, default=4)
    r.add_argument("--no-movie", dest="movie", action="store_false")
    r.add_argument("--no-heldout", dest="heldout", action="store_false")
    r.add_argument("--prior", type=float, default=PRIOR, help="weak prior on the inlet knots (InletCorr.penalty)")
    r.add_argument("--pset", default=PSET_DEFAULT, choices=tuple(PSETS),
                   help="closure numbers freed in the params / both modes (PSETS; default v05n = PARAM_WIDE)")
    r.add_argument("--box-family", dest="box_family", default="auto", choices=("auto", "err", "meas"),
                   help="inlet boxes of the inlet / both / both_wide modes: err/wide or meas/meas_wide (auto: meas under "
                        "an MI inlet, else err)")
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    g = sub.add_parser("gradcheck")
    g.add_argument("--init", default=os.path.join(os.path.dirname(__file__), "..", "data", "closures", "v05n_AB_cmaes_N48.json"))
    g.add_argument("--pour", default="tulip")
    g.add_argument("--N", type=int, default=48)
    g.add_argument("--T", type=float, default=1.0)
    g.add_argument("--device", default="cpu")
    g.add_argument("--eps", type=float, default=1e-5, help="knot step (unit box) and relative step of --keys")
    g.add_argument("--tol", type=float, default=2e-3)
    g.add_argument("--box", default="err", choices=tuple(INLET_BOX), help="inlet correction box")
    g.add_argument("--keys", default="kappa_Q,beta", help="closure numbers checked (comma separated)")
    g.add_argument("--inlet", default=None, help="inlet mode at the closure level (record / law / replay or 0 / 1 / 2)")
    g.add_argument("--rand-knots", dest="rand_knots", type=int, default=2, help="random active knots checked per variable")
    g.add_argument("--vj-knots", dest="vj_knots", type=int, default=0,
                   help="also check the dx / dy knots with the largest V_j share (vj closures; FD with and without V_j)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    elif a.cmd == "summary":
        summary(a.out)
    else:
        gradcheck(a)


if __name__ == "__main__":
    main()
