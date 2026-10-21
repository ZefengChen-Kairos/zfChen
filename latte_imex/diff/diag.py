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
# inlet correction boxes: 'err' = the measured inlet's own uncertainty (pose 5-10 mm at the spout tip and ~1 deg of
# tilt = 30-50 % of the flow per degree; fall height +-5-10 mm of 15-55 mm = +-15-25 % of U_perp; arrival direction
# from the same pose); 'wide' = 2.5-3x that
INLET_BOX = {"err": dict(dx_mm=8.0, Q_x=2.0, U_x=1.4, du_ms=0.10, knot_s=0.2),
             "wide": dict(dx_mm=20.0, Q_x=5.0, U_x=2.5, du_ms=0.30, knot_s=0.3)}
IV = ("dx", "dy", "aQ", "aU", "dux", "duy")
MODE_SET = {"params": (True, None), "inlet": (False, "err"), "both": (True, "err"), "both_wide": (True, "wide")}
KEY_T = {"tulip": (1.92, 2.78, 4.02, 4.87, 13.21), "leaf": (2.21, 3.03, 4.41, 8.79)}
SEP_FRAC = 0.3           # petal separation over the last 30 % of the observed frames


# ----------------------------------------------------------------------------------------------- parameter box
def p_to_unit(P):
    u = []
    for k in PKEYS:
        lo, hi, sc = PARAM_WIDE[k]
        v = min(max(float(P[k]), lo), hi)
        u.append(math.log(v / lo) / math.log(hi / lo) if sc == "log" else (v - lo) / (hi - lo))
    return np.array(u)


def p_from_unit(u, base):
    P = dict(base)
    for k, x in zip(PKEYS, np.clip(u, 0.0, 1.0)):
        lo, hi, sc = PARAM_WIDE[k]
        P[k] = lo * (hi / lo) ** x if sc == "log" else lo + (hi - lo) * x
    return P


def p_dunit(P):
    return np.array([float(P[k]) * math.log(PARAM_WIDE[k][1] / PARAM_WIDE[k][0]) if PARAM_WIDE[k][2] == "log"
                     else PARAM_WIDE[k][1] - PARAM_WIDE[k][0] for k in PKEYS])


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
        self.lim = np.array([dx, dx, math.log(b["Q_x"]), math.log(b["U_x"]), du, du])
        self.h, self.safety = 1.0 / D.N, 0.5                 # solver.Numerics.scan_safety
        self.cur = None

    def zero(self):
        return np.full((6, self.K), 0.5)

    def theta(self, u):
        return (2.0 * np.asarray(u, float) - 1.0) * self.lim[:, None]

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

    def chain(self, G):
        """dL/du (6, K) from the per-frame schedule gradient G (dict of numpy) at the last apply()."""
        a, c = self.act, self.cur
        n = len(self.t)
        g = np.zeros((6, n))
        Gx = {k: np.asarray(G[k], float)[a] for k in W.SCHED}
        R1, R2, Dd = Gx["r1"] * c["r1"], Gx["r2"] * c["r2"], Gx["d"] * c["d"]
        g[0][a], g[1][a] = Gx["x"], Gx["y"]
        g[2][a] = Gx["Q"] * c["Q"] + 0.5 * (R1 + R2 + Dd)
        s = c["U"] ** 2 / c["s2"]
        g[3][a] = Gx["U"] * c["U"] + (0.5 * s - 1.0) * R1 - 0.5 * s * (R2 + Dd)
        okv = c["v2"] > 1e-24
        iv2 = np.where(okv, 1.0 / np.where(okv, c["v2"], 1.0), 0.0)
        com = 0.5 * (R1 - R2 - Dd) / c["s2"]
        g[4][a] = Gx["ux"] + c["ux"] * com - c["uy"] * iv2 * Gx["phi"]
        g[5][a] = Gx["uy"] + c["uy"] * com + c["ux"] * iv2 * Gx["phi"]
        return self.frames_T(g) * (2.0 * self.lim[:, None])

    def scan_ratio(self, S):
        """Largest landing-point step per frame over the solver's own bound scan_safety min(h, r1, r2) (<= ~1 in the
        measured record)."""
        a = self.active
        both = a[1:] & a[:-1]
        if not both.any():
            return 0.0
        step = np.hypot(np.diff(S["x"]), np.diff(S["y"]))[both]
        lim = self.safety * np.minimum(self.h, np.minimum(np.maximum(S["r1"], 1.5 * self.h), np.maximum(S["r2"], 1.5 * self.h)))[1:][both]
        return float(np.max(step / lim))

    def series(self, S):
        """Measured vs corrected inlet on the active frames, physical units (for the figure)."""
        a, L = self.act, self.L_m
        S0 = self.S0
        return dict(t=self.t[a], Q0=S0["Q"][a] * L ** 3 * 1e6, Q=S["Q"][a] * L ** 3 * 1e6,
                    dx=(S["x"][a] - S0["x"][a]) * L * 1e3, dy=(S["y"][a] - S0["y"][a]) * L * 1e3,
                    U0=S0["U"][a] * L, U=S["U"][a] * L,
                    uh0=np.hypot(S0["ux"][a], S0["uy"][a]) * L, uh=np.hypot(S["ux"][a], S["uy"][a]) * L,
                    du=np.hypot(S["ux"][a] - S0["ux"][a], S["uy"][a] - S0["uy"][a]) * L)


def assign(D, S):
    for k in W.SCHED:
        D.Sw[k].assign(np.asarray(S[k], float))
        D.S[k][:] = S[k]


# ----------------------------------------------------------------------------------------------- one pour (worker side)
def make_pour(p, sw, kw):
    from .run import DiffPour
    D = DiffPour(p, N=kw["N"], device=kw["device"], T_max=kw.get("T"), w_gap=2.0, tau=0.10, loss="video",
                 scales=(0.3, 0.15, 0.0), e0=sw["e0"], exit_cal=sw["exit_cal"], upwell=sw["upwell"], circ=sw["circ"],
                 circ_shape=sw["circ_shape"], milk=sw.get("milk", 0), sink=sw.get("sink", 0),
                 visc_tr=sw.get("visc_tr", 0), log=lambda *x: print(*x, flush=True))
    D.inl = InletCorr(D, kw["box"]) if kw.get("box") else None
    return D


def full_P(D, num, sw):
    P = dict(D.s1)
    if not sw["circ"]:
        P.update(kappa_M=0.0, eps_f=0.0, kappa_w=0.0)
    P.update({k: v for k, v in num.items() if k in W.PARAMS})
    return P


def step(D, P, tau, u):
    scan = 0.0
    if D.inl is not None and u is not None:
        S = D.inl.apply(u)
        assign(D, S)
        scan = D.inl.scan_ratio(S)
    D.tau = tau
    L, g, _ = D.loss_grad(P)
    gu = D.inl.chain(D.sched_grad) if (D.inl is not None and u is not None) else None
    gvj = float(np.abs(D.sched_grad["vjx"]).sum() + np.abs(D.sched_grad["vjy"]).sum())
    viou = float(np.mean([v for _, v in D.vmetrics])) if D.vmetrics else float("nan")
    return dict(L=float(L), g=g, gu=gu, viou=viou, scan=scan, gvj=gvj, timing=D.timing)


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
    fig.suptitle(f"{p}: inlet correction (box '{box}')")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{p}_inlet.png"), dpi=110)
    plt.close(fig)


def inlet_stats(ser):
    if ser is None or len(ser["t"]) == 0:
        return None
    sh = np.hypot(ser["dx"], ser["dy"])
    qr = ser["Q"] / np.maximum(ser["Q0"], 1e-12)
    ur = ser["U"] / np.maximum(ser["U0"], 1e-12)
    return dict(shift_mm_p50=float(np.median(sh)), shift_mm_p90=float(np.percentile(sh, 90)), shift_mm_max=float(sh.max()),
                Q_ratio_p10=float(np.percentile(qr, 10)), Q_ratio_p50=float(np.median(qr)), Q_ratio_p90=float(np.percentile(qr, 90)),
                Q_total_ratio=float(ser["Q"].sum() / max(ser["Q0"].sum(), 1e-12)),
                U_ratio_p10=float(np.percentile(ur, 10)), U_ratio_p90=float(np.percentile(ur, 90)),
                du_ms_p50=float(np.median(ser["du"])), du_ms_p90=float(np.percentile(ser["du"], 90)))


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
        inlet_figure(p, out, ser, D.inl.box)
    _, C1, ML1 = RR.run_pour(D, P1, sw, every)
    r1 = dict(RR.score48(p, sw, D, ML1, T, D48=D48), **separation(D, ML1))
    msk = D.geo.mask_np
    lab0 = "起点（测量入口）"
    stills(p, out, [(t_end, C0, msk), (t_end, C1, msk)], [lab0, label])
    if movie:
        RR.movies(p, out, (t_end, C1, msk, D.N), (t_end, C0, msk, D.N), label, lab0, every)
    tobs = np.array([o["t"] for o in D.obs])
    c128 = np.array([np.clip(RR.to_H(C1[int(o["k"])], msk, 128), 0, 1) for o in D.obs])
    np.savez_compressed(os.path.join(out, f"{p}_c128.npz"), t=tobs, c=(c128 * 255 + 0.5).astype(np.uint8))
    if D.inl is not None:
        assign(D, D.inl.S0)
    return dict(start=r0, best=r1, inlet=inlet_stats(ser))


def _worker(conn, p, sw, kw):
    import traceback
    try:
        D = make_pour(p, sw, kw)
        conn.send(("ok", dict(s1=D.s1, K=(D.inl.K if D.inl else 0), n=int(D.S["n"]))))
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
            info = dict(s1=self.D.s1, K=(self.D.inl.K if self.D.inl else 0), n=int(self.D.S["n"]))
        self.s1, self.K, self.n = info["s1"], info["K"], info["n"]
        self.sw = sw

    def send(self, *msg):
        if self.process:
            self.conn.send(msg)
        else:
            try:
                fn = step if msg[0] == "step" else (lambda D, *a: render(D, self.sw, *a))
                self._res = ("ok", fn(self.D, *msg[1:]))
            except RuntimeError as e:
                self._res = ("err", str(e))

    def recv(self):
        st, r = self.conn.recv() if self.process else self._res
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
    cal = tuple(a.cal.split(","))
    kw = dict(N=a.N, device=a.device, T=a.T, box=box)
    workers = (a.device or "cuda").startswith("cuda") if a.workers < 0 else a.workers > 0
    t_start = time.time()
    log(f"== {mode}: N {a.N}, pours {cal}, start closure {a.init}, closure {'free (wide box)' if free_p else 'fixed'}, "
        f"inlet {'box ' + box + ' ' + json.dumps(INLET_BOX[box]) if box else 'as measured'}; slot {a.time_h:.2f} h")
    pours = {p: Pour(p, sw, kw, workers) for p in cal}
    base = full_P_host(pours[cal[0]].s1, num, sw)
    up = p_to_unit(base)
    ui = {p: np.full((6, pours[p].K), 0.5) for p in cal} if box else {}
    nP = len(PKEYS)
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
        P = p_from_unit(up_, base) if free_p else dict(base)
        t0 = time.time()
        try:
            for p in cal:
                pours[p].send("step", P, tau, ui_.get(p))
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
        nfail = 0
        t_it = time.time() - t0
        L = float(np.mean([r["L"] for r in res.values()]))
        vI = float(np.mean([r["viou"] for r in res.values()]))
        gparts = []
        if free_p:
            gth = np.mean([[r["g"][k] for k in PKEYS] for r in res.values()], axis=0)
            gparts.append(gth * p_dunit(P))
        for p in cal:
            if box:
                gparts.append(res[p]["gu"].ravel() / len(cal))
        gx = np.concatenate(gparts)
        rec = dict(it=it, loss=L, viou=vI, tau=tau, wall_s=t_it, params={k: float(P[k]) for k in PKEYS},
                   per={p: dict(loss=r["L"], viou=r["viou"], scan=r["scan"], gvj=r["gvj"], fwd_s=r["timing"]["forward_s"],
                                bwd_s=r["timing"]["backward_s"]) for p, r in res.items()})
        recs.append(rec)
        with open(fn, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"{mode} it {it:3d}  loss {L:.4f}  vIoU {vI:.3f}  " + "  ".join(
            f"{p} v{r['viou']:.3f}" + (f" scan {r['scan']:.2f}" if box else "") for p, r in res.items())
            + f"  |g| {np.linalg.norm(gx):.2e}  {t_it:.0f} s")
        if best is None or vI > best["viou"]:
            best = dict(it=it, viou=vI, loss=L, params=dict(P), u_in={p: ui_[p].tolist() for p in ui_})
            json.dump(best, open(os.path.join(out, "best.json"), "w"), default=float)
            json.dump(dict(P, _note=f"diag {mode} best (it {it}, video IoU {vI:.4f})"),
                      open(os.path.join(out, "best_closure.json"), "w"), indent=1, default=float)
        m1 = b1 * m1 + (1 - b1) * gx
        m2 = b2 * m2 + (1 - b2) * gx ** 2
        mh, vh = m1 / (1 - b1 ** (it + 1)), m2 / (1 - b2 ** (it + 1))
        x_prev = x.copy()
        x = np.clip(x - lr * mh / (np.sqrt(vh) + eps), 0.0, 1.0)
        it += 1
        np.savez(stf, x=x, m1=m1, m2=m2, it=it)
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
        ev[p] = pours[p].recv()
        b, s = ev[p]["best"], ev[p]["start"]
        log(f"{p}: J {s['J']:.3f} -> {b['J']:.3f}   separation {s['sep']:.2f} -> {b['sep']:.2f} "
            f"(kept {b['kept']}/{b['pairs']}, merged {b['merged']}, missing {b['missing']})"
            + (f"   inlet: shift p50 {ev[p]['inlet']['shift_mm_p50']:.1f} mm, Q x{ev[p]['inlet']['Q_ratio_p10']:.2f}-"
               f"{ev[p]['inlet']['Q_ratio_p90']:.2f}" if ev[p].get("inlet") else ""))
    for po in pours.values():
        po.close()
    if mode == "params" and a.heldout:                          # frozen-parameter test on the held-out pours
        kwh = dict(kw, box=None)
        for p in HELDOUT:
            try:
                po = Pour(p, sw, kwh, workers)
                P0h = full_P_host(po.s1, num, sw)
                P1h = dict(P0h, **{k: float(P1[k]) for k in PKEYS})
                po.send("render", out, P0h, P1h, None, label, a.every, a.movie, a.T)
                ev[p] = po.recv()
                po.close()
                log(f"held-out {p}: J {ev[p]['start']['J']:.3f} -> {ev[p]['best']['J']:.3f}  separation "
                    f"{ev[p]['start']['sep']:.2f} -> {ev[p]['best']['sep']:.2f}")
            except Exception as e:                               # noqa: BLE001 -- report and go on
                log(f"held-out {p}: failed ({e})")
    json.dump(dict(mode=mode, init=os.path.abspath(a.init), N=a.N, best_it=best["it"], best_viou=best["viou"],
                   params={k: float(P1[k]) for k in PKEYS}, eval=ev, iterations=it, hours=(time.time() - t_start) / 3600),
              open(os.path.join(out, "done.json"), "w"), indent=1, default=float)
    log(f"== {mode} done in {(time.time() - t_start) / 3600:.2f} h -> {out}")


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
    return bool(b and b["J"] >= OPEN_J and b["sep"] >= OPEN_SEP)


def summary(root):
    rows, R = [], {}
    for m in MODES:
        fn = os.path.join(root, m, "done.json")
        if os.path.exists(fn):
            R[m] = json.load(open(fn))
    lines = ["# 三方诊断结果（N = %s）" % (next(iter(R.values()))["N"] if R else "?"), "",
             f"判据（事先定）：郁金香 J ≥ {OPEN_J} 且 末段花瓣分离率 ≥ {OPEN_SEP} 记为“打开”。",
             "J 与标定同一把尺（0.25 IoU + 0.75 结构，按视频自身 0.25 s 一致性归一，1 = 像视频自己）；"
             "分离率 = 视频里成对分开的白色部件在模拟里也分开的比例（validate.part_separation，末 30 % 的观测帧）。", "",
             "| 试验 | tulip J 起点→结果 | tulip 分离率 | leaf J | leaf 分离率 | 入口改动（tulip：落点中位 / Q 倍率 p10–p90） | 打开？ |",
             "|---|---|---|---|---|---|---|"]
    name = {"params": "A 只放开参数（宽）", "inlet": "B 只修正入口（误差内）", "both": "C 参数+入口（误差内）",
            "both_wide": "D 参数+入口（放宽）"}
    for m in MODES:
        if m not in R:
            lines.append(f"| {name[m]} | 未完成 | | | | | |")
            continue
        ev = R[m]["eval"]
        t, l = ev.get("tulip", {}), ev.get("leaf", {})
        inl = t.get("inlet")
        lines.append(f"| {name[m]} | {t['start']['J']:.3f} → {t['best']['J']:.3f} | {t['start']['sep']:.2f} → "
                     f"{t['best']['sep']:.2f} | {l['start']['J']:.3f} → {l['best']['J']:.3f} | {l['start']['sep']:.2f} → "
                     f"{l['best']['sep']:.2f} | " + (f"{inl['shift_mm_p50']:.1f} mm / ×{inl['Q_ratio_p10']:.2f}–"
                                                    f"{inl['Q_ratio_p90']:.2f}" if inl else "—")
                     + f" | {'是' if opens(ev) else '否'} |")
    if "params" in R:
        ho = [p for p in HELDOUT if p in R["params"]["eval"]]
        if ho:
            lines += ["", "留出三杯（只放开参数的结果冻结后直接跑，不参与任何选择）：" + "，".join(
                f"{p} J {R['params']['eval'][p]['start']['J']:.3f} → {R['params']['eval'][p]['best']['J']:.3f}" for p in ho)]
    o = {m: opens(R[m]["eval"]) for m in R}
    lines += ["", "## 判定"]
    if not R:
        lines.append("没有完成的试验。")
    elif "both_wide" in o and not o["both_wide"] and not o.get("both", False):
        lines.append("1 物理模型的问题：参数随便放、入口放宽到测量误差的 2.5–3 倍，郁金香仍打不开。表层模型本身缺机制。")
    elif o.get("both") and o.get("params"):
        lines.append("2 物理参数的问题：入口按测量值，只放开参数就能打开（误差内修正入口也能）。倒奶模型在误差内够用。")
    elif o.get("both") and not o.get("params") and o.get("inlet"):
        lines.append("3 倒奶模型的问题：参数固定、只在测量误差内修正入口就能打开；只放开参数打不开。")
    elif o.get("both") and not o.get("params") and not o.get("inlet"):
        lines.append("2 + 3：参数和倒奶模型都有问题——两者一起放开（入口在误差内）才打开，单放一个都不够。")
    elif o.get("both_wide") and not o.get("both"):
        lines.append("1（为主）+ 3：只有把入口改到超出测量误差才打开。要么倒奶误差比估计的大得多，要么表层模型缺机制、"
                     "靠扭曲入口来凑；看 D 的入口改动图判断改动是否像真实的手。")
    else:
        lines.append("结果不全或互相矛盾，见上表和各试验的 run.log。")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(root, "summary.md"), "w").write(txt)
    print(txt)


# ----------------------------------------------------------------------------------------------- gradient check
def gradcheck(a):
    """Adjoint vs central finite differences of the loss w.r.t. inlet knot variables and two closure numbers, and the
    bitwise identity of a zero correction."""
    from .render512 import load_closure
    num, sw = load_closure(a.init)
    D = make_pour(a.pour, sw, dict(N=a.N, device=a.device, T=a.T, box="err"))
    P = full_P_host(D.s1, num, sw)
    S0 = {k: D.Sw[k].numpy().copy() for k in W.SCHED}
    Sz = D.inl.apply(D.inl.zero())
    same = all(np.array_equal(Sz[k], S0[k]) for k in W.SCHED)
    print(f"zero correction reproduces the record bitwise: {same}")
    rng = np.random.default_rng(3)
    u = np.clip(0.5 + 0.15 * rng.standard_normal((6, D.inl.K)), 0.05, 0.95)
    r = step(D, P, 0.05, u)
    gu = r["gu"]
    act_k = np.unique(np.r_[D.inl.j[D.inl.act], D.inl.j[D.inl.act] + 1])
    worst, ok = 0.0, same
    for v in range(6):
        for k in rng.choice(act_k, size=min(2, len(act_k)), replace=False):
            h = a.eps
            up, dn = u.copy(), u.copy()
            up[v, k] += h; dn[v, k] -= h
            Lp = step_loss(D, P, up)
            Lm = step_loss(D, P, dn)
            fd = (Lp - Lm) / (2 * h)
            ad = gu[v, k]
            rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-12)
            worst = max(worst, rel if max(abs(fd), abs(ad)) > 1e-9 else 0.0)
            print(f"  {IV[v]:4s} knot {k:3d}: adjoint {ad:+.6e}  FD {fd:+.6e}  rel {rel:.2e}")
    for key in ("kappa_Q", "beta"):
        h = 1e-5 * float(P[key])
        Pp, Pm = dict(P), dict(P)
        Pp[key] += h; Pm[key] -= h
        fd = (step_loss(D, Pp, u) - step_loss(D, Pm, u)) / (2 * h)
        ad = r["g"][key]
        rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-12)
        worst = max(worst, rel)
        print(f"  {key:8s}: adjoint {ad:+.6e}  FD {fd:+.6e}  rel {rel:.2e}")
    ok = ok and worst < a.tol
    print(f"gradcheck {'PASS' if ok else 'FAIL'} (worst relative error {worst:.2e}, tolerance {a.tol:g}; scan {r['scan']:.2f})")
    if not ok:
        raise SystemExit(1)


def step_loss(D, P, u):
    S = D.inl.apply(u)
    assign(D, S)
    D.tau = 0.05
    L, _, _ = D.loss(P)
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
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    g = sub.add_parser("gradcheck")
    g.add_argument("--init", default=os.path.join(os.path.dirname(__file__), "..", "data", "closures", "v05n_AB_cmaes_N48.json"))
    g.add_argument("--pour", default="tulip")
    g.add_argument("--N", type=int, default=48)
    g.add_argument("--T", type=float, default=1.0)
    g.add_argument("--device", default="cpu")
    g.add_argument("--eps", type=float, default=1e-5)
    g.add_argument("--tol", type=float, default=2e-3)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    elif a.cmd == "summary":
        summary(a.out)
    else:
        gradcheck(a)


if __name__ == "__main__":
    main()
