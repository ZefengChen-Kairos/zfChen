"""Per-frame arrival record of one pour, exactly as batch/run.run feed solver.Solver: frame length from the inlet at t
(scan-speed bound), inlet sampled at t + fdt/2, the frame re-limited by that inlet's scan speed (Solver.advance_frame).
The record does not depend on the solver state, so it is built once on the host and handed to the Warp solver as
arrays (x_hit, Q = S_eff / kQ, u_in, U_perp, d_jet, r1, r2, phi per frame)."""
from __future__ import annotations

import math

import numpy as np

from ..solver import Numerics

SCHED = ("x", "y", "Q", "ux", "uy", "U", "d", "r1", "r2", "phi", "vjx", "vjy")   # = wsolver.SCHED (vjx, vjy: landing-point velocity)


def build(script, N, num=None, T_max=None):
    """script: s1_forward.ProgScript (or anything with .T, .sample(t), .prog.kQ).  Returns dict of numpy arrays
    (SCHED keys, active, fdt, q) + 'inlets' (the solver.Inlet objects) + kQ, T, return_law."""
    num = num or Numerics(N=N)
    h = 1.0 / N
    T = script.T if T_max is None else min(script.T, T_max)
    kQ = float(script.prog.kQ)
    t = 0.0
    rows, inlets = [], []
    while t < T - 1e-9:
        probe, _ = script.sample(t)
        fdt = num.frame_dt
        if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
            fdt = min(fdt, max(num.scan_safety * min(h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
        inl, _ = script.sample(t + 0.5 * fdt)
        if inl.active and inl.S_eff > 0 and inl.scan_speed > 0:
            fdt = min(fdt, max(num.scan_safety * min(h, inl.r1, inl.r2) / inl.scan_speed, 1e-4))
        act = bool(inl.active and inl.S_eff > 0)
        r1, r2 = max(inl.r1, 1.5 * h), max(inl.r2, 1.5 * h)
        q = min(max(num.kernel_quadrature, int(math.ceil(4 * h / min(r1, r2)))), 12)
        rows.append((inl.x_hit[0], inl.x_hit[1], inl.S_eff / kQ, inl.u_in[0], inl.u_in[1], inl.U_perp, inl.d_jet,
                     inl.r1, inl.r2, inl.phi, float(act), fdt, q))
        inlets.append(inl)
        t += fdt
    A = np.array(rows, float)
    nb = len(SCHED) - 2                                    # the per-frame inlet columns; vjx, vjy follow
    out = {k: A[:, i].copy() for i, k in enumerate(SCHED[:nb])}
    out["vjx"], out["vjy"] = foot_velocity(A[:, 0], A[:, 1], A[:, nb], A[:, nb + 1])
    for inl, vx, vy in zip(inlets, out["vjx"], out["vjy"]):
        inl.v_hit = (float(vx), float(vy))
    out.update(active=A[:, nb] > 0, fdt=A[:, nb + 1], q=A[:, nb + 2].astype(int), inlets=inlets, kQ=kQ, T=T,
               return_law=script.params.get("return_law", "v1"), n=len(rows))
    return out


V_FOOT_TAU = 0.15        # [s] smoothing time of the landing-point velocity V_j (filters the frame-to-frame E0 jitter)


def foot_velocity(x, y, active, fdt, tau=V_FOOT_TAU):
    """V_j: velocity of the landing point itself (exponential moving average of its frame-to-frame displacement over
    tau, between consecutive active frames; reset to 0 when the stream is off), the speed at which the stream column
    moves through the surface (solver.Params.vj)."""
    vx, vy = np.zeros(len(x)), np.zeros(len(x))
    v = np.zeros(2)
    for k in range(1, len(x)):
        if active[k] > 0 and active[k - 1] > 0:
            raw = np.array([x[k] - x[k - 1], y[k] - y[k - 1]]) / max(fdt[k - 1], 1e-6)
            a = min(1.0, fdt[k - 1] / tau)
            v = (1.0 - a) * v + a * raw
        else:
            v = np.zeros(2)
        vx[k], vy[k] = v
    return vx, vy


FOOT = ("foot_a", "foot_b")     # global landing-point correction [cup diameters]: along / across the jet direction


def foot_dirs(ux, uy):
    """Unit vector of the jet's horizontal arrival velocity per frame (zero where it is zero)."""
    u = np.hypot(ux, uy)
    ok = u > 1e-12
    return np.where(ok, ux / np.where(ok, u, 1.0), 0.0), np.where(ok, uy / np.where(ok, u, 1.0), 0.0)


def foot_shift(x, y, ux, uy, a, b):
    """Landing point moved by a along the jet's horizontal direction and b across it (to its left)."""
    ex, ey = foot_dirs(np.asarray(ux, float), np.asarray(uy, float))
    return x + a * ex - b * ey, y + a * ey + b * ex


def shift_program(prog, a, b):
    """Make prog(t) return inlets with the landing point shifted (foot_shift); a, b numbers or functions of t."""
    if not callable(a) and not callable(b) and not a and not b:
        return prog
    base = type(prog)
    fa = a if callable(a) else (lambda t, v=float(a): v)
    fb = b if callable(b) else (lambda t, v=float(b): v)

    class Shifted(base):
        def __call__(self, t):
            inl = base.__call__(self, t)
            if inl.active:
                x, y = foot_shift(inl.x_hit[0], inl.x_hit[1], inl.u_in[0], inl.u_in[1], fa(t), fb(t))
                inl.x_hit = (float(x), float(y))
            return inl
    prog.__class__ = Shifted
    return prog


E0_WIN = 0.5             # [s] half window of the running median of the measured offsets
E0_CODES = {0: False, 1: "jet", 2: "world"}     # the "e0" value stored in closure json files


def e0_mode(e0):
    """Normalise an E0 switch: False / 0 (none), True / 1 / 'jet' (e0_offsets, along / across u_in), 2 / 'world'
    (e0_world_offsets, cup frame)."""
    if isinstance(e0, str):
        if e0 not in ("jet", "world"):
            raise ValueError(f"e0 must be 'jet' or 'world', got {e0!r}")
        return e0
    return E0_CODES[int(round(float(e0 or 0)))]


def e0_code(e0):
    return {v: k for k, v in E0_CODES.items()}[e0_mode(e0)]


def shift_program_world(prog, fx, fy):
    """Make prog(t) return inlets with the landing point moved by (fx(t), fy(t)) in the cup frame (solver units)."""
    base = type(prog)

    class ShiftedW(base):
        def __call__(self, t):
            inl = base.__call__(self, t)
            if inl.active:
                inl.x_hit = (float(inl.x_hit[0] + fx(t)), float(inl.x_hit[1] + fy(t)))
            return inl
    prog.__class__ = ShiftedW
    return prog


# E0 world-frame estimator (e0_world_offsets).  Candidate feet: a well-resolved stream (score, centre-line rms, slope)
# reaching the surface within the path search bound (stream_foot.py searches to 12 mm below the tip; its own `trusted`
# flag stops at TOP_OK = 10 mm and so dropped good frames of the fast heart-1 stroke while keeping weak ones that follow
# foam behind the tip).  Feet more than E0_BEHIND behind the spout tip are dropped from every tier; a candidate whose
# offset disagrees with its neighbours by more than E0_GATE is dropped; the offset at time t is a quality-weighted robust
# (Tukey) mean of the kept frames under a Gaussian time window (audit wf_61f4987d-e3b, C1; V2_SURFACE_MODEL.md §9).
E0_CAND = dict(q=0.6, rms_mm=1.0, slope=0.8, top_mm=12.0)
E0_PLAUS = dict(q=0.2, rms_mm=3.5, slope=0.8, top_mm=12.0)
E0_BEHIND = 3.0e-3      # [m] a foot further behind the spout tip (along the stream's horizontal direction u_in) is not
                        # the stream: the falling milk only moves forward; checked on the camera images of heart, where all
                        # such feet were the pitcher's lower edge or the liquid edge in its shadow by the rim
E0_GATE = (0.30, 0.11)   # [s] neighbour window, [solver units ~ 10 mm] consistency limit
E0_SIG, E0_CUT, E0_TUKEY, E0_WMIN = 0.10, 0.30, 0.08, 0.3     # [s], [s], [su], minimum summed weight


def _e0_sel(r, c):
    return r["quality"] > c["q"] and r["rms_mm"] <= c["rms_mm"] and r["slope"] <= c["slope"] and r["top_mm"] <= c["top_mm"]


def e0_world_estimate(t, O, q, keep, plaus, tt):
    """Offset at time tt from per-frame offsets O (n, 2) at times t with quality q: Tukey IRLS (3 iterations from the
    median) of the kept frames weighted by q exp(-(t - tt)^2 / 2 E0_SIG^2) within E0_CUT; where those weigh less than
    E0_WMIN, the median of the plausible frames within E0_WIN (>= 3), else the median of all kept frames."""
    n = keep & (np.abs(t - tt) <= E0_CUT)
    if n.sum() >= 2:
        w0 = q[n] * np.exp(-0.5 * ((t[n] - tt) / E0_SIG) ** 2)
        if w0.sum() > E0_WMIN:
            m = np.median(O[n], 0)
            for _ in range(3):
                r = np.hypot(*(O[n] - m).T) / E0_TUKEY
                w = w0 * np.where(r < 1.0, (1.0 - r * r) ** 2, 0.0)
                if w.sum() <= 0:
                    break
                m = (w[:, None] * O[n]).sum(0) / w.sum()
            return m
    n2 = plaus & (np.abs(t - tt) <= E0_WIN)
    if n2.sum() >= 3:
        return np.median(O[n2], 0)
    return np.median(O[keep], 0)


def e0_world_offsets(p, prog, grid=False):
    """E0 replay in the cup frame: the measured stream foot (rig/stream_foot.py) minus the landing point of prog (the
    model as built, e.g. with the calibrated exit) at the same time, as functions of the sim time (robust estimator
    above).  Unlike e0_offsets it does not use the jet direction (u_in), whose frame-to-frame noise turns a near-
    constant offset into landing-point jitter.  grid: also return (t grid, offsets, kept frame times)."""
    import json
    import os
    from ..rig.real_program import RES
    fn = os.path.join(RES, p, "stream_foot.json")
    if not os.path.exists(fn):
        return None
    rows = []
    for r in json.load(open(fn))["frames"]:
        if _e0_sel(r, E0_PLAUS):
            inl = prog(r["t"])
            un = math.hypot(*inl.u_in) if inl.active else 0.0
            if un > 0:
                # spout tip in solver units (same board -> solver map as the foot); foot ahead of it along u_in [m]
                fb, tb = r["foot_board"], r["tip_board"]
                ahead = ((fb[0] - tb[0]) * inl.u_in[0] + (tb[1] - fb[1]) * inl.u_in[1]) / un     # board y = -solver y
                if ahead < -E0_BEHIND:
                    continue
                rows.append((r["t"], r["x"] - inl.x_hit[0], r["y"] - inl.x_hit[1], r["quality"], _e0_sel(r, E0_CAND)))
    if not rows:
        return None
    R = np.array(rows, float)
    t, O, q, plaus, cand = R[:, 0], R[:, 1:3], R[:, 3], np.ones(len(R), bool), R[:, 4] > 0
    keep = cand.copy()
    for i in np.where(cand)[0]:
        nb = cand & (np.abs(t - t[i]) <= E0_GATE[0])
        nb[i] = False
        if nb.sum() >= 2 and np.hypot(*(O[i] - np.median(O[nb], 0))) > E0_GATE[1]:
            keep[i] = False
    if keep.sum() < 5:
        return None
    med = np.median(O[keep], 0)
    tg = np.arange(t.min() - 1.0, t.max() + 1.0, 0.05)
    G = np.array([e0_world_estimate(t, O, q, keep, plaus, tt) for tt in tg])
    f = (lambda tt: float(np.interp(tt, tg, G[:, 0], left=med[0], right=med[0])),
         lambda tt: float(np.interp(tt, tg, G[:, 1], left=med[1], right=med[1])))
    return (f, (tg, G, t[keep])) if grid else f


def e0_offsets(p):
    """E0 input replay: the measured landing-point offset of pour p (rig/stream_foot.py: the stream foot seen by the
    three cameras minus the model's landing point, along / across the jet), as functions of the sim time: running
    median of the trusted frames over +-E0_WIN s (where they are sparse, of the plausible ones), else the pour median.  A measurement of
    the input, not a fit to the pattern (V1 plan 4.1).  None if the pour has no stream_foot.json."""
    import json
    import os
    from ..rig.real_program import RES
    fn = os.path.join(RES, p, "stream_foot.json")
    if not os.path.exists(fn):
        return None
    allf = [r for r in json.load(open(fn))["frames"] if "along" in r]
    F = [r for r in allf if r.get("trusted")]
    if len(F) < 5:
        return None
    # second tier where trusted frames are sparse (thin, fast stream: the final cut): plausible frames -- a stream
    # path that is roughly straight and starts near the tip, but a weaker image response
    G2 = [r for r in allf if r["quality"] > 0.2 and r["rms_mm"] <= 3.5 and r["slope"] <= 0.8 and r["top_mm"] <= 12.0]
    t = np.array([r["t"] for r in F])
    A = np.array([[r["along"], r["across"]] for r in F])
    t2 = np.array([r["t"] for r in G2])
    A2 = np.array([[r["along"], r["across"]] for r in G2]).reshape(-1, 2)
    med = np.median(A, 0)
    tg = np.arange(min(t.min(), t2.min() if len(t2) else t.min()) - 1.0, max(t.max(), t2.max() if len(t2) else t.max()) + 1.0, 0.05)
    G = np.empty((len(tg), 2))
    for i, tt in enumerate(tg):
        near = np.abs(t - tt) <= E0_WIN
        near2 = np.abs(t2 - tt) <= E0_WIN if len(t2) else np.zeros(0, bool)
        G[i] = (np.median(A[near], 0) if near.sum() >= 3 else
                np.median(A2[near2], 0) if near2.sum() >= 3 else med)
    return (lambda tt: float(np.interp(tt, tg, G[:, 0], left=med[0], right=med[0])),
            lambda tt: float(np.interp(tt, tg, G[:, 1], left=med[1], right=med[1])))


def pour_script(p, N, e0=False, exit_cal=False):
    """ProgScript of pattern p's measured pour through the Kona 700 mL pitcher (tulip_feas.Pour inlet) and the S1
    closure as starting parameters.  exit_cal: the stream exit calibrated to the measured stream (real_program.EXIT_CAL);
    e0: landing point corrected by the measured stream foot -- True / 'jet': e0_offsets (along / across u_in, relative to
    the default exit), 'world': e0_world_offsets (cup frame, relative to the program as built)."""
    from ..rig.tulip_feas import Pour
    from ..rig import real_program as RP
    from .. import s1_forward as SF
    pour = Pour(p, law=RP.EXIT_CAL if exit_cal else None)
    prog = SF.FwdProgram(p, pour.R, N, meta=pour.meta)
    e0 = e0_mode(e0)
    if e0:
        off = e0_world_offsets(p, prog) if e0 == "world" else e0_offsets(p)
        if off is None:
            raise FileNotFoundError(f"{p}: no E0 stream-foot measurement (python3 -m latte_imex.rig.stream_foot {p})")
        (shift_program_world if e0 == "world" else shift_program)(prog, *off)
    return SF.ProgScript(prog, p), pour
