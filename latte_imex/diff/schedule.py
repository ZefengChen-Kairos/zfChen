"""Per-frame arrival record of one pour, exactly as batch/run.run feed solver.Solver: frame length from the inlet at t
(scan-speed bound), inlet sampled at t + fdt/2, the frame re-limited by that inlet's scan speed (Solver.advance_frame).
The record does not depend on the solver state, so it is built once on the host and handed to the Warp solver as
arrays (x_hit, Q = S_eff / kQ, u_in, U_perp, d_jet, r1, r2, phi per frame)."""
from __future__ import annotations

import math

import numpy as np

from ..solver import Numerics

SCHED = ("x", "y", "Q", "ux", "uy", "U", "d", "r1", "r2", "phi")


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
    out = {k: A[:, i].copy() for i, k in enumerate(SCHED)}
    out.update(active=A[:, 10] > 0, fdt=A[:, 11], q=A[:, 12].astype(int), inlets=inlets, kQ=kQ, T=T,
               return_law=script.params.get("return_law", "v1"), n=len(rows))
    return out


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


def pour_script(p, N, e0=False):
    """ProgScript of pattern p's measured pour through the Kona 700 mL pitcher (tulip_feas.Pour inlet) and the S1
    closure as starting parameters; e0: landing point corrected by the measured stream foot (e0_offsets)."""
    from ..rig.tulip_feas import Pour
    from .. import s1_forward as SF
    pour = Pour(p)
    prog = SF.FwdProgram(p, pour.R, N, meta=pour.meta)
    if e0:
        off = e0_offsets(p)
        if off is None:
            raise FileNotFoundError(f"{p}: no E0 stream-foot measurement (python3 -m latte_imex.rig.stream_foot {p})")
        shift_program(prog, *off)
    return SF.ProgScript(prog, p), pour
