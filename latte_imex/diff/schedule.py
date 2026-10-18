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
    """Make prog(t) return inlets with the landing point shifted (foot_shift), for runs outside the Warp solver."""
    if not a and not b:
        return prog
    base = type(prog)

    class Shifted(base):
        def __call__(self, t):
            inl = base.__call__(self, t)
            if inl.active:
                x, y = foot_shift(inl.x_hit[0], inl.x_hit[1], inl.u_in[0], inl.u_in[1], a, b)
                inl.x_hit = (float(x), float(y))
            return inl
    prog.__class__ = Shifted
    return prog


def pour_script(p, N):
    """ProgScript of pattern p's measured pour through the Kona 700 mL pitcher (tulip_feas.Pour inlet) and the S1
    closure as starting parameters."""
    from ..rig.tulip_feas import Pour
    from .. import s1_forward as SF
    pour = Pour(p)
    prog = SF.FwdProgram(p, pour.R, N, meta=pour.meta)
    return SF.ProgScript(prog, p), pour
