"""calib.py model adapter for the v3-min prototypes (px_model.py: PXSim on a Cartesian grid, c[ix, iy], SI units).

Observed-inlet mode (senior V1 'observed_inlet'): the stream foot x_j(t), the flow Q(t), the spout height z(t) and the
horizontal jet direction e(t) = (cos yaw, sin yaw) come directly from the physical-time stage script (the barista's
intended path as pitcher.Barista.intent defines it: cosine easing, optional Bezier, wiggle with 0.25 s ease-in/out);
the pitcher chain is not used.  The delayed supply clock t_k is the onset of the current pour episode (Q rising from 0).

    m = V3MinModel("/path/to/proto_dir")          # directory with px_model.py
    out = m.run(calib.physical_script("tulip"), {"Gamma": 0.04}, 128, t_out)
"""
import importlib.util
import math
import os
import sys

import numpy as np

from . import video_extract as VE

D_CUP = VE.D_CUP


def load_module(code_dir, name="px_model"):
    path = os.path.join(code_dir, f"{name}.py")
    key = f"_v3min_{abs(hash(os.path.abspath(path)))}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, code_dir)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(code_dir)
    sys.modules[key] = mod
    return mod


class Inlet:
    """Observed inlet from a physical-time stage script: t -> (x_D, y_D, z_m, Q_m3s) in the M frame."""

    def __init__(self, script, Q_scale=1.0, Q_off=1e-9):
        moves, _ = VE.stages_to_moves(script, away=False)
        self.moves = moves
        self.T = sum(m.dur for m in moves)
        self.Q_scale = Q_scale
        self.yaw = VE.script_yaw(script)
        # pour episodes: onset times where the wanted flow rises from zero
        ts = np.arange(0.0, self.T + 1e-9, 0.005)
        q = np.array([self.at(t)[3] for t in ts])
        on = (q > Q_off)
        self.onsets = [float(ts[i]) for i in range(len(ts)) if on[i] and (i == 0 or not on[i - 1])]

    def at(self, t):
        acc = 0.0
        for m in self.moves:
            if t < acc + m.dur:
                f = (t - acc) / m.dur
                e = 0.5 - 0.5 * math.cos(math.pi * f)
                amp = m.wobble + (m.wobble1 - m.wobble) * f
                amp *= min(1.0, (t - acc) / 0.25, (acc + m.dur - t) / 0.25)
                wob = amp * math.sin(2 * math.pi * m.wfreq * (t - acc))
                if m.bezier is not None:
                    (ax, ay), (bx, by) = m.bezier
                    x = (1 - e) ** 3 * m.p0[0] + 3 * (1 - e) ** 2 * e * ax + 3 * (1 - e) * e * e * bx + e ** 3 * m.p1[0]
                    y = (1 - e) ** 3 * m.p0[1] + 3 * (1 - e) ** 2 * e * ay + 3 * (1 - e) * e * e * by + e ** 3 * m.p1[1]
                else:
                    x = m.p0[0] + (m.p1[0] - m.p0[0]) * e
                    y = m.p0[1] + (m.p1[1] - m.p0[1]) * e
                if m.wobble_dir is not None:
                    x += wob * math.cos(m.wobble_dir); y += wob * math.sin(m.wobble_dir)
                else:
                    x += wob if m.wobble_axis == "x" else 0.0
                    y += wob if m.wobble_axis == "y" else 0.0
                z = m.z0 + (m.z1 - m.z0) * e
                Q = (m.Q0 + (m.Q1 - m.Q0) * f) * self.Q_scale
                return x, y, z, Q
            acc += m.dur
        last = self.moves[-1]
        return last.p1[0], last.p1[1], last.z1, 0.0

    def onset_before(self, t):
        k = 0.0
        for o in self.onsets:
            if o <= t + 1e-9:
                k = o
        return k


class V3MinModel:
    """space: the fitted parameters (v3-min-2 section 4.2 bounds); extra fitted parameters of a later version are
    passed in `space_extra`.  `fixed` overrides the prototype's FIXED/V2 options."""
    SPACE = dict(Gamma=(1.5e-3, 0.7, "log"), tau_w=(0.3, 3.0, "log"), delta_e=(0.4e-3, 9e-3, "log"),
                 Pi_coh=(30.0, 2000.0, "log"))

    def __init__(self, code_dir, fixed=None, space_extra=None, base="V2", dt_max=0.02):
        self.mod = load_module(code_dir)
        self.code_dir = code_dir
        self.fixed = dict(getattr(self.mod, base, {}) or {})
        if fixed:
            self.fixed.update(fixed)
        self.space = dict(self.SPACE)
        if space_extra:
            self.space.update(space_extra)
        self.dt_max = dt_max
        self._grids = {}

    def grid(self, N):
        if N not in self._grids:
            self._grids[N] = self.mod.Grid(N)
        return self._grids[N]

    def run(self, script, params, N, t_out):
        mod = self.mod
        inl = Inlet(script, Q_scale=float(params.get("Q_scale", 1.0)))
        ex, ey = math.cos(inl.yaw), math.sin(inl.yaw)
        p = dict(self.fixed)
        p.update({k: v for k, v in params.items() if k != "Q_scale"})
        sim = mod.PXSim(grid=self.grid(N), params=p, dt_max=self.dt_max)

        def inp(t):
            x, y, z, Q = inl.at(t)
            return dict(x=x * D_CUP, y=y * D_CUP, Q=Q, v=mod.v_from_zs(max(z, 0.0)), ex=ex, ey=ey)

        t_out = np.asarray(t_out, float)
        snaps = np.zeros((len(t_out), N, N), np.float32)
        filled = np.zeros(len(t_out), bool)

        def cb(s, t):
            for i in np.flatnonzero(np.abs(t_out - t) < 1e-9):
                snaps[i] = s.c.T
                filled[i] = True

        bps = sorted(set([m for m in np.cumsum([mv.dur for mv in inl.moves])] + inl.onsets))
        cb_t = sorted(set(float(t) for t in t_out if t > 1e-12))
        for i in np.flatnonzero(t_out <= 1e-12):
            filled[i] = True
        ok, err = True, ""
        try:
            sim.run(inp, float(t_out.max()), inl.onset_before, breakpoints=bps, callback=cb, cb_times=cb_t)
        except Exception as e:                         # a failed simulation is recorded, never silently scored
            ok, err = False, f"{type(e).__name__}: {e}"
        L = sim.ledgers()
        led = dict(ok=ok and bool(filled.all()), error=err, foam_rel=float(L["foam_rel"]), surf_err=float(L["surf_err"]),
                   clip_c=float(L["clip_c"]), max_c=float(sim.stats["max_c"]), min_c=float(sim.stats["min_c"]),
                   steps=int(sim.stats["steps"]), wall_s=float(sim.stats["t_wall"]), T=float(sim.t))
        return dict(c=snaps, ledger=led)
