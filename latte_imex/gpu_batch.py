"""Batched forward engine for the design and calibration loops: many simulations on one grid, advanced together on the
GPU (cuda_solver.BatchCudaSolver, device buffers reused across calls via BatchCudaSolver.reset) or one after the other
on the CPU (solver.Solver, the reference).  Each simulation is a *member* object that decides its own control frames,
so one engine serves two frame semantics:

  ProgMember   run.run / batch_run.run_batch semantics (S1 programs, s1_forward.ProgScript): frame = num.frame_dt
               limited by the scan speed of the inlet probed at t, inlet sampled at t + fdt/2, until T; per-phase
               accumulators (S-weighted chi and |u_in|, received volume) for the S1 design terms
               (v05_pourable.inlet_strokes / finalize); final field only.
  ClipMember   v05_adapter.V05Model.run semantics (barista's real actions through the physical inlet): frames end
               exactly on the clip's output times, snapshots c (outside-cup cells filled with the nearest cup cell),
               the same ledger keys calib.py reads (ok, error, layer_error, ...).

    eng = BatchEngine(N=96, backend="cuda")           # or "cpu"
    eng.run(members, chunk=0)                          # members get .result filled in

A member whose frame keeps failing (non-finite / negative layer after the retries) is marked failed (result['ok'] =
False, error text) without stopping the batch (BatchCudaSolver.advance_frames(drop_failed=True)).
"""
from __future__ import annotations
import math
import time

import numpy as np

from .solver import Numerics, Params, Solver

ML_D3 = 512.0          # mL per D^3 (D = 0.08 m)
D_M = 0.08


def _probe(script, t, num, h):
    """batch_run._probe: frame length from the inlet at t (scan-speed bound), inlet and phase label at t + fdt/2."""
    probe, _ = script.sample(t)
    fdt = num.frame_dt
    if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
        fdt = min(fdt, max(num.scan_safety * min(h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
    inl, phase = script.sample(t + 0.5 * fdt)
    return inl, phase, fdt


# =========================================================================== members
class ProgMember:
    """One run.run-style simulation of a program script (ProgScript: .T, .params, .sample(t), .prog).

    params: solver.Params; kQ: the program's kappa_Q (S_eff = kQ Q) for the received volume; T_max truncates."""

    def __init__(self, script, params, T_max=None, tag=None):
        self.script, self.P, self.tag = script, params, tag
        self.T = script.T if T_max is None else min(float(T_max), script.T)
        self.prog = getattr(script, "prog", None)
        self.kQ = float(getattr(self.prog, "kQ", params.kappa_Q))
        self.done = False
        self.failed = None
        self.acc = {}                     # phase index -> [S dt, S dt chi, S dt |u_in| D, received volume mL, frames]
        self.n_frames = 0
        self.result = None
        self._cur = None

    def snap_due(self, t):
        return False

    def next_frame(self, t, num, h):
        if t >= self.T - 1e-9:
            return None
        inl, ph, fdt = _probe(self.script, t, num, h)
        k = self.prog.phase_index(t + 0.5 * fdt) if self.prog is not None else 0
        self._cur = (inl, k)
        return inl, fdt

    def after(self, t0, t1, chi, Z):
        inl, k = self._cur
        dt = t1 - t0
        self.n_frames += 1
        if not (inl.active and inl.S_eff > 0):
            return
        a = self.acc.setdefault(k, [0.0, 0.0, 0.0, 0.0, 0])
        w = inl.S_eff * dt
        a[0] += w
        a[1] += w * chi
        a[2] += w * math.hypot(*inl.u_in) * D_M
        a[3] += inl.S_eff / self.kQ * dt * ML_D3
        a[4] += 1

    def fail(self, msg):
        self.failed = msg
        self.done = True

    def finish(self, c, ledger, stats, wall):
        strokes = {}
        for k, a in sorted(self.acc.items()):
            if a[0] > 0:
                strokes[k + 1] = dict(chi=a[1] / a[0], uh_ms=a[2] / a[0], vol_mL=a[3], frames=a[4])
        ok = self.failed is None and bool(np.isfinite(ledger["layer"]))
        self.result = dict(c=c, ledger=ledger, stats=dict(stats), strokes=strokes, ok=bool(ok), error=self.failed or "",
                           T=self.T, wall_s=wall)


class ClipMember:
    """v05_adapter.V05Model.run on one clip: same frame logic, snapshots and ledger keys (see module docstring)."""

    def __init__(self, model, script, params, N, t_out, cup_radius=0.49, T_max=None, tag=None):
        self.model, self.params, self.tag = model, dict(params or {}), tag
        self.P = model.solver_params(params)
        self.t_out = np.asarray(t_out, float)
        self.inl, self.t_start = model.inlet(script, params, cup_radius, t_end=float(self.t_out.max()))
        self.tm = self.t_out - self.t_start
        tg = np.unique(self.tm[self.tm > 1e-9])
        self.truncated = False
        if T_max is not None and len(tg) and tg[-1] > T_max:
            tg = tg[tg <= T_max + 1e-9]
            self.truncated = True
        self.targets = tg
        self.j = 0
        self.N = N
        self.snaps = np.empty((len(self.t_out), N, N), np.float32)
        self.filled = np.zeros(len(self.t_out), bool)
        self.need0 = bool(np.any(self.tm <= 1e-9))
        self.done = False
        self.failed = None
        self.Vw = self.Vr = self.chi_w = self.chi_q = 0.0
        self.n_off = 0
        self.rng = {}
        self.result = None
        self._st = None
        self._fill_idx = None

    def _fill(self, c, mask):
        if self._fill_idx is None:
            self._fill_idx = self.model._nearest(mask)
        iy, ix = self._fill_idx
        c = np.nan_to_num(c, nan=0.0)
        return np.where(mask, c, c[iy, ix]).astype(np.float32)

    def snap_due(self, t):
        if self.need0:
            return True
        return self.j < len(self.targets) and t >= self.targets[self.j] - 1e-9

    def store(self, c, t, mask):
        cf = self._fill(c, mask)
        if self.need0:
            for i in np.flatnonzero(self.tm <= 1e-9):
                self.snaps[i] = cf; self.filled[i] = True
            self.need0 = False
        while self.j < len(self.targets) and t >= self.targets[self.j] - 1e-9:
            for i in np.flatnonzero(np.abs(self.tm - self.targets[self.j]) < 1e-9):
                self.snaps[i] = cf; self.filled[i] = True
            self.j += 1

    def next_frame(self, t, num, h):
        if self.j >= len(self.targets):
            return None
        tgt = self.targets[self.j]
        fdt = num.frame_dt
        probe = self.inl.state(t)["inlet"]
        if probe.active and probe.scan_speed > 0:
            fdt = min(fdt, max(num.scan_safety * min(h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
        fdt = min(fdt, tgt - t)
        st = self.inl.state(t + 0.5 * fdt)
        self._st = st
        return st["inlet"], fdt

    def after(self, t0, t1, chi, Z):
        st = self._st
        dt = t1 - t0
        self.Vw += st["Q_w"] * dt
        if st["active"]:
            self.Vr += st["Q"] * dt
            self.chi_w += chi * st["Q"] * dt
            self.chi_q += st["Q"] * dt
            for key, v in (("Q_mLs", st["Q"] / 1e-6), ("z_cm", st["z"] * 100), ("U_ms", st.get("U", np.nan)),
                           ("chi", chi)):
                if v is not None and np.isfinite(v):
                    lo, hi = self.rng.get(key, (v, v))
                    self.rng[key] = (min(lo, v), max(hi, v))
        self.n_off += int(st["off_cup"])

    def fail(self, msg):
        self.failed = msg
        self.done = True

    def finish(self, c, ledger, stats, wall, mask, t_model):
        if not self.filled.all():
            self.snaps[~self.filled] = self._fill(c, mask)
        ok = self.failed is None and (bool(self.filled.all()) or self.truncated)
        L = ledger
        led = dict(ok=bool(ok and np.isfinite(L["layer"])), error=self.failed or "", truncated=self.truncated,
                   t_start=self.t_start, T_model=float(t_model), layer_error=float(L["layer_error"]),
                   milk_error=float(L["milk_error"]), layer=float(L["layer"]), milk=float(L["milk"]),
                   deposited=float(L["deposited"]), spilled=float(L.get("spilled", 0.0)),
                   offcup_frames=int(stats.get("offcup_frames", 0)), kernel_clamp=int(stats.get("kernel_clamp", 0)),
                   V_wanted_mL=self.Vw / 1e-6, V_received_mL=self.Vr / 1e-6,
                   chi_flow_weighted=self.chi_w / self.chi_q if self.chi_q > 0 else None, frames_off_cup=self.n_off,
                   ranges={k: [float(v[0]), float(v[1])] for k, v in self.rng.items()},
                   steps=int(stats["steps"]), frames=int(stats["frames"]), retries=int(stats["retries"]),
                   max_substeps=int(stats["max_substeps"]), wall_s=wall, closure=self.model.closure(self.params),
                   u_exit=self.inl.u_exit, tau_spout=self.inl.tau, N=self.N)
        self.result = dict(c=self.snaps, ledger=led)


# =========================================================================== engine
class BatchEngine:
    """Advance a list of members on one grid.  backend 'cuda': BatchCudaSolver per batch shape, kept and reset between
    calls (no reallocation, no recompilation); 'cpu': solver.Solver per member (reference)."""

    def __init__(self, N, backend="cuda", numerics=None, dtype=np.float64, log=None):
        self.num = Numerics(N=N, **(numerics or {}))
        self.N = N
        self.backend = backend
        self.dtype = dtype
        self.log = log
        self._solvers = {}
        self.calls = 0

    def _solver(self, Ps):
        from .cuda_solver import BatchCudaSolver
        B = len(Ps)
        keys = lambda P: (P.visc, P.offcup, P.nu > 0, P.D > 0, P.closure)
        S = self._solvers.get(B)
        if S is not None and keys(S.P) == keys(Ps[0]):
            S.reset(Ps)
            return S
        if S is not None:
            del self._solvers[B]
        S = BatchCudaSolver(Ps, self.num, dtype=self.dtype)
        self._solvers[B] = S
        return S

    def run(self, members, chunk=0):
        chunk = chunk or len(members)
        for c0 in range(0, len(members), chunk):
            part = members[c0:c0 + chunk]
            if self.backend == "cuda":
                self._run_cuda(part)
            else:
                for m in part:
                    self._run_cpu(m)
        self.calls += 1
        return members

    # --------------------------------------------------------------------- CPU reference
    def _run_cpu(self, m):
        t0 = time.time()
        num = self.num
        sol = Solver(m.P, num)
        g = sol.g
        try:
            while True:
                if m.snap_due(sol.t):
                    m.store(sol.concentration(), sol.t, g.mask)
                r = m.next_frame(sol.t, num, g.h)
                if r is None:
                    break
                tb = sol.t
                chi, _ = sol.advance_frame(r[0], frame_dt=r[1])
                m.after(tb, sol.t, chi, float(getattr(sol, "last_Z", 1.0)))
        except Exception as e:                         # recorded, scored by the caller
            m.fail(f"{type(e).__name__}: {e}")
        c = sol.concentration()
        L = sol.ledger()
        if isinstance(m, ClipMember):
            m.finish(c, L, sol.stats, time.time() - t0, g.mask, sol.t)
        else:
            m.finish(c, L, sol.stats, time.time() - t0)

    # --------------------------------------------------------------------- GPU batch
    def _run_cuda(self, mem):
        t0 = time.time()
        S = self._solver([m.P for m in mem])
        num, g = self.num, S.g
        B = len(mem)
        frame = 0
        while True:
            due = [b for b, m in enumerate(mem) if not m.done and m.snap_due(float(S.t[b]))]
            if due:
                C = S.concentrations_all()
                for b in due:
                    mem[b].store(C[b], float(S.t[b]), g.mask)
            inlets, fdts = [None] * B, [None] * B
            for b, m in enumerate(mem):
                if m.done:
                    continue
                try:
                    r = m.next_frame(float(S.t[b]), num, g.h)
                except Exception as e:
                    m.fail(f"{type(e).__name__}: {e}")
                    continue
                if r is None:
                    m.done = True
                    continue
                inlets[b], fdts[b] = r
            if all(i is None for i in inlets):
                break
            tb = S.t.copy()
            res = S.advance_frames(inlets, fdts, drop_failed=True)
            for b, m in enumerate(mem):
                if inlets[b] is None:
                    continue
                if res[b] is None:
                    m.fail("solver: sub-step failed repeatedly (non-finite or negative layer)")
                    continue
                m.after(float(tb[b]), float(S.t[b]), res[b][0], float(S.last_Z[b]))
            frame += 1
            if self.log and frame % 2000 == 0:
                act = sum(1 for m in mem if not m.done)
                self.log(f"    [engine N={self.N} B={B}] frame {frame}: t {S.t.min():.2f}..{S.t.max():.2f} s, {act} running, "
                         f"{time.time() - t0:.0f} s")
        C = S.concentrations_all()
        wall = time.time() - t0
        for b, m in enumerate(mem):
            L = S.ledger(b)
            if isinstance(m, ClipMember):
                m.finish(C[b], L, S.stats[b], wall, g.mask, float(S.t[b]))
            else:
                m.finish(C[b], L, S.stats[b], wall)
