"""run.run for many programs at once: B members advanced together on the GPU (cuda_solver.BatchCudaSolver), each with
the same per-frame logic, outputs and metrics as latte_imex.run.run (backend 'cpu' runs the members one after the
other with solver.Solver, the reference).

    from latte_imex.batch_run import run_batch
    mets = run_batch([dict(script=s, pattern="heart", out="runs/x/heart"), ...], N=256, video=True)

A job's script has .T, .params (Params kwargs) and .sample(t) -> (Inlet, phase label), like run.V05Script.
"""
from __future__ import annotations
import json
import os
import time

import numpy as np

from . import render
from .solver import Params, Numerics, Solver

CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="压纹郁金香", leaf="树叶", swan="天鹅")


class _Member:
    def __init__(self, job, snap_dt):
        self.job = job
        self.script = job["script"]
        self.T = self.script.T if job.get("T_max") is None else min(job["T_max"], self.script.T)
        self.snaps, self.traj, self.events = [], [], []
        self.next_snap = 0.0
        self.last_phase = None
        self.done = False
        self.snap_dt = snap_dt
        self.t_end = None


def _probe(script, t, num, h):
    probe, _ = script.sample(t)
    fdt = num.frame_dt
    if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
        fdt = min(fdt, max(num.scan_safety * min(h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
    inl, phase = script.sample(t + 0.5 * fdt)
    return inl, phase, fdt


def run_batch(jobs, N=256, backend="cuda", snaps=True, video=False, snap_dt=1.0 / 24.0, numerics=None, quiet=False,
              log=print, dtype=np.float64, save_state=True):
    """Run every job; writes <out>/{final.png, final_state.npz, c.npz, metrics.json, trajectory.json[, snapshots_c.npz,
    movie.mp4]} per job and returns the list of metrics dicts."""
    num = Numerics(N=N, **(numerics or {}))
    Ps = [Params(**dict(j["script"].params, **(j.get("params") or {}))) for j in jobs]
    mem = [_Member(j, snap_dt) for j in jobs]
    B = len(jobs)
    t0 = time.time()
    if backend == "cuda":
        from .cuda_solver import BatchCudaSolver
        S = BatchCudaSolver(Ps, num, dtype=dtype)
        g = S.g
        tget = lambda b: float(S.t[b])
        frame = 0
        while True:
            inlets, fdts = [None] * B, [None] * B
            phases = [None] * B
            for b, m in enumerate(mem):
                if m.done:
                    continue
                t = tget(b)
                if t >= m.T - 1e-9:
                    m.done = True; m.t_end = time.time() - t0
                    continue
                inl, ph, fdt = _probe(m.script, t, num, g.h)
                inlets[b], fdts[b], phases[b] = inl, fdt, ph
            if all(i is None for i in inlets):
                break
            res = S.advance_frames(inlets, fdts)
            want = []
            for b, m in enumerate(mem):
                if res[b] is None:
                    continue
                _record(m, inlets[b], phases[b], res[b][0], tget(b), float(S.last_Z[b]))
                if snaps and tget(b) + 1e-9 >= m.next_snap:
                    want.append(b)
            if want:
                cs = S.concentrations(want)
                for b in want:
                    mem[b].snaps.append((tget(b), cs[b].astype(np.float16)))
                    mem[b].next_snap += snap_dt
            frame += 1
            if not quiet and frame % 240 == 0:
                act = sum(1 for m in mem if not m.done)
                log(f"  [batch N={N} B={B}] frame {frame}: t = {min(tget(b) for b in range(B)):.2f}..{max(tget(b) for b in range(B)):.2f} s, "
                    f"{act} running, {time.time() - t0:.0f} s")
        solver_s = time.time() - t0
        out = []
        cs = S.concentrations(list(range(B)))
        for b, m in enumerate(mem):
            out.append(_finish(m, Ps[b], num, cs[b], S.host_state(b), S.ledger(b), S.energy(b), S.stats[b], g,
                               m.t_end or solver_s, backend, video, save_state))
        return out
    out = []
    for b, m in enumerate(mem):
        tb = time.time()
        sol = Solver(Ps[b], num)
        while sol.t < m.T - 1e-9:
            inl, ph, fdt = _probe(m.script, sol.t, num, sol.g.h)
            chi, nsub = sol.advance_frame(inl, frame_dt=fdt)
            _record(m, inl, ph, chi, sol.t, float(getattr(sol, "last_Z", 1.0)))
            if snaps and sol.t + 1e-9 >= m.next_snap:
                m.snaps.append((sol.t, sol.concentration().astype(np.float16)))
                m.next_snap += snap_dt
        out.append(_finish(m, Ps[b], num, sol.concentration(), sol.host_state(), sol.ledger(), sol.energy(), sol.stats,
                           sol.g, time.time() - tb, backend, video, save_state))
    return out


def _record(m, inl, phase, chi, t, Z):
    if phase != m.last_phase:
        m.events.append(dict(t=round(t, 4), phase=phase))
        m.last_phase = phase
    act = bool(inl.active and inl.S_eff > 0)
    Zr = Z if act else 1.0
    m.traj.append(dict(t=t, x=inl.x_hit[0], y=inl.x_hit[1], active=act, phase=phase,
                       S=float(inl.S_eff if inl.active else 0.0), chi=float(chi), Z=Zr, off_cup=bool(act and Zr < 1.0)))


def _finish(m, P, num, c, hs, L, en, stats, g, elapsed, backend, video, save_state):
    job = m.job
    out = job["out"]
    pattern = job["pattern"]
    os.makedirs(out, exist_ok=True)
    c = np.where(g.mask, c, np.nan)
    np.savez_compressed(os.path.join(out, "c.npz"), c=c.astype(np.float32), mask=g.mask)
    if save_state:
        np.savez_compressed(os.path.join(out, "final_state.npz"), l=hs["l"], qx=hs["qx"], qy=hs["qy"], m=hs["m"], mask=g.mask)
    if m.snaps:
        np.savez_compressed(os.path.join(out, "snapshots_c.npz"), t=np.array([s[0] for s in m.snaps]),
                            c=np.stack([s[1] for s in m.snaps]).astype(np.float16))
    t_end = m.traj[-1]["t"] if m.traj else 0.0
    render.save_final(os.path.join(out, "final.png"), c, g.mask, title=f"{job.get('label', pattern)}  t={t_end:.2f}s")
    ke, pe = en
    metrics = dict(pattern=pattern, label=job.get("label", pattern), control=job.get("control", "s1_forward"), N=g.N,
                   T=m.T, elapsed_s=elapsed, params=P.as_dict(), numerics=vars(num), backend=backend, stats=stats,
                   ledger=L, kinetic_energy=ke, pressure_energy=pe,
                   white_area_c_gt_0p4=float(np.nansum(c > 0.4) * g.area), events=m.events, extra=job.get("extra", {}))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, ensure_ascii=False, default=float)
    with open(os.path.join(out, "trajectory.json"), "w") as f:
        json.dump(m.traj, f)
    if video and m.snaps:
        snaps = [(t, s.astype(np.float32)) for t, s in m.snaps]
        render.make_video(os.path.join(out, "movie.mp4"), snaps, m.traj, g.mask, pattern,
                          fps=int(round(1 / m.snap_dt)), hold_frames=int(round(1 / m.snap_dt)), cn=CN.get(pattern, ""))
    m.snaps = []
    return metrics
