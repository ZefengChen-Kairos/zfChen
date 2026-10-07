"""Run one pattern: python -m latte_imex.run --pattern heart --N 256 --out runs/heart"""
import argparse, json, os, sys, time
import numpy as np
from .solver import Solver, Params, Numerics
from .actions import SCRIPTS
from . import render
from .v05_program import program as v05_program, NAMES as V05_NAMES


class V05Script:
    """Adapter: V0.5 control JSON -> (Inlet, label) sampler with the V0.5 shared parameters."""

    def __init__(self, name):
        self.prog = v05_program(name)
        self.name = name
        self.T = self.prog.duration
        self.params = dict(self.prog.config["parameters"])
        self.params["return_law"] = "v05" if self.prog.config.get("return_law", {}).get("mode") == "return_speed" else "v1"

    def sample(self, t):
        if t >= self.T:
            return __import__("latte_imex.solver", fromlist=["Inlet"]).Inlet(active=False), "settle"
        return self.prog(t), self.prog.phase_label(t)

DEFAULT_PARAMS = dict(cp=0.5, beta=3.0, nu=1e-3, D=1e-7, kappa_Q=125.0, kappa_c=1.0, kappa_t=0.7, kappa_r=0.3,
                      B_dep=600.0, p_dep=2.0, return_law="v1")


CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="压纹郁金香", leaf="树叶", swan="天鹅")


def run(pattern, N=256, out=None, params=None, snap_dt=1.0 / 24.0, video=True, quiet=False, T_max=None, script_kw=None,
        control="guess", best=None):
    if control == "v05":
        script = V05Script(pattern)
        pdict = dict(script.params); pdict.update(params or {})
    elif control == "zefeng":
        from .controls import ZScript
        with open(best or f"runs/opt/{pattern}/best.json") as f:
            script = ZScript(pattern, params=json.load(f)["params"])
        pdict = dict(script.params); pdict.update(params or {})
    else:
        script = SCRIPTS[pattern](**(script_kw or {}))
        pdict = dict(DEFAULT_PARAMS); pdict.update(params or {})
    P = Params(**pdict)
    num = Numerics(N=N)
    sol = Solver(P, num)
    T = script.T if T_max is None else min(T_max, script.T)
    out = out or f"runs/{pattern}_{control}_N{N}"
    os.makedirs(out, exist_ok=True)
    snaps, traj, events = [], [], []
    next_snap = 0.0
    t0 = time.time()
    last_phase = None
    while sol.t < T - 1e-9:
        # sample the control at the middle of the frame the solver is about to take
        probe, _ = script.sample(sol.t)
        fdt = num.frame_dt
        if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
            fdt = min(fdt, max(num.scan_safety * min(sol.g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
        inl, phase = script.sample(sol.t + 0.5 * fdt)
        if phase != last_phase:
            events.append(dict(t=round(sol.t, 4), phase=phase))
            last_phase = phase
        chi, nsub = sol.advance_frame(inl, frame_dt=fdt)
        traj.append(dict(t=sol.t, x=inl.x_hit[0], y=inl.x_hit[1], active=bool(inl.active and inl.S_eff > 0),
                         phase=phase, S=float(inl.S_eff if inl.active else 0.0), chi=float(chi)))
        if sol.t + 1e-9 >= next_snap:
            snaps.append((sol.t, sol.concentration().astype(np.float32)))
            next_snap += snap_dt
            if not quiet and len(snaps) % 12 == 1:
                L = sol.ledger()
                print(f"[{pattern}] t={sol.t:6.2f}/{T:.2f} phase={phase:<34s} chi={chi:.2f} nsub={nsub:2d} "
                      f"lmax={sol.l[sol.g.mask].max():.2f} lmin={sol.l[sol.g.mask].min():.2f} "
                      f"dep={L['deposited']:.3f} lerr={L['layer_error']:.1e} el={time.time()-t0:.0f}s", flush=True)
    elapsed = time.time() - t0
    c = sol.concentration()
    np.savez_compressed(os.path.join(out, "final_state.npz"), l=sol.l, qx=sol.qx, qy=sol.qy, m=sol.m, mask=sol.g.mask)
    np.savez_compressed(os.path.join(out, "snapshots_c.npz"), t=np.array([s[0] for s in snaps]),
                        c=np.stack([s[1] for s in snaps]).astype(np.float16))
    render.save_final(os.path.join(out, "final.png"), c, sol.g.mask, title=f"{pattern}  t={sol.t:.2f}s")
    L = sol.ledger(); ke, pe = sol.energy()
    metrics = dict(pattern=pattern, control=control, N=N, T=T, elapsed_s=elapsed, params=pdict, numerics=vars(num),
                   stats=sol.stats, ledger=L, kinetic_energy=ke, pressure_energy=pe,
                   white_area_c_gt_0p4=float(np.nansum(c > 0.4) * sol.g.area), events=events, script_kw=script_kw or {})
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, ensure_ascii=False)
    if video:
        render.make_video(os.path.join(out, "movie.mp4"), snaps, traj, sol.g.mask, pattern,
                          fps=int(round(1 / snap_dt)), hold_frames=int(round(1 / snap_dt)), cn=CN.get(pattern, ""))
    with open(os.path.join(out, "trajectory.json"), "w") as f:
        json.dump(traj, f)
    return sol, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="heart")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--param", action="append", default=[], help="key=value overrides")
    ap.add_argument("--control", default="guess", choices=["guess", "v05", "zefeng"])
    ap.add_argument("--best", default=None, help="best.json from latte_imex.optimize (control=zefeng)")
    a = ap.parse_args()
    overrides = {}
    for kv in a.param:
        k, v = kv.split("=")
        overrides[k] = v if k == "return_law" else float(v)
    run(a.pattern, N=a.N, out=a.out, params=overrides, video=not a.no_video, T_max=a.T, control=a.control, best=a.best)
