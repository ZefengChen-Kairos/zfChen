"""Self-contained S1 forward-pitcher pipeline data: recorded ideal-hand poses, the design programs, the solver program
fed by forward-pitcher arrivals, and the hand-error perturbations in pose space.

Data (latte_imex/data/s1_forward/, written once by `export` from the review scratch; small):
  poses_<pattern>.npz      t (design clock, preroll < 0), tip (n, 3) [m], yaw, tilt, roll [rad], fill_mL, alpha_deg, dt
                           (v05_forward.regen: the S1 ideal-hand loop's poses, preroll included)
  program_<pattern>.json   the design program of v05_forward.forward_program (v05_pourable.pl_program 'pitcher'): config
                           (phases, closure parameters), kQ, stop, stroke classes, n_out (arrival bins), the flowing
                           strokes of the robustness prereg
  export_check.json        FwdProgram here == v05_forward.forward_program (bitwise Inlet records) on the stored forward
                           tables, per pattern

FwdProgram(pattern, R, N) is v05_pourable.pl_program(kind 'pitcher')'s LoopProgram on an arrival record R
(ForwardPitcher.arrivals) without files: Program phases + drvf_record on the 1 ms forward table.

Hand errors in pose space (prereg v05_pourable/data/robust_prereg.json, applied per flowing stroke, path / yaw / roll
untouched): dz -> tip height + dz (profile as v05_pourable.rb_profile, ztr transitions); dt -> the tilt history shifted
by dt (flow timing relative to the path, 0.08 s ramps); f -> a tilt offset per stroke calibrated with the forward
pitcher so that the stroke's emitted volume is f times that of the same pose sequence without the flow error
(calibrate_flow; secant iterations, batched).
"""
from __future__ import annotations
import json
import math
import os

import numpy as np

from .solver import Inlet
from .v05_program import Program

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "s1_forward")
PREREG = os.path.join(HERE, "v05_pourable", "data", "robust_prereg.json")
PATTERNS = ("heart", "push_heart", "layered_heart", "tulip", "leaf", "swan")
CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="郁金香", leaf="树叶", swan="天鹅")
D = 0.08
D3 = D ** 3
TABLE_DT = 1e-3
RB_ZHW = 0.11            # [m/s] height-transition speed (v05_pourable.RB_ZHW)
RB_THW = 0.08            # [s] timing-ramp half-width (v05_pourable.RB_THW)
NAMES = ("nominal", "P1", "P2", "P3", "P4", "P5", "P6", "R1", "R2", "R3")
LAB = dict(nominal="nominal S1", P1="P1 z +3 mm", P2="P2 z -3 mm", P3="P3 Q x0.8", P4="P4 Q x1.2",
           P5="P5 flow 50 ms late", P6="P6 flow 50 ms early", R1="R1 random", R2="R2 random", R3="R3 random")


# ----------------------------------------------------------------------------- data
def load_poses(pattern, data=DATA):
    z = np.load(os.path.join(data, f"poses_{pattern}.npz"))
    return {k: z[k] for k in z.files}


def load_meta(pattern, data=DATA):
    with open(os.path.join(data, f"program_{pattern}.json")) as f:
        return json.load(f)


def geometry(alpha_deg):
    """The NX cavity at nr = 6 with lip slope alpha (== v05_pourable.loop_geometry('nx', alpha))."""
    from .pitcher import GridPitcherGeometry
    return GridPitcherGeometry.load(nr=6, lip_slope=math.radians(alpha_deg))


def export(patterns=PATTERNS, data=DATA):
    """Copy the poses and the design programs out of the review scratch (needs it; run once on the dev machine) and
    check FwdProgram against v05_forward.forward_program on the stored forward tables."""
    from . import v05_forward as VF
    from . import v05_pitcher_check as PC
    os.makedirs(data, exist_ok=True)
    with open(PREREG) as f:
        flowing = json.load(f)["flowing_strokes"]
    checks = {}
    for p in patterns:
        P = VF.load_poses(p)
        np.savez_compressed(os.path.join(data, f"poses_{p}.npz"), t=P["t"], tip=P["tip"], yaw=P["yaw"], tilt=P["tilt"],
                            roll=P["roll"], fill_mL=P["fill_mL"], alpha_deg=P["alpha_deg"], dt=P["dt"])
        prog = VF.forward_program(p)
        z = np.load(os.path.join(VF.FWD, VF.fwd_name(p) + ".npz"))
        meta = dict(pattern=p, config=prog.config, kQ=prog.kQ, stop=bool(prog.stop), cls=list(prog.cls),
                    events=list(prog.events), duration=prog.duration, n_out=int(len(z["a_t"])), dt=float(z["dt"]),
                    fill_mL=float(P["fill_mL"]), alpha_deg=float(P["alpha_deg"]), flowing=list(flowing[p]),
                    source="v05_forward.forward_program (v05_pourable.pl_program 'pitcher' on the S1 forward table)")
        with open(os.path.join(data, f"program_{p}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        # check: same Inlet records as the review pipeline on the stored forward arrivals
        R = {k[2:]: z[k] for k in z.files if k.startswith("a_")}
        mine = FwdProgram(p, R, 256, data=data)
        ts = np.concatenate([np.arange(0.0, prog.duration + 0.3, 1.0 / 240.0), np.random.default_rng(1).uniform(0, prog.duration, 400)])
        bad = 0
        for t in ts:
            if not PC._inlet_same(mine(float(t)), prog(float(t))):
                bad += 1
        tab_old = np.load(os.path.join(VF.FWD, "tables", VF.fwd_name(p) + ".npz"))
        tab_new = table_from_arrivals(R, meta["dt"])
        same_tab = all(np.array_equal(tab_old["p_" + k], tab_new[k]) for k in tab_new)
        checks[p] = dict(n=int(len(ts)), mismatch=bad, table_identical=bool(same_tab))
        print(p, checks[p], flush=True)
    with open(os.path.join(data, "export_check.json"), "w") as f:
        json.dump(checks, f, indent=1)
    return checks


# ----------------------------------------------------------------------------- arrivals -> program
def table_from_arrivals(R, dt=TABLE_DT):
    """v05_forward.write_table's 'p_' columns (D units) from an arrival record."""
    from .v05_pitcher_check import _foot_speed
    on = R["on"] & (R["Q"] > 0)
    T = dict(on=on.astype(np.int8), x=np.where(on, R["x_hit"][:, 0], 0.0) / D, y=np.where(on, R["x_hit"][:, 1], 0.0) / D,
             Q=np.where(on, R["Q"], 0.0) / D3, U=np.where(on, R["U_perp"], 0.0) / D,
             ux=np.where(on, R["u_h"][:, 0], 0.0) / D, uy=np.where(on, R["u_h"][:, 1], 0.0) / D)
    T["vfoot"] = _foot_speed(on, T["x"], T["y"], dt)
    return T


class FwdProgram(Program):
    """v05_pourable.pl_program(kind 'pitcher') on an in-memory forward arrival record (see module docstring)."""

    def __init__(self, pattern, R, N=256, meta=None, data=DATA):
        meta = meta or load_meta(pattern, data)
        super().__init__(dict(meta["config"], continuous_actions=False))   # the stored phases are final (bridges included)
        assert np.allclose(self.events, meta["events"], rtol=0, atol=1e-12)
        self.meta = meta
        self.kQ, self.stop, self.cls = meta["kQ"], meta["stop"], list(meta["cls"])
        self.tab = table_from_arrivals(R, meta["dt"])
        self.dt = meta["dt"]
        self.n = len(self.tab["on"])
        self.vcap = 0.5 / (N * TABLE_DT)
        self.slog = []

    def index(self, t):
        return min(max(int(math.floor(t / self.dt + 1e-9)), 0), self.n - 1)

    def __call__(self, t):
        from .v05_pitcher_check import drvf_record
        k = self.phase_index(t)
        inl0 = Program.__call__(self, t)
        c = self.tab
        i = self.index(t)
        args = (bool(c["on"][i]), (float(c["x"][i]), float(c["y"][i])), float(c["Q"][i]), float(c["U"][i]),
                float(c["ux"][i]), float(c["uy"][i]), max(inl0.scan_speed, min(float(c["vfoot"][i]), self.vcap)))
        o, Q = drvf_record(self, k, inl0, *args)
        self.slog.append((t, k, float(args[0]), args[2] * 512.0, (Q or 0.0) * 512.0))
        return o


class ProgScript:
    """run.V05Script on a program object: (Inlet, label) sampler, Params from the program's parameters."""

    def __init__(self, prog, pattern):
        self.prog, self.name = prog, pattern
        self.T = prog.duration
        self.params = dict(prog.config["parameters"])
        self.params["return_law"] = "v05" if prog.config.get("return_law", {}).get("mode") == "return_speed" else "v1"

    def sample(self, t):
        if t >= self.T:
            return Inlet(active=False), "settle"
        return self.prog(t), self.prog.phase_label(t)


# ----------------------------------------------------------------------------- hand errors in pose space
def prereg_pert(name, pattern):
    """{phase index (0-based): (dz_mm, f, dt_ms)} of a named perturbation (v05_pourable.rb_pert, from the repo copy of
    the prereg)."""
    with open(PREREG) as fh:
        pre = json.load(fh)
    ns = pre["flowing_strokes"][pattern]
    if name == "nominal":
        return {n - 1: (0.0, 1.0, 0.0) for n in ns}
    P = pre["perturbations"]
    if name in P and name != "R":
        q = P[name]
        return {n - 1: (q["dz_mm"], q["f"], q["dt_ms"]) for n in ns}
    d = P["R"]["draws"][name][pattern]
    return {n - 1: (d[str(n)]["dz_mm"], d[str(n)]["f"], d[str(n)]["dt_ms"]) for n in ns}


def profile(events, vals, hw_fn, tg):
    """v05_pourable.rb_profile: per-stroke values {k: v} -> series on tg (v_k inside stroke k, linear across a gap
    between two flowing strokes, held outside, ramp of half-width hw_fn(|dv|) at a boundary of adjacent strokes)."""
    fl = sorted(vals)
    nt, nv = [], []
    for i, k in enumerate(fl):
        t0, t1 = events[k], events[k + 1]
        a = t0
        if i > 0 and fl[i - 1] == k - 1:
            a = t0 + hw_fn(abs(vals[k] - vals[k - 1]))
        b = t1
        if i + 1 < len(fl) and fl[i + 1] == k + 1:
            b = t1 - hw_fn(abs(vals[k + 1] - vals[k]))
        nt += [a, b]
        nv += [vals[k], vals[k]]
    return np.interp(tg, nt, nv)


def tilt_offset(events, offs, tg, tail=0.1):
    """Per-stroke tilt offsets as a series: profile() semantics, tapered to 0 over `tail` s after the last flowing
    stroke (the stop after the pour is not tilted)."""
    if not offs or all(v == 0.0 for v in offs.values()):
        return np.zeros_like(tg)
    y = profile(events, offs, lambda dv: RB_THW, tg)
    tend = events[max(offs) + 1]
    return y * np.clip(1.0 - (tg - tend) / tail, 0.0, 1.0)


def perturb_poses(P, meta, pert, offs=None):
    """Pose sequence with the hand errors pert ({k: (dz_mm, f, dt_ms)}): tip z + dz(t), tilt(t - delta(t)) + offs(t).
    Returns a dict in the ForwardPitcher.run input format (emission times t - dt/2) + 'dz', 'delta', 'dtilt'."""
    ev = meta["events"]
    t = np.asarray(P["t"], float)
    dz = profile(ev, {k: v[0] * 1e-3 for k, v in pert.items()}, lambda dv: 0.5 * max(dv / RB_ZHW, 0.05), t)
    delta = profile(ev, {k: v[2] * 1e-3 for k, v in pert.items()}, lambda dv: RB_THW, t)
    tilt = np.interp(t - delta, t, P["tilt"]) if np.any(delta != 0) else np.asarray(P["tilt"], float).copy()
    dth = tilt_offset(ev, offs or {}, t)
    tip = np.asarray(P["tip"], float).copy()
    tip[:, 2] += dz
    dt = float(P["dt"])
    return dict(t=t - 0.5 * dt, tip=tip, yaw=np.asarray(P["yaw"], float), tilt=tilt + dth, roll=np.asarray(P["roll"], float),
                V0=float(P["fill_mL"]) * 1e-6, dt=dt, dz=dz, delta=delta, dtilt=dth)


def stroke_volumes(E, meta):
    """Emitted volume [mL] per flowing stroke (emission time windows; the first stroke includes the preroll, the last
    one the tail)."""
    fl = sorted(k - 1 for k in meta["flowing"])
    ev = meta["events"]
    t = E["t"]
    out = {}
    for i, k in enumerate(fl):
        a = -np.inf if i == 0 else ev[k]
        b = np.inf if i == len(fl) - 1 else ev[k + 1]
        m = (t >= a) & (t < b)
        out[k] = float(np.sum(E["Q_o"][m]) * E["dt"] * 1e6)
    return out


_GPU_FP = {}


def run_pitcher(seqs, fp, backend="gpu"):
    """ForwardPitcher.run on a list of pose sequences: batched on the GPU or one by one on the CPU.  The GPU object
    (geometry and law on the device) is kept per ForwardPitcher, so repeated calls upload only the poses."""
    if backend == "gpu":
        from .cuda_pitcher import GPUForwardPitcher
        g = _GPU_FP.get(id(fp))
        if g is None or g.fp is not fp:
            g = _GPU_FP[id(fp)] = GPUForwardPitcher(fp)
        return g.run_batch(seqs)
    return [fp.run(s["t"], s["tip"], s["yaw"], s["tilt"], s["roll"], s["V0"], s["dt"]) for s in seqs]


MAX_OFFSET = 6.0         # [deg] largest tilt offset of the flow-factor calibration


def _next_offset(hist, tg, step):
    """Next tilt offset for one stroke from its (offset, volume) history: regula falsi inside a bracket of the target
    (the volume grows with the offset), secant / growing steps outside it (max 8 deg per step, |offset| <= MAX_OFFSET)."""
    lo = [(x, v) for x, v in hist if v < tg]
    hi = [(x, v) for x, v in hist if v > tg]
    if lo and hi:
        xl, vl = max(lo); xh, vh = min(hi)
        if xh > xl:
            x = xl + (tg - vl) * (xh - xl) / max(vh - vl, 1e-12)
            return min(max(x, xl + 0.05 * (xh - xl)), xh - 0.05 * (xh - xl))
    x1, v1 = hist[-1]
    sgn = 1.0 if v1 < tg else -1.0
    if len(hist) >= 2:
        x0, v0 = hist[-2]
        slope = (v1 - v0) / (x1 - x0) if x1 != x0 else 0.0
        if slope > 1e-9:
            x = x1 + (tg - v1) / slope
            x = float(np.clip(x, x1 - math.radians(8.0), x1 + math.radians(8.0)))
            return float(np.clip(x, -math.radians(MAX_OFFSET), math.radians(MAX_OFFSET)))
        x = x1 + sgn * 2.0 * max(abs(x1 - x0), step)
    else:
        x = x1 + step * (tg / max(v1, 1e-6) - 1.0) / 0.2
    return float(np.clip(x, -math.radians(MAX_OFFSET), math.radians(MAX_OFFSET)))


def calibrate_flow(jobs, fp, backend="gpu", tol=0.003, maxit=8, step_deg=3.0, sweeps=2, log=print):
    """jobs: list of (P, meta, pert).  For every job with a flow error f != 1, find per-stroke tilt offsets so that the
    emitted stroke volumes are f times those of the same job without the flow error (dz, dt kept).  The strokes are
    calibrated in time order (a stroke's volume depends on the earlier strokes through the pitcher fill; the offset
    ramps at a boundary of two adjacent strokes couple them back weakly, hence `sweeps` passes), all jobs batched in
    every pitcher call; a stroke that cannot reach its target (e.g. a short cut stroke whose ramps alone
    pour more) keeps the closest offset and the job is reported as not converged.
    Returns per job dict(seq, E, offs, V, target, ratio, converged, iters)."""
    n = len(jobs)
    base = [perturb_poses(P, meta, {k: (v[0], 1.0, v[2]) for k, v in pert.items()}) for P, meta, pert in jobs]
    E0 = run_pitcher(base, fp, backend)
    V0 = [stroke_volumes(E, jobs[j][1]) for j, E in enumerate(E0)]
    target = [{k: V0[j][k] * jobs[j][2][k][1] for k in V0[j]} for j in range(n)]
    res = [dict(seq=base[j], E=E0[j], offs={k: 0.0 for k in V0[j]}, V=V0[j], target=target[j], iters=0) for j in range(n)]
    todo = [j for j in range(n) if any(abs(v[1] - 1.0) > 1e-12 for v in jobs[j][2].values())]
    offs = {j: {k: 0.0 for k in V0[j]} for j in todo}
    calls = 1
    step = math.radians(step_deg)

    def run(js):
        seqs = [perturb_poses(*jobs[j], offs=offs[j]) for j in js]
        Es = run_pitcher(seqs, fp, backend)
        for j, sq, E in zip(js, seqs, Es):
            res[j].update(seq=sq, E=E, offs=dict(offs[j]), V=stroke_volumes(E, jobs[j][1]), iters=res[j]["iters"] + 1)

    nstroke = max((len(V0[j]) for j in todo), default=0)
    for si in [i for _ in range(sweeps) for i in range(nstroke)]:
        cur = {j: sorted(V0[j])[si] for j in todo if si < len(V0[j])}
        hist = {j: [(offs[j][k], res[j]["V"][k])] for j, k in cur.items()}
        err = lambda j: abs(res[j]["V"][cur[j]] / max(target[j][cur[j]], 1e-9) - 1.0)
        act = [j for j in cur if err(j) > tol]
        for it in range(maxit):
            if not act:
                break
            for j in act:
                offs[j][cur[j]] = _next_offset(hist[j], target[j][cur[j]], step)
            run(act)
            calls += 1
            for j in act:
                hist[j].append((offs[j][cur[j]], res[j]["V"][cur[j]]))
            act = [j for j in act if err(j) > tol]
        # keep the closest offset of every stroke that did not converge
        redo = []
        for j in act:
            k = cur[j]
            xb, vb = min(hist[j], key=lambda xv: abs(xv[1] / max(target[j][k], 1e-9) - 1.0))
            if xb != offs[j][k]:
                offs[j][k] = xb
                redo.append(j)
        if redo:
            run(redo)
            calls += 1
        log(f"  flow calibration, stroke {si + 1}: {len(cur) - len(act)} of {len(cur)} sequences within {tol:.1%}")
    for j in range(n):
        r = res[j]
        r["ratio"] = {k: r["V"][k] / max(V0[j][k], 1e-9) for k in r["V"]}
        r["converged"] = all(abs(r["V"][k] / max(target[j][k], 1e-9) - 1.0) <= 2 * tol for k in r["V"])
    if todo:
        log(f"  flow calibration: {calls} batched pitcher calls, {sum(res[j]['converged'] for j in todo)} of {len(todo)} "
        f"sequences with every stroke within {2 * tol:.1%}")
    return res
