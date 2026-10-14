"""Forward, pose-driven pitcher (V1 mode B) replayed on the S1 ideal-hand poses.

The S1 pitcher-loop runs (v05_pourable.pl_simulate, IdealHand on the full latte_imex.pitcher.Pitcher, stored under
<review>/v05pour/loop/runs/S1_<pattern>_ideal.npz) keep the hand's poses only after the 0.3 s preroll.  `regen` re-runs
pl_simulate (vtime 'landing', i.e. exactly as the stored runs were made) and records every pose the hand gave the pitcher,
preroll included; the post-preroll part is checked against the stored run.  `replay` feeds those poses open-loop to
pitcher.ForwardPitcher (V1 mode B: front-edge weir with projected widths, h = [z_f - z_lip]_+, no hysteresis, no slosh,
no lip buffer, U = C_v sqrt(2 g H), rigid-body lip velocity incl. omega x r, ballistic parcels to z = 0) and writes
  <FWD>/S1_<pattern>_<variant>.npz          arrival record (SI) + emission history + ledger
  <FWD>/tables/S1_<pattern>_<variant>.npz   the solver table in v05_pourable.pl_table format (p_ = forward arrivals,
                                            w_ = the design inlet of the stored loop table)
so that v05_pourable.pl_program / pl_solve (the pitcher-delivered path) run on it unchanged:
  prog = forward_program("heart")            # a run.run program (pl_program on the forward table)
  run_forward("heart", N=256)                # latte_imex.run.run on it -> <FWD>/runs256/PLS1fwd_heart_N256/
`compare` reports per stroke the forward delivery against the old pitcher model's delivery on the same poses.
Variants (ablations of what changed against the old Pitcher): fwd (V1 defaults), arc (rim arc length instead of the
projected width), hk (old 0.8 mm dead head subtracted), hbar (exit speed from the discharge-weighted head instead of
the max head).

usage (from /home/user/zfChen; NUMBA_NUM_THREADS=1):
  python3 -m latte_imex.v05_forward regen --pattern heart
  python3 -m latte_imex.v05_forward replay --pattern heart [--variant fwd]
  python3 -m latte_imex.v05_forward compare            (all patterns, all variants present) -> <FWD>/compare.json
  python3 -m latte_imex.v05_forward solve --pattern heart --N 256
"""
import argparse
import json
import math
import os
import time

import numpy as np

from . import v05_pitcher_check as PC
from . import v05_pourable as VP
from .pitcher import ForwardLaw, ForwardPitcher

ML = 1e-6
D = PC.D
TAG = "S1"
FWD = os.path.join(os.path.dirname(PC.REVIEW), "align", "forward")
REGEN = os.path.join(FWD, "regen")
VARIANTS = dict(fwd={}, arc=dict(width="arc"), hk=dict(h_k=0.8e-3), hbar=dict(U_head="bar"))


def old_name(pattern):
    return VP.pl_name(TAG, pattern, "ideal")


def fwd_name(pattern, variant="fwd"):
    return f"{TAG}_{pattern}_{variant}"


# ----------------------------------------------------------------------- poses
def regen(pattern, out=REGEN, check=True):
    """Re-run the stored ideal-hand loop simulation (vtime 'landing', as stored) and record the poses of the last pass,
    preroll included.  Saves <out>/poses_<pattern>.npz: t (end of the 1 ms step, design clock: preroll < 0), tip (n, 3),
    yaw, tilt, roll [rad], fill_mL, alpha_deg."""
    logs = []
    orig = VP.SpoutPitcher.step

    def step(self, pose, dt, z_surface=0.0):
        rec = orig(self, pose, dt, z_surface)
        if not logs or logs[-1][0] is not self:
            logs.append((self, []))
        logs[-1][1].append((rec["t"], *pose.tip, pose.yaw, pose.tilt, pose.roll))
        return rec
    vt = VP.VTIME_DEFAULT
    VP.SpoutPitcher.step = step
    VP.VTIME_DEFAULT = "landing"
    try:
        W0, meta0, A, info = VP.pl_simulate(TAG, pattern, "ideal", out=out, verbose=False)
    finally:
        VP.SpoutPitcher.step = orig
        VP.VTIME_DEFAULT = vt
    L = np.array(logs[-1][1])
    t = L[:, 0] - info["preroll_s"]
    res = dict(n=len(t), preroll_steps=int(np.sum(t < 1e-9)))
    if check:
        Wo, mo, Ao, io = PC.load_run(old_name(pattern), VP.PLOUT)
        m = t > 1e-9
        n = min(m.sum(), len(Ao["t"]))
        d = max(float(np.max(np.abs(L[m, 1][:n] - Ao["tip_x"][:n]))), float(np.max(np.abs(L[m, 2][:n] - Ao["tip_y"][:n]))),
                float(np.max(np.abs(L[m, 3][:n] - Ao["tip_z"][:n]))), float(np.max(np.abs(L[m, 5][:n] - Ao["tilt"][:n]))))
        res["max_abs_diff_vs_stored"] = d
        res["Q_hit_max_diff_vs_stored"] = float(np.max(np.abs(np.nan_to_num(A["Q_hit"][:n]) - np.nan_to_num(Ao["Q_hit"][:n]))))
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f"poses_{pattern}.npz"), t=t, tip=L[:, 1:4], yaw=L[:, 4], tilt=L[:, 5],
                        roll=L[:, 6], fill_mL=info["fill_mL"], alpha_deg=info["alpha_deg"], dt=info["dt"],
                        meta=json.dumps(dict(pattern=pattern, tag=TAG, source="v05_forward.regen (pl_simulate ideal, "
                                             "vtime landing), last pass", check=res)))
    print(pattern, res, flush=True)
    return res


def load_poses(pattern, out=REGEN):
    z = np.load(os.path.join(out, f"poses_{pattern}.npz"))
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------- forward replay
def replay(pattern, variant="fwd", law=None, save=True, out=FWD):
    """Feed the recorded poses to ForwardPitcher; arrival record on the stored run's 1 ms grid (bin j = arrivals in
    [j dt, (j+1) dt) on the design clock).  Saves the arrival npz and the solver table."""
    t0w = time.time()
    P = load_poses(pattern)
    dt = float(P["dt"])
    geom = VP.loop_geometry("nx", float(P["alpha_deg"]))
    lw = law or ForwardLaw(**VARIANTS[variant])
    FP = ForwardPitcher(geom, lw)
    te = P["t"] - 0.5 * dt                                # pose i holds over the step [t_i - dt, t_i]
    E = FP.run(te, P["tip"], P["yaw"], P["tilt"], P["roll"], float(P["fill_mL"]) * ML, dt)
    Wo, mo, Ao, io = PC.load_run(old_name(pattern), VP.PLOUT)
    n_out = len(Ao["t"])
    R = ForwardPitcher.arrivals(E, 0.0, n_out, dt)
    led = R["V_P"] + R["V_jet"] + R["V_cup"] + R["V_spill"] - E["V0"]
    pre = R["t"] <= 0
    info = dict(pattern=pattern, variant=variant, law=lw.__dict__, fill_mL=float(P["fill_mL"]),
                alpha_deg=float(P["alpha_deg"]), dt=dt, n_emit=len(te), n_out=n_out, beak_phi_deg=math.degrees(FP.beak_phi),
                front_segments=int(len(FP.seg_a)), ledger_err_max_m3=float(np.max(np.abs(led))),
                V_released_before_0_mL=float(np.sum(E["Q_o"][te < 0]) * dt / ML),
                V_arrived_before_0_mL=float(np.sum(R["Q"][pre]) * dt / ML) if pre.any() else 0.0,
                V_spill_mL=float(R["V_spill"][-1] / ML), wall_s=time.time() - t0w,
                source_poses=os.path.join(REGEN, f"poses_{pattern}.npz"), old_run=os.path.join(VP.PLOUT, "runs", old_name(pattern) + ".npz"))
    if save:
        os.makedirs(os.path.join(out, "tables"), exist_ok=True)
        np.savez_compressed(os.path.join(out, fwd_name(pattern, variant) + ".npz"),
                            **{"a_" + k: v for k, v in R.items()}, **{"e_" + k: v for k, v in E.items() if k not in ("dt", "V0")},
                            dt=dt, V0=E["V0"], meta=json.dumps(info, default=float))
        write_table(pattern, R, variant, out)
    return E, R, info


def write_table(pattern, R, variant="fwd", out=FWD):
    """The solver table (pl_table format: p_ forward arrivals in D units, w_ = the stored loop table's design columns)."""
    old_tab = np.load(VP.pl_table(old_name(pattern), VP.PLOUT))
    tmeta = json.loads(str(old_tab["meta"]))
    dt = float(old_tab["dt"])
    on = R["on"] & (R["Q"] > 0)
    T = dict(p_on=on.astype(np.int8), p_x=np.where(on, R["x_hit"][:, 0], 0.0) / D, p_y=np.where(on, R["x_hit"][:, 1], 0.0) / D,
             p_Q=np.where(on, R["Q"], 0.0) / PC.D3, p_U=np.where(on, R["U_perp"], 0.0) / D,
             p_ux=np.where(on, R["u_h"][:, 0], 0.0) / D, p_uy=np.where(on, R["u_h"][:, 1], 0.0) / D)
    T["p_vfoot"] = PC._foot_speed(on, T["p_x"], T["p_y"], dt)
    for k in old_tab.files:
        if k.startswith("w_"):
            T[k] = old_tab[k]
    tmeta = dict(tmeta, source_run=os.path.join(out, fwd_name(pattern, variant) + ".npz"),
                 source="v05_forward.replay: pitcher.ForwardPitcher on the S1 ideal-hand poses", variant=variant)
    path = os.path.join(out, "tables", fwd_name(pattern, variant) + ".npz")
    np.savez_compressed(path, dt=dt, **T, meta=json.dumps(tmeta))
    return path


def forward_program(pattern, N=256, variant="fwd", out=FWD, kind="pitcher"):
    """A latte_imex.run.run program fed by the forward arrival table (v05_pourable.pl_program: the design program's
    phases, closure and duration; the same inlet rules as the pitcher-delivered path)."""
    return VP.pl_program(fwd_name(pattern, variant), kind, N, out)


def run_forward(pattern, N=256, variant="fwd", out=FWD):
    """latte_imex.run.run (control 'v05') on the forward table -> <out>/runs256/PLS1<variant>_<pattern>_N<N>/."""
    return VP.pl_solve(fwd_name(pattern, variant), "pitcher", N, out)


# ----------------------------------------------------------------------- old vs forward
def _fw(x, w):
    w = np.where(np.isfinite(x), w, 0.0)
    s = w.sum()
    return float(np.sum(np.nan_to_num(x) * w) / s) if s > 0 else None


def _first(t, m):
    return float(t[m][0]) if m.any() else None


def compare_pattern(pattern, variants=("fwd", "arc", "hk", "hbar"), out=FWD):
    Wo, meta, Ao, io = PC.load_run(old_name(pattern), VP.PLOUT)
    dt = io["dt"]
    t = Ao["t"] - 0.5 * dt
    Tt = meta["T"]
    Qo = np.nan_to_num(Ao["Q_hit"])
    on_o = (np.nan_to_num(Ao["h_on"]) > 0) & (Qo > 0)
    Uo, uho = Ao["h_U"], np.hypot(Ao["h_uhx"], Ao["h_uhy"])
    xo, yo = Ao["h_xh"], Ao["h_yh"]
    tw = Wo["t"]
    Qw = np.where(t <= tw[-1] + dt, np.interp(t, tw, np.nan_to_num(Wo["Q"])), 0.0)
    xw, yw = np.interp(t, tw, Wo["x"]) * D, np.interp(t, tw, Wo["y"]) * D
    res = dict(pattern=pattern, T=Tt, fill_mL=io["fill_mL"], variants={})
    for var in variants:
        path = os.path.join(out, fwd_name(pattern, var) + ".npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        finfo = json.loads(str(z["meta"]))
        Qf = z["a_Q"]
        on_f = z["a_on"] & (Qf > 0)
        Uf = np.where(on_f, z["a_U_perp"], np.nan)
        uhf = np.where(on_f, np.hypot(z["a_u_h"][:, 0], z["a_u_h"][:, 1]), np.nan)
        xf, yf = z["a_x_hit"][:, 0], z["a_x_hit"][:, 1]
        strokes, bridges = [], []
        for s in meta["strokes"]:
            m = (t >= s["t0"]) & (t < s["t1"])
            if not m.any():
                continue
            Vw, Vo, Vf = (float(np.sum(np.where(m, q, 0.0)) * dt / ML) for q in (Qw, np.where(on_o, Qo, 0.0), np.where(on_f, Qf, 0.0)))
            if s["cls"] not in VP.FLOWING:
                bridges.append(dict(n=s["n"], cls=s["cls"], V_des_mL=Vw, V_old_mL=Vo, V_fwd_mL=Vf))
                continue
            wo, wf = np.where(m & on_o, Qo, 0.0), np.where(m & on_f, Qf, 0.0)
            both = m & on_o & on_f
            wb = np.where(both, np.minimum(Qo, Qf), 0.0)
            dfo = np.hypot(xf - xo, yf - yo) * 1e3
            dfw = np.hypot(xf - xw, yf - yw) * 1e3
            dow = np.hypot(xo - xw, yo - yw) * 1e3
            o = dict(n=s["n"], cls=s["cls"], t0=s["t0"], t1=s["t1"], V_des_mL=Vw, V_old_mL=Vo, V_fwd_mL=Vf,
                     V_ratio_fwd_old=Vf / max(Vo, 1e-12),
                     Q_old_fw=_fw(Qo / ML, wo), Q_fwd_fw=_fw(Qf / ML, wf),
                     U_old=_fw(Uo, wo), U_fwd=_fw(Uf, wf), uh_old=_fw(uho, wo), uh_fwd=_fw(uhf, wf),
                     foot_fwd_old_rms_mm=(math.sqrt(_fw(dfo ** 2, wb)) if wb.sum() > 0 else None),
                     foot_fwd_old_dx_mm=_fw((xf - xo) * 1e3, wb), foot_fwd_old_dy_mm=_fw((yf - yo) * 1e3, wb),
                     foot_fwd_des_rms_mm=(math.sqrt(_fw(dfw ** 2, wf)) if wf.sum() > 0 else None),
                     foot_old_des_rms_mm=(math.sqrt(_fw(dow ** 2, wo)) if wo.sum() > 0 else None),
                     start_old=_first(t, m & on_o & (Qo >= ML)), start_fwd=_first(t, m & on_f & (Qf >= ML)))
            for a_, b_ in (("U", "U"), ("uh", "uh")):
                if o[a_ + "_old"] and o[b_ + "_fwd"]:
                    o[a_ + "_rel"] = o[b_ + "_fwd"] / o[a_ + "_old"] - 1
            if o["start_old"] is not None and o["start_fwd"] is not None:
                o["start_shift_ms"] = (o["start_fwd"] - o["start_old"]) * 1e3
            strokes.append(o)
        mT = (t >= 0) & (t < Tt)
        res["variants"][var] = dict(
            info=finfo, strokes=strokes, bridges=bridges,
            V_des_mL=float(np.sum(Qw[mT]) * dt / ML), V_old_in_T_mL=float(np.sum(np.where(on_o, Qo, 0.0)[mT]) * dt / ML),
            V_fwd_in_T_mL=float(np.sum(np.where(on_f, Qf, 0.0)[mT]) * dt / ML),
            V_P_end_old_mL=float(Ao["V"][-1] / ML), V_P_end_fwd_mL=float(z["a_V_P"][-1] / ML),
            H_max_mm=float(np.nanmax(z["e_H"]) * 1e3), U_exit_range=[float(np.min(z["e_U"][z["e_Q_o"] > 0])), float(np.max(z["e_U"]))],
            t_f_ms_range=[float(np.nanmin(z["e_t_f"]) * 1e3), float(np.nanmax(z["e_t_f"]) * 1e3)])
    return res


def compare(patterns=PC.PATTERNS, out=FWD):
    allr = {p: compare_pattern(p, out=out) for p in patterns}
    with open(os.path.join(out, "compare.json"), "w") as f:
        json.dump(allr, f, indent=1, default=float)
    return allr


def print_compare(allr, var="fwd"):
    for p, r in allr.items():
        v = r["variants"].get(var)
        if v is None:
            continue
        print(f"== {p} ({var}) V in T: des {v['V_des_mL']:.1f} old {v['V_old_in_T_mL']:.1f} fwd {v['V_fwd_in_T_mL']:.1f} mL;"
              f" V_P end old {v['V_P_end_old_mL']:.1f} fwd {v['V_P_end_fwd_mL']:.1f}; ledger {v['info']['ledger_err_max_m3']:.1e} m3")
        for o in v["strokes"]:
            f_ = lambda x, fmt="{:.2f}": "-" if x is None else fmt.format(x)
            print(f"  #{o['n']} {o['cls']:4s} V {o['V_old_mL']:.2f}->{o['V_fwd_mL']:.2f} ({f_(o['V_ratio_fwd_old'])})"
                  f" Q {f_(o['Q_old_fw'], '{:.1f}')}->{f_(o['Q_fwd_fw'], '{:.1f}')} U {f_(o['U_old'], '{:.3f}')}->{f_(o['U_fwd'], '{:.3f}')}"
                  f" uh {f_(o['uh_old'], '{:.3f}')}->{f_(o['uh_fwd'], '{:.3f}')} foot d {f_(o['foot_fwd_old_rms_mm'])} mm"
                  f" (dx {f_(o['foot_fwd_old_dx_mm'])}, dy {f_(o['foot_fwd_old_dy_mm'])}) start {f_(o.get('start_shift_ms'), '{:+.0f}')} ms")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["regen", "replay", "compare", "solve"])
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--N", type=int, default=256)
    a = ap.parse_args(argv)
    pats = [a.pattern] if a.pattern else list(PC.PATTERNS)
    if a.cmd == "regen":
        for p in pats:
            regen(p)
    elif a.cmd == "replay":
        for p in pats:
            for v in ([a.variant] if a.variant else list(VARIANTS)):
                E, R, info = replay(p, v)
                print(p, v, {k: info[k] for k in ("ledger_err_max_m3", "V_spill_mL", "V_released_before_0_mL", "wall_s")}, flush=True)
    elif a.cmd == "compare":
        r = compare(pats)
        print_compare(r, a.variant or "fwd")
    elif a.cmd == "solve":
        for p in pats:
            print(run_forward(p, a.N, a.variant or "fwd"))


if __name__ == "__main__":
    main()
