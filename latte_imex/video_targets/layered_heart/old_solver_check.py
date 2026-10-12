"""Sanity check of the layered_heart stage script: stages_to_moves -> pitcher.BaristaScript -> OLD solver (the
action_opt.run_one setup: physics #9, closure 'old'), N = 64.  NOT a model test: it only checks that the moves run,
that the stream lands where the video shows it, that the wiggle has the measured frequency/amplitude and that the
durations line up with the clip.

    cd /home/user/zfChen && NUMBA_NUM_THREADS=1 python latte_imex/video_targets/layered_heart/old_solver_check.py [--N 64] [--realtime]

(NUMBA_NUM_THREADS=1: with several solver jobs on the 4-core box the parallel numba kernels oversubscribe and one N=64
run takes > 30 min instead of ~20 s.)

Default: the clip-time script stage_scripts/layered_heart.json (the 5.20-6.53 s video freeze is a zero-flow hold).
--realtime: video_targets/layered_heart/stages_realtime.json (freeze cut out; sim time is then mapped back to clip
time for the comparison: t_clip = t_sim + 1.333 for t_sim > 5.2).
Writes runs/stage_check/layered_heart[_rt]/{snapshots.npz, inlet.npz, summary.json, compare.json} and, next to this
file, old_solver_check[_rt].png (video | w | simulated field at matching clip times, all in the T frame, the solver
field row-flipped with video_extract.sim_to_t) and old_solver_check[_rt].mp4 (rectified video | w | sim at 5 fps).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import action_opt as AO                                   # noqa: E402
from latte_imex import video_extract as VX                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_CLIP = os.path.join(HERE, "..", "..", "stage_scripts", "layered_heart.json")
SCRIPT_RT = os.path.join(HERE, "stages_realtime.json")
FREEZE = (5.2, 6.533)                    # clip-time freeze removed in the realtime script
SHIFT = FREEZE[1] - FREEZE[0]


def clip_of_sim(t_sim, t_clip0, realtime):
    t = np.asarray(t_sim, float) + t_clip0
    return np.where(t > FREEZE[0], t + SHIFT, t) if realtime else t


def sim_of_clip(t_clip, t_clip0, realtime):
    t = np.asarray(t_clip, float)
    if realtime:
        t = np.where(t >= FREEZE[1], t - SHIFT, np.minimum(t, FREEZE[0]))
    return t - t_clip0


def run(N=64, realtime=False, fps_snap=5.0):
    from latte_imex.pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript
    from latte_imex.solver import Solver, Params, Numerics
    from latte_imex.optimize import visible
    script_path = SCRIPT_RT if realtime else SCRIPT_CLIP
    out = "runs/stage_check/layered_heart" + ("_rt" if realtime else "")
    os.makedirs(out, exist_ok=True)
    moves, t_clip0 = VX.stages_to_moves(script_path)
    geom = GridPitcherGeometry.load(nr=6)
    ph, kQ = AO.physics9()
    cp = Coupling(c_S=kQ / AO.D_L ** 3, c_U=1.0 / AO.D_L, c_u=1.0 / AO.D_L, footprint="physical")
    script = BaristaScript(moves=moves, pitcher=Pitcher(geom=geom, V0=250e-6), coupling=cp)
    pd = dict(ph, D=1e-7, kappa_Q=125.0, return_law="v05"); pd.update(AO.MODELS["old"])
    num = Numerics(N=N)
    sol = Solver(Params(**pd), num)
    t_snap = list(np.arange(0.0, script.T + 1e-9, 1.0 / fps_snap))
    snaps, snap_t, track = [], [], []
    t0 = time.time()
    while sol.t < script.T - 1e-9:
        probe, _ = script.sample(sol.t)
        fdt = num.frame_dt
        if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
            fdt = min(fdt, max(num.scan_safety * min(sol.g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
        inl, lab = script.sample(sol.t + 0.5 * fdt)
        track.append((sol.t, float(inl.active), *(inl.x_hit if inl.active else (np.nan, np.nan)),
                      float(inl.S_eff) if inl.active else 0.0))
        sol.advance_frame(inl, frame_dt=fdt)
        while t_snap and sol.t >= t_snap[0] - 1e-9:
            snaps.append(visible(sol.concentration(), sol.g.mask).astype(np.float32))
            snap_t.append(t_snap.pop(0))
    c = sol.concentration()
    mask = sol.g.mask
    ok = bool(np.all(np.isfinite(c[mask])))
    tr = np.array(track, float)
    rec = script.records
    Q = np.array([r["Q"] for r in rec]); tq = np.array([r["t"] for r in rec])
    z_tip = np.array([r["tip"][2] for r in rec])
    st = np.array(snap_t)
    np.savez_compressed(os.path.join(out, "snapshots.npz"), t_sim=st, t_clip=clip_of_sim(st, t_clip0, realtime),
                        s_M=np.array(snaps), mask=mask, N=N)
    np.savez_compressed(os.path.join(out, "inlet.npz"), t=tr[:, 0], active=tr[:, 1], x_M=tr[:, 2], y_M=tr[:, 3],
                        S_eff=tr[:, 4], t_pitcher=tq, Q=Q, z_tip=z_tip)
    summary = dict(script=os.path.relpath(script_path, "/home/user/zfChen"), realtime=realtime, N=N, ok=ok,
                   elapsed_s=time.time() - t0, T_sim=script.T, t_clip0=t_clip0, n_moves=len(moves),
                   moves=[dict(name=m.name, dur=round(m.dur, 3), p0=m.p0, p1=m.p1, z0=m.z0, z1=m.z1, Q0_mLs=m.Q0 * 1e6,
                               Q1_mLs=m.Q1 * 1e6, wobble=m.wobble, wobble1=m.wobble1, wfreq=m.wfreq) for m in moves],
                   V_end_mL=float(script.pitcher.state.V) * 1e6, deposited=float(sol.ledger()["deposited"]),
                   poured_mL=float(getattr(np, "trapezoid", getattr(np, "trapz", None))(Q, tq) * 1e6))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return summary


def _wiggle(t, x, a, b):
    """dominant frequency [Hz] and half-amplitude [D] of x(t) on [a, b] (detrended), from the inlet track."""
    m = (t >= a) & (t <= b) & np.isfinite(x)
    if m.sum() < 20:
        return None
    tt, xx = t[m], x[m]
    tu = np.arange(tt[0], tt[-1], 1 / 60.0)
    xu = np.interp(tu, tt, xx)
    d = xu - np.polyval(np.polyfit(tu, xu, 2), tu)
    n = 4096
    f = np.fft.rfftfreq(n, 1 / 60.0)
    P = np.abs(np.fft.rfft(d * np.hanning(len(d)), n)) ** 2
    k = np.argmax(P[f > 0.8]) + np.argmax(f > 0.8)
    return dict(freq_Hz=round(float(f[k]), 2), half_amp_D=round(float(np.sqrt(2) * d.std()), 4))


def compare(realtime=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    script_path = SCRIPT_RT if realtime else SCRIPT_CLIP
    out = "runs/stage_check/layered_heart" + ("_rt" if realtime else "")
    tag = "_rt" if realtime else ""
    st = json.load(open(script_path))
    t_clip0 = st.get("t_clip0", 0.0)
    sn = np.load(os.path.join(out, "snapshots.npz"))
    il = np.load(os.path.join(out, "inlet.npz"))
    fr = VX.load_frames("layered_heart")
    act = il["active"] > 0
    t_il_clip = clip_of_sim(il["t"], t_clip0, realtime)
    rep = dict(script=os.path.relpath(script_path, "/home/user/zfChen"), realtime=realtime)
    rep["t_first_inlet_clip"] = float(t_il_clip[act][0]) if act.any() else None
    rep["t_last_inlet_clip"] = float(t_il_clip[act][-1]) if act.any() else None
    # inlet-off intervals longer than 0.1 s (clip time)
    off, k = [], 0
    a = act.astype(int)
    edges = np.flatnonzero(np.diff(a))
    for e in edges:
        if a[e] == 1:                       # falls
            k = e + 1
        elif k:
            if t_il_clip[e + 1] - t_il_clip[k] > 0.1:
                off.append([round(float(t_il_clip[k]), 2), round(float(t_il_clip[e + 1]), 2)])
    rep["inlet_off_intervals_clip"] = off
    # pitcher flow vs the stage table at the stage mid-points
    tq, Q = il["t_pitcher"], il["Q"]
    flows = []
    for s in st["stages"]:
        tm = 0.5 * (s["t0"] + s["t1"])
        ts = tm - t_clip0                   # stage times are clip times (realtime script: real times)
        flows.append(dict(stage=f"{s['kind']} {s['t0']}-{s['t1']}", wanted=round(0.5 * (s.get("flow0", 0) + s.get("flow1", s.get("flow0", 0))), 3),
                          sim=round(float(np.interp(ts, tq, Q)) / st["Q_ref"], 3),
                          z_tip_sim_m=round(float(np.interp(ts, tq, il["z_tip"])), 4)))
    rep["flow_vs_stage_mid"] = flows
    # impact vs measured feet (wiggle-averaged over +-0.15 s in the sim)
    errs = {}
    for ts_, xy in st["foot_measurements_M"].items():
        tc = float(ts_)
        if realtime and FREEZE[0] < tc < FREEZE[1]:
            continue
        tsim = float(sim_of_clip(tc, t_clip0, realtime))
        m = act & (np.abs(il["t"] - tsim) <= 0.15)
        if not m.any():
            errs[ts_] = None
            continue
        sim = np.array([np.nanmean(il["x_M"][m]), np.nanmean(il["y_M"][m])])
        errs[ts_] = dict(sim=sim.round(3).tolist(), measured=xy, err_D=round(float(np.hypot(*(sim - np.array(xy)))), 3))
    rep["impact_vs_video"] = errs
    e = np.array([v["err_D"] for v in errs.values() if v])
    rep["impact_err_median_D"] = float(np.median(e)) if len(e) else None
    rep["impact_err_p90_D"] = float(np.percentile(e, 90)) if len(e) else None
    rep["impact_err_max_D"] = float(np.max(e)) if len(e) else None
    # wiggle frequency / amplitude of the simulated impact (measured: 3.3 Hz, ~0.035 D; 3.7 Hz after the freeze)
    tcl = t_il_clip
    x = np.where(act, il["x_M"], np.nan)
    rep["wiggle_sim"] = {"2.1-5.1s": _wiggle(tcl, x, 2.1, 5.1), "6.7-7.3s": _wiggle(tcl, x, 6.7, 7.3)}
    # final frame
    sT = VX.sim_to_t(sn["s_M"][-1])
    N = sT.shape[0]
    from scipy.ndimage import zoom
    wt = zoom(fr["white"][-1].astype(np.float32), N / fr["white"].shape[1], order=1)
    g = (np.arange(N) + 0.5) / N - 0.5
    X, Y = np.meshgrid(g, g); dm = np.hypot(X, Y) < 0.49

    def iou(a, b):
        a, b = (a > 0.5) & dm, (b > 0.5) & dm
        return float((a & b).sum() / max((a | b).sum(), 1))
    j_end = int(np.argmin(np.abs(sn["t_clip"] - float(fr["t"][-1]))))
    sT = VX.sim_to_t(sn["s_M"][j_end])
    rep["final_t_clip_sim"] = float(sn["t_clip"][j_end])
    rep["final_iou_T_rowflipped"] = iou(sT, wt)
    rep["final_iou_T_noflip"] = iou(sn["s_M"][j_end], wt)
    rep["final_white_frac_sim"] = float(((sT > 0.5) & dm).sum() / dm.sum())
    rep["final_white_frac_video"] = float(((wt > 0.5) & dm).sum() / dm.sum())
    json.dump(rep, open(os.path.join(out, "compare.json"), "w"), indent=1)

    frames, _ = VX.read_video(str(fr["clip"]))
    rot, flip = float(fr["rot"]), bool(fr["flip"])
    # hand-fitted ellipses for the tilt phase (quality 0/1, t < 4.4 s), as in the stage script
    HAND = {0.0: (220, 318), 1.2: (220, 318), 1.8: (218, 320), 2.4: (215, 322), 3.0: (213, 325), 3.6: (212, 327),
            4.0: (212, 328), 4.4: (211.2, 329.0)}

    def ell(tc):
        i = min(int(round(tc * 15)), len(fr["track_ell"]) - 1)
        if tc < 4.4:
            kk = sorted(HAND)
            return (float(np.interp(tc, kk, [HAND[q][0] for q in kk])), float(np.interp(tc, kk, [HAND[q][1] for q in kk])),
                    112.5, 86.0, -3.0)
        return tuple(fr["track_ell"][i])

    def panel_video(tc):
        i = min(int(round(tc * 15)), len(frames) - 1)
        return np.clip(VX.rectify(frames[i].astype(np.float64) / 255.0, ell(tc), 192, rot, flip), 0, 1)

    def marks(axs, tc):
        ts = float(sim_of_clip(tc, t_clip0, realtime))
        kk = np.searchsorted(il["t"], ts)
        pm, _ = VX.stage_point(script_path if not realtime else SCRIPT_CLIP, tc)
        for a_ in axs:
            a_.axis("off")
            if kk < len(il["t"]) and il["active"][kk]:
                a_.plot(il["x_M"][kk], -il["y_M"][kk], "r+", ms=8, mew=1.5)
            if pm is not None:
                a_.plot(pm[0], -pm[1], "yx", ms=5)

    times = [1.2, 1.5, 2.4, 3.8, 5.2, 6.6, 7.4, 8.0, 9.4, 9.8, 10.07, 10.7, 11.8]
    fig, ax = plt.subplots(3, len(times), figsize=(1.9 * len(times), 6.2), dpi=90)
    for k, tc in enumerate(times):
        ax[0, k].imshow(panel_video(tc), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5])
        ax[0, k].set_title(f"video {tc:.2f}s", fontsize=7)
        j = int(np.argmin(np.abs(fr["t"] - tc)))
        w = np.where(fr["valid"][j], fr["white"][j].astype(np.float32), 0.4)
        ax[1, k].imshow(w, origin="lower", cmap="gray", vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
        ax[1, k].set_title(f"w {fr['t'][j]:.1f}s", fontsize=7)
        m = int(np.argmin(np.abs(sn["t_clip"] - tc)))
        ax[2, k].imshow(VX.sim_to_t(sn["s_M"][m]), origin="lower", cmap="gray", vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
        ax[2, k].set_title(f"old solver {sn['t_clip'][m]:.1f}s", fontsize=7)
        marks(ax[:, k], tc)
        sel = act & (t_il_clip <= tc)
        ax[2, k].plot(il["x_M"][sel], -il["y_M"][sel], "r-", lw=0.4)
        for ts_, xy in st["foot_measurements_M"].items():
            if float(ts_) <= tc + 1e-6:
                ax[0, k].plot(xy[0], -xy[1], "c.", ms=2)
    fig.suptitle("layered_heart stage script%s through the OLD solver (N=%d), T frame (+y up: pours start at -y, cut exits at "
                 "+y). red + = sim inlet, yellow x = stage intent, cyan = measured feet; t<4.4 s rectified with hand-fit "
                 "ellipses" % (" (realtime)" if realtime else "", sn["N"]), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"old_solver_check{tag}.png"))
    plt.close(fig)
    # movie: every snapshot (5 fps of sim time), rectified video | w | sim
    tmp = tempfile.mkdtemp(prefix="lhc_")
    try:
        n = 0
        for m, tc in enumerate(sn["t_clip"]):
            if tc > fr["t"][-1] + 1e-6:
                break
            f2, a2 = plt.subplots(1, 3, figsize=(9.0, 3.4), dpi=100)
            a2[0].imshow(panel_video(tc), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5])
            a2[0].set_title(f"video clip t={tc:.1f}s (T frame)", fontsize=8)
            j = int(np.argmin(np.abs(fr["t"] - tc)))
            a2[1].imshow(np.where(fr["valid"][j], fr["white"][j].astype(np.float32), 0.4), origin="lower", cmap="gray",
                         vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
            a2[1].set_title("extracted w (grey = invalid)", fontsize=8)
            a2[2].imshow(VX.sim_to_t(sn["s_M"][m]), origin="lower", cmap="gray", vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
            frz = (FREEZE[0] < tc < FREEZE[1] + 0.06) or tc < 1.2
            a2[2].set_title(f"old solver t_sim={sn['t_sim'][m]:.1f}s" + ("  [video freeze]" if frz else ""), fontsize=8)
            marks(a2, tc)
            f2.tight_layout(); f2.savefig(os.path.join(tmp, f"f{n:04d}.png")); plt.close(f2); n += 1
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "5", "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        os.path.join(HERE, f"old_solver_check{tag}.mp4")], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    if not a.compare_only:
        print(json.dumps(run(a.N, a.realtime), indent=1), flush=True)
    print(json.dumps(compare(a.realtime), indent=1))
