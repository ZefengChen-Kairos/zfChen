"""Sanity check of the swan stage script: stages_to_moves -> pitcher.BaristaScript -> OLD solver (the action_opt.run_one
setup: physics #9, closure 'old'), N = 64.  NOT a model test: it only checks that the moves run, that the stream lands
where the video shows it, that the wiggle has the scripted frequency/amplitude and that the durations line up.

    cd /home/user/zfChen && NUMBA_NUM_THREADS=2 python latte_imex/video_targets/swan/old_solver_check.py [--N 64]
        [--realtime] [--yaw default|video] [--compare-only]

Default: the clip-time script stage_scripts/swan.json (the two mid-clip video freezes are zero-flow holds).
--realtime: video_targets/swan/stages_realtime.json (freezes cut out; sim time mapped back to clip time).
--yaw video: Barista yaw = script['barista_yaw_rad'] (+45 deg: spout toward D, pitcher on the -y side as in this clip)
instead of the pipeline default -pi/2 (pitcher handle at +y, see the script's orientation_warning).
Writes runs/stage_check/swan<tag>/{snapshots.npz, inlet.npz, summary.json, compare.json} and, next to this file,
old_solver_check<tag>.png (video | w | simulated field at matching clip times, T frame, the solver field row-flipped
with video_extract.sim_to_t) and old_solver_check<tag>.mp4 (rectified video | w | sim, 5 fps of clip time).
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, "/home/user/zfChen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latte_imex import action_opt as AO                                   # noqa: E402
from latte_imex import video_extract as VX                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_CLIP = os.path.join(HERE, "..", "..", "stage_scripts", "swan.json")
SCRIPT_RT = os.path.join(HERE, "stages_realtime.json")
FRZ = [(3.867, 5.0), (6.133, 7.267)]
SH = 17 / 15.0


def clip_of_sim(t_sim, realtime):
    t = np.asarray(t_sim, float)
    if not realtime:
        return t
    return np.where(t < FRZ[0][0], t, np.where(t < FRZ[1][0] - SH, t + SH, t + 2 * SH))


def sim_of_clip(t_clip, realtime):
    t = np.asarray(t_clip, float)
    if not realtime:
        return t
    return np.where(t <= FRZ[0][0], t, np.where(t < FRZ[0][1], FRZ[0][0], np.where(
        t <= FRZ[1][0], t - SH, np.where(t < FRZ[1][1], FRZ[1][0] - SH, t - 2 * SH))))


def tag_of(realtime, yaw):
    return ("_rt" if realtime else "") + ("_yawvideo" if yaw == "video" else "")


def run(N=64, realtime=False, yaw="default", fps_snap=5.0):
    from latte_imex.pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript
    from latte_imex.solver import Solver, Params, Numerics
    from latte_imex.optimize import visible
    script_path = SCRIPT_RT if realtime else SCRIPT_CLIP
    out = "runs/stage_check/swan" + tag_of(realtime, yaw)
    os.makedirs(out, exist_ok=True)
    st = json.load(open(script_path))
    moves, t_clip0 = VX.stages_to_moves(script_path)
    kw = {}
    if yaw == "video":
        kw["yaw"] = float(st["barista_yaw_rad"])
    geom = GridPitcherGeometry.load(nr=6)
    ph, kQ = AO.physics9()
    cp = Coupling(c_S=kQ / AO.D_L ** 3, c_U=1.0 / AO.D_L, c_u=1.0 / AO.D_L, footprint="physical")
    script = BaristaScript(moves=moves, pitcher=Pitcher(geom=geom, V0=250e-6), coupling=cp, **kw)
    pd = dict(ph, D=1e-7, kappa_Q=125.0, return_law="v05"); pd.update(AO.MODELS["old"])
    num = Numerics(N=N)
    sol = Solver(Params(**pd), num)
    t_snap = list(np.arange(0.0, script.T + 1e-9, 1.0 / fps_snap))
    snaps, snap_t, track = [], [], []
    t0 = time.time()
    err = None
    try:
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
    except Exception as e:  # noqa
        err = f"{type(e).__name__}: {e}"
    c = sol.concentration()
    mask = sol.g.mask
    ok = err is None and bool(np.all(np.isfinite(c[mask])))
    tr = np.array(track, float)
    rec = script.records
    Q = np.array([r["Q"] for r in rec]); tq = np.array([r["t"] for r in rec])
    z_tip = np.array([r["tip"][2] for r in rec])
    lift = np.array([r.get("lift", 0.0) for r in rec])
    stt = np.array(snap_t)
    np.savez_compressed(os.path.join(out, "snapshots.npz"), t_sim=stt, t_clip=clip_of_sim(stt + t_clip0, realtime),
                        s_M=np.array(snaps), mask=mask, N=N)
    np.savez_compressed(os.path.join(out, "inlet.npz"), t=tr[:, 0], active=tr[:, 1], x_M=tr[:, 2], y_M=tr[:, 3],
                        S_eff=tr[:, 4], t_pitcher=tq, Q=Q, z_tip=z_tip, lift=lift)
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    summary = dict(script=os.path.relpath(script_path, "/home/user/zfChen"), realtime=realtime, yaw=yaw,
                   yaw_rad=kw.get("yaw", -math.pi / 2), N=N, ok=ok, error=err, elapsed_s=round(time.time() - t0, 1),
                   T_sim=script.T, t_clip0=t_clip0, n_moves=len(moves),
                   moves=[dict(name=m.name, dur=round(m.dur, 3), p0=m.p0, p1=m.p1, z0=m.z0, z1=m.z1, Q0_mLs=m.Q0 * 1e6,
                               Q1_mLs=m.Q1 * 1e6, wobble=m.wobble, wobble1=m.wobble1, wfreq=m.wfreq,
                               wobble_dir=m.wobble_dir, bezier=m.bezier) for m in moves],
                   V_end_mL=float(script.pitcher.state.V) * 1e6, deposited=float(sol.ledger()["deposited"]),
                   poured_mL=float(trap(Q, tq) * 1e6), max_collision_lift_mm=float(lift.max() * 1e3) if len(lift) else 0.0)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return summary


def _wiggle(t, x, y, a, b, ang):
    """dominant frequency [Hz] and half-amplitude [D] of the impact along the direction ang [deg] on [a, b]."""
    m = (t >= a) & (t <= b) & np.isfinite(x)
    if m.sum() < 20:
        return None
    u = math.cos(math.radians(ang)) * x[m] + math.sin(math.radians(ang)) * y[m]
    tt = t[m]
    tu = np.arange(tt[0], tt[-1], 1 / 60.0)
    uu = np.interp(tu, tt, u)
    d = uu - np.polyval(np.polyfit(tu, uu, 2), tu)
    n = 4096
    f = np.fft.rfftfreq(n, 1 / 60.0)
    P = np.abs(np.fft.rfft(d * np.hanning(len(d)), n)) ** 2
    k = np.argmax(P[f > 0.8]) + np.argmax(f > 0.8)
    return dict(freq_Hz=round(float(f[k]), 2), half_amp_D=round(float(np.sqrt(2) * d.std()), 4))


def compare(realtime=False, yaw="default"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import make_tiles as MT
    script_path = SCRIPT_RT if realtime else SCRIPT_CLIP
    tag = tag_of(realtime, yaw)
    out = "runs/stage_check/swan" + tag
    st = json.load(open(script_path))
    stc = json.load(open(SCRIPT_CLIP))
    sn = np.load(os.path.join(out, "snapshots.npz"))
    il = np.load(os.path.join(out, "inlet.npz"))
    summ = json.load(open(os.path.join(out, "summary.json")))
    fr = VX.load_frames("swan")
    act = il["active"] > 0
    t_il_clip = clip_of_sim(il["t"], realtime)
    rep = dict(script=os.path.relpath(script_path, "/home/user/zfChen"), realtime=realtime, yaw=yaw, ok=summ["ok"],
               error=summ["error"], elapsed_s=summ["elapsed_s"], T_sim=summ["T_sim"], poured_mL=summ["poured_mL"],
               V_end_mL=summ["V_end_mL"], max_collision_lift_mm=summ["max_collision_lift_mm"])
    rep["t_first_inlet_clip"] = float(t_il_clip[act][0]) if act.any() else None
    rep["t_last_inlet_clip"] = float(t_il_clip[act][-1]) if act.any() else None
    off, k = [], None
    a = act.astype(int)
    for e in np.flatnonzero(np.diff(a)):
        if a[e] == 1:
            k = e + 1
        elif k is not None:
            if t_il_clip[e + 1] - t_il_clip[k] > 0.1:
                off.append([round(float(t_il_clip[k]), 2), round(float(t_il_clip[e + 1]), 2)])
            k = None
    rep["inlet_off_intervals_clip"] = off
    tq, Q = il["t_pitcher"], il["Q"]
    flows = []
    for s in st["stages"]:
        ts = 0.5 * (s["t0"] + s["t1"]) - st.get("t_clip0", 0.0)
        flows.append(dict(stage=f"{s['kind']} {s['t0']}-{s['t1']}",
                          wanted=round(0.5 * (s.get("flow0", 0) + s.get("flow1", s.get("flow0", 0))), 3),
                          sim=round(float(np.interp(ts, tq, Q)) / st["Q_ref"], 3),
                          z_tip_sim_m=round(float(np.interp(ts, tq, il["z_tip"])), 4),
                          z_wanted_m=round(0.5 * (s["z0"] + s["z1"]), 4)))
    rep["flow_vs_stage_mid"] = flows
    errs = {}
    for ts_, xy in stc["foot_measurements_M"].items():
        tc = float(ts_)
        if realtime and any(f0 < tc < f1 for f0, f1 in FRZ):
            continue
        tsim = float(sim_of_clip(tc, realtime))
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
    rep["n_feet_without_inlet"] = sum(1 for v in errs.values() if v is None)
    x = np.where(act, il["x_M"], np.nan); y = np.where(act, il["y_M"], np.nan)
    rep["wiggle_sim_along_135deg"] = {"1.2-2.1s (script 2.9 Hz, 0.03-0.04)": _wiggle(t_il_clip, x, y, 1.2, 2.1, 135),
                                     "2.3-3.8s (3.2 Hz, 0.033)": _wiggle(t_il_clip, x, y, 2.3, 3.8, 135),
                                     "5.1-6.1s (3.1 Hz, 0.04)": _wiggle(t_il_clip, x, y, 5.1, 6.1, 135),
                                     "7.4-9.0s (3.2 Hz, 0.04->0.025)": _wiggle(t_il_clip, x, y, 7.4, 9.0, 135)}
    # final frame vs the extracted reference frame (T frame)
    from scipy.ndimage import zoom
    N = sn["s_M"].shape[1]
    wt = zoom(fr["white"][-1].astype(np.float32), N / fr["white"].shape[1], order=1)
    g = (np.arange(N) + 0.5) / N - 0.5
    X, Y = np.meshgrid(g, g); dm = np.hypot(X, Y) < 0.49

    def iou(a_, b_):
        a_, b_ = (a_ > 0.5) & dm, (b_ > 0.5) & dm
        return float((a_ & b_).sum() / max((a_ | b_).sum(), 1))
    j_end = int(np.argmin(np.abs(sn["t_clip"] - float(fr["t"][-1]))))
    sT = VX.sim_to_t(sn["s_M"][j_end])
    rep["final_t_clip_sim"] = float(sn["t_clip"][j_end])
    rep["final_iou_T_rowflipped"] = iou(sT, wt)
    rep["final_iou_T_noflip"] = iou(sn["s_M"][j_end], wt)
    rep["final_white_frac_sim"] = float(((sT > 0.5) & dm).sum() / dm.sum())
    rep["final_white_frac_video"] = float(((wt > 0.5) & dm).sum() / dm.sum())
    json.dump(rep, open(os.path.join(out, "compare.json"), "w"), indent=1)

    _, frames = MT.data()

    def panel_video(tc):
        i = min(int(round(tc * 15)), len(frames) - 1)
        p, _ = MT.ell_of(i)
        return np.clip(VX.rectify(frames[i].astype(np.float64) / 255.0, p, 192, 0.0, False), 0, 1)

    def marks(axs, tc):
        ts = float(sim_of_clip(tc, realtime))
        kk = min(np.searchsorted(il["t"], ts), len(il["t"]) - 1)
        pm, _ = VX.stage_point(SCRIPT_CLIP, tc)
        for a_ in axs:
            a_.axis("off")
            if il["active"][kk]:
                a_.plot(il["x_M"][kk], -il["y_M"][kk], "r+", ms=8, mew=1.5)
            if pm is not None:
                a_.plot(pm[0], -pm[1], "yx", ms=5)

    times = [0.8, 1.6, 2.6, 3.8, 5.6, 7.6, 8.6, 9.6, 10.4, 11.2, 12.8, 13.6, 14.4, 15.6, 16.6, 17.0, 20.5]
    fig, ax = plt.subplots(3, len(times), figsize=(1.8 * len(times), 6.0), dpi=90)
    for k, tc in enumerate(times):
        ax[0, k].imshow(panel_video(tc), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5])
        ax[0, k].set_title(f"video {tc:.1f}s", fontsize=7)
        j = int(np.argmin(np.abs(fr["t"] - tc)))
        w = np.where(fr["valid"][j], fr["white"][j].astype(np.float32), 0.4)
        ax[1, k].imshow(w, origin="lower", cmap="gray", vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
        ax[1, k].set_title(f"w {fr['t'][j]:.1f}s q{int(fr['quality'][j])}", fontsize=7)
        m = int(np.argmin(np.abs(sn["t_clip"] - tc)))
        ax[2, k].imshow(VX.sim_to_t(sn["s_M"][m]), origin="lower", cmap="gray", vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
        ax[2, k].set_title(f"old solver {sn['t_clip'][m]:.1f}s", fontsize=7)
        marks(ax[:, k], tc)
        sel = act & (t_il_clip <= tc)
        ax[2, k].plot(il["x_M"][sel], -il["y_M"][sel], "r-", lw=0.3)
        for ts_, xy in stc["foot_measurements_M"].items():
            if float(ts_) <= tc + 1e-6:
                ax[0, k].plot(xy[0], -xy[1], "c.", ms=2)
    fig.suptitle("swan stage script%s through the OLD solver (N=%d, Barista yaw %s), T frame (rot 0; the pitcher is at +y_T "
                 "here). red + = sim inlet, yellow x = stage intent, cyan = measured feet; video rectified with hand-fit "
                 "ellipses for t<=8.6 s; w row: frames.npz (propagated ellipse for t<7.8 s)"
                 % (" (realtime)" if realtime else "", N, yaw), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"old_solver_check{tag}.png"))
    plt.close(fig)
    tmp = tempfile.mkdtemp(prefix="swc_")
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
            frz = any(f0 - 1e-6 <= tc < f1 for f0, f1 in FRZ) or tc < 0.333
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
    ap.add_argument("--yaw", choices=["default", "video"], default="default")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    if not a.compare_only:
        print(json.dumps(run(a.N, a.realtime, a.yaw), indent=1, default=str), flush=True)
    r = compare(a.realtime, a.yaw)
    print(json.dumps({k: v for k, v in r.items() if k not in ("impact_vs_video", "flow_vs_stage_mid")}, indent=1))
