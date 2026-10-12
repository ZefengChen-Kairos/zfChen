"""Sanity check of the tulip stage script: stages_to_moves -> pitcher.BaristaScript -> OLD solver (action_opt.run_one
setup: physics #9 of runs/joint_sweep_r2fix, closure 'old', physical footprint), N = 64.  NOT a model test: it only
checks that the moves run, that the stream lands where the video shows it, and that the durations/flows line up.

    cd /home/user/zfChen && python latte_imex/video_targets/tulip/tulip_solver_check.py [--N 64] [--physical] [--compare-only]

--physical runs stage_scripts/tulip_physical.json (freeze-frames removed, see freeze_tools.py) and maps sim time back
to clip time with freeze_tools.phys_to_clip for the comparison with the video.
Writes runs/stage_check/tulip[_physical]/{snapshots.npz, inlet.npz, summary.json, compare.json} and, next to this
file, tulip_solver_check[_physical].{png,mp4} (rectified video | extracted w | simulated field, all in the T frame
displayed with +y_M up, i.e. the M frame; solver field row-flipped with video_extract.sim_to_t).
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
from latte_imex import action_opt as AO                                   # noqa: E402
from latte_imex import video_extract as VX                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import freeze_tools as FT                                                  # noqa: E402

SCRIPT = os.path.abspath(os.path.join(HERE, "..", "..", "stage_scripts", "tulip.json"))
SCRIPT_PHYS = os.path.abspath(os.path.join(HERE, "..", "..", "stage_scripts", "tulip_physical.json"))
T_SWITCH, T_REF_DIR = 3.95, 4.0          # tilt-phase orientation fix (see stage script "orientation_fix")


def run(script, N=64, out="runs/stage_check/tulip", fps_snap=5.0):
    from latte_imex.pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript
    from latte_imex.solver import Solver, Params, Numerics
    from latte_imex.optimize import visible
    os.makedirs(out, exist_ok=True)
    moves, t_clip0 = VX.stages_to_moves(script)
    geom = GridPitcherGeometry.load(nr=6)
    ph, kQ = AO.physics9()
    cp = Coupling(c_S=kQ / AO.D_L ** 3, c_U=1.0 / AO.D_L, c_u=1.0 / AO.D_L, footprint="physical")
    bs = BaristaScript(moves=moves, pitcher=Pitcher(geom=geom, V0=250e-6), coupling=cp)
    pd = dict(ph, D=1e-7, kappa_Q=125.0, return_law="v05"); pd.update(AO.MODELS["old"])
    num = Numerics(N=N)
    sol = Solver(Params(**pd), num)
    t_snap = list(np.arange(0.0, bs.T + 1e-9, 1.0 / fps_snap))
    snaps, snap_t, track = [], [], []
    t0 = time.time()
    while sol.t < bs.T - 1e-9:
        probe, _ = bs.sample(sol.t)
        fdt = num.frame_dt
        if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
            fdt = min(fdt, max(num.scan_safety * min(sol.g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
        inl, lab = bs.sample(sol.t + 0.5 * fdt)
        track.append((sol.t, float(inl.active), *(inl.x_hit if inl.active else (np.nan, np.nan)),
                      float(inl.S_eff) if inl.active else 0.0))
        sol.advance_frame(inl, frame_dt=fdt)
        if int(sol.t) != int(sol.t - fdt):
            print(f"  sim t = {sol.t:.1f} / {bs.T:.1f} s  ({time.time() - t0:.0f} s wall)", flush=True)
        while t_snap and sol.t >= t_snap[0] - 1e-9:
            snaps.append(visible(sol.concentration(), sol.g.mask).astype(np.float32))
            snap_t.append(t_snap.pop(0))
    c = sol.concentration()
    ok = bool(np.all(np.isfinite(c[sol.g.mask])))
    tr = np.array(track, float)
    rec = bs.records
    Q = np.array([r["Q"] for r in rec]); tq = np.array([r["t"] for r in rec])
    z_tip = np.array([r["tip"][2] for r in rec])
    np.savez_compressed(os.path.join(out, "snapshots.npz"), t_sim=np.array(snap_t), s_M=np.array(snaps), mask=sol.g.mask, N=N,
                        t_clip0=t_clip0)
    np.savez_compressed(os.path.join(out, "inlet.npz"), t=tr[:, 0], active=tr[:, 1], x_M=tr[:, 2], y_M=tr[:, 3], S_eff=tr[:, 4],
                        t_pitcher=tq, Q=Q, z_tip=z_tip)
    summary = dict(script=script, N=N, finite=ok, wall_s=round(time.time() - t0, 1), T_sim=bs.T, t_clip0=t_clip0,
                   n_moves=len(moves), V_end_mL=round(float(bs.pitcher.state.V) * 1e6, 1),
                   poured_mL=round(float(np.trapezoid(Q, tq) * 1e6), 1), deposited=float(sol.ledger()["deposited"]))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return summary


def _fixed_ell(fr, t):
    """Ellipse at clip time t; for t < T_SWITCH keep centre/size, choose phi so +y_M has the level-phase image direction."""
    tt, te = fr["track_t"], fr["track_ell"]
    k = min(int(np.argmin(np.abs(tt - t))), len(te) - 1)
    p = te[k].copy()
    if tt[k] >= T_SWITCH:
        return p
    rot, flip = float(fr["rot"]), bool(fr["flip"])

    def ydir(q):
        a = math.radians(rot)
        ex, ey = VX.basis(q)
        Xr, Yr = -math.sin(a) * (-0.3), math.cos(a) * (-0.3)            # (x_T, y_T) = (0, -0.3) = (0, +0.3)_M
        u, v = 2 * Xr, -2 * Yr
        return math.atan2(q[3] * v * ey[1] + q[2] * u * ex[1], q[3] * v * ey[0] + q[2] * u * ex[0])
    want = ydir(te[int(np.argmin(np.abs(tt - T_REF_DIR)))])
    phs = np.linspace(-math.pi, math.pi, 1441)
    errs = []
    for ph in phs:
        q = p.copy(); q[4] = ph
        errs.append(abs((ydir(q) - want + math.pi) % (2 * math.pi) - math.pi))
    p[4] = phs[int(np.argmin(errs))]
    return p


def compare(script, out="runs/stage_check/tulip", physical=False, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = json.load(open(SCRIPT))                         # measured feet / freezes always in clip time
    d = VX.load_stages(script)
    sn = np.load(os.path.join(out, "snapshots.npz"))
    il = np.load(os.path.join(out, "inlet.npz"))
    fr = VX.load_frames("tulip")
    t_clip0 = float(d["t_clip0"])
    to_clip = (lambda ts: FT.phys_to_clip(ts + t_clip0, SCRIPT)) if physical else (lambda ts: ts + t_clip0)
    to_sim = (lambda tc: FT.clip_to_phys(tc, SCRIPT) - t_clip0) if physical else (lambda tc: tc - t_clip0)
    act = il["active"] > 0
    rep = dict(script=os.path.basename(script), physical_time=physical)
    rep["t_first_inlet_clip"] = float(to_clip(il["t"][act][0])) if act.any() else None
    rep["t_last_inlet_clip"] = float(to_clip(il["t"][act][-1])) if act.any() else None

    def sim_at(tc):
        k = int(np.searchsorted(il["t"], to_sim(tc)))
        if k >= len(il["t"]) or not il["active"][k]:
            return None
        return np.array([il["x_M"][k], il["y_M"][k]])
    # impact vs measured feet (clip time)
    errs = {}
    for ts, xy in st["foot_measurements_M"].items():
        tc = float(ts)
        s = sim_at(tc)
        if s is None:
            errs[ts] = None
            continue
        errs[ts] = dict(sim=s.round(3).tolist(), measured=xy, err_D=round(float(np.hypot(*(s - np.array(xy)))), 3))
    rep["impact_vs_video"] = errs
    e = [v["err_D"] for v in errs.values() if v]
    rep["impact_err_median_D"] = float(np.median(e)) if e else None
    rep["impact_err_p90_D"] = float(np.percentile(e, 90)) if e else None
    rep["impact_err_max_D"] = float(np.max(e)) if e else None
    rep["n_feet_without_sim_stream"] = int(sum(v is None for v in errs.values()))
    # per stage: impact vs intent, flow vs wanted
    per = []
    tq, Q = il["t_pitcher"], il["Q"]
    for s in d["stages"]:
        a, b = s["t0"] - t_clip0, s["t1"] - t_clip0        # stage times are in the script's own time base
        sel = (il["t"] >= a) & (il["t"] < b) & act
        ee = []
        for k in np.flatnonzero(sel)[::5]:
            p, _ = VX.stage_point(d, float(il["t"][k] + t_clip0))
            if p is not None:
                ee.append(float(np.hypot(il["x_M"][k] - p[0], il["y_M"][k] - p[1])))
        qs = (tq >= a) & (tq < b)
        per.append(dict(kind=s["kind"], t0=s["t0"], t1=s["t1"], freeze=bool(s.get("freeze", False)),
                        impact_vs_intent_mean_D=round(float(np.mean(ee)), 4) if ee else None,
                        impact_vs_intent_max_D=round(float(np.max(ee)), 4) if ee else None,
                        Q_mean_mLs=round(float(Q[qs].mean() * 1e6), 2) if qs.any() else None,
                        Q_want_mLs=round(0.5 * (s["flow0"] + s["flow1"]) * d["Q_ref"] * 1e6, 2)))
    rep["per_stage"] = per
    # final frame vs extracted reference frame (both in T)
    from scipy import ndimage as ndi
    N = int(sn["N"])
    wt = np.clip(ndi.zoom(fr["white"][-1].astype(np.float32), N / fr["white"].shape[1], order=1), 0, 1)
    t_snap_clip = np.array([to_clip(float(x)) for x in sn["t_sim"]])
    m_end = int(np.argmin(np.abs(t_snap_clip - float(fr["t"][-1]))))
    sT = VX.sim_to_t(sn["s_M"][m_end])
    g = (np.arange(N) + 0.5) / N - 0.5
    X, Y = np.meshgrid(g, g); dm = np.hypot(X, Y) < 0.49

    def iou(a_, b_):
        a_, b_ = (a_ > 0.5) & dm, (b_ > 0.5) & dm
        return float((a_ & b_).sum() / max((a_ | b_).sum(), 1))
    rep["final_t_clip_sim"] = float(t_snap_clip[m_end])
    rep["final_iou_T_rowflipped"] = round(iou(sT, wt), 3)
    rep["final_iou_T_noflip"] = round(iou(sn["s_M"][m_end], wt), 3)
    rep["final_white_frac_sim"] = round(float(((sT > 0.5) & dm).sum() / dm.sum()), 3)
    rep["final_white_frac_video"] = round(float(((wt > 0.5) & dm).sum() / dm.sum()), 3)
    json.dump(rep, open(os.path.join(out, "compare.json"), "w"), indent=1)

    frames, _ = VX.read_video(str(fr["clip"]))
    rot, flip = float(fr["rot"]), bool(fr["flip"])

    def video_rect(tc, M=192):
        i = min(int(round(tc * 15)), len(frames) - 1)
        p = _fixed_ell(fr, i / 15.0)
        r = VX.rectify(frames[i].astype(np.float64) / 255.0, p, M, rot, flip)
        Xg, Yg = VX.cup_grid(M)
        r[np.hypot(Xg, Yg) > 0.5] *= 0.3
        return np.clip(r, 0, 1)
    ext = [-0.5, 0.5, -0.5, 0.5]             # origin='upper': T row 0 (y_T = -0.5 = y_M = +0.5) drawn at y = +0.5 -> M frame, +y_M up

    def marks(a, tc):
        s = sim_at(tc)
        if s is not None:
            a.plot(s[0], s[1], "r+", ms=8, mew=1.6)
        pm, _ = VX.stage_point(d, to_clip(to_sim(tc)) if not physical else tc)
        if pm is not None and not physical:
            a.plot(pm[0], pm[1], "yx", ms=6)
    # figure at stage boundaries
    times = [0.85, 2.4, 3.37, 5.07, 5.33, 6.5, 8.75, 9.7, 10.45, 11.8, 13.45, 14.3, 14.85, 15.8, 16.37, 17.75]
    fig, ax = plt.subplots(3, len(times), figsize=(1.75 * len(times), 5.6), dpi=90)
    for k, tc in enumerate(times):
        ax[0, k].imshow(video_rect(tc), extent=ext); ax[0, k].set_title(f"video {tc:.2f}s", fontsize=7)
        j = int(np.argmin(np.abs(fr["t"] - tc)))
        w = np.where(fr["valid"][j], fr["white"][j].astype(np.float32), 0.4)
        ax[1, k].imshow(w, cmap="gray", vmin=0, vmax=1, extent=ext); ax[1, k].set_title(f"w {fr['t'][j]:.1f}s", fontsize=6)
        m = int(np.argmin(np.abs(t_snap_clip - tc)))
        ax[2, k].imshow(VX.sim_to_t(sn["s_M"][m]), cmap="gray", vmin=0, vmax=1, extent=ext)
        ax[2, k].set_title(f"old sim {t_snap_clip[m]:.1f}s", fontsize=7)
        for a in ax[:, k]:
            a.set_xlim(-0.5, 0.5); a.set_ylim(-0.5, 0.5); a.axis("off")
            marks(a, tc)
        sel = act & (il["t"] <= to_sim(tc))
        ax[2, k].plot(il["x_M"][sel], il["y_M"][sel], "r-", lw=0.5)
        for ts, xy in st["foot_measurements_M"].items():
            if float(ts) <= tc + 1e-6:
                ax[0, k].plot(xy[0], xy[1], "c.", ms=2)
    fig.suptitle(f"tulip stage script ({os.path.basename(script)}) through the OLD solver (N={N}), M frame (+y_M = B/spout side up); "
                 "red + = sim impact, yellow x = stage intent, cyan = measured feet (tilt-phase orientation fixed)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"tulip_solver_check{tag}.png"))
    plt.close(fig)
    # movie 5 fps over the clip: video | w | sim
    tmp = tempfile.mkdtemp(prefix="tc_")
    try:
        n = 0
        for tc in np.arange(0.0, float(fr["t"][-1]) + 1e-6, 0.2):
            f2, a2 = plt.subplots(1, 3, figsize=(9.0, 3.4), dpi=100)
            a2[0].imshow(video_rect(tc), extent=ext); a2[0].set_title(f"video t={tc:.1f}s", fontsize=8)
            j = int(np.argmin(np.abs(fr["t"] - tc)))
            a2[1].imshow(np.where(fr["valid"][j], fr["white"][j].astype(np.float32), 0.4), cmap="gray", vmin=0, vmax=1, extent=ext)
            a2[1].set_title("extracted w (grey = invalid)", fontsize=8)
            m = int(np.argmin(np.abs(t_snap_clip - tc)))
            a2[2].imshow(VX.sim_to_t(sn["s_M"][m]), cmap="gray", vmin=0, vmax=1, extent=ext)
            frz = any(s.get("freeze") and s["t0"] <= tc < s["t1"] for s in st["stages"])
            a2[2].set_title("old solver (sanity only)" + (" [FREEZE]" if frz else ""), fontsize=8)
            for a in a2:
                a.set_xlim(-0.5, 0.5); a.set_ylim(-0.5, 0.5); a.axis("off")
                marks(a, tc)
            f2.tight_layout(); f2.savefig(os.path.join(tmp, f"f{n:04d}.png")); plt.close(f2); n += 1
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "5", "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        os.path.join(HERE, f"tulip_solver_check{tag}.mp4")], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--physical", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    script = SCRIPT_PHYS if a.physical else SCRIPT
    out = "runs/stage_check/tulip" + ("_physical" if a.physical else "")
    tag = "_physical" if a.physical else ""
    if a.physical and not os.path.exists(SCRIPT_PHYS):
        FT.drop_freezes(SCRIPT, SCRIPT_PHYS)
    if not a.compare_only:
        print(json.dumps(run(script, a.N, out), indent=1), flush=True)
    r = compare(script, out, a.physical, tag)
    print(json.dumps({k: v for k, v in r.items() if k not in ("impact_vs_video",)}, indent=1))
