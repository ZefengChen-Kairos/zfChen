"""Sanity check of the heart stage script: stages_to_moves -> pitcher.BaristaScript -> OLD solver (action_opt.run_one
setup: physics #9 of runs/joint_sweep_r2fix, closure 'old'), N = 64.  NOT a model test: it only checks that the moves
run, that the stream lands where the video shows it, and that the durations line up with the clip.

    cd /home/user/zfChen && python latte_imex/video_targets/heart/old_solver_check.py [--N 64]

Writes runs/stage_check/heart/{snapshots.npz, inlet.npz, summary.json} and, next to this file,
old_solver_check.png (video frame | simulated field at matching clip times, both in the T frame, solver field
row-flipped with video_extract.sim_to_t) and old_solver_check.mp4 (same at 5 fps).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import action_opt as AO                                   # noqa: E402
from latte_imex import video_extract as VX                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "..", "stage_scripts", "heart.json")


def run(N=64, out="runs/stage_check/heart", fps_snap=5.0):
    from latte_imex.pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript
    from latte_imex.solver import Solver, Params, Numerics
    from latte_imex.optimize import visible
    os.makedirs(out, exist_ok=True)
    moves, t_clip0 = VX.stages_to_moves(SCRIPT)
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
    np.savez_compressed(os.path.join(out, "snapshots.npz"), t_sim=np.array(snap_t), t_clip=np.array(snap_t) + t_clip0,
                        s_M=np.array(snaps), mask=mask, N=N)
    np.savez_compressed(os.path.join(out, "inlet.npz"), t=tr[:, 0], active=tr[:, 1], x_M=tr[:, 2], y_M=tr[:, 3], S_eff=tr[:, 4],
                        t_pitcher=tq, Q=Q, z_tip=z_tip)
    summary = dict(N=N, ok=ok, elapsed_s=time.time() - t0, T_sim=script.T, t_clip0=t_clip0, n_moves=len(moves),
                   V_end_mL=float(script.pitcher.state.V) * 1e6, deposited=float(sol.ledger()["deposited"]),
                   poured_mL=float(np.trapezoid(Q, tq) * 1e6))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return summary


def compare(out="runs/stage_check/heart"):
    """Impact track vs the stage script / measured feet, first-flow time, and side-by-side figure + movie."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = json.load(open(SCRIPT))
    sn = np.load(os.path.join(out, "snapshots.npz"))
    il = np.load(os.path.join(out, "inlet.npz"))
    fr = VX.load_frames("heart")
    act = il["active"] > 0
    rep = {}
    # first time the stream lands in the sim
    rep["t_first_inlet_sim"] = float(il["t"][act][0]) if act.any() else None
    rep["t_last_inlet_sim"] = float(il["t"][act][-1]) if act.any() else None
    # impact vs measured feet (clip time = sim time + t_clip0)
    t_clip0 = st.get("t_clip0", 0.0)
    errs = {}
    for ts, xy in st["foot_measurements_M"].items():
        t = float(ts) - t_clip0
        k = np.searchsorted(il["t"], t)
        if k >= len(il["t"]) or not il["active"][k]:
            errs[ts] = None
            continue
        sim = np.array([il["x_M"][k], il["y_M"][k]])
        intent, _ = VX.stage_point(SCRIPT, float(ts))
        errs[ts] = dict(sim=sim.round(3).tolist(), measured=xy, intent=None if intent is None else intent.round(3).tolist(),
                        err_D=round(float(np.hypot(*(sim - np.array(xy)))), 3))
    rep["impact_vs_video"] = errs
    e = [v["err_D"] for v in errs.values() if v]
    rep["impact_err_median_D"] = float(np.median(e)) if e else None
    rep["impact_err_max_D"] = float(np.max(e)) if e else None
    # final-frame orientation check against targets (sim field row-flipped into T)
    w_t = fr["white"][-1].astype(np.float32)
    from scipy import ndimage as ndi
    N = int(sn["N"])
    wt = np.clip(ndi.zoom(w_t, N / w_t.shape[0], order=1), 0, 1)
    s_end = sn["s_M"][np.argmin(np.abs(sn["t_clip"] - fr["t"][-1]))]
    sT = VX.sim_to_t(s_end)
    g = (np.arange(N) + 0.5) / N - 0.5
    X, Y = np.meshgrid(g, g); dm = np.hypot(X, Y) < 0.49

    def iou(a, b):
        a, b = (a > 0.5) & dm, (b > 0.5) & dm
        return float((a & b).sum() / max((a | b).sum(), 1))
    rep["final_iou_T_rowflipped"] = iou(sT, wt)
    rep["final_iou_T_noflip"] = iou(s_end, wt)
    rep["final_white_frac_sim"] = float(((sT > 0.5) & dm).sum() / dm.sum())
    rep["final_white_frac_video"] = float(((wt > 0.5) & dm).sum() / dm.sum())
    json.dump(rep, open(os.path.join(out, "compare.json"), "w"), indent=1)
    # figure at stage boundaries
    times = [1.0, 2.2, 3.6, 5.0, 6.15, 6.9, 8.0, 8.7, 9.73, 11.6]
    frames, _ = VX.read_video(str(fr["clip"]))
    fig, ax = plt.subplots(3, len(times), figsize=(2.0 * len(times), 6.3), dpi=90)
    for k, tc in enumerate(times):
        j = int(np.argmin(np.abs(fr["t"] - tc)))
        i = int(round(tc * 15))
        p = fr["track_ell"][min(i, len(fr["track_ell"]) - 1)]
        rgbw = VX.rectify(frames[min(i, len(frames) - 1)].astype(np.float64) / 255.0, p, 192, float(fr["rot"]), bool(fr["flip"]))
        ax[0, k].imshow(np.clip(rgbw, 0, 1), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5])
        ax[0, k].set_title(f"video {tc:.2f}s", fontsize=7)
        w = np.where(fr["valid"][j], fr["white"][j].astype(np.float32), 0.4)
        ax[1, k].imshow(w, origin="lower", cmap="gray", vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
        ax[1, k].set_title(f"w (grey=invalid) {fr['t'][j]:.1f}s", fontsize=6)
        m = int(np.argmin(np.abs(sn["t_clip"] - tc)))
        ax[2, k].imshow(VX.sim_to_t(sn["s_M"][m]), origin="lower", cmap="gray", vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
        ax[2, k].set_title(f"old solver {sn['t_clip'][m]:.1f}s", fontsize=7)
        # impact markers in T: sim inlet (red), measured feet (cyan), intent (yellow)
        kk = np.searchsorted(il["t"], tc - t_clip0)
        for a in ax[:, k]:
            a.axis("off")
            if kk < len(il["t"]) and il["active"][kk]:
                a.plot(il["x_M"][kk], -il["y_M"][kk], "r+", ms=7, mew=1.5)
            pm, _ = VX.stage_point(SCRIPT, tc)
            if pm is not None:
                a.plot(pm[0], -pm[1], "yx", ms=5)
        # sim impact path so far
        sel = act & (il["t"] <= tc - t_clip0)
        ax[2, k].plot(il["x_M"][sel], -il["y_M"][sel], "r-", lw=0.6)
        for ts, xy in st["foot_measurements_M"].items():
            if float(ts) <= tc + 1e-6:
                ax[0, k].plot(xy[0], -xy[1], "c.", ms=2)
    fig.suptitle("heart stage script through the OLD solver (N=%d) — T frame (+y up, pours start at -y); red + = sim inlet, "
                 "yellow x = stage intent, cyan = measured feet" % N, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "old_solver_check.png"))
    plt.close(fig)
    # movie at 5 fps: rectified video | sim
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="hc_")
    try:
        for m, tc in enumerate(sn["t_clip"]):
            if tc > fr["t"][-1] + 1e-6:
                break
            i = int(round(tc * 15))
            p = fr["track_ell"][min(i, len(fr["track_ell"]) - 1)]
            rgbw = VX.rectify(frames[min(i, len(frames) - 1)].astype(np.float64) / 255.0, p, 192, float(fr["rot"]), bool(fr["flip"]))
            f2, a2 = plt.subplots(1, 2, figsize=(6.4, 3.4), dpi=100)
            a2[0].imshow(np.clip(rgbw, 0, 1), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5]); a2[0].set_title(f"video t={tc:.1f}s (T frame)", fontsize=8)
            a2[1].imshow(VX.sim_to_t(sn["s_M"][m]), origin="lower", cmap="gray", vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
            a2[1].set_title("old solver, stage script (sanity check only)", fontsize=8)
            kk = np.searchsorted(il["t"], tc - t_clip0)
            for a in a2:
                a.axis("off")
                if kk < len(il["t"]) and il["active"][kk]:
                    a.plot(il["x_M"][kk], -il["y_M"][kk], "r+", ms=9, mew=1.5)
            f2.tight_layout(); f2.savefig(os.path.join(tmp, f"f{m:04d}.png")); plt.close(f2)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "5", "-i", os.path.join(tmp, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        os.path.join(HERE, "old_solver_check.mp4")], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    if not a.compare_only:
        print(json.dumps(run(a.N), indent=1), flush=True)
    print(json.dumps(compare(), indent=1))
