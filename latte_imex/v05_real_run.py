"""V0.5 (single-layer IMEX solver, closure 'chi', return law 'v05') driven by the barista's REAL actions (the
physical-time stage scripts of the six clips) through the physical inlet of `v05_adapter`, evaluated against the
videos with `calib` (same observation operator, validity masks and metrics for video and model; senior V1 5.4).

    python -m latte_imex.v05_real_run check  --out DIR                      # calib.simulate_clip / pattern_video smoke
    python -m latte_imex.v05_real_run run    --sets FS,FSp2 --N 128 --out DIR [--names a,b] [--procs 3]
    python -m latte_imex.v05_real_run final  --sets FS --N 256 --out DIR    # final frame only (images)
    python -m latte_imex.v05_real_run run    --sets FS --u_exit 0.10,0.27 --names push_heart,tulip --out DIR
    python -m latte_imex.v05_real_run figs   --best FS --other FSp2 --out DIR
    python -m latte_imex.v05_real_run videos --sets FS --N 128 --k 3 --out DIR
    python -m latte_imex.v05_real_run summary --out DIR                     # summary.json of every evaluated run

Layout of DIR:  runs/<tag>_<name>_N<N>/{c.npz (t, c float16, M frame), ledger.json, inlet_log.npy, eval.json,
wT.npz (model whiteness on the clip's frames.npz grid and time axis, T frame)}, final_vs_real.png,
process_<name>.png, videos/<name>.mp4.  tag = closure set (FS | FSp2) + '_u<u_exit>' when u_exit != 0.15 m/s + '_q<Q_scale>' for the diagnostic
flow-scale runs (--q_scale, wanted flow x Q_scale; not a calibrated parameter).

A full run stores c at the union of the clip's 5 fps analysis frames and its 15 fps video frames, so the videos reuse
the same simulation (`Cached` returns those snapshots to make_videos.pattern_video).
No pre-roll (V0.5 has no first-white delay; `v05_adapter.strip_preroll`).
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

from . import calib as CB
from . import video_extract as VE
from .v05_adapter import CLOSURE_SETS, U_EXIT, V05Model

ORDER = ("heart", "push_heart", "layered_heart", "tulip", "leaf", "swan")
CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="郁金香", leaf="树叶", swan="天鹅")


def tag_of(set_name, u_exit, q_scale=1.0):
    t = set_name if abs(u_exit - U_EXIT) < 1e-12 else f"{set_name}_u{u_exit:.2f}"
    return t if abs(q_scale - 1.0) < 1e-12 else f"{t}_q{q_scale:.2f}"


def run_dir(out, set_name, u_exit, name, N, q_scale=1.0):
    return os.path.join(out, "runs", f"{tag_of(set_name, u_exit, q_scale)}_{name}_N{N}")


def video_times(name):
    from . import make_videos as MV
    return MV.clip_frames_phys(name)[1]


def _json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def job(args):
    set_name, name, N, u_exit, mode, out, q_scale = args
    t0 = time.time()
    d = run_dir(out, set_name, u_exit, name, N, q_scale)
    params = {} if abs(q_scale - 1.0) < 1e-12 else dict(Q_scale=q_scale)     # diagnostic only (flow scale)
    os.makedirs(d, exist_ok=True)
    clip = CB.load_clip(name)
    tc = clip["res"]["t"]
    if mode == "full":
        t_all = np.unique(np.concatenate([tc, video_times(name)]))
    else:
        t_all = np.array([tc[-1]])
    model = V05Model(CLOSURE_SETS[set_name], u_exit=u_exit)
    r = model.run(clip["script"], params, N, t_all)
    led = r["ledger"]
    np.savez_compressed(os.path.join(d, "c.npz"), t=t_all, c=r["c"].astype(np.float16))
    np.save(os.path.join(d, "inlet_log.npy"), r["inlet_log"])
    _json(dict(ledger=led, inlet_cols=list(r["inlet_cols"]), set=set_name, closure=CLOSURE_SETS[set_name],
               u_exit=u_exit, Q_scale=q_scale, N=N, mode=mode, name=name), os.path.join(d, "ledger.json"))
    idx = np.searchsorted(t_all, tc) if mode == "full" else np.array([0])
    wT = CB.observe_model(r["c"][idx], clip["M"])
    np.savez_compressed(os.path.join(d, "wT.npz"), t=tc if mode == "full" else tc[-1:], wT=wT.astype(np.float16))
    res = clip["res"]
    disk = res["disk"].astype(bool)
    if mode == "full":
        ev = CB.evaluate(clip, wT, led)
        keep = CB.compact(ev)
        keep["frames"] = ev["frames"]
        keep["events_full"] = ev["events"]
    else:
        fin = CB.shape_metrics(res["white"][-1].astype(np.float32), res["valid"][-1], wT[-1], disk)
        keep = dict(name=name, final=fin, ledger=led)
    keep.update(set=set_name, u_exit=u_exit, Q_scale=q_scale, N=N, wall_s=time.time() - t0)
    _json(keep, os.path.join(d, "eval.json"))
    msg = (f"{tag_of(set_name, u_exit, q_scale)} {name} N{N} {mode}: ok {led['ok']} lerr {led['layer_error']:.1e} "
           f"merr {led['milk_error']:.1e} Vrec {led['V_received_mL']:.1f} mL chi {led['chi_flow_weighted']} "
           f"final IoU {keep['final']['iou']} bnd {keep['final']['bnd_D']}")
    if mode == "full":
        msg += f" | proc IoU {keep['iou']['value']:.3f} dice {keep['dice']['value']:.3f} loss {keep['loss']:.3f}"
    msg += f" | {time.time() - t0:.0f} s"
    print(msg, flush=True)
    return msg


def run_jobs(jobs, procs):
    if procs <= 1:
        return [job(j) for j in jobs]
    from multiprocessing import get_context
    with get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        return pool.map(job, jobs, chunksize=1)


# =========================================================================== cached model (for videos)
class Cached:
    """Returns the snapshots of a stored full run at the requested times (exact match within 1e-6 s)."""
    space = {}

    def __init__(self, d):
        z = np.load(os.path.join(d, "c.npz"))
        self.t, self.c = z["t"], z["c"]
        self.ledger = json.load(open(os.path.join(d, "ledger.json")))["ledger"]

    def run(self, script, params, N, t_out):
        idx = np.searchsorted(self.t, np.asarray(t_out) - 1e-6)
        assert np.all(np.abs(self.t[idx] - t_out) < 1e-6), "time not stored"
        return dict(c=self.c[idx].astype(np.float32), ledger=self.ledger)


# =========================================================================== figures
def final_photo(clip):
    """Rectified sRGB final frame of the clip (T grid; displayed as is = physical top view, as make_videos)."""
    res = clip["res"]
    frames, _ = VE.read_video(VE.clip_path(clip["name"]))
    img = frames[int(res["frame_index"][-1])].astype(np.float64) / 255.0
    rect = VE.rectify(img, res["ell"][-1], clip["M"], float(res["rot"]), bool(res["flip"]))
    rect[~res["disk"].astype(bool)] = 1.0
    return np.clip(rect, 0, 1)


def grey(w, valid=None, disk=None):
    img = np.stack([w, w, w], -1).astype(np.float64) * 0.9 + 0.05
    if valid is not None:
        img[~valid] = [0.75, 0.35, 0.35]
    if disk is not None:
        img[~disk] = 1.0
    return img


def fig_final(out, best, other, names=ORDER, N=128, N_hi=256, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["real photo (rectified, final frame)", "real whiteness w (red = invalid)", f"model {best} N={N}",
            f"model {other} N={N}", f"model {best} N={N_hi}"]
    fig, axes = plt.subplots(len(names), len(cols), figsize=(2.75 * len(cols), 2.95 * len(names)), dpi=110, squeeze=False)
    for i, n in enumerate(names):
        clip = CB.load_clip(n)
        res = clip["res"]
        disk = res["disk"].astype(bool)
        panels = [(final_photo(clip), ""), (grey(res["white"][-1].astype(float), res["valid"][-1], disk),
                                             f"{n}, t = {res['t'][-1]:.1f} s (physical)")]
        for s, NN in ((best, N), (other, N), (best, N_hi)):
            d = run_dir(out, s, U_EXIT, n, NN)
            try:
                wT = np.load(os.path.join(d, "wT.npz"))["wT"][-1].astype(float)
                fin = json.load(open(os.path.join(d, "eval.json")))["final"]
                lab = f"IoU {fin['iou']:.2f}  bnd {fin['bnd_D']:.3f} D ({fin['bnd_D'] * 80:.1f} mm)"
                panels.append((grey(wT, None, disk), lab))
            except (OSError, KeyError, TypeError):
                panels.append((np.ones((clip["M"], clip["M"], 3)), "missing"))
        for j, (img, lab) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img, origin="upper", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_xlabel(lab, fontsize=7.5)
            if i == 0:
                ax.set_title(cols[j], fontsize=8)
    fig.suptitle("V0.5 with the barista's real actions (stage scripts) + physical inlet, final frame vs the real cup "
                 "(physical top view; metrics: calib final frame, threshold 0.5, video validity mask)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path or os.path.join(out, "final_vs_real.png"))
    plt.close(fig)


def fig_process(out, set_name, names=ORDER, N=128, suffix=""):
    for n in names:
        clip = CB.load_clip(n)
        d = run_dir(out, set_name, U_EXIT, n, N)
        wT = np.load(os.path.join(d, "wT.npz"))["wT"].astype(np.float32)
        ev = json.load(open(os.path.join(d, "eval.json")))
        title = (f"V0.5 {set_name}, real actions, N={N}: process IoU {ev['iou']['value']:.2f}, "
                 f"bnd {ev['boundary_D']['value']:.3f} D, final IoU {ev['final']['iou']:.2f}")
        CB.contact_sheet(clip, wT, os.path.join(out, f"process_{n}{suffix}.png"), n=10, title=title)


def videos(out, set_name, k=3.0, N=128, names=ORDER):
    from . import make_videos as MV
    vd = os.path.join(out, "videos")
    os.makedirs(vd, exist_ok=True)
    rep = {}
    for n in names:
        t0 = time.time()
        m = Cached(run_dir(out, set_name, U_EXIT, n, N))
        r = MV.pattern_video(m, {}, n, k, os.path.join(vd, f"{n}.mp4"), N=N,
                             title=f"V0.5 {set_name}, real actions, physical inlet")
        rep[n] = dict(path=r["path"], n_frames=r["n_frames"], T=r["T"], wall_s=time.time() - t0)
        print("video", n, rep[n], flush=True)
    _json(rep, os.path.join(vd, "videos.json"))
    return rep


# =========================================================================== summary
def summarize(out):
    rows = {}
    base = os.path.join(out, "runs")
    for dname in sorted(os.listdir(base)):
        p = os.path.join(base, dname, "eval.json")
        if not os.path.exists(p):
            continue
        ev = json.load(open(p))
        led = json.load(open(os.path.join(base, dname, "ledger.json")))["ledger"]
        r = dict(final_iou=ev["final"]["iou"], final_dice=ev["final"]["dice"], final_bnd_D=ev["final"]["bnd_D"],
                 final_mae=ev["final"]["mae"], ok=led["ok"], layer_error=led["layer_error"],
                 milk_error=led["milk_error"], V_received_mL=led["V_received_mL"], V_wanted_mL=led["V_wanted_mL"],
                 chi_flow_weighted=led["chi_flow_weighted"], ranges=led["ranges"], wall_s=ev.get("wall_s"))
        if "iou" in ev:
            pr = [p_ for p_ in ev["profiles"] if p_["at"] != "final"]
            pf = [p_ for p_ in ev["profiles"] if p_["at"] == "final"]
            fw = ev["first_white"]
            r.update(loss=ev["loss"], iou=ev["iou"]["value"], dice=ev["dice"]["value"], mae=ev["coverage_mae"]["value"],
                     bnd_D=ev["boundary_D"]["value"], n_shape=ev["iou"]["valid_count"],
                     fw_err_s=fw["value"], fw_onsets=[(o["t_onset"], o["video_s"], o["model_s"]) for o in fw["onsets"]],
                     stripes_stage_err=(float(np.mean([abs(p_["n_white_model"] - p_["n_white_video"]) for p_ in pr]))
                                        if pr else None),
                     stripes_stage=[(p_["at"], p_["n_white_video"], p_["n_white_model"]) for p_ in pr],
                     stripes_final=[(p_["n_white_video"], p_["n_white_model"]) for p_ in pf],
                     push_blocks=ev.get("push_blocks"), cut=ev.get("cut"))
        rows[dname] = r
    _json(rows, os.path.join(out, "summary.json"))
    return rows


# =========================================================================== smoke check of the two entry points
def check(out):
    from . import make_videos as MV
    m = V05Model(CLOSURE_SETS["FS"])
    clip = CB.load_clip("heart")
    wT, led = CB.simulate_clip(m, {}, clip, N=48)
    assert wT.shape == clip["res"]["white"].shape and led["ok"], led
    ev = CB.evaluate(clip, wT, led)
    r = MV.pattern_video(m, {}, "tulip", 3.0, os.path.join(out, "check_tulip.mp4"), N=48, max_frames=6)
    rep = dict(simulate_clip=dict(shape=list(wT.shape), ok=led["ok"], layer_error=led["layer_error"],
                                  milk_error=led["milk_error"], final_iou=ev["final"]["iou"], loss=ev["loss"]),
               pattern_video=dict(n_frames=r["n_frames"], ok=r["ledger"]["ok"], T=r["T"]))
    print(json.dumps(rep, default=float), flush=True)
    _json(rep, os.path.join(out, "check.json"))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run", "final", "figs", "videos", "summary"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="FS,FSp2")
    ap.add_argument("--names", default=",".join(ORDER))
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--u_exit", default=str(U_EXIT))
    ap.add_argument("--q_scale", type=float, default=1.0, help="diagnostic flow scale (not a calibrated parameter)")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--best", default="FS")
    ap.add_argument("--other", default="FSp2")
    ap.add_argument("--k", type=float, default=3.0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    names = [n for n in a.names.split(",") if n]
    sets = [s for s in a.sets.split(",") if s]
    us = [float(u) for u in a.u_exit.split(",")]
    if a.cmd == "check":
        check(a.out)
    elif a.cmd in ("run", "final"):
        mode = "full" if a.cmd == "run" else "final"
        jobs = [(s, n, a.N, u, mode, a.out, a.q_scale) for s in sets for u in us for n in names]
        jobs.sort(key=lambda j: -CB.load_clip(j[1])["res"]["t"][-1])        # longest clips first
        run_jobs(jobs, a.procs)
        summarize(a.out)
    elif a.cmd == "figs":
        fig_final(a.out, a.best, a.other, names)
        fig_process(a.out, a.best, names)
        fig_process(a.out, a.other, names, suffix=f"_{a.other}")
    elif a.cmd == "videos":
        videos(a.out, sets[0], a.k, a.N, names)
    elif a.cmd == "summary":
        rows = summarize(a.out)
        for k, r in rows.items():
            print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()
                      if kk in ("loss", "iou", "dice", "bnd_D", "fw_err_s", "final_iou", "final_bnd_D", "stripes_stage_err")})


if __name__ == "__main__":
    main()
