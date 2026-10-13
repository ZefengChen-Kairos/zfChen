"""Frozen-parameter evaluation of the V0.5 calibration (senior V1 5.1-5.4): the best AB (tulip + leaf) and the best A
(tulip only) parameters of `v05_calibrate` are FROZEN and evaluated on all six patterns at N = 128 with
`calib.evaluate(delay_in_loss=False)` (first-white error reported separately), side by side with the zero-tuning FSp2
baseline; final-frame figure at N = 256; process contact sheets and six 4-panel videos for the AB parameters.

    python -m latte_imex.v05_predict run    --cal CALDIR --out OUT [--stages eval128,baseline,videos,final256]
    python -m latte_imex.v05_predict figs   --cal CALDIR --out OUT --base V05VIDEO
    python -m latte_imex.v05_predict tables --cal CALDIR --out OUT --base V05VIDEO

CALDIR = the calibration output (calib_AB/calib.json, calib_A/calib.json, prereg.json); OUT may equal CALDIR.
V05VIDEO = the zero-tuning run of `v05_real_run` (runs/FSp2_<name>_N128/wT.npz, runs/FSp2_<name>_N256/...).

PARAMETER SETS (read from the calibration records; nothing here is chosen by looking at a test pattern):
  AB      calib_AB best (lowest mean N = 96 calibration loss over all CMA samples of the three inits)
  A       calib_A best
  AB_i1, AB_i2, AB_i3   best point of each AB init (one of them is AB itself; evaluated to show the spread of the
          prediction across inits)
  FSp2    zero-tuning start (FSp2 closure + u_exit 0.15 + Q_scale 1.0 = prereg x0 1), run through the SAME path as
          the calibrated sets (consistency check of the stored v05video baseline, which used other output times)
  FSp2v   the stored v05video FSp2 N = 128 fields (runs/FSp2_<name>_N128/wT.npz) re-evaluated with
          calib.evaluate(delay_in_loss=False): THE baseline of the comparison tables (same metrics, same N)

STAGES:
  eval128   calib.simulate_clip + calib.evaluate(delay_in_loss=False) at N = 128: AB, A, FSp2 and the AB inits that
            are not AB, on all six patterns -> OUT/runs/<set>_<name>_N128/{wT.npz, eval.json}
  baseline  FSp2v re-evaluation -> OUT/runs/FSp2v_<name>_N128/eval.json
  videos    make_videos.pattern_video(model, AB params, name, k = 3, N = 128) -> OUT/videos/<name>.mp4 (the model is
            simulated again at the 15 fps frame times inside pattern_video)
  final256  final frame only at N = 256 for AB and A -> OUT/runs/<set>_<name>_N256_final/{wT.npz, final.json}
            (calib.shape_metrics of the final frame + fine lines on the B -> D profile, `final_lines`)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

from . import calib as CB
from . import video_extract as VE

ALL = tuple(CB.CAL_AB) + tuple(CB.TEST)
CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="郁金香", leaf="树叶", swan="天鹅")
K_APPEAR = 3.0          # appearance parameter of the RGB panel of the videos (same as the v05video videos)
N_EVAL, N_FIG = 128, 256


def _spec():
    from . import v05_calibrate as VC
    return VC.SPEC


def _jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    os.replace(tmp, path)


def _jload(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


# =========================================================================== parameter sets
def param_sets(cal):
    ab = json.load(open(os.path.join(cal, "calib_AB", "calib.json")))
    a = json.load(open(os.path.join(cal, "calib_A", "calib.json")))
    pre = json.load(open(os.path.join(cal, "prereg.json")))
    sets = dict(AB=ab["best"]["params"], A=a["best"]["params"], FSp2=pre["x0s"][0])
    src = dict(AB=f"calib_AB best (loss {ab['best']['loss']:.4f})", A=f"calib_A best (loss {a['best']['loss']:.4f})",
               FSp2="prereg x0 1 (FSp2 closure + u_exit 0.15 + Q_scale 1.0), zero tuning")
    best_init = None
    for i, ini in enumerate(ab["inits"]):
        k = f"AB_i{i + 1}"
        sets[k] = ini["best"]["params"]
        src[k] = f"calib_AB init {i + 1} best (loss {ini['best']['loss']:.4f}, gen {ini['best']['gen'] + 1})"
        if all(abs(ini["best"]["params"][q] - sets["AB"][q]) < 1e-12 for q in sets["AB"]):
            best_init = k
    return sets, src, best_init


def eval_jobs(cal):
    sets, _, best_init = param_sets(cal)
    names = ["AB", "A", "FSp2"] + [k for k in sets if k.startswith("AB_i") and k != best_init]
    return [("eval", s, sets[s], n, N_EVAL) for s in names for n in ALL]


# =========================================================================== fine lines on the final frame
def final_lines(clip, w_final):
    """White runs on the pre-registered B -> D profile of the FINAL frame (video: causal fill as calib.profile_metrics;
    model: the final field itself, read through the video's observed mask).  Returns (video, model) axis_runs dicts."""
    res = clip["res"]
    disk = res["disk"].astype(bool)
    Wv, obs = VE.filled_white(res["white"].astype(np.float32), res["valid"] & disk[None], res["t"])
    j = len(res["t"]) - 1
    prof = clip["profiles"][0]
    a = VE.axis_runs(Wv[j], obs[j], prof["p0_T"], prof["p1_T"])
    b = VE.axis_runs(np.asarray(w_final, np.float32), obs[j], prof["p0_T"], prof["p1_T"])
    return a, b


def _n_runs(r):
    """(interior count as calib uses it = n_white, or 0 when there are fewer than 2 white runs; all white runs)."""
    if r is None:
        return None, None
    return int(r.get("n_white", 0)), int(sum(1 for x in r["runs"] if x[0] == "W"))


PROM = 0.10             # prominence of the threshold-free peak count (v05video/extra/peaks.py, same definition)


def npeaks(w, valid, p0_T, p1_T):
    """Threshold-free stripe count: whiteness maxima with prominence >= PROM along the B -> D profile (axis_runs
    sampling, valid samples only, counted on each contiguous valid segment padded with 0), as v05video/extra/peaks.py."""
    from scipy.signal import find_peaks
    s, prof = profile_curve(w, valid, p0_T, p1_T)
    ok = np.isfinite(prof)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return 0
    n = 0
    for sg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(sg) < 5:
            continue
        pk, _ = find_peaks(np.concatenate([[0.0], prof[sg], [0.0]]), prominence=PROM)
        n += len(pk)
    return int(n)


def final_peaks(clip, w_final):
    """(video, model) peak counts on the final frame; the video's raw whiteness and its validity mask (no fill), as
    peaks.py."""
    res = clip["res"]
    vv = res["valid"][-1] & res["disk"].astype(bool)
    prof = clip["profiles"][0]
    return (npeaks(res["white"][-1].astype(np.float32), vv, prof["p0_T"], prof["p1_T"]),
            npeaks(np.asarray(w_final, np.float32), vv, prof["p0_T"], prof["p1_T"]))


# =========================================================================== workers
def _job(task):
    kind = task[0]
    t0 = time.time()
    if kind == "eval":
        _, s, params, name, N = task
        out = os.environ["V05P_OUT"]
        d = os.path.join(out, "runs", f"{s}_{name}_N{N}")
        if os.path.exists(os.path.join(d, "eval.json")):
            return f"{s} {name} N{N}: cached"
        os.makedirs(d, exist_ok=True)
        model = CB.make_model(_spec())
        clip = CB.load_clip(name)
        wT, led = CB.simulate_clip(model, params, clip, N)
        ev = CB.evaluate(clip, wT, led, delay_in_loss=False)
        np.savez_compressed(os.path.join(d, "wT.npz"), t=clip["res"]["t"], wT=wT.astype(np.float16))
        ev.update(set=s, params=params, N=N, wall_s=time.time() - t0, delay_in_loss=False)
        _jdump(ev, os.path.join(d, "eval.json"))
        return (f"{s} {name} N{N}: ok {(led or {}).get('ok')} loss {ev['loss']:.3f} IoU {ev['iou']['value']:.3f} "
                f"final IoU {ev['final']['iou']:.3f} | {time.time() - t0:.0f} s")
    if kind == "final":
        _, s, params, name, N = task
        out = os.environ["V05P_OUT"]
        d = os.path.join(out, "runs", f"{s}_{name}_N{N}_final")
        if os.path.exists(os.path.join(d, "final.json")):
            return f"{s} {name} N{N} final: cached"
        os.makedirs(d, exist_ok=True)
        model = CB.make_model(_spec())
        clip = CB.load_clip(name)
        res = clip["res"]
        r = model.run(clip["script"], params, N, res["t"][-1:], keep_log=False)
        wT = CB.observe_model(r["c"], clip["M"])
        disk = res["disk"].astype(bool)
        fin = CB.shape_metrics(res["white"][-1].astype(np.float32), res["valid"][-1], wT[0], disk)
        a, b = final_lines(clip, wT[0])
        np.savez_compressed(os.path.join(d, "wT.npz"), t=res["t"][-1:], wT=wT.astype(np.float16))
        _jdump(dict(set=s, name=name, N=N, params=params, final=fin, lines_video=a, lines_model=b,
                    ledger=r["ledger"], wall_s=time.time() - t0), os.path.join(d, "final.json"))
        return f"{s} {name} N{N} final: ok {r['ledger'].get('ok')} IoU {fin['iou']:.3f} | {time.time() - t0:.0f} s"
    if kind == "video":
        _, s, params, name, N, path, title = task
        if os.path.exists(path):
            return f"video {name}: exists"
        from . import make_videos as MV
        model = CB.make_model(_spec())
        r = MV.pattern_video(model, params, name, K_APPEAR, path, N=N, title=title)
        meta = dict(path=r["path"], n_frames=r["n_frames"], T=r["T"], N=N, k=K_APPEAR, set=s, params=params,
                    ledger_ok=(r.get("ledger") or {}).get("ok"), wall_s=time.time() - t0)
        _jdump(meta, path.replace(".mp4", ".json"))
        return f"video {name}: {r['n_frames']} frames | {time.time() - t0:.0f} s"
    if kind == "baseline":
        _, name, base = task
        out = os.environ["V05P_OUT"]
        d = os.path.join(out, "runs", f"FSp2v_{name}_N{N_EVAL}")
        if os.path.exists(os.path.join(d, "eval.json")):
            return f"FSp2v {name}: cached"
        os.makedirs(d, exist_ok=True)
        src = os.path.join(base, "runs", f"FSp2_{name}_N{N_EVAL}")
        z = np.load(os.path.join(src, "wT.npz"))
        led = json.load(open(os.path.join(src, "ledger.json")))["ledger"]
        clip = CB.load_clip(name)
        assert np.allclose(z["t"], clip["res"]["t"]), "time axis differs"
        wT = z["wT"].astype(np.float32)
        ev = CB.evaluate(clip, wT, led, delay_in_loss=False)
        ev.update(set="FSp2v", source=src, N=N_EVAL, delay_in_loss=False)
        _jdump(ev, os.path.join(d, "eval.json"))
        return f"FSp2v {name}: loss {ev['loss']:.3f} | {time.time() - t0:.0f} s"
    raise ValueError(kind)


def _pool_map(tasks, procs):
    from multiprocessing import get_context
    if procs <= 1:
        return [print(_job(t), flush=True) for t in tasks]
    with get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(_job, tasks, chunksize=1):
            print(time.strftime("%H:%M:%S"), msg, flush=True)


def provenance():
    here = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    files = ("v05_predict.py", "v05_calibrate.py", "calib.py", "v05_adapter.py", "solver.py", "kernels.py",
             "v3min_adapter.py", "video_extract.py", "make_videos.py", "appearance.py")
    return dict(git_head=head, sha1={f: hashlib.sha1(open(os.path.join(here, f), "rb").read()).hexdigest()[:12]
                                     for f in files}, numpy=np.__version__)


def run(cal, out, base, stages, procs=4):
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    os.environ["V05P_OUT"] = out
    sets, src, best_init = param_sets(cal)
    _jdump(dict(sets=sets, source=src, ab_best_is=best_init, spec=_spec(), N_eval=N_EVAL, N_fig=N_FIG,
                delay_in_loss=False, k_video=K_APPEAR, provenance=provenance(), date=time.strftime("%Y-%m-%d %H:%M:%S"),
                cal=os.path.abspath(cal), base=os.path.abspath(base)), os.path.join(out, "predict_setup.json"))
    for st in stages:
        t0 = time.time()
        print(time.strftime("%H:%M:%S"), f"stage {st} start", flush=True)
        if st == "eval128":
            _pool_map(eval_jobs(cal), procs)
        elif st == "baseline":
            _pool_map([("baseline", n, os.path.abspath(base)) for n in ALL], procs)
        elif st == "videos":
            vd = os.path.join(out, "videos")
            os.makedirs(vd, exist_ok=True)
            title = "V0.5 calibrated on tulip+leaf (AB), frozen"
            _pool_map([("video", "AB", sets["AB"], n, N_EVAL, os.path.join(vd, f"{n}.mp4"), title) for n in ALL], procs)
        elif st == "final256":
            order = ("swan", "tulip", "push_heart", "leaf", "heart", "layered_heart")     # longest first
            jobs = [("final", s, sets[s], n, N_FIG) for n in order for s in ("AB", "A")]
            _pool_map(jobs, procs)
        else:
            raise ValueError(st)
        print(time.strftime("%H:%M:%S"), f"stage {st} done in {time.time() - t0:.0f} s", flush=True)


# =========================================================================== loading results
def load_eval(out, s, n, N=None):
    N = N or N_EVAL
    return _jload(os.path.join(out, "runs", f"{s}_{n}_N{N}", "eval.json"))


def load_wT(out, s, n, N=None):
    N = N or N_EVAL
    return np.load(os.path.join(out, "runs", f"{s}_{n}_N{N}", "wT.npz"))["wT"].astype(np.float32)


def decompose(ev):
    """Terms of calib.loss_of(delay=False) + the reported-only quantities."""
    d = ev["dice"]["value"]
    b = ev["boundary_D"]["value"]
    pr = [p for p in ev["profiles"] if p["at"] != "final"]
    pf = [p for p in ev["profiles"] if p["at"] == "final"]
    line = (float(np.mean([abs(p["n_white_model"] - p["n_white_video"]) / max(p["n_white_video"], 1) for p in pr]))
            if pr else 0.0)
    return dict(loss=ev["loss"], one_minus_dice=1.0 - d if d is not None else 1.0,
                bnd_term=(b if b is not None else CB.EMPTY_BND) / CB.BND_SCALE, bnd_mm=(b * 80 if b is not None else None),
                line_term=line, iou=ev["iou"]["value"], mae=ev["coverage_mae"]["value"],
                first_white_err_s=ev["events"]["first_white"]["value"],
                first_white_n=ev["events"]["first_white"]["valid_count"],
                final_iou=ev["final"]["iou"], final_bnd_mm=(ev["final"]["bnd_D"] or 0) * 80,
                stage_lines=[(p["at"], p["n_white_video"], p["n_white_model"]) for p in pr],
                final_lines=[(p["n_white_video"], p["n_white_model"]) for p in pf])


# =========================================================================== figures
def _grey(w, valid=None, disk=None):
    img = np.stack([w, w, w], -1).astype(np.float64) * 0.9 + 0.05
    if valid is not None:
        img[~valid] = [0.75, 0.35, 0.35]
    if disk is not None:
        img[~disk] = 1.0
    return img


def _style(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _profile_xy(clip):
    prof = clip["profiles"][0]
    M = clip["M"]
    p0, p1 = np.array(prof["p0_T"]), np.array(prof["p1_T"])
    return [(p0[0] + 0.5) * M - 0.5, (p1[0] + 0.5) * M - 0.5], [(p0[1] + 0.5) * M - 0.5, (p1[1] + 0.5) * M - 0.5]


def _role(n):
    return "calibration (AB)" if n in CB.CAL_AB else "TEST"


def fig_final(cal, out, base, path=None):
    """Rows = six patterns (calibration first), columns: real photo | real whiteness (B -> D profile in cyan) |
    zero-tuning FSp2 | calibrated AB | calibrated A, all final frames at N = 256 in the physical top view
    (origin upper, as make_videos), calib final-frame IoU / boundary distance and B -> D fine lines below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .v05_real_run import final_photo
    cols = ["real photo (rectified final frame)", "real whiteness w (red = invalid,\ncyan = B->D profile)",
            "zero tuning FSp2", "calibrated AB (tulip + leaf)", "calibrated A (tulip only)"]
    fig, axes = plt.subplots(len(ALL), len(cols), figsize=(2.75 * len(cols), 3.5 * len(ALL)), dpi=90, squeeze=False)
    for i, n in enumerate(ALL):
        clip = CB.load_clip(n)
        res = clip["res"]
        disk = res["disk"].astype(bool)
        panels = [(final_photo(clip), f"{n} ({_role(n)})"),
                  (_grey(res["white"][-1].astype(float), res["valid"][-1], disk), None)]
        # zero tuning FSp2 (v05video, N = 256 full run; final metrics from its eval.json)
        d = os.path.join(base, "runs", f"FSp2_{n}_N256")
        w = np.load(os.path.join(d, "wT.npz"))["wT"][-1].astype(np.float32)
        fin = json.load(open(os.path.join(d, "eval.json")))["final"]
        a, b = final_lines(clip, w)
        pv, pm = final_peaks(clip, w)
        e128 = load_eval(out, "FSp2v", n)
        panels.append((_grey(w, None, disk), _lab(fin, b, pm, e128)))
        for s in ("AB", "A"):
            f = _jload(os.path.join(out, "runs", f"{s}_{n}_N{N_FIG}_final", "final.json"))
            w = np.load(os.path.join(out, "runs", f"{s}_{n}_N{N_FIG}_final", "wT.npz"))["wT"][0].astype(np.float32)
            lab = _lab(f["final"], f["lines_model"], final_peaks(clip, w)[1], load_eval(out, s, n))
            if s == "A" and n == "leaf":
                lab += "\n(leaf not used by A)"
            panels.append((_grey(w, None, disk), lab))
        nv = _n_runs(a)
        panels[1] = (panels[1][0], f"t = {res['t'][-1]:.1f} s (physical)\nB->D: lines {nv[0]}, peaks {pv}")
        for j, (img, lab) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img, origin="upper", interpolation="nearest")
            if j == 1:
                xs, ys = _profile_xy(clip)
                ax.plot(xs, ys, color="c", lw=0.8, alpha=0.8)
                ax.plot(xs[1], ys[1], "o", color="c", ms=2.5)
            _style(ax)
            ax.set_xlabel(lab, fontsize=7)
            if i == 0:
                ax.set_title(cols[j], fontsize=8)
    fig.suptitle("V0.5, barista's real actions + physical inlet: final frame vs the real cup.  Zero tuning (FSp2) vs "
                 "frozen calibrated parameters (AB = tulip + leaf, A = tulip only); N = 256, physical top view as "
                 "make_videos\nmetrics: calib final frame (threshold 0.5, video validity mask); B->D profile (cyan, "
                 "dot = D end): 'lines' = white runs at 0.5 as calib counts them, 'peaks' = whiteness maxima with "
                 "prominence >= 0.1 (no threshold)",
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path or os.path.join(out, "final_vs_real.png"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _lab(fin, lines_model, peaks, e128):
    nm = _n_runs(lines_model)
    s = f"IoU {fin['iou']:.2f}  bnd {fin['bnd_D'] * 80:.1f} mm  (256²)\nB->D: lines {nm[0]}, peaks {peaks}"
    if e128 is not None:
        pf = [p for p in e128["profiles"] if p["at"] == "final"]
        s += (f"\n128²: IoU {e128['final']['iou']:.2f}  bnd {e128['final']['bnd_D'] * 80:.1f} mm"
              + (f"  lines {pf[0]['n_white_model']}" if pf else ""))
    return s


def fig_process(out):
    """calib.contact_sheet of the AB parameters at N = 128 (arrays flipped vertically, so that calib's origin='lower'
    shows the physical top view of make_videos / final_vs_real.png)."""
    paths = []
    for n in ALL:
        clip = CB.load_clip(n)
        ev = load_eval(out, "AB", n)
        wT = load_wT(out, "AB", n)
        res = clip["res"]
        res_f = dict(res, white=res["white"][:, ::-1], valid=res["valid"][:, ::-1], disk=res["disk"][::-1])
        clip_f = dict(clip, res=res_f)
        title = (f"V0.5 calibrated AB (frozen), N=128 [{_role(n)}]: loss {ev['loss']:.2f}, process IoU "
                 f"{ev['iou']['value']:.2f}, bnd {ev['boundary_D']['value'] * 80:.1f} mm, final IoU "
                 f"{ev['final']['iou']:.2f} (physical top view)")
        p = os.path.join(out, f"process_{n}_AB.png")
        CB.contact_sheet(clip_f, wT[:, ::-1], p, n=10, title=title)
        paths.append(p)
    return paths


def fig_inits(cal, out, base):
    """Final frames at N = 128 of every evaluated parameter set: real whiteness | FSp2 (v05video) | AB init 1 / 2 / 3
    best | A, with final IoU and B -> D lines (spread of the prediction across the AB inits)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets, src, best_init = param_sets(cal)
    cols = [("video", "real whiteness"), ("FSp2v", "zero tuning FSp2")] + [
        (("AB" if k == best_init else k), f"AB init {k[-1]} best" + (" (= AB)" if k == best_init else ""))
        for k in ("AB_i1", "AB_i2", "AB_i3")] + [("A", "A best")]
    fig, axes = plt.subplots(len(ALL), len(cols), figsize=(2.3 * len(cols), 2.75 * len(ALL)), dpi=90, squeeze=False)
    for i, n in enumerate(ALL):
        clip = CB.load_clip(n)
        res = clip["res"]
        disk = res["disk"].astype(bool)
        for j, (s, lab) in enumerate(cols):
            ax = axes[i, j]
            _style(ax)
            if i == 0:
                ax.set_title(lab, fontsize=8)
            if s == "video":
                ax.imshow(_grey(res["white"][-1].astype(float), res["valid"][-1], disk), origin="upper")
                ax.set_xlabel(f"{n} ({_role(n)})", fontsize=7)
                continue
            ev = load_eval(out, s, n)
            if ev is None:
                ax.set_xlabel("not run", fontsize=7)
                continue
            w = (load_wT(out, s, n) if s != "FSp2v" else
                 np.load(os.path.join(base, "runs", f"FSp2_{n}_N128", "wT.npz"))["wT"].astype(np.float32))[-1]
            ax.imshow(_grey(w, None, disk), origin="upper")
            pf = [p for p in ev["profiles"] if p["at"] == "final"]
            ax.set_xlabel(f"L {ev['loss']:.2f}  proc IoU {ev['iou']['value']:.2f}\nfinal IoU {ev['final']['iou']:.2f}  "
                          f"lines {pf[0]['n_white_model'] if pf else '-'}/{pf[0]['n_white_video'] if pf else '-'}",
                          fontsize=7)
    fig.suptitle("Frozen parameter sets at N = 128 (final frame, physical top view): spread of the prediction across "
                 "the three AB inits\nL = calib loss without the delay term; lines = B->D interior white runs model/video",
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(out, "inits_N128.png")
    fig.savefig(p, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return p


def profile_curve(w, valid, p0_T, p1_T, band=0.02, step=0.004):
    """The averaged profile that VE.axis_runs thresholds (same sampling), NaN where not observed."""
    M = w.shape[0]
    p0, p1 = np.asarray(p0_T, float), np.asarray(p1_T, float)
    L = np.linalg.norm(p1 - p0)
    u = (p1 - p0) / L
    nrm = np.array([-u[1], u[0]])
    s = np.arange(0, L, step)
    prof = []
    for si in s:
        vals, vs = [], []
        for o in np.linspace(-band, band, 5):
            q = p0 + si * u + o * nrm
            ix, iy = int((q[0] + 0.5) * M), int((q[1] + 0.5) * M)
            if 0 <= ix < M and 0 <= iy < M:
                vals.append(w[iy, ix]); vs.append(valid[iy, ix])
        prof.append(np.mean(vals) if vals and np.mean(vs) > 0.6 else np.nan)
    return s, np.array(prof)


def _final_fields(out, base, n, N):
    """Final model frames (T grid) of FSp2 (v05video) / AB / A at N (N_FIG: final-only runs; N_EVAL: protocol runs)."""
    ws = dict(FSp2=np.load(os.path.join(base, "runs", f"FSp2_{n}_N{N}", "wT.npz"))["wT"][-1].astype(np.float32))
    for k in ("AB", "A"):
        if N == N_FIG:
            ws[k] = np.load(os.path.join(out, "runs", f"{k}_{n}_N{N}_final", "wT.npz"))["wT"][0].astype(np.float32)
        else:
            ws[k] = load_wT(out, k, n, N)[-1]
    return ws


def fig_lines(out, base, N=None):
    """B -> D profile of the final frame at N (N_FIG or N_EVAL): video vs FSp2 / AB / A, with the 0.5 threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = N or N_FIG
    fig, axes = plt.subplots(len(ALL), 1, figsize=(9, 1.55 * len(ALL)), dpi=100, squeeze=False)
    for i, n in enumerate(ALL):
        clip = CB.load_clip(n)
        res = clip["res"]
        disk = res["disk"].astype(bool)
        Wv, obs = VE.filled_white(res["white"].astype(np.float32), res["valid"] & disk[None], res["t"])
        j = len(res["t"]) - 1
        prof = clip["profiles"][0]
        ax = axes[i, 0]
        s, pv = profile_curve(Wv[j], obs[j], prof["p0_T"], prof["p1_T"])
        a, _ = final_lines(clip, Wv[j])
        ax.plot(s * 80, pv, color="k", lw=1.6, label=f"video ({_n_runs(a)[0]} / {final_peaks(clip, Wv[j])[0]})")
        ws = _final_fields(out, base, n, N)
        for k, c in (("FSp2", "tab:gray"), ("AB", "tab:red"), ("A", "tab:blue")):
            _, pm = profile_curve(ws[k], obs[j], prof["p0_T"], prof["p1_T"])
            _, b = final_lines(clip, ws[k])
            ax.plot(s * 80, pm, color=c, lw=1.0, label=f"{k} ({_n_runs(b)[0]} / {final_peaks(clip, ws[k])[1]})")
        ax.axhline(CB.THR, color="0.6", lw=0.6, ls=":")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, s[-1] * 80)
        ax.set_ylabel(f"{n}\n{_role(n)}", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, ncol=4, loc="upper right", framealpha=0.8)
    axes[-1, 0].set_xlabel("distance along the B -> D profile [mm] (B end = 0); whiteness averaged over +-1.6 mm across",
                           fontsize=7)
    fig.suptitle(f"Final frame, B -> D profile (N = {N}, through the calib observation operator; video: causal fill as "
                 "calib)\nlegend (lines / peaks): white runs at 0.5 counted as calib does / maxima with prominence "
                 ">= 0.1 (no threshold)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, f"lines_BD_N{N}.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def figs(cal, out, base):
    fig_final(cal, out, base)
    fig_process(out)
    fig_inits(cal, out, base)
    fig_lines(out, base, N_EVAL)
    fig_lines(out, base, N_FIG)


# =========================================================================== tables
def tables(cal, out, base):
    sets, src, best_init = param_sets(cal)
    rep = dict(sets=sets, source=src, ab_best_is=best_init, N=N_EVAL, delay_in_loss=False)
    evs = {}
    for s in ["FSp2v", "FSp2", "AB", "A"] + [k for k in sets if k.startswith("AB_i")]:
        e = {n: load_eval(out, s if not (s.startswith("AB_i") and s == best_init) else "AB", n) for n in ALL}
        e = {n: v for n, v in e.items() if v is not None}
        if not e:
            continue
        evs[s] = e
        rep[s] = dict(calibration_fit=CB.summary({n: e[n] for n in CB.CAL_AB if n in e}),
                      test_prediction=CB.summary({n: e[n] for n in CB.TEST if n in e}),
                      terms={n: decompose(e[n]) for n in e})
        for n in e:
            s_ = "AB" if s == best_init else s
            w = (np.load(os.path.join(base, "runs", f"FSp2_{n}_N{N_EVAL}", "wT.npz"))["wT"][-1].astype(np.float32)
                 if s == "FSp2v" else load_wT(out, s_, n)[-1])
            rep[s]["terms"][n]["final_peaks_video_model"] = final_peaks(CB.load_clip(n), w)
    fin = {}
    for s in ("AB", "A"):
        for n in ALL:
            f = _jload(os.path.join(out, "runs", f"{s}_{n}_N{N_FIG}_final", "final.json"))
            if f:
                w = np.load(os.path.join(out, "runs", f"{s}_{n}_N{N_FIG}_final", "wT.npz"))["wT"][0].astype(np.float32)
                fin.setdefault(s, {})[n] = dict(iou=f["final"]["iou"], bnd_mm=f["final"]["bnd_D"] * 80,
                                                lines_model=_n_runs(f["lines_model"]),
                                                lines_video=_n_runs(f["lines_video"]),
                                                peaks_video_model=final_peaks(CB.load_clip(n), w),
                                                chi=f["ledger"].get("chi_flow_weighted"),
                                                V_mL=f["ledger"].get("V_received_mL"), wall_s=f["wall_s"])
    for n in ALL:
        d = os.path.join(base, "runs", f"FSp2_{n}_N256")
        if os.path.exists(os.path.join(d, "wT.npz")):
            clip = CB.load_clip(n)
            w = np.load(os.path.join(d, "wT.npz"))["wT"][-1].astype(np.float32)
            f = json.load(open(os.path.join(d, "eval.json")))["final"]
            a, b = final_lines(clip, w)
            led = json.load(open(os.path.join(d, "ledger.json")))["ledger"]
            fin.setdefault("FSp2", {})[n] = dict(iou=f["iou"], bnd_mm=f["bnd_D"] * 80, lines_model=_n_runs(b),
                                                 lines_video=_n_runs(a), peaks_video_model=final_peaks(clip, w),
                                                 chi=led.get("chi_flow_weighted"),
                                                 V_mL=led.get("V_received_mL"))
    rep["final_N256"] = fin
    led128 = {s: {n: {k: (e[n].get("ledger") or {}).get(k) for k in ("chi_flow_weighted", "V_received_mL", "ok")}
                  for n in e} for s, e in evs.items()}
    rep["ledger_N128"] = led128
    _jdump(rep, os.path.join(out, "prediction.json"))
    return rep


SET_CN = dict(FSp2v="零调参 FSp2", FSp2="FSp2（同路径重算）", AB="AB 校准", A="A 校准", AB_i1="AB 起点 1", AB_i2="AB 起点 2",
              AB_i3="AB 起点 3")


def _f(v, fmt="{:.3f}"):
    return "–" if v is None else fmt.format(v)


def tables_md(rep):
    """Markdown tables (Chinese headers) of prediction.json."""
    L = []
    hdr = ("| 图案 | 参数 | L | 1 − Dice | 边界项（mm） | 细纹项 | 过程 IoU | 覆盖 MAE | 阶段末细纹 视频→模型 | 末帧 IoU | "
           "末帧边界 mm | 末帧白段 视频/模型 | 末帧峰数 视频/模型 | 首白误差 s（个数） |")
    sep = "|" + "---|" * 14

    def rows(names, sets):
        out = [hdr, sep]
        for n in names:
            for s in sets:
                if s not in rep or n not in rep[s]["terms"]:
                    continue
                t = rep[s]["terms"][n]
                st = "，".join(f"{v}→{m}" for _, v, m in t["stage_lines"]) or "–"
                fl = "，".join(f"{v}/{m}" for v, m in t["final_lines"]) or "–"
                pk = t.get("final_peaks_video_model")
                fw = t["first_white_err_s"]
                note = "（A 未用）" if (s == "A" and n == "leaf") else ""
                out.append(f"| {CN[n]}{note} | {SET_CN[s]} | {t['loss']:.3f} | {t['one_minus_dice']:.3f} | "
                           f"{t['bnd_term']:.3f}（{_f(t['bnd_mm'], '{:.2f}')}） | {t['line_term']:.3f} | {_f(t['iou'])} | "
                           f"{_f(t['mae'])} | {st} | {_f(t['final_iou'])} | {t['final_bnd_mm']:.2f} | {fl} | "
                           f"{(str(pk[0]) + '/' + str(pk[1])) if pk else '–'} | "
                           f"{_f(fw, '{:.2f}')}（{t['first_white_n']}） |")
        return out

    def means(names, sets):
        out = ["| 参数 | L | 1 − Dice | 边界项（mm） | 细纹项 | 过程 IoU | 覆盖 MAE | 末帧 IoU | 末帧边界 mm | 首白误差 s |",
               "|" + "---|" * 10]
        for s in sets:
            if s not in rep:
                continue
            ts = [rep[s]["terms"][n] for n in names if n in rep[s]["terms"]]
            if not ts:
                continue
            m = lambda k: float(np.mean([t[k] for t in ts if t[k] is not None]))
            fw = [t["first_white_err_s"] for t in ts if t["first_white_err_s"] is not None]
            out.append(f"| {SET_CN[s]} | {m('loss'):.3f} | {m('one_minus_dice'):.3f} | {m('bnd_term'):.3f}（"
                       f"{m('bnd_mm'):.2f}） | {m('line_term'):.3f} | {m('iou'):.3f} | {m('mae'):.3f} | {m('final_iou'):.3f} | "
                       f"{m('final_bnd_mm'):.2f} | {(np.mean(fw) if fw else float('nan')):.2f} |")
        return out

    L.append("### 校准拟合（郁金香、树叶），N = 128")
    L += rows(CB.CAL_AB, ["FSp2v", "AB", "A"])
    L.append("")
    L.append("两图等权平均（A 的平均含树叶，但 A 没有用树叶校准；只看郁金香见上表）：")
    L.append("")
    L += means(CB.CAL_AB, ["FSp2v", "AB", "A"])
    L.append("")
    L.append("### 测试预测（大白心、推推乐、千层心、天鹅），N = 128")
    L += rows(CB.TEST, ["FSp2v", "AB", "A"])
    L.append("")
    L.append("四图等权平均：")
    L.append("")
    L += means(CB.TEST, ["FSp2v", "AB", "A"])
    L.append("")
    L.append("### AB 三个起点的最好点（冻结），N = 128")
    L += rows(CB.TEST + CB.CAL_AB, ["AB_i1", "AB_i2", "AB_i3"])
    L.append("")
    L += means(CB.TEST, ["AB_i1", "AB_i2", "AB_i3"])
    L.append("")
    L.append("### 一致性检查：FSp2 零调参，v05video 存的场重评 vs 同一路径重算，N = 128")
    L += rows(ALL, ["FSp2v", "FSp2"])
    L.append("")
    L.append("### 末帧 N = 256（calib 末帧指标；白段 = 阈值 0.5 的 calib 计数，峰数 = 显著度 ≥ 0.1 的极大值）")
    L.append("| 图案 | 视频 白段/峰数 | 零调参 FSp2 IoU / 边界 mm / 白段 / 峰数 | AB IoU / 边界 mm / 白段 / 峰数 | "
             "A IoU / 边界 mm / 白段 / 峰数 |")
    L.append("|---|---|---|---|---|")
    fin = rep["final_N256"]
    for n in ALL:
        cells = []
        v = None
        for s in ("FSp2", "AB", "A"):
            f = fin.get(s, {}).get(n)
            if f is None:
                cells.append("–")
                continue
            v = (f["lines_video"][0], f["peaks_video_model"][0])
            cells.append(f"{f['iou']:.3f} / {f['bnd_mm']:.2f} / {f['lines_model'][0]} / {f['peaks_video_model'][1]}")
        L.append(f"| {CN[n]} | {v[0]}/{v[1]} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### 流量加权 χ 与收到的奶量（N = 128 的账本）")
    L.append("| 图案 | " + " | ".join(SET_CN[s] for s in ("FSp2v", "AB", "A", "AB_i1", "AB_i3")) + " |")
    L.append("|---|" + "---|" * 5)
    for n in ALL:
        cells = []
        for s in ("FSp2v", "AB", "A", "AB_i1", "AB_i3"):
            g = rep.get("ledger_N128", {}).get(s, {}).get(n)
            cells.append("–" if not g or g.get("chi_flow_weighted") is None else
                         f"χ {g['chi_flow_weighted']:.3f}，{g['V_received_mL']:.0f} mL")
        L.append(f"| {CN[n]} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "figs", "tables"])
    ap.add_argument("--cal", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", required=True)
    ap.add_argument("--stages", default="eval128,baseline,videos,final256")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out or a.cal)
    if a.cmd == "run":
        run(a.cal, out, a.base, [s for s in a.stages.split(",") if s], a.procs)
    elif a.cmd == "figs":
        figs(a.cal, out, a.base)
    elif a.cmd == "tables":
        rep = tables(a.cal, out, a.base)
        md = tables_md(rep)
        open(os.path.join(out, "prediction_tables.md"), "w").write(md + "\n")
        print(json.dumps({s: dict(cal=rep[s]["calibration_fit"]["mean"], test=rep[s]["test_prediction"]["mean"])
                          for s in rep if isinstance(rep[s], dict) and "calibration_fit" in rep[s]}, indent=1,
                         default=float))


if __name__ == "__main__":
    from latte_imex import v05_predict as _me
    _me.main()
