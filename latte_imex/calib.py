"""Calibration / evaluation harness for the fluid parameters, following the senior's V1 protocol (2026-10-10):
section 5.1 data split (calibrate A = tulip, B = leaf; frozen-parameter test on heart, push_heart, layered_heart, swan;
an A-only baseline on the same four tests), 5.2 dynamics first (shape/boundary/old-pattern/event observables of the
process frames; appearance parameter k only afterwards), 5.4 metrics, 5.5 record format (value / unit / valid_count /
valid_duration / status / reason).

Everything below is fixed BEFORE any model is run and is never changed by looking at a test pattern:

  time base     physical time = clip time with the freeze frames removed (both encodings of the scripts: stages with
                "freeze": true + "physical_s", and "freezes": [[t0, t1, note], ...] lists with no physical time).
                Frames inside a freeze are dropped (they are copies of one image).  Clips that start mid-pour get a
                pre-roll of the first stage (PREROLL, from calibration clips only: tulip's first white appears 0.26 s
                after the clip starts while its later onsets show 1.4-2.1 s physical delays); their first onset, and
                the leaf's (pour already running at the clip start), are excluded from the delay metric.
  observation   video: video_extract's whiteness w (frames.npz) and its validity mask (pitcher, stream, overlay, glare
                invalid).  model: c (raft coverage = whiteness, M frame) -> T frame -> bilinear resampling onto the
                frames.npz grid -> Gaussian camera PSF SIGMA_CAM_D.  Same threshold THR, same validity mask, same
                observable code (video_extract.compute_observables) for both.
  shape frames  ellipse quality >= Q_MIN (the propagated tilt-phase frames are not rectified reliably), valid fraction
                of the disc >= FV_MIN, no instructor overlay.
  profiles      fine-line counts / pitch along the instructor's B -> D axis through C (landmarks_M of the scripts;
                heart: the cut direction through the cup centre), extended to r = 0.47 D.
  loss          L = <1 - Dice>_t + <d_bnd>_t / BND_SCALE + mean |delay error| / DELAY_SCALE
                    + mean over stage-end profiles of |n_white error| / max(n_video, 1)
                (time averages over shape frames weighted by physical time; each pattern weighted equally).

A model is any object with  .space = {name: (lo, hi, 'log'|'lin')}  and
  .run(script_phys: dict, params: dict, N: int, t_out: array) -> dict(c=[n, N, N] float in [0, 1] (M frame), ledger=...)
where script_phys is a physical-time stage script (video_extract format, plus 'barista_yaw_rad') and t_out are
simulation times (0 = start of the script).

    python -m latte_imex.calib selftest [N=128] [names]   # video-as-sharp-model and blank-model check of the harness
"""
import json
import math
import os
import sys

import numpy as np
from scipy import ndimage as ndi

from . import video_extract as VE

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "stage_scripts")
TARGETS = os.path.join(HERE, "video_targets")

CAL_A = ("tulip",)
CAL_AB = ("tulip", "leaf")
TEST = ("heart", "push_heart", "layered_heart", "swan")
FILE_ID = dict(tulip="pattern_composite", leaf="pattern_rosetta", heart="pattern_heart", push_heart="pattern_push",
               layered_heart="pattern_layered_heart", swan="pattern_swan")

SIGMA_CAM_D = 0.005        # camera PSF of the whiteness extraction [cup diameters] (1 px of the 192 grid)
THR = 0.5                  # white / brown threshold, same for video and model
Q_MIN = 1                  # minimum ellipse quality of a shape frame
FV_MIN = 0.3               # minimum valid fraction of the disc of a shape frame
BND_SCALE = 0.05           # boundary-distance scale in the loss [D]
DELAY_SCALE = 0.3          # first-white delay scale in the loss [s]
EMPTY_BND = 0.5            # boundary distance when only one side has white [D] (domain scale)
PREROLL = dict(tulip=1.5)  # [s] pre-roll of the first stage for clips that start mid-pour
SKIP_FIRST_ONSET = ("tulip", "leaf")


# =========================================================================== time base
def freeze_cuts(d):
    """[(t0, t1, keep_s)] clip-time freeze intervals of a stage script, sorted, overlapping entries merged."""
    raw = []
    for s in d["stages"]:
        if s.get("freeze"):
            raw.append((float(s["t0"]), float(s["t1"]), float(s.get("physical_s", 0.0))))
    for f in d.get("freezes") or []:
        if isinstance(f, dict):
            raw.append((float(f["t0"]), float(f["t1"]), float(f.get("physical_s", 0.0))))
        else:
            raw.append((float(f[0]), float(f[1]), 0.0))
    raw.sort()
    out = []
    for c in raw:
        if out and c[0] < out[-1][1] - 1e-6:          # same freeze given twice (tulip): keep the stage entry
            p = out[-1]
            out[-1] = (min(p[0], c[0]), max(p[1], c[1]), max(p[2], c[2]))
        else:
            out.append(c)
    return out


def clip_to_phys(t, cuts):
    shift = 0.0
    for t0, t1, keep in cuts:
        rem = max(t1 - t0 - keep, 0.0)
        if t >= t1:
            shift += rem
        elif t > t0:
            return t0 - shift + keep * (t - t0) / (t1 - t0)
    return t - shift


def phys_to_clip(tp, cuts):
    shift = 0.0
    for t0, t1, keep in cuts:
        rem = max(t1 - t0 - keep, 0.0)
        if tp >= t0 - shift + keep:
            shift += rem
        elif tp > t0 - shift:
            return t0 + (tp - (t0 - shift)) * (t1 - t0) / max(keep, 1e-9)
    return tp + shift


def clip_script(name):
    return VE.load_stages(os.path.join(SCRIPTS, f"{name}.json"))


def physical_script(name, preroll=None):
    """Stage script in physical time (freezes removed, zero-length stages dropped, optional pre-roll of the first
    stage), with 'barista_yaw_rad', 'cuts' and 'preroll' recorded.  Time 0 = start of the (pre-rolled) script."""
    d = clip_script(name)
    cuts = freeze_cuts(d)
    pre = PREROLL.get(name, 0.0) if preroll is None else preroll
    st = []
    for s in d["stages"]:
        t0p, t1p = clip_to_phys(s["t0"], cuts) + pre, clip_to_phys(s["t1"], cuts) + pre
        if t1p - t0p < 1e-3:
            continue
        s2 = dict(s)
        s2.pop("freeze", None)
        s2.update(t0=round(t0p, 4), t1=round(t1p, 4), t_clip=[s["t0"], s["t1"]])
        st.append(s2)
    for a, b in zip(st[:-1], st[1:]):
        if b["t0"] < a["t1"]:
            b["t0"] = a["t1"]
    if pre > 0:
        s0 = dict(st[0])
        s0.update(t0=0.0, t1=st[0]["t0"], p1=s0["p0"], z1=s0["z0"], flow1=s0["flow0"], t_clip=None,
                  note=f"pre-roll {pre} s (clip starts mid-pour)")
        st.insert(0, s0)
    out = {k: v for k, v in d.items() if k not in ("stages", "freezes", "foot_measurements_M")}
    out.update(stages=st, barista_yaw_rad=VE.script_yaw(d), cuts=cuts, preroll=pre, t_clip0=0.0,
               time_base="physical time + preroll (calib.physical_script)")
    return VE.load_stages(out)


# =========================================================================== clip data
_CLIPS = {}


def load_clip(name):
    """Video frames on the physical time axis + masks + the video's own observables (same code as the model's)."""
    if name in _CLIPS:
        return _CLIPS[name]
    z = dict(np.load(os.path.join(TARGETS, name, "frames.npz"), allow_pickle=True))
    d = clip_script(name)
    cuts = freeze_cuts(d)
    pre = PREROLL.get(name, 0.0)
    tc = z["t"].astype(float)
    keep, tp, last = [], [], -1e9
    for k, t in enumerate(tc):
        if any(t0 + 1e-6 < t < t1 - 1e-6 for t0, t1, _ in cuts):
            continue
        p = clip_to_phys(t, cuts) + pre
        if p <= last + 1e-6:
            continue
        keep.append(k); tp.append(p); last = p
    keep = np.array(keep)
    res = {k: v for k, v in z.items()}
    for key in ("white", "valid", "findex", "ell", "quality", "gain", "lo", "hi", "levels_own", "occ_frac", "tip",
                "stream_px", "overlay", "frame_index"):
        if key in res and np.ndim(res[key]) >= 1 and len(res[key]) == len(tc):
            res[key] = res[key][keep]
    res["t"] = np.array(tp)
    res["t_clip"] = tc[keep]
    res["name"] = name
    disk = res["disk"].astype(bool)
    fv = (res["valid"] & disk[None]).sum((1, 2)) / disk.sum()
    shape_ok = (res["quality"] >= Q_MIN) & (fv >= FV_MIN) & ~res["overlay"].astype(bool)
    phys = physical_script(name)
    obs = VE.compute_observables(res, phys)
    clip = dict(name=name, res=res, script=phys, cuts=cuts, shape_ok=shape_ok, frac_valid=fv, obs=obs,
                profiles=profiles(name), M=int(res["white"].shape[1]))
    _CLIPS[name] = clip
    return clip


def profiles(name):
    """Pre-registered fine-line profiles (T frame): the B -> D pattern axis through C, extended to r = 0.47 D."""
    d = json.load(open(os.path.join(SCRIPTS, f"{name}.json")))
    lm = d.get("landmarks_M") or {}
    if "B" in lm and ("D" in lm or "D_side_line_point" in lm):
        B = np.array(lm["B"], float); Dp = np.array(lm.get("D", lm.get("D_side_line_point")), float)
        C = np.array(lm.get("C", (B + Dp) / 2), float)
        u = (Dp - B) / np.linalg.norm(Dp - B)
    else:                                          # heart: the cut direction through the cup centre
        cut = [s for s in d["stages"] if s["kind"] == "high_cut"]
        u = np.array(cut[-1]["p1"], float) - np.array(cut[0]["p0"], float)
        u /= np.linalg.norm(u)
        C = np.zeros(2)
    # chord through C along u, clipped to r = 0.47
    b = C @ u; cc = C @ C - 0.47 ** 2
    s0, s1 = -b - math.sqrt(b * b - cc), -b + math.sqrt(b * b - cc)
    pM0, pM1 = C + s0 * u, C + s1 * u
    return [dict(id="axis_BD", p0_T=VE.m_to_t(pM0).round(4).tolist(), p1_T=VE.m_to_t(pM1).round(4).tolist())]


# =========================================================================== observation operator
def observe_model(cM, Mo):
    """Model coverage frames c_M [n, N, N] (M frame) -> whiteness on the frames.npz grid (T frame, Mo x Mo)."""
    cM = np.asarray(cM, np.float32)
    n, N = cM.shape[0], cM.shape[1]
    xo = (np.arange(Mo) + 0.5) / Mo * N - 0.5                     # output cell centres in model index units
    I, J = np.meshgrid(xo, xo, indexing="ij")
    out = np.empty((n, Mo, Mo), np.float32)
    sig = SIGMA_CAM_D * Mo
    for j in range(n):
        cT = cM[j][::-1, :]
        w = ndi.map_coordinates(cT, [I, J], order=1, mode="nearest")
        out[j] = np.clip(ndi.gaussian_filter(w, sig), 0.0, 1.0) if sig > 0 else w
    return out


def model_res(clip, wT):
    """A frames.npz-like dict for the model, on the clip's physical time axis, with the VIDEO's validity mask."""
    r = dict(clip["res"])
    r["white"] = wT.astype(np.float16)
    return r


# =========================================================================== metrics (senior V1 section 5.4)
def _boundary(m, ok):
    b = m ^ ndi.binary_erosion(m, iterations=1, border_value=0)
    return b & ok


def shape_metrics(wv, vv, wm, disk):
    """One frame: probability MAE, IoU at THR, Dice, symmetric mean boundary distance [D] on the valid region."""
    v = vv & disk
    ok = ndi.binary_erosion(v, iterations=2)                      # boundaries next to occlusion / rim are excluded
    n = v.sum()
    mae = float(np.abs(wv[v].astype(float) - wm[v]).mean()) if n else None
    A, B = (wv > THR) & v, (wm > THR) & v
    inter, uni = (A & B).sum(), (A | B).sum()
    iou = None if uni == 0 else float(inter / uni)
    dice = None if uni == 0 else float(2 * inter / (A.sum() + B.sum()))
    bA, bB = _boundary(A, ok), _boundary(B, ok)
    M = wv.shape[0]
    if not bA.any() and not bB.any():
        bnd = None
    elif not bA.any() or not bB.any():
        bnd = EMPTY_BND
    else:
        dA = ndi.distance_transform_edt(~bA) / M
        dB = ndi.distance_transform_edt(~bB) / M
        bnd = float(0.5 * (dA[bB].mean() + dB[bA].mean()))
    return dict(mae=mae, iou=iou, dice=dice, bnd_D=bnd, valid_px=int(n), empty_both=bool(uni == 0),
                false_white_on_empty=float(B.sum() / max(n, 1)) if A.sum() == 0 else None)


def _rec(value, unit, valid_count, valid_duration, status="completed", reason=""):
    return dict(value=value, unit=unit, valid_count=int(valid_count), valid_duration=valid_duration, status=status,
                reason=reason)


def evaluate(clip, wT, ledger=None):
    """All metrics of one pattern for model whiteness wT [n, Mo, Mo] on the clip's physical time axis."""
    res = clip["res"]
    t = res["t"]
    disk = res["disk"].astype(bool)
    dt = np.gradient(t) if len(t) > 1 else np.ones(1)
    frames = []
    for j in np.flatnonzero(clip["shape_ok"]):
        m = shape_metrics(res["white"][j].astype(np.float32), res["valid"][j], wT[j], disk)
        m.update(t=round(float(t[j]), 3), w=float(dt[j]))
        frames.append(m)

    def tavg(key):
        xs = [(f[key], f["w"]) for f in frames if f[key] is not None]
        if not xs:
            return _rec(None, "-", 0, 0.0, "not_applicable", "no frame with a defined value")
        v = np.array(xs)
        return _rec(float((v[:, 0] * v[:, 1]).sum() / v[:, 1].sum()), "-", len(xs), round(float(v[:, 1].sum()), 2))

    out = dict(name=clip["name"], file_id=FILE_ID.get(clip["name"]), frames=frames)
    out["coverage_mae"] = tavg("mae")
    out["iou"] = tavg("iou")
    out["dice"] = tavg("dice")
    out["boundary_D"] = tavg("bnd_D")
    out["boundary_D"]["unit"] = "D"
    # final frame
    jf = len(t) - 1
    out["final"] = shape_metrics(res["white"][jf].astype(np.float32), res["valid"][jf], wT[jf], disk)
    # observables with the same code
    mo = VE.compute_observables(model_res(clip, wT), clip["script"])
    vo = clip["obs"]
    out["events"] = event_metrics(clip, vo, mo)
    out["profiles"] = profile_metrics(clip, res, wT)
    if ledger is not None:
        out["ledger"] = ledger
    out["loss"] = loss_of(out)
    return out


def event_metrics(clip, vo, mo):
    """First-white delays per onset, push-block displacement / gap, cut tip and stem width (physical units)."""
    ev = {}
    skip = clip["name"] in SKIP_FIRST_ONSET
    dl = []
    for k, (a, b) in enumerate(zip(vo["first_white"], mo["first_white"])):
        if k == 0 and skip:
            continue
        if a["delay_s"] is None:
            continue
        err = None if b["delay_s"] is None else b["delay_s"] - a["delay_s"]
        dl.append(dict(t_onset=a["t_onset"], video_s=a["delay_s"], model_s=b["delay_s"], err_s=err))
    errs = [abs(x["err_s"]) if x["err_s"] is not None else 2.0 for x in dl]     # no model white = 2 s error
    ev["first_white"] = dict(onsets=dl, **_rec(float(np.mean(errs)) if errs else None, "s", len(errs), None,
                                                "completed" if errs else "not_applicable",
                                                "" if errs else "no usable onset"))
    pb = []
    for a, b in zip(vo["push_blocks"], mo["push_blocks"]):
        for ba in a["blocks"]:
            bb = min(b["blocks"], key=lambda x: np.hypot(*(np.array(x["centroid0_T"]) - ba["centroid0_T"])), default=None)
            if bb is None or not ba["series"] or not bb["series"]:
                continue
            sa, sb = ba["series"][-1], bb["series"][-1]
            pb.append(dict(t_onset=a["t_onset"], near_disp_video_cm=sa.get("d_near_cm"), near_disp_model_cm=sb.get("d_near_cm"),
                           far_disp_video_cm=sa.get("d_far_cm"), far_disp_model_cm=sb.get("d_far_cm"),
                           gap_video_cm=sa.get("gap_to_other_white_cm"), gap_model_cm=sb.get("gap_to_other_white_cm")))
    ev["push_blocks"] = pb
    cu = []
    for a in vo["cut"]:
        b = next((x for x in mo["cut"] if x["at"] == a["at"] and abs(x["stage_t0"] - a["stage_t0"]) < 1e-6), None)
        if b is None:
            continue
        cu.append(dict(at=a["at"], t=a["t"], tip_r_video_D=a.get("tip_r_D"), tip_r_model_D=b.get("tip_r_D"),
                       stem_width_video_D=a.get("stem_width_D"), stem_width_model_D=b.get("stem_width_D")))
    ev["cut"] = cu
    return ev


def profile_metrics(clip, res, wT):
    """Fine lines along the pre-registered profiles, at the end of every wiggle / retreat / push / low_pour stage and at
    the final frame: number of white runs, pitch, brown-line width (video vs model, same axis_runs code)."""
    t = res["t"]
    disk = res["disk"].astype(bool)
    Wv, obs = VE.filled_white(res["white"].astype(np.float32), res["valid"] & disk[None], t)
    Wm, _ = VE.filled_white(wT.astype(np.float32), res["valid"] & disk[None], t)
    times = [("final", float(t[-1]))] + [(f"{s['kind']}@{s['t1']:.2f}", float(s["t1"])) for s in clip["script"]["stages"]
                                         if s["kind"] in ("wiggle", "retreat", "push")]
    out = []
    for prof in clip["profiles"]:
        for lab, tq in times:
            j = min(int(np.searchsorted(t, tq - 1e-6)), len(t) - 1)
            a = VE.axis_runs(Wv[j], obs[j], prof["p0_T"], prof["p1_T"])
            b = VE.axis_runs(Wm[j], obs[j], prof["p0_T"], prof["p1_T"])
            if a is None or b is None:
                continue
            out.append(dict(profile=prof["id"], at=lab, t=round(float(t[j]), 3),
                            n_white_video=a.get("n_white", 0), n_white_model=b.get("n_white", 0),
                            pitch_video_D=a.get("pitch_D"), pitch_model_D=b.get("pitch_D"),
                            brown_line_video_D=a.get("brown_line_D"), brown_line_model_D=b.get("brown_line_D")))
    return out


def loss_of(ev):
    """Pre-registered calibration loss of one pattern (see module docstring)."""
    d = ev["dice"]["value"]; b = ev["boundary_D"]["value"]
    L = (1.0 - d if d is not None else 1.0) + (b if b is not None else EMPTY_BND) / BND_SCALE
    fw = ev["events"]["first_white"]["value"]
    if fw is not None:
        L += fw / DELAY_SCALE
    pr = [p for p in ev["profiles"] if p["at"] != "final"]
    if pr:
        L += float(np.mean([abs(p["n_white_model"] - p["n_white_video"]) / max(p["n_white_video"], 1) for p in pr]))
    return float(L)


# =========================================================================== drivers
def sample_times(clip):
    return clip["res"]["t"]


def simulate_clip(model, params, clip, N=128):
    r = model.run(clip["script"], params, N, sample_times(clip))
    wT = observe_model(r["c"], clip["M"])
    return wT, r.get("ledger")


def evaluate_model(model, params, names, N=128):
    out = {}
    for n in names:
        clip = load_clip(n)
        wT, led = simulate_clip(model, params, clip, N)
        out[n] = evaluate(clip, wT, led)
    return out


def summary(evals):
    """Per-pattern table + equal-weight means (5.4: each pattern listed, then pooled)."""
    rows = {}
    for n, e in evals.items():
        rows[n] = dict(loss=round(e["loss"], 4), iou=e["iou"]["value"], dice=e["dice"]["value"],
                       mae=e["coverage_mae"]["value"], boundary_D=e["boundary_D"]["value"],
                       first_white_err_s=e["events"]["first_white"]["value"], final_iou=e["final"]["iou"],
                       final_boundary_D=e["final"]["bnd_D"])
    keys = ["loss", "iou", "dice", "mae", "boundary_D", "first_white_err_s", "final_iou", "final_boundary_D"]
    mean = {k: (float(np.mean([r[k] for r in rows.values() if r[k] is not None]))
                if any(r[k] is not None for r in rows.values()) else None) for k in keys}
    return dict(patterns=rows, mean=mean)


# =========================================================================== self-test models
class VideoModel:
    """The video itself as a SHARP model field (w > THR after the causal fill of occluded pixels, nearest frame, resampled
    to the model grid, M frame): the round trip through the observation operator gives the resolution floor of the
    metrics at grid N."""
    space = {}

    def __init__(self, name):
        self.clip = load_clip(name)
        res = self.clip["res"]
        disk = res["disk"].astype(bool)
        Wf, obs = VE.filled_white(res["white"].astype(np.float32), res["valid"] & disk[None], res["t"], hold=1e9)
        self.sharp = ((Wf > THR) & obs).astype(np.float32)

    def run(self, script, params, N, t_out):
        res = self.clip["res"]
        Mo = self.clip["M"]
        idx = [int(np.argmin(np.abs(res["t"] - t))) for t in t_out]
        xi = (np.arange(N) + 0.5) / N * Mo - 0.5
        I, J = np.meshgrid(xi, xi, indexing="ij")
        c = np.stack([(ndi.map_coordinates(self.sharp[i], [I, J], order=1, mode="nearest") > 0.5)[::-1, :]
                      for i in idx]).astype(np.float32)
        return dict(c=c, ledger=None)


class BlankModel:
    space = {}

    def run(self, script, params, N, t_out):
        return dict(c=np.zeros((len(t_out), N, N), np.float32), ledger=None)


def selftest(names=VE.PATTERNS, N=128):
    """Harness sanity check: the video as a sharp model (resolution floor at grid N) and an all-brown model."""
    res = {}
    for n in names:
        clip = load_clip(n)
        ev_v = evaluate(clip, *simulate_clip(VideoModel(n), {}, clip, N))
        ev_b = evaluate(clip, *simulate_clip(BlankModel(), {}, clip, N))
        res[n] = dict(video=summary({n: ev_v})["patterns"][n], blank=summary({n: ev_b})["patterns"][n],
                      n_frames=len(clip["res"]["t"]), n_shape=int(clip["shape_ok"].sum()),
                      T_phys=round(float(clip["res"]["t"][-1]), 2), cuts=clip["cuts"])
        print(n, json.dumps(res[n]), flush=True)
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        args = sys.argv[2:]
        N = int(args.pop(0)[2:]) if args and args[0].startswith("N=") else 128
        out = selftest(args or VE.PATTERNS, N)
        os.makedirs(os.path.join(HERE, "runs", "calib"), exist_ok=True)
        json.dump(out, open(os.path.join(HERE, "runs", "calib", f"selftest_N{N}.json"), "w"), indent=1)


# =========================================================================== figures
def contact_sheet(clip, wT, path, n=10, title=""):
    """Video whiteness (top) vs model whiteness through the same operator (bottom), T frame, invalid pixels red."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    res = clip["res"]
    t = res["t"]
    disk = res["disk"].astype(bool)
    idx = np.unique(np.linspace(0, len(t) - 1, n).round().astype(int))
    fig, axes = plt.subplots(2, len(idx), figsize=(1.9 * len(idx), 4.2), dpi=100, squeeze=False)
    for col, j in enumerate(idx):
        for row, w in enumerate((res["white"][j].astype(float), wT[j].astype(float))):
            img = np.stack([w, w, w], -1) * 0.85 + 0.08
            img[~res["valid"][j]] = [0.7, 0.3, 0.3]
            img[~disk] = 1.0
            ax = axes[row, col]
            ax.imshow(img, origin="lower")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{t[j]:.1f}s" + ("" if clip["shape_ok"][j] else " (q0)"), fontsize=7)
    fig.suptitle(f"{clip['name']}: video (top) / model (bottom), physical time; {title}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
