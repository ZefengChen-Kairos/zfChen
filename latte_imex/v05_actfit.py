"""Per-pattern ACTION fit of the V0.5 model with the physics FIXED: closure FSp2 and every inlet formula of
`v05_adapter.V05Model` unchanged; for each of the six clips SEPARATELY the barista's action read off that clip (the
physical-time stage script) is adjusted to that clip's own video, inside the physical envelope.  CMA-ES through
`calib.calibrate`, with the cache / timeout / resume wrapper of `v05_calibrate` (`v05_calibrate.patched`, inner worker
`_weval_act` = calib's worker with the loss L_act below).

    python -m latte_imex.v05_actfit prereg   [--path P]                           # write the pre-registered setup
    python -m latte_imex.v05_actfit run      --out DIR [--patterns a,b] [--procs 4]  # resumable, DIR/<pattern>/calib.json
    python -m latte_imex.v05_actfit smoke    --out DIR [--patterns heart,swan]      # 1 generation, popsize 4, N = 64
    python -m latte_imex.v05_actfit status   --out DIR
    python -m latte_imex.v05_actfit evaluate --out DIR [--N 128 --N_fig 256]       # SCRIPT / FIT / PROJECTED
    python -m latte_imex.v05_actfit timing   --out DIR [--N 96]                    # one forward call per pattern (init)
    python -m latte_imex.v05_actfit test                                           # unit tests

PRE-REGISTERED SETUP (`PREREG`, `PREREG_TEXT`; written by `prereg` before any fitting run, checked by `run`):
  physics   v05_adapter.FSP2 closure (cp, beta, nu, D, kappa_Q, kappa_c, kappa_t, kappa_r, B_dep, p_dep NOT fitted);
            inlet of V05Model unchanged: spout lag 0.08 s, Q clip 1-32 mL/s while pouring, U_perp = sqrt(0.35^2 + 2 g z),
            continuity footprint, no pulsation.
  action    per pattern, for every stage kind K in KINDS that occurs in the pattern's physical script with nonzero flow:
            z_K (multiplier on z0, z1 of every stage of kind K) and q_K (multiplier on flow0, flow1), bounds [1/3, 3],
            log, init 1; u_exit [0.10, 0.27] m/s, lin, init 0.15; dx, dy = rigid offset of the whole foot path (p0, p1,
            Bezier control points of every stage, dry ones included) [-0.03, 0.03] D, lin, init 0.  Dry stages
            (reposition / stop) keep their heights and flows.  After the multipliers the heights of the scaled stages
            are clipped to Z_CLIP = [0.003, 0.10] m (counted); flows are clipped by the adapter (counted from the inlet
            log: pour time with the 1 mL/s floor / the 32 mL/s cap active).
  box       VIDEO-CONSISTENT box = every multiplier in [0.5, 1.5] (the scripts' stated +-50 %), offsets within
            +-0.03 D.  The fit may leave it (reported); PROJECTED = FIT with the multipliers clipped into the box.
  loss      L_act = <1 - Dice>_t + <d_bnd>_t / BND_SCALE + P   (calib.evaluate's shape-frame averages),
            P = mean over {stage-end profiles exactly as calib.profile_metrics chooses them (ends of wiggle / retreat /
            push stages)} + {final frame} of |peaks_model - peaks_video| / max(peaks_video, 1); peaks =
            v05_predict.npeaks (prominence >= 0.10) on the pre-registered B -> D profile; stage ends: video and model
            fields both through VE.filled_white(., valid & disk, t) and read with the fill's observed mask (as
            v05calib/diag/stage_peaks.py); final frame: v05_predict.final_peaks (raw video, valid & disk).  No
            first-white term.  calib.loss_of(ev, delay=False) is still computed and stored as 'loss_calib'.
  CMA-ES    calib.calibrate per pattern: N = 96, popsize 12, sigma0 0.25 (unit box), 20 generations, one initial point
            (the script as read: multipliers 1, u_exit 0.15, offsets 0; evaluated once as the reference, not a
            candidate), seed 1.  Best = lowest L_act over all CMA samples.  Failures (exception, ledger ok False, wall
            > 1200 s) score +inf (1e6 for CMA) and are counted.  Order heart, push_heart, layered_heart, tulip, leaf,
            swan; nothing shared between patterns (separate calib.json; the cache key holds the pattern's spec).
  evaluate  SCRIPT (= zero tuning FSp2), FIT, PROJECTED: N = 128 (calib.evaluate + L_act + loss_of + final IoU + final
            peaks) and N = 256 (final frame, figures).

INTERPRETATION NOTES (fixed with the setup): the height clip acts only on the scaled stages (dry stages are "left as
they are"); a stage of a scaled kind with zero flow (layered_heart's third wiggle) is scaled like the others of its
kind (its z matters only for flow leaking from the spout lag); the dry gaps that `stages_to_moves` inserts between
stages keep their rule z = max(previous z1, 3 cm), so they follow the scaled z1; the tulip pre-roll stage is
transformed too but V05Model drops it (V0.5 has no first-white delay).

CACHE / RESUME: as v05_calibrate (DIR/cache/<sha1>.json keyed by spec (with the pattern), parameter reprs, pattern, N;
fixed-seed cma replays finished generations from the cache).  A pattern whose calib.json is complete with the same
setup is skipped.  Bookkeeping: DIR/run_setup.json, DIR/run_meta.json (v05_calibrate.Meta, tags = patterns),
DIR/<pattern>.log.
"""
import argparse
import copy
import glob
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

from . import calib as CB
from . import video_extract as VE
from . import v05_predict as VP
from .v05_adapter import CLOSURE_SETS, FSP2, LOGCOLS, ML, Q_MAX, Q_MIN, U_EXIT, V05Model

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("low_pour", "wiggle", "retreat", "push", "lift", "high_cut")
DRY_KINDS = ("reposition", "stop")
ORDER = ("heart", "push_heart", "layered_heart", "tulip", "leaf", "swan")
MULT = (1.0 / 3.0, 3.0, "log")
U_EXIT_RANGE = (0.10, 0.27, "lin")
OFFSET = (-0.03, 0.03, "lin")
Z_CLIP = (0.003, 0.10)               # m
BOX = (0.5, 1.5)                     # video-consistent multiplier range (+-50 %)
STAGE_END_KINDS = ("wiggle", "retreat", "push")   # calib.profile_metrics
TIMEOUT_S = 1200
AT_BOUND = 0.02
INNER = "latte_imex.v05_actfit:_weval_act"
PREREG_PATH = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review/v05actfit/prereg.json"
SETS = ("SCRIPT", "FIT", "PROJECTED")
CN = VP.CN

PREREG_TEXT = """PRE-REGISTERED SETUP (write it verbatim into /tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review/v05actfit/prereg.json BEFORE any fitting run; do not change it after seeing results):
- Physics fixed: v05_adapter.FSP2 closure, every inlet formula of V05Model unchanged (spout lag 0.08 s, Q clip 1-32 mL/s, U_perp = sqrt(0.35^2 + 2 g z), continuity footprint, no pulsation). Not fitted: cp, beta, nu, D, kappa_Q, kappa_c, kappa_t, kappa_r, B_dep, p_dep.
- Action parameters, per pattern (fitted separately on each pattern's own video):
  * for every stage kind K in {low_pour, wiggle, retreat, push, lift, high_cut} that occurs in the pattern's physical script with nonzero flow: z_K = multiplier on z0 and z1 of every stage of kind K, q_K = multiplier on flow0 and flow1 of every stage of kind K; bounds [1/3, 3], log scale, init 1. Stages of kind reposition / stop (dry) are left as they are.
  * u_exit (exit horizontal speed along the spout direction) [0.10, 0.27] m/s, linear, init 0.15.
  * dx, dy: rigid offset of the whole foot path (p0, p1 and Bezier control points of every stage) [-0.03, +0.03] D each, linear, init 0 (= the foot-picking error).
  * after the multipliers, heights are clipped to [0.003, 0.10] m; flows are clipped by the adapter (1-32 mL/s while pouring). Count how often each clip is active and report it.
  * the VIDEO-CONSISTENT box = every multiplier in [0.5, 1.5] (the scripts' stated +-50 %), offsets within +-0.03 D. The fit may leave it; that is reported, and a PROJECTED variant (multipliers clipped into [0.5, 1.5], everything else as fitted) is evaluated too.
- Fitted loss (per pattern, minimised): L_act = <1 - Dice>_t + <d_bnd>_t / BND_SCALE + P, with P = mean over {stage-end profiles exactly as calib.profile_metrics chooses them (ends of wiggle / retreat / push stages)} plus {the final frame} of |peaks_model - peaks_video| / max(peaks_video, 1); peaks = v05_predict.npeaks (threshold-free maxima, prominence >= 0.10) on the pre-registered B -> D profile(s); stage-end video/model fields with the same fill and validity convention as /tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review/v05calib/diag/stage_peaks.py, final frame as v05_predict.final_peaks. No first-white term. This replaces the white-run count of calib.loss_of, which a noisy field can satisfy. The original calib.loss_of(ev, delay=False) is still computed and reported for every evaluation.
- CMA-ES: N = 96, popsize 12, sigma0 0.25 in the unit box, 20 generations, one initial point (the script as read off the video: all multipliers 1, u_exit 0.15, offsets 0; evaluated once as the reference), seed 1. Best = lowest L_act over all CMA samples. Failures (exception, ledger ok False, wall > 1200 s) score +inf and are counted. Patterns are fitted one after another in the order heart, push_heart, layered_heart, tulip, leaf, swan; nothing is shared between patterns.
- Evaluation of each pattern at N = 128 (all metrics of calib.evaluate + L_act + loss_of + final IoU + final peaks) and N = 256 (final frame, figures) for: SCRIPT (zero tuning, FSp2), FIT (best), PROJECTED (best clipped into the video-consistent box)."""

PREREG = dict(
    date="2026-10-13",
    physics=dict(closure="FSp2", closure_values=FSP2,
                 not_fitted=["cp", "beta", "nu", "D", "kappa_Q", "kappa_c", "kappa_t", "kappa_r", "B_dep", "p_dep"],
                 inlet="v05_adapter.V05Model unchanged: spout lag 0.08 s, Q clip 1-32 mL/s while pouring, U_perp = "
                       "sqrt(0.35^2 + 2 g z), continuity footprint, no pulsation"),
    action=dict(kinds=list(KINDS), dry_kinds_untouched=list(DRY_KINDS),
                multiplier=dict(keys="z_K (z0, z1), q_K (flow0, flow1)", bounds=[MULT[0], MULT[1]], scale="log",
                                init=1.0, kinds="kinds present in the pattern's physical script with nonzero flow"),
                u_exit=dict(bounds=[U_EXIT_RANGE[0], U_EXIT_RANGE[1]], scale="lin", init=0.15, unit="m/s"),
                offsets=dict(keys=["dx", "dy"], bounds=[OFFSET[0], OFFSET[1]], scale="lin", init=0.0, unit="D",
                             applies_to="p0, p1 and Bezier control points of every stage (rigid shift of the path)"),
                z_clip_m=list(Z_CLIP), z_clip_applies_to="z0, z1 of the scaled stages, after the multipliers",
                flow_clip="adapter: 1 mL/s floor and 32 mL/s cap while pouring (counted from the inlet log)",
                video_consistent_box=dict(multipliers=list(BOX), offsets=[OFFSET[0], OFFSET[1]]),
                projected="FIT with every multiplier clipped into [0.5, 1.5], u_exit / dx / dy as fitted"),
    loss=dict(name="L_act", formula="<1 - Dice>_t + <d_bnd>_t / BND_SCALE + P",
              BND_SCALE=CB.BND_SCALE, EMPTY_BND=CB.EMPTY_BND, PROM=VP.PROM,
              P="mean over stage-end profiles (calib.profile_metrics selection: ends of wiggle / retreat / push "
                "stages) + the final frame of |peaks_model - peaks_video| / max(peaks_video, 1)",
              stage_end_fields="VE.filled_white(white, valid & disk, t) for video and model, read with its observed "
                               "mask (stage_peaks.py convention)",
              final_frame="v05_predict.final_peaks (raw video whiteness, valid & disk)",
              first_white="not in the loss", reported="calib.loss_of(ev, delay=False) as loss_calib"),
    cma=dict(N=96, popsize=12, sigma0=0.25, gens=20, seed=1,
             init="the script as read off the video: multipliers 1, u_exit 0.15, offsets 0 (reference, evaluated once)"),
    selection="best = lowest L_act over all CMA samples of the pattern (calib.calibrate rec['best'])",
    failures=f"exception / ledger ok=False / wall > {TIMEOUT_S} s -> +inf (1e6 for CMA), counted",
    order=list(ORDER),
    evaluate=dict(sets=list(SETS), N=128, N_fig=256,
                  N128="calib.evaluate(delay_in_loss=False) + L_act + loss_of + final IoU + final peaks",
                  N256="final frame only (shape metrics, peaks, figures)"),
    interpretation=[
        "the height clip acts only on the scaled stages; dry stages (reposition / stop) keep every value",
        "a stage of a scaled kind with zero flow is scaled like the other stages of its kind",
        "dry gaps inserted by stages_to_moves keep z = max(previous z1, 3 cm), i.e. follow the scaled z1",
        "the tulip pre-roll stage is transformed too and then dropped by V05Model (preroll='drop')",
        "stage-end frame index j = min(searchsorted(t, t1 - 1e-6), n - 1) exactly as calib.profile_metrics",
    ],
)


# =========================================================================== action space
def kinds_of(name_or_script):
    """Stage kinds of KINDS that occur with nonzero flow in the pattern's physical script (canonical order)."""
    sc = CB.physical_script(name_or_script) if isinstance(name_or_script, str) else name_or_script
    have = {s["kind"] for s in sc["stages"] if max(float(s["flow0"]), float(s["flow1"])) > 0}
    return tuple(k for k in KINDS if k in have)


def action_space(kinds):
    sp = {}
    for k in kinds:
        sp[f"z_{k}"] = MULT
        sp[f"q_{k}"] = MULT
    sp["u_exit"] = U_EXIT_RANGE
    sp["dx"] = OFFSET
    sp["dy"] = OFFSET
    return sp


def x0_of(space):
    """The script as read off the video: multipliers 1, u_exit 0.15, offsets 0."""
    return {k: (1.0 if k[:2] in ("z_", "q_") else (U_EXIT if k == "u_exit" else 0.0)) for k in space}


def is_mult(k):
    return k[:2] in ("z_", "q_")


def project(params):
    """PROJECTED variant: every multiplier clipped into the video-consistent box, everything else unchanged."""
    return {k: (min(max(float(v), BOX[0]), BOX[1]) if is_mult(k) else float(v)) for k, v in params.items()}


def outside_box(params, tol=1e-12):
    """Parameters outside the video-consistent box: [(key, value)]."""
    out = []
    for k, v in params.items():
        if is_mult(k) and not (BOX[0] - tol <= v <= BOX[1] + tol):
            out.append((k, float(v)))
        if k in ("dx", "dy") and abs(v) > OFFSET[1] + tol:
            out.append((k, float(v)))
    return out


def stage_counts(name):
    """Stages per kind of the poured script (pre-roll stage excluded) and the ranges of z [m] and flow [x Q_ref] of
    the scaled kinds as read off the video."""
    sc = CB.physical_script(name, preroll=0)
    kinds = kinds_of(sc)
    c = {}
    for s in sc["stages"]:
        c[s["kind"]] = c.get(s["kind"], 0) + 1
    zs = [s[e] for s in sc["stages"] if s["kind"] in kinds for e in ("z0", "z1")]
    fs = [s[e] for s in sc["stages"] if s["kind"] in kinds for e in ("flow0", "flow1")]
    return dict(stages_by_kind=c, z_range_m=[min(zs), max(zs)], flow_range_rel=[min(fs), max(fs)])


# =========================================================================== script transform
def act_script(script, params, kinds=None):
    """Physical script + action params -> (new physical script, info).  The input is not modified.

    kinds   the scaled stage kinds (default: those with a z_K / q_K key in params)
    z0, z1 of every stage of a scaled kind K  x z_K, then clipped to Z_CLIP (counted in info['height_clip']);
    flow0, flow1 of every stage of kind K     x q_K (the adapter clips the received flow later);
    p0, p1, Bezier control points of EVERY stage + (dx, dy) [D]; everything else unchanged."""
    if kinds is None:
        kinds = tuple(k for k in KINDS if f"z_{k}" in params or f"q_{k}" in params)
    sc = copy.deepcopy(script)
    dx, dy = float(params.get("dx", 0.0)), float(params.get("dy", 0.0))
    hc = dict(low=0, high=0, n_scaled_ends=0, entries=[])
    n_scaled = {}
    for i, s in enumerate(sc["stages"]):
        k = s["kind"]
        if k in kinds:
            mz, mq = float(params.get(f"z_{k}", 1.0)), float(params.get(f"q_{k}", 1.0))
            n_scaled[k] = n_scaled.get(k, 0) + 1
            for e in ("z0", "z1"):
                raw = s[e] * mz
                v = min(max(raw, Z_CLIP[0]), Z_CLIP[1])
                hc["n_scaled_ends"] += 1
                if v != raw:
                    hc["low" if raw < Z_CLIP[0] else "high"] += 1
                    hc["entries"].append(dict(stage=i, kind=k, end=e, raw_m=float(raw), clipped_m=float(v)))
                s[e] = v
            s["flow0"] = s["flow0"] * mq
            s["flow1"] = s["flow1"] * mq
        s["p0"] = [s["p0"][0] + dx, s["p0"][1] + dy]
        s["p1"] = [s["p1"][0] + dx, s["p1"][1] + dy]
        if s.get("bezier"):
            s["bezier"] = [[c[0] + dx, c[1] + dy] for c in s["bezier"]]
    return sc, dict(height_clip=hc, n_scaled_stages=n_scaled, offset_D=[dx, dy])


def flow_clips(log, cols=LOGCOLS):
    """How often the adapter's flow clip is active, from the inlet log of a run (one row per control frame with
    pouring or flow; weight = the frame length fdt).  floor = pouring (Q_w > 0) with the 1 mL/s floor active (Q = 1
    mL/s while the lag is below it); cap = Q at the 32 mL/s cap (pouring, and separately while the lag leaks after a
    stage).  Seconds, frame counts, fractions of the pouring time, maximum wanted flow, wanted volume above the cap."""
    if log is None or len(log) == 0:
        return dict(pour_s=0.0, floor_s=0.0, cap_s=0.0, cap_leak_s=0.0, floor_frames=0, cap_frames=0,
                    floor_frac=None, cap_frac=None, max_wanted_mLs=None, wanted_above_cap_mL=0.0)
    c = {k: i for i, k in enumerate(cols)}
    lg = np.asarray(log, float)
    qw, q, dt = lg[:, c["Q_w_mLs"]], lg[:, c["Q_mLs"]], lg[:, c["fdt"]]
    lo, hi = Q_MIN / ML, Q_MAX / ML
    pour = qw > 0
    floor = pour & (np.abs(q - lo) <= 1e-9 * lo)
    cap = np.abs(q - hi) <= 1e-9 * hi
    ps = float(dt[pour].sum())
    return dict(pour_s=ps, floor_s=float(dt[floor].sum()), cap_s=float(dt[cap & pour].sum()),
                cap_leak_s=float(dt[cap & ~pour].sum()), floor_frames=int(floor.sum()), cap_frames=int((cap & pour).sum()),
                floor_frac=float(dt[floor].sum() / ps) if ps > 0 else None,
                cap_frac=float(dt[cap & pour].sum() / ps) if ps > 0 else None,
                max_wanted_mLs=float(qw.max()), wanted_above_cap_mL=float((np.maximum(qw - hi, 0.0) * dt).sum()))


# =========================================================================== model
class V05ActModel:
    """V0.5 with a FIXED closure (default FSp2) and the action space of ONE pattern (see module docstring).

    .space = {z_K, q_K for the pattern's kinds, u_exit, dx, dy};  .run(script, params, N, t_out) transforms the
    physical script with `act_script` and runs v05_adapter.V05Model on it with u_exit from params.  The ledger gets
    'act' = height-clip and flow-clip counts, the offset, the scaled stage counts and the box check."""

    def __init__(self, pattern, base="FSp2", numerics=None):
        self.pattern = pattern
        self.base_name = base if isinstance(base, str) else "custom"
        self.inner = V05Model(base=dict(CLOSURE_SETS[base]) if isinstance(base, str) else dict(base),
                              numerics=numerics)
        self.kinds = kinds_of(pattern)
        self.space = action_space(self.kinds)

    def full(self, params):
        p = x0_of(self.space)
        p.update({k: float(v) for k, v in (params or {}).items()})
        return p

    def transform(self, script, params):
        return act_script(script, self.full(params), self.kinds)

    def run(self, script, params, N, t_out, keep_log=True):
        p = self.full(params)
        sc, info = act_script(script, p, self.kinds)
        out = self.inner.run(sc, {"u_exit": float(p["u_exit"])}, N, t_out, keep_log=True)
        info["flow_clip"] = flow_clips(out.get("inlet_log"), out.get("inlet_cols", LOGCOLS))
        info["outside_box"] = outside_box(p)
        info["params"] = p
        out["ledger"]["act"] = info
        if not keep_log:
            out.pop("inlet_log", None)
            out.pop("inlet_cols", None)
        return out


# =========================================================================== loss L_act
_VID = {}


def _video_filled(clip):
    res = clip["res"]
    key = (clip["name"], id(res["white"]))
    if key not in _VID:
        disk = res["disk"].astype(bool)
        _VID[key] = VE.filled_white(res["white"].astype(np.float32), res["valid"] & disk[None], res["t"])
    return _VID[key]


def stage_end_frames(clip):
    """[(label, j)]: the stage ends calib.profile_metrics uses (wiggle / retreat / push), same frame index rule."""
    t = clip["res"]["t"]
    return [(f"{s['kind']}@{s['t1']:.2f}", min(int(np.searchsorted(t, float(s["t1"]) - 1e-6)), len(t) - 1))
            for s in clip["script"]["stages"] if s["kind"] in STAGE_END_KINDS]


def structure_term(clip, wT):
    """P of L_act: threshold-free peak-count error on the B -> D profile at the stage ends (both fields through the
    causal fill, observed mask) and at the final frame (v05_predict.final_peaks).  Returns dict(P, entries)."""
    res = clip["res"]
    t = res["t"]
    disk = res["disk"].astype(bool)
    Wv, obs = _video_filled(clip)
    Wm, _ = VE.filled_white(np.asarray(wT, np.float32), res["valid"] & disk[None], t)
    ents = []
    for prof in clip["profiles"]:
        for lab, j in stage_end_frames(clip):
            if (VE.axis_runs(Wv[j], obs[j], prof["p0_T"], prof["p1_T"]) is None
                    or VE.axis_runs(Wm[j], obs[j], prof["p0_T"], prof["p1_T"]) is None):
                continue                                  # skipped by calib.profile_metrics as well
            pv = VP.npeaks(Wv[j], obs[j], prof["p0_T"], prof["p1_T"])
            pm = VP.npeaks(Wm[j], obs[j], prof["p0_T"], prof["p1_T"])
            ents.append(dict(profile=prof["id"], at=lab, t=round(float(t[j]), 3), j=int(j), peaks_video=pv,
                             peaks_model=pm, err=abs(pm - pv) / max(pv, 1)))
    pv, pm = VP.final_peaks(clip, np.asarray(wT[-1], np.float32))
    ents.append(dict(profile=clip["profiles"][0]["id"], at="final", t=round(float(t[-1]), 3), j=len(t) - 1,
                     peaks_video=int(pv), peaks_model=int(pm), err=abs(pm - pv) / max(pv, 1)))
    return dict(P=float(np.mean([e["err"] for e in ents])), entries=ents)


def act_loss(clip, wT, ev):
    """L_act of one evaluation (ev = calib.evaluate(clip, wT, ..., delay_in_loss=False)).  Returns the fields merged
    into the worker's record: loss (= L_act, what CMA minimises), loss_calib (= calib.loss_of(ev, delay=False)),
    L_act (terms) and structure (P entries)."""
    d = ev["dice"]["value"]
    b = ev["boundary_D"]["value"]
    st = structure_term(clip, wT)
    omd = 1.0 - d if d is not None else 1.0
    bt = (b if b is not None else CB.EMPTY_BND) / CB.BND_SCALE
    L = omd + bt + st["P"]
    return dict(loss=float(L), loss_calib=float(CB.loss_of(ev, delay=False)),
                L_act=dict(value=float(L), one_minus_dice=float(omd), bnd_term=float(bt), P=st["P"],
                           bnd_mm=(b * VE.D_CUP * 1000 if b is not None else None)),
                structure=st)


_MODELS = {}


def _model_for(name):
    if name not in _MODELS:
        _MODELS[name] = CB.make_model(spec_of(name))
    return _MODELS[name]


def _weval_act(args):
    """calib._weval with L_act as 'loss' (inner worker of v05_calibrate's cached / time-limited wrapper)."""
    params, name, N = args
    clip = CB.load_clip(name)
    model = CB._W.get("model")
    if getattr(model, "pattern", None) != name:
        model = _model_for(name)
    try:
        wT, led = CB.simulate_clip(model, params, clip, N)
        if led is not None and not led.get("ok", True):
            return name, dict(name=name, loss=float("inf"), loss_calib=float("inf"), ledger=led, failed=True)
        ev = CB.evaluate(clip, wT, led, delay_in_loss=False)
        out = CB.compact(ev)
        out.update(act_loss(clip, wT, ev))
        return name, out
    except Exception as e:                         # recorded as a failed forward call, scored +inf
        return name, dict(name=name, loss=float("inf"), loss_calib=float("inf"), failed=True,
                          error=f"{type(e).__name__}: {e}")


# =========================================================================== setup / prereg
def spec_of(name):
    return dict(kind="v05act", pattern=name, base="FSp2", loss_delay=False, loss="L_act")


def pattern_setup(name):
    m = V05ActModel(name)
    x0 = x0_of(m.space)
    return dict(kinds=list(m.kinds), keys=list(m.space), dimension=len(m.space),
                space={k: list(v) for k, v in m.space.items()}, x0=x0,
                x0_unit=[round(u, 6) for u in CB.encode(x0, m.space)], spec=spec_of(name), **stage_counts(name))


def prereg_dict():
    return dict(setup=json.loads(json.dumps(PREREG, default=float)),
                patterns={n: json.loads(json.dumps(pattern_setup(n), default=float)) for n in ORDER},
                text_verbatim=PREREG_TEXT)


def _jload(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    os.replace(tmp, path)


def _sha(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def provenance():
    import subprocess
    try:
        head = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = None
    files = ("v05_actfit.py", "v05_calibrate.py", "v05_predict.py", "calib.py", "v05_adapter.py", "solver.py",
             "kernels.py", "v3min_adapter.py", "video_extract.py")
    sha = {f: hashlib.sha1(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in files
           if os.path.exists(os.path.join(HERE, f))}
    import cma
    return dict(git_head=head, sha1=sha, cma=cma.__version__, numpy=np.__version__)


def write_prereg(path=PREREG_PATH, force=False):
    """Write the pre-registered setup (refuses to overwrite a different one unless force)."""
    d = prereg_dict()
    old = _jload(path)
    if old is not None and _sha({k: old.get(k) for k in d}) != _sha(d) and not force:
        raise SystemExit(f"{path} holds a different setup; refusing to overwrite (use --force only before any run)")
    if old is not None and _sha({k: old.get(k) for k in d}) == _sha(d):
        print(f"{path}: unchanged (sha {_sha(d)[:12]})")
        return old
    rec = dict(d, sha1=_sha(d), written=time.strftime("%Y-%m-%d %H:%M:%S"), code=provenance(),
               note="written before any fitting run; `run` refuses to start if the code's setup differs from this")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _jdump(rec, path)
    print(f"wrote {path} (sha {rec['sha1'][:12]})")
    return rec


def check_prereg(path=PREREG_PATH):
    d = prereg_dict()
    old = _jload(path)
    if old is None:
        raise SystemExit(f"no pre-registration at {path}: run `prereg` first")
    if _sha({k: old.get(k) for k in d}) != _sha(d):
        raise SystemExit(f"the code's setup differs from the pre-registration {path}; not starting")
    return old["sha1"]


# =========================================================================== run
def setup_matches(rec, name, x0, cfg):
    if rec is None:
        return False
    same = (rec.get("spec") == json.loads(json.dumps(spec_of(name))) and rec.get("names") == [name]
            and rec.get("N") == cfg["N"] and rec.get("gens") == cfg["gens"] and rec.get("popsize") == cfg["popsize"]
            and abs(rec.get("sigma0", -1) - cfg["sigma0"]) < 1e-12 and len(rec.get("inits", [])) == 1)
    if not same:
        return False
    if any(abs(rec["inits"][0]["x0"][k] - x0[k]) > 1e-12 for k in x0):
        return False
    return (rec.get("driver") or {}).get("seed", cfg["seed"]) == cfg["seed"]


def _summ(ev):
    """Short record of one cached evaluation (reference / best)."""
    if ev is None:
        return None
    la = ev.get("L_act") or {}
    led = ev.get("ledger") or {}
    act = led.get("act") or {}
    return dict(L_act=ev.get("loss"), loss_calib=ev.get("loss_calib"), failed=bool(ev.get("failed")),
                one_minus_dice=la.get("one_minus_dice"), bnd_term=la.get("bnd_term"), P=la.get("P"),
                iou=(ev.get("iou") or {}).get("value"), dice=(ev.get("dice") or {}).get("value"),
                final_iou=(ev.get("final") or {}).get("iou"),
                peaks=[(e["at"], e["peaks_video"], e["peaks_model"]) for e in (ev.get("structure") or {}).get("entries", [])],
                height_clip={k: (act.get("height_clip") or {}).get(k) for k in ("low", "high", "n_scaled_ends")},
                flow_clip=act.get("flow_clip"), chi=led.get("chi_flow_weighted"), V_received_mL=led.get("V_received_mL"),
                wall_s=led.get("wall_s"), error=ev.get("error") or led.get("error"))


def run(out, patterns=ORDER, gens=20, popsize=12, N=96, procs=4, sigma0=0.25, seed=1, label="run", force=False,
        prereg_path=PREREG_PATH):
    from multiprocessing import Pool
    from . import v05_calibrate as VC
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    pre_sha = check_prereg(prereg_path)
    cache = os.path.join(out, "cache")
    cfg = dict(gens=gens, popsize=popsize, N=N, sigma0=sigma0, seed=seed, label=label)
    sp = os.path.join(out, "run_setup.json")
    old = _jload(sp)
    if old is not None and old.get("run_config") != json.loads(json.dumps(cfg)) and not force:
        raise SystemExit(f"{sp} holds a different run_config ({old.get('run_config')}); use another --out or --force")
    setup = dict(prereg_path=os.path.abspath(prereg_path), prereg_sha1=pre_sha, run_config=cfg,
                 procs=procs, timeout_s=TIMEOUT_S, numba_threads=os.environ.get("NUMBA_NUM_THREADS"),
                 patterns=[n for n in ORDER if n in patterns], code=provenance(),
                 history=(old or {}).get("history", []) + [dict(start=time.strftime("%Y-%m-%d %H:%M:%S"), procs=procs,
                                                                argv=sys.argv)])
    _jdump(setup, sp)
    meta = VC.Meta(out, sys.argv)
    for name in [n for n in ORDER if n in patterns]:
        spec = spec_of(name)
        space = CB.make_model(spec).space
        x0 = x0_of(space)
        d = os.path.join(out, name)
        cj = os.path.join(d, "calib.json")
        logp = os.path.join(out, f"{name}.log")
        rec_old = _jload(cj)
        if rec_old is not None and "best" in rec_old and setup_matches(rec_old, name, x0, cfg):
            print(f"[{VC._now()}] {name}: complete with the same setup, skipped", flush=True)
            continue
        if rec_old is not None and not setup_matches(rec_old, name, x0, cfg) and not force:
            raise SystemExit(f"{cj} was written with a different setup; use another --out or --force")
        n0 = VC.n_cache(cache)
        seg = meta.segment(name, n0)
        t0 = time.time()
        state = dict(last=n0, tlast=t0)

        def log(line, name=name, seg=seg, t0=t0):
            nc = VC.n_cache(cache)
            new = nc - state["last"]
            dt = time.time() - state["tlast"]
            state.update(last=nc, tlast=time.time())
            s = (f"[{VC._now()} +{time.time() - t0:7.0f}s] {line} | computed {new} in {dt:.0f}s"
                 f"{' (replayed from cache)' if new == 0 else ''}, cache {nc}")
            print(s, flush=True)
            with open(logp, "a") as fh:
                fh.write(s + "\n")
            meta.touch(seg, nc)

        log(f"{name} start: session {meta.sid}, dim {len(space)} {list(space)}, N {N}, gens {gens}, popsize {popsize}, "
            f"sigma0 {sigma0}, seed {seed}, procs {procs}")
        with VC.patched(cache, name, inner=INNER):
            with Pool(1, initializer=VC._winit, initargs=(spec,)) as pool:
                ref = pool.map(VC._weval, [(x0, name, N)], chunksize=1)[0][1]
            r = _summ(ref)
            log(f"{name} x0 reference L_act {r['L_act']:.4f} (1-Dice {r['one_minus_dice']:.3f}, bnd {r['bnd_term']:.3f}, "
                f"P {r['P']:.3f}; loss_of {r['loss_calib']:.4f}) peaks {r['peaks']}"
                if not r["failed"] else f"{name} x0 reference FAILED {r['error']}")
            rec = CB.calibrate(spec, [name], d, [x0], gens=gens, popsize=popsize, sigma0=sigma0, N=N, procs=procs,
                               seed=seed, log=log)
        nc = VC.n_cache(cache)
        meta.touch(seg, nc, completed=True)
        segs = meta.d["tags"][name]["segments"]
        comp = [c for c in (_jload(f) for f in glob.glob(os.path.join(cache, "*.json"))) if c and c.get("tag") == name]
        best = rec["best"]["params"]
        rec["driver"] = dict(
            prereg_sha1=pre_sha, run_config=cfg, procs=procs, seed=seed, label=label, x0_reference=_summ(ref),
            best=_summ(rec["inits"][0]["best"]["evals"][name]), outside_box=outside_box(best),
            projected=project(best), unit_best={k: CB.to_unit(best[k], *space[k]) for k in space},
            wall_s=float(sum(s["end"] - s["start"] for s in segs)), segments=segs,
            computed_by_tag=len(comp), computed_failed=sum(bool(c["eval"].get("failed")) for c in comp),
            compute_s_by_tag=float(sum(c["wall_s"] for c in comp)),
            note="n_forward = forward calls requested by CMA-ES (cache hits included); the x0 reference is extra")
        _jdump(rec, cj)
        b = rec["driver"]["best"]
        log(f"{name} done: best L_act {rec['best']['loss']:.4f} (x0 {r['L_act']:.4f}), loss_of {b['loss_calib']:.4f}, "
            f"P {b['P']:.3f}, outside box {rec['driver']['outside_box']}, n_forward {rec['n_forward']}, "
            f"n_failed {rec['n_failed']}")
    meta.d["sessions"][meta.sid].update(end=time.time(), end_str=VC._now(), finished=True)
    meta.save()


# =========================================================================== status
def status(out):
    from . import v05_calibrate as VC
    out = os.path.abspath(out)
    print(f"cache entries: {VC.n_cache(os.path.join(out, 'cache'))}")
    meta = _jload(os.path.join(out, "run_meta.json")) or {}
    for name in ORDER:
        rec = _jload(os.path.join(out, name, "calib.json"))
        if rec is None:
            print(f"{name}: no calib.json")
            continue
        ini = rec["inits"][0]
        h = ini["history"]
        drv = rec.get("driver") or {}
        ref = (drv.get("x0_reference") or {}).get("L_act")
        w = sum(s["end"] - s["start"] for s in (meta.get("tags", {}).get(name) or {}).get("segments", []))
        print(f"{name}: {'COMPLETE' if 'best' in rec else 'running'}  dim {len(rec['space'])}  gens {len(h)}/{rec['gens']}"
              f"  best L_act {ini['best']['loss']:.4f} (gen {ini['best']['gen']})"
              + (f"  x0 {ref:.4f}" if ref is not None else "")
              + (f"  last min {h[-1]['loss_min']:.4f} med {h[-1]['loss_med']:.4f} sigma {h[-1]['sigma']:.3f}" if h else "")
              + f"  n_forward {rec['n_forward']} failed {rec['n_failed']}  wall {w / 60:.1f} min")
        if "best" in rec:
            print(f"   outside box: {drv.get('outside_box')}")


# =========================================================================== evaluate
def param_sets(out, name):
    rec = _jload(os.path.join(out, name, "calib.json"))
    if rec is None or "best" not in rec:
        return None
    space = {k: tuple(v) for k, v in rec["space"].items()}
    fit = {k: float(v) for k, v in rec["best"]["params"].items()}
    return dict(SCRIPT=x0_of(space), FIT=fit, PROJECTED=project(fit)), space, rec


def _job(task):
    kind, s, params, name, N, out = task
    t0 = time.time()
    model = CB.make_model(spec_of(name))
    clip = CB.load_clip(name)
    res = clip["res"]
    if kind == "eval":
        d = os.path.join(out, "eval", f"{s}_{name}_N{N}")
        if os.path.exists(os.path.join(d, "eval.json")):
            return f"{s} {name} N{N}: cached"
        os.makedirs(d, exist_ok=True)
        wT, led = CB.simulate_clip(model, params, clip, N)
        ev = CB.evaluate(clip, wT, led, delay_in_loss=False)
        ev.update(act_loss(clip, wT, ev))
        ev.update(set=s, params=params, N=N, wall_s=time.time() - t0, delay_in_loss=False,
                  final_peaks_video_model=list(VP.final_peaks(clip, wT[-1])), outside_box=outside_box(params))
        np.savez_compressed(os.path.join(d, "wT.npz"), t=res["t"], wT=wT.astype(np.float16))
        np.save(os.path.join(d, "final.npy"), wT[-1].astype(np.float32))
        _jdump(ev, os.path.join(d, "eval.json"))
        return (f"{s} {name} N{N}: ok {(led or {}).get('ok')} L_act {ev['loss']:.3f} loss_of {ev['loss_calib']:.3f} "
                f"IoU {ev['iou']['value']:.3f} final IoU {ev['final']['iou']:.3f} | {time.time() - t0:.0f} s")
    if kind == "final":
        d = os.path.join(out, "eval", f"{s}_{name}_N{N}_final")
        if os.path.exists(os.path.join(d, "final.json")):
            return f"{s} {name} N{N} final: cached"
        os.makedirs(d, exist_ok=True)
        r = model.run(clip["script"], params, N, res["t"][-1:], keep_log=False)
        wT = CB.observe_model(r["c"], clip["M"])
        disk = res["disk"].astype(bool)
        fin = CB.shape_metrics(res["white"][-1].astype(np.float32), res["valid"][-1], wT[0], disk)
        a, b = VP.final_lines(clip, wT[0])
        pv, pm = VP.final_peaks(clip, wT[0])
        np.save(os.path.join(d, "final.npy"), wT[0].astype(np.float32))
        _jdump(dict(set=s, name=name, N=N, params=params, final=fin, lines_video=a, lines_model=b,
                    peaks_video=pv, peaks_model=pm, ledger=r["ledger"], outside_box=outside_box(params),
                    wall_s=time.time() - t0), os.path.join(d, "final.json"))
        return f"{s} {name} N{N} final: ok {r['ledger'].get('ok')} IoU {fin['iou']:.3f} peaks {pv}/{pm} | " \
               f"{time.time() - t0:.0f} s"
    raise ValueError(kind)


def _pool_map(tasks, procs):
    from multiprocessing import get_context
    if procs <= 1:
        for t in tasks:
            print(time.strftime("%H:%M:%S"), _job(t), flush=True)
        return
    with get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(_job, tasks, chunksize=1):
            print(time.strftime("%H:%M:%S"), msg, flush=True)


def evaluate(out, patterns=ORDER, N=128, N_fig=256, procs=4):
    """SCRIPT / FIT / PROJECTED of every complete pattern: full evaluation at N, final frame at N_fig (PROJECTED is
    not recomputed when it equals FIT).  -> DIR/eval/..., DIR/evaluation.json, evaluation_tables.md, figures."""
    out = os.path.abspath(out)
    sets = {}
    for n in ORDER:
        if n in patterns:
            ps = param_sets(out, n)
            if ps is None:
                print(f"{n}: no complete calib.json, skipped", flush=True)
                continue
            sets[n] = ps
    same = {n: all(abs(ps[0]["PROJECTED"][k] - ps[0]["FIT"][k]) == 0 for k in ps[0]["FIT"]) for n, ps in sets.items()}
    jobs = []
    order = [n for n in ("swan", "tulip", "push_heart", "leaf", "heart", "layered_heart") if n in sets]   # long first
    for n in order:
        for s in SETS:
            if s == "PROJECTED" and same[n]:
                continue
            jobs.append(("eval", s, sets[n][0][s], n, N, out))
    if N_fig:
        for n in order:
            for s in SETS:
                if s == "PROJECTED" and same[n]:
                    continue
                jobs.append(("final", s, sets[n][0][s], n, N_fig, out))
    _jdump(dict(sets={n: ps[0] for n, ps in sets.items()}, projected_equals_fit=same, N=N, N_fig=N_fig,
                provenance=provenance(), date=time.strftime("%Y-%m-%d %H:%M:%S")),
           os.path.join(out, "evaluate_setup.json"))
    _pool_map(jobs, procs)
    rep = summarise(out, sets, same, N, N_fig)
    _jdump(rep, os.path.join(out, "evaluation.json"))
    open(os.path.join(out, "evaluation_tables.md"), "w").write(tables_md(rep) + "\n")
    if N_fig:
        fig_final(out, rep, N_fig)
        fig_lines(out, rep, N_fig)
    return rep


def _eval_dir(out, s, n, N, same, final=False):
    s_ = "FIT" if (s == "PROJECTED" and same) else s
    return os.path.join(out, "eval", f"{s_}_{n}_N{N}" + ("_final" if final else ""))


def summarise(out, sets, same, N, N_fig):
    rep = dict(N=N, N_fig=N_fig, patterns={})
    for n, (ps, space, rec) in sets.items():
        drv = rec.get("driver") or {}
        row = dict(space={k: list(v) for k, v in space.items()}, dimension=len(space), fit_N=rec["N"],
                   best_L_act_N96=rec["best"]["loss"], x0_L_act_N96=(drv.get("x0_reference") or {}).get("L_act"),
                   n_forward=rec["n_forward"], n_failed=rec["n_failed"], projected_equals_fit=same[n], sets={})
        for s in SETS:
            ev = _jload(os.path.join(_eval_dir(out, s, n, N, same[n]), "eval.json"))
            fi = _jload(os.path.join(_eval_dir(out, s, n, N_fig, same[n], True), "final.json")) if N_fig else None
            r = dict(params=ps[s], unit={k: CB.to_unit(ps[s][k], *space[k]) for k in space},
                     outside_box=outside_box(ps[s]))
            if ev is not None:
                act = (ev.get("ledger") or {}).get("act") or {}
                r.update(L_act=ev["L_act"], loss_calib=ev["loss_calib"], iou=ev["iou"]["value"],
                         dice=ev["dice"]["value"], boundary_mm=(ev["boundary_D"]["value"] or 0) * 80,
                         mae=ev["coverage_mae"]["value"], first_white_err_s=ev["events"]["first_white"]["value"],
                         final_iou=ev["final"]["iou"], final_bnd_mm=(ev["final"]["bnd_D"] or 0) * 80,
                         final_peaks=ev["final_peaks_video_model"],
                         peaks=[(e["at"], e["peaks_video"], e["peaks_model"]) for e in ev["structure"]["entries"]],
                         runs=[(p["at"], p["n_white_video"], p["n_white_model"]) for p in ev["profiles"]],
                         height_clip={k: act.get("height_clip", {}).get(k) for k in ("low", "high", "n_scaled_ends")},
                         flow_clip=act.get("flow_clip"), chi=ev["ledger"].get("chi_flow_weighted"),
                         V_received_mL=ev["ledger"].get("V_received_mL"), V_wanted_mL=ev["ledger"].get("V_wanted_mL"),
                         ok=ev["ledger"].get("ok"))
            if fi is not None:
                r["final_N_fig"] = dict(iou=fi["final"]["iou"], bnd_mm=(fi["final"]["bnd_D"] or 0) * 80,
                                        peaks_video=fi["peaks_video"], peaks_model=fi["peaks_model"],
                                        runs_video=VP._n_runs(fi["lines_video"])[0],
                                        runs_model=VP._n_runs(fi["lines_model"])[0], ok=fi["ledger"].get("ok"))
            row["sets"][s] = r
        rep["patterns"][n] = row
    return rep


SET_CN = dict(SCRIPT="照读脚本（零调参）", FIT="动作拟合", PROJECTED="投影回 ±50 % 盒")


def tables_md(rep):
    def f(v, fmt="{:.3f}"):
        return "–" if v is None else fmt.format(v)
    L = [f"### 每个图案：L_act 与各项（N = {rep['N']}），末帧（N = {rep['N']} 与 {rep['N_fig']}）", "",
         "| 图案 | 动作 | L_act | 1−Dice | 边界项（mm） | P | calib loss_of | 过程 IoU | 末帧 IoU | 末帧峰数 视频/模型 | "
         f"末帧 IoU（{rep['N_fig']}） | 峰数（{rep['N_fig']}） | 盒外参数 | 高度截断 低/高 | 流量截断 下限/上限 s |",
         "|" + "---|" * 15]
    for n, row in rep["patterns"].items():
        for s in SETS:
            r = row["sets"][s]
            if "L_act" not in r:
                continue
            la = r["L_act"]
            fn = r.get("final_N_fig") or {}
            hc = r.get("height_clip") or {}
            fc = r.get("flow_clip") or {}
            ob = "，".join(f"{k} {v:.2f}" for k, v in r["outside_box"]) or "无"
            L.append(f"| {CN[n]} | {SET_CN[s]}{'（= 拟合）' if s == 'PROJECTED' and row['projected_equals_fit'] else ''} | "
                     f"{la['value']:.3f} | {la['one_minus_dice']:.3f} | {la['bnd_term']:.3f}（{f(la['bnd_mm'], '{:.2f}')}） | "
                     f"{la['P']:.3f} | {r['loss_calib']:.3f} | {f(r['iou'])} | {f(r['final_iou'])} | "
                     f"{r['final_peaks'][0]}/{r['final_peaks'][1]} | {f(fn.get('iou'))} | "
                     f"{fn.get('peaks_video', '–')}/{fn.get('peaks_model', '–')} | {ob} | {hc.get('low')}/{hc.get('high')} | "
                     f"{f(fc.get('floor_s'), '{:.2f}')}/{f(fc.get('cap_s'), '{:.2f}')} |")
    L.append("")
    L.append("### 拟合出的动作参数（值，括号里是单位盒坐标；* = 贴边界，! = 出了 ±50 % 盒）")
    for n, row in rep["patterns"].items():
        r = row["sets"]["FIT"]
        cells = []
        for k in row["space"]:
            u = r["unit"][k]
            mark = ("*" if (u <= AT_BOUND or u >= 1 - AT_BOUND) else "") + ("!" if k in dict(r["outside_box"]) else "")
            cells.append(f"{k} {r['params'][k]:.3g}（{u:.2f}）{mark}")
        L.append(f"- {CN[n]}（{row['dimension']} 维；N = {row['fit_N']} 上 L_act {f(row['x0_L_act_N96'])} → {row['best_L_act_N96']:.3f}，"
                 f"失败 {row['n_failed']}）：" + "，".join(cells))
    return "\n".join(L)


def fig_final(out, rep, N_fig):
    """Rows = patterns, columns: real whiteness | SCRIPT | FIT | PROJECTED, final frame at N_fig (physical top view
    as v05_predict, origin upper)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(rep["patterns"])
    cols = ["real whiteness"] + [f"{s} N={N_fig}" for s in SETS]
    fig, axes = plt.subplots(len(names), len(cols), figsize=(2.6 * len(cols), 3.1 * len(names)), dpi=100, squeeze=False,
                             constrained_layout=True)
    for i, n in enumerate(names):
        clip = CB.load_clip(n)
        res = clip["res"]
        disk = res["disk"].astype(bool)
        row = rep["patterns"][n]
        for j, c in enumerate(cols):
            ax = axes[i, j]
            VP._style(ax)
            if i == 0:
                ax.set_title(c, fontsize=8)
            if j == 0:
                ax.imshow(VP._grey(res["white"][-1].astype(float), res["valid"][-1], disk), origin="upper")
                pv = VP.final_peaks(clip, res["white"][-1].astype(np.float32))[0]
                ax.set_xlabel(f"{n}: video, peaks {pv}", fontsize=7)
                continue
            s = SETS[j - 1]
            p = os.path.join(_eval_dir(out, s, n, N_fig, row["projected_equals_fit"], True), "final.npy")
            if not os.path.exists(p):
                ax.set_xlabel("not run", fontsize=7)
                continue
            w = np.load(p)
            ax.imshow(VP._grey(w, None, disk), origin="upper")
            r = row["sets"][s]
            fn = r.get("final_N_fig") or {}
            ob = len(r["outside_box"])
            ax.set_xlabel(f"IoU {fn.get('iou', float('nan')):.2f}  peaks {fn.get('peaks_model', '-')}  "
                          f"L_act({rep['N']}) {r.get('L_act', {}).get('value', float('nan')):.2f}"
                          + (f"\n{ob} param(s) outside +-50 %" if ob else ""), fontsize=7)
    fig.suptitle("V0.5, physics fixed (FSp2), barista's action fitted per pattern: final frame "
                 f"(N = {N_fig}); SCRIPT = as read off the video, FIT = best L_act, PROJECTED = multipliers clipped "
                 "into [0.5, 1.5]", fontsize=8)
    fig.savefig(os.path.join(out, f"final_N{N_fig}.png"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_lines(out, rep, N_fig):
    """Final-frame B -> D profile at N_fig: video vs SCRIPT / FIT / PROJECTED (legend: peaks)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(rep["patterns"])
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 1.6 * len(names)), dpi=100, squeeze=False)
    for i, n in enumerate(names):
        clip = CB.load_clip(n)
        res = clip["res"]
        vv = res["valid"][-1] & res["disk"].astype(bool)
        prof = clip["profiles"][0]
        ax = axes[i, 0]
        s_, pv = VP.profile_curve(res["white"][-1].astype(np.float32), vv, prof["p0_T"], prof["p1_T"])
        ax.plot(s_ * 80, pv, color="k", lw=1.6, label=f"video ({VP.final_peaks(clip, res['white'][-1].astype(np.float32))[0]})")
        row = rep["patterns"][n]
        for s, c in zip(SETS, ("tab:gray", "tab:red", "tab:blue")):
            p = os.path.join(_eval_dir(out, s, n, N_fig, row["projected_equals_fit"], True), "final.npy")
            if not os.path.exists(p):
                continue
            w = np.load(p)
            _, pm = VP.profile_curve(w, vv, prof["p0_T"], prof["p1_T"])
            ax.plot(s_ * 80, pm, color=c, lw=1.0, ls="--" if s == "PROJECTED" else "-",
                    label=f"{s} ({VP.final_peaks(clip, w)[1]})")
        ax.axhline(CB.THR, color="0.6", lw=0.6, ls=":")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(n, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, ncol=4, loc="upper right", framealpha=0.8)
    axes[-1, 0].set_xlabel("distance along the B -> D profile [mm]; legend: peaks (prominence >= 0.1)", fontsize=7)
    fig.suptitle(f"Final frame, B -> D profile, N = {N_fig} (raw video whiteness and validity, as v05_predict.final_peaks)",
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out, f"lines_BD_N{N_fig}.png"))
    plt.close(fig)


# =========================================================================== timing
def _time_one(args):
    name, N = args
    clip = CB.load_clip(name)
    model = CB.make_model(spec_of(name))
    x0 = x0_of(model.space)
    t0 = time.time()
    wT, led = CB.simulate_clip(model, x0, clip, N)
    t1 = time.time()
    ev = CB.evaluate(clip, wT, led, delay_in_loss=False)
    la = act_loss(clip, wT, ev)
    return dict(name=name, N=N, dim=len(model.space), wall_s=time.time() - t0, simulate_s=t1 - t0,
                solver_wall_s=led["wall_s"], ok=led["ok"], L_act=la["loss"], loss_calib=la["loss_calib"],
                P=la["L_act"]["P"], one_minus_dice=la["L_act"]["one_minus_dice"], bnd_term=la["L_act"]["bnd_term"],
                peaks=[(e["at"], e["peaks_video"], e["peaks_model"]) for e in la["structure"]["entries"]],
                flow_clip=led["act"]["flow_clip"], height_clip=led["act"]["height_clip"], pid=os.getpid())


def timing(out, N=96, procs=4, patterns=ORDER):
    from multiprocessing import get_context
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    load0 = os.getloadavg()
    t0 = time.time()
    order = [n for n in ("swan", "tulip", "push_heart", "leaf", "heart", "layered_heart") if n in patterns]
    with get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        res = list(pool.imap_unordered(_time_one, [(n, N) for n in order], chunksize=1))
    rep = dict(N=N, procs=procs, load_before=load0, load_after=os.getloadavg(), wall_total_s=time.time() - t0,
               cpus=os.cpu_count(), results={r["name"]: r for r in res})
    est = {}
    c = PREREG["cma"]
    for p in (2, 3, 4):
        rounds = math.ceil(c["popsize"] / p)
        per = {n: (1 + c["gens"] * rounds) * r["wall_s"] for n, r in rep["results"].items()}
        est[p] = dict(per_pattern_h={n: v / 3600 for n, v in per.items()}, total_h=sum(per.values()) / 3600)
    rep["estimate"] = dict(rule="(1 reference + gens x ceil(popsize / procs)) x t_forward(x0), no straggler factor",
                           by_procs=est)
    _jdump(rep, os.path.join(out, f"timing_N{N}.json"))
    return rep


# =========================================================================== unit tests
def test_identity(names=("heart", "tulip"), N=48, verbose=True):
    """Identity params (multipliers 1, u_exit 0.15, offsets 0) reproduce V05Model(FSp2) on the unmodified script
    bit for bit (c arrays and ledger volumes), and the transformed script equals the input."""
    rep = {}
    for name in names:
        sc = CB.physical_script(name)
        m = V05ActModel(name)
        x0 = x0_of(m.space)
        sc2, info = m.transform(sc, x0)
        assert sc2 == sc, name
        assert info["height_clip"]["low"] == 0 and info["height_clip"]["high"] == 0
        T = max(s["t1"] for s in sc["stages"])
        t_out = np.array([0.0, 0.25 * T, 0.5 * T, 0.8 * T, T + 0.3])
        a = V05Model(base=FSP2).run(sc, {}, N, t_out)
        b = m.run(sc, x0, N, t_out)
        same = bool(np.array_equal(a["c"], b["c"]))
        led = {k: (a["ledger"][k], b["ledger"][k]) for k in ("V_received_mL", "V_wanted_mL", "deposited", "steps")}
        rep[name] = dict(bit_identical=same, max_abs_diff=float(np.abs(a["c"] - b["c"]).max()), ledger=led,
                         flow_clip=b["ledger"]["act"]["flow_clip"])
        if verbose:
            print("identity", name, rep[name], flush=True)
        assert same and all(x == y for x, y in led.values()), rep[name]
    return rep


def _random_params(space, rng, offsets=True):
    p = {k: CB.from_unit(rng.uniform(), *space[k]) for k in space}
    if not offsets:
        p["dx"] = p["dy"] = 0.0
    return p


def test_multipliers(names=ORDER, seed=3, verbose=True):
    """Multipliers change exactly z0 / z1 (then the height clip) and flow0 / flow1 of the stages of the scaled kinds;
    every other field of every stage (and every field of the dry stages) is unchanged."""
    rng = np.random.default_rng(seed)
    rep = {}
    for name in names:
        sc = CB.physical_script(name)
        m = V05ActModel(name)
        n_checked = 0
        for _ in range(3):
            p = _random_params(m.space, rng, offsets=False)
            sc2, info = m.transform(sc, p)
            assert sc2["stages"] is not sc["stages"] and len(sc2["stages"]) == len(sc["stages"])
            assert {k: v for k, v in sc2.items() if k != "stages"} == {k: v for k, v in sc.items() if k != "stages"}
            for s, s2 in zip(sc["stages"], sc2["stages"]):
                k = s["kind"]
                changed = {f for f in set(s) | set(s2) if s.get(f) != s2.get(f)}
                if k in m.kinds:
                    for e in ("z0", "z1"):
                        assert s2[e] == min(max(s[e] * p[f"z_{k}"], Z_CLIP[0]), Z_CLIP[1]), (name, k, e)
                    for e in ("flow0", "flow1"):
                        assert s2[e] == s[e] * p[f"q_{k}"], (name, k, e)
                    assert changed <= {"z0", "z1", "flow0", "flow1"}, (name, k, changed)
                else:
                    assert not changed, (name, k, changed)
                n_checked += 1
        rep[name] = dict(kinds=list(m.kinds), stages_checked=n_checked)
        if verbose:
            print("multipliers", name, rep[name], flush=True)
    return rep


def test_offsets(names=ORDER, d=(0.021, -0.017), dt=0.005, verbose=True):
    """Offsets move every p0 / p1 / Bezier point of every stage by exactly (dx, dy) and nothing else; the wanted foot
    path of the whole script (dry gaps and wiggles included) moves rigidly, height and flow unchanged."""
    from .v3min_adapter import Inlet as WantedPath
    rep = {}
    for name in names:
        sc = CB.physical_script(name)
        m = V05ActModel(name)
        p = x0_of(m.space)
        p.update(dx=d[0], dy=d[1])
        sc2, _ = m.transform(sc, p)
        nb = 0
        for s, s2 in zip(sc["stages"], sc2["stages"]):
            for e in ("p0", "p1"):
                assert abs(s2[e][0] - s[e][0] - d[0]) < 1e-15 and abs(s2[e][1] - s[e][1] - d[1]) < 1e-15
            if s.get("bezier"):
                for c, c2 in zip(s["bezier"], s2["bezier"]):
                    assert abs(c2[0] - c[0] - d[0]) < 1e-15 and abs(c2[1] - c[1] - d[1]) < 1e-15
                nb += 1
            changed = {f for f in set(s) | set(s2) if s.get(f) != s2.get(f)}
            assert changed <= {"p0", "p1", "bezier"}, changed
        a, b = WantedPath(sc), WantedPath(sc2)
        worst = dict(path=0.0, zQ=0.0)
        for t in np.arange(0.0, a.T + 0.2, dt):
            xa, ya, za, qa = a.at(t)
            xb, yb, zb, qb = b.at(t)
            worst["path"] = max(worst["path"], abs(xb - xa - d[0]), abs(yb - ya - d[1]))
            worst["zQ"] = max(worst["zQ"], abs(zb - za), abs(qb - qa))
        rep[name] = dict(bezier_stages=nb, **worst)
        if verbose:
            print("offsets", name, rep[name], flush=True)
        assert worst["path"] < 1e-12 and worst["zQ"] == 0.0, worst
    return rep


def test_clip_counts(verbose=True):
    """Height clip counts against an independent count; flow-clip counting on a synthetic log; a real run (heart,
    N = 32) with q_low_pour = 3 activates the 32 mL/s cap for about as long as the lag exceeds it on a fine grid."""
    from .v05_adapter import PhysInlet, strip_preroll
    rep = {}
    for name in ORDER:
        sc = CB.physical_script(name)
        m = V05ActModel(name)
        rep[name] = {}
        # inside the bounds (3 and 1/3), and below them (0.2: the low clip; 1/3 x 10 mm = 3.3 mm never reaches it)
        for tag, zlo in (("bounds", 1.0 / 3.0), ("z0.2", 0.2)):
            p = x0_of(m.space)
            for i, k in enumerate(m.kinds):
                p[f"z_{k}"] = 3.0 if i % 2 == 0 else zlo
            sc2, info = m.transform(sc, p)
            lo = hi = ends = 0
            for s in sc["stages"]:
                if s["kind"] in m.kinds:
                    for e in ("z0", "z1"):
                        v = s[e] * p[f"z_{s['kind']}"]
                        lo += v < Z_CLIP[0]
                        hi += v > Z_CLIP[1]
                        ends += 1
            hc = info["height_clip"]
            assert (hc["low"], hc["high"], hc["n_scaled_ends"]) == (lo, hi, ends), (name, hc, lo, hi, ends)
            assert len(hc["entries"]) == lo + hi
            assert all(Z_CLIP[0] <= s[e] <= Z_CLIP[1] for s in sc2["stages"] if s["kind"] in m.kinds
                       for e in ("z0", "z1"))
            rep[name][tag] = dict(low=lo, high=hi, ends=ends)
    assert sum(r["z0.2"]["low"] for r in rep.values()) > 0 and sum(r["bounds"]["high"] for r in rep.values()) > 0
    # synthetic log: 4 frames of 0.01 s: floor, normal, cap while pouring, cap while leaking
    c = {k: i for i, k in enumerate(LOGCOLS)}
    lg = np.zeros((4, len(LOGCOLS)))
    lg[:, c["fdt"]] = 0.01
    lg[:, c["Q_w_mLs"]] = [15.0, 15.0, 40.0, 0.0]
    lg[:, c["Q_mLs"]] = [Q_MIN / ML, 14.0, Q_MAX / ML, Q_MAX / ML]
    fc = flow_clips(lg)
    assert abs(fc["floor_s"] - 0.01) < 1e-15 and abs(fc["cap_s"] - 0.01) < 1e-15 and abs(fc["cap_leak_s"] - 0.01) < 1e-15
    assert abs(fc["pour_s"] - 0.03) < 1e-15 and abs(fc["wanted_above_cap_mL"] - 0.08) < 1e-12
    rep["synthetic_log"] = fc
    # real run: heart at N = 32, x0 (no cap) and q_low_pour = 3 (cap)
    m = V05ActModel("heart")
    sc = CB.physical_script("heart")
    T = max(s["t1"] for s in sc["stages"])
    out = {}
    for tag, q in (("x0", 1.0), ("q3", 3.0)):
        p = x0_of(m.space)
        p["q_low_pour"] = q
        r = m.run(sc, p, 32, np.array([T]))
        fcr = r["ledger"]["act"]["flow_clip"]
        sc2, _ = m.transform(sc, p)
        inl = PhysInlet(strip_preroll(sc2)[0], kappa_Q=FSP2["kappa_Q"])
        tg = np.arange(0.0005, inl.T, 0.001)
        cap_grid = 0.001 * sum(1 for t in tg if inl.path.at(t)[3] > 0 and inl.q_lag(t) > Q_MAX)
        out[tag] = dict(cap_s_log=fcr["cap_s"], cap_s_grid=cap_grid, floor_s_log=fcr["floor_s"],
                        max_wanted_mLs=fcr["max_wanted_mLs"])
    assert out["x0"]["cap_s_log"] == 0.0 and out["x0"]["cap_s_grid"] == 0.0
    assert out["q3"]["cap_s_log"] > 1.0, out
    assert abs(out["q3"]["cap_s_log"] - out["q3"]["cap_s_grid"]) < max(0.1, 0.03 * out["q3"]["cap_s_grid"]), out
    rep["heart_run"] = out
    if verbose:
        print("clip counts", rep, flush=True)
    return rep


def test_structure_zero(names=ORDER, verbose=True):
    """P = 0 when the model field equals the video field; the stage-end entries are the non-final profiles of
    calib.profile_metrics (same frames); P > 0 for an all-brown model field when the video has stripes."""
    rep = {}
    for name in names:
        clip = CB.load_clip(name)
        res = clip["res"]
        wv = res["white"].astype(np.float32)
        st = structure_term(clip, wv)
        pm = [p for p in CB.profile_metrics(clip, res, wv) if p["at"] != "final"]
        se = [e for e in st["entries"] if e["at"] != "final"]
        assert [e["at"] for e in se] == [p["at"] for p in pm] and [e["t"] for e in se] == [p["t"] for p in pm], name
        blank = structure_term(clip, np.zeros_like(wv))
        rep[name] = dict(P_video=st["P"], n_entries=len(st["entries"]), P_blank=blank["P"],
                         peaks_video=[(e["at"], e["peaks_video"]) for e in st["entries"]])
        if verbose:
            print("structure", name, rep[name], flush=True)
        assert st["P"] == 0.0
        if any(e["peaks_video"] > 0 for e in st["entries"]):
            assert blank["P"] > 0
    return rep


def test_init_finite(names=ORDER, N=48, verbose=True):
    """L_act (and loss_of) finite on all six patterns at the init (through the same worker as the fit)."""
    rep = {}
    for name in names:
        n, ev = _weval_act((x0_of(V05ActModel(name).space), name, N))
        rep[name] = dict(L_act=ev["loss"], loss_calib=ev.get("loss_calib"), P=(ev.get("L_act") or {}).get("P"),
                         failed=bool(ev.get("failed")), wall_s=(ev.get("ledger") or {}).get("wall_s"))
        if verbose:
            print("init", name, rep[name], flush=True)
        assert np.isfinite(ev["loss"]) and np.isfinite(ev["loss_calib"]) and not ev.get("failed"), rep[name]
    return rep


def selftest():
    return dict(multipliers=test_multipliers(), offsets=test_offsets(), clip_counts=test_clip_counts(),
                structure=test_structure_zero(), identity=test_identity(), init=test_init_finite())


# =========================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["prereg", "run", "smoke", "status", "evaluate", "timing", "test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--path", default=PREREG_PATH, help="pre-registration file (prereg / run)")
    ap.add_argument("--patterns", default=",".join(ORDER))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--N_fig", type=int, default=PREREG["evaluate"]["N_fig"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    pats = tuple(p for p in a.patterns.split(",") if p)
    for p in pats:
        assert p in ORDER, p
    c = PREREG["cma"]
    if a.cmd == "prereg":
        write_prereg(a.path, a.force)
    elif a.cmd == "run":
        run(a.out, pats, c["gens"], c["popsize"], a.N or c["N"], a.procs, c["sigma0"], c["seed"], "run", a.force, a.path)
    elif a.cmd == "smoke":
        if a.patterns == ",".join(ORDER):
            pats = ("heart", "swan")
        run(a.out, pats, 1, 4, a.N or 64, a.procs, c["sigma0"], c["seed"], "smoke", a.force, a.path)
    elif a.cmd == "status":
        status(a.out)
    elif a.cmd == "evaluate":
        rep = evaluate(a.out, pats, a.N or PREREG["evaluate"]["N"], a.N_fig, a.procs)
        print(tables_md(rep))
    elif a.cmd == "timing":
        rep = timing(a.out, a.N or c["N"], a.procs, pats)
        print(json.dumps({n: dict(wall_s=round(r["wall_s"], 1), solver_s=round(r["solver_wall_s"], 1), dim=r["dim"],
                                  L_act=round(r["L_act"], 4), loss_of=round(r["loss_calib"], 4))
                          for n, r in rep["results"].items()}, indent=1))
        print(json.dumps(rep["estimate"], indent=1, default=float))
    elif a.cmd == "test":
        r = selftest()
        print(json.dumps(r, default=float))
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    # run through the package module so that pool workers pickle functions as latte_imex.v05_actfit.*
    from latte_imex import v05_actfit as _me
    _me.main()
