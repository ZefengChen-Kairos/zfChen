"""Calibration of the V0.5 closure with the GUESSED all-physical actions FS (senior V1 5.1-5.4 protocol: calibrate on
tulip + leaf, freeze, test on heart, push_heart, layered_heart, swan), final frame against the real video.

    python -m latte_imex.v05_guessed_calib prereg   --out DIR                    # write DIR/prereg.json (no runs)
    python -m latte_imex.v05_guessed_calib nrule    --out DIR [--procs 3]        # N rule of PREREG (FS closure, tulip +
                                                                                 # leaf, N = 96 / 128 / 256) + FS repro
    python -m latte_imex.v05_guessed_calib run      --out DIR [--procs 4]        # pre-registered CMA-ES (resumable)
    python -m latte_imex.v05_guessed_calib run      --out SMOKE --smoke --prereg DIR/prereg.json
    python -m latte_imex.v05_guessed_calib status   --out DIR
    python -m latte_imex.v05_guessed_calib evaluate --out DIR [--procs 4]        # frozen best + FS, six patterns, N = 256
    (run / evaluate --seed_from DIR2: reuse finished calls with the same cache key from DIR2/cache, e.g. nrule's FS)
    python -m latte_imex.v05_guessed_calib test     [--out DIR]                  # unit tests

ACTIONS (fixed during calibration and test, never re-designed for a new closure).  The final all-physical set FS of
latte_imex/v05_allphys (V05_ALLPHYS_RESULT.md sections 1, 3, 4): Codex's paths, timings and wiggles
(v05_controls/*.json); flow = Codex's taper with 0.1 s on/off ramps and bridges >= 0.3 s, passed through a first-order
spout lag tau = 0.08 s, 5 Hz pulsation applied after the lag at depth min(Codex, 0.41) with the flow-peak phase 0
(tulip, leaf, swan), pi/2 (layered_heart) or Codex's own (heart, push_heart: no pulsation); per-stroke spout heights of
the FS inverse design (allphys_experiments.json 'P2', embedded below as FS_U), U_perp = sqrt(0.35^2 + 2 g z); height
transitions <= 0.11 m/s; horizontal speed 'exitmin' (lateral = hand speed, exit part along Codex's jet >= 0.10 m/s);
continuity footprint.  The inlet object is drvf.PhysF of v05_allphys/drvf.py (imported from its file, not edited), built
exactly as drvf.main builds it for
    python3 -I drvf.py FS <pattern> --exp P2 --pulse 0.41 --lat exitmin --lag 0.08 --ztr [--flow_phase 0 | pi/2]
(final/jobs.txt), and the run is latte_imex.run.run(control='v05') with run.v05_program pointed at it, as in drvf.main.
CLOSURE OVERRIDE.  drvf.main puts kappa_Q, cp, beta, nu, B_dep, p_dep, kappa_r into the control's 'parameters' and leaves
kappa_t, kappa_c, D at the control's values (0.7, 1.0, 1e-7).  Here all ten closure keys go through the same 'parameters'
dict (-> V05Script.params -> solver.Params), and kappa_Q is also PhysF's kQ: received flow Q = clip(F / kappa_Q, 1, 32 mL/s),
S_eff = kappa_Q Q (the solver itself does not read Params.kappa_Q).  With the FS closure this is drvf's call, value for
value (test: identity closure = drvf subprocess bit for bit; nrule: = the stored FS 256^2 fields).

LOSS (version LOSS_VERSION, final frame only: the guessed actions have their own timeline).  Canonical frame of the
real-video comparison (review/v05real/compare.py, imported, not copied): target w from compare.target (leaf 180 deg,
swan 135 deg fixed frame corrections), sim -> target grid by compare.Sampler, video white w > 0.5, model white c > 0.4,
cup R = 0.49 D.  Best rigid alignment = argmax IoU(c > 0.4) over theta in {-25, -22.5, .., 25} deg x dx, dy in
{-0.06, -0.05, .., 0.06} D (first maximum in that loop order, as compare.best_align, no refinement); with that transform
    L_fin = (1 - Dice) + d_bnd / 0.05 D + S,
    d_bnd = compare.boundary_distance(...)['sym'] [D] (EMPTY_BND = 0.5 D when one side has no boundary),
    S = mean over the five lines x = -0.10, -0.05, 0, 0.05, 0.10 D of |n_model - n_video| / max(n_video, 1),
    n = structnum.stripes(...)['total'] (brown lines on that line = cf_runner.axis_gaps gaps >= 1 mm + dips with
    prominence >= 0.1 and FWHM >= 0.5 mm inside the white extent; model threshold 0.4, video 0.5).
Calibration loss = mean of L_fin over tulip and leaf.  Raw (unaligned) IoU / Dice / boundary / S are reported too.
The video side (w, mask, white mask, line counts) is computed once per pattern and cached in DIR/video/.

CMA-ES (calib.calibrate, unchanged), with v05_calibrate.patched's disk cache + wall-time limit around the forward
worker (inner = this module's _inner).  Cache key = v05_calibrate.cache_key(params, pattern, N, SPEC), SPEC holding the
loss version, the action-definition hash (FS_U, phases, flags, sha1 of drvf.py and the six controls) and the fixed
closure part; the final field of every call is kept as DIR/cache/<key>.npz.  What the key does not hold (PREREG text and
constants, forward / loss code sha1, video targets) is checked against the pre-registration by check_unchanged before
run and evaluate.  Failures (exception, non-finite field or
ledger, wall > TIMEOUT_S) score +inf (1e6 for CMA) and are counted.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

from . import calib as CB
from . import v05_calibrate as VC
from .v05_adapter import FS, FSP2, SPACE

PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(PKG)
DRVF = os.path.join(PKG, "v05_allphys", "drvf.py")
CONTROLS = os.path.join(PKG, "v05_controls")
REVIEW = os.environ.get("V05GCAL_REVIEW",
                        "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review")
REAL = os.path.join(REVIEW, "v05real")                       # compare.py, structnum.py (they import v05audit/cf_runner)
FS_RUNS = os.path.join(REVIEW, "v05allphys", "final", "runs")  # stored FS 256^2 runs (FS_<pattern>/c.npz, res.json)
ALLPHYS_EXP = os.path.join(REVIEW, "v05allphys", "allphys_experiments.json")

PATTERNS = ("heart", "push_heart", "layered_heart", "tulip", "leaf", "swan")
NAMES = CB.CAL_AB                                            # ("tulip", "leaf")
TEST = CB.TEST                                               # ("heart", "push_heart", "layered_heart", "swan")
CLOSURE_KEYS = ("cp", "beta", "nu", "D", "kappa_Q", "kappa_c", "kappa_t", "kappa_r", "B_dep", "p_dep")
CLOSURE8 = VC.CLOSURE8                                       # calibrated: B_dep, p_dep, kappa_r, kappa_Q, cp, beta, kappa_t, kappa_c
FIXED = dict(nu=1e-3, D=1e-7)
ML = 512.0                                                   # mL per D^3

# ---------------------------------------------------------------- the FS action set (fixed)
# per-stroke impact speed U_perp [m/s] of the FS inverse design = allphys_experiments.json['P2']['inlet'][p]['U']
FS_U = {
    "heart": {"0": 0.6187466042832238, "1": 0.5618302660868209, "2": 0.6716144314223843},
    "push_heart": {"0": 0.6760038173559236, "2": 0.6492869662180937, "4": 0.6577490325604356, "6": 0.6951097066984131},
    "layered_heart": {"0": 0.7025497621782993, "1": 0.7828976149442493, "2": 0.6728876150955637},
    "tulip": {"0": 0.7025497621782993, "1": 0.7825559058181663, "3": 0.7814983312060321, "5": 0.4980515934687648,
              "7": 0.691427145529588},
    "leaf": {"0": 0.7025497621782993, "1": 0.7828976149442493, "2": 0.6278252465339513, "3": 0.5003756917732478,
             "5": 0.6277819106059029},
    "swan": {"0": 0.7025497621782993, "1": 0.7825559058181663, "3": 0.4805353804311261, "5": 0.671928872540564,
             "7": 0.43100192902387174, "8": 0.4258638280013929, "10": 0.598280870494787},
}
# --flow_phase of the FS jobs (final/jobs.txt): None = 'keep' (Codex's own; heart / push_heart do not pulse)
FLOW_PHASE = dict(heart=None, push_heart=None, layered_heart=1.5707963267948966, tulip=0.0, leaf=0.0, swan=0.0)
ACTION = dict(exp="P2", pulse=0.41, lat="exitmin", lag=0.08, ramp=0.1, stop=False, ztr=True, bridge=0.3)

# ---------------------------------------------------------------- loss constants (pre-registered)
LOSS_VERSION = "Lfin-1"
THR_SIM, THR_VID = 0.4, 0.5
BND_SCALE = 0.05
EMPTY_BND = 0.5
TH_MAX, TH_STEP, SH_MAX, SH_STEP = 25.0, 2.5, 0.06, 0.01
LINES = (-0.10, -0.05, 0.0, 0.05, 0.10)                      # = structnum.LINES (checked in real())
TIMEOUT_S = 1500
INNER = "latte_imex.v05_guessed_calib:_inner"


def _sha1_file(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def action_definition():
    return dict(FS_U=FS_U, FLOW_PHASE={k: (None if v is None else repr(v)) for k, v in FLOW_PHASE.items()},
                ACTION=ACTION, drvf_sha1=_sha1_file(DRVF),
                controls_sha1={p: _sha1_file(os.path.join(CONTROLS, f"{p}.json")) for p in PATTERNS})


def actions_id():
    return hashlib.sha1(json.dumps(action_definition(), sort_keys=True).encode()).hexdigest()


def make_spec():
    """calib spec (make_model reads kind/base only -> space = v05_adapter.SPACE); the other keys go into the cache key."""
    return dict(kind="v05", base="FS", loss=LOSS_VERSION, actions="FS", actions_id=actions_id(), fixed=FIXED,
                driver="v05_guessed_calib")


def closure_of(params):
    """the 8 calibrated values (subset allowed) on top of FS; nu, D = FIXED (the values the cache key holds)."""
    c = {k: float(FS[k]) for k in CLOSURE_KEYS}
    c.update({k: float(v) for k, v in FIXED.items()})
    for k, v in (params or {}).items():
        if k not in CLOSURE8:
            raise KeyError(f"not a calibrated closure parameter: {k}")
        c[k] = float(v)
    return c


def init_points():
    fs = {k: float(FS[k]) for k in CLOSURE8}
    fsp2 = {k: float(FSP2[k]) for k in CLOSURE8}
    centre = CB.decode([0.5] * len(SPACE), SPACE)
    return dict(FS=fs, FSp2=fsp2, centre=centre)


# =========================================================================== forward model
_MOD = {}


def drvf():
    """v05_allphys/drvf.py as a module (imported from its file; nothing in it is changed)."""
    if "drvf" not in _MOD:
        spec = importlib.util.spec_from_file_location("v05_allphys_drvf", DRVF)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MOD["drvf"] = m
    return _MOD["drvf"]


def fs_program(pattern, closure):
    """drvf.PhysF exactly as drvf.main builds it for the FS job of `pattern`, with the closure override."""
    D = drvf()
    params = {k: float(closure[k]) for k in CLOSURE_KEYS}
    return D.PhysF(pattern, FS_U[pattern], params["kappa_Q"], params, ACTION["pulse"], FLOW_PHASE[pattern],
                   bridge=ACTION["bridge"], ramp=ACTION["ramp"], lat=ACTION["lat"], lag=ACTION["lag"],
                   stop=ACTION["stop"], ztr=ACTION["ztr"], zset={}, diag_uh={}, diag_r={})


def inlet_summary(log):
    """applied inlet samples (second sample of every frame, drvf log parity 0): received flow range, clip activity of
    Q = clip(F / kappa_Q, 1, 32 mL/s) (sample fraction and time fraction), wanted / received volumes."""
    if len(log) == 0:
        return dict(n_active=0)
    L = np.asarray(log)
    A = L[L[:, 1] == 0]
    t, qraw, q = A[:, 0], A[:, 4], A[:, 5]
    dt = np.diff(np.concatenate([t, [t[-1] + (np.median(np.diff(t)) if len(t) > 1 else 0.0)]]))
    dt = np.where((dt > 0) & (dt < 0.1), dt, 0.0)            # a gap > 0.1 s is a bridge / stop (no flow logged)
    lo, hi = qraw < 1.0 - 1e-12, qraw > 32.0 + 1e-12
    T = float(dt.sum()) or 1.0
    return dict(n_active=int(len(A)), Q_mLs=[float(q.min()), float(q.max())], Qraw_mLs=[float(qraw.min()), float(qraw.max())],
                clip_low_frac=float(lo.mean()), clip_high_frac=float(hi.mean()),
                clip_low_time_frac=float((dt * lo).sum() / T), clip_high_time_frac=float((dt * hi).sum() / T),
                V_wanted_mL=float((qraw * dt).sum()), V_received_mL=float((q * dt).sum()),
                U_ms=[float(A[:, 6].min()), float(A[:, 6].max())], r_mm=[float(A[:, 14].min()), float(A[:, 14].max())])


def run_fs(pattern, closure, N, T_max=None, workdir=None, keep=False):
    """One forward run with the FS actions -> dict(c = final c field (NaN outside the cup, M frame [iy, ix]), mask,
    ledger, stats, solver_params (what the solver actually received), kQ, inlet summary, wall_s)."""
    from . import run as R
    D = drvf()
    assert D.R is R
    prog = fs_program(pattern, closure)
    old = R.v05_program
    R.v05_program = lambda name: prog                         # as drvf.main
    tmp = tempfile.mkdtemp(prefix=f"fs_{pattern}_N{N}_", dir=workdir)
    try:
        t0 = time.time()
        sol, m = R.run(pattern, N=N, out=tmp, video=False, quiet=True, control="v05", T_max=T_max)
        wall = time.time() - t0
    finally:
        R.v05_program = old
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
    c = sol.concentration()
    mask = sol.g.mask.copy()
    P = sol.P
    return dict(c=c, mask=mask, wall_s=wall, T=float(m["T"]), t_end=float(sol.t), stats=m["stats"], ledger=m["ledger"],
                solver_params={k: getattr(P, k) for k in CLOSURE_KEYS + ("closure", "return_law")}, kQ=prog.kQ,
                program_params=dict(prog.config["parameters"]), inlet=inlet_summary(prog.log),
                log=np.array(prog.log), out=tmp if keep else None)


# =========================================================================== loss
def real():
    """compare.py / structnum.py of the real-video comparison (imported from their files)."""
    if "C" not in _MOD:
        if REAL not in sys.path:
            sys.path.insert(0, REAL)
        import compare as C                                   # noqa: E402
        import structnum as SN                                # noqa: E402
        assert os.path.realpath(C.__file__) == os.path.realpath(os.path.join(REAL, "compare.py")), C.__file__
        assert os.path.realpath(SN.__file__) == os.path.realpath(os.path.join(REAL, "structnum.py")), SN.__file__
        assert tuple(SN.LINES) == LINES and C.TH_MAX == TH_MAX and C.SH_MAX == SH_MAX and C.CF.THR_VID == THR_VID
        _MOD["C"], _MOD["SN"] = C, SN
    return _MOD["C"], _MOD["SN"]


def code_sha1():
    C, SN = real()
    return dict(compare=_sha1_file(C.__file__), structnum=_sha1_file(SN.__file__), cf_runner=_sha1_file(C.CF.__file__),
                drvf=_sha1_file(DRVF), calib=_sha1_file(os.path.join(PKG, "calib.py")),
                v05_calibrate=_sha1_file(os.path.join(PKG, "v05_calibrate.py")),
                solver=_sha1_file(os.path.join(PKG, "solver.py")), kernels=_sha1_file(os.path.join(PKG, "kernels.py")),
                run=_sha1_file(os.path.join(PKG, "run.py")), v05_program=_sha1_file(os.path.join(PKG, "v05_program.py")),
                this=_sha1_file(os.path.abspath(__file__)))


_VID = {}


def _video_key(p):
    C, _ = real()
    return dict(pattern=p, PHI=C.PHI[p], target_sha1=_sha1_file(os.path.join(PKG, "targets", f"{p}.npz")),
                compare=_sha1_file(C.__file__), structnum=_sha1_file(_MOD["SN"].__file__), cf_runner=_sha1_file(C.CF.__file__),
                thr=THR_VID, loss=LOSS_VERSION)


def video(p, vdir=None):
    """video side of the loss for pattern p (canonical frame): w, wm, white mask, per-line brown-line counts.  Cached in
    memory and, with vdir, on disk (vdir/<p>.npz + .json, recomputed when the key differs)."""
    if p in _VID:
        return _VID[p]
    C, SN = real()
    key = _video_key(p)
    if vdir:
        fj, fz = os.path.join(vdir, f"{p}.json"), os.path.join(vdir, f"{p}.npz")
        if os.path.exists(fj) and os.path.exists(fz):
            meta = json.load(open(fj))
            if meta.get("key") == key:
                z = np.load(fz)
                V = dict(meta, w=z["w"], wm=z["wm"])
                V["vw"] = (V["w"] > THR_VID) & V["wm"]
                _VID[p] = V
                return V
    w, wm = C.target(p, canonical=True)
    st = SN.stripes(w, wm, THR_VID)
    vw = (w > THR_VID) & wm
    meta = dict(key=key, Mt=int(w.shape[0]), n=[int(s["total"]) for s in st], gaps=[int(s["gaps"]) for s in st],
                dips=[int(s["dips"]) for s in st], white_frac=float(vw.sum() / wm.sum()),
                source=f"latte_imex/targets/{p}.npz via compare.target (cf_runner.load_target), PHI {C.PHI[p]} deg")
    if vdir:
        os.makedirs(vdir, exist_ok=True)
        tmp = os.path.join(vdir, f".{p}.{os.getpid()}.npz")
        np.savez_compressed(tmp, w=w, wm=wm)
        os.replace(tmp, os.path.join(vdir, f"{p}.npz"))
        _jdump(meta, os.path.join(vdir, f"{p}.json"))
    V = dict(meta, w=w, wm=wm, vw=vw)
    _VID[p] = V
    return V


def seam_term(n_model, n_video):
    return float(np.mean([abs(a - b) / max(b, 1) for a, b in zip(n_model, n_video)]))


def terms(sT, V):
    """L_fin terms of a model field sT already on the target grid (T frame)."""
    C, SN = real()
    wm, vw = V["wm"], V["vw"]
    a = (sT > THR_SIM) & wm
    na, nb, ni = int(a.sum()), int(vw.sum()), int((a & vw).sum())
    dice = 2.0 * ni / (na + nb) if na + nb else float("nan")
    bd = C.boundary_distance(a, vw, wm)
    empty = not np.isfinite(bd["sym"])
    bnd = EMPTY_BND if empty else float(bd["sym"])
    st = SN.stripes(sT, wm, THR_SIM)
    n_m = [int(s["total"]) for s in st]
    S = seam_term(n_m, V["n"])
    return dict(L=float((1.0 - dice) + bnd / BND_SCALE + S), dice=float(dice), iou=C.CF.iou(a, vw), bnd_D=bnd,
                bnd_dir_D=[bd["a_to_b"], bd["b_to_a"]], bnd_empty=bool(empty), S=S, n_model=n_m,
                gaps_model=[int(s["gaps"]) for s in st], dips_model=[int(s["dips"]) for s in st],
                white_frac=float(na / wm.sum()))


def grid():
    ths = np.arange(-TH_MAX, TH_MAX + 1e-9, TH_STEP)
    shs = np.round(np.arange(-SH_MAX, SH_MAX + 1e-9, SH_STEP), 4)
    return ths, shs


def iou_grid_exact(smp, V):
    """IoU(c > 0.4) on the whole alignment grid with compare.Sampler.field_T (the reference operator; slow)."""
    C, _ = real()
    ths, shs = grid()
    out = np.empty((len(ths), len(shs), len(shs)))
    for a, th in enumerate(ths):
        for i, dx in enumerate(shs):
            for j, dy in enumerate(shs):
                s = smp.field_T(float(th), float(dx), float(dy))
                out[a, i, j] = C.CF.iou((s > THR_SIM) & V["wm"], V["vw"])
    return out


_NB = {}


def _iou_kernel():
    """numba kernel: IoU of (bilinear sample > thr) vs the video white mask over the cup cells, for every transform.
    Same coordinates as compare.Sampler.field_T + cf_runner.sample (rotation R(-theta) of q - d, cell-centred index
    (y + 0.5) N - 0.5, map_coordinates order 1 with mode 'nearest' = clamped index); only cup cells enter
    cf_runner.iou((s > thr) & wm, (w > 0.5) & wm), so only they are sampled."""
    if "k" not in _NB:
        import numba

        @numba.njit(cache=False)
        def k(ff, X, Y, vw, cs, sn, shs, thr):
            N = ff.shape[0]
            nT, nS, nP = cs.shape[0], shs.shape[0], X.shape[0]
            out = np.empty((nT, nS, nS))
            for a in range(nT):
                c, s = cs[a], sn[a]
                for i in range(nS):
                    dx = shs[i]
                    for j in range(nS):
                        dy = shs[j]
                        inter = 0
                        uni = 0
                        for p in range(nP):
                            qx = X[p] - dx
                            qy = Y[p] - dy
                            px = c * qx + s * qy
                            py = -s * qx + c * qy
                            I = (py + 0.5) * N - 0.5
                            J = (px + 0.5) * N - 0.5
                            I = min(max(I, 0.0), N - 1.0)
                            J = min(max(J, 0.0), N - 1.0)
                            i0 = min(int(math.floor(I)), N - 2)
                            j0 = min(int(math.floor(J)), N - 2)
                            ti = I - i0
                            tj = J - j0
                            v = ((1.0 - ti) * ((1.0 - tj) * ff[i0, j0] + tj * ff[i0, j0 + 1])
                                 + ti * ((1.0 - tj) * ff[i0 + 1, j0] + tj * ff[i0 + 1, j0 + 1]))
                            w = v > thr
                            if w and vw[p]:
                                inter += 1
                            if w or vw[p]:
                                uni += 1
                        out[a, i, j] = inter / uni if uni > 0 else np.nan
            return out
        _NB["k"] = k
    return _NB["k"]


def iou_grid_fast(smp, V):
    """same grid as iou_grid_exact, by _iou_kernel (tested equal to the reference operator in `test`)."""
    ths, shs = grid()
    Mt = V["Mt"]
    g = (np.arange(Mt) + 0.5) / Mt - 0.5                      # as compare.Sampler
    r, col = np.nonzero(V["wm"])                               # T-frame cup cells; field_T flips rows: row r <-> g[Mt-1-r]
    X, Y = g[col].copy(), g[Mt - 1 - r].copy()
    t = [math.radians(float(th)) for th in ths]
    cs = np.array([math.cos(x) for x in t])
    sn = np.array([math.sin(x) for x in t])
    return _iou_kernel()(np.ascontiguousarray(smp.ff, dtype=np.float64), X, Y, V["vw"][r, col].copy(), cs, sn,
                         np.asarray(shs, dtype=np.float64), THR_SIM)


def align(smp, V, exact=False):
    """argmax IoU(c > 0.4) on the pre-registered grid (first maximum in loop order theta, dx, dy, as the strict '>' scan
    of compare.best_align)."""
    ths, shs = grid()
    G = iou_grid_exact(smp, V) if exact else iou_grid_fast(smp, V)
    a, i, j = np.unravel_index(int(np.argmax(G)), G.shape)
    th, dx, dy = float(ths[a]), float(shs[i]), float(shs[j])
    return dict(theta_deg=th, dx_D=dx, dy_D=dy, iou_search=float(G[a, i, j]), exact=bool(exact),
                at_bound=dict(theta=abs(abs(th) - TH_MAX) < 1e-9, dx=abs(abs(dx) - SH_MAX) < 1e-9,
                              dy=abs(abs(dy) - SH_MAX) < 1e-9))


def lfin(cM, mM, pattern, vdir=None):
    """L_fin of a model field c_M (M frame [iy, ix], NaN outside mM) against the real video of `pattern`."""
    C, _ = real()
    V = video(pattern, vdir)
    if not np.all(np.isfinite(cM[mM])):
        raise FloatingPointError("non-finite model field inside the cup")
    smp = C.Sampler(cM, mM, V["Mt"])
    raw = terms(smp.field_T(), V)
    tr = align(smp, V)
    al = terms(smp.field_T(tr["theta_deg"], tr["dx_D"], tr["dy_D"]), V)
    return dict(loss=al["L"], aligned=al, transform=tr, raw=raw,
                video=dict(n=V["n"], gaps=V["gaps"], dips=V["dips"], white_frac=V["white_frac"], Mt=V["Mt"]))


# =========================================================================== cached forward worker
def _cache_dir():
    return os.environ[VC.ENV_CACHE]


def _inner(args):
    """calib._weval-compatible worker: (params, name, N) -> (name, ev); the final field goes to cache/<key>.npz."""
    params, name, N = args
    cdir = _cache_dir()
    root = os.path.dirname(cdir)
    key = VC.cache_key(params, name, N, VC._WS["spec"])
    try:
        closure = closure_of(params)
        os.makedirs(os.path.join(root, "tmp"), exist_ok=True)
        r = run_fs(name, closure, N, workdir=os.path.join(root, "tmp"))
        led = r["ledger"]
        if not (np.all(np.isfinite(r["c"][r["mask"]])) and all(np.isfinite(float(led[k])) for k in ("layer_error", "milk_error"))):
            raise FloatingPointError("non-finite field or ledger")
        np.savez_compressed(os.path.join(cdir, f"{key}.npz"), c=r["c"].astype(np.float32), mask=r["mask"])
        t1 = time.time()
        lf = lfin(r["c"], r["mask"], name, os.path.join(root, "video"))
        ev = dict(name=name, N=int(N), loss=lf["loss"], failed=False, closure=closure,
                  dice=lf["aligned"]["dice"], iou=lf["aligned"]["iou"], bnd_D=lf["aligned"]["bnd_D"], S=lf["aligned"]["S"],
                  n_model=lf["aligned"]["n_model"], n_video=lf["video"]["n"], white_frac=lf["aligned"]["white_frac"],
                  aligned=lf["aligned"], transform=lf["transform"], raw=lf["raw"], video_white_frac=lf["video"]["white_frac"],
                  inlet=r["inlet"], ledger=dict(T=r["t_end"], layer_error=led["layer_error"], milk_error=led["milk_error"],
                                                deposited=led["deposited"]),
                  stats={k: r["stats"].get(k) for k in ("steps", "frames", "max_substeps", "retries", "clip_mass", "max_u")},
                  solver_params={k: r["solver_params"][k] for k in CLOSURE_KEYS}, kQ=r["kQ"],
                  wall_fwd_s=r["wall_s"], wall_loss_s=time.time() - t1)
        return name, ev
    except Exception as e:                                    # incl. the TimeoutError of the wall-time limit
        return name, dict(name=name, N=int(N), loss=float("inf"), failed=True, error=f"{type(e).__name__}: {e}")


def load_field(cdir, params, name, N, spec=None):
    key = VC.cache_key(params, name, N, spec or make_spec())
    z = np.load(os.path.join(cdir, f"{key}.npz"))
    m = z["mask"]
    return np.where(m, z["c"].astype(np.float64), np.nan), m, key


def seed_cache(src_root, root, jobs, spec=None):
    """copy finished forward calls of `jobs` from another output dir's cache (same key = same spec, closure values,
    pattern, N); returns the number copied."""
    if not src_root:
        return 0
    spec = spec or make_spec()
    src, dst = os.path.join(os.path.abspath(src_root), "cache"), os.path.join(root, "cache")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for params, name, N in jobs:
        key = VC.cache_key(params, name, N, spec)
        if os.path.exists(os.path.join(dst, f"{key}.json")):
            continue
        if os.path.exists(os.path.join(src, f"{key}.json")) and os.path.exists(os.path.join(src, f"{key}.npz")):
            shutil.copy(os.path.join(src, f"{key}.npz"), os.path.join(dst, f"{key}.npz"))
            shutil.copy(os.path.join(src, f"{key}.json"), os.path.join(dst, f"{key}.json"))
            n += 1
    return n


def forward_many(jobs, root, procs, tag, spec=None):
    """jobs = [(params, name, N)] through v05_calibrate's cached / time-limited wrapper around _inner."""
    from multiprocessing import Pool
    spec = spec or make_spec()
    cache = os.path.join(root, "cache")
    for n in sorted({j[1] for j in jobs}):
        video(n, os.path.join(root, "video"))                 # video side once per pattern, before the workers start
    old = VC.TIMEOUT_S
    VC.TIMEOUT_S = TIMEOUT_S
    try:
        with VC.patched(cache, tag, inner=INNER):
            with Pool(max(1, min(procs, len(jobs))), initializer=VC._winit, initargs=(spec,)) as pool:
                outs = pool.map(VC._weval, jobs, chunksize=1)
    finally:
        VC.TIMEOUT_S = old
    return [ev for _, ev in outs]


# =========================================================================== pre-registration
PREREG = dict(
    date="2026-10-13",
    question="physical closure parameters calibrated on tulip + leaf (then frozen) with the GUESSED all-physical actions "
             "FS; the four other patterns are predictions",
    actions="FS of latte_imex/v05_allphys (V05_ALLPHYS_RESULT.md sections 1, 3, 4) exactly as in the stored FS runs: "
            "drvf.py FS <p> --exp P2 --pulse 0.41 --lat exitmin --lag 0.08 --ztr, --flow_phase 0 (tulip, leaf, swan), "
            "pi/2 (layered_heart), keep (heart, push_heart); bridges >= 0.3 s, ramps 0.1 s, spout lag 0.08 s, per-stroke "
            "heights of the FS inverse design (FS_U), Codex's paths / timings / wiggles.  FIXED during calibration and "
            "test; not re-designed for the new closure",
    calibrated=dict(keys=list(CLOSURE8), space={k: list(v) for k, v in SPACE.items()},
                    fixed=FIXED, note="kappa_Q also sets the received flow Q = clip(F / kappa_Q, 1, 32 mL/s) as in "
                                      "drvf.py; clip activity is recorded for every call"),
    loss=dict(version=LOSS_VERSION, frame="final frame only",
              definition="L_fin = (1 - Dice) + d_bnd / 0.05 D + S after the best rigid alignment; calibration loss = "
                         "mean of L_fin over tulip and leaf (equal weights)",
              canonical="compare.py: compare.target (leaf PHI 180 deg, swan 135 deg), compare.Sampler, cup R = 0.49 D",
              thresholds=dict(model=THR_SIM, video=THR_VID),
              alignment=f"argmax IoU(c > {THR_SIM}) over theta in [-{TH_MAX}, {TH_MAX}] step {TH_STEP} deg x dx, dy in "
                        f"[-{SH_MAX}, {SH_MAX}] step {SH_STEP} D (first maximum in loop order theta, dx, dy); the same "
                        "transform for Dice, d_bnd and S.  The IoU grid is computed by a numba kernel with the "
                        "coordinates of compare.Sampler.field_T (test: identical grid, max |diff| = 0, on the six stored "
                        "FS 256^2 fields); Dice, d_bnd and S at the chosen transform use compare.Sampler.field_T itself",
              d_bnd=f"compare.boundary_distance(...)['sym'] [D]; {EMPTY_BND} D when one side has no white boundary",
              S="mean over the structnum.LINES x = -0.10, -0.05, 0, 0.05, 0.10 D of |n_model - n_video| / "
                "max(n_video, 1); n = structnum.stripes(...)['total'] = cf_runner.axis_gaps gaps (>= 1 mm brown between "
                "white) + dips (prominence >= 0.1, FWHM >= 0.5 mm inside the white extent) on that line",
              reported_also="raw (unaligned) IoU, Dice, boundary distance, S; white fraction; gaps / dips separately",
              video="computed once per pattern and cached (DIR/video)"),
    grid_rule=dict(rule="N = 96 for the search IF the FS closure at N = 96 keeps the per-line brown-line counts of N = 256 "
                        "on tulip and leaf within +-1 on every line AND |L_fin(96) - L_fin(256)| <= 0.15 on both; "
                        "otherwise N = 128 with only 2 initial points (FS, FSp2)",
                   counts="the per-line counts n_model that enter L_fin (after each field's own best alignment); the "
                          "raw counts are reported too",
                   decided_in="Build phase (nrule), written into this file as N_decision before any calibration run"),
    cma=dict(optimizer="calib.calibrate (unchanged) inside v05_calibrate.patched (disk cache + wall-time limit)",
             popsize=12, sigma0=0.25, gens=20, seed=1, seeds_per_init="seed + init index (1, 2, 3)",
             inits=["FS closure", "FSp2 closure (B_dep 2149.443, p_dep 2, kappa_r 0.042312, rest as FS)",
                    "centre of the unit box"],
             references="the initial points are evaluated once as references (not candidates, not in the CMA budget)",
             best="lowest calibration loss over all CMA samples (calib.calibrate rec['best'])"),
    failures=f"exception, non-finite field / ledger, or wall > {TIMEOUT_S} s -> +inf (1e6 for CMA), counted",
    test=dict(patterns=list(TEST), calibration_patterns=list(NAMES), N=256,
              what="the frozen best closure and FS on all six patterns with the FS actions (fields .npy + metrics JSON)"),
    never="the test patterns are not loaded by run / smoke and are not used to choose anything",
    note_zh="预注册：动作固定为全物理猜测动作 FS（不随闭合重新设计）；只校准 v05_adapter.SPACE 的 8 个闭合参数（ν、D 固定）；"
            "损失只用末帧：在 compare.py 的规范坐标里先做 ±25°、±0.06 D 的最佳刚体对齐（按 IoU），再算 (1 − Dice) + "
            "边界距离/0.05 D + 五条线上褐线数的相对误差；校准损失 = 郁金香、树叶等权平均。N 的选择规则见 grid_rule。",
)


def _jdump(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=float, ensure_ascii=False)
    os.replace(tmp, path)


def _jload(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _core_sha1(pre):
    core = {k: v for k, v in pre.items() if k not in ("N_decision",)}
    return hashlib.sha1(json.dumps(core, sort_keys=True, default=float).encode()).hexdigest()


def check_unchanged(pre, root, names):
    """refuse to run when something a cached result depends on, but the cache key does not hold, has changed since the
    pre-registration: the PREREG text (loss constants, alignment grid, time limit, CMA setup), the sha1 of the forward
    and loss code (this driver excepted: its later edits are listed in DIR/code_changes*.json) and the video targets
    (key of DIR/video/<p>.json)."""
    bad = [k for k in PREREG if json.loads(json.dumps(PREREG[k], default=float)) != pre.get(k)]
    if pre.get("spec") != json.loads(json.dumps(make_spec())):
        bad.append("spec")
    cs = code_sha1()
    bad += [f"code_sha1.{k}" for k, v in pre["code_sha1"].items() if k != "this" and cs.get(k) != v]
    for p in names:
        meta = _jload(os.path.join(root, "video", f"{p}.json"))
        if meta is not None and meta.get("key") != json.loads(json.dumps(_video_key(p))):
            bad.append(f"video.{p}")
    if bad:
        raise SystemExit(f"changed since the pre-registration (cached results would be stale): {bad}")


def prereg(out):
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    pp = os.path.join(out, "prereg.json")
    if os.path.exists(pp):
        raise SystemExit(f"{pp} exists; a pre-registration is written once")
    pre = dict(PREREG, spec=make_spec(), action_definition=action_definition(), init_points=init_points(),
               code_sha1=code_sha1(), numba_threads=os.environ.get("NUMBA_NUM_THREADS"), N_decision=None,
               written=time.strftime("%Y-%m-%d %H:%M:%S"))
    _jdump(pre, pp)
    print(f"wrote {pp} (core sha1 {_core_sha1(pre)[:12]})")


# =========================================================================== N rule + FS reproduction
def nrule(out, procs=3, Ns=(96, 128, 256)):
    out = os.path.abspath(out)
    pp = os.path.join(out, "prereg.json")
    pre = _jload(pp)
    if pre is None:
        raise SystemExit("write the pre-registration first (prereg)")
    fs = init_points()["FS"]
    jobs = [(fs, n, N) for N in sorted(Ns, reverse=True) for n in NAMES]
    la0 = os.getloadavg()
    t0 = time.time()
    evs = forward_many(jobs, out, procs, "nrule")
    la1 = os.getloadavg()
    res = {}
    for (p, n, N), ev in zip(jobs, evs):
        if ev.get("failed"):
            raise SystemExit(f"FS {n} N {N} failed: {ev.get('error')}")
        res.setdefault(n, {})[N] = ev
    # reproduction: N = 256 fields vs the stored FS runs
    repro = {}
    for n in NAMES:
        if 256 not in res[n]:
            continue
        cM, mM, key = load_field(os.path.join(out, "cache"), fs, n, 256)
        z = np.load(os.path.join(FS_RUNS, f"FS_{n}", "c.npz"))
        cs, ms = z["c"], z["mask"]
        c32 = np.where(mM, cM, np.nan).astype(np.float32)
        same_mask = bool(np.array_equal(ms, mM))
        d = np.abs(c32[ms].astype(np.float64) - cs[ms].astype(np.float64))
        st = _jload(os.path.join(FS_RUNS, f"FS_{n}", "res.json"), {})
        repro[n] = dict(bitwise_equal=bool(np.array_equal(c32, cs, equal_nan=True)), same_mask=same_mask,
                        max_abs_diff=float(d.max()), stored=os.path.join(FS_RUNS, f"FS_{n}", "c.npz"),
                        steps=[res[n][256]["stats"]["steps"], st.get("stats", {}).get("steps")],
                        T=[res[n][256]["ledger"]["T"], st.get("ledger", {}).get("t")],
                        deposited=[res[n][256]["ledger"]["deposited"], st.get("ledger", {}).get("deposited")])
        # L_fin of the stored field itself (same function)
        cS = np.where(ms, cs.astype(np.float64), np.nan)
        repro[n]["L_fin_stored"] = lfin(cS, ms, n, os.path.join(out, "video"))["loss"]
    # rule
    rows, ok96 = {}, True
    for n in NAMES:
        r256 = res[n][256]
        rows[n] = {}
        for N in Ns:
            ev = res[n][N]
            dn = [a - b for a, b in zip(ev["n_model"], r256["n_model"])]
            dn_raw = [a - b for a, b in zip(ev["raw"]["n_model"], r256["raw"]["n_model"])]
            rows[n][N] = dict(L_fin=ev["loss"], dL=ev["loss"] - r256["loss"], n_model=ev["n_model"], dn=dn,
                              n_raw=ev["raw"]["n_model"], dn_raw=dn_raw, dice=ev["dice"], bnd_D=ev["bnd_D"], S=ev["S"],
                              iou_raw=ev["raw"]["iou"], bnd_raw_D=ev["raw"]["bnd_D"], transform=ev["transform"],
                              white_frac=ev["white_frac"], wall_fwd_s=ev["wall_fwd_s"], wall_loss_s=ev["wall_loss_s"],
                              inlet_clip=dict(low=ev["inlet"]["clip_low_time_frac"], high=ev["inlet"]["clip_high_time_frac"]))
        r96 = rows[n].get(96)
        if r96 is None or max(abs(x) for x in r96["dn"]) > 1 or abs(r96["dL"]) > 0.15:
            ok96 = False
    decision = dict(N=96 if ok96 else 128, inits=["FS", "FSp2", "centre"] if ok96 else ["FS", "FSp2"],
                    rule_passed_at_96=ok96, numbers=rows, fs_reproduction=repro,
                    loadavg_start=la0, loadavg_end=la1, wall_s=time.time() - t0, procs=procs,
                    decided=time.strftime("%Y-%m-%d %H:%M:%S"))
    pre = _jload(pp)
    core = _core_sha1(pre)
    if pre.get("N_decision") is not None:
        print("N_decision already in prereg.json; not changed.  Measured now:")
        _jdump(dict(decision, core_sha1=core), os.path.join(out, "nrule_remeasure.json"))
    else:
        pre["N_decision"] = dict(decision, core_sha1=core)
        _jdump(pre, pp)
        assert _core_sha1(_jload(pp)) == core
    _jdump(dict(decision, core_sha1=core), os.path.join(out, "nrule.json"))
    for n in NAMES:
        for N in Ns:
            r = rows[n][N]
            print(f"{n:6s} N {N:3d}  L_fin {r['L_fin']:.4f} (dL {r['dL']:+.4f})  n {r['n_model']} dn {r['dn']}  raw n "
                  f"{r['n_raw']} dn_raw {r['dn_raw']}  Dice {r['dice']:.3f} bnd {r['bnd_D']:.4f} S {r['S']:.3f}  "
                  f"fwd {r['wall_fwd_s']:.0f}s loss {r['wall_loss_s']:.0f}s")
    print("video n:", {n: res[n][256]["n_video"] for n in NAMES})
    print("reproduction:", json.dumps(repro, default=float))
    print(f"decision: N = {decision['N']}, inits {decision['inits']} (rule at 96 {'passed' if ok96 else 'failed'})")


# =========================================================================== calibration run
def run(out, procs=4, smoke=False, prereg_path=None, seed_from=None):
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    pp = os.path.abspath(prereg_path or os.path.join(out, "prereg.json"))
    pre = _jload(pp)
    if pre is None or pre.get("N_decision") is None:
        raise SystemExit(f"{pp}: no pre-registration with an N decision")
    spec = make_spec()
    if pre["spec"] != json.loads(json.dumps(spec)):
        raise SystemExit("the current code's spec (loss version / action definition) differs from the pre-registration")
    check_unchanged(pre, out, NAMES)
    dec = pre["N_decision"]
    N = int(dec["N"])
    pts = init_points()
    x0s = [pts[k] for k in dec["inits"]]
    for k, x in zip(dec["inits"], x0s):
        assert all(abs(x[q] - pre["init_points"][k][q]) <= 1e-12 * max(1.0, abs(x[q])) for q in CLOSURE8), k
    cma = pre["cma"]
    cfg = dict(N=N, popsize=cma["popsize"], sigma0=cma["sigma0"], gens=cma["gens"], seed=cma["seed"], procs=procs,
               inits=dec["inits"], smoke=bool(smoke))
    if smoke:
        cfg.update(gens=1, popsize=4)
    if pp != os.path.join(out, "prereg.json"):
        _jdump(dict(source=pp, core_sha1=_core_sha1(pre), prereg=pre), os.path.join(out, "prereg_used.json"))
    d = os.path.join(out, "calib")
    cj = os.path.join(d, "calib.json")
    old = _jload(cj)
    if old is not None and "best" in old and (old.get("driver") or {}).get("cfg") == json.loads(json.dumps(cfg)):
        print(f"{cj}: complete with the same setup, skipped")
        return
    logp = os.path.join(out, "calib.log")
    cache = os.path.join(out, "cache")
    t0 = time.time()

    def ncache():
        return len([f for f in os.listdir(cache) if f.endswith(".json")]) if os.path.isdir(cache) else 0

    state = dict(n=ncache(), t=t0)

    def log(line):
        nc = ncache()
        s = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')} +{time.time() - t0:7.0f}s] {line} | computed {nc - state['n']} "
             f"in {time.time() - state['t']:.0f}s, cache {nc}, load {os.getloadavg()[0]:.1f}")
        state.update(n=nc, t=time.time())
        print(s, flush=True)
        with open(logp, "a") as fh:
            fh.write(s + "\n")

    log(f"start: names {list(NAMES)}, cfg {cfg}, prereg {pp} (core {_core_sha1(pre)[:12]})")
    jobs = [(x, n, N) for x in x0s for n in NAMES]
    ns = seed_cache(seed_from, out, jobs, spec)
    if ns:
        log(f"seeded {ns} reference call(s) from {seed_from}/cache")
    evs = forward_many(jobs, out, procs, "ref", spec)
    refs = []
    for i, k in enumerate(dec["inits"]):
        e = dict(zip(NAMES, evs[i * len(NAMES):(i + 1) * len(NAMES)]))
        refs.append(dict(init=k, x0=x0s[i], loss=float(np.mean([e[n]["loss"] for n in NAMES])),
                         per_pattern={n: _compact(e[n]) for n in NAMES}))
        log(f"reference {k}: loss {refs[-1]['loss']:.4f} " + " ".join(f"{n} {e[n]['loss']:.4f}" for n in NAMES))
    old_t = VC.TIMEOUT_S
    VC.TIMEOUT_S = TIMEOUT_S
    try:
        with VC.patched(cache, "smoke" if smoke else "cal", inner=INNER):
            rec = CB.calibrate(spec, NAMES, d, x0s, gens=cfg["gens"], popsize=cfg["popsize"], sigma0=cfg["sigma0"], N=N,
                               procs=procs, seed=cfg["seed"], log=log)
    finally:
        VC.TIMEOUT_S = old_t
    recs = [_jload(os.path.join(cache, f)) for f in os.listdir(cache) if f.endswith(".json")]
    recs = [r for r in recs if r]
    rec["driver"] = dict(cfg=cfg, prereg=pp, prereg_core_sha1=_core_sha1(pre), x0_reference=refs, wall_s=time.time() - t0,
                         cache_records=len(recs), cache_failed=sum(bool(r["eval"].get("failed")) for r in recs),
                         compute_s=float(sum(r["wall_s"] for r in recs)), code_sha1=code_sha1(),
                         note="n_forward = forward calls requested by CMA-ES (cache hits included); x0 references are "
                              "extra and not in n_forward")
    _jdump(rec, cj)
    b = rec["best"]
    log(f"done: best {b['loss']:.4f} spread {[round(v, 4) for v in b['spread']]} n_forward {rec['n_forward']} "
        f"n_failed {rec['n_failed']}; best params {json.dumps({k: round(v, 6) for k, v in b['params'].items()})}")


def _compact(ev):
    if ev.get("failed"):
        return dict(loss=ev["loss"], failed=True, error=ev.get("error"))
    return dict(loss=ev["loss"], dice=ev["dice"], bnd_D=ev["bnd_D"], S=ev["S"], n_model=ev["n_model"],
                n_video=ev["n_video"], iou_raw=ev["raw"]["iou"], bnd_raw_D=ev["raw"]["bnd_D"], transform=ev["transform"],
                clip_low=ev["inlet"]["clip_low_time_frac"], clip_high=ev["inlet"]["clip_high_time_frac"],
                V_received_mL=ev["inlet"]["V_received_mL"], wall_fwd_s=ev["wall_fwd_s"], wall_loss_s=ev["wall_loss_s"])


# =========================================================================== status
def status(out):
    out = os.path.abspath(out)
    cache = os.path.join(out, "cache")
    recs = [_jload(os.path.join(cache, f)) for f in os.listdir(cache) if f.endswith(".json")] if os.path.isdir(cache) else []
    recs = [r for r in recs if r]
    fail = [r for r in recs if r["eval"].get("failed")]
    print(f"cache {len(recs)} records, failed {len(fail)}")
    if recs:
        w = np.array([r["wall_s"] for r in recs])
        print(f"wall per call: median {np.median(w):.1f} s, max {w.max():.1f} s; first {min(r['t_start'] for r in recs):.0f}"
              f" last {max(r['t_end'] for r in recs):.0f}")
    rec = _jload(os.path.join(out, "calib", "calib.json"))
    if rec is None:
        print("no calib.json")
        return
    pre = _jload(os.path.join(out, "prereg.json")) or (_jload(os.path.join(out, "prereg_used.json")) or {}).get("prereg")
    print(f"{'COMPLETE' if 'best' in rec else 'running'}: N {rec['N']} gens {rec['gens']} popsize {rec['popsize']} "
          f"n_forward {rec['n_forward']} n_failed {rec['n_failed']}")
    for i, r in enumerate(rec["inits"]):
        h = r["history"]
        if h:
            print(f"  init {i}: gens {len(h)}/{rec['gens']} best {r['best']['loss']:.4f} (gen {r['best']['gen']}) "
                  f"last min {h[-1]['loss_min']:.4f} med {h[-1]['loss_med']:.4f} sigma {h[-1]['sigma']:.3f}")
    if pre and pre.get("N_decision") and "best" not in rec and recs:
        n_inits = len(pre["N_decision"]["inits"])
        total = n_inits * rec["gens"] * rec["popsize"] * len(rec["names"])
        done = rec["n_forward"]
        span = max(r["t_end"] for r in recs) - min(r["t_start"] for r in recs)
        if done:
            print(f"  progress {done}/{total} forward calls; ETA ~{(total - done) * span / max(len(recs), 1) / 3600:.1f} h")


# =========================================================================== evaluate (frozen best + FS, six patterns)
def evaluate(out, procs=4, N=256, seed_from=None):
    out = os.path.abspath(out)
    rec = _jload(os.path.join(out, "calib", "calib.json"))
    if rec is None or "best" not in rec:
        raise SystemExit("no finished calibration in DIR/calib/calib.json")
    pre = _jload(os.path.join(out, "prereg.json")) or (_jload(os.path.join(out, "prereg_used.json")) or {}).get("prereg")
    if pre is None:
        raise SystemExit("no pre-registration in DIR (prereg.json or prereg_used.json)")
    check_unchanged(pre, out, PATTERNS)
    closures = dict(best={k: float(rec["best"]["params"][k]) for k in CLOSURE8}, FS=init_points()["FS"])
    jobs = [(closures[t], p, N) for t in closures for p in PATTERNS]
    t0 = time.time()
    ns = seed_cache(seed_from, out, jobs)
    if ns:
        print(f"seeded {ns} call(s) from {seed_from}/cache")
    evs = forward_many(jobs, out, procs, "evaluate")
    ed = os.path.join(out, "evaluate")
    os.makedirs(ed, exist_ok=True)
    res = dict(N=N, closures={t: closure_of(v) for t, v in closures.items()}, calibration_patterns=list(NAMES),
               test_patterns=list(TEST), calib_best_loss=rec["best"]["loss"], wall_s=time.time() - t0, patterns={})
    for (params, p, _), ev, t in zip(jobs, evs, [t for t in closures for _ in PATTERNS]):
        res["patterns"].setdefault(p, {})[t] = ev
        if not ev.get("failed"):
            cM, mM, key = load_field(os.path.join(out, "cache"), params, p, N)
            np.save(os.path.join(ed, f"{t}_{p}_N{N}.npy"), np.where(mM, cM, np.nan).astype(np.float32))
    for t in closures:
        L = {p: res["patterns"][p][t]["loss"] for p in PATTERNS}
        res[f"summary_{t}"] = dict(L_fin=L, calibration_mean=float(np.mean([L[p] for p in NAMES])),
                                   test_mean=float(np.mean([L[p] for p in TEST])))
    _jdump(res, os.path.join(ed, "metrics.json"))
    for t in closures:
        s = res[f"summary_{t}"]
        print(f"{t:5s} cal {s['calibration_mean']:.4f} test {s['test_mean']:.4f} "
              + " ".join(f"{p} {s['L_fin'][p]:.3f}" for p in PATTERNS))


# =========================================================================== unit tests
def test(out=None, quick=False):
    out = os.path.abspath(out or tempfile.mkdtemp(prefix="v05gcal_test_"))
    os.makedirs(out, exist_ok=True)
    rep = {}
    # 1. the embedded action set = the stored FS runs' arguments and the P2 inverse design
    e = _jload(ALLPHYS_EXP)
    if e is not None:
        for p in PATTERNS:
            assert e["P2"]["inlet"][p]["U"] == FS_U[p], p
            assert e["P2"]["inlet"][p]["kappa_Q"] == FS["kappa_Q"], p
        assert all(e["P2"]["params"][k] == FS[k] for k in ("B_dep", "p_dep", "kappa_r"))
    for p in PATTERNS:
        a = (_jload(os.path.join(FS_RUNS, f"FS_{p}", "res.json")) or {}).get("args")
        if a is None:
            continue
        fp = None if a["flow_phase"] == "keep" else float(a["flow_phase"])
        assert fp == FLOW_PHASE[p] and a["exp"] == ACTION["exp"] and a["fit"] is None, p
        for k in ("pulse", "lat", "lag", "ramp", "stop", "ztr"):
            assert a[k] == ACTION[k], (p, k)
        assert a["zset"] == a["diag_uh"] == a["diag_r"] == "" and a["kr_scale"] == 1.0 and a["N"] == 256
        assert (a["cp"], a["beta"], a["nu"]) == (FS["cp"], FS["beta"], FS["nu"]), p
    rep["actions_match_stored_FS_args"] = True
    # 2. the override reaches the solver (and kappa_Q reaches the received flow)
    X = dict(cp=1.11, beta=2.22, nu=1.3e-3, D=2e-7, kappa_Q=1.7, kappa_c=0.55, kappa_t=0.45, kappa_r=0.05, B_dep=900.0,
             p_dep=3.3)
    r = run_fs("tulip", X, 48, T_max=1.0)
    for k in CLOSURE_KEYS:
        assert r["solver_params"][k] == X[k], (k, r["solver_params"][k])
    assert r["solver_params"]["closure"] == "chi" and r["solver_params"]["return_law"] == "v05"
    assert r["kQ"] == 1.7
    L = r["log"]
    assert len(L) > 0
    qraw = L[:, 16] / 1.7 * ML                              # F / kappa_Q in mL/s
    assert np.allclose(L[:, 4], qraw, rtol=1e-12) and np.allclose(L[:, 5], np.clip(qraw, 1.0, 32.0), rtol=1e-12)
    r_fs = run_fs("tulip", closure_of({}), 48, T_max=1.0)
    rr = run_fs("tulip", dict(closure_of({}), cp=1.2), 48, T_max=1.0)
    d_cp = float(np.nanmax(np.abs(rr["c"] - r_fs["c"])))
    assert d_cp > 0
    rep["override"] = dict(solver_params=r["solver_params"], kQ=r["kQ"], max_dc_cp_1p36_to_1p2=d_cp)
    # 3. identity closure = drvf.py as a subprocess (copy of the driver + the P2 design in a scratch tree), bit for bit
    if e is not None:
        tree = os.path.join(out, "drvf_tree")
        os.makedirs(os.path.join(tree, "final"), exist_ok=True)
        shutil.copy(DRVF, os.path.join(tree, "final", "drvf.py"))
        shutil.copy(ALLPHYS_EXP, os.path.join(tree, "allphys_experiments.json"))
        rep["identity"] = {}
        for p, fpa in (("tulip", "0"), ("layered_heart", "1.5707963267948966"), ("heart", None)):
            cmd = [sys.executable, "-I", os.path.join(tree, "final", "drvf.py"), "T", p, "--exp", "P2", "--pulse", "0.41",
                   "--lat", "exitmin", "--lag", "0.08", "--ztr", "--N", "48"] + (["--flow_phase", fpa] if fpa else [])
            env = dict(os.environ, NUMBA_NUM_THREADS="1")
            fz = os.path.join(tree, "final", "runs", f"T_{p}", "c.npz")
            if os.path.exists(fz):
                os.remove(fz)
            pr = subprocess.run(cmd, capture_output=True, text=True, env=env)
            # drvf.main saves c.npz, then compares with the 256^2 V0.5 reference (IoU), which fails for N != 256 with a
            # broadcast error; only that failure is accepted
            assert pr.returncode == 0 or "could not be broadcast together with shapes (48,48) (256,256)" in pr.stderr, pr.stderr[-2000:]
            z = np.load(fz)
            mine = run_fs(p, closure_of({}), 48)
            c32 = mine["c"].astype(np.float32)
            eq = bool(np.array_equal(c32, z["c"], equal_nan=True))
            rep["identity"][p] = dict(bitwise_equal=eq, max_abs_diff=float(np.nanmax(np.abs(c32 - z["c"]))))
            assert eq, (p, rep["identity"][p])
    # 4. loss sanity: the real whiteness as the model field (through the whole model path: M frame, Sampler, alignment)
    #    f = w - 0.1 above w = 0.5 and 0.8 w below: c > 0.4 <=> w > 0.5, exact zeros of the clipped w stay exact zeros.
    #    (Plain w - 0.1 gives Dice 1 and d_bnd 0 too, but its brown plateaus -0.1 come back from the bilinear line
    #    sampling as two rounded values, and every run of the larger one is a separate maximum of -c with the full
    #    prominence for scipy.find_peaks (ties do not block each other): many extra 'dips'.  A solver field has no
    #    exact non-zero plateaus; its exact zeros (never-white cells) interpolate to exact zeros.)
    vdir = os.path.join(out, "video")
    rep["loss_sanity"] = {}
    for p in (NAMES if quick else PATTERNS):
        V = video(p, vdir)
        w, wm = V["w"], V["wm"]
        mM = wm[::-1, :]
        f = np.where(w > THR_VID, w - (THR_VID - THR_SIM), (THR_SIM / THR_VID) * w)
        a = lfin(np.where(wm, f, np.nan)[::-1, :], mM, p, vdir)
        tr = (a["transform"]["theta_deg"], a["transform"]["dx_D"], a["transform"]["dy_D"])
        assert a["aligned"]["dice"] == 1.0 and a["aligned"]["bnd_D"] == 0.0 and tr == (0.0, 0.0, 0.0), (p, a["aligned"], tr)
        assert a["aligned"]["S"] <= 0.05 and max(abs(x - y) for x, y in zip(a["aligned"]["n_model"], V["n"])) <= 1, p
        if p == "tulip":
            assert a["loss"] == 0.0, a["aligned"]
        a2 = lfin(np.where(wm, w - (THR_VID - THR_SIM), np.nan)[::-1, :], mM, p, vdir)
        b = lfin(np.where(wm, w, np.nan)[::-1, :], mM, p, vdir)          # w itself (model threshold 0.4 vs video 0.5)
        z = lfin(np.where(mM, 0.0, np.nan), mM, p, vdir)                 # blank model
        assert a2["aligned"]["dice"] == 1.0 and a2["aligned"]["bnd_D"] == 0.0
        assert z["aligned"]["dice"] == 0.0 and z["aligned"]["bnd_D"] == EMPTY_BND
        rep["loss_sanity"][p] = dict(
            video_as_model=dict(L=a["loss"], dice=a["aligned"]["dice"], bnd_D=a["aligned"]["bnd_D"], S=a["aligned"]["S"],
                                n=a["aligned"]["n_model"], n_video=V["n"], transform=tr),
            w_minus_0p1=dict(L=a2["loss"], dice=a2["aligned"]["dice"], bnd_D=a2["aligned"]["bnd_D"], S=a2["aligned"]["S"],
                             n=a2["aligned"]["n_model"]),
            w_itself=dict(L=b["loss"], dice=b["aligned"]["dice"], bnd_D=b["aligned"]["bnd_D"], S=b["aligned"]["S"],
                          n=b["aligned"]["n_model"], transform=b["transform"], raw_L=b["raw"]["L"]),
            blank=dict(L=z["loss"], S=z["aligned"]["S"]))
    # 5. the fast IoU search = the reference operator (compare.Sampler.field_T) on stored FS fields
    C, _ = real()
    rep["iou_grid"] = {}
    for p in (NAMES if quick else PATTERNS):
        f = os.path.join(FS_RUNS, f"FS_{p}", "c.npz")
        if not os.path.exists(f):
            continue
        cM, mM = C.load_sim(f)
        V = video(p, vdir)
        smp = C.Sampler(cM, mM, V["Mt"])
        F, E = iou_grid_fast(smp, V), iou_grid_exact(smp, V)
        d = float(np.abs(F - E).max())
        rep["iou_grid"][p] = dict(max_abs_diff=d, argmax_equal=bool(np.argmax(F) == np.argmax(E)), best=float(E.max()))
        assert d == 0.0 and np.argmax(F) == np.argmax(E), (p, d)
    # 6. cache key: changes with every input
    spec = make_spec()
    base = init_points()["FS"]
    k0 = VC.cache_key(base, "tulip", 96, spec)
    keys = {"base": k0}
    for k in CLOSURE8:
        keys[k] = VC.cache_key(dict(base, **{k: base[k] * (1 + 1e-12) + 1e-15}), "tulip", 96, spec)
    keys["pattern"] = VC.cache_key(base, "leaf", 96, spec)
    keys["N"] = VC.cache_key(base, "tulip", 128, spec)
    keys["loss"] = VC.cache_key(base, "tulip", 96, dict(spec, loss="Lfin-x"))
    keys["actions"] = VC.cache_key(base, "tulip", 96, dict(spec, actions_id="0" * 40))
    keys["fixed"] = VC.cache_key(base, "tulip", 96, dict(spec, fixed=dict(FIXED, nu=2e-3)))
    assert len(set(keys.values())) == len(keys), keys
    assert VC.cache_key(dict(base), "tulip", 96, make_spec()) == k0             # and is stable
    rep["cache_key"] = dict(n_inputs=len(keys) - 1, all_distinct=True)
    # 7. what the key does not hold is guarded: PREREG text, spec, code sha1 (driver excepted), video targets
    pre = json.loads(json.dumps(dict(PREREG, spec=make_spec(), code_sha1=code_sha1()), default=float))
    check_unchanged(pre, out, PATTERNS)
    refused = 0
    for bad in (dict(pre, loss=dict(pre["loss"], thresholds=dict(model=0.45, video=THR_VID))),
                dict(pre, failures="x"), dict(pre, cma=dict(pre["cma"], popsize=8)),
                dict(pre, code_sha1=dict(pre["code_sha1"], solver="0" * 40)), dict(pre, spec=dict(pre["spec"], loss="x"))):
        try:
            check_unchanged(bad, out, PATTERNS)
        except SystemExit:
            refused += 1
    assert refused == 5, refused
    assert closure_of({})["nu"] == FIXED["nu"] and closure_of({})["D"] == FIXED["D"]
    rep["guard"] = dict(passes_unchanged=True, refused_changed=refused)
    _jdump(rep, os.path.join(out, "test_report.json"))
    print(json.dumps(rep, indent=1, default=float))
    print("ALL TESTS PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["prereg", "nrule", "run", "status", "evaluate", "test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--quick", action="store_true", help="test: iou-grid check on tulip + leaf only")
    ap.add_argument("--seed_from", default=None, help="run / evaluate: copy finished calls with the same key from "
                                                     "DIR2/cache (e.g. the FS calls of nrule)")
    a = ap.parse_args()
    if a.cmd == "test":
        test(a.out, quick=a.quick)
        return
    if not a.out:
        raise SystemExit("--out DIR")
    if a.cmd == "prereg":
        prereg(a.out)
    elif a.cmd == "nrule":
        nrule(a.out, procs=a.procs)
    elif a.cmd == "run":
        run(a.out, procs=a.procs, smoke=a.smoke, prereg_path=a.prereg, seed_from=a.seed_from)
    elif a.cmd == "status":
        status(a.out)
    elif a.cmd == "evaluate":
        evaluate(a.out, procs=a.procs, N=a.N, seed_from=a.seed_from)


if __name__ == "__main__":
    main()
