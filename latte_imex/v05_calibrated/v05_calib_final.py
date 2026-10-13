"""V0.5 calibration with the barista's real actions (senior V1 5.1-5.4): the final step after the review.

It produces the numbers, tables and the figure of CALIBRATION_RESULT.md that were not yet in calibration.md /
prediction.md (review verify.md, issues 1-12), and copies the results to latte_imex/v05_calibrated/.

Nothing here is a new calibration, and no test pattern (heart, push_heart, layered_heart, swan) is used to choose
anything: the forward calls below run on the calibration patterns only (tulip, leaf); the test-pattern numbers are read
from the frozen-parameter runs that already exist.

Inputs (defaults = the scratch directories of this task; all read-only):
  CAL      v05calib/: prereg.json, calib_AB/calib.json, calib_A/calib.json (v05_calibrate), cache/ (2130 compact N = 96
           evaluations), runs/<set>_<name>_N128/{wT.npz, eval.json} and runs/<set>_<name>_N256_final/ (v05_predict)
  VERIFY   CAL/verify_work/: the review's recompute of the A init 2 / 3 best points (recompute_Ainits_N128.json and
           wT_A_i<k>_<name>_N128.npz) and its finite differences around the AB best point (fd_rel0.1.json, fd_extra.json)
  BASE     v05video/: the zero-tuning FSp2 fields (runs/FSp2_<name>_N128/wT.npz, runs/FSp2_<name>_N256/wT.npz)

    python3 latte_imex/v05_calibrated/v05_calib_final.py probe   [--procs 4]
    python3 latte_imex/v05_calibrated/v05_calib_final.py mech    [--procs 4]
    python3 latte_imex/v05_calibrated/v05_calib_final.py analyze
    python3 latte_imex/v05_calibrated/v05_calib_final.py figs
    python3 latte_imex/v05_calibrated/v05_calib_final.py export  [--dest latte_imex/v05_calibrated]

probe   forward calls (calib.simulate_clip + calib.evaluate(delay_in_loss=False); deterministic, NUMBA_NUM_THREADS = 1):
        (a) A best point: each of the ten parameters x (1 +- 0.10), tulip, N = 96, plus the centre -> flat directions and
            compensating pairs (same method as the review's fd_analyse.py for the AB best point);
        (b) AB best point: three of the review's probes repeated (kappa_t +10 %, Q_scale +2 %, B_dep = 2149), tulip +
            leaf, N = 96 -> reproduction of the review's finite differences;
        (c) A init 2 best point (B_dep and Q_scale at the upper bound): centre, B_dep -10 %, Q_scale -10 %, tulip, N = 96;
        (d) POST-HOC diagnostic, never used to replace the frozen parameter sets: the 10 lowest-loss CMA samples of AB
            (tulip + leaf) and of A (tulip), from a replay of the recorded CMA runs out of the cache, re-run at N = 128
            on their own calibration patterns -> how much of the N = 96 ranking survives one grid step.
        -> FINAL/probe.json
mech    POST-HOC mechanism diagnostic on the calibration patterns only (tulip, leaf), N = 128: the frozen AB / A sets with
        beta, or the exit momentum (u_exit, kappa_t), put back to the zero-tuning FSp2 values (and FSp2 with AB's beta)
        -> white centroid offset along B -> D, area ratio, loss parts -> FINAL/mech.json, FINAL/mech/*.npz
analyze tables for the seven parameter sets FSp2 (zero tuning), AB, A, AB_i1, AB_i3, A_i2, A_i3: calibration fit and
        test prediction (N = 128), test means with and without heart, AB - A paired by init, loss decomposition;
        white centroid offset along B -> D (final frame N = 256 / 128, process mean N = 128) and white-area ratio;
        boundary-length ratio; row peak counts of the right lobe of layered_heart; received milk vs cup level;
        flow-weighted chi per stage kind; finite-difference tables (AB: the review's data re-analysed here; A: probe)
        -> FINAL/final_tables.json, FINAL/final_tables.md
figs    FINAL/offset_fd.png
export  copies the note, figures, calibration records, drivers and videos to --dest
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from latte_imex import calib as CB          # noqa: E402

SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review"
D_CAL = os.path.join(SCR, "v05calib")
D_BASE = os.path.join(SCR, "v05video")

SPEC = dict(kind='v05', base='FSp2', loss_delay=False,
            space_extra=dict(u_exit=(0.10, 0.27, 'lin'), Q_scale=(0.5, 2.0, 'log')))
PAT = ("tulip", "leaf", "heart", "push_heart", "layered_heart", "swan")
TEST = tuple(CB.TEST)
CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="郁金香", leaf="树叶", swan="天鹅")
GROUPS = ("FSp2", "AB", "A", "AB_i1", "AB_i3", "A_i2", "A_i3")
GCN = dict(FSp2="零调参 FSp2", AB="AB（起点 2，冻结）", A="A（起点 1，冻结）", AB_i1="AB 起点 1", AB_i3="AB 起点 3",
           A_i2="A 起点 2", A_i3="A 起点 3")
A_CUP_CM2 = np.pi * 4.0 ** 2          # D = 8 cm
REL = 0.10
TOPK = 10


def _paths(a):
    cal = a.cal
    return dict(cal=cal, base=a.base, verify=os.path.join(cal, "verify_work"), final=os.path.join(cal, "final"))


def _jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    os.replace(tmp, path)


def _sha(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:8]


# =========================================================================== loss parts (calib.loss_of, delay=False)
def parts(ev):
    """1 - Dice, boundary term, fine-line term of calib.loss_of(delay=False); works on full and compact evaluations."""
    d = ev["dice"]["value"]
    b = ev["boundary_D"]["value"]
    pr = [p for p in ev["profiles"] if p["at"] != "final"]
    line = (float(np.mean([abs(p["n_white_model"] - p["n_white_video"]) / max(p["n_white_video"], 1) for p in pr]))
            if pr else 0.0)
    fw = ev["events"]["first_white"] if "events" in ev else ev.get("first_white", {})
    led = ev.get("ledger") or {}
    return dict(loss=float(ev["loss"]), one_minus_dice=1.0 - d if d is not None else 1.0,
                bnd_term=(b if b is not None else CB.EMPTY_BND) / CB.BND_SCALE,
                bnd_mm=b * 80 if b is not None else None, line_term=line,
                iou=ev["iou"]["value"], mae=ev["coverage_mae"]["value"], final_iou=ev["final"]["iou"],
                final_bnd_mm=(ev["final"]["bnd_D"] or 0) * 80, first_white_err_s=fw.get("value"),
                stage_lines=[(p["at"], p["n_white_video"], p["n_white_model"]) for p in pr],
                V_received_mL=led.get("V_received_mL"), chi=led.get("chi_flow_weighted"),
                ledger_ok=led.get("ok"), layer_error=led.get("layer_error"), milk_error=led.get("milk_error"))


# =========================================================================== probe (forward calls)
_M = {}


def _winit():
    _M["model"] = CB.make_model(SPEC)


def _job(task):
    lab, params, name, N, frames = task
    clip = CB.load_clip(name)
    t0 = time.time()
    wT, led = CB.simulate_clip(_M["model"], params, clip, N)
    ev = CB.evaluate(clip, wT, led, delay_in_loss=False)
    r = parts(ev)
    r["wall_s"] = time.time() - t0
    if frames:
        r["frames"] = [dict(t=f["t"], w=f["w"], dice=f["dice"], bnd=f["bnd_D"]) for f in ev["frames"]]
    return lab, name, N, json.loads(json.dumps(r, default=float))


def _canon(p):
    return tuple(sorted((k, repr(float(v))) for k, v in p.items()))


def cache_index(cal):
    idx = {}
    d = os.path.join(cal, "cache")
    for f in os.listdir(d):
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(d, f)))
        idx[(r["name"], int(r["N"]), _canon(r["params"]))] = float(r["eval"]["loss"])
    return idx


def cma_candidates(cal, tag, idx):
    """Replay the recorded CMA run of `tag` from the cache (same seeds, same x0) -> every candidate with its loss."""
    import cma
    rec = json.load(open(os.path.join(cal, f"calib_{tag}", "calib.json")))
    space = CB.make_model(SPEC).space
    names = CB.CAL_AB if tag == "AB" else CB.CAL_A
    out = []
    for ii, ini in enumerate(rec["inits"]):
        es = cma.CMAEvolutionStrategy(CB.encode(ini["x0"], space), rec["sigma0"],
                                      dict(bounds=[0, 1], popsize=rec["popsize"], seed=rec["driver"]["seed"] + ii,
                                           maxiter=rec["gens"], verbose=-9))
        g = 0
        while not es.stop():
            X = es.ask()
            P = [CB.decode(x, space) for x in X]
            L = [float(np.mean([idx[(n, rec["N"], _canon(p))] for n in names])) for p in P]
            es.tell(X, L)
            out += [dict(init=ii + 1, gen=g + 1, params=p, loss96=l) for p, l in zip(P, L)]
            g += 1
    best = min(out, key=lambda c: c["loss96"])
    assert abs(best["loss96"] - rec["best"]["loss"]) < 1e-12, (tag, best["loss96"], rec["best"]["loss"])
    assert all(abs(best["params"][k] - rec["best"]["params"][k]) < 1e-12 for k in best["params"])
    return out


def probe(a):
    P = _paths(a)
    os.makedirs(P["final"], exist_ok=True)
    ab = json.load(open(os.path.join(P["cal"], "calib_AB", "calib.json")))
    ar = json.load(open(os.path.join(P["cal"], "calib_A", "calib.json")))
    pA, pAB, pAi2 = dict(ar["best"]["params"]), dict(ab["best"]["params"]), dict(ar["inits"][1]["best"]["params"])
    tasks, points = [], {}

    def add(lab, p, names, N, frames=False):
        points[lab] = dict(p)
        tasks.extend((lab, dict(p), n, N, frames) for n in names)

    # (a) A best point, +-10 %, tulip, N = 96
    add("A:centre", pA, CB.CAL_A, 96, True)
    for k in pA:
        for s in (+1, -1):
            q = dict(pA)
            q[k] = pA[k] * (1 + s * REL)
            add(f"A:{k}{'+' if s > 0 else '-'}", q, CB.CAL_A, 96, True)
    # (b) AB best point: three of the review's probes
    for lab, k, v in (("AB:kappa_t+10%", "kappa_t", pAB["kappa_t"] * 1.10), ("AB:Q_scale+2%", "Q_scale", pAB["Q_scale"] * 1.02),
                      ("AB:B_dep=2149", "B_dep", float(json.load(open(os.path.join(P["cal"], "prereg.json")))["x0s"][0]["B_dep"]))):
        q = dict(pAB)
        q[k] = v
        add(lab, q, CB.CAL_AB, 96)
    # (c) A init 2 best point
    add("Ai2:centre", pAi2, CB.CAL_A, 96)
    for k in ("B_dep", "Q_scale"):
        q = dict(pAi2)
        q[k] = pAi2[k] * 0.9
        add(f"Ai2:{k}-10%", q, CB.CAL_A, 96)
    # (d) post-hoc top-10 re-evaluation at N = 128 on the calibration patterns
    idx = cache_index(P["cal"])
    top = {}
    for tag in ("AB", "A"):
        cand = cma_candidates(P["cal"], tag, idx)
        cand.sort(key=lambda c: c["loss96"])
        top[tag] = cand[:TOPK]
        for r, c in enumerate(top[tag], 1):
            add(f"top{tag}:{r}", c["params"], CB.CAL_AB if tag == "AB" else CB.CAL_A, 128)
    tasks.sort(key=lambda t: (-t[3], 0 if t[2] == "tulip" else 1))      # N = 128 and tulip first (longest)
    from multiprocessing import Pool
    t0 = time.time()
    res = []
    with Pool(a.procs, initializer=_winit) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, tasks), 1):
            res.append(r)
            if i % 8 == 0 or i == len(tasks):
                print(f"[probe] {i}/{len(tasks)} done, {time.time() - t0:.0f} s", flush=True)
    ev = {}
    for lab, n, N, r in res:
        ev.setdefault(lab, {})[n] = r
    out = dict(points=points, evals=ev, top={t: [dict(init=c["init"], gen=c["gen"], loss96=c["loss96"]) for c in v]
                                             for t, v in top.items()},
               n_calls=len(tasks), wall_s=time.time() - t0, procs=a.procs, rel=REL,
               provenance=dict(driver=_sha(os.path.abspath(__file__)), calib=_sha(CB.__file__),
                               v05_adapter=_sha(os.path.join(REPO, "latte_imex", "v05_adapter.py")),
                               solver=_sha(os.path.join(REPO, "latte_imex", "solver.py"))))
    _jdump(out, os.path.join(P["final"], "probe.json"))
    print(f"[probe] {len(tasks)} forward calls in {time.time() - t0:.0f} s", flush=True)


# =========================================================================== mechanism probe (calibration patterns only)
MECH = (("AB|beta=3.0", "AB", dict(beta=3.0)),
        ("AB|u_exit=0.15,kappa_t=0.7", "AB", dict(u_exit=0.15, kappa_t=0.7)),
        ("AB|beta=3.0,u_exit=0.15,kappa_t=0.7", "AB", dict(beta=3.0, u_exit=0.15, kappa_t=0.7)),
        ("A|beta=3.0", "A", dict(beta=3.0)),
        ("FSp2|beta=1.556", "FSp2", dict(beta=1.5559599288009482)))


def _mjob(task):
    lab, params, name, N, path = task
    clip = CB.load_clip(name)
    t0 = time.time()
    wT, led = CB.simulate_clip(_M["model"], params, clip, N)
    ev = CB.evaluate(clip, wT, led, delay_in_loss=False)
    np.savez_compressed(path, wT=wT.astype(np.float32), t=np.asarray(clip["res"]["t"]))
    r = parts(ev)
    r["wall_s"] = time.time() - t0
    r["offsets"] = offsets(clip, wT[-1].astype(float), wT.astype(float))
    return lab, name, json.loads(json.dumps(r, default=float))


def mech(a):
    """Which closure change moves the white downstream?  One- and two-parameter swaps between the frozen AB / A sets
    and the zero-tuning FSp2 (beta; the exit momentum u_exit, kappa_t), on the CALIBRATION patterns only (tulip, leaf),
    N = 128; white centroid offset along B -> D and the loss parts.  Post-hoc diagnostic, not a selection."""
    P = _paths(a)
    ab = json.load(open(os.path.join(P["cal"], "calib_AB", "calib.json")))
    ar = json.load(open(os.path.join(P["cal"], "calib_A", "calib.json")))
    pre = json.load(open(os.path.join(P["cal"], "prereg.json")))
    base = dict(AB=ab["best"]["params"], A=ar["best"]["params"], FSp2=pre["x0s"][0])
    d = os.path.join(P["final"], "mech")
    os.makedirs(d, exist_ok=True)
    tasks, points = [], {}
    for lab, b, ch in MECH:
        q = dict(base[b])
        q.update(ch)
        points[lab] = q
        tasks += [(lab, q, n, 128, os.path.join(d, f"{lab.replace('|', '__').replace(',', '_')}_{n}_N128.npz"))
                  for n in CB.CAL_AB]
    tasks.sort(key=lambda t: 0 if t[2] == "tulip" else 1)
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.procs, initializer=_winit) as pool:
        res = pool.map(_mjob, tasks, chunksize=1)
    ev = {}
    for lab, n, r in res:
        ev.setdefault(lab, {})[n] = r
    _jdump(dict(points=points, evals=ev, wall_s=time.time() - t0, n_calls=len(tasks)), os.path.join(P["final"], "mech.json"))
    for lab in points:
        print(lab, " | ".join(f"{n}: L {ev[lab][n]['loss']:.3f} offset {ev[lab][n]['offsets']['final']['along_mm']:+.1f} mm "
                              f"(proc {ev[lab][n]['offsets']['process']['along_mm']:+.1f}) area x{ev[lab][n]['offsets']['final']['area_ratio']:.2f}"
                              for n in CB.CAL_AB), flush=True)
    print(f"[mech] {len(tasks)} calls, {time.time() - t0:.0f} s")


# =========================================================================== analysis helpers
def load_eval(P, g, p):
    if g in ("A_i2", "A_i3"):
        d = json.load(open(os.path.join(P["verify"], "recompute_Ainits_N128.json")))
        return d["evals"][g][p]
    if g == "FSp2":
        e = json.load(open(os.path.join(P["cal"], "runs", f"FSp2v_{p}_N128", "eval.json")))
        if not e.get("ledger"):        # the stored-field re-evaluation has no ledger: take the same-path recompute's
            e["ledger"] = json.load(open(os.path.join(P["cal"], "runs", f"FSp2_{p}_N128", "eval.json")))["ledger"]
        return e
    return json.load(open(os.path.join(P["cal"], "runs", f"{g}_{p}_N128", "eval.json")))


def field128(P, g, p):
    if g == "FSp2":
        f = os.path.join(P["base"], "runs", f"FSp2_{p}_N128", "wT.npz")
    elif g in ("A_i2", "A_i3"):
        f = os.path.join(P["verify"], f"wT_{g}_{p}_N128.npz")
    else:
        f = os.path.join(P["cal"], "runs", f"{g}_{p}_N128", "wT.npz")
    return np.load(f)["wT"].astype(float)


def final256(P, g, p):
    if g == "FSp2":
        f = os.path.join(P["base"], "runs", f"FSp2_{p}_N256", "wT.npz")
    else:
        f = os.path.join(P["cal"], "runs", f"{g}_{p}_N256_final", "wT.npz")
    return np.load(f)["wT"][-1].astype(float)


def _grid(clip):
    M = clip["M"]
    x = (np.arange(M) + .5) / M - .5
    return np.meshgrid(x, x)


def _bd_axis(clip):
    a = np.array(clip["profiles"][0]["p1_T"], float) - np.array(clip["profiles"][0]["p0_T"], float)
    return a / np.linalg.norm(a)


def centroid(w, v, X, Y):
    m = (w > 0.5) & v
    if m.sum() == 0:
        return None, 0.0
    return np.array([X[m].mean(), Y[m].mean()]), m.sum() / v.sum()


def offsets(clip, w_final, w_proc=None):
    """White centroid offset model - video along the clip's B -> D profile [mm] (+ = downstream), |offset| [mm], white
    area ratio model / video; final frame and (w_proc given) the dt-weighted mean over calib's shape frames.
    Same definition as v05video/extra/centroid2.py and the review's centroid_cal.py."""
    res = clip["res"]
    disk = res["disk"].astype(bool)
    X, Y = _grid(clip)
    ax = _bd_axis(clip)
    vT = res["valid"][-1] & disk
    cv, av = centroid(res["white"][-1].astype(float), vT, X, Y)
    cm, am = centroid(w_final, vT, X, Y)
    d = cm - cv
    out = dict(final=dict(along_mm=float(d @ ax) * 80, abs_mm=float(np.hypot(*d)) * 80, area_ratio=float(am / av)))
    if w_proc is not None:
        dt = np.gradient(res["t"])
        al, ar, ws = [], [], []
        for j in np.flatnonzero(clip["shape_ok"]):
            v = res["valid"][j] & disk
            cvj, avj = centroid(res["white"][j].astype(float), v, X, Y)
            cmj, amj = centroid(w_proc[j], v, X, Y)
            if cvj is None or cmj is None or avj < 0.02:
                continue
            al.append(float((cmj - cvj) @ ax) * 80)
            ar.append(amj / avj)
            ws.append(dt[j])
        ws = np.array(ws)
        out["process"] = dict(along_mm=float(np.sum(np.array(al) * ws) / ws.sum()),
                              area_ratio=float(np.sum(np.array(ar) * ws) / ws.sum()), n=len(ws))
    return out


def blen(w, v):
    from scipy import ndimage as ndi
    ok = ndi.binary_erosion(v, iterations=2)
    m = (w > 0.5) & v
    b = m ^ ndi.binary_erosion(m, iterations=1, border_value=0)
    return int((b & ok).sum())


def boundary_ratio(clip, w_proc):
    """Boundary length of the model white mask / the video's, dt-weighted over the shape frames (review structure.py)."""
    res = clip["res"]
    disk = res["disk"].astype(bool)
    dt = np.gradient(res["t"])
    rs, ws = [], []
    for j in np.flatnonzero(clip["shape_ok"]):
        v = res["valid"][j] & disk
        a = blen(res["white"][j].astype(float), v)
        if a < 10:
            continue
        rs.append(blen(w_proc[j], v) / a)
        ws.append(dt[j])
    return float(np.average(rs, weights=ws))


def lobe_rows(w, v, prom=0.05):
    """Right lobe of layered_heart: rows above the white centroid, columns right of it (array = physical top view);
    number of whiteness peaks (prominence >= prom) per row -> median and 90th percentile over the rows."""
    from scipy.signal import find_peaks
    m = (w > 0.5) & v
    if m.sum() == 0:
        return None
    rr, cc = np.nonzero(m)
    r0, rc, cc0 = rr.min(), int(round(rr.mean())), int(round(cc.mean()))
    n = []
    for r in range(r0, rc):
        cols = np.flatnonzero(v[r, cc0:]) + cc0
        if len(cols) < 5:
            continue
        x = w[r, cols]
        if (x > 0.5).sum() < 3:
            continue
        pk, _ = find_peaks(np.r_[0.0, x, 0.0], prominence=prom)
        n.append(len(pk))
    return dict(median=float(np.median(n)), p90=float(np.percentile(n, 90)), rows=len(n))


def chi_kind(script, params):
    from latte_imex import v05_adapter as VA
    r = VA.chi_by_kind(script, dict(VA.CLOSURE_SETS["FSp2"]), params=params)
    return {k: dict(mL=float(v[0]), chi=float(v[1])) for k, v in r.items()}


def fd_rows(ev, keys, names, centre="centre", pre=""):
    def L(lab):
        return float(np.mean([ev[pre + lab][n]["loss"] for n in names]))

    def part(lab, q):
        return float(np.mean([ev[pre + lab][n]["parts"][q] for n in names]))
    L0 = L(centre)
    rows = []
    for k in keys:
        r = dict(k=k)
        for s in "+-":
            lab = k + s
            r["dL" + s] = L(lab) - L0
            r["d" + s] = {q: part(lab, q) - part(centre, q) for q in ("one_minus_dice", "bnd_term", "line_term")}
        rows.append(r)
    return L0, rows


def gauss_newton(ev, keys, names, rel, centre="centre", pre=""):
    """Sensitivity of the CONTINUOUS observables (per-frame 1 - Dice and boundary / 0.05 D, calib.evaluate weights)
    to ln(theta) by central differences -> J^T J eigenvalues, column cosines, linearised correlations (pinv)."""
    def vec(lab):
        v = []
        for n in names:
            fr, fr0 = ev[pre + lab][n]["frames"], ev[pre + centre][n]["frames"]
            W = sum(f["w"] for f in fr0 if f["dice"] is not None)
            for f, f0 in zip(fr, fr0):
                if f0["dice"] is None or f["dice"] is None:
                    continue
                s = np.sqrt(f0["w"] / W / len(names))
                v.append(s * (1 - f["dice"]))
                v.append(s * (f["bnd"] if f["bnd"] is not None else 0.5) / 0.05)
        return np.array(v)
    h = np.log(1 + rel) - np.log(1 - rel)
    J = np.stack([(vec(k + "+") - vec(k + "-")) / h for k in keys], 1)
    norm = np.linalg.norm(J, axis=0)
    Jn = J / np.where(norm > 0, norm, 1)
    COS = Jn.T @ Jn
    F = J.T @ J
    w, V = np.linalg.eigh(F)
    Cov = np.linalg.pinv(F)
    sd = np.sqrt(np.diag(Cov))
    R = Cov / np.outer(sd, sd)
    pairs = [dict(a=keys[i], b=keys[j], r=float(R[i, j]), cos=float(COS[i, j]))
             for i in range(len(keys)) for j in range(i + 1, len(keys)) if abs(R[i, j]) > 0.7 or abs(COS[i, j]) > 0.85]
    flat = [dict(eig=float(w[j]), dir={k: round(float(x), 2) for k, x in zip(keys, V[:, j]) if abs(x) > 0.2})
            for j in range(3)]
    return dict(eig=w.tolist(), norm=dict(zip(keys, norm.tolist())), pairs=pairs, flattest=flat,
                cond=float(np.sqrt(w.max() / max(w.min(), 1e-300))))


# =========================================================================== analyze
def analyze(a):
    P = _paths(a)
    os.makedirs(P["final"], exist_ok=True)
    T = dict(groups=GROUPS, patterns=PAT)
    ab = json.load(open(os.path.join(P["cal"], "calib_AB", "calib.json")))
    ar = json.load(open(os.path.join(P["cal"], "calib_A", "calib.json")))
    pre = json.load(open(os.path.join(P["cal"], "prereg.json")))
    params = dict(FSp2=pre["x0s"][0], AB=ab["best"]["params"], A=ar["best"]["params"],
                  AB_i1=ab["inits"][0]["best"]["params"], AB_i3=ab["inits"][2]["best"]["params"],
                  A_i2=ar["inits"][1]["best"]["params"], A_i3=ar["inits"][2]["best"]["params"])
    loss96 = dict(FSp2=ab["driver"]["x0_reference"][0]["loss"],
                  AB=ab["best"]["loss"], A=ar["best"]["loss"], AB_i1=ab["inits"][0]["best"]["loss"],
                  AB_i3=ab["inits"][2]["best"]["loss"], A_i2=ar["inits"][1]["best"]["loss"],
                  A_i3=ar["inits"][2]["best"]["loss"])
    T["params"], T["loss96"] = params, loss96
    # ---- metrics per set and pattern (N = 128)
    E = {g: {p: parts(load_eval(P, g, p)) for p in PAT} for g in GROUPS}
    T["evals"] = E

    def mean(g, names, k):
        v = [E[g][p][k] for p in names]
        return float(np.mean(v)) if all(x is not None for x in v) else None
    keys = ("loss", "one_minus_dice", "bnd_term", "bnd_mm", "line_term", "iou", "mae", "final_iou", "final_bnd_mm")
    T["means"] = {g: dict(cal_AB={k: mean(g, CB.CAL_AB, k) for k in keys}, cal_A={k: mean(g, CB.CAL_A, k) for k in keys},
                          test={k: mean(g, TEST, k) for k in keys},
                          test_no_heart={k: mean(g, ("push_heart", "layered_heart", "swan"), k) for k in keys})
                  for g in GROUPS}
    T["paired_AB_minus_A_test"] = {f"init{i}": T["means"][x]["test"]["loss"] - T["means"][y]["test"]["loss"]
                                   for i, (x, y) in enumerate((("AB_i1", "A"), ("AB", "A_i2"), ("AB_i3", "A_i3")), 1)}
    T["dL_vs_FSp2"] = {g: {p: {k: E[g][p][k] - E["FSp2"][p][k] for k in ("loss", "one_minus_dice", "bnd_term", "line_term",
                                                                          "iou", "final_iou")} for p in PAT}
                       for g in GROUPS if g != "FSp2"}
    # ---- volumes and cup level
    T["volume"] = {g: {p: dict(V_mL=E[g][p]["V_received_mL"], level_cm=E[g][p]["V_received_mL"] / A_CUP_CM2,
                               chi=E[g][p]["chi"]) for p in PAT} for g in GROUPS}
    # ---- chi per stage kind (wanted action, same formulas as the solver)
    chk = {}
    for p in CB.CAL_AB:
        sc = CB.load_clip(p)["script"]
        q = dict(params["AB"])
        q["B_dep"] = params["FSp2"]["B_dep"]
        chk[p] = {g: chi_kind(sc, params[g]) for g in ("FSp2", "AB", "A", "A_i2")}
        chk[p]["AB_Bdep2149"] = chi_kind(sc, q)
    T["chi_by_kind"] = chk
    # ---- centroid offsets, area ratios, boundary-length ratios
    off, bl = {}, {}
    for p in PAT:
        clip = CB.load_clip(p)
        off[p], bl[p] = {}, {}
        for g in GROUPS:
            w = field128(P, g, p)
            assert w.shape[0] == len(clip["res"]["t"]), (g, p, w.shape)
            off[p][g + "_N128"] = offsets(clip, w[-1], w)
            bl[p][g] = boundary_ratio(clip, w)
        for g in ("FSp2", "AB", "A"):
            off[p][g + "_N256"] = offsets(clip, final256(P, g, p))
    T["offsets"], T["boundary_ratio_proc128"] = off, bl
    T["offset_summary"] = {k: dict(mean_abs_mm=float(np.mean([off[p][k]["final"]["abs_mm"] for p in PAT])),
                                   along_min=float(min(off[p][k]["final"]["along_mm"] for p in PAT)),
                                   along_max=float(max(off[p][k]["final"]["along_mm"] for p in PAT)),
                                   proc_along_mean=(float(np.mean([off[p][k]["process"]["along_mm"] for p in PAT]))
                                                    if "process" in off[PAT[0]][k] else None))
                           for k in off[PAT[0]]}
    # ---- layered_heart right-lobe row peaks
    clip = CB.load_clip("layered_heart")
    res = clip["res"]
    vT = res["valid"][-1] & res["disk"].astype(bool)
    lr = dict(video=lobe_rows(res["white"][-1].astype(float), vT))
    for g in ("FSp2", "AB", "A"):
        lr[g + "_N256"] = lobe_rows(final256(P, g, "layered_heart"), vT)
    for g in GROUPS:
        lr[g + "_N128"] = lobe_rows(field128(P, g, "layered_heart")[-1], vT)
    T["layered_lobe_rows"] = lr
    # ---- finite differences: AB (review data, re-analysed), A (probe)
    ks = list(params["AB"])
    fdab = json.load(open(os.path.join(P["verify"], "fd_rel0.1.json")))
    L0, rows = fd_rows(fdab["evals"], ks, CB.CAL_AB)
    T["fd_AB"] = dict(L0=L0, rows=rows, gn=gauss_newton(fdab["evals"], ks, CB.CAL_AB, fdab["rel"]))
    ex = json.load(open(os.path.join(P["verify"], "fd_extra.json")))
    T["fd_AB_extra"] = {lab: float(np.mean([ex["evals"][lab][n]["loss"] for n in CB.CAL_AB])) - L0 for lab in ex["evals"]}
    pr_path = os.path.join(P["final"], "probe.json")
    if os.path.exists(pr_path):
        pr = json.load(open(pr_path))
        pe = pr["evals"]
        L0A, rowsA = fd_rows(pe, ks, CB.CAL_A, pre="A:")
        T["fd_A"] = dict(L0=L0A, L0_calib=ar["best"]["loss"], rows=rowsA,
                         gn=gauss_newton(pe, ks, CB.CAL_A, pr["rel"], pre="A:"))
        T["fd_AB_repro"] = {lab: dict(mine=float(np.mean([pe[lab][n]["loss"] for n in CB.CAL_AB])) - L0,
                                      review=None) for lab in ("AB:kappa_t+10%", "AB:Q_scale+2%", "AB:B_dep=2149")}
        rv = {r["k"]: r for r in rows}
        T["fd_AB_repro"]["AB:kappa_t+10%"]["review"] = rv["kappa_t"]["dL+"]
        T["fd_AB_repro"]["AB:Q_scale+2%"]["review"] = T["fd_AB_extra"].get("Q_scale+2%")
        T["fd_AB_repro"]["AB:B_dep=2149"]["review"] = T["fd_AB_extra"].get("B_dep=2149")
        c2 = pe["Ai2:centre"]["tulip"]["loss"]
        T["Ai2_probe"] = dict(centre=c2, centre_calib=ar["inits"][1]["best"]["loss"],
                              **{lab: pe[lab]["tulip"]["loss"] - c2 for lab in ("Ai2:B_dep-10%", "Ai2:Q_scale-10%")})
        top = {}
        for tag, names in (("AB", CB.CAL_AB), ("A", CB.CAL_A)):
            lst = []
            for r, c in enumerate(pr["top"][tag], 1):
                lab = f"top{tag}:{r}"
                e = pe[lab]
                lst.append(dict(rank=r, init=c["init"], gen=c["gen"], loss96=c["loss96"],
                                loss128=float(np.mean([e[n]["loss"] for n in names])),
                                line128=float(np.mean([e[n]["line_term"] for n in names])),
                                shape128=float(np.mean([e[n]["one_minus_dice"] + e[n]["bnd_term"] for n in names])),
                                **{f"{n}128": e[n]["loss"] for n in names}))
            from scipy.stats import spearmanr
            l96 = [x["loss96"] for x in lst]
            l128 = [x["loss128"] for x in lst]
            top[tag] = dict(rows=lst, spearman=float(spearmanr(l96, l128).correlation),
                            best128=min(lst, key=lambda x: x["loss128"]))
        T["topk"] = top
        T["probe_meta"] = dict(n_calls=pr["n_calls"], wall_s=pr["wall_s"], procs=pr["procs"], provenance=pr["provenance"])
    mp = os.path.join(P["final"], "mech.json")
    if os.path.exists(mp):
        mj = json.load(open(mp))
        T["mech"] = {lab: {n: dict(loss=e["loss"], one_minus_dice=e["one_minus_dice"], bnd_term=e["bnd_term"],
                                   line_term=e["line_term"], iou=e["iou"], chi=e["chi"], V_mL=e["V_received_mL"],
                                   along_mm=e["offsets"]["final"]["along_mm"], abs_mm=e["offsets"]["final"]["abs_mm"],
                                   proc_along_mm=e["offsets"]["process"]["along_mm"],
                                   area=e["offsets"]["final"]["area_ratio"]) for n, e in d.items()}
                     for lab, d in mj["evals"].items()}
        for g in ("FSp2", "AB", "A"):
            T["mech"][g] = {n: dict(loss=E[g][n]["loss"], one_minus_dice=E[g][n]["one_minus_dice"], bnd_term=E[g][n]["bnd_term"],
                                    line_term=E[g][n]["line_term"], iou=E[g][n]["iou"], chi=E[g][n]["chi"],
                                    V_mL=E[g][n]["V_received_mL"], along_mm=off[n][g + "_N128"]["final"]["along_mm"],
                                    abs_mm=off[n][g + "_N128"]["final"]["abs_mm"],
                                    proc_along_mm=off[n][g + "_N128"]["process"]["along_mm"],
                                    area=off[n][g + "_N128"]["final"]["area_ratio"]) for n in CB.CAL_AB}
        T["mech_points"] = mj["points"]
    T["provenance"] = dict(driver=_sha(os.path.abspath(__file__)),
                           calib=_sha(CB.__file__), v05_adapter=_sha(os.path.join(REPO, "latte_imex", "v05_adapter.py")),
                           solver=_sha(os.path.join(REPO, "latte_imex", "solver.py")),
                           kernels=_sha(os.path.join(REPO, "latte_imex", "kernels.py")))
    _jdump(T, os.path.join(P["final"], "final_tables.json"))
    open(os.path.join(P["final"], "final_tables.md"), "w").write(tables_md(T))
    print(tables_md(T))


# =========================================================================== markdown tables
def _f(v, fmt="{:.3f}"):
    return "–" if v is None else fmt.format(v)


def tables_md(T):
    E, M = T["evals"], T["means"]
    s = []
    s.append("### T1 校准拟合与测试预测（N = 128）\n")
    s.append("| 参数组 | N = 96 校准 L | 校准拟合 L（AB：郁金香 + 树叶；A：郁金香） | 郁金香 L | 树叶 L | 测试四图 L | 测试三图 L（去掉大白心） | "
             "测试 1 − Dice | 测试边界 mm | 测试细纹项 | 测试过程 IoU | 测试末帧 IoU |")
    s.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in GROUPS:
        cal = M[g]["cal_A"]["loss"] if g.startswith("A_") or g == "A" else M[g]["cal_AB"]["loss"]
        t, t3 = M[g]["test"], M[g]["test_no_heart"]
        s.append(f"| {GCN[g]} | {_f(T['loss96'].get(g), '{:.4f}')} | {cal:.3f} | {E[g]['tulip']['loss']:.3f} | "
                 f"{E[g]['leaf']['loss']:.3f} | {t['loss']:.3f} | {t3['loss']:.3f} | {t['one_minus_dice']:.3f} | "
                 f"{t['bnd_mm']:.2f} | {t['line_term']:.3f} | {t['iou']:.3f} | {t['final_iou']:.3f} |")
    s.append("\nAB − A（测试四图 L，按起点配对）：" + "，".join(f"{k} {v:+.3f}" for k, v in T["paired_AB_minus_A_test"].items()))
    s.append("\n### T2 相对零调参的逐图变化（ΔL = Δ(1 − Dice) + Δ边界项 + Δ细纹项）\n")
    s.append("| 图案 | 参数组 | ΔL | Δ(1 − Dice) | Δ边界项 | Δ细纹项 | Δ过程 IoU | Δ末帧 IoU |")
    s.append("|---|---|---|---|---|---|---|---|")
    for p in PAT:
        for g in ("AB", "A"):
            d = T["dL_vs_FSp2"][g][p]
            s.append(f"| {CN[p]} | {g} | {d['loss']:+.3f} | {d['one_minus_dice']:+.3f} | {d['bnd_term']:+.3f} | "
                     f"{d['line_term']:+.3f} | {d['iou']:+.3f} | {d['final_iou']:+.3f} |")
    s.append("\n### T3 白质心沿 B→D 的偏移（mm，+ = 下游），末帧\n")
    s.append("| 图案 | " + " | ".join(f"{GCN[g]} N128" for g in GROUPS) + " | FSp2 N256 | AB N256 | A N256 |")
    s.append("|---|" + "---|" * (len(GROUPS) + 3))
    O = T["offsets"]
    for p in PAT:
        s.append(f"| {CN[p]} | " + " | ".join(f"{O[p][g + '_N128']['final']['along_mm']:+.1f}" for g in GROUPS) + " | " +
                 " | ".join(f"{O[p][g + '_N256']['final']['along_mm']:+.1f}" for g in ("FSp2", "AB", "A")) + " |")
    S = T["offset_summary"]
    s.append("| 平均 \\|d\\| | " + " | ".join(f"{S[g + '_N128']['mean_abs_mm']:.1f}" for g in GROUPS) + " | " +
             " | ".join(f"{S[g + '_N256']['mean_abs_mm']:.1f}" for g in ("FSp2", "AB", "A")) + " |")
    s.append("\n过程平均（N = 128，形状帧按 dt 加权），沿 B→D mm：\n")
    s.append("| 图案 | " + " | ".join(GCN[g] for g in GROUPS) + " |")
    s.append("|---|" + "---|" * len(GROUPS))
    for p in PAT:
        s.append(f"| {CN[p]} | " + " | ".join(f"{O[p][g + '_N128']['process']['along_mm']:+.1f}" for g in GROUPS) + " |")
    s.append("\n### T4 末帧白区面积比（模型 / 视频）与过程边界长度比\n")
    s.append("| 图案 | 面积比 FSp2 / AB / A（N256） | 面积比 N128：" + " / ".join(GROUPS) + " | 边界长度比 N128 过程：" + " / ".join(GROUPS) + " |")
    s.append("|---|---|---|---|")
    for p in PAT:
        s.append(f"| {CN[p]} | " + " / ".join(f"{O[p][g + '_N256']['final']['area_ratio']:.2f}" for g in ("FSp2", "AB", "A")) +
                 " | " + " / ".join(f"{O[p][g + '_N128']['final']['area_ratio']:.2f}" for g in GROUPS) + " | " +
                 " / ".join(f"{T['boundary_ratio_proc128'][p][g]:.2f}" for g in GROUPS) + " |")
    s.append("\n### T5 收到的奶量（mL）与相当的液位上升（cm，A_cup = 50.3 cm²），流量加权 χ\n")
    s.append("| 图案 | " + " | ".join(GCN[g] for g in GROUPS) + " |")
    s.append("|---|" + "---|" * len(GROUPS))
    for p in PAT:
        s.append(f"| {CN[p]} | " + " | ".join(
            f"{T['volume'][g][p]['V_mL']:.0f} mL / {T['volume'][g][p]['level_cm']:.1f} cm / χ {T['volume'][g][p]['chi']:.3f}"
            for g in GROUPS) + " |")
    s.append("\n### T6 千层心右瓣逐行峰数（显著度 0.05），中位 / p90\n")
    s.append(" ，".join(f"{k} {v['median']:.0f}/{v['p90']:.0f}（{v['rows']} 行）" for k, v in T["layered_lobe_rows"].items() if v))
    s.append("\n### T7 有限差分（±10 %，N = 96）\n")
    for lab in ("fd_AB", "fd_A"):
        if lab not in T:
            continue
        F = T[lab]
        s.append(f"\n{lab}：中心 L = {F['L0']:.4f}\n")
        s.append("| 参数 | ΔL +10 % | （1 − Dice，边界，细纹） | ΔL −10 % | （1 − Dice，边界，细纹） |")
        s.append("|---|---|---|---|---|")
        for r in F["rows"]:
            dp, dm = r["d+"], r["d-"]
            s.append(f"| {r['k']} | {r['dL+']:+.3f} | ({dp['one_minus_dice']:+.3f}, {dp['bnd_term']:+.3f}, {dp['line_term']:+.3f}) | "
                     f"{r['dL-']:+.3f} | ({dm['one_minus_dice']:+.3f}, {dm['bnd_term']:+.3f}, {dm['line_term']:+.3f}) |")
        g = F["gn"]
        s.append("\nJ^T J 特征值：" + "，".join(f"{x:.2g}" for x in g["eig"]))
        s.append("\n|r| > 0.7 或 |cos| > 0.85 的参数对：" + "；".join(f"{q['a']}–{q['b']} r {q['r']:+.3f} cos {q['cos']:+.3f}"
                                                          for q in g["pairs"]))
    if "fd_AB_repro" in T:
        s.append("\nAB 探针复现（本文 / 审稿）：" + "；".join(f"{k} {v['mine']:+.4f} / {_f(v['review'], '{:+.4f}')}"
                                                   for k, v in T["fd_AB_repro"].items()))
    if "Ai2_probe" in T:
        q = T["Ai2_probe"]
        s.append(f"\nA 起点 2：中心 {q['centre']:.4f}（calib.json {q['centre_calib']:.4f}），B_dep −10 % {q['Ai2:B_dep-10%']:+.4f}，"
                 f"Q_scale −10 % {q['Ai2:Q_scale-10%']:+.4f}")
    if "topk" in T:
        s.append("\n### T8 事后诊断：前 10 个 N = 96 候选在 N = 128 上复评（只用校准图）\n")
        for tag, d in T["topk"].items():
            s.append(f"\n{tag}：Spearman(N96, N128) = {d['spearman']:+.2f}；N = 128 最低的是第 {d['best128']['rank']} 名"
                     f"（N96 {d['best128']['loss96']:.4f} → N128 {d['best128']['loss128']:.4f}）\n")
            s.append("| 名次 | 起点 | 代 | L N96 | L N128 | 形状项 N128 | 细纹项 N128 |")
            s.append("|---|---|---|---|---|---|---|")
            for r in d["rows"]:
                s.append(f"| {r['rank']} | {r['init']} | {r['gen']} | {r['loss96']:.4f} | {r['loss128']:.4f} | "
                         f"{r['shape128']:.3f} | {r['line128']:.3f} |")
    if "mech" in T:
        s.append("\n### T10 机制检验（事后，只用校准图，N = 128）：把 β 或出口动量换回零调参的值\n")
        s.append("| 参数组 | 郁金香 L | 郁金香末帧偏移 mm（过程） | 郁金香面积比 | 树叶 L | 树叶末帧偏移 mm（过程） | 树叶面积比 |")
        s.append("|---|---|---|---|---|---|---|")
        order = ["FSp2", "FSp2|beta=1.556", "AB", "AB|beta=3.0", "AB|u_exit=0.15,kappa_t=0.7",
                 "AB|beta=3.0,u_exit=0.15,kappa_t=0.7", "A", "A|beta=3.0"]
        for lab in order:
            if lab not in T["mech"]:
                continue
            m = T["mech"][lab]
            s.append(f"| {lab} | " + " | ".join(f"{m[n]['loss']:.3f} | {m[n]['along_mm']:+.1f}（{m[n]['proc_along_mm']:+.1f}） | "
                                                f"{m[n]['area']:.2f}" for n in CB.CAL_AB) + " |")
    s.append("\n### T9 各阶段的流量加权 χ（想要的动作，校准图）\n")
    for p, d in T["chi_by_kind"].items():
        s.append(f"\n{CN[p]}：")
        for g, kk in d.items():
            s.append(f"- {g}: " + "，".join(f"{k} {v['chi']:.3f}（{v['mL']:.0f} mL）" for k, v in kk.items()))
    return "\n".join(s) + "\n"


# =========================================================================== figure
def figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    P = _paths(a)
    T = json.load(open(os.path.join(P["final"], "final_tables.json")))
    for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "Droid Sans Fallback", "AR PL UMing CN"):
        if any(f == x.name for x in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = [f, "DejaVu Sans"]
            break
    C = dict(FSp2="#2a78d6", AB="#eb6834", A="#1baf7a", other="#8a8984")
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlesize": 10.5, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9.5})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6), dpi=100)
    # (a) downstream offset
    ax = axes[0, 0]
    O = T["offsets"]
    ys = np.arange(len(PAT))[::-1]
    for y, p in zip(ys, PAT):
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
        for g in ("AB_i1", "AB_i3", "A_i2", "A_i3"):
            ax.plot(O[p][g + "_N128"]["final"]["along_mm"], y, "o", ms=5, mfc="none", mec=C["other"], mew=1.2, zorder=2)
        for g, mk, dy in (("FSp2", "s", 0.0), ("AB", "o", 0.0), ("A", "D", 0.0)):
            v = O[p][g + "_N256"]["final"]["along_mm"]
            ax.plot(v, y + dy, mk, ms=8, color=C[g], mec="white", mew=1.0, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([CN[p] + ("（校准）" if p in CB.CAL_AB else "（测试）") for p in PAT])
    ax.set_xlabel("末帧白质心沿 B→D 的偏移，模型 − 视频（mm，+ = 下游）")
    S = T["offset_summary"]
    ax.set_title(f"(a) 下游偏移：平均 |d| 零调参 {S['FSp2_N256']['mean_abs_mm']:.1f} → AB {S['AB_N256']['mean_abs_mm']:.1f}"
                 f" / A {S['A_N256']['mean_abs_mm']:.1f} mm（N = 256）", loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="s", ms=8, color=C["FSp2"], label="零调参 FSp2（N256）"),
         Line2D([], [], ls="", marker="o", ms=8, color=C["AB"], label="AB 冻结（N256）"),
         Line2D([], [], ls="", marker="D", ms=7, color=C["A"], label="A 冻结（N256）"),
         Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec=C["other"], label="其余四个起点最好点（N128）")]
    ax.legend(handles=h, loc="lower right", fontsize=8.5, frameon=False)
    ax.set_xlim(-3, 17)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # (b), (c) finite differences at the AB best point, shape part and fine-line part
    F = T["fd_AB"]["rows"]
    lab = {"B_dep": "B_dep", "p_dep": "p_dep", "kappa_r": "κ_r", "kappa_Q": "κ_Q", "cp": "c_p", "beta": "β",
           "kappa_t": "κ_t", "kappa_c": "κ_c", "u_exit": "u_exit", "Q_scale": "Q_scale"}
    yk = np.arange(len(F))[::-1]
    for ax, which, title in ((axes[0, 1], "shape", "(b) AB 最好点 ±10 %：形状部分 Δ(1 − Dice) + Δ边界项"),
                             (axes[1, 0], "line", "(c) AB 最好点 ±10 %：细纹部分 Δ细纹项")):
        for y, r in zip(yk, F):
            ax.axhline(y, color=GRID, lw=0.8, zorder=0)
            for s, col, mk in (("+", C["FSp2"], "o"), ("-", C["AB"], "s")):
                d = r["d" + s]
                v = d["one_minus_dice"] + d["bnd_term"] if which == "shape" else d["line_term"]
                ax.plot([0, v], [y, y], color=col, lw=2, zorder=2, solid_capstyle="round")
                ax.plot(v, y, mk, ms=7, color=col, mec="white", mew=1, zorder=3)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(yk)
        ax.set_yticklabels([lab[r["k"]] for r in F])
        ax.set_xlim(-0.06, 0.22)
        ax.set_xlabel("ΔL（两图平均，N = 96）")
        ax.set_title(title, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        h = [Line2D([], [], ls="-", marker="o", ms=7, color=C["FSp2"], lw=2, label="+10 %"),
             Line2D([], [], ls="-", marker="s", ms=7, color=C["AB"], lw=2, label="−10 %")]
        ax.legend(handles=h, loc="lower right", fontsize=8.5, frameon=False)
    # (d) top-10 N96 -> N128
    ax = axes[1, 1]
    if "topk" in T:
        for tag, col, mk in (("AB", C["AB"], "o"), ("A", C["A"], "D")):
            rows = T["topk"][tag]["rows"]
            x = [r["loss96"] for r in rows]
            y = [r["loss128"] for r in rows]
            ax.plot(x, y, mk, ms=7, color=col, mec="white", mew=1, ls="",
                    label=f"{tag} 前 10（Spearman {T['topk'][tag]['spearman']:+.2f}）")
            ax.annotate("冻结的最好点", (x[0], y[0]), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
        lo, hi = 1.40, 2.05
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        ax.text(hi - 0.01, hi - 0.03, "y = x", ha="right", va="top", fontsize=8, color=MUTED)
        ax.set_xlim(lo, 1.62)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("校准损失 L，N = 96（选点时用的网格）")
        ax.set_ylabel("同一点的校准损失 L，N = 128")
        ax.set_title("(d) 事后诊断：N = 96 的前 10 名到 N = 128 还剩多少", loc="left")
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)
        ax.grid(color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(P["final"], "offset_fd.png")
    fig.savefig(out, facecolor="white")
    print("wrote", out, os.path.getsize(out))


# =========================================================================== export
def export(a):
    P = _paths(a)
    dest = a.dest
    os.makedirs(os.path.join(dest, "videos"), exist_ok=True)
    os.makedirs(os.path.join(dest, "drivers"), exist_ok=True)
    os.makedirs(os.path.join(dest, "records"), exist_ok=True)
    cp = []

    def c(src, dst):
        shutil.copy2(src, os.path.join(dest, dst))
        cp.append(dst)
    c(os.path.join(P["cal"], "CALIBRATION_RESULT.md"), "CALIBRATION_RESULT.md")
    for f in ("final_vs_real.png", "history.png", "inits_N128.png", "lines_BD_N256.png"):
        c(os.path.join(P["cal"], f), f)
    for p in PAT:
        c(os.path.join(P["cal"], f"process_{p}_AB.png"), f"process_{p}_AB.png")
        c(os.path.join(P["cal"], "videos", f"{p}.mp4"), os.path.join("videos", f"{p}.mp4"))
        c(os.path.join(P["cal"], "videos", f"{p}.json"), os.path.join("videos", f"{p}.json"))
    c(os.path.join(P["final"], "offset_fd.png"), "offset_fd.png")
    c(os.path.join(P["cal"], "calib_AB", "calib.json"), os.path.join("records", "calib_AB.json"))
    c(os.path.join(P["cal"], "calib_A", "calib.json"), os.path.join("records", "calib_A.json"))
    c(os.path.join(P["cal"], "prereg.json"), os.path.join("records", "prereg.json"))
    c(os.path.join(P["cal"], "predict_setup.json"), os.path.join("records", "predict_setup.json"))
    c(os.path.join(P["final"], "final_tables.json"), os.path.join("records", "final_tables.json"))
    c(os.path.join(P["final"], "probe.json"), os.path.join("records", "probe.json"))
    c(os.path.join(P["verify"], "fd_analysis_rel0.1.json"), os.path.join("records", "review_fd_analysis_AB.json"))
    # drivers: the versions that produced the results (from git: v05_calibrate a12954fc, v05_predict 2c1ab41d)
    import subprocess
    for f, want in (("v05_calibrate.py", "a12954fc"), ("v05_predict.py", "2c1ab41d")):
        txt = subprocess.run(["git", "-C", REPO, "show", f"4caec97:latte_imex/{f}"], capture_output=True, check=True).stdout
        assert hashlib.sha1(txt).hexdigest()[:8] == want, f
        open(os.path.join(dest, "drivers", f), "wb").write(txt)
        cp.append(os.path.join("drivers", f))
    big = [(f, os.path.getsize(os.path.join(dest, f))) for f in cp if f.endswith(".png")]
    assert all(s < 1_000_000 for _, s in big), big
    print("\n".join(f"{f}  {os.path.getsize(os.path.join(dest, f))}" for f in cp))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("probe", "mech", "analyze", "figs", "export"))
    ap.add_argument("--cal", default=D_CAL)
    ap.add_argument("--base", default=D_BASE)
    ap.add_argument("--dest", default=HERE)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    dict(probe=probe, mech=mech, analyze=analyze, figs=figs, export=export)[a.cmd](a)


if __name__ == "__main__":
    main()
