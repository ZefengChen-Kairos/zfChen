"""viz29: tests of the renderer chain (scene -> raytrace -> render3) on the N 48 fixtures in viz/testdata.

    python test_viz.py            (CPU; ~1-2 min; writes only into a temporary directory that it removes)
1. stream foot == solver landing within 0.5 mm: the centre line of the TUBE that is ray traced, where it crosses the
   liquid surface z = 0, against the schedule landing (S_x, S_y) L of the solver frame containing t (MI-R tulip, record
   control tulip, MI-L held-out heart; every 30 fps frame with a landed stream)
2. poured volume == integral of Q within 0.1 %: the HUD's poured volume against the trapezoid integral of the pour
   model's lip flow Q_o(t), against V0 - V_P (the ledger) and against the arrivals V_cup at the end
3. ledger monotone: milk left in the pitcher non-increasing, poured and arrived non-decreasing on the frame clock
4. the ray tracer renders a known sphere / plane scene identically on repeated runs, and the sphere's silhouette has the
   analytic projected radius
5. CPU smoke of each view (64 x 36, 3 frames): four mp4 files with 3 frames of the right size + the sheet; the caption
   names model, which closure (起点 / 拟合后), inlet and N once (a label's own 'N=96' is dropped)
6. job planner (jobs21.py, no solver): the start closure of a finished params run from done.json 'init' (tree moved),
   run.log, the launcher's INIT table, the repository default; agreement with best_closure.json on the unfitted
   numbers; the run's inlet; job list (start before best, held-out pours for measured inlets, and for record closures
   with --held-record, VIZ_CLOSURES); .done markers per (model, which, pour) bound to the closure sha256 and the
   render settings (N, fps, T, size, ss, cup; --keep-render, --force); cached fields bound to closure, inlet, N, fps,
   T; the first layout moved to viz/best/; summary
   (captions: a held-out pour says 'e0 关' only when e0 was off); the cup model is in the markers (a marker without
   it = the old level cup: re-rendered, the cached fields reused)
7. cup (test_cup): the stand-in mug on the measured rim (viz/cup_rim.json) never intersects the posed CAD pitcher at
   any 30 fps frame and the liquid never spills; the old level cup did intersect (tulip fixture, heart fixture, and
   the files in VIZ_REAL_FIELDS=a.npz:b.npz, e.g. the real N 256 tulip / leaf fields).  VIZ_TESTDATA = fixture dir.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jobs21 as J  # noqa: E402
import raytrace as RT  # noqa: E402
import render3 as R3  # noqa: E402
import scene as S  # noqa: E402

TD = os.environ.get("VIZ_TESTDATA") or os.path.join(HERE, "testdata")


def _heart():
    Z = dict(np.load(os.path.join(TD, "inl_v05n_heart_N48.npz")))
    Z["C"] = np.zeros((2, 48, 48), np.uint8)
    Z["t"] = np.array([0.0, 7.9])
    return Z


def _crossing(P):
    """Point where a centre line (k, 3) crosses z = 0 (first downward crossing), or None."""
    z = P[:, 2]
    for i in range(len(P) - 1):
        if z[i] >= 0.0 >= z[i + 1] and z[i] != z[i + 1]:
            f = z[i] / (z[i] - z[i + 1])
            return P[i] + f * (P[i + 1] - P[i])
    return None


def test_stream_foot(fields, name, tol_mm=0.5, t1=None):
    sc = S.Scene(fields)
    n, worst, trickle = 0, 0.0, 0
    for t in sc.clock(0.0, t1):
        fs = sc.solver_foot(t)
        st = sc.stream(t)
        if fs is None or st["foot"] is None:
            continue
        if not st["segments"]:              # a trickle below scene.q_min (0.1 mL/s, radius ~0.1 mm) is not drawn
            trickle += 1
            continue
        sm = RT.stream_mesh(st["segments"])
        assert sm is not None, (name, t)
        landed = [L for L in sm[3] if L[2]]
        assert landed, (name, t, "no landed tube although the stream lands")
        # the tube the solver's landing belongs to: the landed one whose crossing is nearest (several landed segments
        # only right after a break in the stream)
        d = min(float(np.hypot(*(_crossing(P)[:2] - fs[:2]))) for P, r, _ in landed if _crossing(P) is not None)
        worst = max(worst, d * 1e3)
        n += 1
    assert n > 10, (name, n)
    assert worst <= tol_mm, (name, worst)
    print(f"ok 1 stream foot ({name}): {n} frames, max |tube foot - solver landing| {worst:.2e} mm (<= {tol_mm} mm); "
          f"{trickle} frames with only a trickle < 0.1 mL/s (not drawn)")
    return worst


def test_volumes(fields, name):
    sc = S.Scene(fields)
    E, R = sc.E, sc.R
    V0 = float(E["V0"])
    dt = float(E["dt"])
    # trapezoid integral of Q_o(t) on the step ends (Q_o is piecewise constant per 1 ms step)
    te = np.r_[E["t"][0] - 0.5 * dt, E["t"] + 0.5 * dt]
    cumQ = np.r_[0.0, np.cumsum(E["Q_o"] * np.diff(te))]
    ts = sc.clock()
    worst = dict(intQ=0.0, ledger=0.0)
    prev = None
    for t in ts:
        f = sc.flow(t)
        iq = float(np.interp(t, te, cumQ))
        if iq > 1e-6:                                   # relative test once 1 mL has been poured
            worst["intQ"] = max(worst["intQ"], abs(f["V_out"] - iq) / iq)
            worst["ledger"] = max(worst["ledger"], abs(f["V_out"] - (V0 - f["V_P"])) / iq)
        if prev is not None:
            assert f["V_P"] <= prev["V_P"] + 1e-12, (name, t, "milk left increased")
            assert f["V_out"] >= prev["V_out"] - 1e-12, (name, t, "poured decreased")
            assert f.get("V_cup", 0.0) >= prev.get("V_cup", 0.0) - 1e-12, (name, t, "arrived decreased")
        prev = f
    fe = sc.flow(float(E["t"][-1]) + 2.0)
    end = abs(fe["V_out"] - float(R["V_cup"][-1])) / fe["V_out"]
    assert worst["intQ"] < 1e-3 and worst["ledger"] < 1e-3 and end < 1e-3, (name, worst, end)
    print(f"ok 2 volumes ({name}): poured vs int Q max rel {worst['intQ']:.1e}, vs V0 - V_P {worst['ledger']:.1e}, "
          f"vs cup arrivals at the end {end:.1e} (< 1e-3); "
          f"poured {fe['V_out'] * 1e6:.2f} mL of V0 {V0 * 1e6:.2f} mL")
    print(f"ok 3 ledger monotone ({name}): {len(ts)} frames")


def test_raytracer(device="cpu"):
    V, F, Nv = RT.icosphere(0.02, (0.0, 0.0, 0.03), sub=4)
    r = RT.Renderer(device, pitcher=(V, F, Nv), cup=None, ss=2)
    A = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    r.set_field(A, 0.1, 0.045)
    cam = RT.look_at((0.0, -0.25, 0.20), (0.0, 0.0, 0.02), fov_y_deg=30.0, aspect=160 / 120)
    kw = dict(pitcher_pose=(np.eye(3), np.zeros(3)), table_z=-0.01)
    a = r.render(cam, 160, 120, **kw)
    b = r.render(cam, 160, 120, **kw)
    r2 = RT.Renderer(device, pitcher=(V, F, Nv), cup=None, ss=2)
    r2.set_field(A, 0.1, 0.045)
    c = r2.render(cam, 160, 120, **kw)
    assert a.shape == (120, 160, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b) and np.array_equal(a, c), "renders differ between repeated runs"
    # silhouette: sphere alone on the background (no disc, no table), ss 1: the pixel count of the sphere vs the
    # analytic projected disc (camera on the axis through the sphere centre)
    cam2 = RT.look_at((0.0, -0.30, 0.03), (0.0, 0.0, 0.03), fov_y_deg=30.0, aspect=1.0)
    img = r.render(cam2, 200, 200, pitcher_pose=(np.eye(3), np.zeros(3)), disc=False, ss=1)
    bg = r.render(cam2, 200, 200, pitcher_pose=None, disc=False, ss=1)
    hit = np.any(np.abs(img.astype(int) - bg.astype(int)) > 3, axis=2)
    dist, R_s = 0.30, 0.02
    ang = math.asin(R_s / dist)
    r_px = math.tan(ang) / math.tan(math.radians(15.0)) * 100.0
    r_meas = math.sqrt(hit.sum() / math.pi)
    assert abs(r_meas - r_px) < 1.0, (r_meas, r_px)          # (icosphere sub 4: polygonal, ~0.1 % smaller)
    print(f"ok 4 ray tracer: sphere/plane scene bitwise identical over 3 renders (2 renderers); silhouette radius "
          f"{r_meas:.2f} px vs analytic {r_px:.2f} px")


def _ffprobe(fn):
    exe = shutil.which("ffprobe")
    if exe is None:
        import imageio
        rd = imageio.get_reader(fn)
        n = sum(1 for _ in rd)
        sh = rd.get_meta_data()["size"]
        return n, sh[0], sh[1]
    o = subprocess.run([exe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames,width,height", "-of", "json", fn], capture_output=True, text=True)
    s = json.loads(o.stdout)["streams"][0]
    return int(s["nb_read_frames"]), int(s["width"]), int(s["height"])


def test_smoke(device="cpu"):
    d = tempfile.mkdtemp(prefix="viz29_smoke_")
    try:
        info = R3.main(["--pour", "tulip", "--fields", os.path.join(TD, "ff_s1kq_tulip_N48.npz"), "--out", d,
                        "--size", "64x36", "--t0", "8.0", "--t1", str(8.0 + 2.0 / 30.0 + 1e-6), "--device", device,
                        "--ss", "1", "--label", "演示，参数未重新拟合，N=96", "--model", "s1kq",
                        "--model-desc", "S1kq 物理模型", "--which", "start"])
        assert info["frames"] == 3, info["frames"]
        for v in ("pour", "guest", "top", "combo"):
            fn = os.path.join(d, f"tulip_{v}.mp4")
            assert os.path.exists(fn), fn
            n, w, h = _ffprobe(fn)
            assert n == 3 and w == 64 and (h == 36 or v == "combo"), (v, n, w, h)
        assert os.path.exists(os.path.join(d, "tulip_sheet.jpg"))
        js = json.load(open(os.path.join(d, "tulip_viz.json")))
        assert "杯子为示意" in js["caption"][1] or "stand-in" in js["caption"][1]
        c0 = js["caption"][0]
        assert c0.count("N=") == 1 and "N=48" in c0, c0                      # the fields' N, once
        assert c0.startswith("s1kq") and ("起点参数（未重新拟合" in c0 or "start closure" in c0), c0
        assert ("实测入口 MI-R" in c0 or "MI-R" in c0) and js["which"] == "start" and js["model"] == "s1kq", c0
        assert "杯子为示意" not in c0, c0                                     # (said once, in the second line)
        print(f"ok 5 CPU smoke: pour / guest / top / combo, 3 frames each at 64 x 36 (combo 64 x {h}), sheet, json; "
              f"caption: {c0}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_captions():
    for a, b in (("演示：S1kq，参数未重新拟合，N=96", "演示：S1kq，参数未重新拟合"), ("s1kq/params N = 48", "s1kq/params"),
                 ("mcd_S1kq_mi_N96", "mcd_S1kq_mi_N96"), ("v05n · N=256 · x", "v05n · x"), (None, "")):
        assert R3.strip_grid(a) == b, (a, R3.strip_grid(a), b)
    # held-out pours: 'e0 关' only when the forward run had e0 off (MI-L); a record closure keeps its e0 (control fit)
    import types
    for inl, e0, which, cl, want, bad in (
            ("law", "False", "best", "best_closure.json", "MI-L 出口定律（留出，e0 关，仅报告）", None),
            ("record", "world", "start", "bp256_trained.json", "旧入口 R1（记录 + E0）（留出，仅报告）", "e0 关"),
            ("record", "False", "extra", "r1.json", "旧入口 R1（记录 + E0）（留出，e0 关，仅报告）", None),
            ("law", None, "best", "best_closure.json", "（留出，e0 关，仅报告）", None)):
        sc = types.SimpleNamespace(meta=dict(inlet=inl, e0=e0, heldout=True, N=48, closure="/x/" + cl), N=48)
        h = R3.Hud(types.SimpleNamespace(sc=sc, snap={"p50": 8.2 if inl == "record" else 0.0}),
                   types.SimpleNamespace(cjk=True), None, "heart", 1280, 720, model="v05n_ctrl", desc="对照：旧入口 R1",
                   which=which)
        assert want in h.line1 and (bad is None or bad not in h.line1) and h.pattern == "爱心 heart（留出）", h.line1
    print("ok captions: a label's own grid size is dropped (N shown once, from the fields); held-out e0 as run")


def _wj(fn, obj):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w") as f:
        json.dump(obj, f, indent=1)


def _plan(**kw):
    import contextlib
    import io
    import types
    a = types.SimpleNamespace(**dict(dict(models="s1kq v05n v05n_ctrl", launcher=None, N=256, which="both",
                                          pours=" ".join(J.POURS), extras=""), **kw))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        J.plan(a)
    lines = buf.getvalue().splitlines()
    jobs = [l.split("\t")[1:] for l in lines if l.startswith("J\t")]
    info = [l[2:] for l in lines if l.startswith("I\t")]
    return jobs, info


def test_jobs():
    """jobs21: start closures, inlets, job list, markers, migration (synthetic tree; no solver)."""
    d = tempfile.mkdtemp(prefix="viz29_jobs_")
    try:
        root = os.path.join(d, "u21")
        mi = dict(cp=0.68, beta=0.58, B_dep=2665.0, e0=2.0, inlet=2, _note="bp256_trained + inlet 2")
        tr = dict(cp=0.68, beta=0.58, B_dep=2665.0, e0=2.0)
        s1 = dict(cp=1.77, kappa_c=0.35, tau_s=0.035, B_dep=1267.0, milk=1.0, e0=2.0, inlet=2)
        _wj(os.path.join(root, "latte_imex/data/closures/bp256_mi.json"), mi)
        _wj(os.path.join(root, "latte_imex/data/closures/mcd_S1kq_mi_N96.json"), s1)
        _wj(os.path.join(root, "closures/bp256_trained.json"), tr)
        os.makedirs(os.path.join(root, "scripts"))
        with open(os.path.join(root, "scripts/autodl_diag21.sh"), "w") as f:
            f.write("model_cfg() {\n  case $1 in\n"
                    "    s1kq)      INIT_M=latte_imex/data/closures/mcd_S1kq_mi_N96.json; PSET_M=S1;   MI_M=1 ;;\n"
                    "    v05n)      INIT_M=latte_imex/data/closures/bp256_mi.json;        PSET_M=v05n; MI_M=1 ;;\n"
                    "    v05n_ctrl) INIT_M=closures/bp256_trained.json;                   PSET_M=v05n; MI_M=0 ;;\n"
                    "  esac\n}\n")
        out = os.path.join(root, "results_diag21")
        fields = out + "_vizfields"
        inl_mi = dict(cal=dict(tulip=dict(mode="replay", source="closure"), leaf=dict(mode="replay", source="closure")),
                      box=None, heldout=dict(mode="law", e0=0), status="provisional (marginal pass), SPEC27 rev 2 section 2.4")

        def run(m, init, start, fitted, mi_run, log_init=None):
            rd = os.path.join(out, m, "params")
            P = {k: start[k] * 1.1 for k in fitted}
            best = dict(start, **P, D=1e-7, _note="diag params best (it 3, video IoU 0.5)")    # diag.best_closure
            done = dict(mode="params", init=init, N=256, best_it=3, best_viou=0.5, params=P, eval={}, iterations=9,
                        hours=1.2)
            if m == "s1kq":
                done["pset"] = "S1"
            if mi_run:
                done["inlet"] = inl_mi
            _wj(os.path.join(rd, "best_closure.json"), best)
            _wj(os.path.join(rd, "done.json"), done)
            if log_init:
                with open(os.path.join(rd, "run.log"), "w") as f:
                    f.write(f"== params: N 256, pours ('tulip', 'leaf'), start closure {log_init}, closure free (wide box), "
                            "inlet as measured; slot 1.35 h\n")
            return rd
        # v05n: done.json 'init' is the AutoDL path of the u20 tree (moved): re-rooted under this tree
        rv = run("v05n", "/root/autodl-tmp/u20/latte_imex/data/closures/bp256_mi.json", mi, ("cp", "beta"), True)
        r = J.resolve_start(rv, "v05n", root)
        assert r["path"] == os.path.join(root, "latte_imex/data/closures/bp256_mi.json") and r["source"] == "done.json init" \
            and r["inlet"] == "auto" and r["run_inlet"] == "replay" and not r["warn"], r
        # v05n_ctrl: init gone, no run.log -> launcher INIT table; record inlet (no 'inlet' in done.json)
        rc = run("v05n_ctrl", "/elsewhere/gone/ctrl_start.json", tr, ("cp", "beta"), False)
        r = J.resolve_start(rc, "v05n_ctrl", root, os.path.join(root, "scripts/autodl_diag21.sh"))
        assert r["path"] == os.path.join(root, "closures/bp256_trained.json") and r["source"] == "launcher INIT table" \
            and r["inlet"] == "auto" and r["run_inlet"] == "record", r
        assert J.agrees(os.path.join(root, "latte_imex/data/closures/bp256_mi.json"),
                        J.load_json(os.path.join(rc, "best_closure.json")), J.load_json(os.path.join(rc, "done.json"))) \
            == ["inlet"]                                         # the MI start is not the control's start
        # s1kq: no init file, no launcher -> repository default; then run.log wins over it
        rs = run("s1kq", "/elsewhere/gone/s1.json", s1, ("kappa_c", "tau_s", "B_dep"), True)
        r = J.resolve_start(rs, "s1kq", root, "/nonexistent/launcher.sh")
        assert r["source"] == "repository default" and r["path"].endswith("mcd_S1kq_mi_N96.json"), r
        _wj(os.path.join(root, "closures/s1_copy.json"), s1)
        with open(os.path.join(rs, "run.log"), "w") as f:
            f.write("== params: N 256, pours ('tulip', 'leaf'), start closure closures/s1_copy.json, closure free "
                    "(wide box), inlet as measured; slot 1.35 h\n")
        r = J.resolve_start(rs, "s1kq", root, "/nonexistent/launcher.sh")
        assert r["source"] == "run.log" and r["path"] == os.path.join(root, "closures/s1_copy.json"), r
        os.remove(os.path.join(rs, "run.log"))
        # a run whose start has no inlet key although the run was MI: the run's inlet is passed explicitly
        rx = run("x", os.path.join(root, "closures/bp256_trained.json"), tr, ("cp",), True)
        r = J.resolve_start(rx, "x", root)
        assert r["path"].endswith("bp256_trained.json") and r["inlet"] == "replay" and r["own_inlet"] == "record", r
        # no candidate agrees (an unfitted number differs): the first found, with a warning
        b = J.load_json(os.path.join(rx, "best_closure.json")); b["B_dep"] = 1.0
        _wj(os.path.join(rx, "best_closure.json"), b)
        r = J.resolve_start(rx, "x", root)
        assert r["path"].endswith("bp256_trained.json") and "B_dep" in r["warn"], r
        # the job list: pour-major, extras first, start before best; held-out pours only for measured inlets
        jobs, info = _plan(out=out, root=root, fields=fields, pours="tulip heart",
                           extras="e1=closures/bp256_trained.json,1 bad=nope.json", launcher=os.path.join(root, "scripts/autodl_diag21.sh"))
        key = [(j[0], j[1], j[2]) for j in jobs]
        assert key == [("e1", "extra", "tulip"), ("s1kq", "start", "tulip"), ("s1kq", "best", "tulip"),
                       ("v05n", "start", "tulip"), ("v05n", "best", "tulip"), ("v05n_ctrl", "start", "tulip"),
                       ("v05n_ctrl", "best", "tulip"), ("e1", "extra", "heart"), ("s1kq", "start", "heart"),
                       ("s1kq", "best", "heart"), ("v05n", "start", "heart"), ("v05n", "best", "heart")], key
        assert any("bad" in s and "not found" in s for s in info), info
        # a broken VIZ_CLOSURES file (not an object / an unreadable 'inlet') skips that closure only
        _wj(os.path.join(root, "closures/blist.json"), [1, 2])
        _wj(os.path.join(root, "closures/binl.json"), dict(tr, inlet=7))
        jb, ib = _plan(out=out, root=root, fields=fields, pours="tulip", models="v05n", which="start",
                       extras="bl=closures/blist.json bi=closures/binl.json ok=closures/bp256_trained.json")
        assert [(x[0], x[1], x[2]) for x in jb] == [("ok", "extra", "tulip"), ("v05n", "start", "tulip")], jb
        assert any("bl" in s and "not a json object: skipped" in s for s in ib) and \
            any("bi" in s and "inlet code" in s and "skipped" in s for s in ib), ib
        jv = {(j[0], j[1], j[2]): j for j in jobs}
        assert jv[("e1", "extra", "tulip")][4] == "law" and jv[("v05n_ctrl", "start", "tulip")][3].endswith("bp256_trained.json")
        assert jv[("v05n", "best", "tulip")][5] == os.path.join(out, "v05n/params/viz/best")
        assert jv[("v05n", "start", "tulip")][6] == os.path.join(fields, "v05n/start/tulip_fields_N256.npz")
        assert jv[("v05n", "start", "tulip")][8] == "v05n 基线" and jv[("e1", "extra", "tulip")][8] == "-"
        assert all(len(j) == 9 and all(j) for j in jobs)
        jobs1, _ = _plan(out=out, root=root, fields=fields, pours="tulip leaf", which="start", models="v05n")
        assert [(j[0], j[1], j[2]) for j in jobs1] == [("v05n", "start", "tulip"), ("v05n", "start", "leaf")], jobs1
        # markers per (model, which, pour), bound to the closure: done -> skipped; another closure -> redone
        j = jv[("v05n", "start", "tulip")]
        os.makedirs(j[5])
        J.main(["mark", j[5], "tulip", "model=v05n", "which=start", f"closure={j[3]}", f"closure_sha256={j[7]}",
                "inlet=auto", "N=256", f"t1={time.time() - 5}"])
        mk = J.load_json(os.path.join(j[5], "tulip.done"))
        assert mk["N"] == 256 and mk["wall_s"] > 4 and mk["which"] == "start", mk
        assert mk["cup_model"] == S.CUP_MODEL_VERSION, mk                  # the add-on's cup model is recorded
        jobs2, info2 = _plan(out=out, root=root, fields=fields, pours="tulip heart")
        assert ("v05n", "start", "tulip") not in [(x[0], x[1], x[2]) for x in jobs2] and len(jobs2) == len(jobs) - 3  # (- e1 x 2)
        assert any("v05n start tulip: done before" in s for s in info2), info2
        # ... and bound to the render settings: a marker without a key counts as the launcher default; a changed
        # setting is redone (with the reason), --keep-render ignores size / ss / cup, --force redoes everything
        vst = ("v05n", "start", "tulip")
        keys = lambda js: [(x[0], x[1], x[2]) for x in js]                                    # noqa: E731
        for kw, why in ((dict(size="1920x1080"), "size 1280x720 -> 1920x1080"), (dict(ss=3), "ss 2 -> 3"),
                        (dict(cup="white"), "cup dark -> white"), (dict(fps=25.0), "fps 30.0 -> 25.0"),
                        (dict(N=96), "N 256 -> 96"), (dict(T="1.0"), "T whole pour -> 1.0")):
            js, inf = _plan(out=out, root=root, fields=fields, pours="tulip", **kw)
            assert vst in keys(js) and any(f"v05n start tulip: done before with other settings ({why})" in s for s in inf), \
                (kw, inf)
        js, inf = _plan(out=out, root=root, fields=fields, pours="tulip", size="640x360", ss=1, keep_render=True)
        assert vst not in keys(js), "the CPU fallback must keep the finished GPU renders"
        js, inf = _plan(out=out, root=root, fields=fields, pours="tulip", N=96, keep_render=True)
        assert vst in keys(js), "--keep-render still compares N"
        js, inf = _plan(out=out, root=root, fields=fields, pours="tulip", force=True)
        assert vst in keys(js) and any("VIZ_FORCE=1: redone" in s for s in inf), inf
        J.main(["mark", j[5], "tulip", "model=v05n", "which=start", f"closure={j[3]}", f"closure_sha256={j[7]}",
                "inlet=auto", "N=256", "fps=30", "T=", "size=1920x1080", "ss=3", "cup=white", "device=cuda:0"])
        mk2 = J.load_json(os.path.join(j[5], "tulip.done"))
        assert mk2["T"] is None and mk2["ss"] == 3 and mk2["fps"] == 30.0 and mk2["cup"] == "white", mk2
        js, _ = _plan(out=out, root=root, fields=fields, pours="tulip", size="1920x1080", ss=3, cup="white")
        assert vst not in keys(js)
        js, _ = _plan(out=out, root=root, fields=fields, pours="tulip")
        assert vst in keys(js)
        # a marker from before the cup model was recorded (level cup) is re-rendered by a plain rerun; the cached
        # fields are not part of it
        _wj(os.path.join(j[5], "tulip.done"), {k: v for k, v in mk.items() if k != "cup_model"})
        js, inf = _plan(out=out, root=root, fields=fields, pours="tulip")
        assert vst in keys(js) and any(f"(cup_model level-v0 -> {S.CUP_MODEL_VERSION})" in s for s in inf), inf
        _wj(os.path.join(j[5], "tulip.done"), mk)                       # back to the default marker
        # held-out pours of record closures only with --held-record (VIZ_HELD_RECORD=1)
        js, _ = _plan(out=out, root=root, fields=fields, pours="heart", models="v05n_ctrl")
        assert js == [], js
        js, _ = _plan(out=out, root=root, fields=fields, pours="heart swan", models="v05n_ctrl",
                      extras="r1=closures/bp256_trained.json", held_record=True)
        assert keys(js) == [("r1", "extra", "heart"), ("v05n_ctrl", "start", "heart"), ("v05n_ctrl", "best", "heart"),
                            ("r1", "extra", "swan"), ("v05n_ctrl", "start", "swan"), ("v05n_ctrl", "best", "swan")], js
        assert all(x[4] == "auto" for x in js)
        # cached fields: closure, inlet, N, fps, T
        sha = j[7]
        ffn = os.path.join(d, "f.npz")
        meta = dict(closure_sha256=sha, cal_inlet="replay", cal_inlet_source="closure key 'inlet'", N=256, fps=30.0,
                    T_max=None)
        np.savez(ffn, meta=np.asarray(json.dumps(meta)))
        assert J.fields_check(ffn, sha, "auto", 256, 30, None) == "" and J.fields_ok(ffn, sha, "auto")
        assert J.fields_check(ffn, sha, "auto", "256", "30", "-") == ""
        assert J.fields_check(ffn, sha, "auto", 256, 25, None) == "fps 30.0 -> 25.0"
        assert J.fields_check(ffn, sha, "auto", 256, 30, "1") == "T whole pour -> 1.0"
        assert J.fields_check(ffn, sha, "auto", 96, 30, None) == "N 256 -> 96"
        assert J.fields_check(ffn, "0" * 64, "auto") == "another closure"
        assert J.fields_check(ffn, sha, "replay").startswith("inlet")
        _wj(os.path.join(j[5], "tulip.done"), dict(mk, closure_sha256="0" * 64))
        jobs3, _ = _plan(out=out, root=root, fields=fields, pours="tulip heart")
        assert ("v05n", "start", "tulip") in [(x[0], x[1], x[2]) for x in jobs3]
        # the first layout (viz/<pour>_*, the fitted closure) moves to viz/best/ and counts as done
        vd = os.path.join(out, "v05n/params/viz")
        sb = J.sha256(os.path.join(out, "v05n/params/best_closure.json"))
        _wj(os.path.join(vd, "leaf_viz.json"), dict(pour="leaf", frames=3, meta=dict(closure_sha256=sb, inlet="replay")))
        _wj(os.path.join(vd, "leaf.done"), dict(model="v05n", pour="leaf"))
        for v in ("pour", "guest", "top", "combo"):
            open(os.path.join(vd, f"leaf_{v}.mp4"), "wb").write(b"x")
        _wj(os.path.join(vd, "heart_viz.json"), dict(pour="heart", frames=3, meta=dict(closure_sha256="f" * 64)))
        _wj(os.path.join(vd, "heart.done"), dict(model="v05n", pour="heart"))
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            J.main(["migrate", "--out", out, "--root", root, "--fields", fields, "--models", "v05n"])
        assert os.path.exists(os.path.join(vd, "best", "leaf_combo.mp4")) and not os.path.exists(os.path.join(vd, "leaf.done"))
        assert os.path.exists(os.path.join(vd, "heart.done"))           # another closure's videos stay where they are
        jobs4, inf4 = _plan(out=out, root=root, fields=fields, pours="leaf", models="v05n", which="best")
        # (migrated videos count as done for the closure, but they were rendered with the level cup: redone)
        assert [(x[0], x[1], x[2]) for x in jobs4] == [("v05n", "best", "leaf")], jobs4
        assert any("cup_model level-v0 ->" in s for s in inf4), inf4
        mj = J.load_json(os.path.join(vd, "best", "leaf.done"))
        _wj(os.path.join(vd, "best", "leaf.done"), dict(mj, cup_model=S.CUP_MODEL_VERSION))
        jobs4, _ = _plan(out=out, root=root, fields=fields, pours="leaf", models="v05n", which="best")
        assert jobs4 == [], jobs4
        with contextlib.redirect_stdout(io.StringIO()):
            J.main(["summary", out])
        sm = open(os.path.join(out, "viz_summary.md")).read()
        assert "| leaf | v05n | 拟合后参数 | ok |" in sm, sm
        print(f"ok 6 job planner: start closures from done.json init (moved tree) / run.log / launcher table / repository, "
              f"checked against best_closure.json; inlets; {len(jobs)} jobs in order; markers; migration; summary")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_framing():
    """The stream foot and the spout tip are inside both perspective views at every 0.5 s of the pour."""
    v = R3.Viz(os.path.join(TD, "ff_s1kq_tulip_N48.npz"), device="cpu", ss=1, W=320, H=180)
    sc = v.sc
    worst = 0.0
    for t in np.arange(0.0, sc.t_end(), 0.5):
        pts = [sc.pose(t)["tip"]]
        f = sc.solver_foot(t)
        if f is not None:
            pts.append(f)
        for k in ("pour", "guest"):
            uv, z = RT.project(v.cams[k], np.array(pts))
            assert np.all(z > 0) and np.all((uv > 0.02) & (uv < 0.98)), (k, t, uv)
            worst = max(worst, float(np.abs(uv - 0.5).max()))
    print(f"ok framing: tip and landing inside both views at every 0.5 s (max |uv - 0.5| {worst:.2f})")


def test_cup(fields, name, level_intersects=None, min_clear_mm=0.0):
    """The stand-in cup on the measured rim (viz/cup_rim.json): at every rendered frame (30 fps over the whole pour)
    the posed CAD pitcher stays outside the cup's wall and handle (min signed distance > min_clear_mm, the pitcher's
    surface sampled ~1 mm apart), the liquid stays inside (the lowest rim-top point >= 1 mm above the surface) and
    the straight inner wall reaches below the tilted liquid section.  level_intersects: the old level cup (rim 3 mm
    above the liquid at the end, rising with the poured volume) must intersect in that many frames or more (the bug)."""
    sc = S.Scene(fields)
    tr = sc.cup_track()
    assert tr["source"] == "measured", (name, tr["source"])
    ts = sc.clock()
    r = S.cup_clearance(sc, ts)
    d = r["d_min"] * 1e3
    assert np.all(d > min_clear_mm) and np.all(r["overlap"] == 0), \
        (name, int(np.sum(d <= min_clear_mm)), float(d.min()), float(ts[int(np.argmin(d))]))
    assert np.all(r["rim_low"] >= S.SPILL_MARGIN - 1e-9), (name, float(r["rim_low"].min()))
    prof, hP, hr, zf, r0 = sc.cup_geometry()
    z_low = -r0 - 4e-3                                                   # bottom of the straight inner wall (B)
    for t in ts[::15]:
        c = sc.cup(t)
        tan_t = math.tan(math.radians(c["tilt_deg"]))
        assert S.CUP_RIM0 - c["s"] - sc.R_disc * tan_t >= z_low - 1e-9, (name, t)   # liquid meets the straight wall
        assert abs(np.linalg.det(c["R"]) - 1) < 1e-9 and np.allclose(c["R"][:, 2], c["n"], atol=1e-9)
    msg = ""
    if level_intersects is not None:
        sl = S.Scene(fields, cup_model="level")
        rl = S.cup_clearance(sl, ts)
        nl = int(np.sum(rl["d_min"] < 0))
        assert nl >= level_intersects, (name, nl)
        msg = f"; the old level cup: {nl}/{len(ts)} frames intersect, up to {rl['overlap'].max() * 1e3:.1f} mm below its rim"
    print(f"ok cup ({name}): {len(ts)} frames, no pitcher-cup intersection (min clearance {d.min():.1f} mm), no spill "
          f"(lowest rim point >= {r['rim_low'].min() * 1e3:.1f} mm above the liquid), tilt {r['tilt_deg'].max():.1f} -> "
          f"{r['tilt_deg'][-1]:.1f} deg, spill guard <= {tr['s_guard'].max() * 1e3:.1f} mm" + msg)


def device_check(dev):
    """The ray tracer on `dev` against the CPU on the sphere / plane scene (autodl_viz21.sh runs this before rendering):
    prints 'ok ...' when < 1 % of the pixels differ by more than 8 levels (float order / BVH differences only)."""
    V, F, Nv = RT.icosphere(0.02, (0.0, 0.0, 0.03), sub=3)
    A = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    cam = RT.look_at((0.0, -0.25, 0.20), (0.0, 0.0, 0.02), fov_y_deg=30.0, aspect=160 / 120)
    imgs = []
    for d in (dev, "cpu"):
        r = RT.Renderer(d, pitcher=(V, F, Nv), cup=None, ss=1)
        r.set_field(A, 0.1, 0.045)
        r.set_dyn([RT.icosphere(0.006, (0.03, 0.0, 0.01), sub=2)[:2] + (None, RT.MAT_STREAM)])
        imgs.append(r.render(cam, 160, 120, pitcher_pose=(np.eye(3), np.zeros(3)), table_z=-0.01).astype(int))
    frac = float(np.mean(np.abs(imgs[0] - imgs[1]).max(axis=2) > 8))
    ok = frac < 0.01 and imgs[0].std() > 5
    print(("ok" if ok else "FAILED") + f" ({dev} vs cpu: {100 * frac:.2f} % of the pixels differ by > 8 levels)")
    return ok


if __name__ == "__main__":
    if "--device-check" in sys.argv:
        dev = sys.argv[sys.argv.index("--device-check") + 1]
        sys.exit(0 if device_check(dev) else 1)
    test_captions()
    test_jobs()
    if "--jobs-only" in sys.argv:
        sys.exit(0)
    test_stream_foot(os.path.join(TD, "ff_s1kq_tulip_N48.npz"), "tulip MI-R")
    test_stream_foot(os.path.join(TD, "ff_ctrl_tulip_N48_T4.npz"), "tulip record control, 0-4 s")
    test_stream_foot(_heart(), "heart MI-L held out")
    test_volumes(os.path.join(TD, "ff_s1kq_tulip_N48.npz"), "tulip MI-R")
    test_volumes(_heart(), "heart MI-L")
    test_cup(os.path.join(TD, "ff_s1kq_tulip_N48.npz"), "tulip", level_intersects=100)
    test_cup(_heart(), "heart")
    for fn in [x for x in os.environ.get("VIZ_REAL_FIELDS", "").split(os.pathsep) if x]:   # e.g. the N 256 fields
        test_cup(fn, os.path.basename(fn), level_intersects=1)
    test_raytracer()
    test_framing()
    test_smoke()
    print("all viz tests passed")
