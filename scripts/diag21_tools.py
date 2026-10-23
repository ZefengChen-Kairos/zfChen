#!/usr/bin/env python3
"""Helpers of scripts/autodl_diag21.sh (update21: the diagnosis on the measured inlet; mi27 SPEC27 rev 2 section 6,
PREREG27_2 section 5; README_diag21.md).  Nothing here changes the model: it checks files, times one iteration, plans
the time slots, writes the bridge closures and the summaries.

    python scripts/diag21_tools.py check   --device cuda:0                         start-up checks (files, closures, inlet)
    python scripts/diag21_tools.py timing  --out DIR [--k 2] -- python -m latte_imex.diff.diag run ... --out DIR
                                                                 one N 256 S1kq (or v05n) iteration (killed after k records)
    python scripts/diag21_tools.py plan    --out OUT --runs "..." [--timing T.json] [--timing-v V.json] ...
                                                                 per-run fit hours -> OUT/plan.json
    python scripts/diag21_tools.py slot    --plan OUT/plan.json --budget-h B --t0 T0 RUN [RUN ...]
                                                                 fit hours of the first RUN (the others follow it)
    python scripts/diag21_tools.py bridge-closure SRC DST --inlet 0|1               the params best closure, inlet changed
    python scripts/diag21_tools.py summary --out OUT                                 per-model + top-level summary.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ------------------------------------------------------------------------------------------------ frozen inputs
# sha256 of the inputs the update21 answers rest on (repository fc9f270 + the mi27 build; SPEC27 rev 2, PREREG27_2 D3/D4,
# SC/mi27/HASHLOG.txt blocks 'build step 2', 'probe 8' and 'fix step 1': stream_foot_mi.json with the SPEC27 1.1 gate).  pitcher_pose.npz is the pose v2 track (the v1 track, which
# the u20 tree carried under this name, is pitcher_pose_v1.npz in the repository).
EXPECT = {
    "raw_data/rig/results/tulip/pitcher_pose.npz": "2ff356def550d2b190dcce529af02c187bc60bfa3f5ca0a0b58fca7e075da485",
    "raw_data/rig/results/leaf/pitcher_pose.npz": "754110c04c1f8cb11550b1aa6aa770871225528e3e62dd7e9a93c893cb3977d1",
    "raw_data/rig/results/heart/pitcher_pose.npz": "17f2c50b96c34597cfb045df49f78e347400f4c554f80cc4c8063cfcbc210c5e",
    "raw_data/rig/results/push_heart/pitcher_pose.npz": "9ee8e707ae5e7666f5aaf243819b00235c263ecdf9e1b3a525ebef015f06eabc",
    "raw_data/rig/results/swan/pitcher_pose.npz": "c2251f45d0da19665e90d0cec64c563a9565e4d289d62dfe64ed421c36696a2f",
    "closures/bp256_trained.json": "ea968f3674a4de048f136474d40607a1f91fc3038230bf4650b3fdf0d049d5c2",
    "latte_imex/data/closures/bp256_mi.json": "cec53bb8ce70075b96aba895a2c86db6a7c896e11fb90cba1095a61635365277",
    "latte_imex/data/closures/mcd_S1kq_N96.json": "2fc0366f19b6817c9f27484e481ac7935d24c57eaecf573cfd03c04e44db3542",
    "latte_imex/data/closures/mcd_S1kq_mi_N96.json": "0e237c7f8245304b7e11191e9e8f892c4a454b381b1569b9f3c28f26d550344b",
    "raw_data/rig/results/tulip/stream_kin.json": "e04be3921068394b2c063bf07649c7fb2817018d416e3870bf1b6ee912f6105c",
    "raw_data/rig/results/leaf/stream_kin.json": "d978ac18be87c8aba732a9d24219b96bd6bd811c583f107b7cd3043a4fdae548",
    "raw_data/rig/results/tulip/stream_foot_mi.json": "57965b77e4a15a624c1d310fd9eccef7d5a56e288a32561684146b842b6b80af",
    "raw_data/rig/results/leaf/stream_foot_mi.json": "d08d38353a18ae0a30bbfad025e8e427ab637c97afbbd3ac8471575a97f1052e",
    "raw_data/rig/results/tulip/stream_foot_col.json": "724f51eaa066f096c1db0dabb55fe96584c0e78a587e236f616798c45ce0e550",
    "raw_data/rig/results/leaf/stream_foot_col.json": "1e2fd383415cc44ec6ecde7ff07b06c8b6fda8d6e65818ee328b4d07163ef8d8",
}
POURS = ("tulip", "leaf", "heart", "push_heart", "swan")
CAL, HELD = ("tulip", "leaf"), ("heart", "push_heart", "swan")

# the three models of update21 (SPEC27 6.1); the launcher holds the same table (INIT / PSET / inlet environment)
MODELS = {
    "s1kq": dict(init="latte_imex/data/closures/mcd_S1kq_mi_N96.json", pset="S1", modes=("params", "inlet"), mi=True,
                 name="物理模型 S1kq", short="S1kq"),
    "v05n": dict(init="latte_imex/data/closures/bp256_mi.json", pset="v05n",
                 modes=("params", "inlet", "both", "both_wide"), mi=True, name="基线 v05n", short="v05n"),
    "v05n_ctrl": dict(init="closures/bp256_trained.json", pset="v05n", modes=("params",), mi=False,
                      name="v05n 对照（pose v2 + 旧入口 R1）", short="v05n 对照"),
}
ORDER = ("s1kq/params", "v05n/params", "v05n_ctrl/params", "s1kq/inlet", "v05n/inlet", "v05n/both", "v05n/both_wide")
MI_PARAMS = ("s1kq/params", "v05n/params")

# numbers of the mi27 gate step for the physical-range report of the S1 params mode (SPEC27 4.2; probe/out/stage1.json,
# hashed in SC/mi27/HASHLOG.txt blocks 3 and 6).  delta_s = tau_s x median |u_in| L over the MI windows (MI-R schedule,
# N 96: DS(S-MR) / 0.075); EXB = the excursion-law surfacing centroid ahead of x_hit (fsc24 arcfit2, 8 variants, MI-R);
# B_dep range = the independent video-threshold range 1230-2210 mapped by f_z (windows) and f_z,high (high pours).
U_MED_MS = {"tulip": 19.202824346574406 / 0.075 / 1e3, "leaf": 17.77997005824375 / 0.075 / 1e3}
EXB_MM = {"tulip": (5.3624008689889076, 7.357255106662471), "leaf": (5.182162770619033, 7.169022548979024)}
BDEP_WIN = {"tulip": (789.621891187538, 1418.751528068666), "leaf": (835.1284351870628, 1500.5153185068364)}
BDEP_HIGH = {"tulip": (1112.9413781696398, 1999.6751591503285), "leaf": (1154.514167560571, 2074.3709839909443)}
STATUS_S1 = {"tau_s": "F(t)（D3a 分支 B）", "kappa_Q": "F(t)（D3c，2.06 未变）", "B_dep": "C（D3b）；本模式放开 = F",
             "p_dep": "F", "kappa_c": "F"}

OPEN_J, OPEN_SEP = 0.80, 0.50


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ================================================================================================ check
def cmd_check(a):
    os.chdir(ROOT)
    bad, warn = [], []
    for fn, h in EXPECT.items():
        if not os.path.exists(fn):
            bad.append(f"missing {fn}")
        elif sha256(fn) != h:
            bad.append(f"{fn}: sha256 {sha256(fn)[:12]}... expected {h[:12]}..." +
                       ("  (this is not the pose v2 track)" if fn.endswith("pitcher_pose.npz") else ""))
    for p in POURS:
        for f in ("disc.json", "stream_foot.json"):
            if not os.path.exists(f"raw_data/rig/results/{p}/{f}"):
                bad.append(f"missing raw_data/rig/results/{p}/{f}")
        for f in (f"raw_data/rig/targets/{p}.npz", f"raw_data/rig/video_obs/{p}.npz"):
            if not os.path.exists(f):
                bad.append(f"missing {f}")
    for f in ("raw_data/rig/results/full_pour_kona_poured165.json", "raw_data/rig/results/cup_register.json",
              "latte_imex/pitcher_geom/pitcher_kona700.json"):
        if not os.path.exists(f):
            bad.append(f"missing {f}")
    os.environ.setdefault("LATTE_POSE", "v2")
    try:
        import latte_imex
        from latte_imex.rig import real_program as RP
        from latte_imex.rig import measured_inlet as MI
        if os.path.realpath(os.path.dirname(latte_imex.__file__)) != os.path.join(ROOT, "latte_imex"):
            bad.append(f"latte_imex is imported from another checkout: {os.path.dirname(latte_imex.__file__)}")
        if RP.pose_version() != "v2":
            bad.append(f"pose track {RP.pose_version()} (update21 runs pose v2; LATTE_POSE={os.environ.get('LATTE_POSE')})")
        if RP.INLET_DEFAULT != "record" or "replay" not in RP.INLET_MODES:
            bad.append("latte_imex.rig.real_program is not the mi27 build (INLET_MODES, INLET_DEFAULT 'record')")
        for p in CAL:
            K = MI.load(p)
            b = MI.check_keys(K)
            if b:
                bad.append(f"{p}: stream_kin.json carries keys outside the allow-list: {b[:5]}")
        from latte_imex.diff import diag as DG
        if "S1" not in DG.PSETS or "meas" not in DG.INLET_BOX or DG.HELDOUT_INLET != "law":
            bad.append("latte_imex.diff.diag is not the mi27 build (PSETS['S1'], INLET_BOX['meas'], HELDOUT_INLET)")
    except Exception as e:                                                  # noqa: BLE001
        bad.append(f"import / load failed: {type(e).__name__}: {e}")
    # the start closures (SPEC27 6.1; PREREG27_2 D3, D4)
    try:
        A0 = json.load(open("closures/bp256_trained.json"))
        A0mi = json.load(open("latte_imex/data/closures/bp256_mi.json"))
        S = json.load(open("latte_imex/data/closures/mcd_S1kq_mi_N96.json"))
        if "inlet" in A0:
            bad.append("closures/bp256_trained.json carries an 'inlet' key (the control must run the record inlet)")
        if A0mi.get("inlet") != 2 or any(A0mi.get(k) != v for k, v in A0.items() if not k.startswith("_")):
            bad.append("bp256_mi.json is not bp256_trained.json with inlet 2 (PREREG27_2 D4)")
        if S.get("inlet") != 2 or abs(S.get("tau_s", 0) - 0.0347774) > 1e-6 or S.get("kappa_Q") != 2.06:
            bad.append("mcd_S1kq_mi_N96.json is not the D3a-D3c start (inlet 2, tau_s 0.0347774, kappa_Q 2.06)")
    except Exception as e:                                                  # noqa: BLE001
        bad.append(f"closures: {type(e).__name__}: {e}")
    if not os.path.exists("fonts/wqy-zenhei.ttc"):
        warn.append("fonts/wqy-zenhei.ttc missing: figure labels fall back to a font without Chinese glyphs")
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:                                                  # noqa: BLE001
        warn.append(f"no ffmpeg for the movies ({e}); the scores are unaffected")
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < a.min_free_gb:
        warn.append(f"only {free:.1f} GB free on the disk of {ROOT} (results and videos need a few GB)")
    if a.device.startswith("cuda"):
        try:
            import warp as wp
            wp.init()
            if not wp.is_cuda_available():
                bad.append("warp sees no CUDA device")
        except Exception as e:                                              # noqa: BLE001
            bad.append(f"warp: {e}")
    for w in warn:
        print("WARNING:", w)
    for b in bad:
        print("FAIL:", b)
    print(f"start-up check: {'PASS' if not bad else 'FAIL'} ({len(EXPECT)} frozen inputs by sha256, pose v2, measured "
          f"inlet allow-list, start closures, {len(warn)} warning(s))")
    return 0 if not bad else 1


# ================================================================================================ timing
def cmd_timing(a, rest):
    """Run `rest` (a diag run writing --out a.out) in its own process group; stop it after a.k iteration records (or
    when it ends, or after a.max_min minutes) and write timing.json: the per-iteration wall time (fwd + bwd, both pours
    in parallel); the S1kq / v05n ratio sets the S1kq slots."""
    os.makedirs(a.out, exist_ok=True)
    rec = os.path.join(a.out, "records.jsonl")
    t0 = time.time()
    with open(os.path.join(a.out, "timing.log"), "ab") as lg:
        pr = subprocess.Popen(rest, stdout=lg, stderr=subprocess.STDOUT, start_new_session=True)
        t_first, n = None, 0
        while True:
            rc = pr.poll()
            try:
                n = sum(1 for l in open(rec) if l.strip()) if os.path.exists(rec) else 0
            except OSError:
                n = 0
            if n >= 1 and t_first is None:
                t_first = time.time()
            if n >= a.k or rc is not None or time.time() - t0 > a.max_min * 60:
                break
            time.sleep(3)
        killed = False
        if pr.poll() is None:
            killed = True
            try:
                os.killpg(pr.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(60):
                if pr.poll() is not None:
                    break
                time.sleep(2)
            if pr.poll() is None:
                os.killpg(pr.pid, signal.SIGKILL)
                pr.wait()
        # wait for the worker processes of the group (CUDA contexts) to go
        for _ in range(60):
            try:
                os.killpg(pr.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(2)
        else:
            try:
                os.killpg(pr.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    recs = [json.loads(l) for l in open(rec) if l.strip()] if os.path.exists(rec) else []
    walls = [float(r["wall_s"]) for r in recs]
    res = dict(ok=len(walls) >= 1, iterations=len(walls), wall_s=walls, t_it_s=(walls[-1] if walls else None),
               fwd_s={p: v["fwd_s"] for p, v in recs[-1]["per"].items()} if recs else None,
               bwd_s={p: v["bwd_s"] for p, v in recs[-1]["per"].items()} if recs else None,
               start_s=(t_first - t0 - walls[0]) if (t_first and walls) else None, elapsed_s=time.time() - t0,
               returncode=pr.returncode, stopped_by_timing=killed, cmd=rest)
    with open(os.path.join(a.out, "timing.json"), "w") as f:
        json.dump(res, f, indent=1)
    if res["ok"]:
        print(f"timing: {len(walls)} iteration(s), {walls} s per iteration (both pours), start-up "
              f"{res['start_s']:.0f} s" if res["start_s"] is not None else f"timing: {walls}")
    else:
        print(f"timing: no iteration finished (rc {pr.returncode}); see {a.out}/timing.log")
    return 0 if res["ok"] else 1


# ================================================================================================ plan + slot
def cmd_plan(a):
    """plan.json: per run the planned fitting hours and its overheads (render; held-out; bridges).  The S1kq runs are
    scaled by the measured S1kq iteration time against the v05n iteration time measured in the same start-up step
    (bp256_mi, MI-R, a.timing_v; diag256's 90 s under the record inlet, a.v_it_s, only when that timing failed), so S1kq
    gets the same number of Adam iterations as the v05n run of the same mode, within [s1_min_h, s1_max_h]."""
    runs = a.runs.split()
    tS, why = None, "no timing (S1kq slots as v05n)"
    tV, vsrc = a.v_it_s, f"{a.v_it_s:.0f} s (diag256, record inlet; no v05n timing)"
    if getattr(a, "timing_v", None) and os.path.exists(a.timing_v):
        TV = json.load(open(a.timing_v))
        if TV.get("ok") and TV.get("t_it_s"):
            tV = float(TV["t_it_s"])
            vsrc = f"{tV:.0f} s (measured, v05n bp256_mi, MI-R, N {a.N})"
    if a.timing and os.path.exists(a.timing):
        T = json.load(open(a.timing))
        if T.get("ok") and T.get("t_it_s"):
            tS = float(T["t_it_s"])
            why = f"S1kq {tS:.0f} s per iteration (measured, N {a.N}) vs v05n {vsrc}"
    plan = {}
    for r in runs:
        m, md = r.split("/")
        fit = a.pfit_h if md == "params" else a.fit_h
        ovh = a.render_h + (a.heldout_h if md == "params" else 0.0) + (a.bridge_h if r in MI_PARAMS else 0.0)
        note = ""
        if m == "s1kq" and tS is not None:
            f0 = fit
            fit = min(max(fit * tS / tV, a.s1_min_h), a.s1_max_h)
            note = f"{f0:.2f} h x {tS:.0f} / {tV:.0f} s -> {fit:.2f} h (~{fit * 3600 / tS:.0f} iterations)"
        plan[r] = dict(fit_h=round(fit, 3), ovh_h=round(ovh, 3), note=note)
    J = dict(runs=runs, plan=plan, why=why, t_it_s1kq=tS, v_it_s=tV, v_it_src=vsrc, made=time.strftime("%F %T"))
    with open(os.path.join(a.out, "plan.json"), "w") as f:
        json.dump(J, f, indent=1)
    tot = sum(v["fit_h"] + v["ovh_h"] for v in plan.values())
    print(f"plan ({why}): " + ", ".join(f"{r} {v['fit_h']:.2f}+{v['ovh_h']:.2f} h" for r, v in plan.items())
          + f"; total {tot:.1f} h")
    return 0


def slot_hours(plan, rem_h, runs, floor=0.25):
    """Fitting hours of runs[0], the others following it in order.  Ahead of plan: the spare time is shared in
    proportion to the planned fitting hours.  Behind plan: runs[0] keeps its plan as far as the remaining budget allows
    (nothing is reserved for the later runs, so the deficit falls on the end of the list, the user's priority order);
    once the budget is used up every remaining run gets the floor (the budget is soft: each of them then takes the floor
    plus its render / held-out time beyond BUDGET_H)."""
    P = [plan.get(r, dict(fit_h=1.6, ovh_h=0.3)) for r in runs]
    need = sum(p["fit_h"] + p["ovh_h"] for p in P)
    slack = rem_h - need
    if slack >= 0:
        return P[0]["fit_h"] + slack * P[0]["fit_h"] / max(sum(p["fit_h"] for p in P), 1e-9)
    return max(floor, min(P[0]["fit_h"], rem_h - P[0]["ovh_h"]))


def cmd_slot(a):
    J = json.load(open(a.plan))
    rem = a.budget_h - (time.time() - a.t0) / 3600.0
    print(f"{slot_hours(J['plan'], rem, a.runs):.3f}")
    return 0


# ================================================================================================ bridges
def cmd_bridge_closure(a):
    """SPEC27 6.1 bridges: the params-mode best closure with only its 'inlet' key changed (1: MI-law for tulip and leaf,
    bridge (i); 0: the record inlet R1 with E0 on the v1 feet for the held-out pours, bridge (ii))."""
    C = json.load(open(a.src))
    C["inlet"] = int(a.inlet)
    C["_note_bridge"] = (f"update21 bridge ({'i' if a.inlet == 1 else 'ii'}): {os.path.relpath(a.src, ROOT)} with inlet "
                         f"{a.inlet} ({'MI-law, e0 off' if a.inlet == 1 else 'record R1, E0 world on the v1 feet'}); "
                         "every number unchanged (SPEC27 6.1; report only)")
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    with open(a.dst, "w") as f:
        json.dump(C, f, indent=1)
    return 0


# ================================================================================================ summary
def _opens(ev):
    try:
        from latte_imex.diff.diag import opens
        return bool(opens(ev))
    except Exception:                                                       # noqa: BLE001 (same rule as diag.opens)
        b = ev.get("tulip", {}).get("best") if ev else None
        return bool(b and math.isfinite(b.get("J", float("nan"))) and b["J"] >= OPEN_J and b.get("sep", 0.0) >= OPEN_SEP)


def _f(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return "nan" if not math.isfinite(x) else f"{x:.{nd}f}"


def _load(fn):
    try:
        return json.load(open(fn))
    except Exception:                                                       # noqa: BLE001
        return None


def _tail(fn, n=3):
    try:
        L = [l.rstrip() for l in open(fn, errors="replace") if l.strip() and not l.startswith("Module ")]
        return " / ".join(L[-n:])[-400:]
    except OSError:
        return ""


def _state(out, run):
    """(status text, done dict or None)."""
    m, md = run.split("/")
    d = os.path.join(out, m, md)
    R = _load(os.path.join(d, "done.json"))
    if R is not None:
        return "完成", R
    E = _load(os.path.join(d, "eval_cal.json"))
    if E is not None and E.get("heldout_failed"):       # MI params run: the calibration evaluation exists, held-out fatal
        hf = E["heldout_failed"]
        return (f"留出杯 {hf.get('pour')} 失败（致命，无 done.json；标定杯数字来自 eval_cal.json，只供参考，不作结论；"
                f"{str(hf.get('error', ''))[:160]}）", dict(E, _heldout_fatal=True))
    sf = os.path.join(out, "_smoke", "failed.txt")
    if os.path.exists(sf) and run in open(sf).read().split():
        return "SMOKE 失败，跳过", None
    if os.path.exists(os.path.join(d, "records.jsonl")):
        n = sum(1 for l in open(os.path.join(d, "records.jsonl")) if l.strip())
        return f"未完成（{n} 次迭代；{_tail(os.path.join(out, m, md + '.log'), 1)}）", None
    return "未运行", None


def _ev(R, p):
    e = (R or {}).get("eval", {}).get(p, {})
    return e if "best" in e else None


def _js(R, p, k="J"):
    e = _ev(R, p)
    if e is None:
        return "—"
    nd = 3 if k == "J" else 2
    return f"{_f(e['start'].get(k), nd)} → {_f(e['best'].get(k), nd)}"


def _yn(R):
    if R is not None and R.get("_heldout_fatal"):
        return "不计（留出杯致命失败）"
    return "—" if R is None else ("**是**" if _opens(R["eval"]) else "否")


def _verdict_open(R):
    return None if (R is None or R.get("_heldout_fatal")) else _opens(R["eval"])


MODE_CN = {"params": "params：只放开参数，入口按测量", "inlet": "inlet：参数固定，入口在 'meas' 盒内修正（oracle）",
           "both": "both：参数 + 入口（'meas' 盒）", "both_wide": "both_wide：参数 + 入口（'meas_wide' 盒，×2.5）"}


def _runs_table(out, model):
    L = ["| 试验 | 状态 | tulip J 起点→结果 | tulip 分离率 | leaf J | leaf 分离率 | 迭代 / 小时 | 打开？ |",
         "|---|---|---|---|---|---|---|---|"]
    for md in MODELS[model]["modes"]:
        st, R = _state(out, f"{model}/{md}")
        it = f"{R.get('iterations', '—')} / {_f(R.get('hours'), 2)}" if R else "—"
        lab = MODE_CN[md] if MODELS[model]["mi"] else "params：只放开参数，旧入口 R1（pose v2）"
        L.append(f"| {lab} | {st} | {_js(R, 'tulip')} | {_js(R, 'tulip', 'sep')} | {_js(R, 'leaf')} | "
                 f"{_js(R, 'leaf', 'sep')} | {it} | {_yn(R)} |")
    return L


def _inlet_cmp(R, label):
    """Correction statistics of an oracle run, measured windows vs law-only frames (D11)."""
    L = []
    if R is None:
        return L
    rows = []
    for p in CAL:
        e = _ev(R, p)
        inl = (e or {}).get("inlet") or {}
        for reg, cn in (("windows", "测量窗口（w = 1）"), ("law_only", "只有定律（w = 0）")):
            s = inl.get(reg)
            if not s:
                continue
            w2 = s.get("within_meas_2sigma")
            rows.append(f"| {p} | {cn} | {s.get('frames', '—')} | {_f(s['shift_mm_p50'], 1)} / {_f(s['shift_mm_p90'], 1)} | "
                        f"×{_f(s['Q_ratio_p10'], 2)}–{_f(s['Q_ratio_p90'], 2)}（总量 ×{_f(s['Q_total_ratio'], 2)}） | "
                        f"×{_f(s['U_ratio_p10'], 2)}–{_f(s['U_ratio_p90'], 2)} | {_f(s['du_ms_p50'], 3)} / {_f(s['du_ms_p90'], 3)} | "
                        + (f"{_f(w2, 2)}" if reg == "windows" else f"（{_f(w2, 2)}，不作测量判读）") + " |")
    if rows:
        L += ["", f"**入口修正（{label}；oracle，只作诊断，不进入模型）**", "",
              "| 杯 | 区域 | 帧数 | 落点平移 p50 / p90 [mm] | Q 倍率 p10–p90 | U_perp 倍率 p10–p90 | du p50 / p90 [m/s] | 在测量 2σ 内的帧比例 |",
              "|---|---|---|---|---|---|---|---|"] + rows + [
              "", "“在测量 2σ 内”只在测量窗口内有意义；只有定律的帧（高位倒奶、空档）上的改动是对倒奶定律的差异，不是测量误差。"]
    return L


def _heldout(out, model, R):
    L = []
    if R is None:
        return L
    rows = [f"| {p} | {'MI-L（暂定）' if MODELS[model]['mi'] else 'R1（旧入口）'} | {_js(R, p)} | {_js(R, p, 'sep')} |"
            for p in HELD if _ev(R, p)]
    if not rows:
        return L
    L += ["", "**留出三杯（参数冻结后直接跑；只报告，不参与任何选择）**", "",
          "| 杯 | 入口 | J 起点→结果 | 分离率 |", "|---|---|---|---|"] + rows
    if MODELS[model]["mi"]:
        Bi = _load(os.path.join(out, model, "bridge_mil", "done.json"))
        bl = ["", "桥（SPEC27 6.1，同一组拟合参数，只换入口）：", "",
              "| 杯 | 拟合时的入口 → J | 换入口 → J | 差 |", "|---|---|---|---|"]
        for p in CAL:
            e, b = _ev(R, p), _ev(Bi, p)
            if e and b:
                bl.append(f"| {p}（桥 i：MI-R → MI-L） | {_f(e['best']['J'])} | {_f(b['best']['J'])} | "
                          f"{float(b['best']['J']) - float(e['best']['J']):+.3f} |")
        for p in HELD:
            e = _ev(R, p)
            b = _ev(_load(os.path.join(out, model, f"bridge_r1_{p}", "done.json")), p)
            if e and b:
                bl.append(f"| {p}（桥 ii：MI-L → R1） | {_f(e['best']['J'])} | {_f(b['best']['J'])} | "
                          f"{float(b['best']['J']) - float(e['best']['J']):+.3f} |")
        if len(bl) > 5:
            L += bl + ["", "留出杯的变化只有扣除桥 (i) 的切换效应（标定杯上 MI-R → MI-L 的差）之后，才算重新拟合的效果。"]
        else:
            L += ["", "桥：没有结果（见 " + f"{model}/bridge_*.log）。"]
    return L


def _s1_params(R0, R):
    L = []
    if R is None:
        return L
    S0 = _load(os.path.join(ROOT, MODELS["s1kq"]["init"])) or {}
    L += ["", "**拟合参数（参数集 S1：五个 F 项）与 MI 重推的物理范围（SPEC27 4.2；只报告）**", "",
          "| 参数 | 起点 | 结果 | 起点状态 | 物理范围（测量入口） |", "|---|---|---|---|---|"]
    for k in ("tau_s", "kappa_Q", "B_dep", "p_dep", "kappa_c"):
        v0, v1 = S0.get(k), R["params"].get(k)
        rng = "—（无独立估计）"
        if k == "tau_s" and v1 is not None:
            ds = " / ".join(f"{p} {v1 * U_MED_MS[p] * 1e3:.1f} mm（EXB {EXB_MM[p][0]:.2f}–{EXB_MM[p][1]:.2f}）" for p in CAL)
            rng = f"δ_s = τ_s·中位 \\|u_in\\|：{ds}"
        if k == "B_dep" and v1 is not None:
            rng = "; ".join(f"{p} 窗口 {BDEP_WIN[p][0]:.0f}–{BDEP_WIN[p][1]:.0f}，高位 {BDEP_HIGH[p][0]:.0f}–{BDEP_HIGH[p][1]:.0f}"
                            + ("（在内）" if BDEP_WIN[p][0] <= v1 <= BDEP_WIN[p][1] and BDEP_HIGH[p][0] <= v1 <= BDEP_HIGH[p][1]
                               else "（不全在内）") for p in CAL)
        L.append(f"| {k} | {_f(v0, 4)} | {_f(v1, 4)} | {STATUS_S1[k]} | {rng} |")
    return L


def _v05_params(R):
    if R is None:
        return []
    S0 = _load(os.path.join(ROOT, MODELS["v05n"]["init"])) or {}
    return ["", "**拟合参数（9 个，宽范围，是否物理不管）**", "", "| 参数 | 起点 | 结果 |", "|---|---|---|"] + [
        f"| {k} | {_f(S0.get(k), 4)} | {_f(v, 4)} |" for k, v in R["params"].items()]


def _read(o):
    return "未完成（不知道）" if o is None else ("打开" if o else "没打开")


def summary_s1kq(out):
    _, Rp = _state(out, "s1kq/params")
    _, Ri = _state(out, "s1kq/inlet")
    op, oi = _verdict_open(Rp), _verdict_open(Ri)
    if op is True:
        v = "params 打开：测量入口加上五个 F 项重新拟合，S1kq 能打开郁金香。"
    elif op is False and oi is True:
        v = "params 没打开、inlet 打开：盒内还有一个入口残差在起作用（看下表是测量窗口还是只有定律的帧承担了修正）。"
    elif op is False and oi is False:
        v = "params、inlet 都没打开：S1kq 在测量入口下是模型本身的问题。"
    elif op is False:
        v = "params 没打开；inlet 还没有结果，表的后两行还不能区分。"
    elif Rp is not None and Rp.get("_heldout_fatal"):
        v = "params 的留出杯致命失败（没有 done.json），不作结论；标定杯数字见上表，只供参考。"
    else:
        v = "params 还没有结果，不能下结论。"
    L = ["# update21 · 物理模型 S1kq（新倒奶模型 = 测量入口 MI-R，pose v2，oracle 盒 'meas'）", "",
         f"起点 `{MODELS['s1kq']['init']}`：mcd_S1kq_N96 + τ_s 0.0347774 s（F(t)，D3a 分支 B）、κ_Q 2.06（F(t)）、"
         "B_dep 1267.43（C）、inlet 2（MI-replay，E0 在验证过的落点上，LATTE_FOOT=mi）。",
         f"判据（事先定）：tulip J ≥ {OPEN_J} 且 末段分离率 ≥ {OPEN_SEP} = “打开”。", ""] + _runs_table(out, "s1kq")
    L += ["", "## 判读（PREREG27_2 §5，S1kq 两模式表）", "",
          "| params（参数集 S1） | inlet（'meas'） | 判读 |", "|---|---|---|",
          "| 打开 | 任意 | 测量入口 + 五个 F 项重新拟合就能打开 S1kq |",
          "| 没打开 | 打开 | 盒内的入口残差在起作用（按区域报告） |",
          "| 没打开 | 没打开 | S1kq 在测量入口下是模型问题 |", "",
          f"本次：params {_read(op)}，inlet {_read(oi)} → **{v}**"]
    L += _s1_params(None, Rp) + _heldout(out, "s1kq", Rp) + _inlet_cmp(Ri, "S1kq inlet")
    L += ["", "oracle（入口修正）只作诊断：它不会变成入口、不会进闭合参数，也不用来在不同测量读数之间裁决（D11）。"]
    return L, dict(params=op, inlet=oi, verdict=v)


def summary_v05n(out):
    st = {md: _state(out, f"v05n/{md}") for md in MODELS["v05n"]["modes"]}
    o = {md: _verdict_open(st[md][1]) for md in st}
    _, Rc = _state(out, "v05n_ctrl/params")
    oc = _verdict_open(Rc)
    op = o["params"]
    tab = {(True, False): "测量入口能打开、旧入口不能：旧入口（记录）就是倒奶模型的瓶颈，测量入口把它去掉了。",
           (True, True): "两个都能打开：pose v2 + 重新拟合就够了，入口没有被证明是瓶颈。",
           (False, True): "测量入口反而不行、旧入口可以：对这一族，测量入口比记录差；要对照 R1 / R7 来读，不是看 oracle。",
           (False, False): "两个都打不开：看下面的模式表。"}
    v1 = tab.get((op, oc), "params 或对照还没有结果，不能下结论。")
    if st["params"][1] is not None and st["params"][1].get("_heldout_fatal"):
        v1 = "MI-R params 的留出杯致命失败（没有 done.json），不作结论；标定杯数字见上表，只供参考。"
    if op is True:
        v2 = "params 已打开（见上表）。"
    elif op is False and o["inlet"] is True:
        v2 = "params 没打开、inlet 打开：盒内的入口残差在起作用；在测量窗口内是测量 2σ（含结构项）以内的误差，在只有定律的帧上是倒奶定律的差异（见下表哪个区域承担了修正）。"
    elif op is False and o["both_wide"] is True and not any(o[m] for m in ("inlet", "both")) and \
            all(o[m] is not None for m in ("inlet", "both")):
        v2 = "只有 both_wide（'meas_wide'）打开：入口要离开测量的 2σ 才行，主要是模型问题。"
    elif op is False and all(o[m] is False for m in o):
        v2 = "全部没打开：测量入口下是模型问题。"
    else:
        v2 = "模式表还不完整（见上表），或结果不在表里（例如只有 both 打开：参数和入口都要改）。"
    try:
        from latte_imex.diff.diag import decide
        dv, notes = decide({m: o[m] for m in o if o[m] is not None})
    except Exception as e:                                                  # noqa: BLE001
        dv, notes = f"（diag.decide 不可用：{e}）", []
    L = ["# update21 · 基线 v05n（新倒奶模型 = 测量入口 MI-R，pose v2，oracle 盒 'meas'）", "",
         f"起点 `{MODELS['v05n']['init']}` = bp256_trained 的全部数值 + inlet 2（MI-replay，LATTE_FOOT=mi）。"
         f"对照 `{MODELS['v05n_ctrl']['init']}`：同一 pose v2、同一参数盒，旧入口 R1（记录 + E0 world + v1 落点）。",
         f"判据（事先定）：tulip J ≥ {OPEN_J} 且 末段分离率 ≥ {OPEN_SEP} = “打开”。", ""] + _runs_table(out, "v05n") + \
        [f"| 对照 params（pose v2，R1） | {_state(out, 'v05n_ctrl/params')[0]} | {_js(Rc, 'tulip')} | {_js(Rc, 'tulip', 'sep')} | "
         f"{_js(Rc, 'leaf')} | {_js(Rc, 'leaf', 'sep')} | "
         f"{(str(Rc.get('iterations')) + ' / ' + _f(Rc.get('hours'), 2)) if Rc else '—'} | {_yn(Rc)} |"]
    L += ["", "## 判读 1：测量入口 params vs 对照 params（PREREG27_2 §5）", "",
          "| MI-R params | 对照 params（R1） | 判读 |", "|---|---|---|",
          "| 打开 | 没打开 | 记录入口是倒奶模型的瓶颈；测量入口把它去掉了 |",
          "| 打开 | 打开 | pose v2 + 重新拟合就够了；入口没有被证明是瓶颈 |",
          "| 没打开 | 打开 | 对这一族测量入口比记录差；对照 R1 / R7 读，不看 oracle |",
          "| 没打开 | 没打开 | 看模式表 |", "",
          f"本次：MI-R params {_read(op)}，对照 {_read(oc)} → **{v1}**", "",
          "## 判读 2：模式表（MI-R）", "",
          "| 结果 | 判读 |", "|---|---|",
          "| params 没打开，inlet（'meas'）打开 | 盒内的入口残差在起作用：窗口内 = 测量 2σ 以内（含结构项），只有定律的帧 = 定律差异；报告哪个区域承担修正 |",
          "| 只有 both_wide（'meas_wide'）打开 | 入口要离开测量的 2σ：主要是模型问题 |",
          "| 都没打开 | 测量入口下是模型问题 |", "",
          f"本次：{', '.join(f'{m} {_read(o[m])}' for m in o)} → **{v2}**", "",
          f"README_diag256 的四模式判据（这里读作“在 pose v2 + MI-R、'meas' 盒下”）：{dv}"] + notes + [
          "", "注意：B–D 模式不能直接和 diag256 比——盒子变了（落点 ±4 vs ±8 mm，U ×1.19 vs ×1.4，du 都是 ±0.10 m/s），pose 也变了。"]
    L += _v05_params(st["params"][1]) + _heldout(out, "v05n", st["params"][1])
    for md in ("inlet", "both", "both_wide"):
        L += _inlet_cmp(st[md][1], f"v05n {md}")
    L += ["", "oracle（入口修正）只作诊断：它不会变成入口、不会进闭合参数，也不用来在不同测量读数之间裁决（D11）。"]
    return L, dict(params=op, ctrl=oc, modes=o, verdict_ctrl=v1, verdict_modes=v2)


def summary_ctrl(out):
    _, R = _state(out, "v05n_ctrl/params")
    L = ["# update21 · v05n 对照（pose v2 + 旧入口 R1，参数盒同 v05n params）", "",
         "作用：把 pose 的改变和入口的改变分开。判读在 `v05n/summary.md` 的“判读 1”。", ""] + _runs_table(out, "v05n_ctrl")
    L += _v05_params(R) + _heldout(out, "v05n_ctrl", R)
    return L


def cmd_summary(a):
    out = a.out
    os.makedirs(out, exist_ok=True)
    Ls, S = summary_s1kq(out)
    Lv, V = summary_v05n(out)
    Lc = summary_ctrl(out)
    for m, L in (("s1kq", Ls), ("v05n", Lv), ("v05n_ctrl", Lc)):
        os.makedirs(os.path.join(out, m), exist_ok=True)
        with open(os.path.join(out, m, "summary.md"), "w") as f:
            f.write("\n".join(L) + "\n")
    # top level, in the user's priority order
    N = "?"
    for r in ORDER:
        R = _state(out, r)[1]
        if R:
            N = R.get("N", "?")
            break

    def line(run, who):
        st, R = _state(out, run)
        if R is None:
            return f"- **{who}**：{st}。"
        if R.get("_heldout_fatal"):
            return (f"- **{who}**：{st}。标定杯（只供参考）：tulip J {_js(R, 'tulip')}，分离率 {_js(R, 'tulip', 'sep')}；"
                    f"leaf J {_js(R, 'leaf')}（{R.get('iterations')} 次迭代）。")
        return (f"- **{who}**：tulip J {_js(R, 'tulip')}，分离率 {_js(R, 'tulip', 'sep')}；leaf J {_js(R, 'leaf')} → "
                f"**{'打开' if _opens(R['eval']) else '没打开'}**（{R.get('iterations')} 次迭代，{_f(R.get('hours'), 2)} h）。")
    T = ["# update21：新倒奶模型（测量入口）+ 重新拟合，郁金香打开了吗？", "",
         f"N {N}；pose v2；标定杯 tulip、leaf 用测量入口 MI-R（墙面出口定律 + 测量窗口内的实测流速，暂定：rule 2 勉强通过）；"
         "留出三杯用 MI-L（只有定律，E0 关）。判据（事先定）：tulip J ≥ 0.80 且 末段分离率 ≥ 0.50。", "",
         "## 回答（按运行顺序）", "",
         line("s1kq/params", "物理模型 S1kq，只放开 5 个 F 参数，入口按测量"),
         line("v05n/params", "基线 v05n，9 个参数放开，入口按测量"),
         line("v05n_ctrl/params", "对照：v05n，pose v2 + 旧入口 R1"), "",
         f"- S1kq 判读：{S['verdict']}",
         f"- v05n 判读（测量入口 vs 对照）：{V['verdict_ctrl']}",
         f"- v05n 模式表：{V['verdict_modes']}", "",
         "## 全部试验", "", "| 顺序 | 试验 | 状态 | tulip J 起点→结果 | tulip 分离率 | 打开？ |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(ORDER, 1):
        st, R = _state(out, r)
        T.append(f"| {i} | {r} | {st} | {_js(R, 'tulip')} | {_js(R, 'tulip', 'sep')} | {_yn(R)} |")
    tj = _load(os.path.join(out, "_timing", "timing.json"))
    pj = _load(os.path.join(out, "plan.json"))
    T += ["", "## 说明", "",
          "- 第 4–7 个试验（inlet / both / both_wide）是 oracle：入口修正只作诊断，不会变成入口、不进模型参数，"
          "也不在不同测量读数之间裁决；“在测量 2σ 内”只在测量窗口内读。",
          "- 留出三杯只报告，从不参与选择；它们的入口 MI-L 是暂定的（rule 2 勉强通过）。",
          f"- S1kq 时间槽：{pj['why'] if pj else '—'}" + (f"（实测 {', '.join(_f(x, 0) for x in tj['wall_s'])} s / 次）" if tj and tj.get('wall_s') else ""),
          "- 细节：`s1kq/summary.md`、`v05n/summary.md`、`v05n_ctrl/summary.md`；每个试验目录里先看 `<杯>_stills.jpg`。"]
    sm = os.path.join(out, "_smoke", "status.txt")
    if os.path.exists(sm):
        T += ["", "SMOKE（N 32，1 次迭代）：" + "；".join(l.strip() for l in open(sm) if l.strip())]
    txt = "\n".join(T) + "\n"
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(txt)
    print(txt)
    return 0


# ================================================================================================ main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--min-free-gb", dest="min_free_gb", type=float, default=10.0)
    t = sub.add_parser("timing")
    t.add_argument("--out", required=True)
    t.add_argument("--k", type=int, default=2)
    t.add_argument("--max-min", dest="max_min", type=float, default=40.0)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--runs", default=" ".join(ORDER))
    p.add_argument("--timing", default=None)
    p.add_argument("--timing-v", dest="timing_v", default=None)
    p.add_argument("--N", default="256")
    p.add_argument("--fit-h", dest="fit_h", type=float, default=1.6)
    p.add_argument("--pfit-h", dest="pfit_h", type=float, default=1.35)
    p.add_argument("--render-h", dest="render_h", type=float, default=0.3)
    p.add_argument("--heldout-h", dest="heldout_h", type=float, default=0.3)
    p.add_argument("--bridge-h", dest="bridge_h", type=float, default=0.2)
    p.add_argument("--v-it-s", dest="v_it_s", type=float, default=90.0)
    p.add_argument("--s1-min-h", dest="s1_min_h", type=float, default=0.5)
    p.add_argument("--s1-max-h", dest="s1_max_h", type=float, default=2.5)
    s = sub.add_parser("slot")
    s.add_argument("--plan", required=True)
    s.add_argument("--budget-h", dest="budget_h", type=float, required=True)
    s.add_argument("--t0", type=float, required=True)
    s.add_argument("runs", nargs="+")
    b = sub.add_parser("bridge-closure")
    b.add_argument("src")
    b.add_argument("dst")
    b.add_argument("--inlet", type=int, choices=(0, 1), required=True)
    m = sub.add_parser("summary")
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "timing":
        if not rest:
            ap.error("timing needs the diag command after --")
        return cmd_timing(a, rest)
    return dict(check=cmd_check, plan=cmd_plan, slot=cmd_slot, summary=cmd_summary)[a.cmd](a) if a.cmd != "bridge-closure" \
        else cmd_bridge_closure(a)


if __name__ == "__main__":
    sys.exit(main())
