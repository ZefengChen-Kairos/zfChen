"""viz21 job planner (used by scripts/autodl_viz21.sh; no solver, no Warp: json + numpy only).

Which closures are rendered, where the videos go, which jobs are done:
  - every FINISHED params run results_diag21/<model>/params (done.json + best_closure.json):
      start  the run's START closure, as it is (起点参数，未重新拟合), under the run's own calibration inlet
      best   the run's best_closure.json (拟合后参数)
    -> results_diag21/<model>/params/viz/<start|best>/<pour>_{pour,guest,top,combo}.mp4, <pour>_sheet.jpg, ...
  - extra closures VIZ_CLOSURES="name=path[,inlet] ..." -> results_diag21/viz_extra/<name>/<pour>_*.mp4
  - pours: tulip, leaf (calibration) for every closure; heart, push_heart, swan (held out) for a closure whose
    calibration inlet is measured (law / replay: MI-L, e0 off, as diag's held-out test), and with --held-record also
    for a record closure (record inlet + the closure's own e0, as the control fit / the R1 bridges evaluated them)
  - a job is done when <viz dir>/<pour>.done names the same closure (sha256) and inlet argument AND the same render
    settings (N, fps, simulated length T, size, supersampling, cup; a key the marker lacks counts as the launcher
    default); --keep-render: size / ss / cup are not compared (the launcher's CPU fallback must not overwrite GPU
    renders with smaller ones); --force: every job is planned (markers ignored)

The start closure of a params run, first match of:
  1. done.json 'init' (diag writes the absolute path it was given; when the tree was moved, e.g. u20 -> u21, the same
     relative path is looked up under this tree)
  2. run.log ('== params: ..., start closure <path>, ...')
  3. the launcher's INIT table (scripts/autodl_diag21.sh, model_cfg: '<model>) INIT_M=<path>;')
  4. the repository's start closures (s1kq mcd_S1kq_mi_N96.json, v05n bp256_mi.json, v05n_ctrl bp256_trained.json)
A candidate must exist and agree with best_closure.json on every number that the run did not fit (best_closure.json =
the start closure's numbers with the fitted ones replaced: diag.best_closure); the first one that agrees is taken, else
the first that exists (with a warning).  Inlet: the run's calibration inlet (done.json 'inlet' -> 'cal' modes; no
'inlet' entry = record, as diag writes it), passed as --inlet only when the start closure would resolve differently.

    python jobs21.py plan --out OUT --root ROOT --fields FIELDS --N 256 [--models "s1kq v05n v05n_ctrl"]
        [--which both] [--pours "tulip leaf heart push_heart swan"] [--extras "name=path[,inlet] ..."]
        [--fps 30] [--T secs] [--size 1280x720] [--ss 2] [--cup dark] [--keep-render] [--force] [--held-record]
    python jobs21.py start RUN_DIR MODEL --root ROOT        (prints the start closure resolution as json)
    python jobs21.py migrate --out OUT --root ROOT --fields FIELDS --models ...   (old layout viz/<pour>_* -> viz/best/)
    python jobs21.py fields-ok NPZ SHA INLET [--N n] [--fps f] [--T secs|-]
                                                            (exit 0 when the cached fields are of that closure, inlet,
                                                             grid, frame rate and simulated length; else the reason)
    python jobs21.py mark VD POUR key=value ...             (the .done marker)
    python jobs21.py summary OUT                            (viz_summary.md)
plan prints one line per job, tab separated (no empty field: '-'): J model which pour closure inlet viz_dir
fields_npz sha model_desc, and information lines: I <text>.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

CAL = ("tulip", "leaf")
HELD = ("heart", "push_heart", "swan")
POURS = CAL + HELD
WHICH = ("start", "best")
INLET_CODES = {0: "record", 1: "law", 2: "replay"}
INLET_MODES = ("record", "law", "replay")
REPO_START = {"s1kq": ("latte_imex/data/closures/mcd_S1kq_mi_N96.json",),
              "v05n": ("latte_imex/data/closures/bp256_mi.json",),
              "v05n_ctrl": ("closures/bp256_trained.json", "latte_imex/data/mi27/bp256_trained.json")}
CLOSURE_DIRS = ("closures", "latte_imex/data/closures", "latte_imex/data/mi27")
MODEL_DESC = {"s1kq": "S1kq 物理模型", "v05n": "v05n 基线", "v05n_ctrl": "对照：旧入口 R1"}
VIDEO_FILES = ("{p}_pour.mp4", "{p}_guest.mp4", "{p}_top.mp4", "{p}_combo.mp4", "{p}_sheet.jpg", "{p}_viz.json",
               "{p}.log", "{p}.done")


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(fn):
    try:
        with open(fn) as f:
            return json.load(f)
    except Exception:                                            # noqa: BLE001
        return None


def inlet_mode(v):
    """record / law / replay from a name or a code 0 / 1 / 2; None for None / 'auto'; ValueError otherwise."""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "auto"):
        return None
    if s in INLET_MODES:
        return s
    try:
        c = float(s)
    except ValueError:
        raise ValueError(f"inlet must be record / law / replay / 0 / 1 / 2, got {v!r}") from None
    if c not in INLET_CODES:
        raise ValueError(f"inlet code must be 0, 1 or 2, got {v!r}")
    return INLET_CODES[int(c)]


def run_cal_mode(done):
    """Calibration inlet of a finished diag run from its done.json (diag writes 'inlet' only for an MI run)."""
    cal = ((done or {}).get("inlet") or {}).get("cal") or {}
    ms = sorted({v.get("mode") for v in cal.values() if isinstance(v, dict) and v.get("mode")})
    if len(ms) > 1:
        raise ValueError(f"done.json lists several calibration inlet modes {ms}")
    return ms[0] if ms else "record"


def closure_cal_mode(fn, inlet_arg="auto"):
    """The calibration inlet forward_fields.resolve gives a closure file (--inlet > key 'inlet' > sibling done.json >
    record; LATTE_INLET is unset for the re-runs, as for the fits)."""
    m = inlet_mode(inlet_arg)
    if m is not None:
        return m
    c = load_json(fn) or {}
    if c.get("inlet") is not None:
        return inlet_mode(c["inlet"])
    d = load_json(os.path.join(os.path.dirname(os.path.abspath(fn)), "done.json"))
    return run_cal_mode(d) if d else "record"


def is_mi(fn, inlet_arg="auto"):
    return closure_cal_mode(fn, inlet_arg) in ("law", "replay")


def reroot(path, root):
    """An existing file for `path`: itself, else the longest suffix of it (2+ components) under root, else its name
    in one of root's closure directories; None when nothing exists."""
    if not path:
        return None
    if not os.path.isabs(path):
        for base in (root, os.getcwd()):
            c = os.path.join(base, path)
            if os.path.isfile(c):
                return os.path.abspath(c)
    elif os.path.isfile(path):
        return path
    parts = [x for x in path.replace("\\", "/").split("/") if x not in ("", ".")]
    for k in range(len(parts) - 1):
        if len(parts) - k < 2:
            break
        c = os.path.join(root, *parts[k:])
        if os.path.isfile(c):
            return os.path.abspath(c)
    for d in CLOSURE_DIRS:
        c = os.path.join(root, d, parts[-1]) if parts else None
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def _num(c):
    return {k: float(v) for k, v in c.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
            and not k.startswith("_")}


def agrees(start_fn, best, done):
    """[] when best_closure.json (dict) equals the start closure on every number the run did not fit, else the keys
    that differ (diag.best_closure: dict(start numbers, **fitted))."""
    st = load_json(start_fn)
    if not isinstance(st, dict) or not isinstance(best, dict):
        return ["unreadable"]
    fitted = set(((done or {}).get("params") or {}).keys())
    b = _num(best)
    bad = []
    for k, v in _num(st).items():
        if k in fitted:
            continue
        if k not in b or abs(b[k] - v) > 1e-9 * max(1.0, abs(v)):
            bad.append(k)
    return bad


def launcher_inits(root, launcher=None):
    """{model: INIT_M path} from the launcher's model_cfg table."""
    fn = launcher or os.path.join(root, "scripts", "autodl_diag21.sh")
    out = {}
    try:
        txt = open(fn).read()
    except OSError:
        return out
    for m, p in re.findall(r"^\s*([A-Za-z0-9_]+)\)\s+INIT_M=([^;\s]+)\s*;", txt, re.M):
        out.setdefault(m, p.strip("'\""))
    return out


def runlog_init(run_dir):
    try:
        with open(os.path.join(run_dir, "run.log")) as f:
            for line in f:
                m = re.search(r"start closure (.+?), closure (?:free|fixed)", line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return None


def resolve_start(run_dir, model, root, launcher=None):
    """dict(path, source, inlet, run_inlet, own_inlet, note, warn) of a finished params run's start closure, or
    dict(path=None, warn=...)."""
    done = load_json(os.path.join(run_dir, "done.json")) or {}
    best = load_json(os.path.join(run_dir, "best_closure.json"))
    cands = [("done.json init", done.get("init")), ("run.log", runlog_init(run_dir)),
             ("launcher INIT table", launcher_inits(root, launcher).get(model))]
    cands += [("repository default", p) for p in REPO_START.get(model, ())]
    found = []
    for src, p in cands:
        q = reroot(p, root) if p else None
        if q and q not in [f[1] for f in found]:
            found.append((src, q, p))
    if not found:
        return dict(path=None, warn=f"{model}: start closure not found (tried {[c[1] for c in cands if c[1]]})")
    try:
        run_in = run_cal_mode(done)
    except ValueError as e:
        return dict(path=None, warn=f"{model}: {e}")
    pick, warn = None, ""
    # first a candidate that agrees and resolves to the run's inlet by itself, then one that agrees (its inlet is then
    # passed explicitly), else the first that exists
    for want_inlet in (True, False):
        for src, q, p in found:
            if not agrees(q, best, done) and (not want_inlet or closure_cal_mode(q) == run_in):
                pick = (src, q, p)
                break
        if pick:
            break
    if pick is None:
        src, q, p = found[0]
        pick = found[0]
        warn = (f"{model}: no start-closure candidate agrees with best_closure.json on the unfitted numbers "
                f"({src} {q} differs in {agrees(q, best, done)[:6]}); using it anyway")
    src, q, p = pick
    own = closure_cal_mode(q)
    note = src + ("" if os.path.abspath(p) == q else f" ({p} -> {q})") if p else src
    return dict(path=q, source=src, note=note, run_inlet=run_in, own_inlet=own,
                inlet="auto" if own == run_in else run_in, warn=warn)


def parse_extras(spec, root):
    """'name=path[,inlet] ...' -> [(name, path or None, inlet, error)]"""
    out = []
    for tok in (spec or "").split():
        if "=" not in tok:
            out.append((tok, None, "auto", f"VIZ_CLOSURES entry {tok!r} is not name=path[,inlet]"))
            continue
        name, rest = tok.split("=", 1)
        path, _, inl = rest.partition(",")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or name in (".", ".."):
            out.append((name, None, "auto", f"VIZ_CLOSURES name {name!r}: letters, digits, . _ - only"))
            continue
        try:
            inl_n = inlet_mode(inl) or "auto"
        except ValueError as e:
            out.append((name, None, "auto", f"VIZ_CLOSURES {name}: {e}"))
            continue
        q = path if os.path.isabs(path) else (os.path.join(root, path) if os.path.isfile(os.path.join(root, path))
                                               else os.path.abspath(path))
        if not os.path.isfile(q):
            out.append((name, None, inl_n, f"VIZ_CLOSURES {name}: {path} not found"))
            continue
        out.append((name, os.path.abspath(q), inl_n, None))
    return out


# render settings bound into a .done marker.  A key the marker lacks was written with the launcher's default (the first
# viz21 layout recorded N, size and fps only; T = the simulated length, None = the whole pour)
RENDER_DEFAULTS = dict(N=256, fps=30.0, T=None, size="1280x720", ss=2, cup="dark")
PICTURE_KEYS = ("size", "ss", "cup")          # not compared with --keep-render


def norm_setting(k, v):
    """A render setting in one comparable form (N, ss int; fps, T float; T None = whole pour; size, cup lower case)."""
    if v is None:
        return None
    s = str(v).strip()
    if k == "T":
        return None if s.lower() in ("", "-", "none", "full", "auto") else round(float(s), 6)
    if k == "fps":
        return round(float(s), 6)
    if k in ("N", "ss"):
        return int(round(float(s)))
    return s.lower()


def _show(k, v):
    return "whole pour" if k == "T" and v is None else str(v)


def marker_check(fn, sha, inlet, want=None, keep_render=False):
    """'' when the marker fn names this closure (sha256), inlet argument and render settings `want` ({key of
    RENDER_DEFAULTS: value}), else the reason: 'no marker', 'another closure', 'inlet ...', 'size 640x360 -> 1280x720'."""
    j = load_json(fn)
    if not isinstance(j, dict):
        return "no marker"
    if j.get("closure_sha256") != sha:
        return "another closure"
    if str(j.get("inlet", "auto")) != str(inlet):
        return f"inlet {j.get('inlet', 'auto')} -> {inlet}"
    diff = []
    for k, v in (want or {}).items():
        if keep_render and k in PICTURE_KEYS:
            continue
        try:
            have, w = norm_setting(k, j.get(k, RENDER_DEFAULTS.get(k))), norm_setting(k, v)
        except ValueError:
            have, w = j.get(k), v
        if have != w:
            diff.append(f"{k} {_show(k, have)} -> {_show(k, w)}")
    return ", ".join(diff)


def marker_ok(fn, sha, inlet, want=None, keep_render=False):
    return marker_check(fn, sha, inlet, want, keep_render) == ""


def plan(a):
    out, root, fields = os.path.abspath(a.out), os.path.abspath(a.root), os.path.abspath(a.fields)
    models = [m for m in a.models.split() if m != "none"]
    which = WHICH if a.which == "both" else (a.which,)
    pours = [p for p in POURS if p in a.pours.split()]
    jobs, info = [], []
    # closures: (kind, model, which, closure, inlet, viz dir, fields dir, desc)
    cl = []
    for name, path, inl, err in parse_extras(a.extras, root):
        if not err:
            try:
                if not isinstance(load_json(path), dict):
                    raise ValueError("not a json object")
                closure_cal_mode(path, inl)                      # (an unreadable 'inlet' key: this closure only)
            except Exception as e:                               # noqa: BLE001
                err = f"VIZ_CLOSURES {name}: {e}"
        if err:
            info.append(err + ": skipped")
            continue
        cl.append(("extra", name, "extra", path, inl, os.path.join(out, "viz_extra", name),
                   os.path.join(fields, "extra", name), "-"))
    for m in models:
        run = os.path.join(out, m, "params")
        if not (os.path.isfile(os.path.join(run, "done.json")) and os.path.isfile(os.path.join(run, "best_closure.json"))):
            info.append(f"{m}/params: not finished (no done.json / best_closure.json): skipped")
            continue
        for w in which:
            if w == "start":
                r = resolve_start(run, m, root, a.launcher)
                if r.get("warn"):
                    info.append("WARNING: " + r["warn"])
                if not r.get("path"):
                    continue
                info.append(f"{m} start closure: {r['path']} [{r['note']}]; calibration inlet {r['run_inlet']}"
                            + ("" if r["inlet"] == "auto" else f" (passed as --inlet {r['inlet']}: the closure alone "
                                                               f"would give {r['own_inlet']})"))
                path, inl = r["path"], r["inlet"]
            else:
                path, inl = os.path.join(run, "best_closure.json"), "auto"
            cl.append(("model", m, w, path, inl, os.path.join(run, "viz", w), os.path.join(fields, m, w),
                       MODEL_DESC.get(m) or "-"))
    shas = {c[3]: sha256(c[3]) for c in cl}
    want = dict(N=a.N, fps=getattr(a, "fps", 30.0), T=getattr(a, "T", None), size=getattr(a, "size", "1280x720"),
                ss=getattr(a, "ss", 2), cup=getattr(a, "cup", "dark"))
    force, keep, held_rec = getattr(a, "force", False), getattr(a, "keep_render", False), getattr(a, "held_record", False)
    for p in pours:
        for kind, m, w, path, inl, vd, fd, desc in cl:
            if p in HELD and not (is_mi(path, inl) or held_rec):
                continue
            why = marker_check(os.path.join(vd, f"{p}.done"), shas[path], inl, want, keep)
            if why == "" and not force:
                info.append(f"{m} {w} {p}: done before (skipped)")
                continue
            if why == "":
                info.append(f"{m} {w} {p}: done before; VIZ_FORCE=1: redone")
            elif why not in ("no marker", "another closure"):
                info.append(f"{m} {w} {p}: done before with other settings ({why}): redone")
            ff = os.path.join(fd, f"{p}_fields_N{a.N}.npz")
            jobs.append("\t".join(("J", m, w, p, path, inl, vd, ff, shas[path], desc)))
    for s in info:
        print("I\t" + s)
    for j in jobs:
        print(j)


def fields_check(fn, sha, inlet, N=None, fps=None, T="any"):
    """'' when the cached forward fields are of the same closure (sha256) and inlet argument, and (when given) the same
    grid N, frame rate fps and simulated length T (None / '-' = the whole pour; 'any' = not checked); else the reason."""
    import numpy as np
    try:
        with np.load(fn) as Z:
            meta = json.loads(str(Z["meta"]))
    except Exception as e:                                       # noqa: BLE001
        return f"unreadable ({type(e).__name__})"
    if meta.get("closure_sha256") != sha:
        return "another closure"
    if inlet != "auto":
        if not (meta.get("cal_inlet") == inlet_mode(inlet) and meta.get("cal_inlet_source") == "--inlet"):
            return f"inlet {meta.get('cal_inlet')} [{meta.get('cal_inlet_source')}] -> --inlet {inlet}"
    elif meta.get("cal_inlet_source") == "--inlet":
        return f"inlet --inlet {meta.get('cal_inlet')} -> the closure's own"
    diff = []
    for k, v, mk in (("N", N, "N"), ("fps", fps, "fps"), ("T", T, "T_max")):
        if (k != "T" and v is None) or (k == "T" and str(v) == "any"):
            continue                                             # not checked
        have, w = norm_setting(k, meta.get(mk)), norm_setting(k, v)
        if have != w:
            diff.append(f"{k} {_show(k, have)} -> {_show(k, w)}")
    return ", ".join(diff)


def fields_ok(fn, sha, inlet, N=None, fps=None, T="any"):
    return fields_check(fn, sha, inlet, N, fps, T) == ""


def migrate(a):
    """The first viz21 layout (results_diag21/<model>/params/viz/<pour>_*, fields results_diag21_vizfields/<model>/
    <pour>_fields_N*.npz, both of best_closure.json) -> viz/best/ and fields/<model>/best/ (start/ when a field file is
    of the start closure).  Only files whose closure sha256 matches are moved."""
    out, root, fields = os.path.abspath(a.out), os.path.abspath(a.root), os.path.abspath(a.fields)
    for m in [x for x in a.models.split() if x != "none"]:
        run = os.path.join(out, m, "params")
        bf = os.path.join(run, "best_closure.json")
        if not os.path.isfile(bf):
            continue
        sb = sha256(bf)
        r = resolve_start(run, m, root, a.launcher) if os.path.isfile(os.path.join(run, "done.json")) else {}
        ss = sha256(r["path"]) if r.get("path") else None
        vd = os.path.join(run, "viz")
        for p in POURS:
            if not os.path.isfile(os.path.join(vd, f"{p}.done")):
                continue
            vj = load_json(os.path.join(vd, f"{p}_viz.json")) or {}
            s = (vj.get("meta") or {}).get("closure_sha256")
            if s != sb:
                print(f"I\t{m} {p}: old-layout videos in {vd} are not of best_closure.json (sha {str(s)[:8]}); left there")
                continue
            dst = os.path.join(vd, "best")
            os.makedirs(dst, exist_ok=True)
            n = 0
            for pat in VIDEO_FILES:
                f = os.path.join(vd, pat.format(p=p))
                if os.path.exists(f) and not os.path.exists(os.path.join(dst, os.path.basename(f))):
                    shutil.move(f, dst)
                    n += 1
            mk = os.path.join(dst, f"{p}.done")
            j = load_json(mk) or {}
            j.update(closure=bf, closure_sha256=sb, inlet="auto", which="best", migrated=time.strftime("%F %T"))
            json.dump(j, open(mk, "w"))
            print(f"I\t{m} {p}: old-layout videos moved to {dst} ({n} files)")
        for f in sorted(glob.glob(os.path.join(fields, m, "*_fields_N*.npz"))):
            import numpy as np
            try:
                with np.load(f) as Z:
                    meta = json.loads(str(Z["meta"]))
            except Exception:                                    # noqa: BLE001
                continue
            s = meta.get("closure_sha256")
            w = "best" if s == sb else ("start" if s is not None and s == ss and meta.get("cal_inlet_source") != "--inlet"
                                        else None)
            if w is None:
                continue
            dst = os.path.join(fields, m, w)
            os.makedirs(dst, exist_ok=True)
            if not os.path.exists(os.path.join(dst, os.path.basename(f))):
                shutil.move(f, dst)
                print(f"I\t{m}: old-layout fields {os.path.basename(f)} -> {dst}")


def mark(a):
    kv = dict(x.split("=", 1) for x in a.kv)
    for k in ("N", "fps", "ss", "T"):
        if k in kv:
            try:
                kv[k] = norm_setting(k, kv[k])
            except ValueError:
                pass
    if "t1" in kv:
        kv["wall_s"] = time.time() - float(kv.pop("t1"))
    kv["pour"] = a.pour
    kv["done"] = time.strftime("%F %T")
    fn = os.path.join(a.vd, f"{a.pour}.done")
    with open(fn + ".tmp", "w") as f:
        json.dump(kv, f, indent=1)
    os.replace(fn + ".tmp", fn)


WHICH_ZH = dict(start="起点参数（未重新拟合）", best="拟合后参数", extra="指定闭包")


def summary(a):
    out = os.path.abspath(a.out)
    rows = []
    found = []
    for d in sorted(glob.glob(os.path.join(out, "*", "params", "viz", "*"))):
        if os.path.isdir(d):
            found.append((d.split(os.sep)[-4], os.path.basename(d), d))
    for d in sorted(glob.glob(os.path.join(out, "viz_extra", "*"))):
        if os.path.isdir(d):
            found.append((os.path.basename(d), "extra", d))
    for d in sorted(glob.glob(os.path.join(out, "*", "params", "viz"))):
        if glob.glob(os.path.join(d, "*_viz.json")):
            found.append((d.split(os.sep)[-3], "old", d))               # the first layout, not migrated
    order = {p: i for i, p in enumerate(POURS)}
    recs = []
    for m, w, d in found:
        for fn in glob.glob(os.path.join(d, "*_viz.json")):
            j = load_json(fn)
            if not j:
                continue
            p = j.get("pour")
            recs.append((order.get(p, 9), m, {"start": 0, "best": 1}.get(w, 2), w, p, d, j))
    for _, m, _, w, p, d, j in sorted(recs, key=lambda r: r[:3]):
        ok = os.path.exists(os.path.join(d, f"{p}.done"))
        sn = j.get("snap_mm") or {}
        meta = j.get("meta") or {}
        rows.append(f"| {p} | {m} | {WHICH_ZH.get(w, w)} | {'ok' if ok else 'unfinished'} | {meta.get('inlet')} | "
                    f"{os.path.basename(str(meta.get('closure') or ''))} | "
                    f"{'' if sn.get('p50') is None else round(sn['p50'], 1)} | {j.get('frames')} | "
                    f"{os.path.relpath(d, out)}/{p}_{{pour,guest,top,combo}}.mp4 |")
    txt = ["# 三视角视频（viz21）", "",
           "每个完成的 params 拟合各渲染两套：**起点参数（未重新拟合）** = 拟合的起点闭包（v05n：bp256_mi.json，"
           "s1kq：mcd_S1kq_mi_N96.json，v05n_ctrl：bp256_trained.json + 旧入口 R1），**拟合后参数** = "
           "best_closure.json；同一倒奶、同一入口，只差参数。另有 VIZ_CLOSURES 指定的闭包（viz_extra/）。"
           "每套：倒奶视角 / 客人视角 / 俯视二维 / 2×2 对比（含真实俯视视频）+ <pour>_sheet.jpg。"
           "杯子为示意（只测了液面圆盘）；其余全部来自模拟链。", "",
           "| 倒奶 | 模型 | 参数 | 状态 | 入口 | 闭包 | 落点校正中位 [mm] | 帧数 | 视频 |",
           "|---|---|---|---|---|---|---|---|---|"] + rows
    open(os.path.join(out, "viz_summary.md"), "w").write("\n".join(txt) + "\n")
    print("\n".join(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "migrate"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--root", required=True)
        s.add_argument("--fields", required=True)
        s.add_argument("--models", default="s1kq v05n v05n_ctrl")
        s.add_argument("--launcher", default=None)
        if name == "plan":
            s.add_argument("--N", type=int, default=256)
            s.add_argument("--which", default="both", choices=("start", "best", "both"))
            s.add_argument("--pours", default=" ".join(POURS))
            s.add_argument("--extras", default="")
            s.add_argument("--fps", type=float, default=30.0)
            s.add_argument("--T", default=None, help="simulated length [s] (VIZ_T); none / '-' = the whole pour")
            s.add_argument("--size", default="1280x720")
            s.add_argument("--ss", type=int, default=2)
            s.add_argument("--cup", default="dark")
            s.add_argument("--keep-render", action="store_true",
                           help="do not compare size / ss / cup (CPU fallback: keep the GPU renders)")
            s.add_argument("--force", action="store_true", help="plan every job (VIZ_FORCE=1)")
            s.add_argument("--held-record", action="store_true",
                           help="held-out pours also for record closures (VIZ_HELD_RECORD=1)")
    s = sub.add_parser("start")
    s.add_argument("run_dir")
    s.add_argument("model")
    s.add_argument("--root", required=True)
    s.add_argument("--launcher", default=None)
    s = sub.add_parser("fields-ok")
    s.add_argument("npz")
    s.add_argument("sha")
    s.add_argument("inlet")
    s.add_argument("--N", default=None)
    s.add_argument("--fps", default=None)
    s.add_argument("--T", default="any", help="simulated length; '-' = the whole pour; 'any' (default) = not checked")
    s = sub.add_parser("ismi")
    s.add_argument("closure")
    s.add_argument("--inlet", default="auto")
    s = sub.add_parser("mark")
    s.add_argument("vd")
    s.add_argument("pour")
    s.add_argument("kv", nargs="*")
    s = sub.add_parser("summary")
    s.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan(a)
    elif a.cmd == "migrate":
        migrate(a)
    elif a.cmd == "start":
        print(json.dumps(resolve_start(a.run_dir, a.model, os.path.abspath(a.root), a.launcher), ensure_ascii=False))
    elif a.cmd == "fields-ok":
        why = fields_check(a.npz, a.sha, a.inlet, a.N, a.fps, a.T)
        if why:
            print(why)
        sys.exit(1 if why else 0)
    elif a.cmd == "ismi":
        print(1 if is_mi(a.closure, a.inlet) else 0)
    elif a.cmd == "mark":
        mark(a)
    elif a.cmd == "summary":
        summary(a)


if __name__ == "__main__":
    main()
