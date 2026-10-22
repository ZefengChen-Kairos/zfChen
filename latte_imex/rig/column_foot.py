"""Validated milk-column foot: where the centre line of the milk column crosses the registered liquid surface, per
cup_track.json frame (tulip, leaf), combined from four independent measurements and a blind visual adjudication.  An
input measurement (the hand's action), never corrected from the white pattern.  Written in the stream_foot.json format
as raw_data/rig/results/<p>/stream_foot_col.json, which the E0 replay (schedule.e0_world_offsets) reads when the
environment LATTE_FOOT is 'col' (real_program.stream_foot_file; the default 'v1' keeps stream_foot.json).

Quantity.  x_col(t): the horizontal point where the column's centre line meets the registered plane z_s(t)
(disc.json, cup_register.py).  U, V: board x, y in mm about the registered disc centre (V < 0 toward the pitcher in the
deposit windows); u, v = (U, V) / L_m; solver x = U / L, y = -V / L with L = L_m / 0.49.  Clock: t = cup_track take time
- real_program.cup_poses t0 (as stream_foot.json).

Measurements (col25 campaign; their per-frame outputs are frozen in raw_data/rig/results/<p>/column_foot_meas.json, the
per-method pipelines themselves are not part of the package):
  A  colour silhouettes, anchor-free: a straight 3-D line through the silhouette midpoints of the column in the three
     calibrated cameras, 2-12 mm above the plane (blurred-box edge fit with a shading term; bright-band fallback with a
     per-camera band-to-silhouette offset), evaluated at h = 0; depth veto (the line must stand >= 3 mm above the local
     surface in cameras 1 and 2).  sigma: grade A max(s, 1.5), grade B max(1.5 s, 3); leaf grade B after 4 s and leaf
     fi 98 (a declared ghost) dropped.
  B  depth only: column pixels 3-30 mm above each camera's own surface, clear of the z-buffered pitcher; camera 1 gives
     V, camera 2 gives U (lateral position of the pixel rays is exact whatever the range); robust line to h = 0.
     sigma: max(sigma_v, 3) when the V fit starts <= 4.5 mm above the plane; U max(sigma_u, 3); leaf fi 32, tulip fi 23
     (declared misses) dropped.
  C  blind three-camera arc readings of the visible column along the model heading (V only), sigma 2.  (Method C's
     ballistic landing point is a model and never enters.)
  D  the stream_foot.py tracker re-run from four anchors (v1 tip, v2 tip, v2 tip + predicted stream, a wide window),
     with a blind visual choice between the candidate feet; sigma max(s, 2) when chosen by eye, max(s, 3) otherwise.
  Per frame and axis: Huber IRLS (c = 2 sigma) from the weighted median, sigma = sqrt(1 / sum w + 1^2), inflated by
  sqrt(chi^2 / dof) when > 1.  Flagged frames (two measurements > 5 mm apart in V or U), a random 10 % and every frame
  of the core windows (leaf 0.7-4.3 s, tulip 0.6-1.9 and 3.4-4.3 s) were read blind on native crops of all three
  cameras with every candidate shown as a permuted letter (no record, no model drawn); the reading SELECTS: methods
  within KEEP = 5 mm of it along V are combined, the others rejected for that frame; the reading itself enters the value
  only when the survivors are weak (combined sigma > 1.5 x the reading's sigma), and alone when nothing survives.
  A common systematic SYS = 1 mm is added in quadrature.  Grades: A = read with hi / med confidence and sigma_V <= 3,
  or unread with >= 2 methods and sigma_V <= 2.5; B = other read frames or sigma_V <= 4; C = rest; ok = A or B.
  Single-method frames of the tulip tail (> 7 s) carry SIG_TAIL = 3 mm more (a random check found them 2-4 mm behind
  the thin stream), which makes them C.
  E0 tiers in stream_foot_col.json (the tracker fields are neutral values that place a row in schedule's E0 tiers only
  when it is validated): candidate = grade A or sigma_V <= 3 (quality 0.65 + 0.35 clip((3 - sigma_V) / 2, 0, 1),
  trusted); plausible = other ok rows (quality 0.40); none = C / no foot.  tip_board is the production anchor
  (stream_foot.json's v1 tip, else cup_track.json's), so the E0_BEHIND gate acts as in production.

Validation (col25; record = the production E0 'world' record from stream_foot.json, N 96, pose v2, exit_cal 0;
record - column along V, median and 95 % frame bootstrap, + = column behind the record, toward the pitcher):
  leaf L1 0.7-2.2 s +1.5 [+0.8, +2.0] (23 frames); L2 2.2-4.3 s -0.4 [-1.4, +0.5]; tulip P1 0.6-1.9 s +1.8 [+1.1, +2.4];
  P2 3.4-4.3 s -2.5 [-4.2, +0.4]; P3 +0.3; tail -0.7.  The record steps 4-6.5 mm behind the column at leaf 3.87-4.05 s
  (one tracker foot; the column does not move).  Sources in leaf L1 (record - source): A +1.3, B -0.1, C +3.3,
  D -0.6, the adjudicator's blind reading +2.4; none near the 9-13 mm 'rear foot' hypothesis (that is where the
  v2-anchored tracker feet, stream_foot_v2tip.json, and the default-exit ballistic put the milk: 12-17 mm behind).
  The E0 replay of these feet reproduces the column to a median 0.0-0.1 mm per window (in-sample by construction).
  Independent review check (blind pixel readings by a second observer, 21 frames, V from cameras 0 / 1 by ray-plane
  intersection): in the core windows (L1, L2, P1, P2; 13 frames) |reading - record| along V p50 is 4.1 mm for the
  current record and 1.0 mm for the record rebuilt from these feet; at the leaf spike frames (fi 61, 62) 4.4 / 6.0 mm
  against 0.1 / 0.3 mm.

Limits.
  Systematic, asymmetric: eye readings of the visible contact (three observers) sit 1-3 mm behind (toward the pitcher)
  the machine centre-line values in L1 / P1 (review: -2.5 mm against the rebuilt record, L1 + P1, 7 frames); leaving
  the tracker family D out moves the L1 / P1 value 0.4 / 1.0 mm back.  The quantity here is the centre line at the
  registered plane; a contact read on a raised white mound lies behind it by ~0.6 mm per mm of mound height plus up to
  the column's radius.  Plane height: ~0.6-0.8 mm of V per mm.
  Coverage: leaf after 4.3 s rests mostly on single-method D rows (28, plausible tier) -- the review found 2 of 2
  checked off by 7-8.5 mm; tulip P3 (column into thick white): the review's eye reading sits ~5 mm behind on 2 of 3
  frames.  Treat both as unvalidated.  U is determined less well than V; 5 leaf and 4 tulip frames take U from the
  record (no lateral measurement; outside the core windows).  Camera sync +-40 ms.  Only tulip and leaf.

Files: raw_data/rig/results/<p>/column_foot_meas.json (input), raw_data/rig/results/<p>/stream_foot_col.json (output).
    python3 -m latte_imex.rig.column_foot build [p ...]                  stream_foot_col.json from the frozen table
    python3 -m latte_imex.rig.column_foot check [p ...]                  record - column per window for the record of
                                                                         the current LATTE_FOOT (pose v2, e0 world)
    python3 -m latte_imex.rig.column_foot sheet <p> <out.jpg> fi ...     the foot in all cameras (needs the pours data,
                                                                         as stream_foot.py)"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "raw_data", "rig", "results")
POURS = ("tulip", "leaf")
WINDOWS = {"tulip": [("P1", 0.6, 1.9), ("P2", 3.4, 4.3), ("P3", 5.26, 7.0), ("tail", 7.0, 99.0)],
           "leaf": [("L1", 0.7, 2.2), ("L2", 2.2, 4.3), ("late", 4.3, 99.0)]}
FLOOR = 1.0        # [mm] per-axis floor of a combination
HUB = 2.0          # Huber constant [sigma]
KEEP = 5.0         # [mm] a method survives a visual reading when within KEEP along V
SYS = 1.0          # [mm] common systematic (plane height, ray calibration)
SIG_TAIL = 3.0     # [mm] extra sigma of single-method frames in the tulip tail
TAIL_INFLATE = {"tulip": ("tail",)}
NAN = float("nan")


def meas_file(p, res=RES):
    return os.path.join(res, p, "column_foot_meas.json")


def out_file(p, res=RES):
    return os.path.join(res, p, "stream_foot_col.json")


def _fin(x):
    return x is not None and np.isfinite(x)


def combine(ms):
    """ms: list of (value, sigma) -> (value, sigma): Huber IRLS from the weighted median, 1 mm floor, chi^2 inflation."""
    v = np.array([m[0] for m in ms], float)
    s = np.array([m[1] for m in ms], float)
    w0 = 1.0 / s ** 2
    o = np.argsort(v)
    cw = np.cumsum(w0[o])
    m = v[o][np.searchsorted(cw, 0.5 * cw[-1])]
    for _ in range(20):
        r = (v - m) / s
        w = w0 * np.where(np.abs(r) <= HUB, 1.0, HUB / np.maximum(np.abs(r), 1e-9))
        m2 = float((w * v).sum() / w.sum())
        if abs(m2 - m) < 1e-4:
            m = m2
            break
        m = m2
    sig = 1.0 / np.sqrt(w.sum())
    if len(v) > 1:
        chi = float((w * (v - m) ** 2).sum() / (len(v) - 1))
        if chi > 1:
            sig *= np.sqrt(chi)
    return float(m), float(np.hypot(sig, FLOOR))


def _combined(meas):
    out = {}
    for ax in ("V", "U"):
        ms = [(m[ax], m["s" + ax]) for m in meas.values() if ax in m and _fin(m[ax])]
        if ms:
            out[ax], out["s" + ax] = combine(ms)
            out["n" + ax] = len(ms)
    return out


def decide(r):
    """The validated value of one frame (see the module docstring)."""
    m, vis = r["meas"], r.get("visual")
    out = dict(how=None, used=[], rejected=[])
    if vis is not None:
        surv = [k for k, x in m.items() if "V" in x and _fin(x["V"]) and abs(x["V"] - vis["V"]) <= KEEP]
        out["rejected"] = [k for k, x in m.items() if "V" in x and _fin(x["V"]) and k not in surv]
        if surv:
            ms = [(m[k]["V"], m[k]["sV"]) for k in surv]
            V, sV = combine(ms)
            if sV > 1.5 * vis["sV"]:
                V, sV = combine(ms + [(vis["V"], vis["sV"])])
                out["visual_in_value"] = True
            us = [(m[k]["U"], m[k]["sU"]) for k in surv if "U" in m[k] and _fin(m[k]["U"])]
            U, sU = combine(us) if us else (vis["U"], vis["sU"])
            out.update(V=V, sV=sV, U=U, sU=sU, how=f"visual-select ({vis['kind']})", used=sorted(surv))
        else:
            out.update(V=vis["V"], sV=vis["sV"], U=vis["U"], sU=vis["sU"], how=f"visual-only ({vis['kind']})")
        out["conf"] = vis["conf"]
    else:
        c = _combined(m)
        if c.get("nV", 0) == 0:
            out["how"] = "none"
            return out
        us = [(x["U"], x["sU"]) for x in m.values() if "U" in x and _fin(x["U"])]
        U, sU = combine(us) if us else (NAN, NAN)
        out.update(V=c["V"], sV=c["sV"], U=U, sU=sU, used=sorted(k for k, x in m.items() if "V" in x),
                   how="methods-agree" if c["nV"] >= 2 else "single-method")
    out["sV"] = float(np.hypot(out["sV"], SYS))
    out["sU"] = float(np.hypot(out["sU"], SYS)) if _fin(out["sU"]) else 3.0
    return out


def grade(d):
    if d["how"] == "none":
        return "none"
    if d["how"].startswith("visual-select") and d["conf"] in ("hi", "med") and d["sV"] <= 3.0:
        return "A"
    if d["how"] == "methods-agree" and d["sV"] <= 2.5:
        return "A"
    if d["how"].startswith("visual") or d["sV"] <= 4.0:
        return "B"
    return "C"


def validated(p, res=RES):
    """Per-frame validated foot of pour p from the frozen table: list of dicts (fi, t, window, quality, ok, how, U_mm,
    V_mm, sigma_V_mm, sigma_U_mm, sigma_mm, u, v, x, y, foot_board, methods_used, methods_rejected)."""
    T = json.load(open(meas_file(p, res)))
    Lmm = T["meta"]["L_m_mm"]
    L = Lmm / 0.49
    frames = []
    for r in T["frames"]:
        d = decide(r)
        if d["how"] == "single-method" and r["window"] in TAIL_INFLATE.get(p, ()):
            d["sV"] = float(np.hypot(d["sV"], SIG_TAIL))
            d["sU"] = float(np.hypot(d["sU"], SIG_TAIL))
        g = grade(d)
        f = dict(fi=r["fi"], t=r["t"], window=r["window"], quality=g, ok=g in ("A", "B"), how=d["how"],
                 methods_used=d["used"], methods_rejected=d["rejected"])
        if d["how"] != "none":
            U_, V_ = d["U"], d["V"]
            if not _fin(U_):                       # V-only measurement and no U anywhere: lateral from the record
                U_ = r["record_U_mm"] if _fin(r["record_U_mm"]) else 0.0
                d["sU"] = 5.0
                f["U_from"] = "record (no lateral measurement)"
            cx, cy = r["disc_c"]
            f.update(U_mm=U_, V_mm=V_, u=U_ / Lmm, v=V_ / Lmm, x=U_ * 1e-3 / (L * 1e-3), y=-V_ * 1e-3 / (L * 1e-3),
                     sigma_V_mm=d["sV"], sigma_U_mm=d["sU"], sigma_mm=float(np.sqrt(0.5 * (d["sV"] ** 2 + d["sU"] ** 2))),
                     foot_board=[cx + U_ * 1e-3, cy + V_ * 1e-3, r["zs"]])
        else:
            f.update(U_mm=None, V_mm=None, u=None, v=None, x=None, y=None, sigma_mm=None, foot_board=None)
        frames.append(f)
    return frames, Lmm


def build_file(p, res=RES):
    """stream_foot.json-format rows of the validated feet (E0 tiers as in the module docstring)."""
    F, _ = validated(p, res)
    T = {r["fi"]: r for r in json.load(open(meas_file(p, res)))["frames"]}
    S1 = {r["fi"]: r for r in json.load(open(os.path.join(res, p, "stream_foot.json")))["frames"]}
    rows, tiers = [], {"cand": 0, "plaus": 0, "none": 0}
    for f in F:
        fi = f["fi"]
        s, r = S1.get(fi), T[fi]
        tip = s["tip_board"] if s else r["tip_board_fallback"]
        row = dict(fi=fi, t=f["t"], surf_z=r["zs"], tip_board=[float(x) for x in tip],
                   model_active=bool(s["model_active"]) if s else False)
        has = f["foot_board"] is not None
        if has:
            row.update(foot_board=f["foot_board"], u=f["u"], v=f["v"], x=f["x"], y=f["y"])
        else:
            row.update(foot_board=None, u=None, v=None, x=None, y=None)
        sV = f.get("sigma_V_mm") or 99.0
        if has and f["ok"] and (f["quality"] == "A" or sV <= 3.0):
            row.update(quality=0.65 + 0.35 * float(np.clip((3.0 - sV) / 2.0, 0, 1)), rms_mm=0.5, slope=0.5, top_mm=6.0,
                       trusted=True)
            tiers["cand"] += 1
        elif has and f["ok"]:
            row.update(quality=0.40, rms_mm=2.0, slope=0.5, top_mm=6.0, trusted=False)
            tiers["plaus"] += 1
        else:
            row.update(quality=0.0, rms_mm=99.0, slope=9.9, top_mm=99.0, trusted=False)
            tiers["none"] += 1
        row.update(validated_grade=f["quality"], validated_how=f["how"], sigma_mm=f.get("sigma_mm"))
        if s and "along" in s and has:
            # jet direction of the stream_foot.json row: along + i across = (dx + i dy) conj(e)
            dx0, dy0 = s["x"] - s["model_x"], s["y"] - s["model_y"]
            z = complex(s["along"], s["across"]) / complex(dx0, dy0) if abs(complex(dx0, dy0)) > 1e-9 else None
            if z is not None:
                e = z.conjugate() / abs(z)
                w = complex(f["x"] - s["model_x"], f["y"] - s["model_y"]) * e.conjugate()
                row.update(model_x=s["model_x"], model_y=s["model_y"], along=w.real, across=w.imag, dist=abs(w))
        rows.append(row)
    return dict(pattern=p, cup_radius_solver=0.49, Q_OK=0.35,
                source="latte_imex/rig/column_foot.py: validated milk-column foot (col25 methods A-D + blind adjudication)",
                tip_board="stream_foot.json v1 tip (production anchor)", tiers=tiers, frames=rows)


def check(p, N=96, res=RES, log=print):
    """Record (pour_script(p, N, e0 'world'), whatever LATTE_FOOT selects) - validated column along V per window, over
    the ok frames: median and 95 % frame bootstrap [mm]."""
    from ..diff import schedule as SC
    from .real_program import stream_foot_file
    F, Lmm = validated(p, res)
    L = Lmm / 0.49
    script, _ = SC.pour_script(p, N, e0="world", exit_cal=False)
    S = SC.build(script, N)
    fdt = np.asarray(S["fdt"], float)
    tm = np.cumsum(fdt) - 0.5 * fdt
    act = np.asarray(S["active"]).astype(bool)
    out = dict(pour=p, N=N, foot_file=stream_foot_file(p), windows={})
    rng = np.random.default_rng(1)
    for w, a, b in WINDOWS[p]:
        dV = []
        for f in F:
            if not f["ok"] or f["window"] != w or not act[int(np.argmin(np.abs(tm - f["t"])))]:
                continue
            dV.append(-float(np.interp(f["t"], tm, S["y"])) * L - f["V_mm"])
        if dV:
            dV = np.array(dV)
            bs = np.median(rng.choice(dV, (4000, len(dV))), 1)
            out["windows"][w] = dict(n=len(dV), median=float(np.median(dV)),
                                     ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
            log(f"{p} {w:5} n {len(dV):3d}: record - column V {np.median(dV):+.2f} [{np.percentile(bs, 2.5):+.2f}, "
                f"{np.percentile(bs, 97.5):+.2f}] mm  ({os.path.basename(stream_foot_file(p))})")
    return out


def sheet(p, out, fis, res=RES, half=40, sc=4):
    """The validated foot (green +, 1 sigma_V ring) and every method's foot (letters A-D) in the three cameras."""
    import cv2
    from .stream_foot import Rig, SCR
    F = {f["fi"]: f for f in validated(p, res)[0]}
    T = {r["fi"]: r for r in json.load(open(meas_file(p, res)))["frames"]}
    ct = json.load(open(os.path.join(SCR, p, "cup_track.json")))
    rig = Rig(p)
    rows = []
    for fi in fis:
        f, r = F[fi], T[fi]
        if f["foot_board"] is None:
            continue
        cx, cy = r["disc_c"]
        zs = r["zs"]
        i = int(np.argmin(np.abs(rig.tk.T - ct[fi]["t"])))
        tiles = []
        for c, k, _ in rig.tk.sync(i):
            im = rig.tk.color(c, k)

            def tp(U, V):
                a, b, _ = rig.proj(c, np.array([[cx + U * 1e-3, cy + V * 1e-3, zs]]))
                return float(a[0]), float(b[0])
            a0, b0 = tp(f["U_mm"], f["V_mm"])
            x0, y0 = int(round(a0)) - half, int(round(b0)) - half - 8
            pad = cv2.copyMakeBorder(im, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=0)
            big = cv2.resize(pad[y0 + 200:y0 + 200 + 2 * half, x0 + 200:x0 + 200 + 2 * half], (2 * half * sc, 2 * half * sc),
                             interpolation=cv2.INTER_CUBIC)

            def q(U, V):
                a, b = tp(U, V)
                return int((a - x0 + 0.5) * sc - 0.5), int((b - y0 + 0.5) * sc - 0.5)
            for key, m in r["meas"].items():
                if _fin(m.get("V")):
                    cv2.putText(big, key, q(m["U"] if _fin(m.get("U")) else f["U_mm"], m["V"]), 0, 0.4, (0, 255, 255), 1)
            s = f["sigma_V_mm"]
            ring = np.array([q(f["U_mm"] + s * np.cos(a), f["V_mm"] + s * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 40)])
            cv2.polylines(big, [ring.astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.drawMarker(big, q(f["U_mm"], f["V_mm"]), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(big, f"{p} fi{fi} t{f['t']:.2f} c{c} {f['quality']}", (3, 13), 0, 0.4, (255, 255, 255), 1)
            tiles += [big, np.full((big.shape[0], 3, 3), 60, np.uint8)]
        rows += [np.hstack(tiles), np.full((3, sum(t.shape[1] for t in tiles), 3), 90, np.uint8)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a.pop(0) if a else "build"
    if cmd == "build":
        for p in a or POURS:
            o = build_file(p)
            json.dump(o, open(out_file(p), "w"), indent=0)
            print(p, "->", out_file(p), o["tiers"])
    elif cmd == "check":
        for p in a or POURS:
            check(p)
    elif cmd == "sheet":
        print(sheet(a[0], a[1], [int(x) for x in a[2:]]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
