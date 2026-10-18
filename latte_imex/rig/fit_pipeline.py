"""Full pipeline on the GPU box: fit the V0.5 closure to the real pours driven by the barista's measured hand actions
(pitcher poses -> catalogue pitcher -> inlet), then simulate every pattern with the fitted closure and render
images and movies (real | simulation | hand action).

Inlet (per pattern, built once): measured pitcher poses (tip path, tilt, yaw, roll) through the Kona 700 mL CAD
(pitcher_kona700.json, spout tip TIP_KONA), fill solved so that the whole pour delivers the 165 g that reached the cup
(full_pour_kona_poured165.json) -- the same inlet as tulip_feas.Pour.
Objective (per pattern, tulip_feas.Target.score): J = white IoU + gap IoU; the fit maximises the mean J over the
calibration patterns (--cal, default tulip,heart); the others are predictions.
Stages (resumable; records appended to OUT/records.jsonl, fields to OUT/fields/):
  A  scrambled Sobol sample of v05_adapter.SPACE (--n) + the S1 closure, every calibration pattern, at --N
  B  --rounds rounds of local refinement: --per gaussian samples around each of the --top best closures,
     sigma halved each round (starts at 0.10 in the unit box)
  C  the --confirm best closures and S1 on ALL patterns at --N-fig (calibration and prediction scores)
  R  render: latte_imex.run with video for every pattern with the best closure (CPU, one process per pattern),
     then OUT/render/<p>.mp4 (real | sim | action) and OUT/summary.png
    python -m latte_imex.rig.fit_pipeline fit    --out results/fit [--cal tulip,heart --n 512 --N 128]
    python -m latte_imex.rig.fit_pipeline render --out results/fit [--jobs 5]
    python -m latte_imex.rig.fit_pipeline report --out results/fit
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from . import real_program as RP
from .tulip_feas import KEYS, Pour, Target, from_unit, to_unit, _rot
from ..v05_adapter import SPACE

PATTERNS = RP.PATTERNS
NAMES = dict(heart="heart", push_heart="push heart", leaf="leaf", swan="swan", tulip="tulip")


def _load(out):
    fn = os.path.join(out, "records.jsonl")
    if not os.path.exists(fn):
        return []
    with open(fn) as f:
        return [json.loads(l) for l in f if l.strip()]


def _agg(recs, ids, pats, N):
    """closure id -> mean J over pats (only ids with every pattern done and ok)."""
    by = {}
    for r in recs:
        if r["N"] == N and r["cid"] in ids and r["pattern"] in pats:
            by.setdefault(r["cid"], {})[r["pattern"]] = r
    out = {}
    for cid, d in by.items():
        if all(p in d for p in pats):
            out[cid] = dict(cid=cid, params=d[pats[0]]["params"], J=float(np.mean([d[p]["J"] for p in pats])),
                            per={p: dict(iou=d[p]["iou"], gap_iou=d[p]["gap_iou"], J=d[p]["J"]) for p in pats})
    return out


class Fitter:
    def __init__(self, a, log=print):
        self.a, self.log = a, log
        self.cal = a.cal.split(",")
        self.pours, self.tgts, self.eng = {}, {}, {}
        os.makedirs(os.path.join(a.out, "fields"), exist_ok=True)
        self.done = {(r["cid"], r["pattern"], r["N"]) for r in _load(a.out)}

    def pour(self, p):
        if p not in self.pours:
            self.pours[p], self.tgts[p] = Pour(p), Target(p)
            i = self.pours[p].info
            self.log(f"  inlet {p}: V0 {i['V0_mL']:.0f} mL, poured {i['poured_mL']:.0f} mL, T {self.pours[p].T:.1f} s")
        return self.pours[p]

    def full(self, par):
        m = self.pour(self.cal[0]).meta["config"]["parameters"]
        return dict(par, nu=float(m["nu"]), D=float(m["D"]))

    def engine(self, N):
        from ..gpu_batch import BatchEngine
        if N not in self.eng:
            self.eng[N] = BatchEngine(N, backend=self.a.backend)
        return self.eng[N]

    def go(self, jobs, pats, N, label):
        """jobs: [(cid, stage, closure)] x pats -> records (resumable, batched across patterns)."""
        todo = [(cid, st, cl, p) for cid, st, cl in jobs for p in pats if (cid, p, N) not in self.done]
        if not todo:
            return
        self.log(f"stage {label}: {len(todo)} runs at N={N}")
        B = self.a.batch
        for k in range(0, len(todo), B):
            part = todo[k:k + B]
            mem = [self.pour(p).member(cl, N, tag=i) for i, (cid, st, cl, p) in enumerate(part)]
            t0 = time.time()
            self.engine(N).run(mem)
            wall = time.time() - t0
            rows = []
            for (cid, st, cl, p), m in zip(part, mem):
                r = m.result
                rec = dict(cid=cid, pattern=p, stage=st, N=N, params=cl, ok=bool(r["ok"]), error=r["error"],
                           wall_s=wall / len(part))
                if r["ok"]:
                    M, _, _ = self.tgts[p].score(r["c"])
                    rec.update(M)
                    np.save(os.path.join(self.a.out, "fields", f"{cid}_{p}_N{N}.npy"),
                            np.nan_to_num(r["c"]).astype(np.float16))
                else:
                    rec.update(iou=0.0, gap_iou=0.0, J=0.0)
                rows.append(rec)
                self.done.add((cid, p, N))
            with open(os.path.join(self.a.out, "records.jsonl"), "a") as f:
                for rec in rows:
                    f.write(json.dumps(rec, default=float) + "\n")
            ok = [r for r in rows if r["ok"]]
            self.log(f"  {k + len(part)}/{len(todo)}: batch {len(part)} in {wall:.0f} s, {len(ok)} ok, "
                     f"best J {max([r['J'] for r in ok], default=0):.3f}")


def fit(a, log=print):
    os.makedirs(a.out, exist_ok=True)
    F = Fitter(a, log)
    cal = F.cal
    log(f"calibration patterns: {cal}; predictions: {[p for p in PATTERNS if p not in cal]}")
    s1 = F.pour(cal[0]).s1
    json.dump(dict(vars(a), space={k: list(v) for k, v in SPACE.items()}, s1=s1),
              open(os.path.join(a.out, "config.json"), "w"), indent=1, default=float)
    from scipy.stats import qmc
    U = qmc.Sobol(len(KEYS), scramble=True, seed=a.seed).random(a.n)
    jobs = [("S1", "A", F.full(s1))] + [(f"A{i:04d}", "A", F.full(from_unit(u))) for i, u in enumerate(U)]
    F.go(jobs, cal, a.N, "A (Sobol)")
    rng = np.random.default_rng(a.seed + 1)
    sig = 0.10
    for rd in range(a.rounds):
        ids = {r["cid"] for r in _load(a.out) if r["stage"] in ("A", "B")}
        top = sorted(_agg(_load(a.out), ids, cal, a.N).values(), key=lambda r: -r["J"])[:a.top]
        log(f"round {rd}: best mean J {top[0]['J']:.3f} ({top[0]['cid']}), sigma {sig:.3f}")
        jobs = []
        for k, r in enumerate(top):
            u0 = to_unit(r["params"])
            for j in range(a.per):
                jobs.append((f"B{rd}_{k:02d}_{j:02d}", "B", F.full(from_unit(u0 + rng.normal(0, sig, len(KEYS))))))
        F.go(jobs, cal, a.N, f"B{rd} (refine)")
        sig *= 0.5
    ids = {r["cid"] for r in _load(a.out) if r["stage"] in ("A", "B")}
    top = sorted(_agg(_load(a.out), ids, cal, a.N).values(), key=lambda r: -r["J"])
    pick = [r for r in top if r["cid"] != "S1"][:a.confirm] + [r for r in top if r["cid"] == "S1"]
    F.go([(f"C_{r['cid']}", "C", r["params"]) for r in pick], list(PATTERNS), a.N_fig, "C (all patterns)")
    return report(a, log)


def report(a, log=print):
    recs = [r for r in _load(a.out) if r["ok"]]
    cal = a.cal.split(",")
    pred = [p for p in PATTERNS if p not in cal]
    C = {r["cid"] for r in recs if r["stage"] == "C"}
    agg = _agg(recs, C, list(PATTERNS), a.N_fig)
    rows = []
    for cid, d in agg.items():
        d["J_cal"] = float(np.mean([d["per"][p]["J"] for p in cal]))
        d["J_pred"] = float(np.mean([d["per"][p]["J"] for p in pred])) if pred else None
        rows.append(d)
    rows.sort(key=lambda d: -d["J_cal"])
    best = next((d for d in rows if d["cid"] != "C_S1"), None)
    s1 = agg.get("C_S1")
    summ = dict(cal=cal, pred=pred, best=best, s1=s1, confirmed=rows, n_runs=len(recs),
                n_failed=sum(1 for r in _load(a.out) if not r["ok"]))
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    if best:
        json.dump({k: best["params"][k] for k in best["params"]}, open(os.path.join(a.out, "best_closure.json"), "w"),
                  indent=1)
    log(f"{len(recs)} runs ({summ['n_failed']} failed); calibration {cal}, prediction {pred}")
    for d in rows[:6] + ([s1] if s1 and s1 not in rows[:6] else []):
        log(f"  {d['cid']:14s} J_cal {d['J_cal']:.3f} J_pred {d['J_pred'] if d['J_pred'] is None else round(d['J_pred'], 3)}  " +
            "  ".join(f"{p} {d['per'][p]['iou']:.2f}/{d['per'][p]['gap_iou']:.2f}" for p in PATTERNS))
    if best:
        log("  best closure: " + " ".join(f"{k} {best['params'][k]:.4g}" for k in KEYS))
    if rows:
        sheet(a, rows, cal)
    return summ


def sheet(a, rows, cal):
    """Rows: target, S1, best; columns: patterns (barista view, gaps cyan)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("milk", [(0.36, 0.20, 0.09), (0.66, 0.47, 0.30), (1.0, 0.97, 0.92)])
    cm.set_bad((0.12, 0.12, 0.12))
    best = next(d for d in rows if d["cid"] != "C_S1")
    show = [("target", None)] + [("S1 closure", d) for d in rows if d["cid"] == "C_S1"] + [("fitted closure", best)]
    fig, ax = plt.subplots(len(show), len(PATTERNS), figsize=(3.2 * len(PATTERNS), 3.4 * len(show)), dpi=100, squeeze=False)
    for j, p in enumerate(PATTERNS):
        T = Target(p)
        for i, (lab, d) in enumerate(show):
            x = ax[i, j]
            x.axis("off")
            if d is None:
                x.imshow(_rot(T.img[..., ::-1].copy(), T.rot))
                x.contour(_rot(T.G.astype(np.float32), T.rot), [0.5], colors="cyan", linewidths=0.7)
                x.set_title(f"{NAMES[p]}{' (calibration)' if p in cal else ' (prediction)'}", fontsize=9)
                continue
            M, cs, G = T.score(np.load(os.path.join(a.out, "fields", f"{d['cid']}_{p}_N{a.N_fig}.npy")).astype(float))
            msk = _rot(np.isfinite(cs).astype(np.float32), T.rot) > 0.5
            x.imshow(np.ma.masked_array(_rot(np.nan_to_num(cs).astype(np.float32), T.rot), ~msk), cmap=cm, vmin=0, vmax=0.8)
            x.contour(_rot(G.astype(np.float32), T.rot), [0.5], colors="cyan", linewidths=0.7)
            x.set_title(f"{lab}: IoU {M['iou']:.2f}  gap {M['gap_iou']:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "sheet.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------- render
def _solve_one(args):
    p, out, closure, N = args
    os.environ["PITCHER_GEOM"] = os.path.join(RP.REPO, "latte_imex", "pitcher_geom", "pitcher_kona700.json")
    from .full_pour import TIP_KONA
    RP.TIP_OVERRIDE = TIP_KONA
    RP.SIM_SUB = "sim_fit"
    RP._FP.clear()
    fill = json.load(open(os.path.join(RP.RES, "full_pour_kona_poured165.json")))[p]["fill_at_cut_mL"]
    od, info = RP.solve(p, N, "pitcher", video=True, V0=fill * 1e-6, log=lambda *x: None, closure=closure)
    return p, od, info


def _pitcher_outline():
    """Side silhouette of the Kona CAD (pitcher frame, mm): spout side (phi 0) and handle side (phi 180), with the
    dipped rim; origin at the inside bottom on the axis."""
    G = json.load(open(os.path.join(RP.REPO, "latte_imex", "pitcher_geom", "pitcher_kona700.json")))
    u = 1.0 if G.get("units") == "mm" else 1e3
    zs = np.asarray(G["zs"], float) * u
    phis = np.asarray(G["phis"], float)
    ro = np.asarray(G["r_out"], float) * u
    i0 = int(np.argmin(np.abs(np.angle(np.exp(1j * phis)))))
    i1 = int(np.argmin(np.abs(np.angle(np.exp(1j * (phis - np.pi))))))
    zr = np.asarray(G["z_rim"], float) * u if "z_rim" in G else np.full(len(phis), zs[-1])
    k0, k1 = zs <= zr[i0] + 1e-6, zs <= zr[i1] + 1e-6
    xs = np.r_[ro[k0, i0], -ro[k1, i1][::-1]]
    zz = np.r_[zs[k0], zs[k1][::-1]]
    return np.c_[xs, zz]


def action_panel(I, t, W, H, outline, tip_kona_mm):
    """Hand action at time t: pitcher side view (CAD silhouette at the measured tilt and tip height above the cup)
    and the traces (pitcher axis tilt, flow, fill) with a cursor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tt = I["t"]
    k = int(np.clip(np.searchsorted(tt, t), 0, len(tt) - 1))
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    a1 = fig.add_axes([0.08, 0.42, 0.88, 0.54])
    L = float(I["L"]) * 1e3
    yaw = float(I["yaw"][k])
    tip = I["tip"][k] * 1e3
    s_tip = tip[0] * math.cos(yaw) + tip[1] * math.sin(yaw)     # tip position along the pouring direction
    th = math.radians(float(I["axis_tilt_deg"][k]))
    P = outline - np.array([tip_kona_mm[1], tip_kona_mm[2]])     # spout tip at the origin
    R = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
    P = P @ R.T + np.array([s_tip, tip[2]])
    a1.fill(P[:, 0], P[:, 1], color="0.85", ec="0.3", lw=1)
    a1.plot([-L / 2, -L / 2, L / 2, L / 2], [20, 0, 0, 20], color="saddlebrown", lw=2)
    on = I["Q_o"][k] >= RP.Q_ON
    if on:
        xh = I["x_hit"][k] * 1e3
        a1.plot([s_tip, xh[0] * math.cos(yaw) + xh[1] * math.sin(yaw)], [tip[2], 0], color="ivory", lw=3)
        a1.plot([s_tip, xh[0] * math.cos(yaw) + xh[1] * math.sin(yaw)], [tip[2], 0], color="0.6", lw=0.6)
    a1.plot([s_tip], [tip[2]], "o", color="orange", ms=4)
    a1.set_xlim(-170, 110); a1.set_ylim(-15, 200); a1.set_aspect("equal")
    a1.set_facecolor((0.12, 0.12, 0.14)); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title(f"hand action: tilt {np.degrees(th):4.0f} deg, tip {tip[2]:4.0f} mm above the surface, "
                 f"{'pouring' if on else 'off'}", fontsize=8)
    a2 = fig.add_axes([0.08, 0.07, 0.88, 0.30])
    a2.plot(tt, I["axis_tilt_deg"], "r-", lw=0.8, label="tilt [deg]")
    a2.plot(tt, I["Q_o"] * 1e6, "b-", lw=0.8, label="flow [mL/s]")
    a2.plot(tt, I["V_P"] * 1e6 / 3, "g-", lw=0.8, label="fill/3 [mL]")
    a2.axvline(t, color="k", lw=0.8)
    a2.set_ylim(0, 110); a2.set_xlim(tt[0], tt[-1]); a2.tick_params(labelsize=6); a2.grid(alpha=0.3)
    a2.legend(fontsize=6, ncol=3, loc="upper left")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def movie(p, od, out_mp4, fps=24):
    import cv2
    import imageio
    S, EXT = 480, 1.05
    rot = json.load(open(os.path.join(RP.TGT, "view_rotation.json")))[p]
    M = cv2.getRotationMatrix2D((S / 2 - 0.5, S / 2 - 0.5), rot, 1.0)
    s = np.linspace(-EXT, EXT, S)
    U, V = np.meshgrid(s, s)
    outside = U ** 2 + V ** 2 > 1.0
    rr = imageio.get_reader(os.path.join(RP.REPO, "raw_data", "rig", "real_panel", f"{p}.mp4"))
    n_real = rr.count_frames()
    cap = cv2.VideoCapture(os.path.join(od, "movie.mp4"))
    mfps, n_sim = cap.get(5), int(cap.get(7))
    mx = (383.5 + U * 376.0).astype(np.float32)
    my = (383.5 + V * 376.0).astype(np.float32)
    cur = dict(i=-1, f=None)

    def sim_frame(i):                                    # sequential reader of the simulation movie (left panel)
        while cur["i"] < i:
            ok, f = cap.read()
            if not ok:
                break
            cur["i"] += 1
            fr = cv2.remap(f[:, :768], mx, my, cv2.INTER_LINEAR)
            fr[outside] = 0
            cur["f"] = cv2.warpAffine(fr, M, (S, S))[..., ::-1]
        return cur["f"]

    I = np.load(os.path.join(od, "inlet.npz"))
    outline = _pitcher_outline()
    from .full_pour import TIP_KONA
    T = min(n_real / fps, (n_sim - 1) / mfps)
    font = cv2.FONT_HERSHEY_SIMPLEX
    w = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=7, macro_block_size=2)
    canvas = None
    for t in np.arange(0, T, 1 / fps):
        canvas = np.full((S + 70, 3 * S + 40, 3), 25, np.uint8)
        canvas[60:60 + S, 10:10 + S] = rr.get_data(min(int(round(t * fps)), n_real - 1))
        canvas[60:60 + S, 20 + S:20 + 2 * S] = sim_frame(min(int(round(t * mfps)), n_sim - 1))
        canvas[60:60 + S, 30 + 2 * S:30 + 3 * S] = action_panel(I, t, S, S, outline, TIP_KONA * 1e3)
        cv2.putText(canvas, "real (3 cameras, rectified)", (14, 30), font, 0.7, (230, 230, 230), 2)
        cv2.putText(canvas, "simulation (fitted closure)", (24 + S, 30), font, 0.7, (230, 230, 230), 2)
        cv2.putText(canvas, "hand action (measured)", (34 + 2 * S, 30), font, 0.7, (230, 230, 230), 2)
        cv2.putText(canvas, f"{p}   t = {t:5.2f} s", (14, 54), font, 0.6, (255, 220, 0), 1)
        w.append_data(canvas)
    for _ in range(fps):
        w.append_data(canvas)
    w.close()
    rr.close()
    cap.release()


def render(a, log=print):
    from multiprocessing import get_context
    bc = os.path.join(a.out, "best_closure.json")
    closure = json.load(open(bc)) if os.path.exists(bc) and not a.s1 else None
    log(f"render: {'S1 closure' if closure is None else 'fitted closure ' + bc}, N={a.N_render}, {a.jobs} processes")
    pats = a.patterns.split(",") if a.patterns else list(PATTERNS)
    rd = os.path.join(a.out, "render")
    os.makedirs(rd, exist_ok=True)
    with get_context("spawn").Pool(a.jobs, maxtasksperchild=1) as pool:
        res = {}
        for p, od, info in pool.imap_unordered(_solve_one, [(p, a.out, closure, a.N_render) for p in pats]):
            res[p] = od
            log(f"  {p}: solver {info['solver_wall_s']:.0f} s -> {od}")
    for p in pats:
        movie(p, res[p], os.path.join(rd, f"{p}.mp4"))
        log(f"  {p}: {os.path.join(rd, p + '.mp4')}")
    RP.SIM_SUB = "sim_fit"
    RP.compare(pats, "pitcher", a.N_render, out=os.path.join(rd, "summary.png"))
    log(f"  -> {rd}/summary.png (+ .json)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "report", "render"])
    ap.add_argument("--out", default="results/fit")
    ap.add_argument("--cal", default="tulip,heart", help="calibration patterns (the rest are predictions)")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--N-fig", dest="N_fig", type=int, default=256)
    ap.add_argument("--n", type=int, default=512, help="Sobol samples (stage A)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--confirm", type=int, default=4)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N-render", dest="N_render", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--patterns", default="")
    ap.add_argument("--s1", action="store_true", help="render with the S1 closure instead of the fitted one")
    a = ap.parse_args(argv)
    {"fit": fit, "report": report, "render": render}[a.cmd](a)


if __name__ == "__main__":
    main()
