"""Feasibility check: can ANY V0.5 closure inside the physical ranges open the tulip's petals with the barista's measured
actions?  (Before calibrating anything: if no parameter set produces the brown gaps between the stacked hearts, the
model, not the data, is the bottleneck.)

Inlet: the measured tulip pour through the real-pitcher model with the 165 g whole-pour flow calibration
(full_pour.DTILT axis correction, V0 = fill at the cut from full_pour_rho0.8.json), as sim_flow165.  Built once.
Space: v05_adapter.SPACE (B_dep, p_dep, kappa_r, kappa_Q, cp, beta, kappa_t, kappa_c; physical bounds), nu and D as S1.
Stages (resumable; every finished batch is appended to DIR/records.jsonl and its fields to DIR/fields/):
  A  scrambled Sobol sample of the box (--n, default 256) + the S1 closure as reference, at --N (default 128)
  B  local refinement: --per samples (gaussian, sigma 0.08 in the unit box) around each of the --top best of A
  C  confirmation: the --confirm best (by J) and the S1 reference again at --N-fig (default 256), contact sheet
Metrics (target = raw_data/rig/targets/<p>.npz, white = c > 0.5, disc coordinates):
  iou      white IoU
  gap_iou  IoU of the 'gaps' = brown inside the convex hull of the white pattern (components > 2 % of the white),
           opened with a disc of 0.02 R so that thin lines between nested layers do not count.  This is the petal
           opening: the target tulip has wide brown wedges between its hearts (gap area 0.37 R^2); nested 'onion'
           layers give thin slivers only (S1: gap_iou 0.08-0.10).
  J = iou + gap_iou.  Verdict 'opens' if some run reaches gap_iou >= 0.30 with iou >= 0.35 at N-fig (look at the sheet).

    python -m latte_imex.rig.tulip_feas run    --out results/tulip_feas [--backend cuda --batch 48 --n 256 --N 128]
    python -m latte_imex.rig.tulip_feas report --out results/tulip_feas
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np

from . import real_program as RP
from .. import s1_forward as SF
from ..v05_adapter import SPACE

KEYS = tuple(SPACE)
GAP_IOU_OK, IOU_OK = 0.30, 0.35


# ----------------------------------------------------------------------------- parameters
def from_unit(u):
    out = {}
    for k, x in zip(KEYS, u):
        lo, hi, sc = SPACE[k]
        x = float(np.clip(x, 0.0, 1.0))
        out[k] = lo * (hi / lo) ** x if sc == "log" else lo + (hi - lo) * x
    return out


def to_unit(par):
    u = []
    for k in KEYS:
        lo, hi, sc = SPACE[k]
        v = float(par[k])
        u.append(math.log(v / lo) / math.log(hi / lo) if sc == "log" else (v - lo) / (hi - lo))
    return np.clip(np.array(u), 0.0, 1.0)


# ----------------------------------------------------------------------------- inlet (built once)
class Pour:
    def __init__(self, p, flow=os.environ.get("FEAS_FLOW", "kona")):
        """flow 'kona': catalogue pitcher, no axis correction, fill solved for 165 g poured (full_pour_kona_poured165.json);
        'dtilt': the earlier calibration (real-pitcher parametric geometry + per-pour axis correction, sim_flow165)."""
        if flow == "kona":
            os.environ["PITCHER_GEOM"] = os.path.join(RP.REPO, "latte_imex", "pitcher_geom", "pitcher_kona700.json")
            fill = json.load(open(os.path.join(RP.RES, "full_pour_kona_poured165.json")))[p]["fill_at_cut_mL"]
        else:
            from .full_pour import DTILT
            RP.DTILT.update(DTILT)
            fill = json.load(open(os.path.join(RP.RES, "full_pour_rho0.8.json")))[p]["fill_at_cut_mL"]
        RP._FP.clear()
        C, E, R, info = RP.build_inlet(p, "pitcher", fill * 1e-6, log=lambda *a: None)
        self.p, self.info = p, info
        self.T = info["T_pour"] + RP.TAIL
        self.meta = RP.phases_meta(p, C, R, self.T)
        self.R = RP.scaled_record(R, C["L"])
        self.s1 = {k: float(self.meta["config"]["parameters"][k]) for k in KEYS}

    def member(self, closure, N, tag):
        from ..gpu_batch import ProgMember
        from ..solver import Params
        m = copy.deepcopy(self.meta)
        m["config"]["parameters"].update(closure)
        m["kQ"] = float(closure["kappa_Q"])
        prog = SF.FwdProgram(self.p, self.R, N, meta=m)
        sc = SF.ProgScript(prog, self.p)
        return ProgMember(sc, Params(**sc.params), tag=tag)


# ----------------------------------------------------------------------------- metrics
def _rot(img, a):
    import cv2
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), a, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def gaps(white, valid, px_per_R):
    import cv2
    from scipy import ndimage as ndi
    w = white & valid
    lab, n = ndi.label(w)
    G = np.zeros_like(w)
    if n == 0:
        return G
    sizes = ndi.sum(w, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.where(sizes > 0.02 * sizes.sum())[0])
    pts = cv2.findNonZero(keep.astype(np.uint8))
    H = np.zeros(w.shape, np.uint8)
    cv2.fillPoly(H, [cv2.convexHull(pts)], 1)
    G = (H > 0) & ~w & valid
    r = max(1, int(round(0.02 * px_per_R)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(G.astype(np.uint8), cv2.MORPH_OPEN, k) > 0


class Target:
    def __init__(self, p):
        T = np.load(os.path.join(RP.TGT, f"{p}.npz"))
        self.white, self.valid, self.img = T["white"].astype(bool), T["valid"].astype(bool), T["img"]
        self.ex = float(T["extent"])
        self.n = self.white.shape[0]
        self.px = self.n / (2 * self.ex)
        self.G = gaps(self.white, self.valid, self.px)
        self.rot = json.load(open(os.path.join(RP.TGT, "view_rotation.json"))).get(p, 0.0)

    def score(self, c):
        cs = RP.sim_in_disc(np.asarray(c, float), self.n, self.ex)
        v = self.valid & np.isfinite(cs)
        w = (np.nan_to_num(cs) > 0.5) & v
        t = self.white & v
        G = gaps(w, v, self.px)
        iou = float((w & t).sum() / max((w | t).sum(), 1))
        giou = float((G & self.G).sum() / max((G | self.G).sum(), 1))
        return dict(iou=iou, gap_iou=giou, J=iou + giou, gap_area=float(G.sum() / self.px ** 2),
                    white_ratio=float(w.sum() / max(t.sum(), 1))), cs, G


# ----------------------------------------------------------------------------- run
def _load(out):
    recs = []
    fn = os.path.join(out, "records.jsonl")
    if os.path.exists(fn):
        with open(fn) as f:
            recs = [json.loads(l) for l in f if l.strip()]
    return recs


def _run_batch(pour, tgt, jobs, N, eng, out, log):
    """jobs: list of (id, stage, closure).  Appends records, saves fields (float16)."""
    mem = [pour.member(cl, N, tag=i) for i, st, cl in jobs]
    t0 = time.time()
    eng.run(mem)
    wall = time.time() - t0
    os.makedirs(os.path.join(out, "fields"), exist_ok=True)
    rows = []
    for (i, st, cl), m in zip(jobs, mem):
        r = m.result
        rec = dict(id=i, stage=st, N=N, params=cl, ok=bool(r["ok"]), error=r["error"], wall_s=wall / len(jobs))
        if r["ok"]:
            M, _, _ = tgt.score(r["c"])
            rec.update(M)
            np.save(os.path.join(out, "fields", f"{i}_N{N}.npy"), np.nan_to_num(r["c"]).astype(np.float16))
        else:
            rec.update(iou=0.0, gap_iou=0.0, J=0.0)
        rows.append(rec)
    with open(os.path.join(out, "records.jsonl"), "a") as f:
        for rec in rows:
            f.write(json.dumps(rec, default=float) + "\n")
    good = [r for r in rows if r["ok"]]
    log(f"  batch of {len(jobs)} at N={N}: {wall:.0f} s, {len(good)} ok; best J {max([r['J'] for r in good], default=0):.3f}"
        f" (gap_iou max {max([r['gap_iou'] for r in good], default=0):.3f})")


def run(a, log=print):
    from ..gpu_batch import BatchEngine
    os.makedirs(a.out, exist_ok=True)
    log(f"building the {a.pattern} inlet (real pitcher, 165 g calibration) ...")
    pour, tgt = Pour(a.pattern), Target(a.pattern)
    log(f"  V0 {pour.info['V0_mL']:.1f} mL, poured {pour.info['poured_mL']:.1f} mL, T {pour.T:.1f} s; target gap area "
        f"{tgt.G.sum() / tgt.px ** 2:.3f} R^2")
    json.dump(dict(vars(a), space={k: list(v) for k, v in SPACE.items()}, s1=pour.s1, inlet=pour.info),
              open(os.path.join(a.out, "config.json"), "w"), indent=1, default=float)
    done = {r["id"] for r in _load(a.out)}
    eng = {}

    def engine(N):
        if N not in eng:
            eng[N] = BatchEngine(N, backend=a.backend)
        return eng[N]

    def go(jobs, N, label):
        todo = [j for j in jobs if j[0] not in done]
        if not todo:
            return
        log(f"stage {label}: {len(todo)} runs at N={N}")
        for k in range(0, len(todo), a.batch):
            _run_batch(pour, tgt, todo[k:k + a.batch], N, engine(N), a.out, log)
            done.update(j[0] for j in todo[k:k + a.batch])

    full = lambda par: dict(par, nu=float(pour.meta["config"]["parameters"]["nu"]), D=float(pour.meta["config"]["parameters"]["D"]))
    # A: Sobol + S1 reference
    from scipy.stats import qmc
    U = qmc.Sobol(len(KEYS), scramble=True, seed=a.seed).random(a.n)
    jobs = [("S1", "A", full(pour.s1))] + [(f"A{i:04d}", "A", full(from_unit(u))) for i, u in enumerate(U)]
    go(jobs, a.N, "A (Sobol)")
    # B: local refinement around the best of A
    A = sorted([r for r in _load(a.out) if r["stage"] == "A" and r["ok"]], key=lambda r: -r["J"])[:a.top]
    rng = np.random.default_rng(a.seed + 1)
    jobs = []
    for k, r in enumerate(A):
        u0 = to_unit(r["params"])
        for j in range(a.per):
            jobs.append((f"B{k:02d}_{j:02d}", "B", full(from_unit(u0 + rng.normal(0, 0.08, len(KEYS))))))
    go(jobs, a.N, "B (refine)")
    # C: confirm the best at N-fig
    AB = sorted([r for r in _load(a.out) if r["stage"] in ("A", "B") and r["ok"] and r["N"] == a.N], key=lambda r: -r["J"])
    pick = [r for r in AB if r["id"] != "S1"][:a.confirm] + [r for r in AB if r["id"] == "S1"]
    go([(f"C_{r['id']}", "C", r["params"]) for r in pick], a.N_fig, "C (confirm)")
    report(a, log)


# ----------------------------------------------------------------------------- report
def report(a, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import spearmanr
    recs = [r for r in _load(a.out) if r["ok"]]
    tgt = Target(a.pattern)
    A = [r for r in recs if r["stage"] == "A" and r["id"] != "S1"]
    C = sorted([r for r in recs if r["stage"] == "C"], key=lambda r: -r["J"])
    S1 = [r for r in recs if r["id"] in ("S1", "C_S1")]
    sens = {}
    if len(A) > 8:
        Uu = np.array([to_unit(r["params"]) for r in A])
        for j, k in enumerate(KEYS):
            sens[k] = dict(rho_gap=float(spearmanr(Uu[:, j], [r["gap_iou"] for r in A])[0]),
                           rho_iou=float(spearmanr(Uu[:, j], [r["iou"] for r in A])[0]))
    best = C[0] if C else (max(recs, key=lambda r: r["J"]) if recs else None)
    opens = bool(C and any(r["gap_iou"] >= GAP_IOU_OK and r["iou"] >= IOU_OK for r in C if not r["id"].endswith("S1")))
    n_fail = sum(1 for r in _load(a.out) if not r["ok"])
    summ = dict(pattern=a.pattern, n_runs=len(recs), n_failed=n_fail, verdict="opens" if opens else "does not open",
                thresholds=dict(gap_iou=GAP_IOU_OK, iou=IOU_OK), s1=S1, best=best, confirmed=C, sensitivity=sens,
                stage_A=dict(gap_iou_max=max([r["gap_iou"] for r in A], default=None),
                             gap_iou_p90=float(np.percentile([r["gap_iou"] for r in A], 90)) if A else None,
                             iou_max=max([r["iou"] for r in A], default=None)))
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    # scatter
    if A:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=110)
        ax.scatter([r["iou"] for r in A], [r["gap_iou"] for r in A], s=10, alpha=0.6, label="stage A (Sobol)")
        B = [r for r in recs if r["stage"] == "B"]
        if B:
            ax.scatter([r["iou"] for r in B], [r["gap_iou"] for r in B], s=10, alpha=0.6, label="stage B (refine)")
        for r in S1:
            ax.scatter([r["iou"]], [r["gap_iou"]], s=60, marker="*", color="k", label=f"S1 (N={r['N']})")
        ax.axhline(GAP_IOU_OK, ls="--", lw=0.8, color="0.5"); ax.axvline(IOU_OK, ls="--", lw=0.8, color="0.5")
        ax.set_xlabel("white IoU"); ax.set_ylabel("gap IoU (petal opening)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_title(f"{a.pattern}: {len(recs)} runs")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "scatter.png")); plt.close(fig)
    # contact sheet: target + confirmed runs, barista view, gaps outlined
    show = C[:11]
    if show:
        cm = LinearSegmentedColormap.from_list("milk", [(0.36, 0.20, 0.09), (0.66, 0.47, 0.30), (1.0, 0.97, 0.92)])
        cm.set_bad((0.12, 0.12, 0.12))
        nc = 4
        nr = int(math.ceil((len(show) + 1) / nc))
        fig, ax = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.5 * nr), dpi=100, squeeze=False)
        for x in ax.flat:
            x.axis("off")
        ax[0, 0].imshow(_rot(tgt.img[..., ::-1].copy(), tgt.rot))
        ax[0, 0].contour(_rot(tgt.G.astype(np.float32), tgt.rot), [0.5], colors="cyan", linewidths=0.7)
        ax[0, 0].set_title("target (gaps cyan)", fontsize=9)
        for k, r in enumerate(show, 1):
            fn = os.path.join(a.out, "fields", f"{r['id']}_N{r['N']}.npy")
            M, cs, G = tgt.score(np.load(fn).astype(float))
            x = ax.flat[k]
            msk = _rot(np.isfinite(cs).astype(np.float32), tgt.rot) > 0.5
            img = np.ma.masked_array(_rot(np.nan_to_num(cs).astype(np.float32), tgt.rot), ~msk)
            x.imshow(img, cmap=cm, vmin=0, vmax=0.8)
            x.contour(_rot(G.astype(np.float32), tgt.rot), [0.5], colors="cyan", linewidths=0.7)
            x.set_title(f"{r['id']}\nIoU {M['iou']:.2f}  gap {M['gap_iou']:.2f}", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "sheet.png")); plt.close(fig)
    log(f"{a.pattern}: {len(recs)} runs ({n_fail} failed); verdict: {summ['verdict']}")
    for r in S1:
        log(f"  S1 (N={r['N']}): IoU {r['iou']:.3f} gap {r['gap_iou']:.3f}")
    for r in C[:6]:
        log(f"  {r['id']}: IoU {r['iou']:.3f} gap {r['gap_iou']:.3f}  " +
            " ".join(f"{k} {r['params'][k]:.3g}" for k in KEYS))
    if sens:
        log("  sensitivity (Spearman, stage A) gap / IoU: " +
            ", ".join(f"{k} {v['rho_gap']:+.2f}/{v['rho_iou']:+.2f}" for k, v in sens.items()))
    log(f"  -> {os.path.join(a.out, 'summary.json')}, sheet.png, scatter.png")
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", default="results/tulip_feas")
    ap.add_argument("--pattern", default="tulip")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--N-fig", dest="N_fig", type=int, default=256)
    ap.add_argument("--n", type=int, default=256, help="Sobol samples (stage A)")
    ap.add_argument("--top", type=int, default=8, help="stage-B centres")
    ap.add_argument("--per", type=int, default=8, help="stage-B samples per centre")
    ap.add_argument("--confirm", type=int, default=11, help="stage-C runs (+ S1)")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
