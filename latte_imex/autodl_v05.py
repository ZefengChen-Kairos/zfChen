"""GPU pipeline (AutoDL) for the aligned model on the S1 forward-pitcher poses.

Aligned model = Params defaults: visc 'symmetric' (coupled implicit viscous stress), offcup 'account'; inlet from the
forward pose-driven pitcher (pitcher.ForwardPitcher, V1 mode B, batched on the GPU by cuda_pitcher) replayed on the
recorded S1 ideal-hand poses (latte_imex/data/s1_forward), turned into solver inlets by the S1 design program
(s1_forward.FwdProgram == v05_forward.forward_program).

    python -m latte_imex.autodl_v05 aligned --N 256,512 --video      # (a) six patterns: finals, movies, metrics
    python -m latte_imex.autodl_v05 robust --N 256                   # (b) hand-error grid (prereg P1-P6, R1-R3), batched
    python -m latte_imex.autodl_v05 figures                          # (c) montage, robustness grid, results.json
    python -m latte_imex.autodl_v05 all --video                      # a + b + c
Outputs under --out (default results/autodl_v05):
  pitcher/<pattern>_<name>.npz      arrival records (SI) and the pose perturbation used
  aligned/N<N>/<pattern>/           final.png, c.npz, final_state.npz, snapshots_c.npz, movie.mp4, metrics.json, trajectory.json
  robust/N<N>/<name>/<pattern>/     final.png, c.npz, metrics.json, trajectory.json
  fig_aligned_six.png, fig_robust_grid.png, results.json
Backends: --pitcher-backend gpu|cpu, --solver-backend cuda|cpu (cpu = the reference implementations, slow).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time

import numpy as np

from . import s1_forward as SF
from .pitcher import ForwardLaw, ForwardPitcher

BOXZ = dict(heart=(-0.3, 0.3, -0.35, 0.35), push_heart=(-0.35, 0.35, -0.35, 0.35), layered_heart=(-0.25, 0.25, -0.3, 0.3),
            tulip=(-0.3, 0.3, -0.32, 0.38), leaf=(-0.3, 0.3, -0.38, 0.38), swan=(-0.32, 0.3, -0.2, 0.36))


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _fp(alpha_deg, _cache={}):
    if alpha_deg not in _cache:
        _cache[alpha_deg] = ForwardPitcher(SF.geometry(alpha_deg), ForwardLaw())
    return _cache[alpha_deg]


def pitcher_jobs(patterns, names, backend="gpu", tol=0.003):
    """Forward pitcher on every (pattern, perturbation): pose perturbation, flow calibration (f != 1), arrivals.
    Returns {(pattern, name): dict(R, E-summary, info)}."""
    jobs, keys = [], []
    for p in patterns:
        P, meta = SF.load_poses(p), SF.load_meta(p)
        for nm in names:
            jobs.append((P, meta, SF.prereg_pert(nm, p)))
            keys.append((p, nm))
    out = {}
    for alpha in sorted({float(j[1]["alpha_deg"]) for j in jobs}):
        sel = [i for i, j in enumerate(jobs) if float(j[1]["alpha_deg"]) == alpha]
        t0 = time.time()
        res = SF.calibrate_flow([jobs[i] for i in sel], _fp(alpha), backend, tol=tol, log=_log)
        _log(f"pitcher: {len(sel)} pose sequences (alpha {alpha:g} deg) in {time.time() - t0:.1f} s ({backend})")
        for i, r in zip(sel, res):
            P, meta, pert = jobs[i]
            E = r["E"]
            R = ForwardPitcher.arrivals(E, 0.0, meta["n_out"], meta["dt"])
            fl = sorted(k - 1 for k in meta["flowing"])
            tz = r["seq"]["tip"][:, 2]
            ev = meta["events"]
            zmin = {str(k + 1): float(tz[(P["t"] >= ev[k]) & (P["t"] < ev[k + 1])].min() * 1e3) for k in fl}
            T = meta["duration"]
            info = dict(pert={str(k + 1): list(v) for k, v in pert.items()},
                        tilt_offset_deg={str(k + 1): math.degrees(v) for k, v in r["offs"].items()},
                        stroke_V_mL={str(k + 1): v for k, v in r["V"].items()},
                        stroke_V_ratio={str(k + 1): v for k, v in r["ratio"].items()},
                        flow_calibrated=bool(r["converged"]), calib_iters=int(r["iters"]),
                        tip_z_min_mm=zmin, poured_mL=float(np.sum(E["Q_o"]) * E["dt"] * 1e6),
                        V_pitcher_end_mL=float(E["V_P"][-1] * 1e6), V_spill_pitcher_mL=float(R["V_spill"][-1] * 1e6),
                        V_cup_in_T_mL=float(np.interp(T, R["t"], R["V_cup"]) * 1e6),
                        gpu_status=int(E.get("gpu_status", 0)), fill_mL=float(P["fill_mL"]))
            out[keys[i]] = dict(R=R, info=info, seq=r["seq"])
    return out


def _save_pitcher(outdir, key, d):
    os.makedirs(outdir, exist_ok=True)
    R = d["R"]
    np.savez_compressed(os.path.join(outdir, f"{key[0]}_{key[1]}.npz"), **{"a_" + k: v for k, v in R.items()},
                        dz=d["seq"]["dz"], delta=d["seq"]["delta"], dtilt=d["seq"]["dtilt"],
                        meta=json.dumps(d["info"], default=float))


def solve(pdata, N, root, video=False, snaps=False, backend="cuda", chunk=0, T_max=None, label_fn=None):
    """Batch solve of every (pattern, name) in pdata at N; returns {(pattern, name): metrics}."""
    from .batch_run import run_batch
    keys = list(pdata)
    mets = {}
    chunk = chunk or len(keys)
    for c0 in range(0, len(keys), chunk):
        ks = keys[c0:c0 + chunk]
        jobs = []
        for p, nm in ks:
            prog = SF.FwdProgram(p, pdata[(p, nm)]["R"], N)
            jobs.append(dict(script=SF.ProgScript(prog, p), pattern=p, out=root(p, nm), T_max=T_max,
                             label=label_fn(p, nm) if label_fn else f"{p} {nm} N{N}", extra=pdata[(p, nm)]["info"]))
        t0 = time.time()
        res = run_batch(jobs, N=N, backend=backend, snaps=snaps or video, video=video, quiet=False, log=_log,
                        save_state=snaps or video)
        _log(f"solver: {len(ks)} runs at N={N} in one batch: {time.time() - t0:.1f} s ({backend})")
        for k, m in zip(ks, res):
            mets[k] = m
    return mets


def cmd_aligned(a):
    pats = a.patterns.split(",")
    _log(f"(a) aligned model, forward pitcher on the S1 poses: {', '.join(pats)}")
    pdata = pitcher_jobs(pats, ["nominal"], a.pitcher_backend)
    for k, d in pdata.items():
        _save_pitcher(os.path.join(a.out, "pitcher"), k, d)
        _log(f"  {k[0]}: poured {d['info']['poured_mL']:.1f} mL of {d['info']['fill_mL']:.0f}, in the cup by T "
             f"{d['info']['V_cup_in_T_mL']:.1f} mL")
    for N in [int(v) for v in a.N.split(",")]:
        solve(pdata, N, lambda p, nm: os.path.join(a.out, "aligned", f"N{N}", p), video=a.video, snaps=True,
              backend=a.solver_backend, T_max=a.T, label_fn=lambda p, nm: f"{p} aligned N{N}")


def cmd_robust(a):
    pats = a.patterns.split(",")
    names = a.names.split(",")
    if "nominal" not in names:
        names = ["nominal"] + names
    _log(f"(b) hand-error robustness: {len(pats)} patterns x {len(names)} = {len(pats) * len(names)} runs")
    pdata = pitcher_jobs(pats, names, a.pitcher_backend)
    for k, d in pdata.items():
        _save_pitcher(os.path.join(a.out, "pitcher"), k, d)
    N = int(a.N.split(",")[0])
    solve(pdata, N, lambda p, nm: os.path.join(a.out, "robust", f"N{N}", nm, p), backend=a.solver_backend, chunk=a.chunk,
          T_max=a.T, label_fn=lambda p, nm: f"{p} {nm} N{N}")


# ----------------------------------------------------------------------------- figures and results
def _load_c(d):
    z = np.load(os.path.join(d, "c.npz"))
    return z["c"].astype(float), z["mask"]


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("s", [(0.25, 0.14, 0.06), (0.62, 0.45, 0.30), (1.0, 0.97, 0.90)])
    cm.set_bad((0.86, 0.86, 0.86))
    return cm


def _show(ax, c, p, cm):
    img = np.clip((c - 0.35) / 0.65, 0, 1)
    ax.imshow(img, cmap=cm, vmin=0, vmax=1, origin="upper", extent=(-0.5, 0.5, 0.5, -0.5), interpolation="nearest")
    x0, x1, y0, y1 = BOXZ.get(p, (-0.5, 0.5, -0.5, 0.5))
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])


def compare_c(c, c0, mask):
    """Shape agreement with the nominal run: IoU of the white region c > 0.5 inside the cup, white-area ratio (c > 0.4),
    mean |c - c0|."""
    m = mask & np.isfinite(c) & np.isfinite(c0)
    a, b = (c > 0.5) & m, (c0 > 0.5) & m
    iou = float((a & b).sum() / max((a | b).sum(), 1))
    ar = float(((c > 0.4) & m).sum() / max(((c0 > 0.4) & m).sum(), 1))
    return dict(iou_white=iou, white_area_ratio=ar, mean_abs_dc=float(np.mean(np.abs(c[m] - c0[m]))))


def cmd_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = _cmap()
    res = dict(model=dict(visc="symmetric", offcup="account", pitcher="ForwardPitcher (V1 mode B), S1 ideal-hand poses",
                          data="latte_imex/data/s1_forward"), aligned={}, robust={})
    pats = list(SF.PATTERNS)
    # (a) montage
    root = os.path.join(a.out, "aligned")
    Ns = sorted(int(d[1:]) for d in os.listdir(root)) if os.path.isdir(root) else []
    if Ns:
        fig, ax = plt.subplots(len(Ns), len(pats), figsize=(2.3 * len(pats), 2.5 * len(Ns) + 0.5), dpi=130, squeeze=False)
        for i, N in enumerate(Ns):
            for j, p in enumerate(pats):
                d = os.path.join(root, f"N{N}", p)
                x = ax[i, j]
                x.set_xticks([]); x.set_yticks([])
                if not os.path.exists(os.path.join(d, "c.npz")):
                    x.text(0.5, 0.5, "missing", ha="center", va="center", transform=x.transAxes, fontsize=7)
                    continue
                c, mask = _load_c(d)
                _show(x, c, p, cm)
                m = json.load(open(os.path.join(d, "metrics.json")))
                L = m["ledger"]
                res["aligned"].setdefault(f"N{N}", {})[p] = dict(
                    T=m["T"], elapsed_s=m["elapsed_s"], steps=m["stats"]["steps"], retries=m["stats"]["retries"],
                    deposited=L["deposited"], spilled=L["spilled"], layer_error=L["layer_error"], milk_error=L["milk_error"],
                    white_area_c_gt_0p4=m["white_area_c_gt_0p4"], pitcher=m.get("extra", {}))
                if i == 0:
                    x.set_title(p, fontsize=8)
                if j == 0:
                    x.set_ylabel(f"N = {N}", fontsize=8)
                x.text(0.03, 0.03, f"dep {L['deposited']:.3f}  spill {L['spilled']:.3f}", color="white", fontsize=5.5,
                       transform=x.transAxes)
        fig.suptitle("Aligned model (symmetric viscosity, off-cup accounting) on the forward pitcher (S1 ideal-hand poses): "
                     "c stretched 0.35..1", fontsize=8.5)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fp = os.path.join(a.out, "fig_aligned_six.png")
        fig.savefig(fp); plt.close(fig)
        _log(fp)
    # (b) robustness grid
    root = os.path.join(a.out, "robust")
    Nr = sorted(int(d[1:]) for d in os.listdir(root)) if os.path.isdir(root) else []
    if Nr:
        N = Nr[0]
        names = [n for n in SF.NAMES if os.path.isdir(os.path.join(root, f"N{N}", n))]
        pats = [p for p in SF.PATTERNS if any(os.path.isdir(os.path.join(root, f"N{N}", n, p)) for n in names)]
        fig, ax = plt.subplots(len(pats), len(names), figsize=(2.0 * len(names), 2.15 * len(pats) + 0.6), dpi=130, squeeze=False)
        for i, p in enumerate(pats):
            c0 = None
            d0 = os.path.join(root, f"N{N}", "nominal", p)
            if os.path.exists(os.path.join(d0, "c.npz")):
                c0, mask0 = _load_c(d0)
            for j, nm in enumerate(names):
                x = ax[i, j]
                x.set_xticks([]); x.set_yticks([])
                d = os.path.join(root, f"N{N}", nm, p)
                if not os.path.exists(os.path.join(d, "c.npz")):
                    x.text(0.5, 0.5, "missing", ha="center", va="center", transform=x.transAxes, fontsize=7)
                    continue
                c, mask = _load_c(d)
                _show(x, c, p, cm)
                m = json.load(open(os.path.join(d, "metrics.json")))
                r = dict(white_area_c_gt_0p4=m["white_area_c_gt_0p4"], deposited=m["ledger"]["deposited"],
                         spilled=m["ledger"]["spilled"], retries=m["stats"]["retries"], pitcher=m.get("extra", {}))
                if c0 is not None:
                    r.update(compare_c(c, c0, mask))
                    if nm != "nominal":
                        q = r["iou_white"]
                        col = (0.1, 0.55, 0.2) if q >= 0.8 else (0.85, 0.55, 0.0) if q >= 0.6 else (0.8, 0.1, 0.1)
                        x.text(0.03, 0.97, f"IoU {q:.2f}  A x{r['white_area_ratio']:.2f}", color="white", fontsize=6,
                               ha="left", va="top", transform=x.transAxes,
                               bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none"))
                        for s in x.spines.values():
                            s.set_edgecolor(col); s.set_linewidth(2)
                res["robust"].setdefault(p, {})[nm] = r
                if i == 0:
                    x.set_title(SF.LAB[nm], fontsize=7)
                if j == 0:
                    x.set_ylabel(p, fontsize=8)
        fig.suptitle(f"Hand-error robustness on the forward pitcher (S1 poses; dz on the tip, flow by a calibrated tilt offset, "
                     f"timing by shifting the tilt), aligned model, {N}$^2$\nc stretched 0.35..1; label = IoU of c > 0.5 "
                     f"with the nominal run and the white-area ratio (an indicator, not a verdict)", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fp = os.path.join(a.out, "fig_robust_grid.png")
        fig.savefig(fp); plt.close(fig)
        _log(fp)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=float)
    _log(os.path.join(a.out, "results.json"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["aligned", "robust", "figures", "all"])
    ap.add_argument("--out", default="results/autodl_v05")
    ap.add_argument("--N", default="256,512", help="aligned: list; robust: the first one")
    ap.add_argument("--patterns", default=",".join(SF.PATTERNS))
    ap.add_argument("--names", default=",".join(SF.NAMES))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--chunk", type=int, default=0, help="robust: members per GPU batch (0 = all at once)")
    ap.add_argument("--T", type=float, default=None, help="truncate every run at T [s] (smoke tests)")
    ap.add_argument("--pitcher-backend", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--solver-backend", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.cmd in ("aligned", "all"):
        cmd_aligned(a)
    if a.cmd in ("robust", "all"):
        if a.cmd == "all":
            a.N = a.N.split(",")[0]
        cmd_robust(a)
    if a.cmd in ("figures", "all"):
        cmd_figures(a)
    _log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
