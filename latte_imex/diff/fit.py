"""Gradient-based calibration of the V0.5 closure with the differentiable Warp solver.

Parameters: v05_adapter.SPACE (8 closure parameters) in the unit box (log-scaled where SPACE says 'log'); nu and D
fixed.  Inlet: the measured pour with the landing point corrected by the measured stream foot (E0 input replay,
schedule.e0_offsets; --no-e0: pitcher model alone).  Diagnostic option --foot: also fit a landing-point correction shared by all pours, foot_a / foot_b [cup
diameters] along / across the jet's horizontal direction (schedule.foot_shift) -- off by default (the inlet is not
corrected from the target pattern; see the V1 plan, 4.1 and E0).
After the fit, the frozen parameters are run on every other pattern (frozen-parameter test): test.json, with the same
whole-video metrics.  Objective (--loss video, default): whole-video supervision, run.DiffPour -- every observed frame of each
calibration pour compared with the continuous simulation on its visible surface, normalised per frame, averaged over
physical time; calibration sequences weighted equally.  Each iteration runs every calibration pattern forward and
backward (exact adjoint, run.DiffPour.loss_grad), averages the losses and their gradients and takes an Adam step in
the unit box (clipped to [0, 1]); the sigmoid temperature is annealed from --tau0 to --tau1.  Reported with every
iteration: the mean per-frame IoU over the video (vIoU) and the final-frame metrics of the Sobol fit (tulip_feas
Target.score: IoU, gap IoU, J).  The iterate with the highest whole-video IoU is kept (with --loss final: the highest final J).
Start: --init a closure json (e.g. results/fit/best_closure.json from fit_pipeline, so the gradient refines the
global Sobol search), default the S1 closure.  A failed forward (non-finite state) halves the step and retries.

Outputs in --out: records.jsonl (one line per iteration), best_closure.json (fit_pipeline render reads it),
progress.png, sheet.png (target | start | best), fields/<p>_{start,best}.npy.
    python -m latte_imex.diff.fit --out results/diff_fit [--cal tulip,leaf --N 128 --iters 60 --init ...]
    python -m latte_imex.rig.fit_pipeline render --out results/diff_fit          # movies with the result
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from ..v05_adapter import SPACE
from . import wsolver as W

FOOT_SPACE = {"foot_a": (-0.25, 0.25, "lin"),      # landing-point correction [cup diameters], along the jet
              "foot_b": (-0.15, 0.15, "lin")}      # ... and across it (schedule.foot_shift); shared by all pours
SP = dict(SPACE)
KEYS = tuple(SPACE)


CIRC_SPACE = {"kappa_M": (0.0, 8.0, "lin"),        # surface current driven by the stream's momentum (circulation.py)
              "eps_f": (0.0, 1.0, "lin"),          # fraction of the foam going down with the current
              "kappa_w": (0.0, 2.0, "lin")}        # the received flow's area going down with the current
# start: kappa_M = 1.5 (the time-mean surface current of the optical flow, 0.04 D/s, needs ~10 in total, but most of it
# is the pattern pushed by the jet itself, which the inlet already models; forward runs at S1: tulip best at 1-2)
CIRC_START = {"kappa_M": 1.5, "eps_f": 0.0, "kappa_w": 0.0}
# the foam surfaces tau_s u_in ahead of the landing point (u_in ~ 1 D/s: tau_s = 0.05 s ~ 4 mm; the real deposits grow
# forward of the measured stream foot)
DEP_SPACE = {"tau_s": (0.0, 0.2, "lin"),
             "mu_s": (0.0, 1.0, "lin")}           # share of the jet momentum the surfacing foam has lost
DEP_START = {"tau_s": 0.05, "mu_s": 0.0}


def use_foot(on, circ=False, dep=False):
    global KEYS, SP
    SP = dict(SPACE, **(FOOT_SPACE if on else {}), **(CIRC_SPACE if circ else {}), **(DEP_SPACE if dep else {}))
    KEYS = tuple(SP)


def to_unit(P):
    u = []
    for k in KEYS:
        lo, hi, sc = SP[k]
        v = float(P[k])
        u.append(math.log(v / lo) / math.log(hi / lo) if sc == "log" else (v - lo) / (hi - lo))
    return np.clip(np.array(u), 0.0, 1.0)


def from_unit(u, base):
    P = dict(base)
    for k, x in zip(KEYS, np.clip(u, 0.0, 1.0)):
        lo, hi, sc = SP[k]
        P[k] = lo * (hi / lo) ** x if sc == "log" else lo + (hi - lo) * x
    return P


def dtheta_du(P):
    """d theta / d u per key (chain rule from the unit box)."""
    out = []
    for k in KEYS:
        lo, hi, sc = SP[k]
        out.append(float(P[k]) * math.log(hi / lo) if sc == "log" else hi - lo)
    return np.array(out)


# ----------------------------------------------------------------------------- one pour, in-process or in a worker process
def _step(D, P, tau):
    D.tau = tau
    L, g, final = D.loss_grad(P)
    M, c = score(D, final)
    M["viou"] = float(np.mean([v for _, v in D.vmetrics])) if D.vmetrics else float("nan")
    return dict(L=L, g=g, M=M, c=c, strip=D.strip, vmetrics=D.vmetrics, timing=D.timing)


def _eval(D, P):
    """Forward only with frozen parameters: whole-video loss, per-frame IoU, final-frame metrics."""
    L, final, _ = D.loss(P)
    snaps = D.sol.snaps
    n = D.S["n"]
    after = lambda k: snaps[k]
    vm = D.video_metrics(after) if D.obs else []
    M, c = score(D, final)
    M["viou"] = float(np.mean([v for _, v in vm])) if vm else float("nan")
    return dict(L=L, M=M, c=c, vmetrics=vm)


def _worker(conn, p, kw):
    import traceback
    from .run import DiffPour
    try:
        D = DiffPour(p, log=lambda *x: print(*x, flush=True), **kw)
        conn.send(("ok", dict(s1=D.s1)))
    except Exception:
        conn.send(("err", traceback.format_exc()))
        return
    while True:
        msg = conn.recv()
        if msg is None:
            return
        try:
            conn.send(("ok", _eval(D, msg[1]) if msg[0] == "eval" else _step(D, *msg)))
        except Exception as e:
            conn.send(("err", f"{type(e).__name__}: {e}"))


class Pour:
    """A calibration pour: in a worker process (parallel: own CUDA context, own Python interpreter -- the backward
    replay is launch/Python-bound, so separate processes keep the GPU busy) or in-process."""

    def __init__(self, p, kw, process):
        self.p, self.process = p, process
        if process:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self.conn, child = ctx.Pipe()
            self.proc = ctx.Process(target=_worker, args=(child, p, kw), daemon=True)
            self.proc.start()
            st, info = self.conn.recv()
            if st != "ok":
                raise RuntimeError(f"{p} worker failed to start:\n{info}")
            self.s1 = info["s1"]
        else:
            from .run import DiffPour
            self.D = DiffPour(p, log=lambda *x: print(*x, flush=True), **kw)
            self.s1 = self.D.s1

    def send(self, P, tau):
        if self.process:
            self.conn.send((P, tau))
        else:
            try:
                self._res = ("ok", _step(self.D, P, tau))
            except RuntimeError as e:
                self._res = ("err", str(e))

    def send_eval(self, P):
        if self.process:
            self.conn.send(("eval", P))
        else:
            try:
                self._res = ("ok", _eval(self.D, P))
            except RuntimeError as e:
                self._res = ("err", str(e))

    def recv(self):
        st, r = self.conn.recv() if self.process else self._res
        if st != "ok":
            raise RuntimeError(f"{self.p}: {r}")
        return r

    def close(self):
        if self.process:
            self.conn.send(None)
            self.proc.join(timeout=30)


def score(D, final):
    msk = D.geo.mask_np
    c = np.where(msk, final["m"] / np.where(msk, final["l"], 1.0), np.nan)
    M, _, _ = D.T.score(c)
    return M, c


def sheet(out, cal, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from ..rig.tulip_feas import _rot
    cm = LinearSegmentedColormap.from_list("milk", [(0.36, 0.20, 0.09), (0.66, 0.47, 0.30), (1.0, 0.97, 0.92)])
    cm.set_bad((0.12, 0.12, 0.12))
    fig, ax = plt.subplots(3, len(cal), figsize=(3.3 * len(cal), 10.2), dpi=100, squeeze=False)
    for j, p in enumerate(cal):
        from ..rig.tulip_feas import Target
        T = Target(p)
        ax[0, j].imshow(_rot(T.img[..., ::-1].copy(), T.rot))
        ax[0, j].contour(_rot(T.G.astype(np.float32), T.rot), [0.5], colors="cyan", linewidths=0.7)
        ax[0, j].set_title(f"{p} target", fontsize=9)
        for i, tag in ((1, "start"), (2, "best")):
            c = np.load(os.path.join(out, "fields", f"{p}_{tag}.npy"))
            M, cs, G = T.score(c)
            msk = _rot(np.isfinite(cs).astype(np.float32), T.rot) > 0.5
            ax[i, j].imshow(np.ma.masked_array(_rot(np.nan_to_num(cs).astype(np.float32), T.rot), ~msk), cmap=cm, vmin=0,
                            vmax=0.8)
            ax[i, j].contour(_rot(G.astype(np.float32), T.rot), [0.5], colors="cyan", linewidths=0.7)
            ax[i, j].set_title(f"{tag}: IoU {M['iou']:.2f} gap {M['gap_iou']:.2f}", fontsize=9)
        for i in range(3):
            ax[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sheet.png"))
    plt.close(fig)


def video_strip(out, cal):
    """Per calibration pattern: observed whiteness (visible area) | start | best simulation at 8 observed times."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for p in cal:
        for tag in ("start", "best"):
            fn = os.path.join(out, "fields", f"{p}_{tag}_video.npz")
            if os.path.exists(fn):
                rows.append((p, tag, np.load(fn)))
    if not rows:
        return
    nc = rows[0][2]["c"].shape[0]
    fig, ax = plt.subplots(len(cal) * 3, nc, figsize=(1.9 * nc, 2.0 * 3 * len(cal)), dpi=90, squeeze=False)
    for x in ax.flat:
        x.axis("off")
    for i, p in enumerate(cal):
        Zs = {tag: Z for q, tag, Z in rows if q == p}
        if "best" not in Zs:
            continue
        Zb = Zs["best"]
        for j in range(nc):
            w = np.where(Zb["v"][j], Zb["w"][j], np.nan)
            ax[3 * i, j].imshow(w, cmap="gray", vmin=0, vmax=1)
            ax[3 * i, j].set_title(f"{p} obs t={Zb['t'][j]:.1f}s", fontsize=7)
            for r, tag in ((1, "start"), (2, "best")):
                if tag in Zs:
                    ax[3 * i + r, j].imshow(np.clip(Zs[tag]["c"][j], 0, 1),
                                            cmap="gray", vmin=0, vmax=1)
                    ax[3 * i + r, j].set_title(f"sim ({tag})", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "video_strip.png"))
    plt.close(fig)


def progress(out, recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    it = [r["it"] for r in recs]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=100)
    ax[0].plot(it, [r["loss"] for r in recs], "k.-"); ax[0].set_title("loss (mean over sequences)")
    for p in recs[0]["per"]:
        ax[1].plot(it, [r["per"][p]["J"] for r in recs], ".-", label=f"{p} J")
    ax[1].plot(it, [r["J"] for r in recs], "k.-", label="mean final J")
    ax[1].plot(it, [r["viou"] for r in recs], "r.-", label="mean video IoU"); ax[1].legend(fontsize=7)
    ax[1].set_title("video IoU / final IoU + gap IoU")
    U = np.array([r["u"] for r in recs])
    for i, k in enumerate(KEYS):
        ax[2].plot(it, U[:, i], ".-", label=k)
    ax[2].legend(fontsize=6, ncol=2); ax[2].set_title("parameters (unit box)"); ax[2].set_ylim(-0.02, 1.02)
    for x in ax:
        x.grid(alpha=0.3); x.set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "progress.png"))
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/diff_fit")
    ap.add_argument("--cal", default="tulip,leaf", help="calibration pours (V1 plan 5.1: A = tulip, A+B = tulip,leaf); "
                    "the other patterns are frozen-parameter tests")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.04, help="Adam step in the unit box")
    ap.add_argument("--tau0", type=float, default=0.10)
    ap.add_argument("--tau1", type=float, default=0.03)
    ap.add_argument("--w-gap", dest="w_gap", type=float, default=2.0)
    ap.add_argument("--loss", default="video", choices=["video", "video_iou", "final"])
    ap.add_argument("--scales", default="0.3,0.15,0",
                    help="video loss scales [cup radii] (run.py): blurred residual terms, 0 = per pixel; '0' = pixel term only")
    ap.add_argument("--w-flow", dest="w_flow", type=float, default=0.0,
                    help="weight of the surface-velocity term (optical flow of the real pour, rig/video_flow.py)")
    ap.add_argument("--dep", action="store_true",
                    help="foam surfaces tau_s u_in ahead of the landing point with the share mu_s of the jet momentum "
                    "lost (fit tau_s, mu_s; off: V0.5, tau_s = mu_s = 0)")
    ap.add_argument("--circ", action="store_true",
                    help="surface current driven by the stream's momentum (circulation.py: fit kappa_M, eps_f, kappa_w; use with "
                         "--upwell)")
    ap.add_argument("--upwell", action="store_true",
                    help="volume-conserving surface source: the whole received volume reaches the surface, chi decides "
                         "whether it is new milk or resurfacing base material (wsolver.k_fields)")
    ap.add_argument("--no-e0", dest="e0", action="store_false",
                    help="landing point from the pitcher model alone (default: E0, corrected by the measured stream foot, "
                         "rig/stream_foot.py)")
    ap.add_argument("--w-final", dest="w_final", type=float, default=0.0, help="extra final-frame soft-IoU term (video loss)")
    ap.add_argument("--init", default=None, help="closure json to start from (default: S1)")
    ap.add_argument("--set", default="", help="start values overriding the start closure, e.g. beta=0.5,kappa_t=1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--T", type=float, default=None, help="simulate only the first T seconds (tests)")
    ap.add_argument("--foot", action="store_true",
                    help="DIAGNOSTIC ONLY: also fit a shared landing-point correction (foot_a, foot_b) from the pattern "
                    "loss.  Off by default: the V1 plan does not correct the inlet from the target pattern (4.1, E0); "
                    "a landing bias is to be measured from the stream itself in the input replay")
    ap.add_argument("--no-test", dest="no_test", action="store_true", help="skip the frozen-parameter test")
    ap.add_argument("--workers", type=int, default=-1, help="-1: one process per pour on CUDA, in-process on the CPU; "
                    "0: in-process; >0: worker processes")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "fields"), exist_ok=True)
    log = lambda *x: print(*x, flush=True)
    use_foot(a.foot, a.circ, a.dep)
    cal = a.cal.split(",")
    kw = dict(N=a.N, device=a.device, w_gap=a.w_gap, tau=a.tau0, T_max=a.T, loss=a.loss, w_final=a.w_final,
              scales=tuple(float(x) for x in a.scales.split(",")), e0=a.e0, upwell=a.upwell,
              circ=a.circ, w_flow=a.w_flow)
    workers = a.workers if a.workers >= 0 else (len(cal) if (a.device or "cuda").startswith("cuda") else 0)
    pours = {p: Pour(p, kw, process=workers > 0) for p in cal}
    log(f"{len(cal)} calibration pours, {'one worker process each' if workers > 0 else 'in-process'}")
    base = dict(pours[cal[0]].s1, **({"foot_a": 0.0, "foot_b": 0.0} if a.foot else {}),
                **(CIRC_START if a.circ else {}), **(DEP_START if a.dep else {}))
    if a.init:
        base.update({k: float(v) for k, v in json.load(open(a.init)).items() if k in base})
        log(f"start: {a.init}")
    for k, v in (x.split("=") for x in a.set.split(",") if x):
        base[k] = float(v)
    # the closure is the single source of the model switches: the render (real_program) reads them back
    base["e0"], base["upwell"], base["circ"] = float(a.e0), float(a.upwell), float(a.circ)
    if not a.circ and any(base.get(k, 0.0) for k in ("kappa_M", "kappa_w")):
        log("no --circ: kappa_M, kappa_w of the start closure set to 0")
    if not a.circ:
        for k in ("kappa_M", "eps_f", "kappa_w"):
            base[k] = 0.0
    if not a.dep:
        base["tau_s"] = base["mu_s"] = 0.0
    u = to_unit(base)
    m1, m2 = np.zeros_like(u), np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr = a.lr
    recs, best = [], None
    fn = os.path.join(a.out, "records.jsonl")
    open(fn, "w").close()
    it = 0
    while it < a.iters:
        tau = a.tau0 * (a.tau1 / a.tau0) ** (it / max(a.iters - 1, 1))
        P = from_unit(u, base)
        t0 = time.time()
        try:
            for po in pours.values():                     # all pours run concurrently in their workers
                po.send(P, tau)
            res, timing, errs = {}, {}, []
            for p, po in pours.items():                   # collect every answer (keeps the pipes in step)
                try:
                    r = po.recv()
                except RuntimeError as e:
                    errs.append(str(e))
                    continue
                res[p] = (r["L"], r["g"], r["M"], r["c"], r["strip"], r["vmetrics"])
                timing[p] = r["timing"]
            if errs:
                raise RuntimeError("; ".join(errs))
        except RuntimeError as e:                       # non-finite state: back off
            log(f"it {it}: forward failed ({e}); halving the step")
            lr *= 0.5
            u = u_prev
            continue
        L = float(np.mean([r[0] for r in res.values()]))
        gth = np.mean([[r[1][k] for k in KEYS] for r in res.values()], axis=0)
        gu = gth * dtheta_du(P)
        J = float(np.mean([r[2]["J"] for r in res.values()]))
        vI = float(np.mean([r[2]["viou"] for r in res.values()]))
        rec = dict(it=it, loss=L, J=J, viou=vI, tau=tau, lr=lr, u=u.tolist(), params={k: P[k] for k in KEYS}, grad_u=gu.tolist(),
                   per={p: dict(loss=r[0], viou=r[2]["viou"], iou=r[2]["iou"], gap_iou=r[2]["gap_iou"], J=r[2]["J"],
                                video_iou=r[5]) for p, r in res.items()},
                   wall_s=time.time() - t0)
        recs.append(rec)
        with open(fn, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"it {it:3d}  loss {L:.4f}  vIoU {vI:.3f}  final J {J:.3f}  "
            + "  ".join(f"{p} v{r[2]['viou']:.2f} f{r[2]['iou']:.2f}/{r[2]['gap_iou']:.2f}" for p, r in res.items())
            + f"  |g| {np.linalg.norm(gu):.2e}  {rec['wall_s']:.0f} s ("
            + ", ".join(f"{p} fwd {t['forward_s']:.0f} + bwd {t['backward_s']:.0f} s" for p, t in timing.items()) + ")")
        def save(tag):
            for p, r in res.items():
                np.save(os.path.join(a.out, "fields", f"{p}_{tag}.npy"), r[3].astype(np.float32))
                if r[4]:
                    np.savez_compressed(os.path.join(a.out, "fields", f"{p}_{tag}_video.npz"), t=[x[0] for x in r[4]],
                                        c=np.array([x[1] for x in r[4]]), w=np.array([x[2] for x in r[4]]),
                                        v=np.array([x[3] for x in r[4]]))
        if it == 0:
            save("start")
        # best iterate: video loss -> highest whole-video IoU (the loss itself changes with the annealed tau, so its
        # values are not comparable across iterations); final loss -> highest final J
        better = best is None or ((vI > best["viou"]) if a.loss.startswith("video") else (J > best["J"]))
        if best is None or better:
            best = rec
            save("best")
            json.dump(dict(P), open(os.path.join(a.out, "best_closure.json"), "w"), indent=1)
        # Adam (descent on the loss) in the unit box
        m1 = b1 * m1 + (1 - b1) * gu
        m2 = b2 * m2 + (1 - b2) * gu ** 2
        mh, vh = m1 / (1 - b1 ** (it + 1)), m2 / (1 - b2 ** (it + 1))
        u_prev = u.copy()
        u = np.clip(u - lr * mh / (np.sqrt(vh) + eps), 0.0, 1.0)
        it += 1
        progress(a.out, recs)
    for po in pours.values():
        po.close()
    sheet(a.out, cal, log)
    # frozen-parameter test on the other patterns (and the calibration pours again, for the same table)
    from ..rig.real_program import PATTERNS
    Pbest = {**base, **json.load(open(os.path.join(a.out, "best_closure.json")))}
    tests = [p for p in PATTERNS if p not in cal]
    if tests and not a.no_test:
        log(f"frozen-parameter test on {tests} ...")
        tp = {p: Pour(p, dict(kw, tau=a.tau1), process=workers > 0) for p in tests}
        for po in tp.values():
            po.send_eval(Pbest)
        out = {}
        for p, po in tp.items():
            try:
                r = po.recv()
                out[p] = dict(loss=r["L"], viou=r["M"]["viou"], iou=r["M"]["iou"], gap_iou=r["M"]["gap_iou"],
                              J=r["M"]["J"], video_iou=r["vmetrics"])
                log(f"  test {p:10s} video loss {r['L']:.4f}  video IoU {r['M']['viou']:.3f}  final IoU {r['M']['iou']:.3f} "
                    f"gap {r['M']['gap_iou']:.3f}")
            except RuntimeError as e:
                out[p] = dict(error=str(e))
                log(f"  test {p}: failed ({e})")
            po.close()
        json.dump(dict(cal=cal, params=Pbest, calibration=best["per"], test=out), open(os.path.join(a.out, "test.json"), "w"),
                  indent=1, default=float)
    video_strip(a.out, cal)
    log(f"best: iteration {best['it']}, loss {best['loss']:.4f} (start {recs[0]['loss']:.4f}), video IoU "
        f"{best['viou']:.3f} (start {recs[0]['viou']:.3f}), final J {best['J']:.3f} (start {recs[0]['J']:.3f}); "
        + " ".join(f"{k} {best['params'][k]:.4g}" for k in KEYS))
    log(f"-> {a.out}/best_closure.json, progress.png, sheet.png, video_strip.png")


if __name__ == "__main__":
    main()
