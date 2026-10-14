"""CUDA benchmarks.

    python -m latte_imex.bench_cuda --quick [--N 256] [--T 1.0] [--B 1,8,32]
        heart (V0.5 controls) for T seconds: single run and batches of B copies; wall time, sub-steps per second,
        launches and host syncs per sub-step, member-sub-steps per second (the batched throughput)
    python -m latte_imex.bench_cuda --N 768 --out runs/cuda768 [--video]      (default mode, --six)
        six patterns with the V0.5 controls (as before): wall times, step counts, IoU vs CPU-256
"""
import argparse, json, os, time
import numpy as np


def quick(N=256, T=1.0, Bs=(1, 8, 32), out=None):
    from .run import V05Script
    from .batch_run import run_batch
    rows = []
    out = out or f"runs/bench_cuda_N{N}"
    for B in Bs:
        jobs = [dict(script=V05Script("heart"), pattern="heart", out=os.path.join(out, f"B{B}", f"m{b}"), T_max=T)
                for b in range(B)]
        t0 = time.time()
        mets = run_batch(jobs, N=N, backend="cuda", snaps=False, quiet=True, save_state=False)
        wall = time.time() - t0
        st = mets[0]["stats"]
        steps = sum(m["stats"]["steps"] for m in mets)
        r = dict(N=N, B=B, T=T, wall_s=wall, substeps_member0=st["steps"], frames=st["frames"],
                 launches_per_substep=st["launches"] / max(st["steps"], 1), syncs_per_substep=st["host_syncs"] / max(st["steps"], 1),
                 cg_mass_per_step=st["cg_mass"] / max(st["steps"], 1), cg_visc_per_step=st["cg_visc"] / max(st["steps"], 1),
                 member_substeps_per_s=steps / wall, ms_per_substep_round=1e3 * wall / max(st["steps"], 1))
        rows.append(r)
        print(f"N={N} B={B:3d}: wall {wall:6.1f} s  {r['ms_per_substep_round']:.2f} ms per sub-step round  "
              f"{r['member_substeps_per_s']:.0f} member-sub-steps/s  launches/sub-step {r['launches_per_substep']:.1f}  "
              f"syncs/sub-step {r['syncs_per_substep']:.2f}  cg {r['cg_mass_per_step']:.1f}+{r['cg_visc_per_step']:.1f}", flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump(rows, open(os.path.join(out, "bench_quick.json"), "w"), indent=1)
    return rows


def six(a):
    from .run import run
    from . import finalize
    out = a.out or f"runs/cuda{a.N}"
    os.makedirs(out, exist_ok=True)
    rows = []
    for p in a.patterns.split(","):
        t0 = time.time()
        sol, met = run(p, N=a.N, out=os.path.join(out, p), control="v05", video=a.video, quiet=False, backend="cuda")
        wall = time.time() - t0
        st = met["stats"]
        iou = float("nan")
        ref = os.path.join(a.ref, p, "final.png")
        if os.path.exists(os.path.join(a.ref, p, "metrics.json")):
            try:
                from PIL import Image
                A = np.asarray(Image.open(ref).convert("L"), float) / 255.
                B = np.asarray(Image.open(os.path.join(out, p, "final.png")).convert("L"), float) / 255.
                n = A.shape[0]; y, x = np.mgrid[:n, :n]; m = np.hypot(x - n / 2, y - n / 2) <= 0.46 * n
                a_, b_ = (A > 0.72) & m, (B > 0.72) & m
                iou = float((a_ & b_).sum() / max((a_ | b_).sum(), 1))
            except Exception:
                pass
        rows.append(dict(pattern=p, N=a.N, T=met["T"], wall_s=wall, solver_s=met["elapsed_s"], steps=st["steps"],
                         cg_per_step=st["cg_mass"] / max(st["steps"], 1), max_sub=st["max_substeps"], retries=st["retries"],
                         launches_per_step=st["launches"] / max(st["steps"], 1),
                         layer_err=met["ledger"]["layer_error"], milk_err=met["ledger"]["milk_error"], iou_vs_cpu256=iou))
        print(f"{p:14s} N={a.N} T={met['T']:.2f}s  solver {met['elapsed_s']:.1f}s  (with I/O {wall:.1f}s)  steps {st['steps']}  "
              f"cg/step {rows[-1]['cg_per_step']:.1f}  launches/step {rows[-1]['launches_per_step']:.1f}  retries {st['retries']}  "
              f"lerr {rows[-1]['layer_err']:.1e}  IoU vs CPU256 {iou:.3f}", flush=True)
    json.dump(rows, open(os.path.join(out, "bench.json"), "w"), indent=1)
    finalize.main(out, f" (V0.5 controls, IMEX CUDA {a.N})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--six", action="store_true")
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--B", default="1,8,32")
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--patterns", default="heart,push_heart,layered_heart,tulip,leaf,swan")
    ap.add_argument("--ref", default="results/v05_on_imex_256", help="CPU-256 results for the IoU column (optional)")
    a = ap.parse_args()
    if a.quick:
        quick(a.N or 256, a.T, [int(b) for b in a.B.split(",")], a.out)
    else:                                     # default: the six-pattern benchmark (previous behaviour)
        a.N = a.N or 768
        six(a)


if __name__ == "__main__":
    main()
