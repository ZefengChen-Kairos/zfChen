"""Six patterns with the V0.5 controls on the CUDA back-end at a given N; wall times, step counts, IoU vs CPU-256.

    python -m latte_imex.bench_cuda --N 768 --out runs/cuda768 [--video]
"""
import argparse, json, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=768)
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--patterns", default="heart,push_heart,layered_heart,tulip,leaf,swan")
    ap.add_argument("--ref", default="results/v05_on_imex_256", help="CPU-256 results for the IoU column (optional)")
    a = ap.parse_args()
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
                         layer_err=met["ledger"]["layer_error"], milk_err=met["ledger"]["milk_error"], iou_vs_cpu256=iou))
        print(f"{p:14s} N={a.N} T={met['T']:.2f}s  solver {met['elapsed_s']:.1f}s  (with I/O {wall:.1f}s)  steps {st['steps']}  "
              f"cg/step {rows[-1]['cg_per_step']:.1f}  retries {st['retries']}  lerr {rows[-1]['layer_err']:.1e}  IoU vs CPU256 {iou:.3f}", flush=True)
    json.dump(rows, open(os.path.join(out, "bench.json"), "w"), indent=1)
    finalize.main(out, f" (V0.5 controls, IMEX CUDA {a.N})")


if __name__ == "__main__":
    main()
