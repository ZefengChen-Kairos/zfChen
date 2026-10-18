"""Timing of the differentiable solver on one whole pour: fast forward (CUDA graphs), forward + backward (loss_grad).
    python -m latte_imex.diff.bench [--pattern tulip --N 128 --device cuda:0]"""
import argparse
import time

import warp as wp

from .run import DiffPour


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="tulip")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--T", type=float, default=None)
    a = ap.parse_args()
    D = DiffPour(a.pattern, N=a.N, device=a.device, T_max=a.T)
    P = dict(D.s1)
    t0 = time.time()
    L, _, ns = D.loss(P)                                    # warm-up: kernel compilation + graph capture
    wp.synchronize()
    t_warm = time.time() - t0
    t0 = time.time()
    L, _, ns = D.loss(P)
    wp.synchronize()
    t_fwd = time.time() - t0
    nsub = sum(ns)
    D.loss_grad(P)                                          # warm-up of the recorded path
    t0 = time.time()
    D.loss_grad(P)
    wp.synchronize()
    t_fb = time.time() - t0
    sol = D.sol
    print(f"{a.pattern} N={a.N} on {D.geo.dev}: {D.S['n']} frames, {nsub} sub-steps; PCG modes {sol.opM.mode}/{sol.opV.mode}, "
          f"fused sub-step graph {sol.fused}, graphs {sol.runner.use} ({len(sol.runner.graphs)} captured)")
    print(f"  first forward (compile + capture) {t_warm:.1f} s")
    print(f"  forward (loss only)          {t_fwd:6.2f} s  = {1e6 * t_fwd / nsub:7.0f} us per sub-step")
    print(f"  forward + backward (gradient) {t_fb:6.2f} s  = {1e6 * t_fb / nsub:7.0f} us per sub-step "
          f"(forward {D.timing['forward_s']:.1f} s, backward {D.timing['backward_s']:.1f} s)")


if __name__ == "__main__":
    main()
