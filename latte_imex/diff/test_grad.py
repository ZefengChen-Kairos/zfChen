"""Gradient check: the adjoint dL/dtheta of the Warp solver against central finite differences of the same forward
(sub-step counts frozen, so both differentiate the same discrete map).
    python -m latte_imex.diff.test_grad [--pattern heart --N 32 --T 1.5 --eps 1e-5]"""
import argparse

import numpy as np

from . import wsolver as W
from .run import DiffPour


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="heart")
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--T", type=float, default=1.5, help="pour seconds simulated")
    ap.add_argument("--eps", type=float, default=1e-5, help="relative FD step")
    ap.add_argument("--keys", default=",".join(W.PARAMS) + ",foot_a,foot_b")
    ap.add_argument("--sched", type=int, default=2, help="also check this many schedule entries (x, Q of active frames)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--scales", default="0", help="video loss scales (run.DiffPour)")
    ap.add_argument("--upwell", action="store_true", help="volume-conserving surface source (wsolver.k_fields)")
    ap.add_argument("--w-flow", dest="w_flow", type=float, default=0.0)
    ap.add_argument("--loss", default="video")
    ap.add_argument("--circ", default="", help="kappa_o,kappa_s[,kappa_w]: circulation (circulation.py)")
    a = ap.parse_args()
    D = DiffPour(a.pattern, N=a.N, device=a.device, T_max=a.T, tau=0.1, scales=[float(x) for x in a.scales.split(",")],
                 upwell=a.upwell, circ=bool(a.circ), w_flow=a.w_flow,
                 loss=a.loss)
    P = dict(D.s1, foot_a=0.05, foot_b=-0.02)
    if a.circ:
        for k, x in zip(("kappa_o", "kappa_s", "kappa_w"), a.circ.split(",")):
            P[k] = float(x)
    L, g, _ = D.loss_grad(P)
    ns = D.nsub
    print(f"loss {L:.6f}; forward {D.timing['forward_s']:.1f} s, backward {D.timing['backward_s']:.1f} s; "
          f"PCG iterations (forward + adjoint) {D.sol.cg_iterations()}")
    worst = 0.0
    for k in a.keys.split(","):
        h = a.eps * max(abs(P[k]), 1e-2 if k.startswith("foot") else 1e-8)
        Lp = D.loss(dict(P, **{k: P[k] + h}), nsub=ns)[0]
        Lm = D.loss(dict(P, **{k: P[k] - h}), nsub=ns)[0]
        fd = (Lp - Lm) / (2 * h)
        rel = abs(g[k] - fd) / max(abs(fd), abs(g[k]), 1e-12)
        if abs(fd) > 1e-9 or abs(g[k]) > 1e-9:
            worst = max(worst, rel)
        print(f"  {k:8s} adjoint {g[k]: .6e}  FD {fd: .6e}  rel diff {rel:.1e}")
    # schedule entries (per-frame arrival record): the hit point x and the flow Q of a few pouring frames
    D._apply_foot(P)            # the parameter loop left the last perturbed landing-point shift applied
    act = np.where(D.S["active"])[0]
    for kk in act[len(act) // 2: len(act) // 2 + a.sched]:
        for key in ("x", "Q"):
            ga = D.sched_grad[key][kk]
            base = D.Sw[key].numpy().copy()        # (x, y: already shifted by the landing-point correction)
            h = a.eps * max(abs(base[kk]), 1e-3)
            vals = []
            for sgn in (1, -1):
                arr = base.copy(); arr[kk] += sgn * h
                D.Sw[key].assign(arr)              # in place: captured graphs hold these addresses
                vals.append(D.loss(P, nsub=ns)[0])
            D.Sw[key].assign(base)
            D._foot = (float(P["foot_a"]), float(P["foot_b"]))
            fd = (vals[0] - vals[1]) / (2 * h)
            rel = abs(ga - fd) / max(abs(fd), abs(ga), 1e-12)
            if abs(fd) > 1e-9 or abs(ga) > 1e-9:
                worst = max(worst, rel)
            print(f"  frame {kk} {key:2s}   adjoint {ga: .6e}  FD {fd: .6e}  rel diff {rel:.1e}")
    print("PASS" if worst < 1e-3 else f"CHECK (worst relative difference {worst:.1e})")


if __name__ == "__main__":
    main()
