"""Gradient check: the adjoint dL/dtheta of the Warp solver against central finite differences of the same forward
(sub-step counts frozen, so both differentiate the same discrete map).
    python -m latte_imex.diff.test_grad [--pattern heart --N 32 --T 1.5 --eps 1e-5]
                                        [--set xi_b=0.7,lam_b=5 --keys xi_b,lam_b,cp]   (two-phase open layer)
                                        [--milk mp5fct]   (milk transport scheme, wsolver.MILK_SCHEMES)
                                        [--sink column --set kappa_e=0.032]   (entrainment sink form, SINK_FORMS)
A non-finite adjoint or FD value fails the check.  With the FCT milk schemes (mp5fct, thincfct, mp5fct_nb) the loss is
only piecewise smooth (the limiter's min / max switches: 17 kinks within +-2e-5 relative of kappa_Q on tulip N 32): an
FD step of 1e-5 straddles kinks (kappa_Q 1e-3 - 6e-3 off at N 32, 5e-2 at N 48); --eps 1e-7 compares the local
derivative (<= 1e-6 at N 32).  fit.MILK_GRAD: which schemes the gradient fit takes."""
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
    ap.add_argument("--circ", default="", help="kappa_M,eps_f[,kappa_w]: surface current (circulation.py)")
    ap.add_argument("--set", default="", help="closure values, e.g. tau_s=0.05")
    ap.add_argument("--milk", default="vl", choices=W.MILK_SCHEMES, help="face value of c in the milk transport")
    ap.add_argument("--sink", default="v05", choices=W.SINK_FORMS, help="form of the entrainment sink's area flux")
    a = ap.parse_args()
    D = DiffPour(a.pattern, N=a.N, device=a.device, T_max=a.T, tau=0.1, scales=[float(x) for x in a.scales.split(",")],
                 upwell=a.upwell, circ=bool(a.circ), w_flow=a.w_flow,
                 loss=a.loss, milk=a.milk, sink=a.sink)
    P = dict(D.s1, foot_a=0.05, foot_b=-0.02)
    P.update({k: float(v) for k, v in (x.split("=") for x in a.set.split(",") if x)})
    if a.circ:
        for k, x in zip(("kappa_M", "eps_f", "kappa_w"), a.circ.split(",")):
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
        # with the current on, the landing point also moves the current (host fields, not differentiated): partial
        partial = bool(a.circ) and k.startswith("foot")
        # kappa_e, lam_b = 0 without the current: the exchange step is switched off (WSolver.set_ex), the FD crosses
        # the switch; check them at a value > 0 (--set lam_b=5,xi_b=0.7: the two-phase open layer)
        gate = k in ("kappa_e", "lam_b") and P[k] == 0.0 and not a.circ
        if not (np.isfinite(g[k]) and np.isfinite(fd)):
            worst = float("inf")                  # (max(worst, nan) would keep worst)
        elif (abs(fd) > 1e-9 or abs(g[k]) > 1e-9) and not partial and not gate:
            worst = max(worst, rel)
        print(f"  {k:8s} adjoint {g[k]: .6e}  FD {fd: .6e}  rel diff {rel:.1e}"
              + ("  (partial: the current's dependence on the landing point is not differentiated)" if partial else "")
              + ("  (not compared: the exchange step is off at 0)" if gate else ""))
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
            if not (np.isfinite(ga) and np.isfinite(fd)):
                worst = float("inf")
            elif abs(fd) > 1e-9 or abs(ga) > 1e-9:
                worst = max(worst, rel)
            print(f"  frame {kk} {key:2s}   adjoint {ga: .6e}  FD {fd: .6e}  rel diff {rel:.1e}")
    print("PASS" if worst < 1e-3 else f"CHECK (worst relative difference {worst:.1e})")


if __name__ == "__main__":
    main()
