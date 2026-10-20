"""update17 self-test 2: the production scoring path on DEVICE against the Warp CPU path, over a whole pour.

One evaluation of the leaf term of the calibration objective exactly as the CMA-ES runs do it (cmafit._init +
cmafit.evaluate: DiffPour fast path with CUDA graphs, e0 world, score.set_grid(N), the video's own 0.25 s lag
reference) with the v05g start closure and every mechanism of the fit17 models on at the values the runs start from
(fit.MECH_START: entrainment sink kappa_e, V_j traction kappa_tc with vj = 1, kinematic deposition mu_s = 1, tau_s 0.08
+ landing crater e_F, whole-layer yield tau_b / tau_f), once on DEVICE and once on the CPU.  Both run the same Warp
kernels with the device-side PCG loop; they differ by the order of the device reductions (atomics, ~1e-16 relative)
and, only where the driver lacks conditional graph nodes (chunked PCG fallback), by the convergence-check interval
(DiffPour: 4 on CUDA, 1 on the CPU), so the scores agree far inside tol.  Unlike diff/test_parity (first 200 frames of
heart) this covers the whole pour, its deposition and yield phases included, on the path the fit uses.
    python scripts/gpu_cpu_check.py DEVICE INIT [--pour leaf] [--N 48] [--tol 0.02]
Prints both results and PASS / FAIL (FAIL: an error on either path, a non-finite score, or |d structure| or |d IoU|
> tol).  On DEVICE = cpu there is nothing to compare: one run, PASS when it scores without error."""
import argparse
import json
import math
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("device")
    ap.add_argument("init", help="start closure (json), as cmafit --init")
    ap.add_argument("--pour", default="leaf")
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--tol", type=float, default=0.02, help="largest accepted |d structure|, |d IoU| (raw scores)")
    a = ap.parse_args()
    from latte_imex.diff import cmafit as CF
    from latte_imex.diff import schedule as SC
    from latte_imex.diff.fit import MECH_START
    from latte_imex.diff.run import DiffPour
    # the closure as cmafit.main builds it: s1 defaults, no surface current, the start closure, the mechanism starts,
    # the switches of the model letters (vj = mu_s = 1) and tau_s = 0.08 (the runs' --set)
    P = dict(DiffPour(a.pour, N=16, device="cpu", log=lambda *x: None).s1)
    P.update(kappa_M=0.0, eps_f=0.0, kappa_w=0.0)
    P.update({k: float(v) for k, v in json.load(open(a.init)).items() if isinstance(v, (int, float)) and not k.startswith("_")})
    P.update(MECH_START)
    P.update(vj=1.0, mu_s=1.0, tau_s=0.08)
    print(f"{a.pour}, N {a.N}, e0 world; mechanisms " + " ".join(f"{k} {P[k]:g}" for k in
                                                                   ("kappa_e", "kappa_tc", "vj", "mu_s", "tau_s", "e_F", "tau_b", "tau_f")),
          flush=True)
    devs = [a.device] if a.device == "cpu" else [a.device, "cpu"]
    res = {}
    for dev in devs:
        CF._D.clear()
        t0 = time.time()
        CF._init(dict(pours=[a.pour], N=a.N, device=dev, e0=SC.e0_mode("world"), exit_cal=False, upwell=False, circ=False,
                      circ_shape=dict(L=0.4, r_f=0.06)))
        t1 = time.time()
        r = CF.evaluate(P)[a.pour]
        D = CF._D[a.pour]
        r["substeps"] = D.sol.stats.get("substeps")
        res[dev] = r
        print(f"[{dev}] init {t1 - t0:.0f} s, one evaluation {time.time() - t1:.0f} s, substeps {r['substeps']}: "
              f"structure {r['struct']:.5f} ({r.get('struct_n', 0):.3f} of the video's own)  video IoU {r['viou']:.5f} "
              f"({r.get('viou_n', 0):.3f})" + (f"  ERROR {r['err']}" if "err" in r else ""), flush=True)
    ok = all("err" not in r and math.isfinite(r["struct"]) and math.isfinite(r["viou"]) for r in res.values())
    if ok and len(devs) == 2:
        g, c = res[devs[0]], res["cpu"]
        ds, di = abs(g["struct"] - c["struct"]), abs(g["viou"] - c["viou"])
        print(f"|d structure| {ds:.2e}  |d IoU| {di:.2e}  (tol {a.tol:g})")
        ok = ds <= a.tol and di <= a.tol
    elif len(devs) == 1:
        print("device is the CPU: nothing to compare (one run)")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
