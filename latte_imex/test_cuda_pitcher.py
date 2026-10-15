"""GPU forward pitcher (cuda_pitcher.GPUForwardPitcher) against the CPU ForwardPitcher on recorded S1 poses.

    python -m latte_imex.test_cuda_pitcher                          # GPU: full NX geometry, 3 patterns x 2 windows x 1500 steps
    NUMBA_ENABLE_CUDASIM=1 python -m latte_imex.test_cuda_pitcher --nr 1 --steps 24 --patterns heart
                                                                    # no GPU: simulator, coarse cavity, few steps (slow)
Pass: every emission quantity agrees to 1e-6 relative of its scale (free-surface height to 1e-9 m) and the binned
arrival records to 1e-6 relative.  The histogram of the cavity volumes uses shared-memory float atomics, so the
summation order (and the last bits) differ from the CPU; near a threshold (weir head ~ 0, quantile bin edge) that
rounding is amplified to ~1e-7 relative (seen on the RTX 4090 for one tulip window: z_f 8e-11 m) -- physically nil.
"""
import argparse
import math
import sys
import time

import numpy as np

from .pitcher import ForwardLaw, ForwardPitcher, GridPitcherGeometry


def windows(P, meta, steps):
    """Two pose windows: one from just before the first stroke (the lip wets), one across the end of the last flowing
    stroke (flow stops, dry lip, the tilt-back)."""
    n = len(P["t"])
    i0 = int(np.searchsorted(P["t"], -0.05))
    tend = meta["events"][max(meta["flowing"])]
    i1 = int(np.searchsorted(P["t"], tend)) - steps // 2
    return [min(max(i, 0), n - steps) for i in (i0, i1)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", default="heart,tulip,swan")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--nr", type=int, default=6)
    ap.add_argument("--variants", default="fwd,arc,hbar")
    a = ap.parse_args(argv)
    from .s1_forward import load_poses, load_meta
    from .cuda_pitcher import GPUForwardPitcher, compare
    laws = dict(fwd=ForwardLaw(), arc=ForwardLaw(width="arc"), hbar=ForwardLaw(U_head="bar"), hk=ForwardLaw(h_k=0.8e-3),
                local=ForwardLaw(U_head="local"),
                inertia=ForwardLaw(inertia=True))
    ok = True
    worst = 0.0
    for var in a.variants.split(","):
        seqs, cpu = [], []
        geom = None
        for p in a.patterns.split(","):
            P = load_poses(p)
            meta = load_meta(p)
            if geom is None:
                geom = GridPitcherGeometry.load(nr=a.nr, lip_slope=math.radians(float(P["alpha_deg"])))
                fp = ForwardPitcher(geom, laws[var])
            for s0 in windows(P, meta, a.steps):
                sl = slice(s0, s0 + a.steps)
                te = P["t"][sl] - 0.5 * float(P["dt"])
                V0 = float(P["fill_mL"]) * 1e-6
                seqs.append(dict(t=te, tip=P["tip"][sl], yaw=P["yaw"][sl], tilt=P["tilt"][sl], roll=P["roll"][sl], V0=V0,
                                 dt=float(P["dt"])))
        t0 = time.time()
        for s in seqs:
            cpu.append(fp.run(s["t"], s["tip"], s["yaw"], s["tilt"], s["roll"], s["V0"], s["dt"]))
        t1 = time.time()
        G = GPUForwardPitcher(fp)
        gpu = G.run_batch(seqs)
        t2 = time.time()
        for s, Ec, Eg in zip(seqs, cpu, gpu):
            d = compare(Ec, Eg)
            rel = {k: v[0] / max(v[1], 1e-30) for k, v in d.items()}
            r = max(v for k, v in rel.items() if k != "z_f")
            dz = d["z_f"][0]
            Rc = ForwardPitcher.arrivals(Ec, float(s["t"][0]), len(s["t"]), s["dt"])
            Rg = ForwardPitcher.arrivals(Eg, float(s["t"][0]), len(s["t"]), s["dt"])
            ra = max(float(np.max(np.abs(Rc[k] - Rg[k]))) / max(float(np.max(np.abs(Rc[k]))), 1e-30)
                     for k in ("Q", "x_hit", "U_perp", "V_cup"))
            flow = float(np.sum(Ec["Q_o"]) * s["dt"] * 1e6)
            good = r < 1e-6 and dz < 1e-9 and ra < 1e-6 and Eg["gpu_status"] == 0
            worst = max(worst, r, ra)
            ok &= good
            print(f"[{var}] n={len(s['t'])} poured {flow:.2f} mL  max rel diff emission {r:.1e} (z_f {dz:.1e} m)  "
                  f"arrivals {ra:.1e}  status {Eg['gpu_status']}  {'ok' if good else 'FAIL'}", flush=True)
        print(f"[{var}] cpu {t1 - t0:.1f}s for {len(seqs)} sequences, gpu batch {t2 - t1:.1f}s", flush=True)
    print(f"worst relative difference {worst:.1e}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
