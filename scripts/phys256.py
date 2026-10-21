"""update19: forward runs of fixed closures at a fine grid on the five pours, scored on the calibration's scale
(render512.score48: J = 0.25 viou_n + 0.75 struct_n at the N = 48 equivalent), with movies next to a reference
closure run at the same grid, and the white fraction c at the observed frames on the 128 px video grid (uint8) so the
kinematic checks can be computed offline through the same pipeline as the video.

    python3 scripts/phys256.py --out DIR --N 256 --ref closures/bp256_trained.json:"反传训练 N=256" \
        --run latte_imex/data/closures/mcd_S1_N96.json:"物理版 第一阶段" --run ...  [--pours ...] [--device cuda:0]
        [--T seconds] [--no-movie]
-> DIR/<tag>/eval.json, DIR/<tag>/<p>_simN.mp4, DIR/<tag>/<p>_compare.mp4 (real | target | closure | reference),
   DIR/<tag>/<p>_c128.npz (t, c: uint8 255 = white, observed frames), DIR/summary.txt.
A pour whose c128 file exists is skipped, so the command resumes after an interruption.  A run whose forward fails
(non-finite state, e.g. the Stage-2 column sink emptying the layer under the column in the tail at fine grids, the
known limit noted in wsolver.SINK_FORMS) is recorded as failed (eval.json 'error', <p>_failed.txt) and the others go on."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from latte_imex.diff import render512 as RR  # noqa: E402

PATTERNS = ("tulip", "leaf", "heart", "push_heart", "swan")


def spec(s):
    fn, _, lab = s.partition(":")
    tag = os.path.splitext(os.path.basename(fn))[0]
    return fn, lab or tag, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--run", action="append", required=True, help="closure.json[:label]; repeatable")
    ap.add_argument("--ref", default=None, help="closure.json[:label] shown as the 4th panel, run at the same N")
    ap.add_argument("--pours", default=",".join(PATTERNS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--no-movie", dest="movie", action="store_false")
    a = ap.parse_args()
    runs = [spec(s) for s in a.run]
    ref = spec(a.ref) if a.ref else None
    for p in a.pours.split(","):
        todo = [r for r in runs if not any(os.path.exists(os.path.join(a.out, r[2], f"{p}_{x}"))
                                           for x in ("c128.npz", "failed.txt"))]
        if not todo:
            print(f"{p}: done before, skipped", flush=True)
            continue
        base = None
        if ref is not None and a.movie:
            t0 = time.time()
            num, sw = RR.load_closure(ref[0])
            Db = RR.make_pour(p, sw, a.N, a.device, a.T)
            try:
                bt, bC, _ = RR.run_pour(Db, num, sw, a.every)
                base = (bt, bC, Db.geo.mask_np, a.N)
                print(f"{p}: reference {ref[1]} at N {a.N} ({time.time() - t0:.0f} s)", flush=True)
            except RuntimeError as e:
                print(f"{p}: reference {ref[1]} FAILED ({e}); movies without it", flush=True)
            del Db
        for fn, lab, tag in todo:
            out = os.path.join(a.out, tag)
            os.makedirs(out, exist_ok=True)
            t0 = time.time()
            num, sw = RR.load_closure(fn)
            D = RR.make_pour(p, sw, a.N, a.device, a.T)
            efn = os.path.join(out, "eval.json")
            try:
                t_end, C, ML = RR.run_pour(D, num, sw, a.every)
            except RuntimeError as e:
                k = int(str(e).split("frame ")[1].split()[0]) if "frame " in str(e) else -1
                tf = float(np.cumsum(D.S["fdt"])[k]) if 0 <= k < len(D.S["fdt"]) else float("nan")
                print(f"{p}: {lab}  N {a.N}  FAILED at t = {tf:.2f} s ({e}); recorded, going on", flush=True)
                res = json.load(open(efn)) if os.path.exists(efn) else {}
                res["_meta"] = dict(closure=os.path.abspath(fn), label=lab, N=a.N, T=a.T)
                res[p] = dict(error=str(e), failed_t=tf)
                json.dump(res, open(efn, "w"), indent=1, default=float)
                open(os.path.join(out, f"{p}_failed.txt"), "w").write(f"{e}\nt = {tf:.3f} s\n")
                del D
                continue
            t1 = time.time()
            r = RR.score48(p, sw, D, ML, a.T)
            res = json.load(open(efn)) if os.path.exists(efn) else {}
            res["_meta"] = dict(closure=os.path.abspath(fn), label=lab, N=a.N, T=a.T)
            res[p] = dict(r, forward_s=t1 - t0)
            json.dump(res, open(efn, "w"), indent=1, default=float)
            print(f"{p}: {lab}  N {a.N}  J {r['J']:.3f}  IoU {r['viou']:.3f} ({r['viou_n']:.2f})  structure "
                  f"{r['struct']:.3f} ({r['struct_n']:.2f})  forward {t1 - t0:.0f} s", flush=True)
            if a.movie:
                RR.movies(p, out, (t_end, C, D.geo.mask_np, a.N), base, lab, ref[1] if ref else "", a.every)
            msk = D.geo.mask_np
            tobs = np.array([o["t"] for o in D.obs])
            c128 = np.array([np.clip(RR.to_H(C[int(o["k"])], msk, 128), 0, 1) for o in D.obs])
            np.savez_compressed(os.path.join(out, f"{p}_c128.npz"), t=tobs, c=(c128 * 255 + 0.5).astype(np.uint8))
            print(f"{p}: {lab} written ({time.time() - t1:.0f} s)", flush=True)
            del D, C, ML
    # summary over every run directory
    lines = [f"J = 0.25 IoU_n + 0.75 structure_n at the N = 48 equivalent; N {a.N}", f"{'run':28s} " +
             " ".join(f"{p:>10s}" for p in PATTERNS) + "   cal mean  held-out mean"]
    for fn, lab, tag in runs:
        efn = os.path.join(a.out, tag, "eval.json")
        if not os.path.exists(efn):
            continue
        res = json.load(open(efn))
        J = {p: res[p]["J"] for p in PATTERNS if "J" in res.get(p, {})}
        bad = [f"{p} (t {res[p].get('failed_t', float('nan')):.1f} s)" for p in PATTERNS if "error" in res.get(p, {})]
        cal = [J[p] for p in ("tulip", "leaf") if p in J]
        tst = [J[p] for p in ("heart", "push_heart", "swan") if p in J]
        lines.append(f"{tag:28s} " + " ".join(f"{J[p]:10.3f}" if p in J else f"{'-':>10s}" for p in PATTERNS)
                     + f"   {np.mean(cal) if cal else float('nan'):8.3f}  {np.mean(tst) if tst else float('nan'):13.3f}"
                     + (f"   failed: {', '.join(bad)}" if bad else ""))
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
