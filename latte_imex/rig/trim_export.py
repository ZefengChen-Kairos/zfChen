"""Trim a take to [t1, t2] (s after the first frame of the take) and export: color/<cam>/<step>.png (original bytes),
depth/<cam>/<step>.png (uint16 mm, lossless PNG), metadata.json (original fields restricted to the kept frames + cut
info), calibrate.pkl (original bytes)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from rs_zip import Rec

src, outdir, t1, t2 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
r = Rec(src)
os.makedirs(outdir, exist_ok=True)
meta = {k: v for k, v in r.meta.items() if k != "recording"}
meta["recording"] = {}
n = 0
for c in range(r.ncam):
    os.makedirs(f"{outdir}/color/{c}", exist_ok=True); os.makedirs(f"{outdir}/depth/{c}", exist_ok=True)
    meta["recording"][str(c)] = {}
    T = r.ts[c] - r.t0
    for k, t, ta in zip(r.steps[c], T, r.ts[c]):
        if t1 <= t <= t2:
            open(f"{outdir}/color/{c}/{k}.png", "wb").write(r.z.read(f"{r.root}/color/{c}/{k}.png"))
            d = r.depth(c, k)
            assert d.dtype == np.uint16
            cv2.imwrite(f"{outdir}/depth/{c}/{k}.png", d)
            meta["recording"][str(c)][str(k)] = float(ta)
            n += 1
meta["cut"] = dict(source=os.path.basename(src), t_first_frame_unix=r.t0, start_s=t1, end_s=t2,
                   start_unix=r.t0 + t1, end_unix=r.t0 + t2, depth_format="uint16 PNG, millimetres (D435 depth scale 0.001 m)",
                   color_format="original PNG, BGR as recorded")
json.dump(meta, open(f"{outdir}/metadata.json", "w"))
open(f"{outdir}/calibrate.pkl", "wb").write(r.z.read(f"{r.root}/calibrate.pkl"))
print(os.path.basename(outdir), "frames", n, {c: len(meta["recording"][str(c)]) for c in range(r.ncam)})
