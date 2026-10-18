"""INRIA 3DGS point_cloud.ply -> compact .splat (antimatter15 / SuperSplat format, 32 bytes per Gaussian: position
3 x f32, scale 3 x f32, RGBA 4 x u8 (SH degree 0 colour, sigmoid opacity), rotation 4 x u8), sorted by size x opacity;
the view-dependent SH bands are dropped (about 8x smaller).  --min-opacity drops nearly transparent Gaussians.
    python ply2splat.py out/milk_cup_cad/point_cloud.ply [out.splat] [--min-opacity 0.02]"""
import sys

import numpy as np


def read_ply(fn):
    with open(fn, "rb") as f:
        names, n = [], 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(n * 4 * len(names)), dtype="<f4").reshape(n, len(names))
    return {k: data[:, i] for i, k in enumerate(names)}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mo = float(sys.argv[sys.argv.index("--min-opacity") + 1]) if "--min-opacity" in sys.argv else 0.0
    src = args[0]
    dst = args[1] if len(args) > 1 and not args[1].replace(".", "").isdigit() else src.rsplit(".", 1)[0] + ".splat"
    v = read_ply(src)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True) + 1e-12
    op = 1.0 / (1.0 + np.exp(-v["opacity"]))
    C0 = 0.28209479177387814
    rgb = 0.5 + C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    keep = op >= mo
    order = np.argsort(-(scale.prod(1) * op)[keep])
    xyz, scale, rot, op, rgb = (a[keep][order] for a in (xyz, scale, rot, op, rgb))
    rec = np.zeros(len(xyz), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"], rec["s"] = xyz, scale
    rec["c"] = np.clip(np.c_[rgb, op] * 255, 0, 255).astype(np.uint8)
    rec["r"] = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)
    rec.tofile(dst)
    print(f"{src}: {len(v['x'])} Gaussians -> {dst}: {len(rec)} ({rec.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
