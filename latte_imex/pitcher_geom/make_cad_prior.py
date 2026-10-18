"""Geometric prior for the 3DGS trainer (gs_train.py --shape kona_cad) from the Kona 700 mL CAD
(pitcher_kona700.stl, mm, inside bottom on the axis at z = 0, spout +x, handle -x):
  pts, nrm  surface samples (area-uniform, ~0.8 mm spacing) and their normals
  dist      unsigned distance [mm] to the surface on a 1 mm grid (float16), origin lo, spacing 1 mm
  r_mid     mean body radius 25-60 % of the height, H rim height [mm]
    python -m latte_imex.pitcher_geom.make_cad_prior"""
import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree

D = os.path.dirname(os.path.abspath(__file__))


def main():
    m = trimesh.load(os.path.join(D, "pitcher_kona700.stl"))
    n = int(m.area / 0.8 ** 2 * 1.15)
    pts, fi = trimesh.sample.sample_surface_even(m, n, seed=0)
    nrm = m.face_normals[fi]
    dense, _ = trimesh.sample.sample_surface_even(m, int(m.area / 0.3 ** 2), seed=1)
    lo = np.floor(m.bounds[0]) - 8.0
    hi = np.ceil(m.bounds[1]) + 8.0
    ax = [np.arange(lo[k], hi[k] + 0.5, 1.0) for k in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    d, _ = cKDTree(dense).query(G, workers=-1)
    dist = d.reshape([len(a) for a in ax]).astype(np.float16)
    H = 122.0
    mid = (pts[:, 2] > 0.25 * H) & (pts[:, 2] < 0.6 * H) & (pts[:, 0] > -40) & (nrm[:, 0] * pts[:, 0] + nrm[:, 1] * pts[:, 1] > 0)
    r_mid = float(np.median(np.hypot(pts[mid, 0], pts[mid, 1])))
    fn = os.path.join(D, "pitcher_kona700_prior.npz")
    np.savez_compressed(fn, pts=pts.astype(np.float32), nrm=nrm.astype(np.float32), dist=dist, lo=lo, spacing=1.0,
                        r_mid=r_mid, H=H)
    print(f"{fn}: {len(pts)} samples, grid {dist.shape}, outer mid radius {r_mid:.2f} mm, "
          f"{os.path.getsize(fn) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
