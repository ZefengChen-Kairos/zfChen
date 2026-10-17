"""Posed multi-view dataset for 3D Gaussian splatting from a STATIC recording (coffee_cup / milk_cup): the object stands
on the table and is turned by hand between quiet windows, the three cameras never move.  Every quiet window is
therefore 3 new viewpoints of the object once the windows are registered to each other.

  1. quiet windows: frame differences inside the object's image region (all cameras) below a threshold for >= MIN_WIN s
  2. per window and camera: median colour and median depth over the window's frames (noise, flicker)
  3. table plane (RANSAC on the reference window), object column (densest cell above the table), vertical axis
     (robust circle fit of the body), points in a table frame centred on the axis
  4. window -> reference: 3-DOF rigid (yaw about the axis + table-plane shift) by ICP on xyz + colour from 36 yaw
     starts, best mean residual
  5. images (cropped), masks (255 object, 0 background, 128 ignore = occluder in front of the object or no depth),
     depths, camera -> object poses (object frame: table plane z = 0, z up, origin on the axis), init points.
Output format = gs_dataset.py (transforms.json, images/, masks/, depths/, init_points.npz) + merged.ply for checking.
    python3 latte_imex/rig/gs_static.py <coffee_cup|milk_cup> <out dir> [--wins t0:t1,t0:t1,...]"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from scipy.spatial import cKDTree
from rs_zip import Rec

SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
OBJ = dict(coffee_cup=dict(R=0.085, H=0.085), milk_cup=dict(R=0.090, H=0.140))   # object volume around the axis [m]
MIN_WIN, NMED, PAD = 0.6, 15, 24


def plane_ransac(P, it=400, tol=0.004, rng=np.random.default_rng(0)):
    best = (0, None)
    for _ in range(it):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        k = np.abs((P - a) @ n) < tol
        if k.sum() > best[0]:
            best = (k.sum(), k)
    Q = P[best[1]]
    m = Q.mean(0)
    n = np.linalg.svd(Q - m, full_matrices=False)[2][-1]
    return n, m


def kasa(x, y):
    A = np.stack([x, y, np.ones_like(x)], 1)
    a = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    cx, cy = a[0] / 2, a[1] / 2
    return cx, cy, np.sqrt(max(a[2] + cx * cx + cy * cy, 1e-9))


def backproject(d, K):
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    return np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)


class Scene:
    def __init__(self, name):
        self.r = Rec(f"{SCR}/zips/{name}.zip")
        self.name = name

    def frames_in(self, c, t0, t1):
        t = self.r.ts[c] - self.r.t0
        return self.r.steps[c][(t >= t0) & (t <= t1)]

    def median_view(self, c, t0, t1):
        ks = self.frames_in(c, t0, t1)
        ks = ks[np.linspace(0, len(ks) - 1, min(NMED, len(ks))).round().astype(int)]
        D = np.stack([self.r.depth(c, k) for k in ks]).astype(np.float32)
        D[D == 0] = np.nan
        n_ok = np.isfinite(D).sum(0)
        dep = np.where(n_ok >= max(2, len(ks) // 2), np.nanmedian(D, 0), 0)
        img = np.median(np.stack([self.r.color(c, k) for k in ks]), 0).astype(np.uint8)
        return img, np.nan_to_num(dep).astype(np.uint16)

    def quiet_windows(self, rois, thr=0.006):
        r = self.r
        grid = np.arange(0, min(ts[-1] for ts in r.ts) - r.t0, 0.05)
        A = np.zeros((r.ncam, len(grid)))
        for c in range(r.ncam):
            x0, y0, x1, y1 = rois[c]
            prev, vals = None, []
            for k in r.steps[c]:
                g = cv2.cvtColor(r.color(c, k)[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
                g = cv2.GaussianBlur(g, (5, 5), 0)
                vals.append(1.0 if prev is None else float(np.mean(np.abs(g - prev) > 10)))
                prev = g
            A[c] = np.interp(grid, r.ts[c] - r.t0, np.array(vals))
        q = A.max(0) < thr
        wins, i = [], 0
        while i < len(q):
            if q[i]:
                j = i
                while j < len(q) and q[j]:
                    j += 1
                if grid[j - 1] - grid[i] >= MIN_WIN:
                    wins.append((float(grid[i]) + 0.05, float(grid[j - 1]) - 0.05))
                i = j
            else:
                i += 1
        return wins, grid, A


def icp_3dof(A, CA, B, CB, yaw0, iters=25, wc=0.02):
    """B -> A: rotation about z by yaw, shift (tx, ty).  xyz [m] + colour (0..1) * wc in the NN search."""
    tree = cKDTree(np.hstack([A, CA * wc]))
    th, t = yaw0, np.zeros(2)
    for _ in range(iters):
        c, s = np.cos(th), np.sin(th)
        Bx = np.stack([c * B[:, 0] - s * B[:, 1] + t[0], s * B[:, 0] + c * B[:, 1] + t[1], B[:, 2]], 1)
        d, j = tree.query(np.hstack([Bx, CB * wc]), distance_upper_bound=0.02)
        k = np.isfinite(d)
        if k.sum() < 100:
            return np.inf, th, t
        P, Q = Bx[k, :2], A[j[k], :2]
        mp, mq = P.mean(0), Q.mean(0)
        Hm = (P - mp).T @ (Q - mq)
        dth = np.arctan2(Hm[0, 1] - Hm[1, 0], Hm[0, 0] + Hm[1, 1])
        c2, s2 = np.cos(dth), np.sin(dth)
        R2 = np.array([[c2, -s2], [s2, c2]])
        th += dth
        t = R2 @ t + (mq - R2 @ mp)
    c, s = np.cos(th), np.sin(th)
    Bx = np.stack([c * B[:, 0] - s * B[:, 1] + t[0], s * B[:, 0] + c * B[:, 1] + t[1], B[:, 2]], 1)
    d, j = cKDTree(A).query(Bx)
    dc = np.linalg.norm(CA[j] - CB, axis=1)
    return float(np.mean(np.minimum(d, 0.01)) + 0.01 * np.mean(np.minimum(dc, 0.5))), th, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("out")
    ap.add_argument("--wins", default=None, help="t0:t1,... (default: detected quiet windows)")
    a = ap.parse_args()
    sc = Scene(a.name)
    r = sc.r
    vol = OBJ[a.name]
    # --- reference geometry from a first quiet guess (the first 1.5 s)
    views0 = [sc.median_view(c, 0.0, 1.5) for c in range(r.ncam)]
    Pb = []
    for c in range(r.ncam):
        d = views0[c][1].astype(np.float32) * 1e-3
        X = backproject(d, r.K[c])[d > 0][::7]
        Pb.append(X @ r.c2w[c][:3, :3].T + r.c2w[c][:3, 3])
    Pb = np.concatenate(Pb)
    Pb = Pb[np.linalg.norm(Pb - np.median(Pb, 0), axis=1) < 0.6]
    n, m = plane_ransac(Pb)
    cam_c = np.mean([r.c2w[c][:3, 3] for c in range(r.ncam)], 0)
    if n @ (cam_c - m) < 0:
        n = -n
    e1 = np.cross([0, 1.0, 0], n); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    E = np.stack([e1, e2, n])                                  # board -> table rows
    to_table = lambda P: (P - m) @ E.T
    # object column: densest 2 cm cell of points 2-6 cm above the table
    Q = to_table(Pb)
    k = (Q[:, 2] > 0.02) & (Q[:, 2] < 0.06) & (np.abs(Q[:, 0]) < 0.5) & (np.abs(Q[:, 1]) < 0.5)
    Hh, xe, ye = np.histogram2d(Q[k, 0], Q[k, 1], bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
    i, j = np.unravel_index(np.argmax(cv2.blur(Hh, (5, 5))), Hh.shape)
    c0 = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
    # image ROI of the object per camera (projected object volume)
    ang = np.linspace(0, 2 * np.pi, 48)
    ring = np.concatenate([np.stack([c0[0] + (vol["R"] + 0.12) * np.cos(ang), c0[1] + (vol["R"] + 0.12) * np.sin(ang), np.full_like(ang, h)], 1)
                           for h in (0.0, vol["H"])])
    ring_b = ring @ E + m
    rois = []
    for c in range(r.ncam):
        w2c = np.linalg.inv(r.c2w[c])
        Pc = ring_b @ w2c[:3, :3].T + w2c[:3, 3]
        u = Pc[:, 0] / Pc[:, 2] * r.K[c][0, 0] + r.K[c][0, 2]; v = Pc[:, 1] / Pc[:, 2] * r.K[c][1, 1] + r.K[c][1, 2]
        rois.append((int(max(0, u.min())), int(max(0, v.min())), int(min(847, u.max())), int(min(479, v.max()))))
    if a.wins:
        wins = [tuple(float(x) for x in w.split(":")) for w in a.wins.split(",")]
    else:
        wins, grid, A = sc.quiet_windows(rois)
    print(a.name, "object column at", np.round(c0 * 1e3, 1), "mm; ROIs", rois, "\nquiet windows", [(round(x, 2), round(y, 2)) for x, y in wins], flush=True)
    # --- per window: views, object points in a table frame centred on the fitted axis
    W = []
    for (t0, t1) in wins:
        views = [sc.median_view(c, t0, t1) for c in range(r.ncam)]
        pts, cols = [], []
        for c in range(r.ncam):
            img, dep = views[c]
            d = dep.astype(np.float32) * 1e-3
            ok = d > 0
            X = backproject(d, r.K[c])[ok]
            Qw = to_table(X @ r.c2w[c][:3, :3].T + r.c2w[c][:3, 3])
            sel = (np.hypot(Qw[:, 0] - c0[0], Qw[:, 1] - c0[1]) < 0.30) & (Qw[:, 2] > 0.003) & (Qw[:, 2] < vol["H"])
            pts.append(Qw[sel]); cols.append(img[ok][sel][:, ::-1] / 255.0)
        Qw, Cw = np.concatenate(pts), np.concatenate(cols)
        mid = (Qw[:, 2] > 0.02) & (Qw[:, 2] < 0.06)
        Hh, xe, ye = np.histogram2d(Qw[mid, 0], Qw[mid, 1], bins=120, range=[[c0[0] - 0.3, c0[0] + 0.3], [c0[1] - 0.3, c0[1] + 0.3]])
        i, j = np.unravel_index(np.argmax(cv2.blur(Hh, (9, 9))), Hh.shape)
        cw = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
        near = np.hypot(Qw[:, 0] - cw[0], Qw[:, 1] - cw[1]) < vol["R"] + 0.02
        Qw, Cw = Qw[near], Cw[near]
        body = (Qw[:, 2] > 0.015) & (Qw[:, 2] < vol["H"] * 0.7)
        cx, cy, R = kasa(Qw[body, 0], Qw[body, 1])
        for _ in range(4):
            rr = np.hypot(Qw[:, 0] - cx, Qw[:, 1] - cy)
            kb = body & (np.abs(rr - R) < 0.006)
            cx, cy, R = kasa(Qw[kb, 0], Qw[kb, 1])
        rr = np.hypot(Qw[:, 0] - cx, Qw[:, 1] - cy)
        keep = rr < vol["R"]
        # handle azimuth: points sticking out of the body at mid height (the spout of the pitcher is higher up)
        far = keep & (rr > R + 0.007) & (Qw[:, 2] > 0.01) & (Qw[:, 2] < 0.6 * vol["H"])
        az = np.arctan2(Qw[far, 1] - cy, Qw[far, 0] - cx)
        az_h = float(np.arctan2(np.sin(az).mean(), np.cos(az).mean())) if far.sum() > 50 else None
        W.append(dict(t=(t0, t1), views=views, Q=Qw[keep] - [cx, cy, 0], C=Cw[keep], c=np.array([cx, cy]), R=R,
                      az_h=az_h, n_h=int(far.sum())))
        print(f"  window {t0:.2f}-{t1:.2f}: {keep.sum()} object points, body radius {R * 1e3:.1f} mm, axis at "
              f"{np.round(np.array([cx, cy]) * 1e3, 1)} mm, handle azimuth "
              f"{np.degrees(W[-1]['az_h']) if W[-1]['az_h'] is not None else float('nan'):.1f} deg ({W[-1]['n_h']} pts)", flush=True)
    # --- register to the reference window (the one with the most points)
    ref = int(np.argmax([len(w["Q"]) for w in W]))
    rng = np.random.default_rng(0)
    sub = lambda w, n=20000: (lambda i: (w["Q"][i], w["C"][i]))(rng.choice(len(w["Q"]), min(n, len(w["Q"])), replace=False))
    A, CA = sub(W[ref])
    for k, w in enumerate(W):
        if k == ref:
            w.update(yaw=0.0, shift=np.zeros(2), res=0.0)
            continue
        B, CB = sub(w)
        if w["az_h"] is not None and W[ref]["az_h"] is not None:      # handle-to-handle start, ICP refines
            y = W[ref]["az_h"] - w["az_h"]
            starts = [y + np.radians(d) for d in (-8, -4, 0, 4, 8)]
        else:
            starts = np.radians(np.arange(0, 360, 10))
        best = min((icp_3dof(A, CA, B, CB, y0) for y0 in starts), key=lambda x: x[0])
        w.update(yaw=float(best[1]), shift=best[2], res=best[0])
        print(f"  window {k} -> ref {ref}: yaw {np.degrees(best[1]) % 360:.1f} deg, shift {np.round(best[2] * 1e3, 1)} mm, "
              f"mean residual {best[0] * 1e3:.2f} mm", flush=True)
    # --- dataset
    for dd in ("images", "masks", "depths"):
        os.makedirs(os.path.join(a.out, dd), exist_ok=True)
    frames, allP, allC = [], [], []
    for k, w in enumerate(W):
        c, s = np.cos(w["yaw"]), np.sin(w["yaw"])
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        T_ob = np.eye(4)                                          # board -> object
        T_ob[:3, :3] = Rz @ E
        T_ob[:3, 3] = Rz @ (-E @ m - np.r_[w["c"], 0]) + np.r_[w["shift"], 0]
        allP.append(w["Q"] @ Rz.T + np.r_[w["shift"], 0]); allC.append(w["C"])
        for cam in range(r.ncam):
            img, dep = w["views"][cam]
            K = r.K[cam]
            c2o = T_ob @ r.c2w[cam]
            d = dep.astype(np.float32) * 1e-3
            X = backproject(d, K)
            Po = X @ c2o[:3, :3].T + c2o[:3, 3]
            rr = np.hypot(Po[..., 0], Po[..., 1])
            obj = (d > 0) & (rr < vol["R"]) & (Po[..., 2] > 0.002) & (Po[..., 2] < vol["H"])
            # silhouette of the object volume (cylinder R x H) and its depth
            o2c = np.linalg.inv(c2o)
            g = np.stack(np.meshgrid(np.linspace(-vol["R"], vol["R"], 90), np.linspace(-vol["R"], vol["R"], 90),
                                     np.linspace(0, vol["H"], 60), indexing="ij"), -1).reshape(-1, 3)
            g = g[np.hypot(g[:, 0], g[:, 1]) < vol["R"]]
            Pc = g @ o2c[:3, :3].T + o2c[:3, 3]
            u = (Pc[:, 0] / Pc[:, 2] * K[0, 0] + K[0, 2]).astype(int); v = (Pc[:, 1] / Pc[:, 2] * K[1, 1] + K[1, 2]).astype(int)
            ib = (u >= 0) & (u < d.shape[1]) & (v >= 0) & (v < d.shape[0])
            near = np.full(d.shape, np.inf, np.float32)
            np.minimum.at(near, (v[ib], u[ib]), Pc[ib, 2].astype(np.float32))
            near = cv2.erode(cv2.dilate(np.where(np.isfinite(near), near, 0), np.ones((5, 5), np.uint8)), np.ones((3, 3), np.uint8))
            sil = near > 0
            mask = np.zeros(d.shape, np.uint8)                   # background
            mask[sil & ((d == 0) | (d < near - 0.005)) & ~obj] = 128   # no depth, or something in front of the volume
            mask[obj] = 255
            ys, xs = np.nonzero(sil)
            if len(xs) == 0 or (mask == 255).sum() < 500:
                continue
            x0, x1 = max(0, xs.min() - PAD), min(d.shape[1], xs.max() + PAD + 1)
            y0, y1 = max(0, ys.min() - PAD), min(d.shape[0], ys.max() + PAD + 1)
            nm = f"{a.name}_w{k}_c{cam}"
            cv2.imwrite(f"{a.out}/images/{nm}.png", img[y0:y1, x0:x1])
            cv2.imwrite(f"{a.out}/masks/{nm}.png", mask[y0:y1, x0:x1])
            cv2.imwrite(f"{a.out}/depths/{nm}.png", dep[y0:y1, x0:x1])
            frames.append(dict(name=nm, window=k, t=list(w["t"]), cam=cam, w=int(x1 - x0), h=int(y1 - y0),
                               fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2] - x0), cy=float(K[1, 2] - y0),
                               c2o=c2o.tolist(), n_object_px=int((mask == 255).sum())))
    P = np.concatenate(allP); C = (np.concatenate(allC) * 255).astype(np.uint8)
    key = np.floor(P / 0.0012).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel(); nn = inv.max() + 1
    cnt = np.bincount(inv, None, nn)
    Pm = np.stack([np.bincount(inv, P[:, j], nn) for j in range(3)], 1) / cnt[:, None]
    Cm = np.stack([np.bincount(inv, C[:, j].astype(float), nn) for j in range(3)], 1) / cnt[:, None]
    np.savez_compressed(f"{a.out}/init_points.npz", xyz=Pm.astype(np.float32), rgb=Cm.astype(np.uint8))
    with open(f"{a.out}/merged.ply", "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(Pm)}\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, col in zip(Pm, Cm.astype(np.uint8)):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {col[0]} {col[1]} {col[2]}\n")
    json.dump(dict(frame=f"{a.name} object frame [m]: table plane z = 0 (up), origin on the fitted vertical axis",
                   camera="OpenCV (x right, y down, z forward); c2o = camera -> object",
                   mask="255 object, 128 ignore, 0 background", depth="uint16 mm, aligned to colour",
                   windows=[dict(t=list(w["t"]), yaw_deg=float(np.degrees(w["yaw"])), shift_mm=(w["shift"] * 1e3).tolist(),
                                 icp_residual_mm=w["res"] * 1e3, body_radius_mm=w["R"] * 1e3) for w in W],
                   reference_window=ref, frames=frames), open(f"{a.out}/transforms.json", "w"), indent=1)
    print(a.name, "views", len(frames), "init points", len(Pm), "->", a.out)


if __name__ == "__main__":
    main()
