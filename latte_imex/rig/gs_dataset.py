"""Posed multi-view dataset of the PITCHER for 3D Gaussian splatting, from the five full pour recordings.

The pitcher moves through many attitudes during a pour while the three cameras stay fixed, so in the pitcher's own
frame (tag frame of the joint bundle adjustment, full/ba.pkl: z = axis towards the rim, spout +y, origin at the tag-4
centre height; rim ~ +93 mm, bottom ~ -33 mm) every camera frame is a new viewpoint:
    camera -> pitcher  =  inv(T_board<-pitcher(t)) @ c2w_cam.
Poses at a frame time: the BA sample itself (cam0 / cam1 frames that took part), else slerp between the two
neighbouring samples if they are < GAP s apart and the pitcher turned < ROT_MAX deg / moved < MOV_MAX mm between them
(cam2 never sees a tag, so its views come only from such quiet moments).
Per image (cropped around the pitcher, intrinsics shifted):
  mask  255 = pitcher (pixel's depth point lies in the wall shell |r - r_out(z)| < SHELL or in the spout region),
        0   = background (outside the dilated silhouette of the pitcher model incl. handle),
        128 = ignore (inside the silhouette but not the shell: hand, milk inside, missing depth)
  depth uint16 mm (aligned to colour)
Init points: shell points in the pitcher frame (voxel 1.5 mm) with colours.
    python3 latte_imex/rig/gs_dataset.py <out dir> [--per-pour 220] [--pours heart,leaf,...]"""
import os, sys, json, pickle, argparse, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.ndimage import median_filter
from rs_zip import Rec

SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
POURS = ("heart", "push_heart", "leaf", "swan", "tulip")
Z_BOTTOM = -33.0                         # pitcher bottom in the tag frame [mm] (height 126 -> rim +93)
PROFILE_Z = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 115, 120, 126], float)
PROFILE_R = np.array([47.5, 47.2, 47.0, 46.8, 46.5, 46.3, 46.1, 45.8, 45.9, 45.4, 44.9, 43.9, 43.3, 42.4, 41.2, 39.6])
SHELL = 7.0                              # [mm] wall shell half-width for 'pitcher' pixels
GAP, ROT_MAX, MOV_MAX = 0.12, 3.0, 4.0   # pose interpolation limits [s, deg, mm]
PAD = 24                                 # crop padding [px]


def r_out(z_tag):
    return np.interp(z_tag - Z_BOTTOM, PROFILE_Z, PROFILE_R, left=np.nan, right=np.nan)


def r_out_c(z_tag):
    return np.interp(np.clip(z_tag - Z_BOTTOM, 0.0, 126.0), PROFILE_Z, PROFILE_R)


def in_shell(P):
    """P [n,3] mm, tag frame -> bool: wall shell or spout region."""
    x, y, z = P.T
    r = np.hypot(x, y)
    ro = r_out(z)
    wall = np.abs(r - ro) < SHELL
    az = np.degrees(np.arctan2(x, y))                    # 0 = spout (+y)
    spout = (np.abs(az) < 35) & (z > 60) & (z < 97) & (r > ro - SHELL) & (r < 60)
    return (wall | spout) & (z > Z_BOTTOM - 3) & (z < 97)


def model_points():
    """Dense points of the pitcher model incl. spout and a generous handle box (silhouette only), tag frame [mm]."""
    zs = np.arange(Z_BOTTOM, 93.5, 1.5)
    ph = np.radians(np.arange(0, 360, 1.5))
    Z, PH = np.meshgrid(zs, ph)
    R = r_out_c(Z)
    body = np.stack([R * np.sin(PH), R * np.cos(PH), Z], -1).reshape(-1, 3)    # x = r sin, y = r cos (spout +y)
    bot = []
    for rr in np.arange(0, 47.5, 2.0):
        a = np.radians(np.arange(0, 360, 3))
        bot.append(np.stack([rr * np.sin(a), rr * np.cos(a), np.full_like(a, Z_BOTTOM)], 1))
    top = []
    for rr in np.arange(0, 40.0, 2.0):
        a = np.radians(np.arange(0, 360, 3))
        top.append(np.stack([rr * np.sin(a), rr * np.cos(a), np.full_like(a, 93.0)], 1))
    sp = []
    for zz in np.arange(70, 95, 1.0):
        for az in np.radians(np.arange(-32, 33, 2)):
            for rr in np.arange(r_out_c(zz), 58, 1.0):
                sp.append((rr * np.sin(az), rr * np.cos(az), zz))
    hd = []                                              # handle: opposite the spout (-y), outside the body
    for zz in np.arange(-25, 92, 2.0):
        for az in np.radians(np.arange(155, 206, 3)):
            for rr in np.arange(r_out_c(zz), 88, 2.0):
                hd.append((rr * np.sin(az), rr * np.cos(az), zz))
    return np.concatenate([body] + bot + top + [np.array(sp), np.array(hd)])


def clean_poses(ba):
    o = np.argsort(ba["t"]); t = ba["t"][o]; P = ba["poses"][o]
    k = np.r_[True, np.diff(t) > 1e-4]; t, P = t[k], P[k]
    R = Rot.from_rotvec(P[:, :3]).as_matrix()
    tilt = np.degrees(np.arccos(np.clip(-R[:, 2, 2], -1, 1)))
    g = np.einsum("nji,j->ni", R, np.array([0, 0, 1.0])); az = np.degrees(np.arctan2(g[:, 1], g[:, 0]))
    good = (np.abs(tilt - median_filter(tilt, 7, mode="nearest")) < 4) & \
           ((np.abs(az - median_filter(az, 7, mode="nearest")) < 8) | (tilt < 25))
    return t[good], P[good]


def pose_at(t, tb, Pb):
    """4x4 board<-pitcher [m] at time t, or None."""
    i = np.searchsorted(tb, t)
    def T_of(P):
        T = np.eye(4); T[:3, :3] = Rot.from_rotvec(P[:3]).as_matrix(); T[:3, 3] = P[3:]; return T
    for j in (i - 1, i):
        if 0 <= j < len(tb) and abs(tb[j] - t) < 0.003:
            return T_of(Pb[j])
    if i == 0 or i >= len(tb) or tb[i] - tb[i - 1] > GAP:
        return None
    a, b = Pb[i - 1], Pb[i]
    if np.degrees((Rot.from_rotvec(a[:3]).inv() * Rot.from_rotvec(b[:3])).magnitude()) > ROT_MAX:
        return None
    if np.linalg.norm(a[3:] - b[3:]) * 1e3 > MOV_MAX:
        return None
    w = (t - tb[i - 1]) / (tb[i] - tb[i - 1])
    T = np.eye(4)
    T[:3, :3] = Slerp([0, 1], Rot.from_rotvec([a[:3], b[:3]]))(w).as_matrix()
    T[:3, 3] = (1 - w) * a[3:] + w * b[3:]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pours", default=",".join(POURS))
    ap.add_argument("--per-pour", type=int, default=220, help="images per pour (spread over cameras and time)")
    a = ap.parse_args()
    for d in ("images", "masks", "depths"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    M = model_points() * 1e-3
    frames, pts, cols = [], [], []
    for p in a.pours.split(","):
        rec = Rec(f"{SCR}/zips/{p}.zip")
        tb, Pb = clean_poses(pickle.load(open(f"{SCR}/{p}/full/ba.pkl", "rb")))
        cand = []
        for c in range(rec.ncam):
            for k, ts in zip(rec.steps[c], rec.ts[c]):
                t = ts - rec.t0
                T = pose_at(t, tb, Pb)
                if T is not None:
                    cand.append((c, int(k), float(t), T))
        # spread evenly: per camera, evenly in time; cam2 views are rarer and more valuable -> its share first
        per_cam = {c: [x for x in cand if x[0] == c] for c in range(rec.ncam)}
        take, left = [], a.per_pour
        order = sorted(per_cam, key=lambda c: len(per_cam[c]))
        for i, c in enumerate(order):
            n = min(len(per_cam[c]), left // (len(order) - i))
            if n:
                idx = np.unique(np.linspace(0, len(per_cam[c]) - 1, n).round().astype(int))
                take += [per_cam[c][j] for j in idx]
                left -= len(idx)
        print(p, "candidates per cam", {c: len(v) for c, v in per_cam.items()}, "-> taken", len(take), flush=True)
        for c, k, t, Tbp in take:
            K = rec.K[c]; c2w = rec.c2w[c]
            c2o = np.linalg.inv(Tbp) @ c2w                      # camera -> pitcher (OpenCV camera axes)
            o2c = np.linalg.inv(c2o)
            img = rec.color(c, k); dep = rec.depth(c, k)
            H, W = dep.shape
            # silhouette of the model
            Pc = M @ o2c[:3, :3].T + o2c[:3, 3]
            ok = Pc[:, 2] > 0.05
            u = Pc[ok, 0] / Pc[ok, 2] * K[0, 0] + K[0, 2]; v = Pc[ok, 1] / Pc[ok, 2] * K[1, 1] + K[1, 2]
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if inb.mean() < 0.9:                                # pitcher partly out of the image
                continue
            sil = np.zeros((H, W), np.uint8)
            sil[v[inb].astype(int), u[inb].astype(int)] = 1
            sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            sil = cv2.dilate(sil, np.ones((7, 7), np.uint8))
            bg = cv2.dilate(sil, np.ones((13, 13), np.uint8)) == 0
            # pitcher pixels from depth
            vv, uu = np.nonzero((sil > 0) & (dep > 0))
            z = dep[vv, uu].astype(float) * 1e-3
            Pcam = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], 1)
            Po = (Pcam @ c2o[:3, :3].T + c2o[:3, 3]) * 1e3
            sh = in_shell(Po)
            mask = np.where(bg, 0, 128).astype(np.uint8)
            mask[vv[sh], uu[sh]] = 255
            if (mask == 255).sum() < 800:
                continue
            ys, xs = np.nonzero(sil)
            x0, x1 = max(0, xs.min() - PAD), min(W, xs.max() + PAD + 1)
            y0, y1 = max(0, ys.min() - PAD), min(H, ys.max() + PAD + 1)
            name = f"{p}_c{c}_{k}"
            cv2.imwrite(f"{a.out}/images/{name}.png", img[y0:y1, x0:x1])
            cv2.imwrite(f"{a.out}/masks/{name}.png", mask[y0:y1, x0:x1])
            cv2.imwrite(f"{a.out}/depths/{name}.png", dep[y0:y1, x0:x1].astype(np.uint16))
            frames.append(dict(name=name, pour=p, cam=c, step=k, t=t, w=int(x1 - x0), h=int(y1 - y0),
                               fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2] - x0), cy=float(K[1, 2] - y0),
                               c2o=c2o.tolist(), n_pitcher_px=int((mask == 255).sum())))
            sel = np.nonzero(sh)[0][::3]
            pts.append(Po[sel] * 1e-3); cols.append(img[vv[sel], uu[sel]][:, ::-1])
        print(p, "kept", sum(1 for f in frames if f["pour"] == p), flush=True)
    P = np.concatenate(pts); C = np.concatenate(cols)
    key = np.floor(P / 0.0015).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    n = first.size
    Pm = np.stack([np.bincount(inv, P[:, j], n) for j in range(3)], 1) / np.bincount(inv, None, n)[:, None]
    Cm = np.stack([np.bincount(inv, C[:, j].astype(float), n) for j in range(3)], 1) / np.bincount(inv, None, n)[:, None]
    np.savez_compressed(f"{a.out}/init_points.npz", xyz=Pm.astype(np.float32), rgb=Cm.astype(np.uint8))
    json.dump(dict(frame="pitcher tag frame [m]: z = axis to the rim, spout +y, rim ~ +0.093, bottom ~ -0.033",
                   camera="OpenCV (x right, y down, z forward); c2o = camera -> pitcher",
                   mask="255 pitcher, 128 ignore, 0 background", depth="uint16 mm, aligned to colour",
                   frames=frames), open(f"{a.out}/transforms.json", "w"), indent=1)
    print("frames", len(frames), "init points", len(Pm), "->", a.out)


if __name__ == "__main__":
    main()
