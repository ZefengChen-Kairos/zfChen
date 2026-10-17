"""Robust liquid-disc fit on saved top views: edge points where bright liquid (V>85) meets the dark cup rim (V<60),
RANSAC circle (radius 28-50 mm) + least-squares refine on inliers; then temporal smoothing."""
import sys, os, json, glob
import numpy as np, cv2
from scipy.ndimage import median_filter

def kasa(x, y):
    A = np.stack([x, y, np.ones_like(x)], 1); b = x * x + y * y
    a = np.linalg.lstsq(A, b, rcond=None)[0]; cx, cy = a[0] / 2, a[1] / 2
    return cx, cy, np.sqrt(max(a[2] + cx * cx + cy * cy, 1e-9))


def ransac_disc(img, seen, PX, rng=np.random.default_rng(0), rlim=(0.028, 0.050)):
    V = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    bright = ((V > 85) & seen).astype(np.uint8); dark = ((V < 60) & seen).astype(np.uint8)
    edge = bright & cv2.dilate(dark, np.ones((5, 5), np.uint8)) & (1 - cv2.erode(bright, np.ones((3, 3), np.uint8)))
    pts = np.argwhere(edge)[:, ::-1].astype(float)
    if len(pts) < 30: return None
    best = (0, None); rmin, rmax = rlim[0] / PX, rlim[1] / PX
    for _ in range(1500):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        cx, cy, r = kasa(s[:, 0], s[:, 1])
        if not (rmin < r < rmax): continue
        inl = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < 3
        ang = np.arctan2(pts[inl, 1] - cy, pts[inl, 0] - cx)
        spread = len(np.unique(np.floor((ang + np.pi) / (np.pi / 18))))
        score = inl.sum() * min(spread, 24) / 24
        if score > best[0]: best = (score, (cx, cy, r, inl))
    if best[1] is None: return None
    cx, cy, r, inl = best[1]
    for _ in range(3):
        cx, cy, r = kasa(pts[inl, 0], pts[inl, 1]); inl = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < 2.5
    ang = np.arctan2(pts[inl, 1] - cy, pts[inl, 0] - cx)
    return cx, cy, r, 10 * len(np.unique(np.floor((ang + np.pi) / (np.pi / 18))))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    take = sys.argv[1]
    B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
    tv = json.load(open(f"{B}/{take}/topview.json"))
    def kasa(x, y):
        A = np.stack([x, y, np.ones_like(x)], 1); b = x * x + y * y
        a = np.linalg.lstsq(A, b, rcond=None)[0]; cx, cy = a[0] / 2, a[1] / 2
        return cx, cy, np.sqrt(max(a[2] + cx * cx + cy * cy, 1e-9))
    out = []
    for fi, o in enumerate(tv):
        Z = np.load(f"{B}/{take}/top/{fi:04d}.npz"); img, seen, c0, PX, HALF = Z["img"], Z["seen"], Z["c0"], float(Z["px"]), float(Z["half"])
        V = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
        bright = ((V > 85) & seen).astype(np.uint8); dark = ((V < 60) & seen).astype(np.uint8)
        edge = bright & cv2.dilate(dark, np.ones((5, 5), np.uint8)) & (1 - cv2.erode(bright, np.ones((3, 3), np.uint8)))
        pts = np.argwhere(edge)[:, ::-1].astype(float)
        best = (0, None); rmin, rmax = 0.028 / PX, 0.050 / PX
        for _ in range(1500):
            s = pts[rng.choice(len(pts), 3, replace=False)]
            try: cx, cy, r = kasa(s[:, 0], s[:, 1])
            except Exception: continue
            if not (rmin < r < rmax): continue
            d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
            inl = d < 3
            # require angular spread
            ang = np.arctan2(pts[inl, 1] - cy, pts[inl, 0] - cx)
            spread = len(np.unique(np.floor((ang + np.pi) / (np.pi / 18)))) 
            score = inl.sum() * min(spread, 24) / 24
            if score > best[0]: best = (score, (cx, cy, r, inl))
        if best[1] is None: out.append(None); continue
        cx, cy, r, inl = best[1]
        for _ in range(3):
            cx, cy, r = kasa(pts[inl, 0], pts[inl, 1]); inl = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < 2.5
        ang = np.arctan2(pts[inl, 1] - cy, pts[inl, 0] - cx)
        out.append(dict(cx=float(c0[0] + (cx + 0.5) * PX - HALF), cy=float(c0[1] + (cy + 0.5) * PX - HALF), R=float(r * PX),
                        n_inl=int(inl.sum()), cover_deg=int(10 * len(np.unique(np.floor((ang + np.pi) / (np.pi / 18))))), px=(cx, cy, r)))
    t = np.array([o["t"] for o in tv]); ok = np.array([q is not None and q["cover_deg"] >= 150 for q in out])
    C = np.array([[q["cx"], q["cy"], q["R"]] if q else [np.nan] * 3 for q in out])
    Cs = C.copy()
    for j in range(3):
        v = np.where(ok, C[:, j], np.nan); idx = np.where(ok)[0]
        Cs[:, j] = median_filter(np.interp(np.arange(len(t)), idx, v[idx]), 9, mode="nearest")
    res = []
    for fi, o in enumerate(tv):
        tip = np.array(o["tip_xy"]); cen = Cs[fi, :2]
        res.append(dict(t=o["t"], surf_z=o["surf_z"], disc_centre=cen.tolist(), disc_R=float(Cs[fi, 2]), raw=out[fi] and {k: v for k, v in out[fi].items() if k != "px"},
                        fit_ok=bool(ok[fi]), tip_rel_mm=((tip - cen) * 1e3).tolist(), tip_rel_norm=((tip - cen) / Cs[fi, 2]).tolist(), jet_h_mm=o["jet_h_mm"]))
        img = cv2.imread(f"{B}/{take}/top/{fi:04d}.png")
        Z = np.load(f"{B}/{take}/top/{fi:04d}.npz"); c0, PX, HALF = Z["c0"], float(Z["px"]), float(Z["half"])
        img = cv2.cvtColor(cv2.cvtColor(Z["img"], cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)
        pc = ((cen - c0 + HALF) / PX).astype(int); cv2.circle(img, tuple(int(a) for a in pc), int(Cs[fi, 2] / PX), (0, 255, 0) if ok[fi] else (0, 165, 255), 1)
        tp = ((tip - c0 + HALF) / PX).astype(int); cv2.drawMarker(img, tuple(int(a) for a in tp), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.imwrite(f"{B}/{take}/top/fit_{fi:04d}.png", img)
    json.dump(res, open(f"{B}/{take}/disc.json", "w"))
    print("fits ok", ok.sum(), "/", len(ok), "R mm median", np.nanmedian(Cs[:, 2]) * 1e3, "range", np.nanpercentile(Cs[:, 2], [5, 95]) * 1e3)
