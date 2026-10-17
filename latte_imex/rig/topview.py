"""Rectified top view of the liquid surface per frame: a 0.5 mm grid on the horizontal plane at the measured surface
height (cup_track.json), projected into every camera; a camera contributes a pixel when it is not occluded (its depth
is not nearer than the plane by > OCC) -- missing depth counts as visible. Colour per grid point = per-channel
median over contributing views (MIN=1: minimum brightness view, to suppress specular highlights). Liquid disc =
circle fitted to the boundary of the 'not dark' region (the black cup rim/wall around it)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from trimrec import Take

take = sys.argv[1]; MODE = os.environ.get("MODE", "median")
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
tk = Take(f"{B}/ghdata/{take}"); ct = json.load(open(f"{B}/{take}/cup_track.json"))
PX, HALF, OCC = 0.0005, 0.07, 0.010
os.makedirs(f"{B}/{take}/top", exist_ok=True)
W2C = [np.linalg.inv(T) for T in tk.c2w]
def kasa(x, y):
    A = np.stack([x, y, np.ones_like(x)], 1); b = x * x + y * y
    a = np.linalg.lstsq(A, b, rcond=None)[0]; cx, cy = a[0] / 2, a[1] / 2
    return cx, cy, np.sqrt(max(a[2] + cx * cx + cy * cy, 1e-9))

def fit_disc(img, seen, c0):
    """liquid = bright region enclosed by the dark cup rim (outside reached from the border without crossing dark)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < 70) & seen
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    free = (1 - dark).astype(np.uint8)
    ff = free.copy(); h, w = ff.shape; mask = np.zeros((h + 2, w + 2), np.uint8)
    for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] + [(x, 0) for x in range(0, w, 20)] + [(x, h - 1) for x in range(0, w, 20)] + [(0, y) for y in range(0, h, 20)] + [(w - 1, y) for y in range(0, h, 20)]:
        if ff[y, x] == 1: cv2.floodFill(ff, mask, (x, y), 2)
    inside = (ff == 1) & seen & (hsv[..., 2] > 85)
    inside = cv2.morphologyEx(inside.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nl, lab, st, cen_ = cv2.connectedComponentsWithStats(inside)
    if nl < 2: return c0, np.nan, 0, 0, 0
    L = 1 + int(np.argmax(st[1:, 4])); comp = (lab == L).astype(np.uint8)
    if st[L, 4] < 2000: return c0, np.nan, 0, 0, 0
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=len).reshape(-1, 2).astype(float)
    ex, ey, er = kasa(cnt[:, 0], cnt[:, 1])
    for _ in range(4):
        rr = np.hypot(cnt[:, 0] - ex, cnt[:, 1] - ey); keep = rr > er - 5
        ex, ey, er = kasa(cnt[keep, 0], cnt[keep, 1])
    return c0 + (np.array([ex, ey]) + 0.5) * PX - HALF, er * PX, ex, ey, er

# smooth the depth-based surface height and centre over time (they are noisy frame to frame)
tt = np.array([o["t"] for o in ct]); zs = np.array([o["surf_z_board"] for o in ct]); cxy = np.array([o["surf_centre"] for o in ct])
from scipy.ndimage import median_filter
zs_s = median_filter(zs, 9, mode="nearest"); cx_s = median_filter(cxy[:, 0], 15, mode="nearest"); cy_s = median_filter(cxy[:, 1], 15, mode="nearest")
res = []
n = int(2 * HALF / PX)
g = (np.arange(n) + 0.5) * PX - HALF
for fi, o in enumerate(ct):
    i = int(np.argmin(np.abs(tk.T - o["t"])))
    c0 = np.array([cx_s[fi], cy_s[fi]]); z = zs_s[fi]
    X, Y = np.meshgrid(c0[0] + g, c0[1] + g)
    Pw = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    cols, vis = [], []
    for c, k, dt in tk.sync(i):
        K = tk.K[c]; Pc = Pw @ W2C[c][:3, :3].T + W2C[c][:3, 3]
        u = Pc[:, 0] / Pc[:, 2] * K[0, 0] + K[0, 2]; v = Pc[:, 1] / Pc[:, 2] * K[1, 1] + K[1, 2]
        ok = (u >= 0) & (u < 847) & (v >= 0) & (v < 479) & (Pc[:, 2] > 0)
        img = tk.color(c, k); dep = tk.depth(c, k).astype(float) * 1e-3
        ui, vi = np.clip(u.astype(int), 0, 847), np.clip(v.astype(int), 0, 479)
        dm = dep[vi, ui]
        vis_c = ok & ((dm == 0) | (dm > Pc[:, 2] - OCC))
        cols.append(img[vi, ui].astype(float)); vis.append(vis_c)
    C = np.stack(cols); V = np.stack(vis)
    Cm = np.where(V[..., None], C, np.nan)
    if MODE == "min":
        br = np.where(V, C.mean(2), np.inf); j = np.argmin(br, 0)
        top = C[j, np.arange(C.shape[1])]; top[~V.any(0)] = np.nan
    else:
        top = np.nanmedian(Cm, 0)
    top = top.reshape(n, n, 3); seen = V.any(0).reshape(n, n)
    img = np.nan_to_num(top).astype(np.uint8)
    np.savez_compressed(f"{B}/{take}/top/{fi:04d}.npz", img=img, seen=seen, c0=c0, z=z, px=PX, half=HALF)
    cen, R, ex, ey, er = fit_disc(img, seen, c0)
    tip = np.array(o["tip"])
    res.append(dict(t=o["t"], surf_z=z, disc_centre=cen.tolist(), disc_R=float(R), tip_xy=tip[:2].tolist(),
                    tip_rel_mm=((tip[:2] - cen) * 1e3).tolist(), jet_h_mm=float((z - tip[2]) * 1e3)))
    vis_img = img.copy()
    if np.isfinite(R):
        cv2.circle(vis_img, (int(ex), int(ey)), int(er), (0, 255, 0), 1)
        tp = ((tip[:2] - c0 + HALF) / PX).astype(int); cv2.drawMarker(vis_img, tuple(int(a) for a in tp), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    cv2.imwrite(f"{B}/{take}/top/{fi:04d}.png", vis_img)
json.dump(res, open(f"{B}/{take}/topview.json", "w"))
R = np.array([r["disc_R"] for r in res]) * 1e3
print(len(res), "frames; disc R mm median", np.nanmedian(R).round(1), "pcts", np.nanpercentile(R, [5, 95]).round(1))
