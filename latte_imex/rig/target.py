"""Clean target image of the finished pattern from post-pour frames (pitcher gone): per frame, liquid-surface height
from depth around the last known disc centre, multi-view top view (median over views), RANSAC disc; frames resampled
to disc coordinates (centre 0, radius 1) and composited by a per-pixel median over frames (moving highlights and view
seams drop out). Output <B>/targets/<pattern>.npz: img (BGR, disc coords), valid, white (white-foam mask), + png."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from trimrec import Take, cloud
from disc_fit import ransac_disc

p = sys.argv[1]
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
tk = Take(f"{B}/post/{p}"); D = json.load(open(f"{B}/{p}/disc.json"))
c_prev = np.array(D[-1]["disc_centre"]); z_prev = D[-1]["surf_z"]
PX, HALF, OCC, NOUT = 0.0005, 0.07, 0.010, 256
W2C = [np.linalg.inv(T) for T in tk.c2w]
n = int(2 * HALF / PX); g = (np.arange(n) + 0.5) * PX - HALF
UP = np.array([0, 0, -1.0])
stack, vstack, info = [], [], []
for i in range(len(tk.T)):
    # surface height: depth points near the previous centre (column r < 35 mm), bright, mode of z
    zs = []
    for c, k, dt in tk.sync(i):
        P, C, _ = cloud(tk, c, k, 2)
        m = (np.hypot(P[:, 0] - c_prev[0], P[:, 1] - c_prev[1]) < 0.035) & (np.abs(P[:, 2] - z_prev) < 0.03)
        if m.sum() == 0: continue
        br = C[m].mean(1) > 90
        zs.append(P[m][br, 2])
    zs = np.concatenate(zs) if zs else np.array([])
    if len(zs) < 200: continue
    h, e = np.histogram(zs, bins=np.arange(z_prev - 0.03, z_prev + 0.03, 0.001)); z = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    X, Y = np.meshgrid(c_prev[0] + g, c_prev[1] + g)
    Pw = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    cols, vis = [], []
    for c, k, dt in tk.sync(i):
        K = tk.K[c]; Pc = Pw @ W2C[c][:3, :3].T + W2C[c][:3, 3]
        u = Pc[:, 0] / Pc[:, 2] * K[0, 0] + K[0, 2]; v = Pc[:, 1] / Pc[:, 2] * K[1, 1] + K[1, 2]
        ok = (u >= 0) & (u < 847) & (v >= 0) & (v < 479) & (Pc[:, 2] > 0)
        img = tk.color(c, k); dep = tk.depth(c, k).astype(float) * 1e-3
        ui, vi = np.clip(u.astype(int), 0, 847), np.clip(v.astype(int), 0, 479); dm = dep[vi, ui]
        vis.append(ok & ((dm == 0) | (dm > Pc[:, 2] - OCC))); cols.append(img[vi, ui].astype(float))
    C = np.stack(cols); V = np.stack(vis)
    top = np.nanmedian(np.where(V[..., None], C, np.nan), 0).reshape(n, n, 3); seen = V.any(0).reshape(n, n)
    img = np.nan_to_num(top).astype(np.uint8)
    fit = ransac_disc(img, seen, PX)
    if fit is None or fit[3] < 180: continue
    cx, cy, r, cov = fit
    # resample to disc coordinates [-1.15, 1.15]^2
    s = np.linspace(-1.15, 1.15, NOUT); U, Vv = np.meshgrid(s, s)
    mapx = (cx + U * r).astype(np.float32); mapy = (cy + Vv * r).astype(np.float32)
    stack.append(cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR).astype(float))
    vstack.append(cv2.remap(seen.astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST).astype(bool))
    c_prev = c_prev + (np.array([cx, cy]) + 0.5) * PX - HALF; z_prev = z
    info.append(dict(t=float(tk.T[i]), R_mm=float(r * PX * 1e3), cover=int(cov)))
S = np.stack(stack); Vs = np.stack(vstack)
comp = np.nanmedian(np.where(Vs[..., None], S, np.nan), 0); valid = Vs.sum(0) >= max(3, len(S) // 4)
s = np.linspace(-1.15, 1.15, NOUT); U, Vv = np.meshgrid(s, s); inside = U ** 2 + Vv ** 2 <= 1.0
img = np.nan_to_num(comp).astype(np.uint8)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
L = lab[..., 0]; bch = lab[..., 2]
# white foam: bright and low yellowness relative to the crema inside the disc (Otsu on L - 0.8*(b-128))
score = L - 0.8 * (bch - 128)
sc = score[inside & valid].astype(np.float32)
thr, _ = cv2.threshold(np.clip((sc - sc.min()) / (np.ptp(sc) + 1e-9) * 255, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thr = sc.min() + thr / 255 * np.ptp(sc)
white = inside & valid & (score > thr)
white = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
os.makedirs(f"{B}/targets", exist_ok=True)
np.savez_compressed(f"{B}/targets/{p}.npz", img=img, valid=valid & inside, white=white, extent=1.15, info=json.dumps(info))
vis = img.copy(); vis[~(valid & inside)] //= 3
ov = vis.copy(); ov[white] = (0.5 * ov[white] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
cv2.circle(vis, (NOUT // 2, NOUT // 2), int(NOUT / 2 / 1.15), (0, 255, 0), 1)
cv2.imwrite(f"{B}/targets/{p}.png", np.hstack([vis, ov]))
print(p, "frames used", len(S), "R mm", np.median([q["R_mm"] for q in info]).round(1), "white frac in disc", round(white.sum() / (inside & valid).sum(), 3), "valid frac", round((valid & inside).sum() / inside.sum(), 3))
