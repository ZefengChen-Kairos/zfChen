"""Target v3: one camera (default: the most top-down), no cross-camera blending (no ghosting).
Per post-pour frame: surface height from depth; this camera's image remapped bilinearly onto the horizontal plane
(0.2 mm grid, 140 mm box); liquid disc by RANSAC in that rectified image; resample to disc coordinates (OUT x OUT);
ECC (Euclidean) alignment of all frames to the middle frame; per-pixel median (removes moving highlights); white mask
by Otsu on lightness inside r < 0.92."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from trimrec import Take, cloud
from disc_fit import ransac_disc

p = sys.argv[1]; CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OUT, EXT, PX, HALF, OCC = 512, 1.05, 0.0002, 0.07, 0.010
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
tk = Take(f"{B}/post/{p}"); D = json.load(open(f"{B}/{p}/disc.json"))
c_prev = np.array(D[-1]["disc_centre"]); z_prev = D[-1]["surf_z"]
W2C = np.linalg.inv(tk.c2w[CAM]); K = tk.K[CAM]
n = int(2 * HALF / PX); g = (np.arange(n) + 0.5) * PX - HALF
s = np.linspace(-EXT, EXT, OUT); U, V = np.meshgrid(s, s); RR = np.hypot(U, V)
frames, masks = [], []
for k, t in zip(tk.steps[CAM], tk.ts[CAM]):
    P, C, _ = cloud(tk, CAM, k, 2)
    m = (np.hypot(P[:, 0] - c_prev[0], P[:, 1] - c_prev[1]) < 0.035) & (np.abs(P[:, 2] - z_prev) < 0.03)
    zz = P[m][C[m].mean(1) > 90, 2] if m.sum() else np.array([])
    if len(zz) < 100: continue
    h, e = np.histogram(zz, bins=np.arange(z_prev - 0.03, z_prev + 0.03, 0.001)); z = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    X, Y = np.meshgrid(c_prev[0] + g, c_prev[1] + g)
    Pw = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1); Pc = Pw @ W2C[:3, :3].T + W2C[:3, 3]
    u = (Pc[:, 0] / Pc[:, 2] * K[0, 0] + K[0, 2]).reshape(n, n).astype(np.float32)
    v = (Pc[:, 1] / Pc[:, 2] * K[1, 1] + K[1, 2]).reshape(n, n).astype(np.float32)
    im = tk.color(CAM, k); dep = tk.depth(CAM, k).astype(np.float32) * 1e-3
    top = cv2.remap(im, u, v, cv2.INTER_LINEAR)
    dm = cv2.remap(dep, u, v, cv2.INTER_NEAREST)
    seen = (u >= 0) & (u < 847) & (v >= 0) & (v < 479) & ((dm == 0) | (dm > Pc[:, 2].reshape(n, n) - OCC))
    small = cv2.resize(top, (n // 2, n // 2), interpolation=cv2.INTER_AREA); ss = cv2.resize(seen.astype(np.uint8), (n // 2, n // 2), interpolation=cv2.INTER_NEAREST).astype(bool)
    fit = ransac_disc(small, ss, PX * 2)
    if fit is None or fit[3] < 200: continue
    cx, cy, r, cov = (fit[0] * 2, fit[1] * 2, fit[2] * 2, fit[3])
    mx = (cx + U * r).astype(np.float32); my = (cy + V * r).astype(np.float32)
    frames.append(cv2.remap(top, mx, my, cv2.INTER_LINEAR)); masks.append(cv2.remap(seen.astype(np.uint8), mx, my, cv2.INTER_NEAREST).astype(bool))
    c_prev = c_prev + (np.array([cx, cy]) + 0.5) * PX - HALF; z_prev = z
disc = (RR <= 0.97).astype(np.uint8)
ref = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2GRAY).astype(np.float32)
al, am = [], []
for f, mk in zip(frames, masks):
    gimg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32); M = np.eye(2, 3, dtype=np.float32)
    try:
        _, M = cv2.findTransformECC(ref, gimg, M, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), (disc & mk).astype(np.uint8), 5)
    except cv2.error:
        continue
    al.append(cv2.warpAffine(f, M, (OUT, OUT), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP).astype(np.float32))
    am.append(cv2.warpAffine(mk.astype(np.uint8), M, (OUT, OUT), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP).astype(bool))
A = np.stack(al); Mk = np.stack(am)
img = np.clip(np.nan_to_num(np.nanmedian(np.where(Mk[..., None], A, np.nan), 0)), 0, 255).astype(np.uint8)
valid = (Mk.sum(0) >= max(3, len(al) // 3)) & (RR <= 1.0)
img[RR > 1.0] = 0
Lc = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]; core = valid & (RR < 0.92)
t, _ = cv2.threshold(Lc[core].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
white = cv2.morphologyEx((valid & (Lc > t)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
os.makedirs(f"{B}/targets3", exist_ok=True)
np.savez_compressed(f"{B}/targets3/{p}_cam{CAM}.npz", img=img, valid=valid, white=white, extent=EXT, cam=CAM, n_frames=len(al))
cv2.imwrite(f"{B}/targets3/{p}_cam{CAM}.png", img)
ov = img.copy(); ov[white] = (0.55 * ov[white] + 0.45 * np.array([255, 0, 255])).astype(np.uint8)
cv2.imwrite(f"{B}/targets3/{p}_cam{CAM}_mask.png", ov)
print(p, "cam", CAM, "frames", len(frames), "aligned", len(al), "white frac", round(white.sum() / max(valid.sum(), 1), 3), "valid", round(valid.sum() / (RR <= 1).sum(), 3))
