"""Multi-frame super-resolution target (iterative back-projection) from one camera's post-pour frames.
Geometry per frame as in target3 (surface plane from depth, disc fit on the rectified image, ECC alignment to the
middle frame); each camera pixel that sees the disc is mapped by its ray -> plane -> disc coords -> aligned HR pixel.
Observation model: camera pixel = Gaussian-blurred HR image (sigma ~ 0.45 x camera footprint) sampled at that point.
IBP: H <- H + lam * splat(robust residual) / splat(weights); residuals clipped (Huber-like) to suppress highlights."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from scipy.ndimage import map_coordinates, gaussian_filter
from trimrec import Take, cloud
from disc_fit import ransac_disc

p = sys.argv[1]; CAMS = [int(c) for c in sys.argv[2].split(",")]; HR = int(sys.argv[3]) if len(sys.argv) > 3 else 512; CAM = CAMS[0]
ALPHA = float(os.environ.get("ALPHA", "0.3")); NIT = int(os.environ.get("NIT", "8"))
EXT, PX, HALF, OCC, OUT = 1.05, 0.0002, 0.07, 0.010, 512
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
tk = Take(f"{B}/post/{p}"); D = json.load(open(f"{B}/{p}/disc.json"))
c_prev = np.array(D[-1]["disc_centre"]); z_prev = D[-1]["surf_z"]
s = np.linspace(-EXT, EXT, OUT); U, V = np.meshgrid(s, s); RR = np.hypot(U, V)
n = int(2 * HALF / PX); g = (np.arange(n) + 0.5) * PX - HALF
frames = []
for CAMI in CAMS:
  T = tk.c2w[CAMI]; W2C = np.linalg.inv(T); K = tk.K[CAMI]
  c_prev = np.array(D[-1]["disc_centre"]); z_prev = D[-1]["surf_z"]
  for k in tk.steps[CAMI]:
      P, C, _ = cloud(tk, CAMI, k, 2)
      m = (np.hypot(P[:, 0] - c_prev[0], P[:, 1] - c_prev[1]) < 0.035) & (np.abs(P[:, 2] - z_prev) < 0.03)
      zz = P[m][C[m].mean(1) > 90, 2] if m.sum() else np.array([])
      if len(zz) < 100: continue
      h, e = np.histogram(zz, bins=np.arange(z_prev - 0.03, z_prev + 0.03, 0.001)); z = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
      X, Y = np.meshgrid(c_prev[0] + g, c_prev[1] + g)
      Pw = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1); Pc = Pw @ W2C[:3, :3].T + W2C[:3, 3]
      u = (Pc[:, 0] / Pc[:, 2] * K[0, 0] + K[0, 2]).reshape(n, n).astype(np.float32)
      v = (Pc[:, 1] / Pc[:, 2] * K[1, 1] + K[1, 2]).reshape(n, n).astype(np.float32)
      im = tk.color(CAMI, k); dep = tk.depth(CAMI, k).astype(np.float32) * 1e-3
      top = cv2.remap(im, u, v, cv2.INTER_LINEAR); dm = cv2.remap(dep, u, v, cv2.INTER_NEAREST)
      seen = (u >= 0) & (u < 847) & (v >= 0) & (v < 479) & ((dm == 0) | (dm > Pc[:, 2].reshape(n, n) - OCC))
      small = cv2.resize(top, (n // 2, n // 2), interpolation=cv2.INTER_AREA); ss = cv2.resize(seen.astype(np.uint8), (n // 2, n // 2), interpolation=cv2.INTER_NEAREST).astype(bool)
      fit = ransac_disc(small, ss, PX * 2)
      if fit is None or fit[3] < 200: continue
      cx, cy, r = fit[0] * 2, fit[1] * 2, fit[2] * 2
      mx = (cx + U * r).astype(np.float32); my = (cy + V * r).astype(np.float32)
      frames.append(dict(cam=CAMI, k=k, z=z, c_prev=c_prev.copy(), cx=cx, cy=cy, r=r, img=im, dep=dep,
                         disc=cv2.remap(top, mx, my, cv2.INTER_LINEAR), mask=cv2.remap(seen.astype(np.uint8), mx, my, cv2.INTER_NEAREST).astype(bool)))
      c_prev = c_prev + (np.array([cx, cy]) + 0.5) * PX - HALF; z_prev = z
# ECC alignment (as target3)
discm = (RR <= 0.97).astype(np.uint8)
f0 = [f for f in frames if f["cam"] == CAM]; reff = f0[len(f0) // 2]
ref = cv2.cvtColor(reff["disc"], cv2.COLOR_BGR2GRAY).astype(np.float32)
for f in frames:
    M = np.eye(2, 3, dtype=np.float32)
    try:
        _, M = cv2.findTransformECC(ref, cv2.cvtColor(f["disc"], cv2.COLOR_BGR2GRAY).astype(np.float32), M, cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), (discm & f["mask"]).astype(np.uint8), 5)
        f["M"] = M
    except cv2.error:
        f["M"] = None
frames = [f for f in frames if f["M"] is not None]
# camera pixel -> HR pixel per frame
step = 2 * EXT / (OUT - 1); hr_scale = (OUT - 1) / (HR - 1)
obs = []
vv, uu = np.mgrid[0:480, 0:848]
GEO = {}
for c in CAMS:
    Tc = tk.c2w[c]; rc = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], 1) @ np.linalg.inv(tk.K[c]).T
    GEO[c] = (Tc[:3, 3], rc @ Tc[:3, :3].T, np.linalg.inv(Tc))
# colour gain/offset of each camera to the reference camera (aligned disc images)
ALN = {}
for f in frames: ALN.setdefault(f['cam'], []).append(cv2.warpAffine(f['disc'], f['M'], (OUT, OUT), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP).astype(float))
MED = {c: np.median(np.stack(v), 0) for c, v in ALN.items()}
core = RR < 0.9
GAIN = {}
for c in CAMS:
    GAIN[c] = [np.polyfit(MED[c][..., ch][core], MED[CAM][..., ch][core], 1) for ch in range(3)]
for f in frames:
    o, dirs, W2C = GEO[f['cam']]
    tpar = (f["z"] - o[2]) / dirs[:, 2]; Pp = o + tpar[:, None] * dirs
    idx = (Pp[:, :2] - f["c_prev"] + HALF) / PX - 0.5
    Ud = (idx[:, 0] - f["cx"]) / f["r"]; Vd = (idx[:, 1] - f["cy"]) / f["r"]
    q = np.stack([(Ud + EXT) / step, (Vd + EXT) / step, np.ones_like(Ud)], 1)
    Mi = cv2.invertAffineTransform(f["M"]); xh = q @ Mi.T                       # aligned OUT-grid pixel
    xh = xh / hr_scale                                                            # -> HR grid pixel
    rr = np.hypot(Ud, Vd); expz = (Pp @ W2C[:3, :3].T + W2C[:3, 3])[:, 2]
    dm = f["dep"].ravel()
    ok = (rr < 0.985) & (tpar > 0) & ((dm == 0) | (dm > expz - OCC))
    I = f["img"].reshape(-1, 3)[ok].astype(np.float64)
    I = np.stack([np.polyval(GAIN[f["cam"]][ch], I[:, ch]) for ch in range(3)], 1)
    obs.append((xh[ok], I))
# camera footprint in HR pixels (median local scale)
x0 = obs[len(obs) // 2][0]
foot = None
f = reff; o, dirs, W2C = GEO[CAM]
tpar = (f["z"] - o[2]) / dirs[:, 2]; Pp = o + tpar[:, None] * dirs
pp = Pp.reshape(480, 848, 3)
du = np.linalg.norm(np.diff(pp, axis=1), axis=2); foot_m = float(np.median(du[200:300, 350:500]))
hr_px_m = (2 * EXT / (HR - 1)) * f["r"] * PX
sig = 0.45 * foot_m / hr_px_m
print(p, "cams", CAMS, "frames", len(frames), "camera footprint mm", round(foot_m * 1e3, 2), "HR px mm", round(hr_px_m * 1e3, 3), "psf sigma HRpx", round(sig, 2), flush=True)
# init: v3 target resampled
H = cv2.resize(MED[CAM].astype(np.float32), (HR, HR), interpolation=cv2.INTER_CUBIC).astype(np.float64)
def sample(img, xy):
    return np.stack([map_coordinates(img[..., ch], [xy[:, 1], xy[:, 0]], order=1, mode="nearest") for ch in range(3)], 1)
for it in range(NIT):
    Hb = np.stack([gaussian_filter(H[..., ch], sig) for ch in range(3)], -1)
    num = np.zeros((HR, HR, 3)); den = np.zeros((HR, HR))
    for xh, I in obs:
        res = I - sample(Hb, xh)
        res = np.clip(res, -25, 25)                                   # robust: highlights / outliers
        ix = np.floor(xh).astype(int); fx = xh - ix
        for dx in (0, 1):
            for dy in (0, 1):
                w = (fx[:, 0] if dx else 1 - fx[:, 0]) * (fx[:, 1] if dy else 1 - fx[:, 1])
                jx, jy = np.clip(ix[:, 0] + dx, 0, HR - 1), np.clip(ix[:, 1] + dy, 0, HR - 1)
                np.add.at(den, (jy, jx), w)
                for ch in range(3): np.add.at(num[..., ch], (jy, jx), w * res[:, ch])
    upd = np.stack([gaussian_filter(num[..., ch], sig) for ch in range(3)], -1) / np.maximum(gaussian_filter(den, sig), 1e-6)[..., None]
    H = H + upd
    H = np.clip(H - ALPHA * (H - np.stack([gaussian_filter(H[..., ch], 0.8) for ch in range(3)], -1)), 0, 255)
    print("iter", it, "mean |res|", round(float(np.mean(np.abs(upd[den > 0.5]))), 3), flush=True)
sH = np.linspace(-EXT, EXT, HR); UH, VH = np.meshgrid(sH, sH); RH = np.hypot(UH, VH)
out = H.astype(np.uint8); out[RH > 1.0] = 0
os.makedirs(f"{B}/targets_sr", exist_ok=True)
tag = "".join(map(str, CAMS))
cv2.imwrite(f"{B}/targets_sr/{p}_cam{tag}_sr.png", out)
np.savez_compressed(f"{B}/targets_sr/{p}_cam{tag}_sr.npz", img=out, extent=EXT, cam=CAM, n_frames=len(frames), psf_sigma=sig, foot_mm=foot_m * 1e3)
