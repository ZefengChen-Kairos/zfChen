"""Target v2: sharper, better aligned finished-pattern image.
Per post-pour frame: liquid plane (depth) + disc (RANSAC on a coarse top view) as before; then per camera a bilinear
remap of the disc region to disc coordinates (0.25 mm-equivalent grid, OUT x OUT), with an occlusion mask and a
per-pixel obliqueness weight cos(view ray, vertical). Per camera: frames registered to that camera's middle frame by
ECC (Euclidean: cup rotation in the hand), temporal median. Cameras: colour matched to the most top-down camera
(per-channel mean/std inside the disc), fused by weight cos^4; CLAHE on L. Output targets2/<p>.npz/.png."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from trimrec import Take, cloud
from disc_fit import ransac_disc

p = sys.argv[1]; OUT = 512; EXT = 1.08
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
tk = Take(f"{B}/post/{p}"); D = json.load(open(f"{B}/{p}/disc.json"))
c_prev = np.array(D[-1]["disc_centre"]); z_prev = D[-1]["surf_z"]
PX, HALF, OCC = 0.0005, 0.07, 0.010
W2C = [np.linalg.inv(T) for T in tk.c2w]
n = int(2 * HALF / PX); g = (np.arange(n) + 0.5) * PX - HALF
s = np.linspace(-EXT, EXT, OUT); U, V = np.meshgrid(s, s); RR = np.hypot(U, V)
per_cam = {c: [] for c in range(tk.ncam)}
for i in range(len(tk.T)):
    zs = []
    syn = tk.sync(i)
    for c, k, dt in syn:
        P, C, _ = cloud(tk, c, k, 2)
        m = (np.hypot(P[:, 0] - c_prev[0], P[:, 1] - c_prev[1]) < 0.035) & (np.abs(P[:, 2] - z_prev) < 0.03)
        if m.sum(): zs.append(P[m][C[m].mean(1) > 90, 2])
    zs = np.concatenate(zs) if zs else np.array([])
    if len(zs) < 200: continue
    h, e = np.histogram(zs, bins=np.arange(z_prev - 0.03, z_prev + 0.03, 0.001)); z = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    # coarse top view for the disc fit (as target.py)
    X, Y = np.meshgrid(c_prev[0] + g, c_prev[1] + g); Pw = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    cols, vis = [], []
    imgs = {}
    for c, k, dt in syn:
        K = tk.K[c]; Pc = Pw @ W2C[c][:3, :3].T + W2C[c][:3, 3]
        u = Pc[:, 0] / Pc[:, 2] * K[0, 0] + K[0, 2]; v = Pc[:, 1] / Pc[:, 2] * K[1, 1] + K[1, 2]
        ok = (u >= 0) & (u < 847) & (v >= 0) & (v < 479)
        im = tk.color(c, k); dep = tk.depth(c, k).astype(float) * 1e-3; imgs[c] = (im, dep)
        ui, vi = np.clip(u.astype(int), 0, 847), np.clip(v.astype(int), 0, 479); dm = dep[vi, ui]
        vis.append(ok & ((dm == 0) | (dm > Pc[:, 2] - OCC))); cols.append(im[vi, ui].astype(float))
    Cc = np.stack(cols); Vv = np.stack(vis)
    top = np.nan_to_num(np.nanmedian(np.where(Vv[..., None], Cc, np.nan), 0)).reshape(n, n, 3).astype(np.uint8)
    fit = ransac_disc(top, Vv.any(0).reshape(n, n), PX)
    if fit is None or fit[3] < 180: continue
    cx, cy, r, cov = fit
    cen = c_prev + (np.array([cx, cy]) + 0.5) * PX - HALF; R = r * PX
    c_prev, z_prev = cen, z
    # fine per-camera remap in disc coordinates
    Pw2 = np.stack([cen[0] + U.ravel() * R, cen[1] + V.ravel() * R, np.full(U.size, z)], 1)
    for c, k, dt in syn:
        im, dep = imgs[c]; K = tk.K[c]; Pc = Pw2 @ W2C[c][:3, :3].T + W2C[c][:3, 3]
        u = (Pc[:, 0] / Pc[:, 2] * K[0, 0] + K[0, 2]).reshape(OUT, OUT).astype(np.float32)
        v = (Pc[:, 1] / Pc[:, 2] * K[1, 1] + K[1, 2]).reshape(OUT, OUT).astype(np.float32)
        warped = cv2.remap(im, u, v, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        dm = cv2.remap(dep.astype(np.float32), u, v, cv2.INTER_NEAREST)
        inb = (u >= 0) & (u < 847) & (v >= 0) & (v < 479)
        visc = inb & ((dm == 0) | (dm > Pc[:, 2].reshape(OUT, OUT) - OCC))
        ray = Pw2 - tk.c2w[c][:3, 3]; ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        cosv = np.abs(ray[:, 2]).reshape(OUT, OUT)
        per_cam[c].append((warped, visc, cosv))
# per camera: ECC registration to the middle frame, temporal median
cam_img, cam_w = {}, {}
disc = (RR <= 1.0).astype(np.uint8)
for c, L in per_cam.items():
    if len(L) < 3: continue
    ref = cv2.cvtColor(L[len(L) // 2][0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    al, am = [], []
    for w, vm, cs in L:
        gimg = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY).astype(np.float32); M = np.eye(2, 3, dtype=np.float32)
        try:
            _, M = cv2.findTransformECC(ref, gimg, M, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), (disc * vm).astype(np.uint8), 5)
        except cv2.error:
            pass
        al.append(cv2.warpAffine(w, M, (OUT, OUT), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP).astype(np.float32))
        am.append(cv2.warpAffine(vm.astype(np.uint8), M, (OUT, OUT), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP).astype(bool))
    A = np.stack(al); Mk = np.stack(am)
    med = np.nanmedian(np.where(Mk[..., None], A, np.nan), 0)
    cnt = Mk.sum(0)
    cam_img[c] = med; cam_w[c] = np.where(cnt >= max(3, len(L) // 3), L[0][2] ** 4, 0.0)
# colour match to the most top-down camera, fuse
ref_c = max(cam_w, key=lambda c: (cam_w[c] * disc).sum())
inside = disc.astype(bool)
acc = np.zeros((OUT, OUT, 3)); wsum = np.zeros((OUT, OUT))
for c, im in cam_img.items():
    im = np.nan_to_num(im); ok = inside & (cam_w[c] > 0) & (cam_w[ref_c] > 0)
    if c != ref_c and ok.sum() > 1000:
        for ch in range(3):
            a, b = im[..., ch][ok], np.nan_to_num(cam_img[ref_c])[..., ch][ok]
            im[..., ch] = (im[..., ch] - a.mean()) / (a.std() + 1e-6) * b.std() + b.mean()
    acc += im * cam_w[c][..., None]; wsum += cam_w[c]
fused = np.clip(acc / np.maximum(wsum, 1e-9)[..., None], 0, 255).astype(np.uint8)
valid = (wsum > 0) & inside
lab = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB); clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab[..., 0] = clahe.apply(lab[..., 0]); out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
out[~(RR <= 1.0)] = 0
# white mask: Otsu on L inside r < 0.92
Lc = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0].astype(float); core = valid & (RR < 0.92)
v8 = Lc[core].astype(np.uint8); t, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
white = valid & (Lc > t)
white = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
os.makedirs(f"{B}/targets2", exist_ok=True)
np.savez_compressed(f"{B}/targets2/{p}.npz", img=out, valid=valid, white=white, extent=EXT, ref_cam=ref_c,
                    n_frames={str(c): len(L) for c, L in per_cam.items()})
cv2.imwrite(f"{B}/targets2/{p}.png", out)
ov = out.copy(); ov[white] = (0.55 * ov[white] + 0.45 * np.array([255, 0, 255])).astype(np.uint8)
cv2.imwrite(f"{B}/targets2/{p}_mask.png", ov)
print(p, "frames/cam", {c: len(L) for c, L in per_cam.items()}, "ref cam", ref_c, "white frac", round(white.sum() / valid.sum(), 3))
