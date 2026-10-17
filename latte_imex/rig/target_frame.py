"""Target from ONE hand-picked frame (the finished pattern, chosen by eye) of the raw recording: target3's
rectification on that frame only (surface height from depth -> camera image remapped onto the horizontal plane ->
RANSAC liquid disc -> disc coordinates OUT x OUT, extent EXT), white mask by Otsu on LAB lightness inside r < 0.92.
    python3 latte_imex/rig/target_frame.py <pattern> <cam> <step> [outdir]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from rs_zip import Rec, backproject
from disc_fit import ransac_disc

OUT, EXT, PX, HALF, OCC = 512, 1.05, 0.0002, 0.07, 0.010
SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "raw_data", "rig", "results")


def make(p, cam, k, outdir):
    r = Rec(f"{SCR}/zips/{p}.zip")
    D = json.load(open(os.path.join(RES, p, "disc.json")))
    c0, z0 = np.array(D[-1]["disc_centre"]), D[-1]["surf_z"]
    im = r.color(cam, k)
    dep = r.depth(cam, k).astype(np.float32) * 1e-3
    K, c2w = r.K[cam], r.c2w[cam]
    P, ok = backproject(dep, K, c2w, 2)
    # bright points of the liquid near the last known disc centre -> surface height and a refined centre
    H, W = dep.shape
    col = im[::2, ::2][ok]
    m = (np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1]) < 0.06) & (np.abs(P[:, 2] - z0) < 0.04) & (col.mean(1) > 90)
    zz = P[m, 2]
    h, e = np.histogram(zz, bins=np.arange(z0 - 0.04, z0 + 0.04, 0.001))
    zs = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    near = m & (np.abs(P[:, 2] - zs) < 0.004)
    c = np.median(P[near, :2], 0)
    W2C = np.linalg.inv(c2w)
    n = int(2 * HALF / PX)
    g = (np.arange(n) + 0.5) * PX - HALF
    X, Y = np.meshgrid(c[0] + g, c[1] + g)
    Pw = np.stack([X.ravel(), Y.ravel(), np.full(X.size, zs)], 1)
    Pc = Pw @ W2C[:3, :3].T + W2C[:3, 3]
    mu = (Pc[:, 0] / Pc[:, 2] * K[0, 0] + K[0, 2]).reshape(n, n).astype(np.float32)
    mv = (Pc[:, 1] / Pc[:, 2] * K[1, 1] + K[1, 2]).reshape(n, n).astype(np.float32)
    top = cv2.remap(im, mu, mv, cv2.INTER_CUBIC)
    dm = cv2.remap(dep, mu, mv, cv2.INTER_NEAREST)
    seen = (mu >= 0) & (mu < W - 1) & (mv >= 0) & (mv < H - 1) & ((dm == 0) | (dm > Pc[:, 2].reshape(n, n) - OCC))
    small = cv2.resize(top, (n // 2, n // 2), interpolation=cv2.INTER_AREA)
    ss = cv2.resize(seen.astype(np.uint8), (n // 2, n // 2), interpolation=cv2.INTER_NEAREST).astype(bool)
    fit = ransac_disc(small, ss, PX * 2)
    cx, cy, rr = fit[0] * 2, fit[1] * 2, fit[2] * 2
    s = np.linspace(-EXT, EXT, OUT)
    U, V = np.meshgrid(s, s)
    RR = np.hypot(U, V)
    mx, my = (cx + U * rr).astype(np.float32), (cy + V * rr).astype(np.float32)
    img = cv2.remap(top, mx, my, cv2.INTER_CUBIC)
    valid = cv2.remap(seen.astype(np.uint8), mx, my, cv2.INTER_NEAREST).astype(bool) & (RR <= 1.0)
    img[RR > 1.0] = 0
    # specular glare: small, near-saturated blobs (not the foam, whose white area is large) -> invalid, inpainted
    sat = ((img.min(2) > 235) & (RR <= 1.0)).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(sat, 8)
    disc_px = (RR <= 1.0).sum()
    glare = np.isin(lab, [i for i in range(1, nl) if st[i, cv2.CC_STAT_AREA] < 0.006 * disc_px])
    glare = cv2.dilate(glare.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & (RR <= 1.0)
    if glare.any():
        img = cv2.inpaint(img, glare.astype(np.uint8), 7, cv2.INPAINT_TELEA)
        valid &= ~glare
    Lc = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    core = valid & (RR < 0.92)
    t, _ = cv2.threshold(Lc[core].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = cv2.morphologyEx((valid & (Lc > t)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)        # drop specks (< 0.3 % of the disc)
    white = np.isin(lab, [i for i in range(1, nl) if st[i, cv2.CC_STAT_AREA] >= 0.003 * disc_px])
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(f"{outdir}/{p}.npz", img=img, valid=valid, white=white, extent=EXT, cam=cam, n_frames=1, step=k,
                        source=f"{p}.zip cam{cam} step {k} (hand-picked finished pattern)")
    cv2.imwrite(f"{outdir}/{p}.png", img)
    ov = img.copy()
    ov[white] = (0.55 * ov[white] + 0.45 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.imwrite(f"{outdir}/{p}_mask.png", ov)
    print(p, "cam", cam, "step", k, f"surface z {zs * 1e3:.1f} mm, disc R {rr * PX * 1e3:.1f} mm, white frac "
          f"{white.sum() / max(valid.sum(), 1):.3f}, glare {glare.sum() / disc_px:.4f}, valid {valid.sum() / (RR <= 1).sum():.3f}")


if __name__ == "__main__":
    make(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else f"{SCR}/targets_pick")
