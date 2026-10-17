"""White-foam mask of a target: Otsu on LAB lightness inside r < 0.92 (crema and foam are two clear modes)."""
import sys, json
import numpy as np, cv2
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours/targets"
for p in sys.argv[1:]:
    Z = dict(np.load(f"{B}/{p}.npz")); img = Z["img"]; valid = Z["valid"]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    w = lab[..., 0]
    sg = np.linspace(-1.15, 1.15, img.shape[0]); U, V = np.meshgrid(sg, sg)
    core = valid & (np.hypot(U, V) < 0.92)
    v = w[core]; u8 = np.clip((v - v.min()) / (np.ptp(v) + 1e-9) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); thr = v.min() + t / 255 * np.ptp(v)
    white = valid & (w > thr)
    white = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    Z["white"] = white; Z["whiteness"] = w.astype(np.float32)
    np.savez_compressed(f"{B}/{p}.npz", **Z)
    vis = img.copy(); vis[~valid] //= 3; ov = vis.copy(); ov[white] = (0.5 * ov[white] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.imwrite(f"{B}/{p}.png", np.hstack([vis, ov]))
    print(p, "white frac", round(white.sum() / valid.sum(), 3))
