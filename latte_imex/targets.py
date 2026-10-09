"""Target milk-fraction fields extracted from the reference clips.

For each pattern one clean end frame is chosen by hand (time in the clip); the liquid surface is
found as the largest orange/brown connected region, its convex hull is fitted with an ellipse, and
the ellipse is affinely mapped to the unit cup (weak-perspective view of a circle).  Inside the cup
the saturation is mapped to a whiteness w in [0, 1] (w=1 milk foam, w=0 crema).  The cup frame is
the solver frame (x right, y up, barista at y<0); each frame is rotated by hand so the pattern axis
is the y axis with the cut exit (heart tip / stem) at +y.
"""
import os, sys, json, math
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.colors import rgb_to_hsv

HERE = os.path.dirname(__file__)
TDIR = os.path.join(HERE, "targets")
REF = os.environ.get("LATTE_REF", "/tmp/claude-0/-home-user-zfChen/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/ref")
# method "otsu2": two-class Otsu split of (1-S)V (foam vs crema).  method "foam3": three-class split (dark crema, light-tan
# crema, foam) so that a pale crema halo is not counted as foam; small white blobs touching the rim (reflections) removed.

# clip time [s], rotation [deg, counter-clockwise in the cup frame], mirror x
FRAMES = dict(
    heart=dict(t=11.6, rot=-125.0, flip=False),
    push_heart=dict(t=16.4, rot=-100.0, flip=False),
    layered_heart=dict(t=11.8, rot=-130.0, flip=False),
    tulip=dict(t=17.75, rot=-165.0, flip=False),
    leaf=dict(t=12.2, rot=45.0, flip=False, method="foam3", M=256),   # 10-09: was 14.4 s (watermark over the leaf, glare)
    swan=dict(t=20.5, rot=0.0, flip=False),
)


def grab(name, t, path):
    import subprocess
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t}", "-i", os.path.join(REF, f"reference_{name}.mp4"),
                    "-frames:v", "1", path], check=True)


def load_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def fit_ellipse(pts):
    """Direct least-squares conic fit (Fitzgibbon) -> center, semi-axes, angle."""
    x, y = pts[:, 0], pts[:, 1]
    mx, my = x.mean(), y.mean(); sx, sy = x.std(), y.std()
    x = (x - mx) / sx; y = (y - my) / sy
    D = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], 1)
    S = D.T @ D
    C = np.zeros((6, 6)); C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
    w, v = np.linalg.eig(np.linalg.solve(S, C))
    k = np.argmax(np.real(w)); a = np.real(v[:, k])
    A, B, Cc, Dd, E, F = a
    # back to pixel coordinates
    A, B, Cc = A / sx ** 2, B / (sx * sy), Cc / sy ** 2
    Dd2 = Dd / sx - 2 * A * mx - B * my; E2 = E / sy - 2 * Cc * my - B * mx
    F2 = F + A * mx * mx + B * mx * my + Cc * my * my - Dd * mx / sx - E * my / sy
    A, B, Cc, Dd, E, F = A, B, Cc, Dd2, E2, F2
    M = np.array([[A, B / 2], [B / 2, Cc]])
    c = np.linalg.solve(2 * M, -np.array([Dd, E]))
    k = -(c @ M @ c + np.array([Dd, E]) @ c + F)
    ev, evec = np.linalg.eigh(M)
    axes = np.sqrt(k / ev)
    ang = math.atan2(evec[1, 0], evec[0, 0])
    return c, axes, ang, evec


def liquid_ellipse(rgb):
    hsv = rgb_to_hsv(rgb)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    brown = (H > 0.02) & (H < 0.13) & (S > 0.3) & (V > 0.25)
    brown = ndimage.binary_opening(brown, iterations=2)
    lab, n = ndimage.label(brown)
    sizes = ndimage.sum(brown, lab, range(1, n + 1))
    comp = lab == (1 + int(np.argmax(sizes)))
    comp = ndimage.binary_fill_holes(comp)
    edge = comp & ~ndimage.binary_erosion(comp)
    iy, ix = np.nonzero(edge)
    pts = np.stack([ix, iy], 1).astype(float)
    hull = ConvexHull(pts)
    poly = pts[hull.vertices]
    dense = []
    for p, q in zip(poly, np.roll(poly, -1, 0)):
        n_ = max(2, int(np.hypot(*(q - p)) / 2))
        dense.append(p[None] + (q - p)[None] * np.linspace(0, 1, n_, endpoint=False)[:, None])
    dense = np.concatenate(dense)
    c, axes, ang, evec = fit_ellipse(dense)
    return dict(center=c, axes=axes, angle=ang, evec=evec, comp=comp)


def warp(rgb, ell, M=192, rot=0.0, flip=False):
    """Sample the image on the cup grid [iy, ix] (y up) of M x M cells covering [-0.5, 0.5]^2."""
    g = (np.arange(M) + 0.5) / M - 0.5
    X, Y = np.meshgrid(g, g)
    if flip:
        X = -X
    th = math.radians(rot)
    Xr = math.cos(th) * X - math.sin(th) * Y; Yr = math.sin(th) * X + math.cos(th) * Y
    # cup frame -> image: the image y axis points down, so the cup's +y maps to -row
    u = 2 * Xr; v = -2 * Yr
    ex, ey = ell["evec"][:, 0], ell["evec"][:, 1]
    a, b = ell["axes"]
    px = ell["center"][0] + a * u * ex[0] + b * v * ey[0]
    py = ell["center"][1] + a * u * ex[1] + b * v * ey[1]
    out = np.stack([ndimage.map_coordinates(rgb[..., k], [py, px], order=1, mode="nearest") for k in range(3)], -1)
    return out, np.hypot(X, Y) <= 0.49


def whiteness(rgb_w, mask):
    hsv = rgb_to_hsv(np.clip(rgb_w, 0, 1))
    S, V = hsv[..., 1], hsv[..., 2]
    f = (1 - S) * V          # bright and unsaturated -> foam
    vals = f[mask]
    # Otsu threshold
    hist, edges = np.histogram(vals, 128)
    mids = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1); m1 = (np.cumsum((hist * mids)[::-1])[::-1] / np.maximum(w1, 1))
    m1 = np.roll(m1, -1)
    var = w0[:-1] * w1[:-1] * (m0[:-1] - m1[:-1]) ** 2
    thr = mids[int(np.argmax(var))]
    lo = np.median(vals[vals < thr]); hi = np.median(vals[vals >= thr])
    w = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)
    w = np.where(mask, w, 0.0)
    return w, thr, lo, hi


def _kmeans3(v, it=60):
    c = np.quantile(v, [0.1, 0.5, 0.9])
    for _ in range(it):
        lab = np.argmin(np.abs(v[:, None] - c[None, :]), 1)
        c = np.array([v[lab == j].mean() if np.any(lab == j) else c[j] for j in range(3)])
    return np.sort(c)


def whiteness_foam3(rgb_w, mask, ramp=1.0, min_blob=0.01):
    """Three-class split of f = (1-S)V into dark crema / light-tan crema / foam; w ramps between the tan and foam centres.
    White blobs smaller than min_blob of the cup that touch the rim (reflections) are removed."""
    hsv = rgb_to_hsv(np.clip(rgb_w, 0, 1))
    f = (1 - hsv[..., 1]) * hsv[..., 2]
    c = _kmeans3(f[mask])
    mid = 0.5 * (c[1] + c[2]); half = ramp * 0.5 * (c[2] - c[1])
    w = np.where(mask, np.clip((f - (mid - half)) / (2 * half), 0, 1), 0.0)
    M = w.shape[0]; g = (np.arange(M) + 0.5) / M - 0.5; X, Y = np.meshgrid(g, g); rim = np.hypot(X, Y) > 0.46
    lab, n = ndimage.label(w > 0.5)
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() < min_blob * mask.sum() and (comp & rim).any():
            w[ndimage.binary_dilation(comp, iterations=2)] = 0.0
    return w, c[1], c[1], c[2]


def build(names=None, M=192, preview=True):
    os.makedirs(TDIR, exist_ok=True)
    names = names or list(FRAMES)
    rows = []
    for name in names:
        cfg = FRAMES[name]
        fpath = os.path.join(TDIR, f"{name}_frame.png")
        grab(name, cfg["t"], fpath)
        rgb = load_rgb(fpath)
        ell = liquid_ellipse(rgb)
        Mi = cfg.get("M", M)
        rgb_w, mask = warp(rgb, ell, Mi, cfg["rot"], cfg["flip"])
        if cfg.get("method", "otsu2") == "foam3":
            w, thr, lo, hi = whiteness_foam3(rgb_w, mask)
        else:
            w, thr, lo, hi = whiteness(rgb_w, mask)
        np.savez_compressed(os.path.join(TDIR, f"{name}.npz"), w=w.astype(np.float32), mask=mask, M=Mi,
                            t=cfg["t"], rot=cfg["rot"], flip=cfg["flip"], method=cfg.get("method", "otsu2"))
        rows.append((name, rgb, ell, rgb_w, w, mask))
        print(f"{name}: t={cfg['t']} center={ell['center'].round(1)} axes={ell['axes'].round(1)} "
              f"angle={math.degrees(ell['angle']):.1f} white_area={w[mask].mean():.3f} thr={thr:.3f}")
    if preview:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        fig, axes = plt.subplots(3, len(rows), figsize=(3 * len(rows), 9), dpi=100, squeeze=False)
        for j, (name, rgb, ell, rgb_w, w, mask) in enumerate(rows):
            ax = axes[0, j]; ax.imshow(rgb); ax.set_title(name)
            ax.add_patch(Ellipse(ell["center"], 2 * ell["axes"][0], 2 * ell["axes"][1], angle=math.degrees(ell["angle"]),
                                 fill=False, color="lime", lw=1.5)); ax.axis("off")
            ax = axes[1, j]; ax.imshow(np.where(mask[..., None], rgb_w, 0.2)[::-1]); ax.axis("off"); ax.set_title("warped (y up)")
            ax = axes[2, j]; ax.imshow(w[::-1], cmap="gray", vmin=0, vmax=1); ax.axis("off"); ax.set_title("whiteness w")
        fig.tight_layout(); fig.savefig(os.path.join(TDIR, "preview.png")); plt.close(fig)


def load(name):
    z = np.load(os.path.join(TDIR, f"{name}.npz"))
    return z["w"].astype(np.float64), z["mask"]


if __name__ == "__main__":
    build(sys.argv[1:] or None)
