"""Whole-video supervision targets: every rectified top-view frame of the pattern phase (topview.py: 3-camera fused
image of the liquid surface + 'seen' = some camera saw the point unoccluded) resampled onto the disc grid of the
targets (tulip_feas.Target convention: col = board x, row = board y, extent +-EXT disc radii), on the simulation
clock (t = 0 at the first frame of the pattern phase, as side_by_side.py / real_panel.py).

Per frame:
  img    RGB (uint8) of the surface, outside the disc 0
  L      LAB lightness (uint8)
  valid  observation mask (fixed by the observation, never by the model): seen by a camera, inside the disc
         (r <= 0.96: the meniscus at the wall is excluded), not a specular glare blob (small near-saturated
         components), not within FOOT_R disc radii of the spout tip (stream / impact zone: low confidence)
  t      frame time [s] on the simulation clock
Whiteness (exposure-invariant; the cameras' exposure and the overall brightness drift during a pour, so a fixed
lightness threshold calls the early crema white): two features per pixel,
    f  = (B - R) / (R + G + B)           chromaticity: the foam is a cold white, the crema a warm pink-brown
    Ln = L / median_valid(L) of the frame  lightness relative to the frame
plus LAB a*, b*, and a regularised logistic pixel classifier trained on the two frames whose labels are certain: the
last frame (finished pattern: the hand-checked target mask, targets/<p>.npz) and the first N_BROWN frames (no foam yet: all crema,
under the early lighting); w = its probability (soft whiteness, 0..1), applied unchanged to every frame.  Pixels
that disagree with the median of the +-TC neighbouring frames by more than 0.5 are dropped from that frame's valid
mask (transient misclassification: exposure jumps, reflections).
Writes raw_data/rig/video_obs/<p>.npz (+ <p>_sheet.jpg).  Run where the pipeline's top/ frames are.
    python latte_imex/rig/video_obs.py [pattern ...]"""
import json
import os
import sys

import cv2
import numpy as np

SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT, EXT = 128, 1.05
R_IN = 0.96              # disc radius used (the bright meniscus ring at the wall is excluded)
FOOT_R = 0.12            # [disc radii] exclusion around the spout tip while pouring
N_BROWN = 3              # first frames used as all-crema examples (no foam on the surface yet)
TC = 2                   # temporal-consistency window (frames each side)
SAT_W = 210.0            # min(R, G, B) above which a (non-glare) pixel is over-exposed foam


def frame(p, fi, D, U, V, RR):
    Z = np.load(f"{SCR}/{p}/top/{fi:04d}.npz")
    img, seen, c0, PX, HALF = Z["img"], Z["seen"], Z["c0"], float(Z["px"]), float(Z["half"])
    cen, R = np.array(D[fi]["disc_centre"]), D[fi]["disc_R"]
    mx = ((cen[0] + U * R - c0[0] + HALF) / PX - 0.5).astype(np.float32)
    my = ((cen[1] + V * R - c0[1] + HALF) / PX - 0.5).astype(np.float32)
    im = cv2.remap(img, mx, my, cv2.INTER_AREA if PX < 0 else cv2.INTER_LINEAR)
    sv = cv2.remap(seen.astype(np.uint8), mx, my, cv2.INTER_NEAREST).astype(bool)
    inside = RR <= R_IN
    valid = sv & inside & bool(D[fi].get("fit_ok", True))
    sat = ((im.min(2) > 235) & inside).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(sat, 8)
    glare = np.isin(lab, [i for i in range(1, nl) if st[i, cv2.CC_STAT_AREA] < 0.006 * inside.sum()])
    glare = cv2.dilate(glare.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    valid &= ~glare
    tip = D[fi].get("tip_rel_norm")
    if tip is not None:
        valid &= np.hypot(U - tip[0], V - tip[1]) > FOOT_R
    im[RR > 1.0] = 0
    L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0]
    return im[..., ::-1].copy(), L, valid


def features(imgs, Ls, vals):
    """[n, H, W, 4] exposure-insensitive pixel features: lightness / frame median lightness, chromaticity
    (B - R)/(R + G + B), LAB a*, b* (centred, /50)."""
    rgb = imgs.astype(float) + 1.0
    f = (rgb[..., 2] - rgb[..., 0]) / rgb.sum(-1)
    med = np.array([np.median(L[v]) if v.any() else 1.0 for L, v in zip(Ls, vals)])
    Ln = Ls.astype(float) / np.maximum(med, 1.0)[:, None, None]
    lab = np.array([cv2.cvtColor(np.ascontiguousarray(im[..., ::-1]), cv2.COLOR_BGR2LAB) for im in imgs]).astype(float)
    return np.stack([Ln, f, (lab[..., 1] - 128) / 50, (lab[..., 2] - 128) / 50], -1)


def logistic(X, y, lam=1.0, iters=100):
    """Ridge-regularised logistic regression (Newton) on standardised features; returns (w, mean, std)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(A.shape[1])
    R = lam * np.eye(len(w)); R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-A @ w))
        g = A.T @ (p - y) + R @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + R
        dw = np.linalg.solve(H, g)
        w -= dw
        if np.abs(dw).max() < 1e-8:
            break
    return w, mu, sd


def whiteness(F, model):
    w, mu, sd = model
    z = w[0] + ((F - mu) / sd) @ w[1:]
    return 1.0 / (1.0 + np.exp(-z))


def make(p):
    D = json.load(open(f"{SCR}/{p}/disc.json"))
    s = np.linspace(-EXT, EXT, OUT)
    U, V = np.meshgrid(s, s)
    RR = np.hypot(U, V)
    t0 = D[0]["t"]
    imgs, Ls, vals, ts = [], [], [], []
    for fi in range(len(D)):
        if not os.path.exists(f"{SCR}/{p}/top/{fi:04d}.npz"):
            continue
        im, L, v = frame(p, fi, D, U, V, RR)
        imgs.append(im); Ls.append(L); vals.append(v); ts.append(D[fi]["t"] - t0)
    L_last, v_last = Ls[-1], vals[-1]
    thr, _ = cv2.threshold(L_last[v_last].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Ls, vals, imgs = np.array(Ls), np.array(vals), np.array(imgs)
    feats = features(imgs, Ls, vals)
    # labels: last frame (Otsu white / crema) + the first N_BROWN frames (before the first white: all crema, early
    # lighting) -> a pixel classifier that has to tell foam from crema under both lightings
    T = np.load(f"{REPO}/raw_data/rig/targets/{p}.npz")          # hand-checked finished-pattern mask (512 grid)
    tn = T["white"].shape[0]
    ii = np.clip(np.round((np.linspace(-EXT, EXT, OUT) + float(T["extent"])) / (2 * float(T["extent"])) * (tn - 1)).astype(int), 0, tn - 1)
    tw = T["white"][np.ix_(ii, ii)].astype(bool)
    tv = T["valid"][np.ix_(ii, ii)].astype(bool) & v_last
    Xs = [feats[-1][tv]]
    ys = [tw[tv].astype(float)]
    for k in range(min(N_BROWN, len(ts))):
        Xs.append(feats[k][vals[k]])
        ys.append(np.zeros(vals[k].sum()))
    rng = np.random.default_rng(0)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    sel = rng.choice(len(X), min(len(X), 60000), replace=False)
    model = logistic(X[sel], y[sel])
    W = whiteness(feats, model)
    # over-exposed foam is neutral white, not the cold white the classifier learnt on the finished pattern; large
    # near-saturated areas are foam (small saturated blobs = glare were removed from the valid mask already)
    W = np.maximum(W, 1.0 / (1.0 + np.exp(-(imgs.min(-1).astype(float) - SAT_W) / 8.0)))
    W[~vals] = 0.0
    # temporal consistency: a pixel that disagrees with the median of its +-TC neighbouring frames by more than 0.5
    # is a transient misclassification (exposure jump, reflection) -> not observed in that frame
    n = len(W)
    Wm = np.where(vals, W, np.nan)
    flick = np.zeros(vals.shape, bool)
    for k in range(n):
        nb = Wm[max(0, k - TC):min(n, k + TC + 1)]
        med = np.nanmedian(nb, axis=0)
        flick[k] = np.isfinite(med) & (np.abs(W[k] - med) > 0.5) & vals[k]
    vals = vals & ~flick
    W[~vals] = 0.0
    prior = np.array([W[k][vals[k]].mean() if vals[k].any() else 0.0 for k in range(len(W))])
    os.makedirs(f"{REPO}/raw_data/rig/video_obs", exist_ok=True)
    fn = f"{REPO}/raw_data/rig/video_obs/{p}.npz"
    np.savez_compressed(fn, t=np.array(ts), img=imgs, L=Ls, valid=vals, w=(W * 255).round().astype(np.uint8),
                        white_frac=prior, model_w=model[0], model_mu=model[1], model_sd=model[2], Lthr=float(thr), extent=EXT, R_in=R_IN, foot_r=FOOT_R,
                        source=f"{p}: topview.py frames, disc.json; w = whiteness * 255")
    # contact sheet: 8 frames, image | whiteness with the invalid area hatched dark
    pick = np.linspace(0, len(ts) - 1, 16).astype(int)
    rows = []
    for k in pick:
        w = W[k]
        wv = (np.dstack([w] * 3) * 255).astype(np.uint8)
        wv[~vals[k]] = (60, 0, 60)
        a = cv2.resize(imgs[k][..., ::-1], (128, 128), interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(wv, (128, 128), interpolation=cv2.INTER_NEAREST)
        col = np.vstack([a, b])
        cv2.putText(col, f"t={ts[k]:.1f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        rows.append(col)
    cv2.imwrite(f"{REPO}/raw_data/rig/video_obs/{p}_sheet.jpg", np.hstack(rows))
    vf = vals.reshape(len(ts), -1).sum(1) / (RR <= R_IN).sum()
    acc = ((W[-1] > 0.5) == tw)[tv & vals[-1]].mean()
    fl = flick.reshape(n, -1).sum(1) / np.maximum(vals.reshape(n, -1).sum(1) + flick.reshape(n, -1).sum(1), 1)
    print(f"{p}: {len(ts)} frames, t 0..{ts[-1]:.2f} s, white fraction first/mid/last {prior[0]:.2f}/{prior[len(ts) // 2]:.2f}/"
          f"{prior[-1]:.2f} (last frame vs target mask {acc:.3f}), flicker removed median {np.median(fl):.3f} max {fl.max():.2f}, "
          f"valid fraction median {np.median(vf):.2f} "
          f"(min {vf.min():.2f}), {os.path.getsize(fn) / 1e6:.1f} MB")


if __name__ == "__main__":
    for p in sys.argv[1:] or ["heart", "push_heart", "leaf", "swan", "tulip"]:
        make(p)
