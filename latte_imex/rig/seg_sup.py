"""Supervised video whiteness (the supervision target of video_obs.py), leave-one-pour-out.

The only labels are the finished-pattern target masks (targets/<p>.npz: Otsu on LAB lightness of one hand-picked, evenly
lit camera-1 frame after the pour, registered onto the video by target_register.py) -- one frame per pour, taken after
the pitcher has left.  The last video frames are still under the pitcher's shadow, so a threshold on the video sits
1-2 px inside that edge (seg_gmm alone: area 0.70-0.85 of the mask); a classifier learns where the reference puts the
edge relative to the local contrast and blur.

  features   per pixel of one frame (frame_feats): crema-relative log lightness / chromaticity / LAB at local scales
             4 and 12 px and frame-global (normalised convolution weighted by the seg_gmm crema belief), a combined
             whiteness w = dl12 + 2 df12 with blurs, ridges, local max / min, gradients, local std, blue clipping,
             distance to the invalid region, the seg_gmm posterior log-odds with blur / local max; no absolute
             brightness and no position (the classifier must not learn where a pattern's white usually is)
  labels     of the TRAINING pours only: last NF frames = target mask (1-px boundary band at half weight); first N0
             frames = all crema (no foam yet); every MID_STEP-th frame between = crema farther than DFAR px from the
             final white; replayed under camera gains GAINS (clipped bright foam of heart / swan inside the range)
  model      HistGradientBoosting (300 iterations, 31 leaves, lr 0.06, min leaf 80, L2 1, seed 0)
  temporal   evidence = clip(GAMMA (log-odds - training prior), +-6), blurred 1 px, per-pixel two-state HMM (seg_gmm's
             constants) -> W
Protocol: the W of pour p comes from a classifier AND a seg_gmm geometry trained on the other four pours (LOPO); the
mask of p is never used for p.  Validation (honest, V2_SURFACE_MODEL.md 10.3): last-frame IoU vs the masks
0.848 / 0.867 / 0.858 / 0.843 / 0.871 (tulip / leaf / heart / push_heart / swan; seg_gmm under the same protocol
0.721 / 0.834 / 0.782 / 0.686 / 0.873), area 0.92-1.12, white at t = 0 <= 0.003, class flips 0.004-0.006 per frame;
trained on tulip + leaf only: held-out 0.860 / 0.856 / 0.842."""
import os

import cv2
import numpy as np

from . import seg_gmm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VID = os.path.join(REPO, "raw_data", "rig", "video_obs")
POURS = ("tulip", "leaf", "heart", "push_heart", "swan")
CFG = dict(NF=3, N0=5, MID_STEP=3, DFAR=8, N_MID=600, N_FIRST=1500, W_FINAL=1.0, W_FIRST=0.6, W_MID=0.6, BAND=1,
           W_BAND=0.5, GAINS=(0.75, 1.0, 1.35, 1.8), DROP=("f", "g", "maxc", "minc", "w12_rel4", "w12_rel8", "w12_relmin"),
           ITERS=300, LR=0.06, LEAVES=31, MIN_LEAF=80, L2=1.0, GAMMA=0.35, LLR_MAX=6.0, SIG_LLR=1.0, P_INIT=0.02,
           A_ON=0.02, A_OFF=0.02, SIG_MOVE=1.0)
EPS = 1e-3


# ------------------------------------------------------------------------------------------------ features
def _nconv(a, w, s, fallback):
    num = cv2.GaussianBlur((a * w).astype(np.float32), (0, 0), s)
    den = cv2.GaussianBlur(w.astype(np.float32), (0, 0), s)
    return (num + EPS * fallback) / (den + EPS)


def _blurv(a, v, s):
    num = cv2.GaussianBlur((a * v).astype(np.float32), (0, 0), s)
    den = cv2.GaussianBlur(v.astype(np.float32), (0, 0), s)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), a)


def _wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def _dil(a, v, r):
    return cv2.dilate(np.where(v, a, -9.0).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))


def _ero(a, v, r):
    return cv2.erode(np.where(v, a, 9.0).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))


def frame_feats(img, v, Wg, drop=CFG["DROP"], gain=1.0):
    """img (H, W, 3) uint8 RGB, v valid, Wg seg_gmm posterior -> (H, W, F) float32 (gain: training augmentation)."""
    if gain != 1.0:
        img = np.clip(np.round(img.astype(np.float32) * gain), 0, 255).astype(np.uint8)
    rgb = img.astype(np.float32) + 1.0
    S = rgb.sum(-1)
    f, g, l = (rgb[..., 2] - rgb[..., 0]) / S, (rgb[..., 1] - rgb[..., 0]) / S, np.log(S / 3.0)
    vf = v.astype(np.float32)
    wc = vf * (1.0 - np.clip(Wg, 0, 1)).astype(np.float32)
    if wc.sum() < 50:
        wc = vf
    sel = wc > 0.5
    if sel.sum() < 50:
        sel = v
    lc, fc, gc = (_wmedian(x[sel], wc[sel] + 1e-3) for x in (l, f, g))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    ac, bc = (_wmedian(lab[..., i][sel], wc[sel] + 1e-3) for i in (1, 2))
    F = {"f": f, "g": g, "dlG": l - lc, "dfG": f - fc, "dgG": g - gc, "da": lab[..., 1] - ac, "db": lab[..., 2] - bc}
    for s in (4, 12):
        F[f"dl{s}"] = l - _nconv(l, wc, s, lc)
        F[f"df{s}"] = f - _nconv(f, wc, s, fc)
    F["dg12"] = g - _nconv(g, wc, 12, gc)
    w12 = F["dl12"] + 2.0 * F["df12"]                  # combined whiteness: brighter and colder than the local crema
    F["w12"] = w12
    for s in (1, 2, 4):
        F[f"w12_b{s}"] = _blurv(w12, vf, s)
    F["w12_r1"] = w12 - _blurv(w12, vf, 1.5)           # ridges / valleys (thin lines)
    F["w12_r3"] = _blurv(w12, vf, 1) - _blurv(w12, vf, 4)
    F["dl12_r3"] = _blurv(F["dl12"], vf, 1) - _blurv(F["dl12"], vf, 4)
    F["df12_r3"] = _blurv(F["df12"], vf, 1) - _blurv(F["df12"], vf, 4)
    F["w12_max2"], F["w12_max4"], F["w12_min2"] = _dil(w12, v, 2), _dil(w12, v, 4), _ero(w12, v, 2)
    wb1 = F["w12_b1"]
    F["w12_rel4"] = wb1 / np.maximum(_dil(wb1, v, 4), 0.15)
    F["w12_rel8"] = wb1 / np.maximum(_dil(wb1, v, 8), 0.15)
    F["w12_relmin"] = (wb1 - _ero(wb1, v, 6)) / np.maximum(_dil(wb1, v, 6) - _ero(wb1, v, 6), 0.15)
    ls = cv2.GaussianBlur(l, (0, 0), 1.0)
    F["grad"] = np.hypot(cv2.Sobel(ls, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ls, cv2.CV_32F, 0, 1, ksize=3))
    fs = cv2.GaussianBlur(f, (0, 0), 1.0)
    F["gradf"] = np.hypot(cv2.Sobel(fs, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(fs, cv2.CV_32F, 0, 1, ksize=3))
    F["lstd"] = np.sqrt(np.maximum(_blurv(l * l, vf, 2) - _blurv(l, vf, 2) ** 2, 0))
    F["minc"], F["maxc"] = rgb.min(-1) / 256.0, rgb.max(-1) / 256.0
    F["clipB"] = (img[..., 2] >= 245).astype(np.float32)
    F["dvalid"] = np.minimum(cv2.distanceTransform(v.astype(np.uint8), cv2.DIST_L2, 3), 10.0)
    z = np.log(np.clip(Wg, 1e-4, 1 - 1e-4) / (1 - np.clip(Wg, 1e-4, 1 - 1e-4))).astype(np.float32)
    F["gz"], F["gz_b2"], F["gz_max2"] = z, _blurv(z, vf, 2), _dil(z, v, 2)
    return np.stack([F[k].astype(np.float32) for k in F if k not in drop], -1)


# ------------------------------------------------------------------------------------------------ training
def _video(p):
    Z = np.load(os.path.join(VID, f"{p}.npz"))
    return dict(img=Z["img"], valid=Z["valid"].astype(bool))


def training_set(pours, gmm, masks, cfg, rng):
    """gmm: {q: seg_gmm W of q}, masks: {q: (white, valid) at the video resolution}."""
    X, Y, Wt = [], [], []
    for q in pours:
        Z = _video(q)
        V, n = Z["valid"], len(Z["valid"])
        hw, hv = masks[q]
        far = cv2.dilate(hw.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg["DFAR"] + 1,) * 2)) == 0
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg["BAND"] + 1,) * 2)
        band = (cv2.dilate(hw.astype(np.uint8), k3) > 0) & (cv2.erode(hw.astype(np.uint8), k3) == 0)
        fin, first = list(range(n - cfg["NF"], n)), list(range(cfg["N0"]))
        mid = list(range(cfg["N0"], n - cfg["NF"], cfg["MID_STEP"]))
        for gain in cfg["GAINS"]:
            gw = 1.0 / len(cfg["GAINS"])
            Fs = {k: frame_feats(Z["img"][k], V[k], gmm[q][k], cfg["DROP"], gain) for k in fin + first + mid}
            for k in fin:
                m = V[k] & hv
                X.append(Fs[k][m]); Y.append(hw[m].astype(np.int8))
                Wt.append(np.where(band[m], cfg["W_BAND"], 1.0) * cfg["W_FINAL"] / cfg["NF"] * gw)
            for k in first:
                idx = np.flatnonzero(V[k].ravel())
                idx = rng.choice(idx, min(len(idx), cfg["N_FIRST"]), replace=False)
                X.append(Fs[k].reshape(-1, Fs[k].shape[-1])[idx]); Y.append(np.zeros(len(idx), np.int8))
                Wt.append(np.full(len(idx), cfg["W_FIRST"] * V[k].sum() / max(len(idx), 1) / cfg["N0"] / 3.0 * gw))
            for k in mid:
                idx = np.flatnonzero((V[k] & far).ravel())
                if len(idx) == 0:
                    continue
                idx = rng.choice(idx, min(len(idx), cfg["N_MID"]), replace=False)
                X.append(Fs[k].reshape(-1, Fs[k].shape[-1])[idx]); Y.append(np.zeros(len(idx), np.int8))
                Wt.append(np.full(len(idx), cfg["W_MID"] * (V[k] & far).sum() / max(len(idx), 1) / len(mid) / 3.0 * gw))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(Wt)


def train(pours, gmm, masks, cfg=CFG, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, Y, Wt = training_set(pours, gmm, masks, cfg, np.random.default_rng(seed))
    clf = HistGradientBoostingClassifier(max_iter=cfg["ITERS"], learning_rate=cfg["LR"], max_leaf_nodes=cfg["LEAVES"],
                                         min_samples_leaf=cfg["MIN_LEAF"], l2_regularization=cfg["L2"], early_stopping=False,
                                         random_state=seed)
    clf.fit(X, Y, sample_weight=Wt)
    pi = float((Wt * Y).sum() / Wt.sum())
    return clf, np.log(pi / (1 - pi))


def predict_logit(clf, p, Wg, cfg=CFG):
    Z = _video(p)
    V = Z["valid"]
    out = np.zeros(V.shape, np.float32)
    for k in range(len(V)):
        if V[k].any():
            out[k][V[k]] = clf.decision_function(frame_feats(Z["img"][k], V[k], Wg[k], cfg["DROP"])[V[k]])
    return out


# ------------------------------------------------------------------------------------------------ temporal
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _hmm(E, disc, cfg):
    lo = lambda q: np.log(q / (1 - q))
    F, B = np.zeros_like(E), np.zeros_like(E)
    lam = np.full(E.shape[1:], lo(cfg["P_INIT"]), np.float32)
    for k in range(len(E)):
        if k > 0:
            q = _blurv(_sig(lam), disc, cfg["SIG_MOVE"])
            lam = lo(np.clip(q * (1 - cfg["A_OFF"]) + (1 - q) * cfg["A_ON"], 1e-5, 1 - 1e-5))
        lam = lam + E[k]
        F[k] = lam
    b = np.zeros(E.shape[1:], np.float32)
    for k in range(len(E) - 2, -1, -1):
        q = _blurv(_sig(b + E[k + 1]), disc, cfg["SIG_MOVE"])
        b = np.log(((1 - cfg["A_OFF"]) * q + cfg["A_OFF"] * (1 - q)) / (cfg["A_ON"] * q + (1 - cfg["A_ON"]) * (1 - q)))
        B[k] = b
    return F + B


def temporal(logit, prior, V, cfg=CFG):
    n, H, _ = V.shape
    s = np.linspace(-1.05, 1.05, H)
    disc = (np.hypot(*np.meshgrid(s, s)) <= 0.96).astype(np.float32)
    E = np.zeros_like(logit)
    for k in range(n):
        if V[k].sum() >= 50:
            e = np.clip(cfg["GAMMA"] * (logit[k] - prior), -cfg["LLR_MAX"], cfg["LLR_MAX"])
            E[k] = np.where(V[k], _blurv(e, V[k].astype(np.float32), cfg["SIG_LLR"]), 0)
    return _sig(_hmm(E, disc, cfg)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ driver
def segment_lopo(p, masks, pours=POURS, cfg=CFG, log=print):
    """W of pour p from a seg_gmm geometry and a classifier trained on the other pours' masks."""
    others = [q for q in pours if q != p]
    train_masks = {q: masks[q] for q in others}
    gmm = {q: seg_gmm.segment(q, _video(q), train_masks) for q in others + [p]}
    seg_gmm._GEOM.clear()
    clf, prior = train(others, gmm, masks, cfg)
    W = temporal(predict_logit(clf, p, gmm[p], cfg), prior, _video(p)["valid"], cfg)
    log(f"{p}: supervised whiteness from {', '.join(others)}")
    return W
