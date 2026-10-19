"""Video whiteness (the supervision target of video_obs.py): per-frame adaptive two-class (crema / foam) Gaussian mixture, fitted jointly with a per-pixel two-state
HMM by EM over the whole video.  segment(p, Z, train) -> W (n, 128, 128) = posterior of the foam class.

features per pixel (gamma-encoded RGB + 1, S = R + G + B):  y = (f, g, l)
    f = (B - R) / S, g = (G - R) / S   chromaticity (intensity invariant: foam cold, crema warm)
    l = log(S / 3) - l_ref(x)          log lightness relative to the LOCAL crema lightness (normalised convolution,
                                       sigma SIG_REF = 12 px, weights = current crema belief 1 - W): exposure,
                                       pitcher shadows, camera seams
class model of frame k (crema-relative coordinates, mu_c,k = median of the frame's crema-labelled pixels):
    crema ~ (1 - T_C) N(0, C_c,k) + T_C N(0, S_TAIL C_c,k)        heavy tail: shadows, dark lines, reflections
    foam  ~ N(d_k, C_f,k), evaluated with y clamped onto the plane through the foam mode along d_k
            (monotone: colder + brighter than the foam mode is never less foam)
    evidence LLR_k(x) = log N_f - log p_crema; B-clipped pixels (B >= CLIP_B; glare blobs are already outside valid)
    get +LLR_CLIP; GAMMA * LLR clipped to +-LLR_MAX, smoothed spatially (SIG_LLR px, valid-normalised)
E-step: per-pixel HMM, forward-backward (prior P_INIT at frame 0, transitions A_ON / A_OFF per frame, motion blur
    SIG_MOVE px of the carried posterior) -> W.  A second HMM on LAB_SCALE x the evidence (smoother in time) gives
    the labels for the M-step, so that a few low-contrast frames (pitcher-raised lighting) do not shrink the labels.
M-step (per frame, statistics pooled over +-T_POOL frames, N_SUB random pixels):
    class locations = density modes (mean shift, kernel MS_H^2 C) of the labelled pixels, not label means
    (boundary-independent: no feedback between where the boundary is and where the classes are);
    spread along the class axis from the uncontaminated side only (crema: warm side, foam: cold side, MAD);
    pseudo-counts K_PRIOR toward the learned geometry; d_k >= D_MIN (foam colder and brighter than crema).
validated on the hand-checked final masks (held-out heart / push_heart / swan final-frame IoU 0.794 vs 0.686 for the
logistic classifier it replaces, white at t = 0 0.000, class flips per frame 0.005 vs 0.020; V2_SURFACE_MODEL.md 10.3).
learned geometry (the only use of hand masks: calibration pours in `train`, last frame): robust class means and
    covariances of the hand white / brown pixels -> prior offset d and covariances C_f, C_c.  Iterated N_IT = 4 times.
Everything is shared across pours; per pour only quantities measured from its own frames.
"""
import numpy as np
import cv2

P_INIT = 0.02        # prior foam probability at the first frame
A_ON = 0.02          # crema -> foam per frame
A_OFF = 0.02         # foam -> crema per frame
SIG_MOVE = 1.0       # [px] motion blur of the carried posterior
GAMMA = 0.15         # evidence tempering (pixel noise is correlated)
E_BIAS = 0.0         # evidence log-odds offset
LLR_MAX = 6.0
LLR_CLIP = 4.0
CLIP_B = 245
K_PRIOR = 300.0
T_POOL = 3           # frames pooled each side for the class statistics
N_IT = 4
T_C, S_TAIL = 0.15, 8.0
D_MIN = (0.05, 0.025, 0.12)
SIG_REF = 12.0
SIG_LLR = 1.5
ONE_SIDED = True
MODE = True         # class locations = density modes (mean shift), not label means
MS_H = 1.0
N_SUB = 20000        # pooled pixels used per M-step (random subsample)
LAB_SCALE = 0.3      # M-step labels from an HMM with the evidence scaled by this (smoother label field)
LOC_CHROMA = False
DARK_L = None       # l_rel below this: no evidence, not used in the class statistics
import os
VID = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "raw_data", "rig", "video_obs")

_GEOM = {}


def raw_feats(img):
    rgb = img.astype(np.float32) + 1.0
    S = rgb.sum(-1)
    return np.stack([(rgb[..., 2] - rgb[..., 0]) / S, (rgb[..., 1] - rgb[..., 0]) / S, np.log(S / 3.0)], -1)


def nconv(a, m, s):
    num = cv2.GaussianBlur((a * m).astype(np.float32), (0, 0), s)
    den = cv2.GaussianBlur(m.astype(np.float32), (0, 0), s)
    return num / np.maximum(den, 1e-4)


def rel_l(Y, crema_w, sig=SIG_REF):
    out = Y.copy()
    for c in ((0, 1, 2) if LOC_CHROMA else (2,)):
        out[..., c] = Y[..., c] - nconv(Y[..., c], crema_w, sig)
    return out


def rcov(y, w=None, thr=9.0, it=3):
    w = np.ones(len(y)) if w is None else w
    m = (w[:, None] * y).sum(0) / w.sum()
    C = np.cov(y.T, aweights=w) + 1e-6 * np.eye(3)
    for _ in range(it):
        D = np.einsum("ni,ij,nj->n", y - m, np.linalg.inv(C), y - m)
        ww = w * (D < thr)
        m = (ww[:, None] * y).sum(0) / ww.sum()
        C = np.cov(y.T, aweights=ww) * 1.1 + 1e-6 * np.eye(3)
    return m, C


def set_scale(C, u, s2):
    """covariance C with its variance along the unit vector u replaced by s2 (kept positive definite)."""
    C2 = C + (s2 - u @ C @ u) * np.outer(u, u)
    ev, Q = np.linalg.eigh(C2)
    return (Q * np.maximum(ev, 1e-6)) @ Q.T


def one_sided(z, side):
    """robust scale from one side of the median (the side the other class cannot contaminate)."""
    m = np.median(z)
    h = (m - z[z < m]) if side < 0 else (z[z > m] - m)
    return m, (1.4826 * np.median(h)) ** 2 if len(h) > 10 else None


def mean_shift(y, w, m, C, h=MS_H, it=15):
    """local density mode of the weighted points y (Gaussian kernel h^2 C) starting at m."""
    Ci = np.linalg.inv(C) / (h * h)
    for _ in range(it):
        d = y - m
        k = w * np.exp(-0.5 * np.einsum("ni,ij,nj->n", d, Ci, d))
        if k.sum() < 1e-6:
            break
        m2 = (k[:, None] * y).sum(0) / k.sum()
        if np.abs(m2 - m).max() < 1e-5:
            m = m2; break
        m = m2
    return m


def learn_geometry(train):
    key = tuple(sorted(train))
    if key in _GEOM:
        return _GEOM[key]
    D, Cf, Cc = [], [], []
    for p, (hw, hv) in train.items():
        Z = np.load(f"{VID}/{p}.npz")
        v = Z["valid"][-1] & hv & (Z["img"][-1][..., 2] < CLIP_B)
        Y = rel_l(raw_feats(Z["img"][-1]), (v & ~hw).astype(np.float32))
        mw, Cw = rcov(Y[v & hw]); mc, Ccc = rcov(Y[v & ~hw])
        D.append(mw - mc); Cf.append(Cw); Cc.append(Ccc)
    g = dict(d=np.mean(D, 0), Cf=np.mean(Cf, 0), Cc=np.mean(Cc, 0))
    _GEOM[key] = g
    return g


def blur_valid(a, m, sig):
    if sig <= 0:
        return a
    num = cv2.GaussianBlur((a * m).astype(np.float32), (0, 0), sig)
    den = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sig)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), a)


def gll(Y, mu, C):
    Ci = np.linalg.inv(C)
    d = Y - mu
    return -0.5 * np.einsum("ni,ij,nj->n", d, Ci, d) - 0.5 * np.log(np.linalg.det(C))


def evidence(Yv, mu_c, d, Cc, Cf):
    mu_f = mu_c + d
    u = d / np.linalg.norm(d)
    s = (Yv - mu_f) @ u
    Yf = Yv - np.maximum(s, 0)[:, None] * u[None, :]
    lf = gll(Yf, mu_f, Cf)
    lc = np.logaddexp(np.log(1 - T_C) + gll(Yv, mu_c, Cc), np.log(T_C) + gll(Yv, mu_c, S_TAIL * Cc))
    return lf - lc


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def hmm(E, V, disc):
    """forward-backward on per-frame log-likelihood ratios E (n, H, W) (0 where unobserved) -> posterior log-odds."""
    n = len(E)
    lo = lambda q: np.log(q / (1 - q))
    F = np.zeros_like(E)
    lam = np.full(E.shape[1:], lo(P_INIT), np.float32)
    for k in range(n):
        if k > 0:
            q = blur_valid(sig(lam), disc, SIG_MOVE)
            q = q * (1 - A_OFF) + (1 - q) * A_ON
            lam = lo(np.clip(q, 1e-5, 1 - 1e-5))
        lam = lam + E[k]
        F[k] = lam
    B = np.zeros_like(E)            # log b_k(1)/b_k(0), excludes frame k's evidence
    b = np.zeros(E.shape[1:], np.float32)
    for k in range(n - 2, -1, -1):
        q = sig(b + E[k + 1])
        q = blur_valid(q, disc, SIG_MOVE)
        b = np.log(((1 - A_OFF) * q + A_OFF * (1 - q)) / (A_ON * q + (1 - A_ON) * (1 - q)))
        B[k] = b
    return F + B


def segment(p, Z, train, return_info=False):
    g = learn_geometry(train)
    img, V = Z["img"], Z["valid"].astype(bool)
    n, H, Wd = V.shape
    s = np.linspace(-1.05, 1.05, H)
    disc = (np.hypot(*np.meshgrid(s, s)) <= 0.96).astype(np.float32)
    Yraw = raw_feats(img)
    clip = img[..., 2] >= CLIP_B
    ok = V.reshape(n, -1).sum(1) >= 50
    W = np.full((n, H, Wd), P_INIT, np.float32)
    par = [dict(d=g["d"].copy(), Cc=g["Cc"].copy(), Cf=g["Cf"].copy()) for _ in range(n)]
    rng = np.random.default_rng(0)
    for it in range(N_IT):
        E = np.zeros((n, H, Wd), np.float32)
        for k in range(n):
            v = V[k]
            if not ok[k]:
                continue
            xm = v & ~clip[k]
            Y = rel_l(Yraw[k], (v * (1 - W[k])).astype(np.float32))
            Yv = Y[xm]
            r = W[k][xm]
            cm = r < 0.5
            mu_c = np.median(Yv[cm], 0) if cm.sum() > 100 else np.median(Yv, 0)
            Cc, Cf, d = par[k]["Cc"], par[k]["Cf"], par[k]["d"]
            L = np.zeros((H, Wd), np.float32)
            L[xm] = np.clip(GAMMA * (evidence(Yv, mu_c, d, Cc, Cf) + E_BIAS), -LLR_MAX, LLR_MAX)
            if DARK_L is not None:
                L[xm & (Y[..., 2] - mu_c[2] < DARK_L)] = 0.0
            L[v & clip[k]] = LLR_CLIP
            if SIG_LLR > 0:
                L = blur_valid(L, v.astype(np.float32), SIG_LLR)
            E[k] = np.where(v, L, 0)
        lam = hmm(E, V, disc)
        W = sig(lam).astype(np.float32)
        Wl = W if LAB_SCALE is None else sig(hmm(E * LAB_SCALE, V, disc)).astype(np.float32)
        # M-step: crema-relative statistics with the NEW responsibilities, pooled over +-T_POOL frames
        per = [None] * n
        for j in range(n):
            if not ok[j]:
                continue
            v = V[j]; xm = v & ~clip[j]
            Y = rel_l(Yraw[j], (v * (1 - W[j])).astype(np.float32))
            Yv = Y[xm]; r = Wl[j][xm]
            cm = r < 0.5
            mu_c = np.median(Yv[cm], 0) if cm.sum() > 100 else np.median(Yv, 0)
            keep = (Yv[:, 2] - mu_c[2] >= DARK_L) if DARK_L is not None else np.ones(len(Yv), bool)
            per[j] = ((Yv - mu_c)[keep], r[keep])
        newpar = []
        for k in range(n):
            js = [j for j in range(max(0, k - T_POOL), min(n, k + T_POOL + 1)) if per[j] is not None]
            if not js:
                newpar.append(par[k]); continue
            ys = np.concatenate([per[j][0] for j in js]); wf = np.concatenate([per[j][1] for j in js])
            if len(ys) > N_SUB:
                ii = rng.choice(len(ys), N_SUB, replace=False); ys, wf = ys[ii], wf[ii]
            wc = 1 - wf
            sel = wf > 0.5
            if sel.sum() > 30:
                mf, Cf_ = rcov(ys[sel], wf[sel])
                nf = wf[sel].sum()
            else:
                mf, Cf_, nf = g["d"], g["Cf"], 0.0
            selc = wc > 0.5
            mc, Cc_ = rcov(ys[selc], wc[selc])
            nc = wc[selc].sum()
            if MODE:
                mc = mean_shift(ys, wc, mc, Cc_)
                if sel.sum() > 30:
                    mf = mean_shift(ys, wf, mf, Cf_)
            d = (nf * (mf - mc) + K_PRIOR * g["d"]) / (nf + K_PRIOR)
            d = np.maximum(d, D_MIN)
            if ONE_SIDED:
                u = d / np.linalg.norm(d)
                zc = ys[selc] @ u
                _, s2 = one_sided(zc, -1)          # crema spread from its warm side (no foam there)
                if s2 is not None:
                    Cc_ = set_scale(Cc_, u, s2)
                if sel.sum() > 30:
                    _, s2 = one_sided(ys[sel] @ u, +1)   # foam spread from its cold side (no crema there)
                    if s2 is not None:
                        Cf_ = set_scale(Cf_, u, s2)
            Cf2 = (nf * Cf_ + K_PRIOR * g["Cf"]) / (nf + K_PRIOR)
            Cc2 = (nc * Cc_ + K_PRIOR * g["Cc"]) / (nc + K_PRIOR)
            newpar.append(dict(d=d, Cc=Cc2, Cf=Cf2))
        par = newpar
    if return_info:
        return W, par
    return W
