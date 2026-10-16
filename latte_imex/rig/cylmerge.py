"""Merge the pitcher across static windows using its cylinder axis: find the pitcher column, fit the vertical axis
(robust circle fit), unwrap each window to (angle, height) maps of brightness and radius, and find the yaw between
windows by circular cross-correlation of those maps (logo, handle and spout break the symmetry)."""
import sys, os, json, functools
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print = functools.partial(print, flush=True)

out = sys.argv[1]; srcs = sys.argv[2:]
NA, DH = 360, 0.002          # 1 deg bins, 2 mm height bins
HB = np.arange(0.0, 0.135, DH)


def table_frame(Z):
    n, d = Z["n"], float(Z["d"])
    e3 = n / np.linalg.norm(n); e1 = np.cross([0, 1, 0], e3); e1 /= np.linalg.norm(e1); e2 = np.cross(e3, e1)
    o = -d * e3
    P = np.concatenate([Z[f"P{c}"] for c in range(3)]).astype(float) - o
    C = np.concatenate([Z[f"C{c}"] for c in range(3)])
    return np.stack([P @ e1, P @ e2, P @ e3], 1), C


def kasa(x, y):
    A = np.stack([x, y, np.ones_like(x)], 1); b = x * x + y * y
    a = np.linalg.lstsq(A, b, rcond=None)[0]; cx, cy = a[0] / 2, a[1] / 2
    return cx, cy, np.sqrt(max(a[2] + cx * cx + cy * cy, 1e-9))


def find_pitcher(Q, C):
    m = (Q[:, 2] > 0.003) & (Q[:, 2] < 0.14) & (np.abs(Q[:, 0]) < 0.5) & (np.abs(Q[:, 1]) < 0.5)
    Q, C = Q[m], C[m]
    mid = (Q[:, 2] > 0.03) & (Q[:, 2] < 0.11)
    H, xe, ye = np.histogram2d(Q[mid, 0], Q[mid, 1], bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
    S = cv2.blur(H, (13, 13))
    i, j = np.unravel_index(np.argmax(S), S.shape)
    cx, cy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    for _ in range(5):
        k = mid & (np.hypot(Q[:, 0] - cx, Q[:, 1] - cy) < 0.07)
        cx, cy, R = kasa(Q[k, 0], Q[k, 1])
        rr = np.hypot(Q[:, 0] - cx, Q[:, 1] - cy)
        k2 = mid & (np.abs(rr - R) < 0.006)          # body only (not handle/spout)
        cx, cy, R = kasa(Q[k2, 0], Q[k2, 1])
    rr = np.hypot(Q[:, 0] - cx, Q[:, 1] - cy)
    k = (rr < R + 0.04) & (Q[:, 2] < 0.135)
    return Q[k] - [cx, cy, 0], C[k], R


def unwrap(Q, C):
    phi = (np.degrees(np.arctan2(Q[:, 1], Q[:, 0])) % 360).astype(int)
    hb = np.clip((Q[:, 2] / DH).astype(int), 0, len(HB) - 1)
    r = np.hypot(Q[:, 0], Q[:, 1])
    g = C.astype(float).mean(1)
    W = np.zeros((len(HB), NA)); Sg = np.zeros_like(W); Sr = np.zeros_like(W)
    np.add.at(W, (hb, phi), 1); np.add.at(Sg, (hb, phi), g); np.add.at(Sr, (hb, phi), r)
    ok = W >= 2
    G = np.where(ok, Sg / np.maximum(W, 1), np.nan); Rm = np.where(ok, Sr / np.maximum(W, 1), np.nan)
    return G, Rm, ok


def norm_map(M, ok):
    v = M[ok]; Z = np.zeros_like(M); Z[ok] = (v - v.mean()) / (v.std() + 1e-9); return Z


def yaw_between(A, B):
    """shift s (deg) such that B(phi) ~ A(phi - s)"""
    best = []
    for s in range(NA):
        sc = 0.0; nn = 0
        for (Ma, oa), (Mb, ob), w in ((A["g"], B["g"], 1.0), (A["r"], B["r"], 1.0)):
            Sa = np.roll(Ma, s, 1); So = np.roll(oa, s, 1) & ob
            n = So.sum()
            if n < 300: continue
            sc += w * np.sum(Sa[So] * Mb[So]) / n; nn += n
        best.append((sc, s, nn))
    best.sort(reverse=True)
    return best[0], best[1:4]


wins = []
for s in srcs:
    Q, C, R = find_pitcher(*table_frame(np.load(s)))
    G, Rm, ok = unwrap(Q, C)
    nm = os.path.basename(s).replace("_cloud.npz", "")
    wins.append(dict(name=nm, Q=Q, C=C, R=R, g=(norm_map(G, ok), ok), r=(norm_map(Rm, ok), ok), G=G, ok=ok))
    print(nm, "points", len(Q), "body radius mm", round(R * 1e3, 1), "angular coverage deg", int(ok.any(0).sum()))
ref = 0
res = {}
yaws = {wins[ref]["name"]: 0}
for w in wins[1:]:
    (sc, s, nn), alts = yaw_between(wins[ref], w)
    yaws[w["name"]] = s
    res[w["name"]] = dict(yaw_deg=s, score=round(sc, 3), overlap_bins=int(nn), next_best=[(round(a, 3), b) for a, b, _ in alts])
    print(w["name"], res[w["name"]])
# merge: rotate window by -yaw about the axis
P, Cc, wi = [], [], []
for k, w in enumerate(wins):
    th = -np.radians(yaws[w["name"]]); c, s_ = np.cos(th), np.sin(th)
    X = w["Q"].copy(); X[:, 0], X[:, 1] = c * w["Q"][:, 0] - s_ * w["Q"][:, 1], s_ * w["Q"][:, 0] + c * w["Q"][:, 1]
    P.append(X); Cc.append(w["C"]); wi.append(np.full(len(X), k))
P = np.concatenate(P); Cc = np.concatenate(Cc); wi = np.concatenate(wi)
np.savez_compressed(out + ".npz", P=P.astype(np.float32), C=Cc, win=wi, names=np.array([w["name"] for w in wins]))
cov = np.zeros(NA, bool)
for w in wins: cov |= np.roll(w["ok"].any(0), -yaws[w["name"]])
res["merged_angular_coverage_deg"] = int(cov.sum())
json.dump(res, open(out + ".json", "w"), indent=1)
print("merged angular coverage", cov.sum(), "deg")
fig = plt.figure(figsize=(20, 14))
for k, w in enumerate(wins):
    ax = fig.add_subplot(len(wins) + 2, 1, k + 1)
    ax.imshow(np.roll(w["G"], -yaws[w["name"]], 1), origin="lower", aspect="auto", cmap="gray", extent=(0, 360, 0, 135))
    ax.set_ylabel(f"{w['name']}\nh mm"); ax.set_xticks([])
tab = plt.get_cmap("tab10")
for j, az in enumerate([0, 90, 180, 270]):
    ax = fig.add_subplot(len(wins) + 2, 4, (len(wins)) * 4 + j + 1 + 4, projection="3d")
    ax.scatter(P[:, 0] * 1e3, P[:, 1] * 1e3, P[:, 2] * 1e3, s=0.3, c=Cc / 255.0, depthshade=False)
    ax.view_init(12, az); ax.set_box_aspect((1, 1, 1)); ax.set_xlim(-80, 80); ax.set_ylim(-80, 80); ax.set_zlim(0, 160)
fig.tight_layout(); fig.savefig(out + ".png", dpi=80)
