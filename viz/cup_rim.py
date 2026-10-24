"""Measured cup tilt and rim height per pour (viz/cup_rim.json), for the stand-in cup of the three-view videos.

Why: the liquid disc (disc.json) fixes only the surface plane, the disc centre and its radius.  A LEVEL stand-in cup whose
rim sits ~20 mm above the liquid early in the pour must intersect the pitcher: the measured spout tip is only ~11-17 mm
above the liquid and inside the rim circle.  The barista tilts the cup towards the pitcher early and levels it as it
fills; this module measures that from the depth cameras.

Measurement (per camera-0 frame of the trimmed take, all three RealSense views fused):
  points  every depth pixel, in the cup frame F of viz/scene.py at that instant (origin = registered disc centre on the
          liquid surface, x = board x, y = -board y, z = up; disc.json interpolated as real_program.cup_poses does), that
          is BLACK (HSV value < 80: the mug; the latte, the white pitcher, the stream and the hand are not), lies near
          the cup wall and is farther than 12 mm from the posed CAD pitcher (its tags are black).
  fit     the rim is a circle perpendicular to the cup axis; the axis passes through the disc centre (the centre of the
          liquid section).  Iterated in the current cup frame: the points with radius R_disc - 2.5 mm .. R_disc + 7 mm
          (inner wall, rounded rim, outer wall), 24 azimuth bins, rim-top height = the 92nd percentile of the axial
          height per bin (>= 25 points), a robust plane through the bin rim heights at radius R_disc + 2 mm -> residual
          tilt; repeated until the residual tilt is < 0.02 deg.  Output: the axis n (unit, F) and s = the axial height
          of the rim-top centre above the disc centre.
  smooth  robust local-linear smoothing in time (Gaussian sigma 0.35 s, Tukey reweighting) of n_x, n_y and s.
Cross-check: free ellipse fits of the liquid edge in the fused top views give axis ratios 0.95-0.97 early in tulip with
the major axis towards the pitcher (a cylinder tilted by 14-19 deg), consistent with the rim fit.

    python viz/cup_rim.py [pour ...]      (needs raw_data/rig/results/<p>/disc.json and the session's trimmed takes,
                                           pours/ghdata/<p>; writes viz/cup_rim.json; the renderer only reads the json)
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scene import rot_to, surface_samples  # noqa: E402
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "cup_rim.json")
POURS_DIR = os.environ.get("POURS_DIR", "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours")
PATTERNS = ("tulip", "leaf", "heart", "push_heart", "swan")
V_BLACK = 80                      # HSV value below which a pixel is the black mug
R_IN, R_OUT = -2.5e-3, 7e-3       # radial window around R_disc (cup frame)
Z_LO, Z_HI = -5e-3, 0.05          # axial window above the disc centre
NB, MIN_BIN, PCT = 24, 25, 92
PITCHER_EXCL = 0.012
SIG_T = 0.35
VERSION = 1


def fit_frame(Q, R, it=12):
    """Q (n, 3) black points in F -> dict(n, s, mad, bins, npts) or None."""
    n = np.array([0.0, 0.0, 1.0])
    s = 0.02
    rb = R + 2e-3
    out = None
    for _ in range(it):
        Rc = rot_to(n)
        P = Q @ Rc                                          # cup-frame coordinates (rows: Rc^T q)
        rho = np.hypot(P[:, 0], P[:, 1])
        m = (rho > R + R_IN) & (rho < R + R_OUT) & (P[:, 2] > Z_LO) & (P[:, 2] < Z_HI)
        if m.sum() < 200:
            return out
        az = np.arctan2(P[m, 1], P[m, 0])
        bi = np.floor((az + math.pi) / (2 * math.pi) * NB).astype(int) % NB
        z = P[m, 2]
        rows = []
        for j in range(NB):
            sj = bi == j
            if sj.sum() >= MIN_BIN:
                rows.append((-math.pi + (j + 0.5) * 2 * math.pi / NB, np.percentile(z[sj], PCT), sj.sum()))
        if len(rows) < 10:
            return out
        rows = np.array(rows)
        A = np.c_[np.ones(len(rows)), rb * np.cos(rows[:, 0]), rb * np.sin(rows[:, 0])]
        w = np.ones(len(rows))
        for _ in range(8):                                  # Huber IRLS
            sol = np.linalg.lstsq(A * w[:, None], rows[:, 1] * w, rcond=None)[0]
            res = rows[:, 1] - A @ sol
            sc = 1.4826 * np.median(np.abs(res)) + 3e-4
            w = np.sqrt(1.0 / np.maximum(1.0, np.abs(res) / (1.5 * sc)))
        a, bx, by = sol
        nc = np.array([-bx, -by, 1.0]); nc /= np.linalg.norm(nc)
        lam = a / (nc[2] - bx * nc[0] - by * nc[1])
        n = Rc @ nc
        s = float(lam)
        out = dict(n=n, s=s, mad=float(np.median(np.abs(res))), bins=int(len(rows)), npts=int(m.sum()))
        if math.degrees(math.acos(min(1.0, nc[2]))) < 0.02:
            break
    return out


def robust_smooth(t, X, w0, sig=SIG_T, c=None, it=4):
    """Robust local-linear smoothing of X (n, k) at the times t (Tukey reweighting with per-column scales c)."""
    w = w0.copy()
    for _ in range(it):
        fit = np.empty_like(X)
        for j, tj in enumerate(t):
            k = np.exp(-0.5 * ((t - tj) / sig) ** 2) * w
            A = np.stack([np.ones_like(t), t - tj], 1) * np.sqrt(k)[:, None]
            fit[j] = np.linalg.lstsq(A, X * np.sqrt(k)[:, None], rcond=None)[0][0]
        r = np.max(np.abs(X - fit) / c[None, :], 1)
        w = w0 * np.where(r < 1, (1 - r ** 2) ** 2, 0.0)
    return fit, w


def measure(p, log=print):
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "latte_imex", "rig"))
    import cv2
    from scipy.spatial import cKDTree
    from trimrec import Take, cloud
    from latte_imex.rig import real_program as RP
    import scene as S
    tk = Take(f"{POURS_DIR}/ghdata/{p}")
    D = json.load(open(os.path.join(REPO, "raw_data", "rig", "results", p, "disc.json")))
    td = np.array([d["t"] for d in D])
    dc = np.array([d["disc_centre"] for d in D])
    dzs = np.array([d["surf_z"] for d in D])
    CP = RP.cup_poses(p)
    t0, R = float(CP["t0_abs"]), float(CP["R_disc"])
    geom = S.PitcherGeom.load(os.path.join(REPO, "latte_imex", "pitcher_geom", "pitcher_kona700.json"))
    V, F = S.load_stl(os.path.join(REPO, "latte_imex", "pitcher_geom", "pitcher_kona700.stl"))
    Ps = surface_samples(V, F, 60000)
    ang = {k: np.unwrap(CP[k]) for k in ("yaw", "tilt", "roll")}
    rows = []
    for i in range(len(tk.T)):
        T = float(tk.T[i])
        ts = T - t0
        if T < td[0] or T > td[-1] or ts < CP["t"][0] - 0.05 or ts > CP["t"][-1] + 0.05:
            continue
        cx, cy, cz = np.interp(T, td, dc[:, 0]), np.interp(T, td, dc[:, 1]), np.interp(T, td, dzs)
        tip = np.array([np.interp(ts, CP["t"], CP["tip"][:, k]) for k in range(3)])
        Rp = S.rotation(*(float(np.interp(ts, CP["t"], ang[k])) for k in ("yaw", "tilt", "roll")))
        tst = tip - Rp @ geom.tip - Rp @ np.array([0.0, 0.0, S.STL_BOTTOM_M])
        kd = cKDTree(Ps @ Rp.T + tst)
        Q = []
        for cam, k, _ in tk.sync(i):
            P, C, _ = cloud(tk, cam, k, 1)
            q = np.c_[P[:, 0] - cx, -(P[:, 1] - cy), cz - P[:, 2]]
            r = np.hypot(q[:, 0], q[:, 1])
            m = (r < R + 0.03) & (q[:, 2] > -0.02) & (q[:, 2] < 0.06)
            if not m.any():
                continue
            hsv = cv2.cvtColor(C[m][None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
            q = q[m][hsv[:, 2] < V_BLACK]
            d, _ = kd.query(q, distance_upper_bound=PITCHER_EXCL)
            Q.append(q[~np.isfinite(d)])
        Q = np.concatenate(Q) if Q else np.zeros((0, 3))
        f = fit_frame(Q, R) if len(Q) > 200 else None
        if f is None:
            continue
        rows.append((ts, f["n"][0], f["n"][1], f["s"], f["mad"], f["bins"], f["npts"]))
    A = np.array(rows)
    t = A[:, 0]
    X = A[:, 1:4]
    w0 = np.clip((A[:, 5] - 9) / 15.0, 0.05, 1.0) / np.maximum(A[:, 4] / 1e-3, 0.5)
    fit, w = robust_smooth(t, X, w0, c=np.array([0.06, 0.06, 4e-3]))
    tg = np.round(np.arange(0.0, float(CP["t"][-1]) + 0.05, 0.05), 4)
    sm = np.stack([np.interp(tg, t, fit[:, k]) for k in range(3)], 1)
    th = np.degrees(np.arcsin(np.clip(np.hypot(sm[:, 0], sm[:, 1]), 0, 1)))
    low = np.degrees(np.arctan2(sm[:, 1], sm[:, 0]))
    log(f"{p}: {len(A)} frames fitted ({int((w > 0).sum())} kept); tilt {th[0]:.1f} -> {th[-1]:.1f} deg "
        f"(max {th.max():.1f}), rim-top centre {sm[0, 2] * 1e3:.1f} -> {sm[-1, 2] * 1e3:.1f} mm above the liquid, "
        f"low side az {np.median(low):.0f} deg (F); per-frame rim MAD median {np.median(A[:, 4]) * 1e3:.2f} mm")
    return dict(t0_abs=t0, R_disc_m=R, t=tg.tolist(), n_x=np.round(sm[:, 0], 6).tolist(),
                n_y=np.round(sm[:, 1], 6).tolist(), s_m=np.round(sm[:, 2], 6).tolist(),
                raw=dict(t=np.round(t, 4).tolist(), n_x=np.round(A[:, 1], 5).tolist(), n_y=np.round(A[:, 2], 5).tolist(),
                         s_m=np.round(A[:, 3], 5).tolist(), mad_m=np.round(A[:, 4], 5).tolist(),
                         bins=A[:, 5].astype(int).tolist(), npts=A[:, 6].astype(int).tolist(), w=np.round(w, 3).tolist()))


def main(argv=None):
    ps = (argv if argv is not None else sys.argv[1:]) or list(PATTERNS)
    J = {}
    if os.path.exists(OUT):
        J = json.load(open(OUT))
    J.setdefault("pours", {})
    J["version"] = VERSION
    J["doc"] = ("Measured cup tilt + rim height (viz/cup_rim.py): per pour, on the scene clock t [s] "
                "(= real_program.cup_poses t, t0_abs = take time of t = 0), the cup axis n = (n_x, n_y, sqrt(1 - n_x^2 - "
                "n_y^2)) in the cup frame F and s_m = the axial height [m] of the rim-top centre above the liquid-disc "
                "centre (smoothed, 20 Hz); raw = the per-frame fits.")
    for p in ps:
        J["pours"][p] = measure(p)
    with open(OUT, "w") as f:
        json.dump(J, f, separators=(",", ":"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
