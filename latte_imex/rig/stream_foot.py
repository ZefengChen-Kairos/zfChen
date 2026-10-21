"""E0 (V1 plan): where the milk stream actually meets the liquid surface, measured from the three calibrated cameras --
an input measurement, independent of the pattern (plan 4.1: the action / flow estimates are not corrected from the
target pattern).

Per frame of the pattern phase (cup_track.json frames, the top-view clock):
  cue        the stream is a thin bright line, near vertical in every camera: per camera a horizontal white top-hat
             of LAB lightness (structuring element 19 px), scaled to [0, 1] -- wide white (the foam pattern, the
             pitcher) is suppressed, the stream column kept
  slices     horizontal planes every 1 mm from 2 mm below the spout tip down to 1 mm above the liquid surface (the
             registered plane of cup_register.py; at least MIN_SLICES, so that low pours with the lip 7-12 mm above
             the surface are tracked); on each a 0.5 mm grid +-30 mm around the tip; a point scores the mean
             over the cameras that see it unoccluded (camera depth not nearer than the point by > 15 mm) of 2 r - 1
             (r = top-hat response at its projection), at least two cameras
  path       best continuous path through the slices (Viterbi, <= 1.5 mm per slice, small curvature penalty, start
             within 12 mm of the tip): the stream's centre line
  foot       straight line through the path's lowest 10 mm, extrapolated to the surface plane
  trusted    a stream really connects tip and surface: mean score of the lowest slices > Q_OK, the whole path close
             to a straight line (rms <= RMS_OK), steep (horizontal travel / drop <= SLOPE_OK) and starting within
             TOP_OK of the spout tip -- with the pitcher lifted the best path follows a foam line or the rim instead
Output raw_data/rig/results/<p>/stream_foot.json: per frame t (sim clock: pose / disc start, real_program.cup_poses),
foot on the board, in disc units (u, v as video_obs) and in solver units (x, y; cup radius 0.49), quality, and the
model's landing point at the same time (schedule.build of the measured pour, the arrival record the solver uses) and
the offset decomposed along / across the jet's horizontal direction (schedule.foot_dirs, foot_a / foot_b units).
Sheets: raw_data/rig/results/<p>/stream_foot.jpg (sampled frames, path and foot drawn in every camera) and
raw_data/rig/results/stream_foot.png (offsets over time, all pours).
    python3 -m latte_imex.rig.stream_foot [pattern ...]            (needs the session's pours/ data, as video_obs.py)"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trimrec import Take  # noqa: E402

SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "raw_data", "rig", "results")
PATTERNS = ("heart", "push_heart", "leaf", "swan", "tulip")
GRID = np.arange(-30, 30 + 1e-9, 0.5) * 1e-3     # [m] around the tip
STEP_MAX = 3                                      # cells (1.5 mm) per 1 mm slice
TOPHAT_W, TOPHAT_S = 19, 25.0
OCC = 0.015
LOW = 0.010                                       # [m] fitted part of the path above the surface
SLICE_TOP, SLICE_BOT = 0.002, 0.001               # [m] slices from below the lip down to just above the liquid
MIN_SLICES = 4                                    # a low pour (lip 7-12 mm above the registered plane) still gets a path
Q_OK = 0.35
RMS_OK = 0.0025          # [m]
SLOPE_OK = 0.8
TOP_OK = 0.010           # [m]


def ridge(im):
    L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, np.ones((1, TOPHAT_W), np.uint8))
    return np.clip(th / TOPHAT_S, 0, 1)


class Rig:
    def __init__(self, p):
        self.tk = Take(f"{SCR}/ghdata/{p}")
        self.W2C = [np.linalg.inv(T) for T in self.tk.c2w]

    def proj(self, c, X):
        K = self.tk.K[c]
        Pc = X @ self.W2C[c][:3, :3].T + self.W2C[c][:3, 3]
        return Pc[..., 0] / Pc[..., 2] * K[0, 0] + K[0, 2], Pc[..., 1] / Pc[..., 2] * K[1, 1] + K[1, 2], Pc[..., 2]


def track(rig, t_abs, tip, zs):
    """Stream centre line (z, x, y, score per slice) and foot (board) of the frame nearest t_abs."""
    tk = rig.tk
    i = int(np.argmin(np.abs(tk.T - t_abs)))
    views = [(c, ridge(tk.color(c, k)), tk.depth(c, k).astype(float) * 1e-3) for c, k, _ in tk.sync(i)]
    GX, GY = np.meshgrid(GRID, GRID)
    n = len(GRID)
    zsl = np.arange(tip[2] + SLICE_TOP, zs - SLICE_BOT, 0.001)  # board z points down
    if len(zsl) < MIN_SLICES:
        return None, None, views
    S = []
    for z in zsl:
        X = np.stack([tip[0] + GX, tip[1] + GY, np.full_like(GX, z)], -1)
        sc, nc = np.zeros(GX.shape), np.zeros(GX.shape)
        for c, r, dep in views:
            u, v, d = rig.proj(c, X)
            ok = (u >= 0) & (u < r.shape[1] - 1) & (v >= 0) & (v < r.shape[0] - 1) & (d > 0)
            ui, vi = np.clip(u.astype(int), 0, r.shape[1] - 1), np.clip(v.astype(int), 0, r.shape[0] - 1)
            dm = dep[vi, ui]
            use = ok & ~((dm > 0) & (dm < d - OCC))
            sc += np.where(use, 2 * r[vi, ui] - 1, 0)
            nc += use
        S.append(np.where(nc >= 2, sc / np.maximum(nc, 1), -1.0))
    S = np.array(S)
    R = STEP_MAX
    offs = [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1) if dy * dy + dx * dx <= R * R]
    acc = np.where(np.hypot(GX, GY) < 0.012, S[0], -1e9)
    bps = []
    for k in range(1, len(S)):
        best, arg = np.full(acc.shape, -1e18), np.zeros(acc.shape, int)
        pad = np.pad(acc, R, constant_values=-1e18)
        for m, (dy, dx) in enumerate(offs):
            sh = pad[R + dy:R + dy + n, R + dx:R + dx + n] - 0.05 * (dy * dy + dx * dx)
            b = sh > best
            best, arg = np.where(b, sh, best), np.where(b, m, arg)
        bps.append(arg)
        acc = best + S[k]
    j = np.unravel_index(np.argmax(acc), acc.shape)
    path = [j]
    for k in range(len(S) - 2, -1, -1):
        dy, dx = offs[bps[k][j]]
        j = (j[0] + dy, j[1] + dx)
        path.append(j)
    path = path[::-1]
    A = np.array([(z, tip[0] + GX[j], tip[1] + GY[j], S[k][j]) for k, (z, j) in enumerate(zip(zsl, path))])
    lo = A[A[:, 0] > zs - LOW]
    if len(lo) < 4:
        lo = A[-4:]
    fx, fy = np.polyfit(lo[:, 0], lo[:, 1], 1), np.polyfit(lo[:, 0], lo[:, 2], 1)
    foot = np.array([np.polyval(fx, zs), np.polyval(fy, zs), zs])
    return A, foot, views


def model_landing(p):
    """The solver's arrival record of pour p (schedule.build): frame mid times, landing x, y, jet direction, active."""
    from ..diff import schedule as SC
    script, _ = SC.pour_script(p, 128)
    S = SC.build(script, 128)
    tm = np.cumsum(S["fdt"]) - 0.5 * S["fdt"]
    ex, ey = SC.foot_dirs(S["ux"], S["uy"])
    return dict(t=tm, x=S["x"], y=S["y"], ex=ex, ey=ey, active=S["active"])


def run(p, every=1, sheet_n=12, log=print):
    from .real_program import CUP_R, cup_poses, tip_board_at
    rig = Rig(p)
    ct = json.load(open(f"{SCR}/{p}/cup_track.json"))
    C = cup_poses(p)
    t0, L = C["t0_abs"], C["L"]
    D = json.load(open(os.path.join(RES, p, "disc.json")))
    registered = bool(D and D[0].get("source") == "cup_register.py")
    if not registered:                                    # old free per-frame circles: only the accepted fits
        D = [d for d in D if d.get("fit_ok", True)]
    dt_ = np.array([d["t"] for d in D])
    dc = np.array([d["disc_centre"] for d in D])
    dR = np.array([d["disc_R"] for d in D])
    dz_ = np.array([d["surf_z"] for d in D])
    M = model_landing(p)
    out, keep = [], []
    # The search is anchored at cup_track.json's tip (the v1 pose track), whatever track the simulation uses: the
    # Viterbi start window and TOP_OK were tuned on that anchor.  Re-anchored at the v2 tip (about 5 mm away, at the
    # true spout) the tracker latched onto other bright features in part of the frames -- leaf 1-3 s: feet 15-25 mm
    # towards the pitcher, outside the growing white blob, at lower quality (0.5-0.7 vs 0.8-0.9); 10-20 % of the
    # frames moved > 10 mm (raw_data/rig/results/<p>/stream_foot_v2tip.json keeps that run).  The foot is a
    # measurement of the stream, not of the pitcher; anchoring it robustly at the true spout is open work
    # (start window around the predicted stream at the top slice, not around the tip).
    tips = np.array([o["tip"] for o in ct]) if os.environ.get("STREAM_FOOT_ANCHOR", "v1") == "v1" else \
        tip_board_at(p, [o["t"] for o in ct])
    for fi in range(0, len(ct), every):
        o = ct[fi]
        tip = tips[fi]
        # liquid surface: the registered plane (multi-view consistent, cup_register.py); else the raw depth surface
        zs = float(np.interp(o["t"], dt_, dz_)) if registered else float(o["surf_z_board"])
        A, foot, views = track(rig, o["t"], tip, zs)
        if A is None:
            continue
        q = float(A[A[:, 0] > zs - LOW, 3].mean()) if (A[:, 0] > zs - LOW).any() else float(A[-4:, 3].mean())
        gx, gy = np.polyfit(A[:, 0], A[:, 1], 1), np.polyfit(A[:, 0], A[:, 2], 1)
        rms = float(np.sqrt(np.mean((A[:, 1] - np.polyval(gx, A[:, 0])) ** 2 + (A[:, 2] - np.polyval(gy, A[:, 0])) ** 2)))
        slope = float(np.hypot(gx[0], gy[0]))
        top = float(np.hypot(A[0, 1] - tip[0], A[0, 2] - tip[1]))
        trusted = bool(q > Q_OK and rms <= RMS_OK and slope <= SLOPE_OK and top <= TOP_OK)
        cx, cy = np.interp(o["t"], dt_, dc[:, 0]), np.interp(o["t"], dt_, dc[:, 1])
        Rr = L * CUP_R if registered else float(np.interp(o["t"], dt_, dR))   # one length unit with the sim (L: m per D)
        u, v = (foot[0] - cx) / Rr, (foot[1] - cy) / Rr
        xs, ys = (foot[0] - cx) / L, -(foot[1] - cy) / L
        ts = o["t"] - t0
        k = int(np.clip(np.searchsorted(M["t"], ts), 0, len(M["t"]) - 1))
        rec = dict(fi=fi, t=ts, quality=q, rms_mm=rms * 1e3, slope=slope, top_mm=top * 1e3, trusted=trusted, foot_board=foot.tolist(), u=u, v=v, x=xs, y=ys,
                   tip_board=tip.tolist(), surf_z=zs, model_active=bool(M["active"][k]))
        if M["active"][k]:
            dx, dy = xs - M["x"][k], ys - M["y"][k]
            ex, ey = M["ex"][k], M["ey"][k]
            rec.update(model_x=float(M["x"][k]), model_y=float(M["y"][k]), along=float(dx * ex + dy * ey),
                       across=float(-dx * ey + dy * ex), dist=float(np.hypot(dx, dy)))
        out.append(rec)
        keep.append((fi, A, foot, tip, views))
    json.dump(dict(pattern=p, cup_radius_solver=CUP_R, Q_OK=Q_OK, frames=out), open(os.path.join(RES, p, "stream_foot.json"), "w"),
              indent=0)
    ok = [r for r in out if r["trusted"] and "along" in r]
    al, ac = np.array([r["along"] for r in ok]), np.array([r["across"] for r in ok])
    summ = dict(pattern=p, frames=len(out), trusted_active=len(ok),
                along_median=float(np.median(al)) if len(ok) else None, across_median=float(np.median(ac)) if len(ok) else None,
                along_iqr=[float(x) for x in np.percentile(al, [25, 75])] if len(ok) else None,
                across_iqr=[float(x) for x in np.percentile(ac, [25, 75])] if len(ok) else None)
    log(f"{p}: {len(out)} frames, {len(ok)} trusted while pouring; offset measured - model [cup diameters]: along "
        f"{summ['along_median']:+.3f} (IQR {summ['along_iqr'][0]:+.3f}..{summ['along_iqr'][1]:+.3f}), across "
        f"{summ['across_median']:+.3f} (IQR {summ['across_iqr'][0]:+.3f}..{summ['across_iqr'][1]:+.3f})" if len(ok) else
        f"{p}: no trusted frames")
    _sheet(p, rig, keep, out, sheet_n)
    return summ


def _sheet(p, rig, keep, out, n):
    sel = [keep[i] for i in np.linspace(0, len(keep) - 1, min(n, len(keep))).astype(int)]
    q = {r["fi"]: r for r in out}
    rows = []
    for fi, A, foot, tip, views in sel:
        tiles = []
        for c, _, _ in views:
            k = [s for cc, s, _ in rig.tk.sync(int(np.argmin(np.abs(rig.tk.T - json.load(open(f"{SCR}/{p}/cup_track.json"))[fi]["t"]))))
                 if cc == c][0]
            im = rig.tk.color(c, k).copy()
            for z, x, y, s in A:
                u, v, _ = rig.proj(c, np.array([x, y, z]))
                cv2.circle(im, (int(u), int(v)), 1, (0, 0, 255) if s > 0 else (0, 255, 255), -1)
            u, v, _ = rig.proj(c, foot)
            cv2.drawMarker(im, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            u, v, _ = rig.proj(c, tip)
            cx, cy, h = int(u), int(v) + 40, 90
            cr = im[max(cy - h, 0):cy + h, max(cx - h, 0):cx + h]
            cr = cv2.resize(cr, (240, 240)) if cr.size else np.zeros((240, 240, 3), np.uint8)
            r_ = q[fi]
            cv2.putText(cr, f"cam{c} f{fi} q{r_['quality']:.2f} {'OK' if r_['trusted'] else 'reject'}", (4, 16), 0, 0.45,
                        (0, 255, 0) if r_["trusted"] else (0, 0, 255), 1)
            tiles.append(cr)
        rows.append(np.hstack(tiles))
    half = (len(rows) + 1) // 2
    L, Rr = rows[:half], rows[half:] + [np.zeros_like(rows[0])] * (half - len(rows[half:]))
    cv2.imwrite(os.path.join(RES, p, "stream_foot.jpg"), np.hstack([np.vstack(L), np.vstack(Rr)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def plot(patterns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(patterns), 1, figsize=(10, 2.3 * len(patterns)), sharex=False)
    for a, p in zip(np.atleast_1d(ax), patterns):
        Z = json.load(open(os.path.join(RES, p, "stream_foot.json")))
        F = [r for r in Z["frames"] if "along" in r]
        t = np.array([r["t"] for r in F]); ok = np.array([r["trusted"] for r in F])
        for key, col in (("along", "C0"), ("across", "C1")):
            y = np.array([r[key] for r in F])
            a.plot(t[ok], y[ok], ".", color=col, ms=4, label=f"{key} (trusted)")
            a.plot(t[~ok], y[~ok], ".", color=col, ms=2, alpha=0.25)
            if ok.any():
                a.axhline(np.median(y[ok]), color=col, lw=0.8, ls="--")
        a.axhline(0, color="k", lw=0.5)
        a.set_ylabel(f"{p}\n[cup D]")
        a.set_ylim(-0.4, 0.4)
    np.atleast_1d(ax)[0].legend(fontsize=7, ncol=2)
    np.atleast_1d(ax)[-1].set_xlabel("t [s] (sim clock)")
    fig.suptitle("E0: measured stream foot - model landing point (along / across the jet direction)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "stream_foot.png"), dpi=110)


def main(argv=None):
    pats = (argv or sys.argv[1:]) or list(PATTERNS)
    S = [run(p) for p in pats]
    json.dump(S, open(os.path.join(RES, "stream_foot_summary.json"), "w"), indent=1)
    plot(pats)


if __name__ == "__main__":
    main()
