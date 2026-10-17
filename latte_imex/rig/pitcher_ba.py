"""Pitcher pose from AprilTags on its (cylindrical) wall, jointly over cameras and frames.

Pitcher frame: z = cylinder axis (towards the rim), origin on the axis at the height of tag 4's centre, x towards
tag 4. Tag i: centred at angle phi_i, height z_i on a cylinder of radius r, upright (tag y = -axis since image-y is
down... handled by a fitted roll psi_i), wrapped around the cylinder.
Unknowns: r, (phi_i, z_i, psi_i) for every tag except tag 4 (phi=z=0, psi fitted), one 6-dof pose per sample.
Samples: cam-0 frames, joined with the other cameras' frames within SYNC s; residual = corner reprojection (px)."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from trimrec import Take
from tags import TAG

SYNC = 0.012
REF = 4


def tag_corners(r, phi, z, psi):
    """3-D corners (TL,TR,BR,BL as aruco) of a TAG x TAG square wrapped on the cylinder, in the pitcher frame."""
    h = TAG / 2
    loc = np.array([[-h, h], [h, h], [h, -h], [-h, -h]])           # tag (x right, y up) before roll
    c, s = np.cos(psi), np.sin(psi)
    loc = loc @ np.array([[c, -s], [s, c]]).T
    ang = phi + loc[:, 0] / r                                      # arc length -> angle
    return np.stack([r * np.cos(ang), r * np.sin(ang), z + loc[:, 1]], 1)


def build(rows, tk):
    by_cam = {}
    for d in rows:
        by_cam.setdefault((d["cam"], d["step"]), []).append(d)
    keys0 = sorted({(d["cam"], d["step"]) for d in rows if d["cam"] == 0}, key=lambda k: k[1])
    used = set(); samples = []
    ts = {c: dict(zip(tk.steps[c], tk.ts[c])) for c in range(tk.ncam)}
    for c0, s0 in keys0:
        t = ts[0][s0]; grp = [(0, s0)]
        for c in range(1, tk.ncam):
            j = int(np.argmin(np.abs(tk.ts[c] - t)))
            if abs(tk.ts[c][j] - t) < SYNC and (c, tk.steps[c][j]) in by_cam: grp.append((c, int(tk.steps[c][j])))
        samples.append(grp); used.update(grp)
    for k in by_cam:
        if k not in used: samples.append([k])
    obs = []
    for si, grp in enumerate(samples):
        for k in grp:
            for d in by_cam[k]:
                obs.append((si, d["cam"], d["tag"], d["q"]))
    t_s = [np.mean([ts[c][s] for c, s in g]) - tk.t0 for g in samples]
    return samples, obs, np.array(t_s)


def solve(tk, rows, init_layout):
    samples, obs, t_s = build(rows, tk)
    tags = sorted({o[2] for o in obs})
    others = [t for t in tags if t != REF]
    # initial poses: from any detection, T_board_pitcher = T_board_tag @ inv(T_pitcher_tag)
    def T_pt(r, phi, z, psi):
        # pitcher <- tag transform (tag plane tangent at its centre)
        n = np.array([np.cos(phi), np.sin(phi), 0.0]); ez = np.array([0, 0, 1.0]); ex = np.cross(ez, n)
        Rt = np.stack([ex, ez, n], 1)            # tag x = tangent, tag y = axis, tag z = outward normal
        c, s = np.cos(psi), np.sin(psi); Rt = Rt @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T = np.eye(4); T[:3, :3] = Rt; T[:3, 3] = r * n + [0, 0, z]; return T
    lay = dict(init_layout)
    x0 = [lay["r"], lay[REF][2]]
    for t in others: x0 += list(lay[t])
    first = {}
    for d in rows:
        first.setdefault((d["cam"], d["step"]), d)
    poses = []
    W2C_ = [np.linalg.inv(T) for T in tk.c2w]
    def sample_err(T, cand):
        e = []
        for dd in cand:
            Xp = tag_corners(lay["r"], *lay[dd["tag"]]); Xb = Xp @ T[:3, :3].T + T[:3, 3]
            Xc = Xb @ W2C_[dd["cam"]][:3, :3].T + W2C_[dd["cam"]][:3, 3]; K = tk.K[dd["cam"]]
            if (Xc[:, 2] <= 0).any(): return 1e9
            uv = Xc[:, :2] / Xc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
            e.append(np.abs(uv - dd["q"]).max())
        return max(e)
    for g in samples:
        cand = [dd for dd in rows if (dd["cam"], dd["step"]) in g]
        best = None
        for d in cand:
            for Tb in d.get("T_board_alts", [d["T_board"]]):
                T = Tb @ np.linalg.inv(T_pt(lay["r"], *lay[d["tag"]]))
                e = sample_err(T, cand)
                if best is None or e < best[0]: best = (e, T)
        T = best[1]
        poses += list(Rot.from_matrix(T[:3, :3]).as_rotvec()) + list(T[:3, 3])
    x0 = np.array(x0 + poses)
    nl = 2 + 3 * len(others)
    W2C = [np.linalg.inv(T) for T in tk.c2w]
    oi = np.array([o[0] for o in obs]); oc = np.array([o[1] for o in obs]); ot = [o[2] for o in obs]
    Q = np.array([o[3] for o in obs])

    def unpack(x):
        L = {REF: (0.0, 0.0, x[1])}
        for j, t in enumerate(others): L[t] = tuple(x[2 + 3 * j: 5 + 3 * j])
        return x[0], L, x[nl:].reshape(-1, 6)

    def resid(x):
        r, L, P = unpack(x)
        C = {t: tag_corners(r, *L[t]) for t in tags}
        out = np.empty((len(obs), 8))
        Rs = Rot.from_rotvec(P[:, :3]).as_matrix()
        for c in range(tk.ncam):
            m = oc == c
            if not m.any(): continue
            K = tk.K[c]; R_wc = W2C[c][:3, :3]; t_wc = W2C[c][:3, 3]
            for t in tags:
                mm = m & (np.array(ot) == t)
                if not mm.any(): continue
                idx = np.where(mm)[0]
                Xp = C[t]                                          # 4x3 pitcher frame
                Xb = np.einsum("nij,kj->nki", Rs[oi[idx]], Xp) + P[oi[idx], None, 3:]
                Xc = Xb @ R_wc.T + t_wc
                uv = Xc[..., :2] / Xc[..., 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
                out[idx] = (uv - Q[idx]).reshape(len(idx), 8)
        return out.ravel()

    from scipy.sparse import lil_matrix
    J = lil_matrix((len(obs) * 8, len(x0)), dtype=np.int8)
    for n_, si in enumerate(oi):
        J[n_ * 8:(n_ + 1) * 8, :nl] = 1
        J[n_ * 8:(n_ + 1) * 8, nl + 6 * si: nl + 6 * si + 6] = 1
    res = least_squares(resid, x0, loss="huber", f_scale=1.0, x_scale="jac", max_nfev=400, jac_sparsity=J, method="trf", ftol=1e-12, xtol=1e-12, gtol=1e-12)
    f0 = resid(x0); print('BA: init rms', float(np.sqrt(np.nanmean(f0 ** 2))), 'nonfinite', int((~np.isfinite(f0)).sum()), 'status', res.status, res.message, 'nfev', res.nfev, flush=True)
    r, L, P = unpack(res.x)
    rr = res.fun.reshape(-1, 8)
    return dict(r=r, layout=L, poses=P, t=t_s, samples=samples, rms_px=float(np.sqrt(np.mean(rr ** 2))),
                med_px=float(np.median(np.abs(rr))), cost=float(res.cost), nobs=len(obs))
