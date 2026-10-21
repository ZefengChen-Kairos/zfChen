"""Model-free check of the tag geometry: for every frame in which ONE camera sees two tags (cam 0: tags 4+5, cam 1:
tags 1+0), fit a per-frame 6-dof pose of the first tag and ONE rigid relative transform between the two (planar
19 mm squares, no jug model).  Gives (i) corner rms per corner estimator (integer 'none' = current tags.pkl, aruco
'contour', subpixel 'edge'), (ii) the angle between the tag normals (flats of a 14-gon: k x 25.714 deg), (iii) the
centre distance -> implied apothem for tags centred on flats, and the vertical offset / relative roll.
    python3 pair_check.py  -> ../out/pair_check.json"""
import sys, json, numpy as np, cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from pv2lib import *

OBJ = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * TAG / 2


def fit_pair(D, cam, ta, tb):
    by = D.by_frame()
    fr = [(k, {d['tag']: d for d in v}) for k, v in by.items() if k[0] == cam]
    fr = [(k, v) for k, v in fr if ta in v and tb in v]
    if len(fr) < 5:
        return None
    Kc = D.tk.K[cam]
    x0 = []
    for k, v in fr:                       # tag-a pose in the camera frame (IPPE, best)
        _, rvs, tvs, _ = cv2.solvePnPGeneric(OBJ, v[ta]['q'], Kc, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        x0.append(np.r_[rvs[0].ravel(), tvs[0].ravel()])
    # relative transform init: median over frames of inv(Ta) Tb from IPPE
    rel = []
    for (k, v), xa in zip(fr, x0):
        _, rvs, tvs, _ = cv2.solvePnPGeneric(OBJ, v[tb]['q'], Kc, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        Ta = np.eye(4); Ta[:3, :3] = Rot.from_rotvec(xa[:3]).as_matrix(); Ta[:3, 3] = xa[3:]
        Tb = np.eye(4); Tb[:3, :3] = cv2.Rodrigues(rvs[0])[0]; Tb[:3, 3] = tvs[0].ravel()
        rel.append(np.linalg.inv(Ta) @ Tb)
    rv = np.median([Rot.from_matrix(T[:3, :3]).as_rotvec() for T in rel], 0); tv = np.median([T[:3, 3] for T in rel], 0)
    n = len(fr)
    Qa = np.stack([v[ta]['q'] for k, v in fr]); Qb = np.stack([v[tb]['q'] for k, v in fr])

    def proj(X):
        return np.stack([X[..., 0] / X[..., 2] * Kc[0, 0] + Kc[0, 2], X[..., 1] / X[..., 2] * Kc[1, 1] + Kc[1, 2]], -1)

    def res(x):
        P = x[6:].reshape(n, 6)
        Ra = Rot.from_rotvec(P[:, :3]).as_matrix()
        Xa = np.einsum('nij,kj->nki', Ra, OBJ) + P[:, None, 3:]
        Rr = Rot.from_rotvec(x[:3]).as_matrix()
        Ob = OBJ @ Rr.T + x[3:6]
        Xb = np.einsum('nij,kj->nki', Ra, Ob) + P[:, None, 3:]
        return np.concatenate([(proj(Xa) - Qa).reshape(n, 8), (proj(Xb) - Qb).reshape(n, 8)], 1).ravel()
    J = lil_matrix((16 * n, 6 + 6 * n), dtype=np.int8)
    for i in range(n):
        J[16 * i:16 * i + 16, :6] = 1; J[16 * i:16 * i + 16, 6 + 6 * i:12 + 6 * i] = 1
    x = np.r_[rv, tv, np.ravel(x0)]
    r = least_squares(res, x, jac_sparsity=J, loss='huber', f_scale=1.0, x_scale='jac', method='trf', max_nfev=3000,
                      xtol=1e-14, ftol=1e-14, gtol=1e-14)
    f = r.fun.reshape(n, 16)
    Rr = Rot.from_rotvec(r.x[:3]).as_matrix(); tr = r.x[3:6]
    # tag frames: x right (along t), y up (along the jug axis, image up), z = towards the camera (outward normal)
    # aruco OBJ has y up / z out of the tag (solvePnP convention: z into the scene is the camera; OBJ z = 0 plane);
    # outward normal of tag a in its own frame = -z (camera looks at the tag along +z of the camera; tag z points away
    # from the camera in the IPPE solution for points facing the camera)
    na = np.array([0, 0, -1.0]); nb = Rr @ na
    ang = float(np.degrees(np.arccos(np.clip(na @ nb, -1, 1))))
    # axis direction: common direction perpendicular to both normals (flats are parallel to the axis)
    ax = np.cross(na, nb); ax /= np.linalg.norm(ax)
    if ax @ np.array([0, 1.0, 0]) < 0: ax = -ax
    d = tr                                       # centre of b in a's frame
    dz = float(d @ ax)                           # height difference along the axis
    dh = d - dz * ax
    apo = float(np.linalg.norm(dh) / (2 * np.sin(np.radians(ang) / 2)))
    # in-plane roll of b relative to a about the axis-aligned frames
    yb = Rr @ np.array([0, 1.0, 0])
    roll_a = float(np.degrees(np.arctan2(np.cross(np.array([0, 1.0, 0]), ax) @ na, np.array([0, 1.0, 0]) @ ax)))
    roll_b = float(np.degrees(np.arctan2(np.cross(yb, ax) @ nb, yb @ ax)))
    # where is the axis? solve centre_a + a na' ... the axis passes at distance apo behind both centres along -normal:
    # lateral (u) check: the centre of b vs the point where b's flat would be centred given a centred on its flat
    ca_axis = -apo * na                          # axis point (a's frame) if a is centred on its flat
    cb_pred = ca_axis + apo * nb                 # b centred on its flat (same height)
    u_b = float((d - dz * ax - (cb_pred - (cb_pred @ ax) * ax)) @ np.cross(ax, nb))
    return dict(cam=cam, tags=[ta, tb], n_frames=n, rms_px=float(np.sqrt(np.mean(f ** 2))),
                med_px=float(np.median(np.abs(f))), angle_deg=ang, centre_dist_mm=float(np.linalg.norm(d) * 1e3),
                dz_mm=dz * 1e3, dh_mm=float(np.linalg.norm(dh) * 1e3), apothem_if_centred_mm=apo * 1e3,
                roll_a_deg=roll_a, roll_b_deg=roll_b, status=int(r.status), nfev=int(r.nfev))


if __name__ == '__main__':
    out = {}
    for p in POURS:
        for corners in ('none', 'contour', 'edge'):
            D = PourData(p, corners)
            for cam, ta, tb in ((0, 4, 5), (1, 1, 0)):
                R = fit_pair(D, cam, ta, tb)
                if R is None:
                    continue
                out[f'{p}/{corners}/{ta}{tb}'] = R
                print(p, corners, f'{ta}-{tb}', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R.items()}, flush=True)
    json.dump(out, open(f'{V2}/out/pair_check.json', 'w'), indent=1)
