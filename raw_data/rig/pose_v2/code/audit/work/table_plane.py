"""A11: calibration cross-check without the pitcher: the dominant horizontal support plane (table / mat, world z ~ +0.10 m,
board up = -z) fitted robustly in each camera's depth (median of several frames), mapped to the board frame with each
camera's own extrinsic.  Differences of the world normals = relative tilt errors of the camera extrinsics about the two
horizontal board axes (the axes that matter for pouring tilt)."""
import sys, json, numpy as np
sys.path.insert(0, '/home/user/zfChen/latte_imex/rig')
from trimrec import Take
from scipy.spatial.transform import Rotation as Rot
import pickle
SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
V2 = f'{SCR}/pose_v2'
J = pickle.load(open(f'{V2}/out/joint_v2k.pkl', 'rb')); e1 = np.array(J['layout']['ext1'])
Re = Rot.from_rotvec(e1[:3]).as_matrix()
def plane(X, iters=300, thr=0.003, rng=np.random.default_rng(0)):
    best = None
    for _ in range(iters):
        s = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; d = np.abs((X - s[0]) @ n); c = (d < thr).sum()
        if best is None or c > best[0]: best = (c, d < thr)
    Y = X[best[1]]; c = Y.mean(0); u, s_, vt = np.linalg.svd(Y - c); n = vt[2]
    if n[2] < 0: n = -n
    return n, c, best[1].mean(), len(Y)
out = {}
for p in ['tulip', 'leaf', 'heart', 'push_heart', 'swan']:
    tk = Take(f'{SCR}/pours/ghdata/{p}')
    res = {}
    for c in range(3):
        Xs = []
        for k in tk.steps[c][::max(len(tk.steps[c]) // 6, 1)][:6]:
            dep = tk.depth(c, k).astype(float) * 1e-3
            K = tk.K[c]; T = tk.c2w[c]
            vv, uu = np.mgrid[0:480:3, 0:848:3]; z = dep[vv, uu]; m = z > 0.1
            X = np.stack([(uu[m] - K[0, 2]) / K[0, 0] * z[m], (vv[m] - K[1, 2]) / K[1, 1] * z[m], z[m]], 1) @ T[:3, :3].T + T[:3, 3]
            Xs.append(X[(X[:, 2] > 0.085) & (X[:, 2] < 0.12)])
        X = np.concatenate(Xs)
        if len(X) < 500:
            res[c] = None; continue
        X = X[np.random.default_rng(1).choice(len(X), min(len(X), 30000), replace=False)]
        n, cen, frac, ninl = plane(X)
        res[c] = dict(n=n.tolist(), z=float(cen[2]), frac=float(frac), n_inl=int(ninl), centre=cen.tolist())
    # tilt angles of each plane normal about board x / y (deg); and with the v2 camera-1 correction applied
    line = []
    for c in range(3):
        if res[c] is None: line.append(f'cam{c}: none'); continue
        n = np.array(res[c]['n'])
        ax, ay = np.degrees(np.arctan2(n[1], n[2])), np.degrees(np.arctan2(-n[0], n[2]))
        res[c]['rot_x_deg'], res[c]['rot_y_deg'] = float(ax), float(ay)
        s = f"cam{c}: z {res[c]['z']*1e3:.1f} mm inl {res[c]['n_inl']} ({res[c]['frac']:.2f}) normal tilt about x {ax:+.2f} y {ay:+.2f}"
        if c == 1:
            nc = Re.T @ n          # camera-1 points corrected into camera-0's world: X_true = Re^T (X_b - ...) -> normals by Re^T
            s += f" | corrected: x {np.degrees(np.arctan2(nc[1], nc[2])):+.2f} y {np.degrees(np.arctan2(-nc[0], nc[2])):+.2f}"
        line.append(s)
    out[p] = res
    print(p, ' || '.join(line), flush=True)
json.dump(out, open(f'{V2}/audit/out/table_plane.json', 'w'), indent=1)
