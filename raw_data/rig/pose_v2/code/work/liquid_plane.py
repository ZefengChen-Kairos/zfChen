"""Gravity reference per camera: the liquid surface in the cup is horizontal.  For every frame of each camera, depth
points inside the registered liquid disc (disc.json: centre, radius, surface z; inner 70% of the radius, within 12 mm
of the surface height) are fitted with a robust plane in that camera's (nominal-extrinsic) board frame; the median
normal per camera gives its tilt relative to the board z axis (and so relative to the other cameras and gravity).
    python3 liquid_plane.py -> ../out/liquid_plane.json"""
import sys, json
import numpy as np
sys.path.insert(0, '/home/user/zfChen/latte_imex/rig')
from trimrec import Take, cloud
SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
res = {}
allN = {0: [], 1: [], 2: []}
for p in ('tulip', 'leaf', 'heart', 'push_heart', 'swan'):
    tk = Take(f'{SCR}/pours/ghdata/{p}')
    D = json.load(open(f'{SCR}/pours/{p}/disc.json'))
    Dt = np.array([d['t'] for d in D])
    out = {}
    for c in range(3):
        N = []
        for k, ts in list(zip(tk.steps[c], tk.ts[c] - tk.t0))[::2]:
            d = D[int(np.argmin(np.abs(Dt - ts)))]
            P, col, uv = cloud(tk, c, k, 2)
            cen = np.array(d['disc_centre'][:2]); R = d['disc_R']; zs = d['surf_z']
            m = (np.hypot(P[:, 0] - cen[0], P[:, 1] - cen[1]) < 0.7 * R) & (np.abs(P[:, 2] - zs) < 0.012)
            Q = P[m]
            if len(Q) < 300:
                continue
            for _ in range(3):
                c0 = Q.mean(0); U, S, Vt = np.linalg.svd(Q - c0); n = Vt[2] * np.sign(Vt[2][2])
                r = (Q - c0) @ n
                keep = np.abs(r) < max(2.5 * np.median(np.abs(r)), 0.0008)
                Q = Q[keep]
            N.append(n)
        N = np.array(N)
        if len(N):
            nm = np.median(N, 0); nm /= np.linalg.norm(nm)
            allN[c] += list(N)
            out[f'cam{c}'] = dict(n_frames=len(N), normal=nm.round(5).tolist(),
                                  tilt_about_board_x_deg=float(np.degrees(np.arctan2(-nm[1], nm[2]))),
                                  tilt_about_board_y_deg=float(np.degrees(np.arctan2(nm[0], nm[2]))),
                                  spread_deg=float(np.degrees(np.median(np.arccos(np.clip(N @ nm, -1, 1))))))
    res[p] = out
    print(p, {c: (v['n_frames'], round(v['tilt_about_board_x_deg'], 2), round(v['tilt_about_board_y_deg'], 2), round(v['spread_deg'], 2)) for c, v in out.items()}, flush=True)
pool = {}
for c, N in allN.items():
    N = np.array(N); nm = np.median(N, 0); nm /= np.linalg.norm(nm)
    pool[f'cam{c}'] = dict(n=len(N), tilt_about_board_x_deg=float(np.degrees(np.arctan2(-nm[1], nm[2]))),
                           tilt_about_board_y_deg=float(np.degrees(np.arctan2(nm[0], nm[2]))))
res['pooled'] = pool
print('pooled', pool)
json.dump(res, open(f'{SCR}/pose_v2/out/liquid_plane.json', 'w'), indent=1)
