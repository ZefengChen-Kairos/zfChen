"""zoom: v2 (green) vs the per-sample single-frame pose of that very frame (blue = camera-0 frame pose from its own tags,
magenta = camera-1 frame pose) in cameras 0 and 1 at fast frames."""
import sys, json, pickle
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
V2 = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2'
J = pickle.load(open(f'{V2}/out/joint_v2k.pkl', 'rb'))
e1 = np.array(J['layout']['ext1']); cen = np.array(J['layout']['ext_cen']); Re = Rot.from_rotvec(e1[:3]).as_matrix()
cad = gcm.CAD(z0_mm=-29.0, az_deg=90.0)
p = sys.argv[1]; times = [float(x) for x in sys.argv[3:]]
P = gcm.Pour(p); Z = np.load(f'{V2}/out/{p}_pose_v2.npz'); trv = gcm.Track(Z['t'], Z['poses']); B = Z['B_out_to_tag']
S = np.load(f'{V2}/audit/out/samples_{p}.npz'); c1 = J['pours'][p]['nuis']['c1']
rows = []
for t in times:
    tiles = []
    for c in (0, 1):
        ts = P.tk.ts[c] - P.tk.t0; j = int(np.argmin(np.abs(ts - t))); k = P.tk.steps[c][j]; tc = float(ts[j]); tv = tc + (c1 if c == 1 else 0)
        im = P.tk.color(c, k).copy()
        poses = [('v2', trv(tv), (0, 255, 0))]
        kd = 'c0' if c == 0 else 'c1'
        m = np.where((S['kind'] == kd) & (np.abs(S['t'] - tv) < 1e-4))[0]
        if len(m):
            Ro = Rot.from_rotvec(S['rotvec'][m[0]]).as_matrix(); Rt = Ro @ B[:3, :3].T; po = Rt @ B[:3, 3] + S['pos'][m[0]]
            poses.append(('own', (Ro, po), (255, 0, 0) if c == 0 else (255, 0, 255)))
            dtl = S['tilt'][m[0]] - S['tilt_v2'][m[0]]
        else:
            dtl = np.nan
        ys = []
        for name, (R, pp), col in poses:
            if c == 1: R, pp = Re @ R, Re @ (pp - cen) + cen + e1[3:]
            dr, lab, fid = gcm.render(P, cad, c, R, pp)
            body = ((lab == 1) | (lab == 2) | (lab == 4)).astype(np.uint8)
            cs, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, col, 1)
            ys.append(np.nonzero(body))
        yy = np.concatenate([y[0] for y in ys]); xx = np.concatenate([y[1] for y in ys])
        cx, cy = int(np.median(xx)), int(np.median(yy)); h = 160
        crop = im[max(cy - h, 0):cy + h, max(cx - h, 0):cx + h]
        crop = cv2.resize(crop, (480, 480), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f'{p} cam{c} t={tc:.3f} own-v2 tilt {dtl:+.2f} deg', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(crop)
    rows.append(np.hstack(tiles))
cv2.imwrite(f'{V2}/audit/img/{sys.argv[2]}.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print('ok')
