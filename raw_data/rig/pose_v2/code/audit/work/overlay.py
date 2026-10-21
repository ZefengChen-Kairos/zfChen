"""A9: overlays in all three cameras: v2 (green; camera 1 drawn through the fitted camera-1 correction, cyan = the same
v2 pose through the RAW camera-1 calibration as a consumer would), current tip.npz track (red, CAD z0 -28.5 az 90),
spout tips (dots).  Each row = one time; each camera at its own nearest frame; crops at full resolution.
    python3 overlay.py <name> <pour:t> [<pour:t> ...] -> ../img/<name>.jpg"""
import sys, json, pickle
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
V2 = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2'
J = pickle.load(open(f'{V2}/out/joint_v2k.pkl', 'rb'))
e1 = np.array(J['layout']['ext1']); cen = np.array(J['layout']['ext_cen']); Re = Rot.from_rotvec(e1[:3]).as_matrix()
cad_v = gcm.CAD(z0_mm=-29.0, az_deg=90.0); cad_c = gcm.CAD(z0_mm=-28.5, az_deg=90.0)
CROP = 300
_P = {}
def pour(p):
    if p not in _P:
        P = gcm.Pour(p); Z = np.load(f'{V2}/out/{p}_pose_v2.npz')
        _P[p] = (P, gcm.Track(Z['t'], Z['poses']), Z, J['pours'][p]['nuis']['c1'])
    return _P[p]
def contour(P, cad, c, R, pp):
    dr, lab, fid = gcm.render(P, cad, c, R, pp)
    body = ((lab == 1) | (lab == 2) | (lab == 4)).astype(np.uint8)
    cs, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cs, body
rows = []
for arg in sys.argv[2:]:
    p, t = arg.split(':'); t = float(t)
    P, trv, Z, c1 = pour(p)
    tiles = []
    for c in range(3):
        ts = P.tk.ts[c] - P.tk.t0
        j = int(np.argmin(np.abs(ts - t))); k = P.tk.steps[c][j]; tc = float(ts[j])
        im = P.tk.color(c, k).copy()
        tv = tc + (c1 if c == 1 else 0.0)
        Rv, pv = trv(tv)
        Rc, pc = P.track(tc)
        draws = []
        if c == 1:
            draws.append(('raw', cad_v, Rv, pv, (255, 255, 0), Z['tip_p']))
            Rv2, pv2 = Re @ Rv, Re @ (pv - cen) + cen + e1[3:]
            draws.append(('v2', cad_v, Rv2, pv2, (0, 255, 0), Z['tip_p']))
        else:
            draws.append(('v2', cad_v, Rv, pv, (0, 255, 0), Z['tip_p']))
        draws.append(('cur', cad_c, Rc, pc, (0, 0, 255), P.tip_p))
        allb = None
        for name, cad, R, pp, col, tip in draws:
            cs, body = contour(P, cad, c, R, pp)
            cv2.drawContours(im, cs, -1, col, 1)
            u, v, z = P.project(c, (R @ tip + pp)[None])
            cv2.circle(im, (int(round(u[0])), int(round(v[0]))), 3, col, -1)
            allb = body if allb is None else (allb | body)
        ys, xs = np.nonzero(allb)
        if len(xs):
            cx, cy = int(np.median(xs)), int(np.median(ys))
        else:
            cx, cy = 424, 240
        half = max(CROP // 2, int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.6) if len(xs) else CROP // 2)
        x0, y0 = max(cx - half, 0), max(cy - half, 0)
        crop = im[y0:min(cy + half, 480), x0:min(cx + half, 848)]
        crop = cv2.resize(crop, (CROP, CROP)) if crop.size else np.zeros((CROP, CROP, 3), np.uint8)
        cv2.putText(crop, f'{p} cam{c} t={tc:.3f}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(crop)
    rows.append(np.hstack(tiles))
img = np.vstack(rows)
cv2.putText(img, 'green v2 | cyan v2 via raw cam1 calib | red current', (4, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
cv2.imwrite(f'{V2}/audit/img/{sys.argv[1]}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
print('wrote', img.shape)
