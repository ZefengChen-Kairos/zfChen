"""Visual check: rendered CAD outline at the v2 pose (green) and the current pose (red, tip.npz minus pose_inliers,
CAD z0 -28.5 / az 90) on cameras 0, 1 and the held-out camera 2, plus spout tips (v2: green dot, current: red dot);
crops around the jug, N frames per pour.
    python3 sheet_v2.py <v2_all.pkl> <pour> [n] -> ../img/sheet_<pour>.jpg"""
import sys, pickle
import numpy as np, cv2
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import V2
from heldout_cam import v2_source

V = pickle.load(open(sys.argv[1], 'rb')); p = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
P = gcm.Pour(p)
pose_v, cad_v, rng = v2_source(p, V)
cad_c = gcm.CAD(z0_mm=-28.5, az_deg=90.0)
tip_c = P.tip_p.copy()
rows = []
for i in np.linspace(5, len(P.tk.T) - 6, n).round().astype(int):
    tiles = []
    for c, k, dt in P.tk.sync(i):
        tt = P.cam_time(c, k)
        im = P.tk.color(c, k).copy()
        boxes = []
        for (pose, cad, tip, col) in ((P.track, cad_c, tip_c, (0, 0, 255)), (pose_v, cad_v, cad_v.tip_pf, (0, 255, 0))):
            R, pp = pose(tt)
            dr, lab, fid = gcm.render(P, cad, c, R, pp)
            body = ((lab == 1) | (lab == 2) | (lab == 4)).astype(np.uint8)
            cs, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, col, 1)
            u, v, z = P.project(c, (R @ tip + pp)[None])
            cv2.circle(im, (int(u[0]), int(v[0])), 3, col, -1)
            ys, xs = np.nonzero(body)
            if len(xs):
                boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        if boxes:
            x0 = max(min(b[0] for b in boxes) - 25, 0); y0 = max(min(b[1] for b in boxes) - 25, 0)
            x1 = min(max(b[2] for b in boxes) + 25, im.shape[1]); y1 = min(max(b[3] for b in boxes) + 25, im.shape[0])
            crop = im[y0:y1, x0:x1]
        else:
            crop = im
        crop = cv2.resize(crop, (300, int(300 * crop.shape[0] / max(crop.shape[1], 1))))
        crop = cv2.resize(crop, (300, 300)) if crop.shape[0] > 300 else np.pad(crop, ((0, 300 - crop.shape[0]), (0, 0), (0, 0)))
        cv2.putText(crop, f'cam{c}{" (held out)" if c == 2 else ""} t={tt:.2f}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        tiles.append(crop)
    rows.append(np.hstack(tiles))
img = np.vstack(rows)
cv2.putText(img, f'{p}: green = v2, red = current (tip.npz)', (4, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imwrite(f'{V2}/img/sheet_{p}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
print('wrote', f'{V2}/img/sheet_{p}.jpg', img.shape)
