"""AprilTag (36h11) poses of the pitcher markers in the board frame, per camera frame."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from trimrec import Take

TAG = 0.0182   # edge of the black square [m], estimated from depth (to be confirmed)
_det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), cv2.aruco.DetectorParameters())
OBJ = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * TAG / 2   # aruco corner order: TL, TR, BR, BL


def detect(tk, c, k):
    g = cv2.cvtColor(tk.color(c, k), cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _det.detectMarkers(g)
    out = {}
    if ids is None: return out
    for cc, i in zip(corners, ids.ravel()):
        q = cc.reshape(4, 2).astype(np.float64)
        cv2.cornerSubPix(g, q.reshape(-1, 1, 2).astype(np.float32), (3, 3), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        ok, rv, tv = cv2.solvePnP(OBJ, q, tk.K[c], None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok: continue
        R = cv2.Rodrigues(rv)[0]
        rep = cv2.projectPoints(OBJ, rv, tv, tk.K[c], None)[0].reshape(-1, 2)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tv.ravel()
        out[int(i)] = dict(T_cam=T, T_board=tk.c2w[c] @ T, err=float(np.abs(rep - q).max()), px=float(cv2.contourArea(q.astype(np.float32))))
    return out


def all_detections(tk):
    rows = []
    for c in range(tk.ncam):
        for k, t in zip(tk.steps[c], tk.ts[c] - tk.t0):
            for i, d in detect(tk, c, k).items():
                rows.append(dict(cam=c, step=int(k), t=float(t), tag=i, **d))
    return rows
