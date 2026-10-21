"""Corner-refinement test: integer (current tags.pkl) vs SUBPIX vs APRILTAG refinement.
Metric: per cam-0 frame with tags 4 and 5 both detected (one camera, one exposure: no sync error), fixed-layout pose
from both tags (converged layout of the hold-out audit) -> corner rms; lower = more accurate corners."""
import sys, pickle, numpy as np, cv2
sys.path.insert(0, '/home/user/zfChen/latte_imex/rig')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/holdout')
from common import Geo, load, OUT
from trimrec import Take
p = sys.argv[1] if len(sys.argv) > 1 else 'swan'
tk, ba, rows, tip = load(p)
cv = pickle.load(open(f"{OUT}/conv_{p}.pkl", "rb"))
G = Geo(tk, dict(r=cv['r'], layout=cv['layout']))
D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def det(method, win=3):
    pr = cv2.aruco.DetectorParameters()
    pr.cornerRefinementMethod = method
    pr.cornerRefinementWinSize = win
    pr.cornerRefinementMinAccuracy = 0.01
    pr.cornerRefinementMaxIterations = 50
    return cv2.aruco.ArucoDetector(D, pr)


variants = {'none': det(cv2.aruco.CORNER_REFINE_NONE), 'subpix3': det(cv2.aruco.CORNER_REFINE_SUBPIX, 3),
            'subpix2': det(cv2.aruco.CORNER_REFINE_SUBPIX, 2), 'contour': det(cv2.aruco.CORNER_REFINE_CONTOUR),
            'edge': det(cv2.aruco.CORNER_REFINE_CONTOUR)}
from edgefit import refine
res = {k: [] for k in variants}
cnt = {k: 0 for k in variants}
tags_seen = {k: {} for k in variants}
for c in (0, 1):
    for k in tk.steps[c][::2]:
        g = cv2.cvtColor(tk.color(c, k), cv2.COLOR_BGR2GRAY)
        for name, dt in variants.items():
            corners, ids, _ = dt.detectMarkers(g)
            if ids is None:
                continue
            ob = [dict(cam=c, tag=int(i), q=cc.reshape(4, 2).astype(float)) for cc, i in zip(corners, ids.ravel()) if int(i) in G.L]
            if name == 'edge':
                gb = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 0.8)
                ob2 = []
                for o in ob:
                    qq = refine(gb, o['q'])
                    if qq is not None: ob2.append(dict(o, q=qq))
                ob = ob2
            for o in ob:
                tags_seen[name][(c, o['tag'])] = tags_seen[name].get((c, o['tag']), 0) + 1
            if c == 0 and len(ob) >= 2:
                # init from the hold-out converged per-sample pose nearest in time
                tt = tk.ts[c][list(tk.steps[c]).index(k)] - tk.t0
                j = int(np.argmin(np.abs(np.asarray(cv['t']) - tt)))
                from scipy.optimize import least_squares
                r = least_squares(G.resid, cv['poses'][j], args=(ob,), method='lm')
                res[name].append(np.sqrt(np.mean(r.fun ** 2)) * np.sqrt(len(r.fun) / max(len(r.fun) - 6, 1)))
for name in variants:
    a = np.array(res[name])
    if len(a) == 0: continue
    print(name, 'n', len(a), 'two-tag rms px median', round(np.median(a), 3), 'p90', round(np.percentile(a, 90), 3), 'dets', tags_seen[name])
