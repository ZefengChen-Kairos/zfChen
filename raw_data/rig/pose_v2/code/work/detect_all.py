"""Re-detect the pitcher AprilTags in cameras 0 and 1 (camera 2 is held out: never read here) with three corner
estimates per detection: q_none (= the current tags.pkl, integer pixels), q_contour (aruco CORNER_REFINE_CONTOUR),
q_edge (edgefit.refine: lines through the subpixel gradient maxima of the black-square edges).
IPPE pose alternatives (board frame) of each tag from q_edge for initialisation.
    python3 detect_all.py <pour>  -> ../data/tags_<pour>.pkl"""
import sys, pickle, numpy as np, cv2
sys.path.insert(0, '/home/user/zfChen/latte_imex/rig')
from trimrec import Take
from edgefit import refine
SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
TAG = 0.0190
OBJ = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * TAG / 2
D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def det(method):
    pr = cv2.aruco.DetectorParameters(); pr.cornerRefinementMethod = method
    return cv2.aruco.ArucoDetector(D, pr)


DN, DC = det(cv2.aruco.CORNER_REFINE_NONE), det(cv2.aruco.CORNER_REFINE_CONTOUR)
p = sys.argv[1]
tk = Take(f"{SCR}/pours/ghdata/{p}")
rows = []
for c in (0, 1):
    for k, t in zip(tk.steps[c], tk.ts[c] - tk.t0):
        g = cv2.cvtColor(tk.color(c, k), cv2.COLOR_BGR2GRAY)
        cn, idn, _ = DN.detectMarkers(g)
        cc_, idc, _ = DC.detectMarkers(g)
        if idn is None and idc is None:
            continue
        gb = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 0.8)
        dn = {int(i): q.reshape(4, 2).astype(float) for q, i in zip(cn, idn.ravel())} if idn is not None else {}
        dc = {int(i): q.reshape(4, 2).astype(float) for q, i in zip(cc_, idc.ravel())} if idc is not None else {}
        for i in sorted(set(dn) | set(dc)):
            q0 = dc.get(i, dn.get(i))
            qe = refine(gb, q0)
            alts = []
            if qe is not None:
                n_, rvs, tvs, _ = cv2.solvePnPGeneric(OBJ, qe, tk.K[c], None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                for rv_, tv_ in zip(rvs, tvs):
                    Ta = np.eye(4); Ta[:3, :3] = cv2.Rodrigues(rv_)[0]; Ta[:3, 3] = tv_.ravel(); alts.append(tk.c2w[c] @ Ta)
            rows.append(dict(cam=c, step=int(k), t=float(t), tag=i, q_none=dn.get(i), q_contour=dc.get(i), q_edge=qe,
                             T_board_alts=alts, px=float(cv2.contourArea(q0.astype(np.float32)))))
pickle.dump(rows, open(f"{SCR}/pose_v2/data/tags_{p}.pkl", "wb"))
from collections import Counter
print(p, len(rows), Counter((d['cam'], d['tag']) for d in rows), 'edge ok', sum(d['q_edge'] is not None for d in rows),
      'none', sum(d['q_none'] is not None for d in rows), 'contour', sum(d['q_contour'] is not None for d in rows))
