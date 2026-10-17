import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pickle, cv2, time
from scipy.spatial.transform import Rotation as Rot
from trimrec import Take
import tags as TG
from pitcher_ba import solve
take = sys.argv[1]
P = f"/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours/ghdata/{take}"
O = f"/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours/{take}"
tk = Take(P)
rows = []
for c in range(3):
    for k, t in zip(tk.steps[c], tk.ts[c] - tk.t0):
        g = cv2.cvtColor(tk.color(c, k), cv2.COLOR_BGR2GRAY)
        corners, ids, _ = TG._det.detectMarkers(g)
        if ids is None: continue
        dd = TG.detect(tk, c, k)
        for cc, i in zip(corners, ids.ravel()):
            if int(i) in dd: rows.append(dict(cam=c, step=int(k), t=float(t), tag=int(i), q=cc.reshape(4, 2).astype(float), **dd[int(i)]))
pickle.dump(rows, open(f"{O}/tags.pkl", "wb"))
print("detections", len(rows), flush=True)
# init layout: tag pairs seen by the SAME camera in the SAME frame (no sync error); tag 1 relative to tag 4 from the
# heart fit (never seen together by one camera) -- the layout is a property of the pitcher, identical in every take
import json as _json
HL = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tag_layout.json")))
r0 = HL["r"]
init = {"r": r0, 4: (0.0, 0.0, 0.0), 1: tuple(HL["1"])}
def same_cam(a, b):
    by = {}
    for d in rows: by.setdefault((d["cam"], d["step"]), {})[d["tag"]] = d
    return [np.linalg.inv(v[a]["T_cam"]) @ v[b]["T_cam"] for v in by.values() if a in v and b in v]
def place(ref, j):
    Ts = same_cam(ref, j)
    if not Ts: return None
    t = np.median([T[:3, 3] for T in Ts], 0)
    phr, zr = init[ref][0], init[ref][1]
    # tag-ref frame: x tangent, y axis, z outward normal -> pitcher frame
    n = np.array([np.cos(phr), np.sin(phr), 0]); ex = np.array([-np.sin(phr), np.cos(phr), 0]); ez = np.array([0, 0, 1.0])
    p = r0 * n + zr * ez + t[0] * ex + t[1] * ez + t[2] * n
    return (float(np.arctan2(p[1], p[0])), float(p[2]), 0.0), len(Ts)
for ref, j in [(4, 5), (1, 0)]:
    if any(d["tag"] == j for d in rows):
        pl = place(ref, j)
        if pl: init[j] = pl[0]; print("init tag", j, "phi deg", round(np.degrees(pl[0][0]), 1), "z mm", round(pl[0][1] * 1e3, 1), "pairs", pl[1], flush=True)
t0 = time.time()
out = solve(tk, [d for d in rows if d["tag"] in init], init)
print("solve s", round(time.time() - t0, 1), "rms px", round(out["rms_px"], 2), "med", round(out["med_px"], 2), "r mm", round(out["r"] * 1e3, 1))
print({k: np.round([np.degrees(v[0]), v[1] * 1e3, np.degrees(v[2])], 1).tolist() for k, v in out["layout"].items()})
pickle.dump(out, open(f"{O}/ba.pkl", "wb"))
