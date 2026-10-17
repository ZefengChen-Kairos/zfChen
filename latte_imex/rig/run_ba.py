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
# init layout from synchronised detections: tag j pose in tag-4 frame, median over close-in-time pairs
by = {}
for d in rows: by.setdefault(d["tag"], []).append(d)
def rel_in4(j):
    out = []
    for a in by.get(4, []):
        for b in by.get(j, []):
            if abs(a["t"] - b["t"]) < 0.012:
                out.append(np.linalg.inv(a["T_board"]) @ b["T_board"])
    return out
r0 = 0.048
init = {"r": r0, 4: (0.0, 0.0, 0.0)}
for j in sorted(by):
    if j == 4: continue
    Ts = rel_in4(j)
    if not Ts: print("no sync pair for tag", j); continue
    t = np.median([T[:3, 3] for T in Ts], 0)        # in tag-4 frame: x tangent, y axis, z normal
    # tag 4 centre at (r,0,0) in pitcher frame; tag-4 axes: x=tangent(+phi), y=axis, z=normal
    p = np.array([r0 + t[2], t[0], t[1]])
    init[j] = (float(np.arctan2(p[1], p[0])), float(p[2]), 0.0)
    print("init tag", j, "phi deg", round(np.degrees(init[j][0]), 1), "z mm", round(init[j][1] * 1e3, 1), "n pairs", len(Ts), flush=True)
t0 = time.time()
out = solve(tk, [d for d in rows if d["tag"] in init], init)
print("solve s", round(time.time() - t0, 1), "rms px", round(out["rms_px"], 2), "med", round(out["med_px"], 2), "r mm", round(out["r"] * 1e3, 1))
print({k: np.round([np.degrees(v[0]), v[1] * 1e3, np.degrees(v[2])], 1).tolist() for k, v in out["layout"].items()})
pickle.dump(out, open(f"{O}/ba.pkl", "wb"))
