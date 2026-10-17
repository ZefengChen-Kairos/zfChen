"""Dense contact sheet of one camera between t1 and t2 (every dt s), 5 columns."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from rs_zip import Rec
path, out, cam, t1, t2, dt = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
r = Rec(path)
T = r.ts[cam] - r.t0
ims = []
for tt in np.arange(t1, min(t2, T[-1]) + 1e-9, dt):
    i = int(np.argmin(np.abs(T - tt)))
    im = cv2.resize(r.color(cam, r.steps[cam][i]), (424, 240))
    cv2.putText(im, f"t={T[i]:.2f}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    ims.append(im)
while len(ims) % 5: ims.append(np.zeros_like(ims[0]))
rows = [np.hstack(ims[i:i + 5]) for i in range(0, len(ims), 5)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 75])
print("duration", round(T[-1], 2))
