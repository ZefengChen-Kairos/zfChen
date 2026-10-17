"""Display orientation 'as the barista sees it': rotate disc-coordinate images so that the pitcher body (barista's
side) is at the bottom. Direction = mean horizontal vector from the liquid-disc centre to the pitcher axis point
(pitcher-frame origin) during the pattern phase, in disc/image coordinates (col = board x, row = board y).
Writes view_rotation.json {pattern: angle_deg (cv2.getRotationMatrix2D convention)} and *_barista.png images."""
import json, os, sys
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
B = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
REPO = "/home/user/zfChen/raw_data/rig"
ang = {}
for p in ["heart", "push_heart", "leaf", "swan", "tulip"]:
    T = np.load(f"{B}/{p}/tip.npz"); D = json.load(open(f"{B}/{p}/disc.json"))
    td = np.array([d["t"] for d in D]); cen = np.array([d["disc_centre"] for d in D])
    pos = T["poses"][:, 3:5]; t = T["t"]
    c = np.stack([np.interp(t, td, cen[:, 0]), np.interp(t, td, cen[:, 1])], 1)
    v = np.median(pos - c, 0)                     # board x,y == image col,row direction
    a_img = np.degrees(np.arctan2(v[1], v[0]))    # angle of v in image coords (x right, y down)
    # rotate so v points down (+90 deg in image coords); cv2 angle is counter-clockwise for positive values in
    # standard display, i.e. it rotates image-coordinate vectors by -angle
    ang[p] = float(a_img - 90.0)
    print(p, "pitcher direction (img deg)", round(a_img, 1), "rotate", round(ang[p], 1))
json.dump(ang, open(f"{REPO}/targets/view_rotation.json", "w"), indent=1)
def rot(img, a):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), a, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
rows = []
for p, a in ang.items():
    img = cv2.imread(f"{REPO}/targets/{p}.png"); msk = cv2.imread(f"{REPO}/targets/{p}_mask.png")
    ri, rm = rot(img, a), rot(msk, a)
    cv2.imwrite(f"{REPO}/targets/{p}_barista.png", ri)
    rows.append(np.hstack([cv2.putText(ri.copy(), p, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2), rm]))
cv2.imwrite(f"{REPO}/targets/targets_all_barista.jpg", cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 90])
