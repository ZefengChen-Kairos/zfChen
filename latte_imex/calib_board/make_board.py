"""Printable ChArUco board for the 3-camera rig: A3 landscape, 9 x 6 squares of 40 mm, ArUco 30 mm, DICT_5X5_100.
    python make_board.py   -> charuco_A3_9x6_40mm.pdf / .png
Print at 100 % ("actual size", not "fit to page"); measure the 100 mm bar and one square afterwards."""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SQ, MK, NX, NY, PXMM = 40.0, 30.0, 9, 6, 20
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((NX, NY), SQ / 1000, MK / 1000, d)
W, H = int(NX * SQ * PXMM), int(NY * SQ * PXMM)
img = board.generateImage((W, H), marginSize=0)
cv2.imwrite("charuco_A3_9x6_40mm.png", img)
PW, PH = 420.0, 297.0
fig = plt.figure(figsize=(PW / 25.4, PH / 25.4))
x0, y0 = (PW - NX * SQ) / 2, (PH - NY * SQ) / 2 + 8
ax = fig.add_axes([x0 / PW, y0 / PH, NX * SQ / PW, NY * SQ / PH])
ax.imshow(img, cmap="gray", interpolation="nearest"); ax.axis("off")
t = fig.add_axes([0, 0, 1, 1], facecolor="none"); t.set_xlim(0, PW); t.set_ylim(0, PH); t.axis("off")
t.plot([x0, x0 + 100], [y0 - 9, y0 - 9], "k-", lw=2)
for x in (x0, x0 + 100):
    t.plot([x, x], [y0 - 11, y0 - 7], "k-", lw=1.5)
t.text(x0 + 105, y0 - 9, "100 mm", va="center", fontsize=9)
t.text(x0 + NX * SQ, y0 - 9, "ChArUco 9x6, square 40 mm, marker 30 mm, DICT_5X5_100 | print at 100% (actual size)",
       va="center", ha="right", fontsize=8)
fig.savefig("charuco_A3_9x6_40mm.pdf")
print("ok", W, H)
