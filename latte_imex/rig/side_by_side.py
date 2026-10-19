"""Side-by-side movie (barista view): real pour (multi-view rectified top view of the liquid surface, per frame, from
the pipeline's top/ frames) | simulation (left panel of sim_<variant>/movie.mp4), on one clock (simulation t = 0 at the
first frame of the pattern phase). Both in disc coordinates, rotated by targets/view_rotation.json.
    python latte_imex/rig/side_by_side.py <pattern> [variant=sim_realpitcher]"""
import sys, os, json
import numpy as np, cv2, imageio
SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NAMES = dict(heart="爱心 heart", push_heart="推推乐 push heart", leaf="树叶 leaf", swan="天鹅 swan", tulip="郁金香 tulip")
p = sys.argv[1]; var = sys.argv[2] if len(sys.argv) > 2 else "sim_realpitcher"
S, EXT, FPS = 480, 1.05, 24
rot = json.load(open(f"{REPO}/raw_data/rig/targets/view_rotation.json"))[p]
D = json.load(open(f"{SCR}/{p}/disc.json"))
td = np.array([d["t"] for d in D]); t0 = td[0]
info = json.load(open(f"{REPO}/raw_data/rig/results/{p}/{var}/info.json"))
M = cv2.getRotationMatrix2D((S / 2 - 0.5, S / 2 - 0.5), rot, 1.0)
s = np.linspace(-EXT, EXT, S); U, V = np.meshgrid(s, s); outside = U ** 2 + V ** 2 > 1.0
def real_frame(fi):
    Z = np.load(f"{SCR}/{p}/top/{fi:04d}.npz"); img, c0, PX, HALF = Z["img"], Z["c0"], float(Z["px"]), float(Z["half"])
    cen = np.array(D[fi]["disc_centre"]); R = D[fi].get("L_m") or D[fi]["disc_R"]   # one length unit with the sim
    mx = ((cen[0] + U * R - c0[0] + HALF) / PX - 0.5).astype(np.float32); my = ((cen[1] + V * R - c0[1] + HALF) / PX - 0.5).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR); out[outside] = 0
    return cv2.warpAffine(out, M, (S, S))
cap = cv2.VideoCapture(f"{REPO}/raw_data/rig/results/{p}/{var}/movie.mp4"); n = int(cap.get(7)); mfps = cap.get(5)
sim = []
while True:
    ok, f = cap.read()
    if not ok: break
    L = f[:, :768]; c, r = 383.5, 376.0
    mx = (c + U * r).astype(np.float32); my = (c - (-V) * r).astype(np.float32)   # movie: image up = +F y = -board y; target rows = +board y
    fr = cv2.remap(L, mx, my, cv2.INTER_LINEAR); fr[outside] = 0
    sim.append(cv2.warpAffine(fr, M, (S, S)))
T = min(td[-1] - t0, (len(sim) - 1) / mfps)
font = cv2.FONT_HERSHEY_SIMPLEX
frames = []
for t in np.arange(0, T, 1 / FPS):
    fi = int(np.argmin(np.abs(td - t0 - t))); si = min(int(round(t * mfps)), len(sim) - 1)
    a = real_frame(fi); b = sim[si]
    canvas = np.full((S + 70, 2 * S + 30, 3), 25, np.uint8)
    canvas[60:60 + S, 10:10 + S] = a; canvas[60:60 + S, 20 + S:20 + 2 * S] = b
    cv2.putText(canvas, "real (3 cameras, rectified)", (14, 30), font, 0.7, (230, 230, 230), 2)
    cv2.putText(canvas, f"simulation ({var})", (24 + S, 30), font, 0.7, (230, 230, 230), 2)
    cv2.putText(canvas, f"{p}   t = {t:5.2f} s", (14, 54), font, 0.6, (0, 220, 255), 1)
    frames.append(canvas[..., ::-1])
for k in range(FPS):   # hold the last frame 1 s
    frames.append(frames[-1])
out = f"{REPO}/raw_data/rig/results/{p}/{var}/side_by_side.mp4"
imageio.mimwrite(out, frames, fps=FPS, codec="libx264", quality=7, macro_block_size=2)
print(out, len(frames), "frames")
