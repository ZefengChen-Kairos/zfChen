"""Real-pour panel for the AutoDL renderer: the multi-view rectified top view of the liquid surface (pipeline top/
frames) resampled onto the disc grid, barista view (targets/view_rotation.json), 24 fps on the simulation clock
(t = 0 at the first frame of the pattern phase).  Run locally where the raw frames are; writes
raw_data/rig/real_panel/<p>.mp4 (480 x 480, disc extent +-1.05 R).
    python latte_imex/rig/real_panel.py [pattern ...]"""
import sys, os, json
import numpy as np, cv2, imageio
SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
S, EXT, FPS = 480, 1.05, 24
s = np.linspace(-EXT, EXT, S); U, V = np.meshgrid(s, s); outside = U ** 2 + V ** 2 > 1.0
os.makedirs(f"{REPO}/raw_data/rig/real_panel", exist_ok=True)
for p in sys.argv[1:] or ["heart", "push_heart", "leaf", "swan", "tulip"]:
    rot = json.load(open(f"{REPO}/raw_data/rig/targets/view_rotation.json"))[p]
    M = cv2.getRotationMatrix2D((S / 2 - 0.5, S / 2 - 0.5), rot, 1.0)
    D = json.load(open(f"{SCR}/{p}/disc.json"))
    td = np.array([d["t"] for d in D]); t0 = td[0]
    frames = []
    for t in np.arange(0, td[-1] - t0, 1 / FPS):
        fi = int(np.argmin(np.abs(td - t0 - t)))
        Z = np.load(f"{SCR}/{p}/top/{fi:04d}.npz"); img, c0, PX, HALF = Z["img"], Z["c0"], float(Z["px"]), float(Z["half"])
        cen = np.array(D[fi]["disc_centre"]); R = D[fi]["disc_R"]
        mx = ((cen[0] + U * R - c0[0] + HALF) / PX - 0.5).astype(np.float32)
        my = ((cen[1] + V * R - c0[1] + HALF) / PX - 0.5).astype(np.float32)
        out = cv2.remap(img, mx, my, cv2.INTER_LINEAR); out[outside] = 0
        frames.append(cv2.warpAffine(out, M, (S, S))[..., ::-1])
    fn = f"{REPO}/raw_data/rig/real_panel/{p}.mp4"
    imageio.mimwrite(fn, frames, fps=FPS, codec="libx264", quality=7, macro_block_size=2)
    print(fn, len(frames), "frames", f"{os.path.getsize(fn) / 1e6:.1f} MB")
