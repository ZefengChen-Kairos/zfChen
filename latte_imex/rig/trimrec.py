"""Reader for trimmed takes (pours-trimmed branch layout) + synchronised multi-camera frames."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from safepkl import load_c2ws


class Take:
    def __init__(self, d):
        self.d = d
        self.meta = json.load(open(f"{d}/metadata.json"))
        self.K = [np.array(k, float) for k in self.meta["intrinsics"]]
        self.c2w = load_c2ws(open(f"{d}/calibrate.pkl", "rb").read())
        self.ncam = len(self.K)
        self.steps, self.ts = [], []
        for c in range(self.ncam):
            r = self.meta["recording"][str(c)]
            ks = sorted(int(k) for k in r)
            self.steps.append(np.array(ks)); self.ts.append(np.array([r[str(k)] for k in ks]))
        self.t0 = self.meta["cut"]["t_first_frame_unix"]
        # master clock: camera 0 frames; other cameras matched by nearest timestamp
        self.T = self.ts[0] - self.t0

    def color(self, c, k):
        return cv2.imread(f"{self.d}/color/{c}/{k}.webp")

    def depth(self, c, k):
        return cv2.imread(f"{self.d}/depth/{c}/{k}.png", cv2.IMREAD_UNCHANGED)

    def sync(self, i):
        """frame index i of camera 0 -> [(cam, step, dt_to_master)]"""
        t = self.ts[0][i]; out = []
        for c in range(self.ncam):
            j = int(np.argmin(np.abs(self.ts[c] - t))); out.append((c, self.steps[c][j], self.ts[c][j] - t))
        return out


def cloud(take, c, k, stride=1):
    d = take.depth(c, k).astype(float) * 1e-3
    K, T = take.K[c], take.c2w[c]
    H, W = d.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]; z = d[::stride, ::stride]; ok = z > 0
    x = (u[ok] - K[0, 2]) / K[0, 0] * z[ok]; y = (v[ok] - K[1, 2]) / K[1, 1] * z[ok]
    P = np.stack([x, y, z[ok]], 1) @ T[:3, :3].T + T[:3, 3]
    col = take.color(c, k)[::stride, ::stride][ok][:, ::-1]
    return P, col, np.stack([u[ok], v[ok]], 1)
