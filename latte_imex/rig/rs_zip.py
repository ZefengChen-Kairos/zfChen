"""Read a recording zip (CameraSystem.record format) without extracting: color/<cam>/<step>.png, depth/<cam>/<step>.npy,
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
metadata.json, calibrate.pkl (list of 4x4 camera-to-board)."""
import io
import json
import zipfile

import cv2
import numpy as np

from safepkl import load_c2ws


class Rec:
    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        names = self.z.namelist()
        self.root = names[0].split("/")[0]
        self.meta = json.loads(self.z.read(f"{self.root}/metadata.json"))
        self.K = [np.array(k, float) for k in self.meta["intrinsics"]]
        self.c2w = load_c2ws(self.z.read(f"{self.root}/calibrate.pkl"))
        self.ncam = len(self.K)
        self.steps, self.ts = [], []
        have = set(names)
        for c in range(self.ncam):
            r = self.meta["recording"][str(c)]
            ks = sorted(int(k) for k in r if f"{self.root}/depth/{c}/{k}.npy" in have and f"{self.root}/color/{c}/{k}.png" in have)
            self.steps.append(np.array(ks))
            self.ts.append(np.array([r[str(k)] for k in ks]))
        self.t0 = min(t[0] for t in self.ts)

    def depth(self, c, k):
        return np.load(io.BytesIO(self.z.read(f"{self.root}/depth/{c}/{k}.npy")))

    def color(self, c, k):
        b = np.frombuffer(self.z.read(f"{self.root}/color/{c}/{k}.png"), np.uint8)
        return cv2.imdecode(b, cv2.IMREAD_COLOR)  # BGR as recorded


def backproject(depth_m, K, c2w, stride=1):
    H, W = depth_m.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_m[::stride, ::stride]
    ok = z > 0
    x = (u[ok] - K[0, 2]) / K[0, 0] * z[ok]
    y = (v[ok] - K[1, 2]) / K[1, 1] * z[ok]
    P = np.stack([x, y, z[ok]], 1)
    return P @ c2w[:3, :3].T + c2w[:3, 3], ok
