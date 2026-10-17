"""Playable preview of a trimmed take: the three cameras side by side (top: colour, bottom: depth, turbo 0.2-1.0 m,
black = no depth), resampled to a uniform FPS timeline by nearest timestamp per camera, real-time speed, H.264.
    python3 latte_imex/rig/make_mp4.py <take dir> <out.mp4> [fps]"""
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from trimrec import Take

TW, TH = 424, 240                       # tile size (half of 848 x 480)
D0, D1 = 200, 1000                      # depth colour range [mm]


def depth_tile(d):
    x = np.clip((d.astype(np.float32) - D0) / (D1 - D0), 0, 1)
    im = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    im[d == 0] = 0
    return cv2.resize(im, (TW, TH), interpolation=cv2.INTER_NEAREST)


def label(im, s, y=18, scale=0.5):
    cv2.putText(im, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def main(d, out, fps=15.0):
    tk = Take(d)
    name = os.path.basename(os.path.normpath(d))
    cut = tk.meta["cut"]
    t0 = max(ts[0] for ts in tk.ts)
    t1 = min(ts[-1] for ts in tk.ts)
    grid = np.arange(t0, t1, 1.0 / fps)
    W, H = TW * tk.ncam, TH * 2
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    cache = {}
    for t in grid:
        frame = np.zeros((H, W, 3), np.uint8)
        for c in range(tk.ncam):
            j = int(np.argmin(np.abs(tk.ts[c] - t)))
            k = tk.steps[c][j]
            if (c, k) not in cache:
                col = cv2.resize(tk.color(c, k), (TW, TH), interpolation=cv2.INTER_AREA)
                label(col, f"cam{c}")
                cache[(c, k)] = (col, depth_tile(tk.depth(c, k)))
            col, dep = cache[(c, k)]
            frame[:TH, c * TW:(c + 1) * TW] = col
            frame[TH:, c * TW:(c + 1) * TW] = dep
        label(frame, f"{name}  t = {t - cut['t_first_frame_unix']:6.2f} s (recording)  "
                     f"{t - cut['start_unix']:5.2f} s (since cut)", y=TH - 8, scale=0.55)
        ff.stdin.write(frame.tobytes())
        if len(cache) > 64:
            cache.clear()
    ff.stdin.close()
    ff.wait()
    print(out, len(grid), "frames", f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 15.0)
