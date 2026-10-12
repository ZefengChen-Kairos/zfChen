"""Wiggle of the stream from the 15 fps frames: x of the stream's edge on one image row (the stream is cream, the
crema next to it dark), tracked frame by frame; detrended; dominant frequency and half-amplitude (px and cup
diameters, with the liquid ellipse of make_tiles).   python wiggle_probe.py t0 t1 row x_lo x_hi [edge=right|left]"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tiles as T  # noqa: E402
from latte_imex import video_extract as VX  # noqa: E402


def edge_track(t0, t1, row, x_lo, x_hi, edge="right", rows=3):
    _, frames = T.data()
    out = []
    for i in range(int(round(t0 * 15)), int(round(t1 * 15)) + 1):
        rgb = frames[i, row - rows:row + rows + 1, x_lo:x_hi].astype(float) / 255
        hsv = VX.rgb_to_hsv(rgb)
        V = hsv[..., 2].mean(0)
        x = np.nan
        if edge == "right":
            # stream's right edge: last bright (V > 0.6) pixel followed by >= 10 px of dark crema (V < 0.45)
            dark = V < 0.45
            for k in range(len(V) - 14):
                if V[k] >= 0.55 > V[k + 1] and dark[k + 3:k + 14].mean() >= 0.8 and V[max(k - 6, 0):k].max() > 0.7:
                    x = k + (V[k] - 0.55) / max(V[k] - V[k + 1], 1e-6)
                    break
        else:
            # stream's left edge: first pixel of the bright core (V > 0.88) left of the right edge
            br = np.nonzero(V > 0.88)[0]
            x = br.min() if len(br) else np.nan
        out.append((i / 15, x_lo + x))
    return np.array(out)


def spectrum(t, x):
    m = np.isfinite(x)
    t, x = t[m], x[m]
    d = x - np.polyval(np.polyfit(t, x, 2), t)
    n = 2048
    f = np.fft.rfftfreq(n, 1 / 15.0)
    P = np.abs(np.fft.rfft(d * np.hanning(len(d)), n)) ** 2
    sel = f > 1.0
    k = np.argmax(P[sel])
    return float(f[sel][k]), float(np.sqrt(2) * d.std()), d


if __name__ == "__main__":
    t0, t1 = float(sys.argv[1]), float(sys.argv[2])
    row, x_lo, x_hi = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    edge = sys.argv[6] if len(sys.argv) > 6 else "right"
    tr = edge_track(t0, t1, row, x_lo, x_hi, edge)
    f, amp, d = spectrum(tr[:, 0], tr[:, 1])
    p, q = T.ell_of(int(round(0.5 * (t0 + t1) * 15)))
    print("t, x:", " ".join(f"{a:.2f}:{b:.0f}" for a, b in tr))
    print("detrended:", " ".join(f"{v:+.0f}" for v in d))
    print(f"dominant {f:.2f} Hz, half-amplitude {amp:.1f} px = {amp / (2 * p[2]):.3f} D (lateral, image-x; a = {p[2]:.0f} px)")
