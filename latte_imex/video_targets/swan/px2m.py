"""Image px at clip time t -> M frame [cup diameters], with the ellipse used by make_tiles (hand-fitted liquid ellipse
for t <= 8.6 s, extractor track afterwards).   python px2m.py t px py [t px py ...]"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tiles as T  # noqa: E402


def px2m(t, px, py):
    i = int(round(t * 15))
    p, q = T.ell_of(i)
    xm, ym = T.img_to_m(p, px, py, 0.0)
    return float(xm), float(ym), q


if __name__ == "__main__":
    v = [float(a) for a in sys.argv[1:]]
    for t, px, py in zip(v[::3], v[1::3], v[2::3]):
        xm, ym, q = px2m(t, px, py)
        print(f"t={t:5.2f} q{q} px=({px:.0f},{py:.0f}) -> M ({xm:+.3f}, {ym:+.3f})")
