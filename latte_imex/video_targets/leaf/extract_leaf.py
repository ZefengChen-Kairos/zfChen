"""Leaf extraction = the shared video_extract pipeline + a hand-fitted liquid ellipse for the tilt phase.

video_extract.track_ellipses gives quality 0 for t < 4.67 s in this clip (cup tilted toward the pitcher) and fills
those frames with the rim-attached propagated ellipse, which here lies 35-45 px too high (it covers the far inner wall,
see tiles/ellc_*.png).  This wrapper replaces the track for t < T_TILT_END by the hand-fitted liquid-surface ellipses
stored in stage_scripts/leaf.json ("tilt_ellipses_px", linear interpolation) -- the same geometry the stage positions
were read with -- and then runs video_extract.run_all unchanged.  Quality flags stay 0 for those frames
(approximate geometry).  video_extract.py itself is not modified.

    cd /home/user/zfChen && python latte_imex/video_targets/leaf/extract_leaf.py [--plain] [--out DIR] [--obs-only]

--plain runs the unpatched pipeline (for comparison).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import video_extract as VX  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "..", "stage_scripts", "leaf.json")
T_TILT_END = 4.65


def tilt_keys(script=SCRIPT):
    return np.array(json.load(open(script))["tilt_ellipses_px"]["keys"], float)


def patch_tracker(script=SCRIPT, log=print):
    K = tilt_keys(script)
    orig = VX.track_ellipses

    def track(frames, fps, ref_index=None, seed=0, log=None):
        tr = orig(frames, fps, ref_index=ref_index, seed=seed, log=log)
        sel = tr["t"] < T_TILT_END
        for i in np.nonzero(sel)[0]:
            tr["p"][i] = [np.interp(tr["t"][i], K[:, 0], K[:, j]) for j in range(1, 6)]
        if log:
            log(f"leaf tilt fix: {int(sel.sum())} frames (t < {T_TILT_END} s) use the hand-fitted liquid ellipse")
        return tr

    VX.track_ellipses = track
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--obs-only", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    if not a.plain:
        patch_tracker()
    VX.run_all("leaf", stages=SCRIPT, out_dir=a.out, obs_only=a.obs_only, video=not a.no_video)


if __name__ == "__main__":
    main()
