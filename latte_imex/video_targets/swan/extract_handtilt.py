"""Supplementary swan extraction with the hand-fitted liquid ellipses spliced into the tilt phase.

The shared extractor (video_extract.extract) propagates the ellipse of the first good frames backwards through the tilt
phase (quality 0, swan t < 7.8 s); for this clip that ellipse covers the far inner wall and part of the pitcher, so the
white maps of 0-7.8 s are offset from the stage-script positions (which use hand_ellipses.json).  This script runs the
same pipeline (run_all: extract -> observables -> preview -> QA movie) with track_ellipses wrapped so that frames with
t <= t_until (8.6 s) use the hand ellipses (linear in time between the keys, quality set to 1 = partial fit).  Nothing
else changes; video_extract.py is not modified.

    cd /home/user/zfChen && python latte_imex/video_targets/swan/extract_handtilt.py
      -> latte_imex/video_targets/swan/handtilt/{frames.npz, observables.json, preview.png, extraction.mp4}
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/user/zfChen")
from latte_imex import video_extract as VX  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
H = json.load(open(os.path.join(HERE, "hand_ellipses.json")))
KEYS = sorted(float(k) for k in H["ell"])
T_UNTIL = float(H["t_until"])
_orig = VX.track_ellipses


def hand_p(t):
    return np.array([np.interp(t, KEYS, [H["ell"][f"{k:.2f}"][j] for k in KEYS]) for j in range(5)])


def track_with_hand(frames, fps, ref_index=None, seed=0, log=None):
    tr = _orig(frames, fps, ref_index=ref_index, seed=seed, log=log)
    t = tr["t"]
    m = t <= T_UNTIL + 1e-9
    tr["p"] = tr["p"].copy()
    for i in np.nonzero(m)[0]:
        tr["p"][i] = hand_p(t[i])
    tr["quality"] = np.where(m, 1, tr["quality"])
    # blend into the track over 0.3 s after t_until (no jump)
    i1 = int(np.nonzero(m)[0][-1])
    for i in range(i1 + 1, min(i1 + 6, len(t))):
        w = (t[i] - t[i1]) / (6 / fps)
        tr["p"][i] = (1 - w) * tr["p"][i1] + w * tr["p"][i]
    if log:
        log(f"hand ellipses spliced for t <= {T_UNTIL} s ({int(m.sum())} frames, quality set to 1)")
    return tr


if __name__ == "__main__":
    VX.track_ellipses = track_with_hand
    out = os.path.join(HERE, "handtilt")
    VX.run_all("swan", stages=os.path.join(HERE, "stages.json"), out_dir=out)
