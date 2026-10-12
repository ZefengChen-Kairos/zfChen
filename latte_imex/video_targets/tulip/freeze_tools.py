"""Freeze-frame handling for stage scripts (the reference clips insert still frames while the instructor's overlay is
shown; during a freeze little or no physical time passes).

A stage flagged "freeze": true with "physical_s" (physical seconds hidden behind the freeze) is kept in the clip-time
script so that sim time = clip time.  For a physically timed run:

    from latte_imex.video_targets.tulip import freeze_tools as FT      # or run this file
    phys = FT.drop_freezes("latte_imex/stage_scripts/tulip.json")      # freeze stages shortened to physical_s (or removed)
    t_clip = FT.phys_to_clip(t_phys, script)                           # compare a physical-time sim with frames.npz
    t_phys = FT.clip_to_phys(t_clip, script)

Times in the returned script are "physical clip time": equal to clip time before the first freeze, shifted earlier by
the removed freeze time afterwards.  `detect_freezes(name)` lists runs of duplicated frames in a reference clip.

    python latte_imex/video_targets/tulip/freeze_tools.py      # writes latte_imex/stage_scripts/tulip_physical.json
"""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _load(script):
    return copy.deepcopy(json.load(open(script)) if isinstance(script, str) else script)


def _cuts(d):
    """[(t0_clip, t1_clip, removed_s)] for every freeze stage, in order."""
    out = []
    for s in d["stages"]:
        if s.get("freeze"):
            keep = float(s.get("physical_s", 0.0))
            out.append((float(s["t0"]), float(s["t1"]), max(float(s["t1"]) - float(s["t0"]) - keep, 0.0)))
    return out


def clip_to_phys(t, script):
    """Clip time -> physical clip time (a time inside a freeze maps into its kept physical_s part)."""
    d = _load(script)
    shift = 0.0
    for t0, t1, rem in _cuts(d):
        if t >= t1:
            shift += rem
        elif t > t0:
            keep = (t1 - t0) - rem
            return t0 - shift + keep * (t - t0) / (t1 - t0)
    return t - shift


def phys_to_clip(tp, script):
    """Physical clip time -> clip time (inverse of clip_to_phys; a time inside a kept freeze part maps into the freeze)."""
    d = _load(script)
    shift = 0.0
    for t0, t1, rem in _cuts(d):
        keep = (t1 - t0) - rem
        if tp >= t0 - shift + keep:
            shift += rem
        elif tp > t0 - shift:
            return t0 + (tp - (t0 - shift)) * (t1 - t0) / max(keep, 1e-9)
    return tp + shift


def drop_freezes(script, out=None):
    """Return (and optionally write) the script with every freeze stage shortened to its physical_s (removed if 0) and
    all later times shifted; foot_measurements_M keys are remapped too.  The result has no "freeze" stages."""
    d = _load(script)
    st = []
    for s in d["stages"]:
        s2 = dict(s)
        if s.get("freeze"):
            keep = float(s.get("physical_s", 0.0))
            if keep <= 1e-6:
                continue
            s2["t0"] = round(clip_to_phys(float(s["t0"]), d), 4)
            s2["t1"] = round(s2["t0"] + keep, 4)
            s2.pop("freeze", None)
            s2["note"] = f"(freeze {s['t0']}-{s['t1']} s clip time shortened to its physical {keep} s) " + s.get("note", "")
        else:
            s2["t0"] = round(clip_to_phys(float(s["t0"]), d), 4)
            s2["t1"] = round(clip_to_phys(float(s["t1"]), d), 4)
            s2["t_clip"] = [s["t0"], s["t1"]]
        st.append(s2)
    for a, b in zip(st[:-1], st[1:]):          # close rounding gaps
        if 0 < b["t0"] - a["t1"] < 0.02:
            b["t0"] = a["t1"]
    d["stages"] = st
    if "foot_measurements_M" in d:
        d["foot_measurements_M"] = {f"{clip_to_phys(float(k), _load(script)):.2f}": v
                                    for k, v in d["foot_measurements_M"].items()}
    d["time_base"] = ("physical clip time: clip time with the freeze-frames removed (freezes shortened to physical_s); "
                      "compare with frames.npz at freeze_tools.phys_to_clip(t)")
    d["freezes_removed"] = d.pop("freezes", None)
    if out:
        with open(out, "w") as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
    return d


def detect_freezes(name, thr=0.5, min_len=4):
    """Runs of (near-)duplicated frames in a reference clip: [(t_first_copy, t_last_copy, n_copies)]."""
    sys.path.insert(0, ROOT)
    from latte_imex import video_extract as VX
    F, fps = VX.read_video(VX.clip_path(name))
    d = np.array([np.abs(F[i].astype(np.int16) - F[i - 1].astype(np.int16)).mean() for i in range(1, len(F))])
    out, i = [], 0
    while i < len(d):
        if d[i] < thr:
            j = i
            while j < len(d) and d[j] < thr:
                j += 1
            if j - i >= min_len:
                out.append((round((i + 1) / fps, 3), round(j / fps, 3), j - i))
            i = j
        else:
            i += 1
    return out


if __name__ == "__main__":
    src = os.path.join(ROOT, "latte_imex", "stage_scripts", "tulip.json")
    dst = os.path.join(ROOT, "latte_imex", "stage_scripts", "tulip_physical.json")
    d = drop_freezes(src, dst)
    print("wrote", dst, "stages:", len(d["stages"]), "end", d["stages"][-1]["t1"])
