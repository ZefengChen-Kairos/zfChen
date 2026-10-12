"""Side-by-side pattern videos (physical time axis, 15 fps):

  [reference clip (camera)] [reference, rectified top view] [model, initial-surface-anchored RGB (appearance.py)]
  [model composition C (diagnostic, grey) + stream-foot trajectory (V0.5 style: past path, current foot)]

All top-view panels show the PHYSICAL top view (the M frame, x right, y up; a T-frame array a_T is displayed as is,
because (x, y)_M = (x, -y)_T and T is the mirror image of the camera's view); the stream foot is the observed inlet of
the physical-time script.  Freeze frames of the clip are skipped (physical time only).

    from latte_imex import make_videos as MV
    MV.pattern_video(model, params, "tulip", k, "runs/videos/tulip.mp4", N=128)
"""
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from . import appearance as AP
from . import calib as CB
from . import video_extract as VE
from .v3min_adapter import Inlet

PANEL = 300


def _panel(rgb01, size=PANEL):
    img = Image.fromarray((np.clip(rgb01, 0, 1) * 255).astype(np.uint8))
    return img.resize((size, size), Image.BILINEAR)


def _m_to_px(p_M, size=PANEL):
    """M-frame point [D] -> panel pixel of the physical top view (x right, y_M up)."""
    return ((p_M[0] + 0.5) * size, (0.5 - p_M[1]) * size)


def clip_frames_phys(clip_name):
    """15 fps clip frames on the physical time axis: (frame indices, physical times, ellipses)."""
    clip = CB.load_clip(clip_name)
    res = clip["res"]
    cuts, pre = clip["cuts"], CB.PREROLL.get(clip_name, 0.0)
    tt, ell = res["track_t"], res["track_ell"]
    idx, tp, E, last = [], [], [], -1e9
    for k, t in enumerate(tt):
        if any(t0 + 1e-6 < t < t1 - 1e-6 for t0, t1, _ in cuts):
            continue
        p = CB.clip_to_phys(float(t), cuts) + pre
        if p <= last + 1e-6:
            continue
        idx.append(k); tp.append(p); E.append(ell[k]); last = p
    return np.array(idx), np.array(tp), np.array(E)


def pattern_video(model, params, name, k, out_path, N=128, title="", fps=15, max_frames=None, anc=None):
    clip = CB.load_clip(name)
    res = clip["res"]
    M = clip["M"]
    idx, tp, E = clip_frames_phys(name)
    if max_frames:
        idx, tp, E = idx[:max_frames], tp[:max_frames], E[:max_frames]
    r = model.run(clip["script"], params, N, tp)
    wT = CB.observe_model(r["c"], M)
    if anc is None:
        anc = AP.anchors(clip)
    frames, _ = VE.read_video(VE.clip_path(name))
    inl = Inlet(clip["script"], Q_scale=float(params.get("Q_scale", 1.0)))
    trail = [np.array(inl.at(t)[:2]) for t in np.arange(0, tp[-1] + 1e-9, 0.02)]
    trail_t = np.arange(0, tp[-1] + 1e-9, 0.02)
    trail_q = np.array([inl.at(t)[3] for t in trail_t])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    W, H = 4 * PANEL, PANEL + 40
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20", out_path],
                          stdin=subprocess.PIPE)
    rot, flip = float(res["rot"]), bool(res["flip"])
    disk = res["disk"].astype(bool)
    for j, (fi, t, e) in enumerate(zip(idx, tp, E)):
        cam = frames[fi]
        h0, w0 = cam.shape[:2]
        s = PANEL / max(h0, w0)
        cam_img = Image.fromarray(cam).resize((int(w0 * s), int(h0 * s)), Image.BILINEAR)
        p1 = Image.new("RGB", (PANEL, PANEL), (255, 255, 255))
        p1.paste(cam_img, ((PANEL - cam_img.width) // 2, (PANEL - cam_img.height) // 2))
        rect = VE.rectify(cam.astype(np.float64) / 255.0, e, M, rot, flip)
        rect[~disk] = 1.0
        p2 = _panel(rect)
        p3 = _panel(AP.render(wT[j], anc, k))
        g = np.clip(wT[j], 0, 1)
        diag = np.stack([g * 0.9 + 0.05] * 3, -1)
        diag[~disk] = 1.0
        p4 = _panel(diag)
        d4 = ImageDraw.Draw(p4)
        past = [_m_to_px(q) for q, tq, qq in zip(trail, trail_t, trail_q) if tq <= t]
        for a_, b_, tq, qq in zip(past[:-1], past[1:], trail_t, trail_q):
            d4.line([a_, b_], fill=(220, 60, 40) if qq > 1e-9 else (90, 140, 230), width=2)
        x_, y_, z_, q_ = inl.at(t)
        cx, cy = _m_to_px(np.array([x_, y_]))
        rr = 3 + 5 * min(q_ / 15e-6, 2.0)
        d4.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 200, 0), width=2)
        d3 = ImageDraw.Draw(p3)
        d3.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=(255, 200, 0), width=1)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        for i, p in enumerate((p1, p2, p3, p4)):
            canvas.paste(p, (i * PANEL, 40))
        dc = ImageDraw.Draw(canvas)
        dc.text((6, 4), f"{name}  t = {t:5.2f} s (physical)   Q = {q_ * 1e6:4.1f} mL/s   z = {z_ * 100:3.1f} cm   {title}",
                fill=(0, 0, 0))
        for i, lab in enumerate(("reference clip", "reference (top view)", f"model RGB (k={k:.2f})",
                                 "model C + stream-foot path")):
            dc.text((i * PANEL + 6, 22), lab, fill=(60, 60, 60))
        ff.stdin.write(np.asarray(canvas, np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    return dict(path=out_path, n_frames=len(idx), T=float(tp[-1]), ledger=r.get("ledger"))
