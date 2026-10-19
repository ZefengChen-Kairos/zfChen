"""Whole-video supervision targets: every rectified top-view frame of the pattern phase (cup_register.py: 3-camera
fused image of the liquid surface on the multi-view consistent plane + 'seen' = some camera saw the point unoccluded)
resampled onto the disc grid of the targets (tulip_feas.Target convention: col = board x, row = board y, extent
+-EXT disc radii), on the simulation clock (t = 0 at the first frame of the pattern phase, as side_by_side.py /
real_panel.py).  Disc frame: the registered rigid-cup track of disc.json (one centre track and one length unit L_m
per pour, shared with the simulation's inlet and the stream-foot measurement).

Per frame:
  img    RGB (uint8) of the surface, outside the disc 0
  L      LAB lightness (uint8)
  valid  observation mask (fixed by the observation, never by the model): seen by a camera, inside the liquid disc
         (r <= 0.96 R(t)/L_m: the meniscus at the wall is excluded), not a specular glare blob (small near-saturated
         components), not within FOOT_R disc radii of the spout tip (stream / impact zone: low confidence)
  t      frame time [s] on the simulation clock
  w      whiteness (uint8, 255 = foam): seg_gmm.py -- per-frame crema / foam Gaussian mixture with a per-pixel HMM over
         the whole video, fitted by EM; the class geometry is learnt from the hand-checked final masks of the
         calibration pours only (tulip, leaf; targets/<p>.npz registered onto the video by target_register.py)
Two passes: frames for every pour (frames()), then the whiteness (white()), which reads the calibration pours' last
frames.  Writes raw_data/rig/video_obs/<p>.npz (+ <p>_sheet.jpg).  Run where the pipeline's top/ frames are.
    python3 -m latte_imex.rig.video_obs [pattern ...]"""
import json
import os
import sys

import cv2
import numpy as np

SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT, EXT = 128, 1.05
R_IN = 0.96              # disc radius used (the bright meniscus ring at the wall is excluded)
FOOT_R = 0.12            # [disc radii] exclusion around the spout tip while pouring
CAL = ("tulip", "leaf")  # calibration pours: their hand masks teach seg_gmm the foam / crema geometry


def frame(p, fi, D, U, V, RR):
    Z = np.load(f"{SCR}/{p}/top/{fi:04d}.npz")
    img, seen, c0, PX, HALF = Z["img"], Z["seen"], Z["c0"], float(Z["px"]), float(Z["half"])
    reg = D[fi].get("source") == "cup_register.py"
    # registered track (cup_register.py): one length unit L_m for the whole pour (the sim's), the liquid edge at R(t);
    # old free fits: each frame scaled by its own radius
    cen, R = np.array(D[fi]["disc_centre"]), (D[fi]["L_m"] if reg else D[fi]["disc_R"])
    edge = D[fi]["disc_R"] / R if reg else 1.0
    mx = ((cen[0] + U * R - c0[0] + HALF) / PX - 0.5).astype(np.float32)
    my = ((cen[1] + V * R - c0[1] + HALF) / PX - 0.5).astype(np.float32)
    im = cv2.remap(img, mx, my, cv2.INTER_AREA if PX < 0 else cv2.INTER_LINEAR)
    sv = cv2.remap(seen.astype(np.uint8), mx, my, cv2.INTER_NEAREST).astype(bool)
    inside = RR <= R_IN * edge
    valid = sv & inside & (reg or bool(D[fi].get("fit_ok", True)))
    sat = ((im.min(2) > 235) & inside).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(sat, 8)
    glare = np.isin(lab, [i for i in range(1, nl) if st[i, cv2.CC_STAT_AREA] < 0.006 * inside.sum()])
    glare = cv2.dilate(glare.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    valid &= ~glare
    tip = D[fi].get("tip_rel_norm")
    if tip is not None:
        valid &= np.hypot(U - tip[0], V - tip[1]) > FOOT_R
    im[RR > 1.0] = 0
    L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0]
    return im[..., ::-1].copy(), L, valid


def frames(p):
    """Rectified frames, lightness and observation masks of pour p -> video_obs/<p>.npz (whiteness added by white())."""
    D = json.load(open(f"{SCR}/{p}/disc.json"))
    s = np.linspace(-EXT, EXT, OUT)
    U, V = np.meshgrid(s, s)
    RR = np.hypot(U, V)
    t0 = D[0]["t"]
    imgs, Ls, vals, ts = [], [], [], []
    for fi in range(len(D)):
        if not os.path.exists(f"{SCR}/{p}/top/{fi:04d}.npz"):
            continue
        im, L, v = frame(p, fi, D, U, V, RR)
        imgs.append(im); Ls.append(L); vals.append(v); ts.append(D[fi]["t"] - t0)
    os.makedirs(f"{REPO}/raw_data/rig/video_obs", exist_ok=True)
    np.savez_compressed(f"{REPO}/raw_data/rig/video_obs/{p}.npz", t=np.array(ts), img=np.array(imgs), L=np.array(Ls),
                        valid=np.array(vals), w=np.zeros(np.array(Ls).shape, np.uint8), extent=EXT, R_in=R_IN, foot_r=FOOT_R,
                        source=f"{p}: cup_register.py frames and disc.json (frames only)")
    vf = np.array(vals).reshape(len(ts), -1).sum(1) / (RR <= R_IN).sum()
    print(f"{p}: {len(ts)} frames, t 0..{ts[-1]:.2f} s, valid fraction median {np.median(vf):.2f} (min {vf.min():.2f})")


def hand_mask(p, n=OUT):
    """Hand-checked final pattern (targets/<p>.npz) at the video resolution: white, valid."""
    T = np.load(f"{REPO}/raw_data/rig/targets/{p}.npz")
    k = T["white"].shape[0] // n
    w = T["white"].reshape(n, k, n, k).mean((1, 3)) > 0.5
    v = T["valid"].reshape(n, k, n, k).mean((1, 3)) > 0.5
    return w, v


def white(p):
    """seg_gmm whiteness of pour p (frames() must have run for p and the calibration pours)."""
    from . import seg_gmm
    fn = f"{REPO}/raw_data/rig/video_obs/{p}.npz"
    Z = dict(np.load(fn))
    W = seg_gmm.segment(p, Z, {q: hand_mask(q) for q in CAL})
    V = Z["valid"].astype(bool)
    W = np.where(V, W, 0.0)
    n = len(W)
    prior = np.array([W[k][V[k]].mean() if V[k].any() else 0.0 for k in range(n)])
    Z.update(w=(W * 255).round().astype(np.uint8), white_frac=prior,
             source=f"{p}: cup_register.py frames and disc.json; w = seg_gmm whiteness * 255")
    np.savez_compressed(fn, **Z)
    rows = []
    for k in np.linspace(0, n - 1, 16).astype(int):
        wv = (np.dstack([W[k]] * 3) * 255).astype(np.uint8)
        wv[~V[k]] = (60, 0, 60)
        a = cv2.resize(Z["img"][k][..., ::-1], (128, 128), interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(wv, (128, 128), interpolation=cv2.INTER_NEAREST)
        col = np.vstack([a, b])
        cv2.putText(col, f"t={Z['t'][k]:.1f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        rows.append(col)
    cv2.imwrite(f"{REPO}/raw_data/rig/video_obs/{p}_sheet.jpg", np.hstack(rows))
    hw, hv = hand_mask(p)
    m = hv & V[-1]
    a, b = (W[-1] > 0.5) & m, hw & m
    print(f"{p}: white fraction first/mid/last {prior[0]:.2f}/{prior[n // 2]:.2f}/{prior[-1]:.2f}, last frame vs hand mask "
          f"IoU {(a & b).sum() / max((a | b).sum(), 1):.3f} (area ratio {a.sum() / max(b.sum(), 1):.2f})")


if __name__ == "__main__":
    pats = sys.argv[1:] or ["heart", "push_heart", "leaf", "swan", "tulip"]
    for p in sorted(set(pats) | set(CAL), key=lambda q: q not in CAL):
        if p in pats or not os.path.exists(f"{REPO}/raw_data/rig/video_obs/{p}.npz"):
            frames(p)
    for p in pats:
        white(p)
