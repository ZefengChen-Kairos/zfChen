"""Whole-video supervision targets: every rectified top-view frame of the pattern phase (fuse_occ.py: 3-camera fusion of
the liquid surface on the multi-view consistent plane of cup_register.py, with occluder masks in each camera image)
resampled onto the disc grid of the targets (tulip_feas.Target convention: col = board x, row = board y, extent
+-EXT disc radii), on the simulation clock (t = 0 at the first frame of the pattern phase, as side_by_side.py /
real_panel.py).  Disc frame: the registered rigid-cup track of disc.json (one centre track and one length unit L_m
per pour, shared with the simulation's inlet and the stream-foot measurement).  V2_SURFACE_MODEL.md 10.5.

Per frame:
  img    RGB (uint8) of the surface, outside the disc 0
  L      LAB lightness (uint8)
  valid  observation mask (fixed by the observation, never by the model; observe()): some camera has a usable view
         (fuse_occ: outside every occluder -- near depth, the tracked pitcher, the milk stream -- grown by 8 px),
         inside the liquid disc (r <= 0.96 R(t)/L_m: the meniscus at the wall is excluded), not a specular glare blob
         (small near-saturated components), not within FOOT_R disc radii of the spout tip (stream / impact zone)
  nview  cameras usable at the pixel (0-3; single-view pixels cannot be cross-checked)
  t      frame time [s] on the simulation clock
  w      whiteness (uint8, 255 = foam): seg_sup.py -- a classifier on crema-relative colour / texture features plus the
         seg_gmm posterior, followed by a per-pixel HMM over the whole video, trained leave-one-pour-out on the
         finished-pattern target masks (targets/<p>.npz: Otsu on one hand-picked, evenly lit frame after the pour,
         registered onto the video by target_register.py): the W of pour p never sees p's own mask
Two passes: frames for every pour (frames()), then the whiteness (white()), which reads every pour's frames.  Writes
raw_data/rig/video_obs/<p>.npz (+ <p>_sheet.jpg).  Run where the pipeline's pours/ data are.
    python3 -m latte_imex.rig.video_obs [--frames | --white] [pattern ...]
(<p>_flow.npz, video_flow.py, was computed on the cup_register frames before fuse_occ; weight 0 in the fit.)"""
import json
import os
import sys

import cv2
import numpy as np

SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pours"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VID = os.path.join(REPO, "raw_data", "rig", "video_obs")
OUT, EXT = 128, 1.05
R_IN = 0.96              # disc radius used (the bright meniscus ring at the wall is excluded)
FOOT_R = 0.12            # [disc radii] exclusion around the spout tip while pouring
POURS = ("tulip", "leaf", "heart", "push_heart", "swan")
_S = np.linspace(-EXT, EXT, OUT)
_U, _V = np.meshgrid(_S, _S)
_RR = np.hypot(_U, _V)


def observe(im, seen, D, fi):
    """Observation mask of the rectified frame im (BGR uint8, OUT x OUT) of frame fi whose pixels some camera saw (seen);
    returns (im with the outside of the disc set to 0, valid)."""
    reg = D[fi].get("source") == "cup_register.py"
    # registered track (cup_register.py): one length unit L_m for the whole pour (the sim's), the liquid edge at R(t);
    # old free fits: each frame scaled by its own radius
    R = D[fi]["L_m"] if reg else D[fi]["disc_R"]
    edge = D[fi]["disc_R"] / R if reg else 1.0
    inside = _RR <= R_IN * edge
    valid = seen & inside & (reg or bool(D[fi].get("fit_ok", True)))
    sat = ((im.min(2) > 235) & inside).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(sat, 8)
    glare = np.isin(lab, [i for i in range(1, nl) if st[i, cv2.CC_STAT_AREA] < 0.006 * inside.sum()])
    glare = cv2.dilate(glare.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    valid &= ~glare
    tip = D[fi].get("tip_rel_norm")
    if tip is not None:
        valid &= np.hypot(_U - tip[0], _V - tip[1]) > FOOT_R
    im = im.copy()
    im[_RR > 1.0] = 0
    return im, valid


def frames(p):
    """Rectified frames, lightness, observation masks and view counts of pour p (fuse_occ.py) -> video_obs/<p>.npz
    (whiteness added by white())."""
    from . import fuse_occ
    write_frames(p, fuse_occ.build(p))


def write_frames(p, F):
    """F: fuse_occ.build(p) (t, img RGB, L, valid, nview, frame_idx, pitcher_depth_conflict, pose_rejected_t, source)."""
    os.makedirs(VID, exist_ok=True)
    n = len(F["t"])
    np.savez_compressed(f"{VID}/{p}.npz", t=F["t"], img=F["img"], L=F["L"], valid=F["valid"],
                        w=np.zeros(F["L"].shape, np.uint8), extent=EXT, R_in=R_IN, foot_r=FOOT_R, nview=F["nview"],
                        frame_idx=F["frame_idx"], pitcher_depth_conflict=F["pitcher_depth_conflict"],
                        pose_rejected_t=F["pose_rejected_t"], source=F["source"])
    vf = F["valid"].reshape(n, -1).sum(1) / (_RR <= R_IN).sum()
    print(f"{p}: {n} frames, t 0..{F['t'][-1]:.2f} s, valid fraction median {np.median(vf):.2f} (min {vf.min():.2f})")


def hand_mask(p, n=OUT):
    """Finished-pattern target mask (targets/<p>.npz) at the video resolution: white, valid."""
    T = np.load(f"{REPO}/raw_data/rig/targets/{p}.npz")
    k = T["white"].shape[0] // n
    w = T["white"].reshape(n, k, n, k).mean((1, 3)) > 0.5
    v = T["valid"].reshape(n, k, n, k).mean((1, 3)) > 0.5
    return w, v


def white(p):
    """Supervised leave-one-pour-out whiteness of pour p (frames() must have run for every pour)."""
    from . import seg_sup
    store_white(p, seg_sup.segment_lopo(p, {q: hand_mask(q) for q in POURS}))


def store_white(p, W):
    """Write the whiteness W (n, OUT, OUT) of pour p into video_obs/<p>.npz, with the contact sheet."""
    fn = f"{VID}/{p}.npz"
    Z = dict(np.load(fn))
    V = Z["valid"].astype(bool)
    W = np.where(V, W, 0.0)
    n = len(W)
    prior = np.array([W[k][V[k]].mean() if V[k].any() else 0.0 for k in range(n)])
    Z.update(w=(W * 255).round().astype(np.uint8), white_frac=prior,
             source=f"{p}: fuse_occ.py frames and disc.json; w = seg_sup (leave-one-pour-out) whiteness * 255")
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
    cv2.imwrite(f"{VID}/{p}_sheet.jpg", np.hstack(rows))
    hw, hv = hand_mask(p)
    m = hv & V[-1]
    a, b = (W[-1] > 0.5) & m, hw & m
    print(f"{p}: white fraction first/mid/last {prior[0]:.2f}/{prior[n // 2]:.2f}/{prior[-1]:.2f}, last frame vs target mask "
          f"IoU {(a & b).sum() / max((a | b).sum(), 1):.3f} (area ratio {a.sum() / max(b.sum(), 1):.2f})")


if __name__ == "__main__":
    args = sys.argv[1:]
    only = args.pop(0) if args and args[0] in ("--frames", "--white") else None
    pats = args or list(POURS)
    if only != "--white":
        for p in POURS:
            if p in pats or not os.path.exists(f"{VID}/{p}.npz"):
                frames(p)
    if only != "--frames":
        for p in pats:
            white(p)
