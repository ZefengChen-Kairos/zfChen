# Three-RealSense rig: reading, static fusion, object reconstruction

Recordings: release `raw-pours-v1` (one zip per take: `color/<cam>/<step>.png`, `depth/<cam>/<step>.npy` uint16 mm,
aligned to colour, `metadata.json` intrinsics + per-frame timestamps, `calibrate.pkl` camera-to-board 4x4; the pkl is
read with a restricted unpickler, `safepkl.py`).

    python -I activity.py   <take.zip> <out>                     # frame-difference activity per camera + contact sheet
    python -I static_fuse.py <take.zip> <out> [n] [t1 t2]        # median depth over a still window -> board-frame clouds, table plane
    python -I reg_check.py  <out>_cloud.npz <png>                # table-texture overlay / phase correlation between cameras
    python -I cup_measure.py <cloud.npz> <out> <x> <y> [r]       # rim circle + radius profile around a seed
    python -I cylmerge.py   <out> <w1_cloud.npz> <w2_cloud.npz> ...  # merge a turned pitcher: axis fit, unwrap, yaw by correlation
    python -I pitcher_view.py <merged.npz> <out>                 # views + body radius profile + protrusions (handle / spout)

Pours (trimmed data on branch `pours-trimmed`): `run_ba.py <take>` (AprilTag 36h11 on the pitcher, tag edge 18.2 mm
estimated from depth; joint fit of tag layout on the wall + pitcher pose per sample) -> `pitcher_model.py` (tilt,
downhill azimuth, shape in the pitcher frame) -> `TIP=x,y,z tip_track.py` (spout tip trajectory, overlay) -> `cup_track.py`
(liquid surface below the tip from fused depth). Paths inside the scripts point at the session scratch; adjust.

Results of the first takes are in `raw_data/rig/results/`.
