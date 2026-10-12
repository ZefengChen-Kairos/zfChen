"""Writes the swan stage script (M frame, clip time) read off the reference clip:

    python latte_imex/video_targets/swan/make_stage_script.py
      -> latte_imex/stage_scripts/swan.json            (clip time; the deliverable)
         latte_imex/video_targets/swan/stages.json     (same file; picked up by `python -m latte_imex.video_extract swan`)
         latte_imex/video_targets/swan/stages_realtime.json   (the two video freezes cut out: real pour time)

All positions were read by eye (stream foot = lower end of the stream column where it meets the surface) on native-
resolution crops with a 5/10 px grid (tiles/fz_*.png, foot_*.png, neck_*.png, wig_*.png), mapped to the M frame with
px2m.py (video_extract.image_to_cup with rot = 0, (x, y)_M = (x, -y)_T) on the liquid-surface ellipse of make_tiles.ell_of:
hand-fitted fixed-shape ellipses for t <= 8.6 s (hand_fit.py -> hand_ellipses.json; the extractor's quality-0 track in the
tilt phase covers the outer porcelain wall) and the extractor's quality-2 track after 8.6 s.  Cross-checked on rectified
M-frame views (tiles/rb_*.png, the stream column projects from the foot toward the lower-left there).
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

FREEZES = [  # [t0, t1) clip time [s] of held (duplicated) frames; real time does not advance there
    [0.0, 0.333, "frames 0-5 are one still image (frame 0 with the instructor's B/C/D overlay); kept as the pre-roll stand-in"],
    [3.867, 5.0, "frames 58-75 duplicate frame 58 (instructor's dashed guide line); frame 76 is the next real frame"],
    [6.133, 7.267, "frames 92-109 duplicate frame 92 (instructor's solid + dashed guide lines); frame 110 is the next real frame"],
]
SHIFT = 17 / 15.0      # each mid-clip freeze holds 17 extra frames = 1.133 s

FOOT_M = {  # clip time -> stream foot (M frame, cup diameters); see the module docstring
    "0.00": [-0.071, -0.143], "0.40": [-0.043, -0.143], "0.60": [-0.053, -0.135], "0.67": [-0.028, -0.117],
    "0.80": [-0.018, -0.109], "1.00": [0.015, -0.101], "1.20": [0.037, -0.089], "1.33": [0.022, -0.085],
    "1.40": [0.024, -0.098], "1.60": [-0.041, -0.103], "1.67": [-0.030, -0.077], "1.80": [-0.040, -0.094],
    "2.00": [-0.057, -0.065], "2.20": [-0.054, -0.080], "2.40": [-0.125, -0.026], "2.60": [-0.101, -0.038],
    "2.80": [-0.086, -0.043], "3.00": [-0.076, -0.049], "3.20": [-0.070, -0.060], "3.40": [-0.081, -0.063],
    "3.60": [-0.074, -0.036], "3.87": [-0.051, -0.033], "5.13": [-0.067, -0.037], "5.33": [-0.076, -0.085],
    "5.60": [-0.103, -0.074], "5.87": [-0.139, -0.074], "6.13": [-0.165, -0.094], "7.40": [-0.222, -0.113],
    "7.67": [-0.257, -0.127], "8.00": [-0.283, -0.156], "8.33": [-0.295, -0.171], "8.60": [-0.303, -0.218],
    "8.87": [-0.311, -0.251], "9.13": [-0.309, -0.278], "9.40": [-0.289, -0.256], "9.67": [-0.134, -0.195],
    "9.93": [-0.048, -0.154], "10.20": [0.017, -0.099], "10.47": [0.070, -0.070], "10.73": [0.137, -0.043],
    "11.00": [0.178, -0.024], "11.20": [0.215, -0.030], "12.40": [0.186, -0.030], "12.80": [0.137, -0.069],
    "13.00": [0.120, -0.077], "13.33": [0.091, -0.088], "13.60": [0.029, -0.192], "14.00": [-0.042, -0.271],
    "14.33": [-0.072, -0.376], "14.67": [-0.050, -0.421], "15.00": [-0.049, -0.437], "15.33": [-0.051, -0.435],
    "15.67": [-0.054, -0.440], "16.00": [-0.054, -0.431], "16.33": [-0.027, -0.390], "16.60": [0.022, -0.330],
    "16.80": [0.088, -0.328], "16.93": [0.127, -0.319]}

STREAM_WIDTH_PX = {  # horizontal width of the stream 12-24 px above the foot (camera px; ~34 px/cm on the surface)
    "0.40": 10, "0.67": 18, "1.00": 22, "2.00": 33, "3.60": 27, "5.33": 31, "5.87": 30, "7.40": 34, "7.67": 31,
    "8.00": 28, "8.60": 30, "9.13": 17, "9.67": 16, "9.93": 14, "10.20": 13, "10.47": 11, "10.73": 11, "11.00": 10,
    "12.40": 14, "12.80": 18, "13.33": 19, "13.60": 20, "14.00": 20, "14.33": 22, "14.67": 22, "16.33": 20,
    "16.60": 22, "16.80": 9, "16.93": 3}

W135 = 135.0   # wiggle direction in the M frame [deg]: across the pattern axis B->D (~45 deg), see "method"


def st(t0, t1, kind, p0, p1, z0, z1, f0, f1, wiggle, conf, note, bezier=None):
    d = {"t0": t0, "t1": t1, "kind": kind, "p0": p0, "p1": p1, "z0": z0, "z1": z1, "flow0": f0, "flow1": f1,
         "wiggle": wiggle, "confidence": conf, "note": note}
    if bezier is not None:
        d["bezier"] = bezier
    return d


STAGES = [
    st(0.0, 0.333, "low_pour", [-0.06, -0.145], [-0.06, -0.145], 0.02, 0.02, 0.25, 0.25, None, 0.25,
       "在B点注入流量 (subtitle 0-0.9 s). VIDEO FREEZE: frames 0-5 are one still (frame 0 with the instructor's B/C/D "
       "overlay); the stream already lands at the B dot, so the real pour onset is <= 0 s (first-white delays are lower "
       "bounds) and this stage stands in for the unknown pre-roll. Cup strongly tilted toward the pitcher (near rim hides "
       "~1/3 of the surface). Overlay dots (hand-fit tilt ellipse, +-0.1 D): B (-0.07,-0.18), C (+0.07,-0.07), D side "
       "(+0.19,+0.07): the B->D axis runs at ~45 deg in this M frame. Stream thin (10 px at 0.4 s)."),
    st(0.333, 1.0, "low_pour", [-0.06, -0.145], [0.01, -0.10], 0.02, 0.02, 0.25, 0.55, None, 0.25,
       "cup being righted while pouring: the stream stays nearly fixed in the image, so on the surface the foot moves "
       "toward the centre (0.40 (-0.04,-0.14), 0.67 (-0.03,-0.12), 0.80 (-0.02,-0.11), 1.00 (+0.015,-0.10)); flow "
       "ramps up (stream 10 -> 18 -> 22 px at 0.4/0.67/1.0 s); spout ~2 cm (stream ~60 px long). Positions +-0.08 D "
       "(liquid ellipse size fixed by assumption, only the far contact line visible)."),
    st(1.0, 2.2, "wiggle", [0.01, -0.10], [-0.05, -0.075], 0.02, 0.02, 0.55, 1.0,
       {"amp": 0.03, "amp1": 0.04, "freq": 2.9, "axis": W135}, 0.3,
       "开始摆动 (subtitle 1.0-2.5 s): lateral wiggle from ~1.0 s, stream image-x peaks 1.10 1.47 1.80 s (2.8-2.9 Hz), "
       "image-x half-amplitude 7 px (= 0.03 D along x_M, 0.04 D along the 135 deg wiggle direction); flow up to the "
       "full stream (22 -> 33 px by 2.0 s). FIRST WHITE: a pale curl at the foot at 1.13 s, a white crescent at 1.27-1.33 s "
       "(tiles/fz_f.png); video_obs.md says 1.7 s. Mean foot drifts from (+0.02,-0.095) (1.0-1.4 s) to (-0.05,-0.08) "
       "(1.8-2.2 s); single readings scatter +-0.03 with the wiggle phase."),
    st(2.2, 3.0, "wiggle", [-0.05, -0.075], [-0.085, -0.045], 0.02, 0.02, 1.0, 1.0,
       {"amp": 0.033, "freq": 3.2, "axis": W135}, 0.35,
       "纹路形成半圆形后 (subtitle 3.0-4.0 s): steady wiggle, image-x peaks 2.07 2.37 2.70 3.00 s (3.2 Hz), image-x "
       "half-amplitude 6 px; arcs convex toward the D side stack into a semicircle ~0.5 D wide; thick stream (~30 px), "
       "spout ~2 cm. Foot 2.4-3.4 s around (-0.09,-0.045)."),
    st(3.0, 3.867, "wiggle", [-0.085, -0.045], [-0.06, -0.035], 0.018, 0.018, 1.0, 1.0,
       {"amp": 0.033, "freq": 3.2, "axis": W135}, 0.35,
       "向前摆动两下 (subtitle 4.5-5.0 s, narration lags): two wiggles with a small forward drift toward D "
       "(3.4 (-0.08,-0.06) -> 3.87 (-0.05,-0.03), ~0.03 D); peaks 3.30 3.60 s; spout ~1.8 cm (stream ~55 px)."),
    st(3.867, 5.0, "wiggle", [-0.06, -0.035], [-0.06, -0.035], 0.018, 0.018, 0.0, 0.0, None, 1.0,
       "VIDEO FREEZE, NOT AN ACTION: clip frames 58-75 are one still (frame 58) with the instructor's dashed guide line; "
       "real time advances only one frame across it. Encoded as a zero-flow hold so that the converted moves stay "
       "aligned with clip time and with frames.npz; video_targets/swan/stages_realtime.json has it cut out (use that one "
       "for physically continuous moves)."),
    st(5.0, 6.133, "retreat", [-0.06, -0.035], [-0.165, -0.095], 0.018, 0.018, 1.0, 1.0,
       {"amp": 0.04, "freq": 3.1, "axis": W135}, 0.4,
       "稍微~倾斜一点 (subtitle 5.5-6.5 s: the cup is tilted slightly toward the pitcher again; liquid ellipse a grows "
       "130 -> 137 px). The retreat toward the pitcher (which sits at ~225-240 deg in this M frame) starts here at "
       "~0.11 D/s (5.13 (-0.07,-0.04), 5.33 (-0.08,-0.085), 5.60 (-0.10,-0.07), 5.87 (-0.14,-0.07), 6.13 (-0.165,-0.095)); "
       "the narration '往后快速后退' (7.0-8.0 s) lags. Wiggle continues: image-x peaks 5.40 5.70 6.00 s (3.1 Hz), "
       "image-x half-amplitude 7.5 px; stream ~31 px, spout ~1.8 cm. A new white blob forms beside the foot (5.4-6.0 s)."),
    st(6.133, 7.267, "retreat", [-0.165, -0.095], [-0.165, -0.095], 0.018, 0.018, 0.0, 0.0, None, 1.0,
       "VIDEO FREEZE, NOT AN ACTION: clip frames 92-109 are one still (frame 92) with two instructor guide lines (at "
       "~33 and ~55 deg in M through the pattern); real time advances only one frame across it. Zero-flow hold as above; "
       "cut out in stages_realtime.json."),
    st(7.267, 9.133, "retreat", [-0.165, -0.095], [-0.31, -0.275], 0.015, 0.012, 1.0, 0.6,
       {"amp": 0.04, "amp1": 0.025, "freq": 3.2, "axis": W135}, 0.5,
       "往后快速后退 / 退到杯子边缘 (subtitles 7.0-8.0 / 8.5-9.5 s): retreat with wiggle to the B rim at ~0.1 D/s "
       "(7.40 (-0.22,-0.11), 7.67 (-0.26,-0.13), 8.00 (-0.28,-0.16), 8.33 (-0.30,-0.17), 8.60 (-0.30,-0.22), "
       "8.87 (-0.31,-0.25), 9.13 (-0.31,-0.28)); stream image-x peaks 7.53 7.80 8.07 s (3.2 Hz; FFT 3.1-4.4 Hz noisy), "
       "image-x half-amplitude 8 px; one chevron per wiggle cycle, chevrons perpendicular to the band (band axis ~45 deg "
       "-> wiggle along ~135 deg); thick stream (28-34 px) until 8.6 s, 17 px at 9.13 s; spout low (stream 45 px at "
       "7.67 s, 25-35 px at 8.6-9.1 s: ~1-1.5 cm). The chevron band (rosetta body) spans (-0.31,-0.28) -> (0,0). The "
       "path bends (first along -x, then along -y: bezier); the stage starts from rest at the freeze-hold point, so the "
       "scripted foot lags the measured one by ~0.05 D during the first ~0.4 s.",
       bezier=[[-0.27, -0.12], [-0.31, -0.18]]),
    st(9.133, 9.8, "lift", [-0.31, -0.275], [-0.10, -0.175], 0.012, 0.025, 0.45, 0.3, None, 0.5,
       "小流量拉高 (subtitle 10.0-10.5 s): the stream dwells at the rim tip until ~9.4 s (9.40 (-0.29,-0.26)), the spout "
       "is raised (stream 35 px at 9.13 s, 52 px at 9.67 s) and the flow cut (17 -> 15 px), then the foot sweeps fast "
       "(~0.5 D/s, 15 fps: 9.47-9.73 s) back into the band: 9.67 (-0.13,-0.195)."),
    st(9.8, 10.467, "high_cut", [-0.10, -0.175], [0.07, -0.07], 0.025, 0.032, 0.25, 0.18, None, 0.55,
       "thin stream (13-11 px) runs along the axis of the chevron band toward its D-side end: 9.93 (-0.05,-0.15), "
       "10.20 (+0.02,-0.10), 10.47 (+0.07,-0.07); leaves a brown cut line (the rosetta's midrib) and drags the chevrons "
       "into V's; spout ~2.5-3 cm (stream 52-74 px). 2nd high_cut stage below continues the same line."),
    st(10.467, 11.267, "high_cut", [0.07, -0.07], [0.215, -0.03], 0.032, 0.038, 0.18, 0.1, None, 0.55,
       "沿着纹路的边缘收尾 (subtitle 11.0-12.5 s): the cut continues past the band into the semicircle arcs, slowing "
       "(10.73 (+0.14,-0.04), 11.00 (+0.18,-0.02), 11.20 (+0.215,-0.03)); stream 11 -> 10 px, ~85 px long at 11.0 s "
       "(spout ~3.5-4 cm); last stream frame 11.20 s, gone at 11.27 s."),
    st(11.267, 12.333, "reposition", [0.215, -0.03], [0.19, -0.035], 0.038, 0.02, 0.0, 0.0, None, 0.5,
       "no stream 11.27-12.27 s: the spout hovers above the end of the cut (spout lip at camera px (220-230, 262-268)) "
       "and is lowered (再压低拉花缸, subtitle 13.0-13.5 s, lags); the stream restarts at 12.33 s at the same place."),
    st(12.333, 13.2, "low_pour", [0.19, -0.035], [0.105, -0.08], 0.015, 0.017, 0.22, 0.4, None, 0.45,
       "low pour restarted at the end of the cut (12.40 (+0.19,-0.03), 12.80 (+0.14,-0.07), 13.00 (+0.12,-0.08)); stream "
       "14 -> 18 px, ~40-45 px long (spout ~1.5 cm); the foot creeps toward the pitcher; a white patch spreads at the foot "
       "(base of the neck). No wiggle."),
    st(13.2, 13.8, "retreat", [0.105, -0.08], [0.0, -0.235], 0.017, 0.017, 0.42, 0.42, None, 0.5,
       "快速的往后退到B点附近 (subtitle 14.0-16.0 s): fast retreat toward the B rim WITHOUT wiggle (15 fps tiles "
       "neck_13.3-14.7_15fps.png: no lateral oscillation): 13.33 (+0.09,-0.09), 13.60 (+0.03,-0.19); ~0.45 D/s; the "
       "stream (19-20 px) leaves a smooth white band ~0.05 D wide behind it = the swan's neck."),
    st(13.8, 14.667, "retreat", [0.0, -0.235], [-0.05, -0.42], 0.017, 0.01, 0.42, 0.45, None, 0.45,
       "neck continues as an S toward the B rim: 14.00 (-0.04,-0.27), 14.33 (-0.07,-0.38), 14.67 (-0.05,-0.42) "
       "(bezier); the spout comes down to the rim (stream only ~25 px long at 14.67 s, ~1 cm).",
       bezier=[[-0.06, -0.29], [-0.09, -0.40]]),
    st(14.667, 16.0, "low_pour", [-0.05, -0.42], [-0.054, -0.432], 0.01, 0.01, 0.45, 0.55, None, 0.4,
       "head: spout held just above the surface at the B rim (r = 0.44 D; stream 20-25 px long), a round white blob "
       "(~0.1 D) grows beside the foot on the +x side (15.0-16.0 s: (-0.049,-0.437), (-0.051,-0.435), (-0.054,-0.44), "
       "(-0.054,-0.431)); stream ~22 px (lip ~28 px)."),
    st(16.0, 16.6, "push", [-0.054, -0.432], [0.022, -0.33], 0.012, 0.015, 0.7, 0.8, None, 0.35,
       "大流量点出大白心 (subtitle 16.5-19.0 s): flow increased, the foot moves toward the centre (16.33 (-0.03,-0.39), "
       "16.60 (+0.02,-0.33)) and the head inflates into a heart ~0.15 D (16.33-16.6 s). Width-based flow estimate ~0.5 "
       "(stream 20-22 px at ~1-1.5 cm), narration says big flow: 0.7-0.8 used, +-50 %."),
    st(16.6, 16.967, "high_cut", [0.022, -0.33], [0.13, -0.318], 0.015, 0.022, 0.3, 0.03, None, 0.5,
       "拉高收尾: the stream thins fast (22 -> 9 -> 3 px at 16.6 / 16.8 / 16.93 s) while it is pulled out of the head "
       "toward +x_M (16.80 (+0.09,-0.33), 16.93 (+0.13,-0.32)): the head becomes a notched heart with the beak/tail on "
       "the +x side; stream ~45 px long (spout ~1.8 cm, not 7-9 cm). Last stream frame 16.93 s; a falling drop at 17.0 s."),
    st(16.967, 20.5, "stop", [0.13, -0.318], [0.13, -0.318], 0.09, 0.09, 0.0, 0.0, None, 0.9,
       "pitcher withdrawn (17.0-17.2 s); cup shown, pattern settles; reference frame (targets/swan.npz) at 20.5 s; "
       "outro subtitle at 22.5 s."),
]

META = {
    "name": "swan", "frame": "M", "Q_ref": 1.5e-05, "t_clip0": 0.0,
    "clip": "reference_swan.mp4 (480x640, 15 fps, 23.0 s; not in refclips/, read from targets.REF); reference frame "
            "20.5 s (targets.FRAMES swan: rot 0, flip False)",
    "time_base": "CLIP time. The clip contains three FREEZE FRAMES (see 'freezes'): the mid-clip ones (3.867-5.0 s and "
                 "6.133-7.267 s, 17 held frames = 1.133 s each) are encoded as zero-flow hold stages (confidence 1.0); "
                 "real pour time = clip time - 1.133 s (5.0 <= t <= 6.133) - 2.267 s (t >= 7.267). "
                 "video_targets/swan/stages_realtime.json is the same script with both holds cut out.",
    "freezes": FREEZES,
    "orientation_warning": (
        "In this clip's M frame (targets.FRAMES swan rot = 0, i.e. not rotated into the canonical orientation of the "
        "other five patterns) the pitcher is on the -y side: the stream column comes from azimuth ~225-240 deg, the "
        "pattern axis B -> D runs at ~45 deg (B (-0.07,-0.18) -> D (+0.19,+0.07)), the pours start near (-0.06,-0.14) "
        "and retreat toward (-0.31,-0.28), and the head is poured at the -y rim. pitcher.Barista uses yaw = -pi/2 "
        "(spout pointing -y, pitcher handle at +y), the opposite side. The Barista still lands the stream on the "
        "scripted points (it places the tip from the measured jet offset), but the jet's horizontal momentum and the "
        "pitcher body are on the wrong side. To match the video pass yaw = barista_yaw_rad to BaristaScript "
        "(spout pointing at +45 deg, toward D)."),
    "barista_yaw_rad": 0.785,
    "pattern_axis_M_deg": 45.0,
    "landmarks_M": {
        "B": [-0.07, -0.18], "C": [0.07, -0.07], "D_side_line_point": [0.19, 0.07],
        "note": "instructor's overlay in the frozen frame 0 (cup tilted; hand-fit ellipse, +-0.1 D: C should be the cup "
                "centre, so the tilt-phase positions may be offset by up to ~0.1 D). Final swan (20.5 s, extractor "
                "track): wing/body chevron band from (-0.42,-0.2) to (0.05,0.0) with a brown midrib, broad arcs of the "
                "first semicircle on the +y side, neck S from (+0.1,-0.1) to the head at (-0.03,-0.38), head = notched "
                "heart at the -y rim."},
    "method": (
        "Stream foot read by eye on native-resolution camera crops (2-3.5x zoom, 5/10 px grid; tiles/fz_*.png, "
        "foot_*.png) and mapped with video_extract.image_to_cup (rot 0; (x,y)_M = (x,-y)_T). Liquid-surface ellipse: "
        "for t <= 8.6 s hand-fitted fixed-shape ellipses (b/a 0.752, phi -3.032 rad from the level phase; centre and "
        "size from the far and side contact lines; size fixed by assumption at t < 2.7 s; hand_ellipses.json, "
        "tiles/hand_ellipses.png) because the extractor's quality-0 track (propagated, t < 7.8 s) includes the far inner "
        "wall and part of the pitcher; from 8.6 s the extractor track (quality 2). The cup handle stays at azimuth ~125 "
        "deg in M (+-5 deg) from 0 to 17 s (tiles/wall_rect.png): no cup rotation about its axis, so the camera-aligned "
        "M frame is also the cup frame. Wiggle: lateral position of the stream's edge 12-25 px above the foot at 15 fps "
        "(wiggle_probe.py; image-x half-amplitude 6-8 px) -> amplitude along the wiggle direction 135 deg in M (across "
        "the B->D axis; the chevrons lie perpendicular to the band) = image-x amplitude / (2a cos 45 deg), +-30 %. "
        "Heights from the image length of the stream (camera elevation ~49 deg from b/a, ~34 px/cm on the surface, "
        "~23 px/cm vertical), +-50 %. Flows from the stream width above the foot (STREAM_WIDTH_PX) relative to the "
        "wiggle-phase stream (~30 px = flow 1.0), Q ~ w^2 sqrt(v0^2 + 2 g z), +-50 %. Subtitles (narration, lag the "
        "actions by ~0.5-2 s) read from the clip: 0.0-0.9 在B点注入流量, 1.0-2.5 开始摆动, 3.0-4.0 纹路形成半圆形后, "
        "4.5-5.0 向前摆动两下, 5.5-6.5 稍微~倾斜一点, 7.0-8.0 往后快速后退, 8.5-9.5 退到杯子边缘, 10.0-10.5 小流量拉高, "
        "11.0-12.5 沿着纹路的边缘收尾, 13.0-13.5 再压低拉花缸, 14.0-16.0 快速的往后退到B点附近, 16.5-19.0 "
        "大流量点出大白心拉高收尾, 19.5-21.5 一个姿态优美的天鹅就完成了, 22.5 outro."),
    "differences_from_video_obs": (
        "video_obs.md (section 6) says first white 1.7 s (here 1.13-1.33 s), a 'second chevron band (the wing)' laid by a "
        "wiggling retreat at 13.9-16.4 s (here: the 13.2-14.67 s retreat has no wiggle and draws the smooth neck band; the "
        "only chevrons are those of the first retreat 5.0-9.1 s), and the head blob + neck at 16.4-17.9 s (here: head "
        "poured at the B rim 14.67-16.6 s, final pull 16.6-16.97 s, stream gone at 17.0 s). It also misses the two mid-clip "
        "video freezes (3.867-5.0 s, 6.133-7.267 s)."),
    "first_white_by_eye": {"t": [1.13, 1.33], "note": "pale curl at the foot at 1.13 s, white crescent 1.27-1.33 s (tiles/fz_f.png); real onset of the pour <= 0 s (clip starts mid-pour on a frozen frame), so this is an upper bound of the delay. Auto (frames.npz): 0.42 s, wrong (the tilt-phase ellipse includes pitcher/wall); auto with hand tilt ellipses (handtilt/): 1.63 s (late, centroid biased by a porcelain/thumb patch)."},
    "old_solver_check": (
        "stages_to_moves -> BaristaScript -> old solver N=64 (old_solver_check.py; runs/stage_check/swan*/compare.json): "
        "runs without error in 25-38 s; simulated impact vs the 61 measured feet: median 0.015-0.019 D, p90 0.045 D, max "
        "0.09 D (2.4 s outlier at a wiggle extreme; 7.4-8.0 s lag after the freeze hold); wiggle 2.7-3.2 Hz, half-amplitude "
        "0.025-0.035 D along 135 deg. With the default yaw -pi/2 the pitcher body hangs over the cup on the wrong side: "
        "cup-collision lifts up to 36 mm, tip 1-1.5 cm higher than scripted in the head phase, and flow surges (2.2x the "
        "wanted flow at 2.2-3.0 s and 7.3-9.1 s, 0.27x at 5.0-6.1 s). With yaw = barista_yaw_rad (0.785) flows follow the "
        "script within ~20 % (lift <= 13 mm, heights as scripted): use that yaw for this pattern. Final IoU with "
        "frames.npz[-1]: 0.35-0.39 with rows flipped (sim_to_t) vs 0.13-0.31 unflipped."),
    "foot_measurements_M": FOOT_M,
    "foot_measurements_note": "keys are CLIP times; +-0.03 D after 8.6 s, +-0.05 D 2.7-8.6 s, +-0.08 D before 2.7 s",
    "stream_width_px": STREAM_WIDTH_PX,
    "tilt_phase_liquid_ellipses": None,   # filled from hand_ellipses.json
}


def realtime(script):
    """Cut the two mid-clip freeze holds out (t_real = t_clip - k * SHIFT)."""
    d = copy.deepcopy(script)
    out = []
    for s in d["stages"]:
        if s["confidence"] == 1.0 and s["note"].startswith("VIDEO FREEZE"):
            continue
        k = sum(1 for f in FREEZES[1:] if s["t0"] >= f[1] - 1e-6)
        s["t0"] = round(s["t0"] - k * SHIFT, 3)
        s["t1"] = round(s["t1"] - k * SHIFT, 3)
        out.append(s)
    for a, b in zip(out[:-1], out[1:]):        # make the stages contiguous again across the removed holds
        if abs(b["t0"] - a["t1"]) < 0.01:
            b["t0"] = a["t1"]
    d["stages"] = out
    d["time_base"] = ("REAL pour time (NOT clip time): stage_scripts/swan.json with the 3.867-5.0 s and 6.133-7.267 s "
                      "freezes cut out; t_real = t_clip (t <= 3.867), t_clip - 1.133 (5.0 <= t <= 6.133), t_clip - 2.267 "
                      "(t >= 7.267). The frame-0 freeze is kept as the 0.33 s pre-roll stand-in. Use for physically "
                      "continuous moves; the extractor's observables are in clip time.")
    d["foot_measurements_note"] = "keys are CLIP times"
    return d


if __name__ == "__main__":
    hand = json.load(open(os.path.join(HERE, "hand_ellipses.json")))
    meta = dict(META)
    meta["tilt_phase_liquid_ellipses"] = dict(
        note="(cx, cy, a, b, phi) [px, rad], targets.py axis convention, linear in time between keys, used for the "
             "positions at t <= 8.6 s; from 8.6 s frames.npz track_ell (quality 2). frames.npz itself uses the "
             "propagated track for t < 7.8 s (quality 0), which is offset by up to ~0.1-0.25 D in the tilt phase.",
        **hand["ell"])
    script = dict(meta, stages=STAGES)
    # sanity: contiguous, ordered
    for a, b in zip(STAGES[:-1], STAGES[1:]):
        assert abs(a["t1"] - b["t0"]) < 1e-9, (a["t1"], b["t0"])
        assert a["p1"] == b["p0"] or a["kind"] == "stop", (a["t1"], a["p1"], b["p0"])
    txt = json.dumps(script, indent=1, ensure_ascii=False)
    for p in (os.path.join(ROOT, "stage_scripts", "swan.json"), os.path.join(HERE, "stages.json")):
        open(p, "w").write(txt)
        print("wrote", p)
    rt = realtime(script)
    p = os.path.join(HERE, "stages_realtime.json")
    open(p, "w").write(json.dumps(rt, indent=1, ensure_ascii=False))
    print("wrote", p, "stages", len(rt["stages"]), "T", rt["stages"][-1]["t1"])
