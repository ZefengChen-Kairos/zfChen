# Pose track v2: can the pitcher's time-series pose be fully recovered?

## Answer

**Not completely. The slow, large motion is recovered well. Per-instant consistency is about 0.2-0.3° in tilt and about 1 mm at the spout tip while the jug moves slowly (< 10 °/s, ~85% of the time), but only ~1° (p90 0.9°) at 10-60 °/s and ~2.5° / 5 mm (p90) during the fastest tilt-backs (> 60 °/s), where the two cameras' clocks disagree. The absolute orientation against gravity and the board is limited to roughly 1-3° by the rig calibration, and nothing faster than ~2-5 Hz can be recovered from these recordings.**

> **Independent re-audit (pose_v2/audit/, 2026-10-21).** The delivered files reproduce every validation number of this README, and v2 is better than the current track. Some comparisons below overstate the margin, and the README has been corrected where marked *[audit]*:
> * **Held-out camera 2.** About half of the IoU gain over the current track comes from v2's fitted CAD placement, a constant the current track could also apply. Against the current track with its own best constant placement (fitted on camera 0/1 depth, `audit/out/cur_place.json`), v2 still wins in every pour, by a smaller margin:
>   * IoU: +0.006..+0.016 (median paired difference), better in 57-74% of frames;
>   * depth residual: better in tulip, heart and swan (4.5 vs 7.1, 3.1 vs 4.6, 3.7 vs 9.1 mm), slightly better in push_heart, tied in leaf.
>   * v2 is the steadiest in every pour. Camera-2 residuals change much less from frame to frame (depth 0.7-1.0 vs 0.9-3.1 mm, silhouette 1.0-1.5 vs 2.0-2.8 px), a test that does not depend on calibration.
> * **Leave-one-camera-out.** The 0.7-2.3 px below include the camera-1 extrinsic correction, which was fitted with both cameras. The current chain's 4.6-17.5 px have no such correction. Like-for-like (raw calibration), v2 predicts the held-out camera's corners to 4.4-9.3 px (median), which is still better than the current chain in 4 of 5 pours (heart camera 1: 4.8 vs 4.6 px). The shared correction is consistent across pours to within 0.4-1.2° (per-pour residual rotation; its own size is 2.1°).
> * **Fast motion and timing.** The two single-camera refits disagree more as the jug moves faster: tilt p90 0.5° below 10 °/s, 0.9° at 10-60 °/s, 2.4° above 60 °/s; tip p90 up to 5 mm. In the residual camera-1 clock offset between them (-7.5..+20 ms per pour), swan is the outlier at +20 ms. At swan's final tilt-back (~100 °/s), v2 follows camera 1 and the camera-0 frames lie 2.0-2.6° below it. The claim "±15 ms costs ≤0.1°" applies only to the slow phases.
> * **No over-smoothing.** v2 follows the per-sample converged two-camera and single-camera poses with slope 0.87-1.11 on the >2 Hz-removed tilt (1 = no flattening). Tilt-back minima are reproduced to within ±0.5° of the camera-0 samples. The current track's fast tilt fluctuations hardly track the per-sample poses (slope 0.06-0.63), so they are noise.
> * Format and consumers were checked: every key matches; `fuse_occ.pose_inliers` rejects none of the v2 samples; the production chain reproduces the downstream table exactly when the file is installed as is. The sim-clock origin (`cup_poses` t0) shifts by ≤3 ms. The spout azimuth reproduces as the downhill direction: 90.0° pooled in the output frame, 87.8-93.0° per pour. Two outright defects were fixed: `offsets_s` now stores the tied depth offsets (c1, 0, c1) instead of (c1, 0, 0), and the camera-0/1 spout-visibility statement was corrected.
> * The held-out camera-2 timing scan (rendering at ±60 ms) is insensitive (IoU changes < 0.01). Camera 2 therefore cannot validate the fast dynamics or the clock offsets.
> * The table-plane cross-check of the extrinsics is inconclusive: camera 0 does not see the plane that cameras 1 and 2 see.
> * Audit material, all in `audit/`:
>   * scripts in `work/`: `cam_eval.py` (camera 2, all frames, from the delivered npz), `cur_place.py`, `loco_check.py`, `smooth_check.py`, `xcam_speed.py`, `stream_audit.py`, `ds_check.py`, `inlier_check.py`, `fmt_check.py`, `frame_check.py`, `cam2_shift.py`, `overlay.py`;
>   * numbers in `out/`;
>   * overlays and tilt traces in `img/` (all viewed).

What v2 recovers, with the evidence (per-pour numbers in the tables below):

* **Continuous 6-DoF trajectory, no flips, no camera-switch jumps.**
  * Leave-one-camera-out: the trajectory fitted without camera 0 predicts camera 0's tag corners to 0.7-2.3 px (median), and without camera 1 predicts camera 1's to 0.9-1.8 px. The current chain manages 4.6-17.5 px. *[audit]* This is not like-for-like: v2's numbers use the camera-1 extrinsic correction fitted with both cameras. With the raw calibration, v2 predicts 4.4-9.3 px, which is still better in 4 of 5 pours (`audit/out/loco_check.json`).
  * Cross-camera bias: the cam-0-only and cam-1-only fits agree to 0.4-1.1 mm at the tip (|median vector|), with per-instant differences of 0.8-1.3 mm (p50) / 1.4-2.2 mm (p90), 0.17-0.33° (p50) / 0.4-0.66° (p90) in tilt, and 0.6-1.3° in total rotation. The current track's cross-camera tip bias is 13.6-16.5 mm and its tilt bias 0.4-2.5°.
* **Held-out camera 2 (never used in the fit) is reproduced better than with the current track:**
  * silhouette IoU is higher in 4 pours and equal in leaf (median 0.784-0.811 vs 0.744-0.789), and its p10 is higher in all 5;
  * the depth residual is lower in 4 pours and equal in leaf (tulip 7.8→4.5 mm, heart 5.1→3.1, push_heart 8.8→8.1, swan 12.0→3.7);
  * colour-edge offsets are the same or lower in 4 pours.
  * *[audit]* Against the current track with its own best constant CAD placement (camera 0/1 depth), the margins shrink:
    * IoU 0.784-0.811 vs 0.767-0.809 (paired +0.006..+0.016, better in 57-74% of frames);
    * depth residual 3.1-8.1 vs 4.6-9.1 mm, with leaf tied (6.1 vs 6.5 mm, v2 better in only 46% of frames).
    * About half of the IoU gain above is therefore placement, not trajectory (`audit/out/cam2_summary.json`).
* **The independent stream checks agree better with v2:**
  * The stream line traced up from the measured stream foot meets v2's spout tip at 2.1-6.0 mm (median) versus 4.5-6.6 mm for the current track (better in 4 of 5 pours). The sideways error is now -0.8..+2.0 mm. *[audit]* The production current chain (current poses plus `TIP_KONA`) gives 3.8-8.2 mm, so v2 is better in all 5 pours. The current track with its own constant placement gives 3.2-6.2 mm, so v2 is better in 4 pours, and push_heart is tied (6.0 vs 6.2 mm). There are only 6-23 frames per pour.
  * The landing point predicted by the production inlet chain lies 6.5-9.5 mm (p50) / 10.9-14.8 mm (p90) from the measured stream foot, versus 8.9-10.6 / 12.5-17.8 mm for the current track (better in all 5 pours).
* **Plausible dynamics.** Body angular acceleration is 160-320 °/s² (p95) and tip acceleration 0.4-0.9 m/s² (p95). The current production series reaches 850-1070 °/s² and 1.8-2.1 m/s². The current track's extra fast motion (1.2-2.3 mm rms of the landing point) does **not** correlate with the measured stream foot (r ≤ 0.3), so it is noise.

What cannot be recovered from these recordings, and why:

1. **Absolute orientation against the board / gravity: about 1-3° systematic.**
   * The three cameras' extrinsics (one calibrate.pkl for all pours) are mutually inconsistent. Camera 1 is off camera 0 by 2.1° (0.19, -1.77, 1.08)° plus (1.3, -1.6, -1.2) mm. This was fitted jointly from tags and depth: the tags and an independent depth-only test agree on the ~1.5° rotation about board y.
   * Camera 2 needs a constant 1.2-3.8° correction for either track (fitted on camera 2 alone, 1.3-3.4° for v2's camera-0 world), mostly about board y and z.
   * v2 is expressed in camera 0's world (camera 1 corrected into it). Which camera is right cannot be decided without recalibration. The liquid surface in the cup, used as a gravity reference, was too noisy to decide (1-6° per-frame spread).
   * The pouring flow responds at 28-57 %/° of tilt, so this is the dominant remaining downstream uncertainty.
2. **Bandwidth.**
   * Each camera delivers ~14 fps (40/80 ms gaps), the cameras are unsynchronised, and only 26-47% of instants have both cameras' tags.
   * Motion power stands above the noise floor only below ~1.5-2.5 Hz. The spline (50 ms knots, smoothness prior from the measured accelerations) reproduces up to ~5 Hz.
   * The measured stream foot itself moves only 1.3-3.1 mm rms faster than a 0.5 s running mean, including its own detection noise. Fast wiggle beyond that is not observable, but there is also no evidence that it exists.
3. **Spin about the jug axis.**
   * It comes from the tags only, since depth cannot see the spin of a near-cylinder. Cross-camera agreement in total rotation is 0.6-1.3° p50.
   * The spout is not used by the fit. *[audit]* It appears in camera 0/1 only as a small part of the silhouette, and its depth is masked by the stream capsule. Its azimuth in the tag frame comes from the downhill direction while pouring (liquid leaves at the lowest point of the lip): 89.5° pooled, 87.3-92.4° per pour, which gives about ±2 mm lateral uncertainty at the tip.
4. **Timing.**
   * The camera-1 clock offset is only weakly identified at these slow speeds (-14..+11 ms per pour). The ±15 ms this implies costs ≤0.3 mm / 0.1° at the median speeds. *[audit]* It costs much more during tilt-backs:
     * The single-camera refits leave a residual camera-1 lag of -7.5..+20 ms.
     * The cross-camera tilt disagreement has a p90 of 2.4° above 60 °/s.
     * At swan's final tilt-back, the camera-0 frames lie 2.0-2.6° below v2 (`audit/out/xcam_speed.json`).
   * Depth-vs-colour offsets are unidentifiable (they drifted to 188 ms when freed), so depth is tied to its camera's colour clock.
5. **Jug model.**
   * The tags lean 2.0° relative to the jug axis (consistent in all pours).
   * The CAD's 14-gon facet phase is wrong relative to the spout: the real flats carry the tags at 0° / -51.4° / 180°, but the spout sits at a facet vertex, not at a flat centre as the CAD assumes.
   * Camera 0 sees the surface 0.6-0.9 mm outside the CAD.
   * These are modelled or absorbed (lean, CAD placement), but they set the floor of the depth agreement.

The formal (noise-only) per-sample uncertainty of the fit is 0.07-0.11° in pouring tilt and spin. That is far below the systematic limits above, so treat it as a lower bound only.

**Downstream: production chain, v2 vs current.**
* Flow rms difference 2.1-3.3 mL/s on mean flows of 10-16 mL/s.
* Poured volume in the pattern phase -1.3 to -5.0 mL; leaf -11.6 mL, where v2's axis tilt is 1.1° lower throughout.
* Landing point difference 2.4-4.0 mm (p50) / 4.7-7.7 mm (p90).

The v2 spout tip in the CAD-aligned frame, (-0.2, 50.1, 88.3) mm, agrees with the production `TIP_KONA` (0, 50.2, 87.9) mm. The tip.npz tip (0, 53, 93) mm used by fuse_occ / stream_foot is 5 mm off.

## What was built (read-only on the repo)

1. **Tag corners.**
   * Re-detected with subpixel edge-line refinement (`work/detect_all.py`, `edgefit.py`; data in `data/tags_<p>.pkl`).
   * Same-frame two-tag rigid fits give 0.10-0.17 px rms. The integer corners in the current tags.pkl give 0.28-0.43 px (leaf even falls on a wrong branch).
2. **Tag model and step-1 layout.**
   * The CAD has a 14-gon at the tag height: circumradius 43.70 mm at CAD z 28.5, so apothem **a = 42.60 mm**, flats 19.45 mm wide, and the tags are 19.0 mm squares on the flats.
   * The flat-tag model fits much better than a cylinder of the same radius (0.28-0.41 vs 0.37-0.56 px). Tag 5 comes out two flats from tag 4 (-51.1 ± 0.2°) and tag 1 seven flats away (180.4-181.2°).
   * The per-sample bundle adjustment (analytic JAX Jacobians, Schur LM) converges in 10-25 iterations; the current chain's runs stopped at max_nfev=400.
   * The joint layout from samples joined across cameras within 12 ms is biased by the cross-camera sync and calibration conflicts: it differs from the camera-0-only layout by up to 4.5° in tag-4 roll and 3 mm in tag-5 height. Camera 0 alone gives a consistent layout (z5 2.5 ± 0.6 mm, relative roll 1.5 ± 0.25°). This motivated steps 3-4.
3. **Continuous-time trajectory** (`spline.py`, `ctfit.py`).
   * Cumulative cubic B-spline on SO(3) plus a cubic B-spline for the origin, with 50 ms knots. The knot spacing follows from the measured bandwidth (Lomb-Scargle of the per-sample track: power above the noise floor only below ~2-2.5 Hz; a cubic B-spline reproduces up to 1/(4Δ) = 5 Hz).
   * Evaluated at each frame's true timestamp; Jacobians verified against finite differences (≤1e-9).
   * Residuals:
     * tag corners of cameras 0/1 (Huber k = 1.5σ, σ = 0.18 px from the residual MAD of the tags-only fit);
     * CAD-to-depth point-to-plane of cameras 0/1 (Tukey 10→5 mm, σ 1.5 mm). Each frame is down-weighted to N_eff ≈ 113-140 effective points, so that a single frame's formal pouring-tilt precision equals the audit's measured depth floor (0.5°); hand, cup rim, stream and opening are masked with the audit's `gcm.observe`;
     * a smoothness prior on body angular acceleration and origin acceleration. σ_α = 94 °/s² and σ_a = 0.13 m/s² per component are the raw rms second differences of the converged two-camera samples (an upper bound, so the prior is conservative), never tuned on validation.
   * Optional term (d), a spin prior from the stream, was **not** used, so the stream check stays independent of the fit.
4. **Shared unknowns, fitted jointly over the five pours from cameras 0/1 tags and depth** (`fit_joint.py`):
   * tag layout (tags 1, 5, 0 and the roll of tag 4);
   * a common tag lean (-2.03°; per-pour tags-only fits gave -2.35..-2.50°);
   * a 6-DoF camera-1 extrinsic correction, applied to camera-1 tags and depth. It is decisively favoured by the fit's own likelihood: tag rms falls from 0.36 to 0.25 px for 6 parameters over ~11,000 residuals;
   * a CAD placement correction (axis tilt plus translation; spin fixed);
   * the camera-1 time offset, per pour.
   * Spout azimuth 89.49° (pooled downhill direction while pouring, cameras 0/1 only). CAD z0 = -29.0 mm.
5. **Outputs** in the CAD-aligned pitcher frame (z = jug axis to the rim, spout +y, origin on the axis at the tag-4 height): a 10 ms grid between the first and last observation.

## Disclosure: one validation-informed revision

The first joint fit placed the spout at 77.14° (three flats from tag 4). That assumed the CAD's "flats centred on the spout", and a flank-symmetry estimate seemed to support it (70-80°). Its validation showed a consistent ~10 mm sideways offset between the spout tip and the stream on the stream check and on camera 2. I then replaced it with the downhill-direction estimator, the principle the current chain also uses (tip_track), computed from cameras 0/1 only, and refitted.

The spout azimuth (one constant) is therefore not blind to the stream check and camera 2. Everything else was fixed before any validation number was seen: the prior weights, knot spacing, N_eff, the extrinsic and lean model choice (by the fit's likelihood), and the tying of the depth offsets.

The first variant's camera-2 IoU was similar (0.78-0.83); its depth residual was better in 4 pours, but its stream distance was 9.4-12.5 mm.

## Validation details (per pour; generated by `work/readme_tables.py` from `out/*.json`)
### Corner estimator (same-frame two-tag rigid fit, camera 0 tags 4+5, rms px)

| pour | integer (tags.pkl now) | aruco contour | edge-line subpixel (v2) |
|---|---|---|---|
| tulip | 0.34 | 0.14 | 0.13 |
| leaf | 1.11 | 0.17 | 0.10 |
| heart | 0.32 | 0.18 | 0.10 |
| push_heart | 0.28 | 0.15 | 0.17 |
| swan | 0.43 | 0.17 | 0.15 |

### Step 1: per-sample bundle adjustment, converged (corner rms px; layout tag 5 phi / z, tag 1 phi / z)

| pour | flat a=42.60 (12 ms sync) | flat, cam-0 only | flat, 5 ms sync | cylinder r=42.6 | cylinder r=43.7 | flat + free scale (tag mm) | iters / status (flat) |
|---|---|---|---|---|---|---|---|
| tulip | 0.30 [t5 -51.16 deg / 0.79 mm; t1 180.4 / -7.1] | 0.12 [t5 -50.92 deg / 3.09 mm] | 0.23 | 0.39 | 0.32 | 0.28 (19.18) | 13 / converged |
| leaf | 0.34 [t5 -51.32 deg / -0.05 mm; t1 180.8 / -7.0] | 0.10 [t5 -51.17 deg / 2.88 mm] | 0.25 | 0.43 | 0.36 | 0.30 (19.23) | 15 / converged |
| heart | 0.34 [t5 -51.32 deg / -0.99 mm; t1 180.4 / -7.8] | 0.11 [t5 -51.35 deg / 2.06 mm] | 0.20 | 0.44 | 0.37 | 0.33 (19.15) | 14 / converged |
| push_heart | 0.41 [t5 -51.44 deg / -0.48 mm; t1 179.7 / -6.9] | 0.14 [t5 -51.07 deg / 2.66 mm] | 0.34 | 0.56 | 0.45 | 0.35 (19.33) | 25 / converged |
| swan | 0.28 [t5 -51.29 deg / -0.19 mm; t1 181.2 / -5.3] | 0.15 [t5 -51.15 deg / 1.65 mm] | 0.16 | 0.37 | 0.30 | 0.24 (19.25) | 10 / converged |

### Continuous-time tags-only fits (joint over the five pours; tag rms px cam0 / cam1)

| model | tulip | leaf | heart | push_heart | swan | shared parameters |
|---|---|---|---|---|---|---|
| flat layout + c1 | 0.40/0.49 | 0.36/0.57 | 0.33/0.58 | 0.48/0.62 | 0.40/0.42 | lean 0.00 deg |
| + common lean | 0.23/0.42 | 0.18/0.53 | 0.17/0.51 | 0.29/0.65 | 0.20/0.43 | lean -2.38 deg |
| + camera-1 extrinsic | 0.19/0.26 | 0.17/0.32 | 0.17/0.26 | 0.25/0.43 | 0.20/0.27 | lean -2.05 deg, ext1 [-0.329, -1.19, 0.586, 3.29, -0.38, -4.09] |

### Final joint fit (v2): tags + depth + prior, cameras 0/1

layout (phi deg, z mm, psi deg, u mm, tilt_x deg, tilt_y deg): {4: [0.0, 0.0, -2.99, 0.0, -2.03, 0.0], 5: [-51.09, 2.28, 0.33, 0.0, -2.03, 0.0], 1: [181.98, -1.89, 0.63, 0.0, -2.03, 0.0], 0: [-125.26, -2.75, -0.0, 0.0, -2.03, 0.0], 'ext1': [0.185, -1.765, 1.076, 1.27, -1.63, -1.19], 'pl': [0.659, -0.316, 0.0, -1.14, -1.89, 1.01]}

| pour | c1 offset ms | tag rms px cam0 | tag rms px cam1 | depth med abs mm cam0 | cam1 | signed depth med mm cam0 | cam1 | N_eff |
|---|---|---|---|---|---|---|---|---|
| tulip | -2.2 | 0.20 | 0.30 | 0.70 | 0.29 | 0.64 | 0.21 | 125 |
| leaf | 8.5 | 0.18 | 0.42 | 0.95 | 0.26 | 0.86 | 0.17 | 126 |
| heart | -13.9 | 0.19 | 0.35 | 0.66 | 0.35 | 0.58 | 0.31 | 140 |
| push_heart | 9.4 | 0.25 | 0.52 | 0.64 | 0.27 | 0.57 | 0.19 | 120 |
| swan | 10.6 | 0.20 | 0.30 | 0.77 | 0.28 | 0.69 | 0.21 | 113 |

### Held-out camera 2 (never used in the fit): median [p10 for IoU | p90 otherwise] per pour

| pour | source | IoU | depth abs residual mm | silhouette offset px | colour-edge offset px | n frames |
|---|---|---|---|---|---|---|
| tulip | current | 0.789 [0.748] | 7.77 [10.06] | 5.5 [8.5] | 5.0 [6.0] | 190 |
| tulip | v2 | 0.807 [0.785] | 4.45 [8.59] | 5.5 [8.5] | 5.0 [7.0] | 190 |
| tulip | v2 tags only | 0.810 [0.786] | 6.96 [9.59] | 6.5 [9.5] | 5.0 [7.0] | 190 |
| tulip | tags only, no cam-1 corr. | 0.801 [0.774] | 4.38 [9.19] | 5.2 [7.5] | 5.0 [6.0] | 190 |
| leaf | current | 0.788 [0.761] | 5.99 [9.71] | 4.5 [8.3] | 3.0 [4.0] | 134 |
| leaf | v2 | 0.787 [0.772] | 6.07 [8.87] | 6.0 [9.5] | 4.0 [5.0] | 135 |
| leaf | v2 tags only | 0.799 [0.783] | 7.86 [10.41] | 7.5 [10.0] | 4.0 [5.0] | 135 |
| leaf | tags only, no cam-1 corr. | 0.785 [0.769] | 5.81 [8.47] | 4.5 [8.5] | 4.0 [5.0] | 135 |
| heart | current | 0.785 [0.752] | 5.07 [9.30] | 5.5 [8.5] | 4.0 [6.0] | 127 |
| heart | v2 | 0.811 [0.782] | 3.11 [8.12] | 5.5 [8.5] | 3.5 [4.0] | 127 |
| heart | v2 tags only | 0.811 [0.777] | 5.61 [9.57] | 7.5 [13.5] | 4.0 [5.0] | 127 |
| heart | tags only, no cam-1 corr. | 0.796 [0.777] | 3.25 [8.34] | 5.5 [7.9] | 4.0 [5.0] | 127 |
| push_heart | current | 0.744 [0.714] | 8.77 [10.53] | 9.5 [12.9] | 5.0 [6.0] | 174 |
| push_heart | v2 | 0.784 [0.759] | 8.10 [9.73] | 8.5 [12.5] | 4.0 [6.0] | 175 |
| push_heart | v2 tags only | 0.794 [0.771] | 10.34 [11.38] | 10.5 [14.5] | 6.0 [8.0] | 175 |
| push_heart | tags only, no cam-1 corr. | 0.767 [0.739] | 8.49 [10.37] | 9.5 [12.5] | 5.0 [6.0] | 175 |
| swan | current | 0.761 [0.704] | 11.96 [18.24] | 8.5 [11.7] | 5.0 [7.0] | 198 |
| swan | v2 | 0.811 [0.761] | 3.70 [7.36] | 6.5 [10.5] | 3.5 [5.7] | 198 |
| swan | v2 tags only | 0.828 [0.778] | 7.77 [9.26] | 8.0 [12.5] | 4.0 [5.0] | 198 |
| swan | tags only, no cam-1 corr. | 0.796 [0.758] | 7.40 [9.82] | 7.5 [10.7] | 4.0 [6.0] | 198 |

### Held-out camera 2, calibration-robust: per-frame correction still needed after one constant camera-2 correction (median [p90])

| pour | source | constant cam-2 correction rot deg / t mm | pouring-tilt corr deg | total rot corr deg | spout-tip corr mm | ICP failures |
|---|---|---|---|---|---|---|
| leaf | current | [0.58, -1.78, 0.93] / [-1.8, -1.1, 0.1] | 2.14 [5.22] | 2.91 [5.54] | 3.75 [9.86] | 16/38 |
| leaf | v2 | [1.82, -2.32, 1.06] / [-0.8, -3.2, -1.7] | 4.46 [6.51] | 6.70 [11.29] | 5.85 [9.49] | 6/18 |
| heart | current | [0.01, -2.73, 2.58] / [-3.8, -0.4, 1.3] | 2.01 [4.78] | 3.72 [8.00] | 3.70 [9.26] | 44/97 |
| heart | v2 | [1.16, -2.42, 1.86] / [-0.8, -2.2, -0.3] | 1.97 [4.48] | 3.64 [12.58] | 4.26 [13.17] | 54/74 |
| push_heart | current | [-0.8, 0.03, 0.84] / [-5.6, -1.5, 2.0] | 1.62 [6.43] | 2.23 [8.80] | 2.19 [20.15] | 39/110 |
| push_heart | v2 | [-0.16, -0.91, 0.92] / [-2.7, -3.0, -0.6] | 8.08 [12.52] | 10.57 [13.58] | 23.54 [31.87] | 13/25 |
| swan | current | [-0.06, -0.95, 3.34] / [-7.6, -0.8, -0.2] | 3.28 [7.83] | 5.14 [12.38] | 11.34 [26.45] | 63/141 |
| swan | v2 | [-0.68, -0.31, 3.33] / [-2.2, -2.1, -0.4] | 1.38 [8.74] | 2.12 [13.08] | 10.44 [24.96] | 40/86 |

### In-sample cameras 0 / 1 (every 3rd frame): IoU median, depth abs residual median mm

| pour | cam | current IoU | v2 IoU | current depth | v2 depth |
|---|---|---|---|---|---|
| tulip | 0 | 0.935 | 0.951 | 1.76 | 1.05 |
| tulip | 1 | 0.934 | 0.945 | 2.34 | 0.96 |
| leaf | 0 | 0.930 | 0.946 | 1.97 | 1.25 |
| leaf | 1 | 0.937 | 0.945 | 1.12 | 0.69 |
| heart | 0 | 0.921 | 0.952 | 1.61 | 1.09 |
| heart | 1 | 0.937 | 0.942 | 2.09 | 1.36 |
| push_heart | 0 | 0.935 | 0.956 | 2.08 | 0.90 |
| push_heart | 1 | 0.962 | 0.968 | 2.03 | 0.62 |
| swan | 0 | 0.928 | 0.943 | 2.18 | 1.34 |
| swan | 1 | 0.930 | 0.948 | 2.62 | 0.91 |

### Stream-origin check (independent 3-camera stream tracer; same frames for both): stream line at tip height vs spout tip, horizontal mm, median [p90]; along / across spout heading (signed medians)

| pour | frames | current | v2 | current along / across | v2 along / across |
|---|---|---|---|---|---|
| tulip | 23 | 4.5 [7.7] | 3.5 [7.1] | -2.1 / 0.4 | 2.9 / -0.8 |
| leaf | 16 | 4.8 [8.9] | 3.4 [10.0] | -1.1 / 0.1 | 2.8 / -0.1 |
| heart | 6 | 5.8 [17.8] | 2.1 [10.0] | -4.7 / 3.4 | -0.4 / 2.0 |
| push_heart | 15 | 4.7 [18.2] | 6.0 [18.2] | 0.4 / 2.8 | 5.0 / -0.4 |
| swan | 6 | 6.6 [8.9] | 4.3 [8.5] | -0.0 / 5.6 | 4.2 / 0.7 |

### Leave-one-camera-out (held-out camera tag corners predicted by the fit without that camera, px median [p90]) and cross-camera bias

| pour | v2: cam0 held out (all / joined <12 ms) | v2: cam1 held out (all / joined) | current (audit, joined samples): cam0 / cam1 | v2 cam0-only vs cam1-only tip bias mm (|median vector|) / p50 abs diff | current tip bias mm (audit) | v2 tilt bias deg | current tilt bias deg |
|---|---|---|---|---|---|---|---|
| tulip | 0.71 [1.76] / 0.62 [1.61] | 1.60 [3.07] / 1.49 [2.79] | 15.2 [20.3] / 10.8 [40.1] | 0.7 / 1.0 | 16.4 | -0.12 | 0.44 |
| leaf | 0.97 [1.64] / 0.95 [1.62] | 1.58 [2.42] / 1.54 [2.40] | 16.3 [22.0] / 11.9 [15.4] | 0.8 / 1.3 | 16.1 | 0.02 | 1.19 |
| heart | 1.45 [2.13] / 1.44 [1.98] | 0.93 [1.56] / 0.96 [1.67] | 11.1 [19.4] / 4.6 [11.6] | 1.1 / 1.3 | 13.6 | 0.00 | 0.45 |
| push_heart | 2.30 [3.23] / 2.39 [3.25] | 1.76 [2.95] / 1.67 [2.82] | 13.1 [19.9] / 17.5 [40.0] | 0.6 / 0.9 | 16.5 | -0.29 | 1.37 |
| swan | 0.80 [1.18] / 0.69 [1.08] | 1.65 [3.15] / 1.55 [2.76] | 14.5 [17.9] / 10.1 [18.5] | 0.4 / 0.8 | 16.1 | -0.32 | 2.54 |

### Dynamics (5 ms grid; p50 / p95 / max)

| pour | source | body angular acceleration deg/s^2 | spout-tip acceleration m/s^2 | angular speed deg/s |
|---|---|---|---|---|
| tulip | v2 | 65 / 299 / 629 | 0.20 / 0.94 / 1.78 | 3.82 / 45 / 84 |
| tulip | current, production 40 ms smoothing | 378 / 1067 / 13719 | 0.70 / 2.05 / 9.98 | 16.64 / 57 / 886 |
| tulip | current, raw Slerp | 0.00 / 37254 / 1039902 | 0.10 / 56 / 665 | 49 / 258 / 6055 |
| leaf | v2 | 73 / 263 / 551 | 0.22 / 0.75 / 1.38 | 3.70 / 12.44 / 100 |
| leaf | current, production 40 ms smoothing | 398 / 928 / 7039 | 0.72 / 1.80 / 7.11 | 15.22 / 35 / 529 |
| leaf | current, raw Slerp | 0.00 / 29119 / 122924 | 0.09 / 39 / 159 | 45 / 221 / 784 |
| heart | v2 | 43 / 160 / 376 | 0.12 / 0.40 / 0.80 | 3.45 / 7.89 / 63 |
| heart | current, production 40 ms smoothing | 396 / 1002 / 1571 | 0.71 / 1.75 / 3.51 | 14.99 / 38 / 50 |
| heart | current, raw Slerp | 0.00 / 18414 / 108339 | 0.06 / 34 / 186 | 35 / 209 / 589 |
| push_heart | v2 | 63 / 321 / 493 | 0.17 / 0.87 / 1.45 | 5.35 / 49 / 86 |
| push_heart | current, production 40 ms smoothing | 410 / 851 / 1189 | 0.77 / 1.84 / 4.65 | 19.52 / 48 / 73 |
| push_heart | current, raw Slerp | 0.00 / 41641 / 112454 | 0.17 / 59 / 307 | 61 / 266 / 691 |
| swan | v2 | 48 / 229 / 759 | 0.16 / 0.84 / 2.50 | 3.28 / 9.38 / 142 |
| swan | current, production 40 ms smoothing | 517 / 1001 / 1734 | 0.94 / 1.93 / 3.17 | 18.60 / 38 / 147 |
| swan | current, raw Slerp | 0.00 / 64258 / 149210 | 0.48 / 89 / 206 | 104 / 364 / 729 |

### Flips

| pour | current tip.npz samples > 10 deg from v2 (of n) | > 5 deg | current ba.pkl samples > 10 deg | v2 detections where another pose branch fits > 4x better |
|---|---|---|---|---|
| tulip | 1 / 268 | 26 | 13 / 280 | 4 / 490 |
| leaf | 1 / 171 | 4 | 17 / 190 | 2 / 365 |
| heart | 0 / 152 | 9 | 13 / 165 | 1 / 331 |
| push_heart | 0 / 248 | 16 | 15 / 265 | 10 / 447 |
| swan | 0 / 303 | 27 | 6 / 310 | 1 / 568 |

### Downstream: production inlet chain (cup_poses -> ForwardPitcher kona700, fill at the cut), v2 vs current

| pour | flow rms diff mL/s | mean flow (current) mL/s | poured volume current / v2 mL | dV mL | on/off mismatch | landing diff mm p50 [p90] | landing mean shift mm (x, y) | axis tilt diff deg median / p90 abs |
|---|---|---|---|---|---|---|---|---|
| tulip | 2.52 | 11.3 | 122.6 / 121.1 | -1.5 | 0.011 | 3.2 [5.1] | [-1.16, -2.21] | -0.13 / 1.02 |
| leaf | 3.25 | 13.2 | 114.1 / 102.5 | -11.6 | 0.004 | 2.4 [7.7] | [1.41, -2.31] | -1.12 / 1.78 |
| heart | 3.03 | 16.0 | 123.4 / 120.9 | -2.6 | 0.021 | 2.4 [4.7] | [-1.24, -1.45] | -0.15 / 0.96 |
| push_heart | 3.20 | 13.8 | 122.3 / 121.0 | -1.3 | 0.010 | 3.3 [5.0] | [-1.45, -1.88] | -0.24 / 1.07 |
| swan | 2.12 | 9.9 | 110.7 / 105.7 | -5.0 | 0.000 | 4.0 [5.7] | [-2.88, -2.02] | -0.51 / 1.17 |

### Landing point vs the MEASURED stream foot (trusted stream_foot.json frames)

| pour | n | current: abs diff p50 [p90] mm, bias (x, y) | v2: abs diff p50 [p90] mm, bias | debiased p50 cur / v2 | fast motion rms (x, y) mm: foot / current / v2 | fast-motion correlation with the foot (x, y): current / v2 |
|---|---|---|---|---|---|---|
| tulip | 118 | 9.7 [13.1], [0.62, 6.95] | 7.0 [11.3], [-0.87, 5.23] | 4.8 / 3.9 | [1.26, 2.74] / [1.65, 1.62] / [0.32, 0.78] | [-0.008, 0.301] / [0.02, 0.301] |
| leaf | 53 | 10.6 [16.7], [0.32, 7.81] | 9.5 [14.8], [1.12, 6.26] | 6.3 / 6.0 | [2.0, 2.97] / [2.26, 1.38] / [0.53, 0.73] | [0.079, 0.085] / [-0.052, 0.197] |
| heart | 87 | 9.3 [12.5], [0.72, 4.8] | 7.7 [11.4], [-0.51, 3.5] | 6.4 / 6.3 | [2.24, 3.14] / [1.55, 1.2] / [0.25, 0.47] | [0.109, 0.057] / [0.045, 0.066] |
| push_heart | 81 | 10.4 [17.8], [2.36, 7.85] | 8.5 [14.4], [0.99, 5.84] | 6.9 / 6.4 | [2.18, 2.39] / [1.93, 1.67] / [0.41, 1.0] | [0.166, 0.084] / [0.071, 0.17] |
| swan | 117 | 8.9 [12.9], [1.67, 8.2] | 6.5 [10.9], [-1.17, 6.14] | 3.0 / 2.8 | [1.26, 1.83] / [1.72, 1.41] / [0.72, 0.69] | [0.149, 0.227] / [0.36, 0.302] |

Notes on the tables:
* **Camera-2 numbers include camera 2's own calibration error.** The current track's values match the audit's (IoU 0.74-0.79, depth 5-12 mm).
* **The "calibration-robust" camera-2 ICP check is inconclusive.** It applies one constant camera-2 correction per source, then a per-frame spin-locked ICP. Single-view ICP failed on 30-70% of frames for both sources, the masks depend on each source's pose (so frame counts differ), and tulip crashed (singular system). It is reported for completeness only.
* **The LOCO fits share the global calibration (layout, lean, camera-1 extrinsic, CAD placement) estimated from both cameras of all five pours.** Only the trajectory is held out, and it is initialised from the remaining camera's own per-sample poses (Viterbi branch choice). The audit's LOCO numbers for the current chain also used a layout fitted with both cameras.
* **v2's smoothness is partly shaped by the prior** (σ_α 94 °/s², σ_a 0.13 m/s² per component, from the measured two-camera accelerations).
* **The prior is not suppressing real motion.** Camera-0 tag residuals stay at 0.18-0.25 px, against 0.10-0.15 px for a free per-frame pose. The measured stream foot shows no fast motion beyond 1.3-3.1 mm rms. Single-camera v2 fits agree with each other to ~1 mm.

## Output files (`out/`)

| file | format | read by |
|---|---|---|
| `<p>_pose_v2.npz` | **tip.npz key set**: `t` [s, take clock = tk.ts - tk.t0], `poses` (n,6) rotvec+pos (x_board = R x_p + p), `tip_board` (n,3), `tilt`, `az` [deg, pitcher_model.py convention], `tip_p` [m, pitcher frame]. It also carries the **pitcher_pose.npz keys** (`rotvec`, `pos`, `tilt_deg`, `az_deg`, `note`) and extras: `sd_tilt_deg`, `sd_spin_deg` (formal 1σ), `tip_p_tagframe`, `B_out_to_tag` (4x4 output→tag frame), `spline_*` (tag-frame control points, T0, dt), `offsets_s` (c1, d0, d1) [s]: depth is tied to its camera's colour clock, so this is (c1, 0, c1) (*[audit]* fixed; it was stored as (c1, 0, 0)), `layout_json`, `placement_json` | fuse_occ.PourData (`T["t"]`, `T["poses"]`, `T["tip_p"]`, then pose_inliers + Slerp), cup_track.py, action_summary.py, barista_view.py (`poses[:,3:5]`, `t`) |
| `<p>_pitcher_pose_v2.npz` | exactly the `raw_data/rig/results/<p>/pitcher_pose.npz` key set: t, rotvec, pos, tip_p, tip_board, tilt_deg, az_deg, note | real_program.load_take / cup_poses (Slerp + 40 ms smoothing), and through it stream_foot.py, run_kona.py and the holdout sensitivity chain |
| `<p>_tip_v2.json` | tip.json fields (tip_in_pitcher_mm, n_samples, tilt_good_deg) + frame note | human / scripts reading tip.json |
| `joint_v2k.pkl`, `full_<p>_v2k.pkl`, `placement_v2k.json` | fitted splines (tag frame), shared parameters, residuals | this directory's scripts |

**Frame.** The pitcher frame is CAD-aligned: z = CAD jug axis to the rim, +y = spout, origin on the axis at the height of tag 4's centre (CAD z = 29.0 mm). This is the convention every consumer assumes ("pitcher z = axis to the rim, spout +y"). The CAD body in this frame is the kona700 STL rotated so that its +x (spout) is at +y and shifted by z0 = -29.0 mm, i.e. exactly `gs_train CadShape --spout-az 90 --fix-az` with z0 -29.0.

The v2 frame is not the old tag frame: different radius (42.6 vs 47 mm), axis tilt and spin. Body-frame quantities derived in the old frame must not be mixed with v2 poses:
* tip.npz's tip_p (0, 53, 93) mm;
* pitcher_model.npz;
* pitcher_model.json wall_bottom_mm (v2: CAD bottom at z = -29.0 mm).

Use v2's `tip_p`, (-0.2, 50.1, 88.3) mm, which equals the production `TIP_KONA` within 0.5 mm.

**Coverage.** v2 covers the trimmed pattern phase of each pour, from the first to the last tag/depth observation of cameras 0/1 (≈ the full trimmed take). `gs_dataset.py` and `full_pour.py` read `pours/<p>/full/ba.pkl`, the whole recording, and are **not** covered.

**Not installed.** The repo was read only. Nothing in `raw_data/` or `pours/` was overwritten. To use v2, copy `<p>_pitcher_pose_v2.npz` to `raw_data/rig/results/<p>/pitcher_pose.npz` and `<p>_pose_v2.npz` to `pours/<p>/tip.npz`.

## Caveats

* **Camera 2 is held out** of the fit (never read by the fitting code). Its nominal timestamps are used for validation, since its clock offset is unknown.
* **Indirect use of camera 2 in the masks.** The depth masks of cameras 0/1 use `disc.json` / `top/*.npz` (cup and liquid geometry) and the stream foot. Those came from multi-camera pipelines that include camera 2, but they are used only to exclude non-pitcher pixels; they carry no pitcher pose.
* **Stream checks use camera-2 colour.** The stream tracer (validation only) and `stream_foot.json` use all three cameras' colour. The spout azimuth revision is disclosed above.
* **The absolute orientation of v2 is camera 0's world.** If camera 0's extrinsic is off (cameras 1 and 2 both need corrections of the same sign about board y relative to it), every v2 tilt carries that offset. Recalibrating the three extrinsics (e.g. the board visible to all cameras at the start of each take) is the single most valuable fix for "full" recovery.
* **The CAD needs a facet-phase fix.** The 14-gon flats should be aligned with the tags, with the spout at a vertex. Camera 0 sees the surface 0.6-0.9 mm outside the CAD.

## Scripts (`work/`)

* **Library:** `pv2lib.py` (geometry, tag model, per-sample BA), `spline.py` (SO(3) cumulative B-spline), `ctfit.py` (continuous-time LM with Schur complement over pours, tags/depth/prior, extrinsic, placement and offset parameters).
* **Fitting:**
  * step 1: `layout_fit.py`, `pair_check.py`;
  * prior / bandwidth: `dyn_stats.py`;
  * tags-only CT: `fit_tags.py`;
  * depth points: `depth_pts.py`;
  * placement: `place_cad.py`, `place_cad2.py`;
  * final: `fit_joint.py`;
  * LOCO: `loco_fit.py`;
  * writer: `write_out.py`.
* **Diagnostics:** `diag_*.py`, `profile_check.py`, `depth_xcam.py`, `liquid_plane.py`.
* **Validation:** `heldout_cam.py`, `cam2_icp.py`, `stream_check.py`, `eval_misc.py`, `downstream.py`, `foot_check.py`, `sheet_v2.py`, `make_report.py`, `readme_tables.py`.
* **All numbers:** `out/summary.json`, `out/*.json`. **Figures:** `img/timeseries.jpg`, `img/heldout_cam2.jpg`, `img/sheet_<p>.jpg` (camera 0/1/2 overlays), `img/downstream_<p>.jpg`, `img/foot_<p>.jpg`, `img/bandwidth.jpg`, `img/spout_az.jpg`, `img/profile.jpg`.
