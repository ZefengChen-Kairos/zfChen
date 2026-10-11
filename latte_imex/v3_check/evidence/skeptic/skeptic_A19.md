# Skeptic check of A19 — "观测算子不一致：仿真 clip(c/0.6) 与目标 w 不是同一观测；w 把褐线糊成灰 0.3–0.5；面积重叠损失必奖励软团"

Verdict: **confirmed with corrections** (independently re-verified 2026-10-11; numbers below are from my own
run of `skeptic_A19_check.py` on the six `targets/*.npz` and the best cached `runs/action_opt/**/_N96.{json,npy}`;
no simulation launched).

The *mechanism* "two different observation operators, no shared P" is fully traceable in code and is a direct
violation of notion_senior §1.3 L40 / §5 L273. Two numbers quoted in the claim are wrong (push_heart gap is
w ≈ 0.00–0.04, not ~0.1; grey lines 0.3–0.5 hold for the swan and, after zoom to N = 96, the leaf — not for the
push_heart). The causal link "operator mismatch ⇒ loss rewards soft blobs" is overstated: the blob preference is a
property of `loss_fields` (σ = N/48 blur + IoU + area, single final frame) and would survive an identical P. The
part of the mismatch that measurably bites is the **brightness scale** (v2 W tops out at 0.75 while target foam is
mapped to 1.0 by the Otsu-median ramp).

## 1. What the code DOES (vs what docs say)

**Target side** — `targets.py`
* `liquid_ellipse` L68–89: HSV brown mask (H 0.02–0.13, S > 0.3, V > 0.25) → largest component → convex hull →
  Fitzgibbon ellipse.
* `warp` L92–108: affine ellipse→unit-disc resampling on M×M (192; leaf 256), per-pattern `rot` (L24–31:
  −125, −100, −130, −165, +45, 0°), bilinear `map_coordinates`.
* `whiteness` L110–125: f = (1−S)·V, Otsu split of f inside the cup, then **linear ramp between the medians of the
  two classes**: `w = clip((f − lo)/(hi − lo), 0, 1)`. So by construction the foam-class median maps to exactly 1.0
  and the crema median to 0.0; anything between is grey.
* Leaf: `whiteness_foam3` L137–150 (3-means on f, ramp between tan and foam centres, rim blobs < 1 % of cup removed).
* `optimize.target_at` L72–76: `ndimage.zoom(w, N/M, order=1)` to the loss grid (N = 96 in all action_opt fits).

**Simulation side** — `optimize.py`
* `whiteness_sim` L34–36: `s = clip(c/0.6, 0, 1)` (old / twolayer); `visible` L39–44: for `closure == "v2"` the
  raw opacity `W` clipped to [0, 1]. **No rendering, no HSV, no Otsu, no ramp** on the simulation side.
* `render.py` L1 docstring: "Diagnostic rendering … (not an optical renderer)"; `c_to_rgb` (c_sat 0.8, γ 0.8) is
  never used in any loss.
* Loss `loss_fields` L47–53: `mean|G_σ s − G_σ w|` + 0.5(1 − IoU@0.5) + 0.5|ΔA|. In the real fits σ = `N/48`
  (`action_opt.py` L87) = D/48 ≈ 1.67 mm at D = 80 mm, **not** the 1.5 % D quoted in README L47.
* `cell_test.objective_struct` L81–97 adds gap-fraction / edge / whiteness-quantile penalties on top of the same
  `visible` output — still no shared P.

**Docs**: README L43–47 states this pipeline openly ("Otsu split … s = clip(c/0.6, 0, 1)"). No repo doc claims
the senior's "same P on render and video" is implemented, so this is a *protocol violation* of notion_senior §1.3
("用校准集固定的图像提取算子 P 同样处理观测和模拟外观 … 不将真实白色 mask 直接当成 c"), not a doc-vs-code
contradiction. Confirmed.

## 2. Numbers (my re-run)

**Target w histograms** (fraction of cup pixels in bins [0,.1,.2,.3,.5,.7,.9,1], native grid → N = 96 nearly
identical):

| pattern | [0,.1) | [.1,.3) | [.3,.5) | [.5,.7) | [.7,.9) | [.9,1] | gap_frac (N96) |
|---|---|---|---|---|---|---|---|
| heart | .548 | .053 | .031 | .029 | .054 | .285 | 0.010 |
| push_heart | .646 | .069 | .033 | .037 | .065 | .150 | 0.184 |
| layered_heart | .517 | .033 | .061 | .098 | .207 | .084 | 0.052 |
| tulip | .535 | .096 | .108 | .101 | .110 | .050 | 0.341 |
| leaf (256) | .553 | .053 | .055 | .058 | .073 | .208 | 0.193 |
| swan | .555 | .096 | .131 | .088 | .102 | .027 | 0.320 |

* push_heart centre column (y up): big crescent 0.97–1.00, then **0.83 → 0.01 → 0.00 → 0.59** into the small
  heart; the lower region reads 0.06/0.01/0.16/0.26/0.11/0.02. The gap is ≈ 0.0 and is counted by `gap_stats` as
  18.4 % of the closed outline. **"0.05 D 缝只占 ~0.1" is wrong**; push_heart has the cleanest w of the six
  (only 3.3 % of cup in [0.3, 0.5)).
* leaf 256²: centre row crosses lines at 0.24/0.01/0.00/0.30/0.15/0.01 (clean) but also 0.46/0.56/0.65/0.50
  (grey); at N = 96 the same row gives 0.45/0.09/0.33/0.25/0.26 — grey *after* the zoom to the loss grid.
* swan 192²: wing is mottled 0.1–0.9, 13 % of cup in [0.3, 0.5). "糊成灰 0.3–0.5" holds for the swan, half for
  the leaf, not for push_heart.
* Contamination (preview.png): subtitles are **outside** the cup in all six frames; a semi-transparent watermark
  (glyphs) lies **inside** the warped disc of layered_heart (lower right), tulip (lower right), swan (left, visible
  in w) and faintly leaf (right edge); heart w has a bright rim crescent at the bottom, tulip/swan have rim arcs
  — glare counted as foam. Heart w shows 8×8 JPEG blocks (narrow f-range stretched by the ramp). Leaf frame was
  moved 14.4 s → 12.2 s for this reason (`targets.py` L28 comment). "字幕反光" should read "水印 + 杯沿反光".

**Loss sensitivity** (σ = 2 px at N = 96, `loss_fields`; "gap-closed" = target white set closed by a 6 mm disc +
holes filled; "blurred" = G_σ(w)·1.3; "×0.7" = target scaled to max 0.7):

| pattern | best cached sim: loss / IoU / l1 (closure) | sim gap_frac vs target | gap-closed target | blurred target | target ×0.7 (l1) |
|---|---|---|---|---|---|
| push_heart | 0.403 / 0.48 / 0.141 (v2, v2w09_02) | 0.038 vs 0.184 | **0.198** / 0.82 | 0.142 | 0.176 (0.074) |
| leaf | 0.317 / 0.65 / 0.129 (v2, v2c07_01) | 0.004 vs 0.193 | 0.240 / 0.81 | 0.192 | 0.226 (0.099) |
| swan | 0.476 / 0.34 / 0.138 (v2) | 0.000 vs 0.320 | 0.410 / 0.64 | 0.285 | 0.340 (0.070) |
| tulip | 0.477 / 0.38 / 0.166 (old, jc04_08) | 0.112 vs 0.341 | 0.422 / 0.66 | 0.208 | 0.345 (0.078) |
| heart | 0.235 / 0.76 / 0.108 (v2) | 0 vs 0.010 | 0.042 / 0.99 | 0.076 | 0.166 (0.108) |

Reading:
1. Gaps are **not weightless**: closing every gap of the push_heart target costs 0.20 (half the best sim loss), of
   the swan/tulip 0.41/0.42. So the push_heart "three components not separated" failure *is* seen by the loss.
2. Nevertheless a gap-free blob with the correct outline beats every simulation by ~2× on push_heart/leaf, and a
   blurred target by ~3×: the loss ranks outline ≫ gaps ≫ line sharpness. This ordering comes from blur + IoU +
   area on a single frame, and would be unchanged if both sides went through the same P — if anything, pushing the
   sim through the same Otsu-ramp + JPEG + glare would add blur to the sim side too.
3. **Brightness-scale offset is the measurable operator mismatch.** Best v2 push_heart: `s = W` histogram 0.195 of
   cup in [0.5, 0.7), 0.055 in [0.7, 0.9), **0 pixels ≥ 0.9, max 0.75**; swan max 0.71; target foam is 0.95–1.0 by
   construction of the ramp. Scaling the *target itself* to 0.7 already costs 0.166–0.345 (l1 0.07–0.11), i.e. a
   large part of the sims' l1 ≈ 0.13–0.14 is a calibration offset unrelated to pattern geometry. For the old
   closure, "white = 60 % milk fraction" (c_sat 0.6) has no optical basis (physics_prior L194: foam is optically
   saturated at ≈ 1 mm); neither side models the camera.
4. A 1.2 mm (1.5 % D) line blurred with σ = 1.67 mm keeps ≲ 0.3 of its contrast in the l1 term, so 1–2 % D lines
   are damped by the loss even where w holds them at 0 (push_heart gap, most leaf lines).

## 3. What is wrong or overstated in A19

1. "推心 0.05 D 缝只占 ~0.1" — false; gap w ≈ 0.00–0.04, 5 mm wide, 18 % of outline; it costs 0.2 of loss.
2. "褐线糊成灰 0.3–0.5" — true for swan, partly for leaf (mainly after the zoom to N = 96), false for push_heart.
3. "面积重叠损失必奖励软团" — correct as a statement about `loss_fields` but not a *consequence* of the operator
   mismatch; it follows from σ = D/48 blur + IoU + area on the final frame only.
4. "每图旋转 / 弱透视" are frame-alignment steps (sim is already in the solver frame; `rot` matches the hand-written
   V0.5 script axis), not observation differences; they could only hurt via a wrong hand-chosen angle, which the
   claim neither argues nor measures.
5. "字幕" — subtitles are outside the cup; in-cup contamination is watermark text (layered/tulip/swan/leaf-edge),
   rim glare (heart/tulip/swan) and JPEG blocks (heart).

## 4. Corrected statement (A19′)

The fits never used the senior's shared extraction operator P: the target is HSV → (1−S)V → Otsu median ramp on a
warped video frame (watermark text and rim glare included), the simulation is `clip(c/0.6)` (old) or raw `W` (v2)
with no rendering (`optimize.py` L34–44, `targets.py` L110–150). Measurable consequences: (a) a brightness-scale
offset — best v2 fits never exceed W ≈ 0.75 where the target foam is 1.0, which alone accounts for a large part of
the l1 term; (b) grey lines in the swan (0.1–0.9) and, after zoom to N = 96, the leaf (0.2–0.5). The push_heart gap
is clean (w ≈ 0.0, 5 mm) and closing it costs 0.20 of loss, so that failure is a physics/action failure the loss
does see. Independently of P, the loss (σ = D/48 blur + IoU + area, final frame only) ranks outline ≫ gaps ≫ thin
lines: a gap-free blob with the target outline scores 0.20 vs the best simulation's 0.40.

Experiment that would settle the residual: render the best v2 push_heart/leaf state with a flat two-tone renderer,
pass it through `targets.whiteness` (same Otsu ramp) and re-score; if the loss drops by ≳ 0.08 the operator offset is
first-order. Cheaper: rescale `W` by its foam-class median before `loss_fields`.
