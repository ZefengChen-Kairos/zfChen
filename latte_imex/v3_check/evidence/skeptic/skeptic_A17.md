<!-- Re-verified 2026-10-11 second pass: optimize.py L17-31/L47-53, action_opt.py L85-88, cell_test.py L64-97,
solver.py L310/326 (tau_new derived), POUR_MODEL_SUMMARY.md L51; numbers re-computed from saved fields with
skeptic_A17_check.py / check2.py and runs/joint_sweep_r2fix/results.json (60/60 ok rows: corr(log cp, loss) -0.06,
beta -0.13, kQ +0.42, B_dep +0.35). Target gap_frac (N=96): heart 0.010, push_heart 0.184, layered 0.052, tulip 0.341,
leaf 0.193, swan 0.320. Nothing in the draft below needed changing. -->

# Skeptic check of A17 — "目标函数只看末帧、面积主导"

Verdict: **confirmed (with nuance)**. Every element of the mechanism is traceable in the code and
re-computable from the saved runs; two sub-claims need sharpening (c_p/β/τ_new "不可辨识", and *why* the
S-ribbon beat J_A).

Checks were run with `skeptic_A17_check.py` / `skeptic_A17_check2.py` in this directory (only
`loss_fields`, `gap_stats`, `struct_stats` on saved `.npy/.npz` fields; no simulation launched).

## 1. What the code does

* `optimize.loss_fields` (optimize.py L47–53):
  `L = mean|G_σ s − G_σ w|_mask + 0.5(1 − IoU(s>0.5, w>0.5)) + 0.5·|A_s − A_w|/A_mask`.
  All three terms are overlap/area statistics of the thresholded or blurred whiteness; nothing is
  evaluated on gaps, edges, line counts or time.
* **Final frame only**: `optimize.simulate` (L17–31) returns `sol.concentration()` after the loop;
  `action_opt.run_one` L85–88 computes the loss once from `sol.concentration()` at `script.T`. Neither
  `evaluate` (L79–93), `evaluate_joint`, `search/refine` (action_opt L101–144) nor `objective_struct`
  (cell_test L81–97) read any snapshot. Confirmed.
* Blur: `optimize.main` uses `--blur 0.015` → σ = 1.5 % D (L202, L213); `action_opt.run_one` passes
  `N/48` = 2.08 % D (L88) and J_A inherits that. The diagnosis's "σ = 1.5 % D" is the CMA-ES script
  value; the action/J_A runs used 2.1 % D. Minor.
* `cell_test.objective_struct` (L81–97): `J_A = loss + |gap−gap_t| + 0.5|ln((e+0.01)/(e_t+0.01))| + W1(q,q_t)`;
  `struct_stats` docstring L64–65 says "Location-free on purpose". Confirmed.

## 2. Numbers (re-computed)

Heart target (N=128, σ=1.92): white fraction 0.364, `gap_frac` 0.009 — the heart *target itself* is an
almost structureless blob.

| candidate vs heart target | L | l1 | 0.5(1−IoU) | 0.5·area |
|---|---|---|---|---|
| same-area disc at white centroid | **0.162** | 0.079 | 0.083 (IoU 0.83) | 0.000 |
| target with holes filled | 0.035 | 0.035 | 0 | 0 |
| `runs/final/heart` (V0.5 controls, IMEX 256 → 128) | **0.442** | 0.194 | 0.176 (IoU 0.65) | 0.072 |
| whole cup white | 1.251 | | | |

So "白盘 0.229 优于 Codex 心形 0.424" (conv [95]) holds and is even stronger with a centred disc
(0.16 vs 0.44). The Codex-style heart loses mainly because it is *too big* (white 0.51 vs 0.36) and
mis-placed, i.e. by area/overlap, as [95] said.

Is it heart-specific? Same-area disc vs each target (N=96): heart 0.162, layered 0.252, leaf 0.395,
push_heart 0.498, tulip 0.581, swan 0.563. The disc only *wins outright* for heart/layered; for
push_heart it still ties the best simulated result (0.498 vs best 0.511, base 0.589). The loss is
therefore not blind to the **outline** (IoU 0.34–0.40 for tulip/swan/push discs) — it is blind to
**interior structure**: filling every brown gap of a target costs only 0.03 (heart, push_heart) to
0.12 (leaf); for the tulip, erasing gaps that are 34 % of the outline costs 0.081, less than a 20 %
radius error of a disc (≈0.25). That is the precise sense in which it is "area-dominated".

**E28 ("把千层心层纹填平") confirmed from `runs/action_opt/refine_N96.json`** (old closure):
layered_heart base loss 0.707 / white 0.112 / gap 0.110 → best 0.301 / white 0.346 / gap **0.022**
(target gap 0.052) at h×0.41, q×1.5, s×1.23. Tulip: loss 0.614→0.497 while gap 0.211→0.073 moved
*away* from the target 0.341. Heart: gap 0.028→0.014. The optimiser lowers the spout and raises the
flow until the white covers the outline; gaps are collateral.

**J_A / S-ribbon ([555]) confirmed from `runs/action_opt/stage_old_push_heart_cma/history.json`**:
best gen 18 J = 0.5075 = loss 0.448 + t_gap 0.016 + t_edge 7.7e-5 + t_w1 0.043 (gap_frac 0.167 vs
target 0.184). Nuance: the three location-free terms sum to only 0.06; the ribbon won because the
**pointwise loss itself** rated it 0.448 against 0.682 for the video-derived script (gen −1, J 0.761).
So the statement "J_A 位置无关被骗过" is right in that the structure terms offered no defence, but the
deciding term was again the overlap loss — the same defect as A17's first half, not a separate one.
Also note J_A does reject a solid disc (1.217 vs 0.803, conv [534]), so it is not simply fooled by blobs.

## 3. "运动参数不可辨识" — partly supported, needs sharpening

* c_p: E-10 (conv part1) IoU 0.89–0.96 between c_p 0.3 and 4 at N=128; in the 60-sample LHS sweep
  `runs/joint_sweep_r2fix/results.json` corr(log c_p, mean loss) = −0.06. POUR_MODEL_SUMMARY L51
  "c_p：末帧完全不敏感". Supported.
* β: corr with loss −0.13 (mean), −0.26 (heart), +0.05 (tulip) — weak. But POUR_MODEL_SUMMARY §5 reports
  β ↔ heart gap count r = −0.51 and β ↔ tulip gap count −0.39. So β *does* leave an end-frame
  signature (number of brown gaps); the loss simply does not measure it. "不可辨识" should read
  "末帧损失读不到 β 的痕迹", not "末帧不含 β 信息".
* τ_new: in solver.py L310/326 `tau_new = max(2U⊥/g_red, 1e-3)` — a derived quantity (from g′ and the
  inlet), never a searched parameter. Listing it as an unidentifiable "motion parameter" is sloppy.
* What the loss is actually driven by in the sweep: kQ (corr +0.42 mean, +0.75 tulip, −0.56 heart) and
  B_dep (+0.35 / +0.59 / −0.40) — deposition amount, i.e. white area. Consistent with "面积主导".

## 4. Items quoted as evidence that I could not re-verify cheaply

E20 (tulip loss 0.58 → 0.72 after the barista fix, notion_z1 row E20) is documentary only; no saved
fields were re-scored. It is consistent with everything above but not independently checked here.

## 5. Corrected statement

The objective is a single final-frame overlap score (blurred-L1 + IoU + area; σ 1.5–2.1 % D) that is
sensitive to outline position and white area but almost insensitive to interior structure: erasing all
brown gaps of a target costs 0.03–0.12, a same-area disc beats the V0.5 heart (0.16 vs 0.44) and ties
the best push_heart (0.50 vs 0.51). Consequently the (h,q,s) search filled the layered-heart gaps
(0.110→0.022) and the sweep loss is driven by kQ/B_dep (area) while c_p is invisible and β's
end-frame signature (gap count, r −0.5) is unread. J_A's location-free terms (0.06 of 0.51) did not
stop the S-ribbon, but the ribbon won on the same overlap loss (0.448 vs 0.682 for the video script).
τ_new is derived, not searched, and should be dropped from the list.
