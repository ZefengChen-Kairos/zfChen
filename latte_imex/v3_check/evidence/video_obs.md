# Direct observation of the six reference clips (phenomenology, not theory)

Source clips: `/tmp/claude-0/-home-user-zfChen/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/ref/reference_{heart,push_heart,layered_heart,tulip,leaf,swan}.mp4`
(all 480x640 portrait, 15 fps; durations 12 / 17 / 14 / 20 / 17 / 23 s). The `refclips/` folder only had heart + push_heart.

Frame tiles used (all in `/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review/frames/`):

| file | content |
|---|---|
| `t2_<p>.png` | whole clip at 2 fps, 6x8 tile, burned-in absolute pts (s) |
| `f_<p>_<S>.png` | 7.5 fps, cup-cropped (`crop=480:460:0:110`), window [S, S+4) s; **burned-in timestamps are relative to S** (ffmpeg `-ss` before `-i`) |
| `z_*_15fps.png`, `z_*_final.png`, `z_push_*.png` | 15 fps zooms around cut/lift/push moments, absolute t in the label |
| `cmp_targets.png` | raw `latte_imex/targets/*_frame.png` (top) vs `web/targets/*.png` (bottom) |
| `gapmeas.py` output | run-length (white/brown) along chords of the 256 px web targets, in % of cup diameter D |

All lengths below are given in cup inner diameters **D** (estimated by eye from the frames; +-20 %). "B/C/D" are the instructor's overlay points: a dashed line across the cup through the centre, **B = point nearest the pitcher spout (far rim side, ~0.2 D from the rim), C = cup centre, D = mirror point on the near/handle side**. In every clip the pitcher comes from the top of the frame and the cup handle is at bottom-right, so B = upper-right, D = lower-left of the cup in the frame. The overlay is shown only during the "go back to B" instructions (frames listed per clip below).

---

## 0. Common facts, seen in all six clips

| item | observation | where |
|---|---|---|
| cup tilt | at the start of every pour the cup is tilted ~15-25 deg toward the pitcher (rim ellipse visibly non-level; B is the low side). It is righted while pouring during the first 1-2.5 s ("同时回正杯子", heart 0.9-2.5 s). From then on the cup is level and is only moved slightly (swan: "稍微~倾斜一点" 5.5-6.5 s, a small extra tilt before the retreat). | heart f_heart_0 0.0-1.5; push f_push_heart_0 0-2 |
| stream | thick (~0.03-0.04 D) near-vertical stream from a spout 1-2 cm above the surface while drawing; thin (~0.01 D) stream from 5-10 cm during the final "拉高收尾" lift. Flow rate changes are instructed explicitly: "大流量" (big flow) for pushing/heart, "小流量" (small flow) for the cut. | all |
| **delay before white** | the stream lands for **0.9-2.1 s before any white is visible**; during this time the surface stays uniformly brown with only a darker/glossy impact spot. Measured first-white times: heart 1.47 s, push_heart 2.1 s (stage 1), 1.2 s (stage 2, 6.0->7.2), 1.2 s (stage 3, 10.0->11.2), layered 1.5 s, tulip 0.9 s (clip starts mid-pour) / 1.3 s (stage 2) / ~1.0 s (stage 3), leaf 1.7 s, swan 1.7 s. | t2_*.png, f_*_0.png |
| **impact spot stays brown** | once white appears it appears as a **ring around the impact point**, the impact point itself remains a brown "hole" of ~0.05-0.12 D for as long as the pitcher is low and pouring. The hole gives every freshly poured element a C shape opening toward the pitcher. It fills in only when the pitcher moves away (heart 4.7-6.7 s) and leaves a small notch if the pour is simply stopped (push_heart blob at t=5.6 s, `z_push_s1_after_lift.png`). | heart f_heart_0 2.5-3.9; push t=8.4, 12.6 zooms |
| white boundary | crisp (<=1-2 px at 480 px, i.e. <=0.01 D) everywhere, including after 10+ s; no visible diffusion/blur of stripes or edges. | all finals |
| brown annulus | a brown ring of 0.10-0.25 D between the pattern and the cup wall survives to the end in all six; white never reaches the wall. End-state white fraction inside the cup (web targets, threshold at mid-grey): heart 0.40, layered 0.42, leaf 0.37, tulip 0.28, push_heart 0.27, swan 0.23. (npz `w` means: 0.36/0.34/0.33/0.26/0.25/0.24.) | gapmeas output |
| **final cut** | always: lift + small flow, thin stream moves **from the B side through the pattern to the D rim** at ~0.3-0.6 D/s (heart: ~1.1 s for ~0.6 D; leaf: ~1.2 s for ~0.7 D). Effects: (i) notch at the entry side, (ii) thin stem 0.01-0.02 D wide, 0.1-0.2 D long, from the pattern to the near rim, (iii) a thin brown line along the cut path inside the white (visible in heart z_heart_cut 0.6-1.0, and as the rosetta midrib), (iv) lateral bulging of the white on both sides of the line and V-shaping of stripes. | z_heart_cut_15fps, z_layered_cut_15fps, f_leaf_8 |
| camera | hand-held/chest camera; cup moves in the frame by up to 0.3 frame widths; cup is never static until the show-off shot. | t2 |

---

## 1. Heart (大白心) - 12 s

Overlay B/C/D shown at 0.0-1.1 s and 5.3-6.6 s (absolute).

| t (s) | instruction (subtitle) | what is observable |
|---|---|---|
| 0.0-0.9 | 在B点注入流量 | thick stream lands at B; cup tilted; surface uniformly brown. |
| 0.9-2.5 | 同时回正杯子 | cup righted while pouring; 1.47 s first white: a pale spot ~0.1 D at the impact, grows to ~0.25 D by 2.0 s, 0.3 D by 2.3 s. The spot is brighter on the side away from the spout. |
| 2.5-4.6 | 奶泡形成半圆形后 | white keeps growing radially from the impact to a **C/half-ring of 0.4-0.5 D** whose open side faces the spout; inside the C the impact core stays brown (~0.1 D). Growth rate of the white radius ~0.1 D/s. |
| 4.7-6.7 | 向中心点慢慢的移动拉花缸 | pitcher moves B->C slowly (~0.1 D/s), still low, big flow. The white becomes a near-full disc 0.6-0.65 D; the brown core fills in as the impact moves onto already-white surface; the disc's far edge (D side) hardly moves, the near (B) edge grows. |
| 6.8-8.0 | 最后变小流量拉高收尾 | pitcher lifted, stream becomes thin; stream lands on the B-side half of the disc; a brownish patch appears at the new impact point (f_heart_8 0.27-0.8). |
| 8.0-9.2 | (cut) | thin stream moves C->D (toward the handle side) across the disc and exits to the rim; a thin brown line trails the stream inside the white (z_heart_cut 0.60-1.00); the B-side edge is pulled in to a notch (~0.05 D deep), the D-side edge is pulled out into a stem that reaches the rim (~0.15 D long, ~0.015 D wide). Lobes bulge sideways by ~0.05 D. Takes ~1.1 s. |
| 9.5-12 | 大白心就完成了 | final: heart ~0.55 D wide, white fraction 0.40, brown annulus 0.15-0.2 D, one brown gap only (the notch/stem). |

Gap widths (gapmeas, web target): rows y=0.3-0.5 D are `b W b` with no internal brown; y=0.6 D shows `W29 b1 W38` (the cut line is **1 % D**); y=0.7 D `W18 b11 W36` is the notch region.

Surprise: the white is **not** laid down where the stream hits; the hit point is the last place to turn white.

## 2. Push heart (推推乐) - 17 s

Overlay shown at 0.5-1.9 s, 6.0-7.0 s, 10.0-11.0 s.

| t (s) | instruction | observable |
|---|---|---|
| 0.0-2.0 | 第一段在B点注入流量 | pour at B, cup tilted, no white. |
| 2.0-4.3 | 同时轻轻的把图案推到D点 | 2.1 s first white; while the white grows the pitcher drifts gently from B toward D (~0.15 D total). Result at 4.3 s: a C-shaped blob ~0.35 D sitting left of centre (D side), open side toward B, brown core at the impact. |
| 4.4-5.9 | 抬高缸嘴收流量 | lift + stop. The blob at t=5.6 (`z_push_s1_after_lift.png`): soft-edged rounded blob 0.33 D with a small brown bite on its B side (the remaining impact hole). No tail, no stripes. |
| 6.0-7.2 | 第二段退回B点注入 | pitcher back to B, pours ~1.2 s with **no white**; during this second the old blob is displaced ~0.1 D toward D and flattens on its B side (the side facing the new impact). |
| 7.2-8.0 | (white appears) | new white ring appears at B, ~0.15-0.2 D from the old blob with brown between (gap initially ~0.15 D). |
| 8.0-9.7 | 大流量推到C点 | big flow, pitcher B->C. `z_push_s2_ring.png` (t=8.4): old blob is now an **outer crescent 0.10 D thick spanning 0.5 D**, a **brown arc gap 0.05-0.10 D**, an inner white disc ~0.25 D with a brown crescent core at its B edge. The old blob got thinner radially and longer azimuthally (deformation, not translation). |
| 9.8-10.0 | lift | |
| 10.0-11.2 | 第三段再退回B点 | pour at B, no white for ~1.2 s. |
| 11.2-11.9 | 大流量注入一个爱心 | small blob (0.15 D) appears around B. |
| 11.9-13.3 | (move B->C, big flow) | pitcher moves from B into the centre of the crescents; the small blob is left behind at B as a separate island whose **tail is dragged toward C** (so it becomes a little heart pointing at the main pattern); the stream now builds a third blob inside the inner crescent, pushing both crescents outward (gaps stay ~0.05-0.1 D). `z_push_s3_blob.png` t=12.6. |
| 13.3-14.5 | 小流量拉高收尾 | lift; thin stream drawn from the inner blob through both crescents to the D rim (stem at lower-left, `z_push_cut.png` t=13.3, `z_push_final.png` t=16.2). The inner blob gets a notch on its B side. |

Gap widths (web target): y=0.5 D `W10 b13 W17 b5 W8` -> **brown gaps 0.13 D and 0.05 D**; y=0.6 D `W5 b9 W16 b4 W9`; vertical x=0.5 D `W9 b5 W38`. The outer crescent thickness 0.05-0.10 D, inner elements 0.1-0.2 D.

Three components as seen: (a) outer crescent = stage-1 blob after two pushes; (b) inner heart = stage-3 blob (notch at B); (c) detached small heart at B = the stage-3 pour before the pitcher moved. A fourth, the stage-2 crescent, is squeezed between (a) and (b) and in the final is only ~0.05 D thick (web target shows it merged/thin).

When the gaps form: **during the first ~1 s of each new pour, before the new white appears** (the brown surface pushed ahead of the new source separates the old white from the impact), then they persist and narrow only slightly as the new white expands. The gaps are not created by the lift or the cut.

## 3. Layered heart (千层心) - 14 s

Overlay at 0.0-1.1 s.

| t (s) | instruction | observable |
|---|---|---|
| 0-1.4 | 在B点注入流量 | pour, tilt, brown. |
| 1.47 | first white | spot at impact. |
| 1.6-4.5 | 看到白色的线条出现后开始摆动 | wiggle starts the moment white appears (lateral oscillation of the spout ~0.08-0.1 D amplitude, ~2.5-3 Hz). Each half-cycle leaves a white arc ~0.02-0.03 D wide, convex toward D, separated by brown lines of similar width. Stack grows to ~0.35 D with 6-8 arcs by 4.5 s. |
| 4.5-7.3 | 摆动的纹路形成半圆形后向前移动 | pitcher moves forward (B->C) while wiggling; older arcs are pushed toward D and wrap laterally so the stack closes into a disc ~0.6 D with ~12 arcs; arcs at the D side are the most tightly packed (spacing ~0.025 D), near the impact ~0.04 D. |
| 7.3-10.2 | 拉花缸最后小流量拉高收尾 | flow reduced, pitcher raised, but it **lingers over the B side of the pattern for ~2 s**: the ripples under it are overwritten by a solid white V-region (the smooth lobes of the final heart, f_layered_heart_8 1.6-2.4). |
| 10.2-10.9 | (cut) | thin stream from the pattern's B side to the D rim in ~0.7 s; stem 0.15 D, notch on the B side. |
| 12.2 | final | heart 0.6 D, interior still shows the concentric arcs (unchanged by the cut except for V-shaping near the midline); white fraction 0.42. |

Gap widths: internal arcs 0.01-0.03 D, nothing wider than the notch (`b8` at y=0.6 D is the notch, `b35` at y=0.7 D is the gap between the lobes' tips at the stem).

## 4. Tulip (郁金香) - 20 s (clip starts mid-pour)

Overlay at 6.5-8.4 s (B/C/D line, "第二段再退回B点注入").

| t (s) | instruction | observable |
|---|---|---|
| 0-0.9 | (pouring at B) | brown; 0.9 s first white. |
| 1.0-2.2 | 看到白色的线条出现后开始摆动 | stripe fan 0.2 D. |
| 2.3-4.0 | 纹路形成半圆形后摆动 | semicircle of arcs 0.4-0.45 D, convex toward D. |
| 4.0-5.1 | 向前移动 | pitcher forward toward C while wiggling, stack extends to ~0.5 D. |
| 5.1-6.5 | 让纹路形成回包 | pitcher held near centre, keeps wiggling; the outer arcs wrap around the sides and close into a round base 0.5-0.55 D. |
| 6.5 | lift | |
| 6.6-8.5 | 第二段再退回B点注入 | at B, 1.3 s without white; the base is displaced ~0.05-0.1 D toward D and its B-facing arcs get compressed. |
| 8.5-10.6 | 摆动的同时往前推 | 2nd element appears (striped), is wiggled and pushed toward C; the base is now a thick crescent on the D side; brown arc gap 0.05-0.08 D between base and 2nd element. |
| 10.9-12.3 | 将底纹压开 | pitcher stays low at the centre with bigger flow: the 2nd element inflates radially, the base crescent is pushed out to r~0.3-0.35 D and the gap widens to ~0.1 D. |
| 12.4 | lift | |
| 12.4-13.5 | 第三段回到B点 | pour at B, ~1 s no white. |
| 13.5-14.3 | (3rd element) | small striped blob at B pushing the 2nd element down. |
| 14.4-15.3 | 注入一个爱心收尾 | 4th: a plain round blob at B (the top heart), big flow, no wiggle. |
| 15.3-16.0 | (cut) | thin stream from the heart through all elements to the D rim; each crescent acquires a pointed apex on the D side (tulip petal), the heart gets its notch. |
| 19.8 | final (`z_tulip_final.png`) | 3 nested crescents with internal ripples + top heart + stem; white fraction 0.28. |

Gap widths: y=0.5 D `W4 b7 W14 b4 W20 b5 W5` -> gaps **0.04-0.07 D** between stacked elements; x=0.5 D column `W8 b8 W13 b25 W10 b14` (the 25 % includes the stem region). Stripe lines inside elements 0.01-0.02 D.

## 5. Leaf / rosetta (树叶) - 17 s

Overlay at 0.1-1.5 s.

| t (s) | instruction | observable |
|---|---|---|
| 0-1.6 | 在b点注入流量 / 保持流量开始摆动纹路 | wiggle starts at ~0.8 s **before any white is visible**; 1.7 s first white. |
| 1.9-2.9 | | stripe fan 0.2 D. |
| 2.9-5.4 | 形成半圆形后向前摆动两下 | semicircle grows to ~0.5 D; two forward wiggles move the spout ~0.1 D toward C. |
| 5.5-8.0 | 然后再快速的往后退 | pitcher retreats C->B (~0.4 D in 2.5 s, 0.16 D/s) while wiggling ~2.5-3 Hz. Each wiggle leaves one chevron; chevrons are deposited behind the moving source (on the B side) and are convex toward D. The semicircle made earlier remains as the broad base on the D side. `z_leaf_retreat_15fps.png`. |
| 8.0-8.8 | 退到杯子的边缘 | spout at the B rim; the stripe band now spans 0.7 D. |
| 8.9-10.5 | 变小流量拉高收尾 | lift, thin stream at the tip (B end): the last blob becomes the rounded/heart-shaped tip. |
| 10.5-11.6 | (cut) | thin stream travels B->D along the midline in ~1.2 s; a brown midrib ~0.01 D wide appears, chevrons are dragged into V's pointing toward D, stem 0.15 D to the D rim. |
| 14.8 | final (`z_leaf_final.png`) | ~12 chevron pairs; white chevrons 0.03-0.05 D wide, brown lines between them **0.01-0.02 D**; the base at D side is a broad white region with faint arcs; white fraction 0.37. |

Gap widths (x=0.5 D column, tip to base): `W27 b1 W9 b1 W5 b2 W5 b1 W7 b2 W5 b1 W10 b7 W3` -> brown lines 1-2 % D, chevrons 5-10 % D; the only wide brown (7 %) is between the lowest chevron and the base.

## 6. Swan (天鹅) - 23 s

Overlay at 0.0-0.4 s.

| t (s) | instruction | observable |
|---|---|---|
| 0-1.6 | 在B点注入流量 / 开始摆动 | tilt, pour; wiggle from 0.9 s; first white 1.7 s. |
| 1.7-3.0 | | stripes. |
| 3.0-5.5 | 纹路形成半圆形后 / 向前摆动两下 | semicircle 0.5 D, two forward wiggles. |
| 5.5-6.5 | 稍微~倾斜一点 | cup tilted slightly more toward the pitcher. |
| 6.5-8.9 | 往后快速后退 / 退到杯子边缘 | fast retreat to the B rim with wiggle -> chevron body (first rosetta), ~0.6 D long. |
| 8.9-11.0 | 小流量拉高 | lift at the B rim, thin stream. |
| 11.0-12.6 | 沿着纹路的边缘收尾 | thin stream is moved **along the left edge of the body** from B to the lower-left, drawing a thin white line (lower neck) that hugs the pattern edge. |
| 12.7-13.9 | 再压低拉花缸 | pitcher lowered at the lower-left with big flow: a new white blob there (1 s delay again, then white). |
| 13.9-16.4 | 快速的往后退到B点附近 | second fast retreat with wiggle from lower-left back to B: a second chevron band (the wing) laid over/next to the body. |
| 16.4-17.6 | 大流量点出大白心拉高收尾 | big flow near the top: a plain blob (head) ~0.15 D; then lift and a thin stream draws the S-curve neck from the head down to the body (16.9-17.9). |
| 21.8 | final (`z_swan_final.png`) | body + wing chevrons (brown lines 0.01-0.02 D), neck = white band ~0.05 D with a thin brown line inside, head = heart-notched blob; white fraction 0.23. |

Gap widths (x=0.5 D): `W9 b10 W2 b10 W1 b2 W5 b2 ...` -> the only wide brown (10 %) is between head/neck and body; inside the wing 1-2 %.

---

## 7. How existing white moves when a new pour starts (the "push")

Seen directly in push_heart (two pushes) and tulip (three pushes), partly in swan (head vs body):

1. During the ~1 s white-free start of the new pour the old element is displaced away from the new impact by ~0.05-0.15 D. Its near edge (facing the impact) moves more than its far edge, so the element **thins radially and stretches azimuthally** into a crescent (push_heart: 0.33 D round -> 0.10 D thick, 0.5 D span). This is not a rigid translation.
2. The brown gap between old and new white is born in this interval: it is the brown surface swept ahead of the new source. Initial width ~0.15 D, final 0.04-0.13 D.
3. Fine internal stripes of the old element survive the push (tulip, swan) with the stripes on the B-facing side compressed; so the push is a smooth, nearly area-preserving deformation of the surface with sharp interfaces kept sharp.
4. Elements further than ~0.3 D from the impact barely move (displacement decays quickly with distance from the source). The brown annulus at the wall stays brown throughout.

## 8. Targets

* `/home/user/zfChen/latte_imex/targets/<p>_frame.png` are raw 480x640 final frames (heart 11.6 s, push_heart 16.4 s, layered 11.8 s, tulip 17.75 s, leaf 12.2 s, swan 20.5 s); `<p>.npz` holds a 192x192 (leaf: 256, `method='foam3'`) whiteness `w` in [0,1] + circular `mask`, **rotated into a canonical orientation** (`rot` = -125, -100, -130, -165, +45, 0 deg; `flip=False`). `preview.png` shows the ellipse fit (green), the warped disc ("y up") and `w`.
* `/home/user/zfChen/web/targets/<p>.png` (256x256) are the same `w` maps rendered two-tone (brown #5a3a28-ish / cream) with soft edges; they are **not** photos. Because of the per-pattern rotation the heart target has its stem at the top and the lobes at the bottom, i.e. the opposite of the video frame orientation (stem lower-left). Anyone comparing simulated frames to these targets must apply the same `rot`.
* The whiteness maps blur the thin brown lines: in `w` of leaf/swan the 1-2 % D brown lines are grey (~0.3-0.5), not 0; in the push_heart `w` the two crescents and the small heart are at ~1.0 while the 0.05 D gap is ~0.1. Any loss that is dominated by area overlap will reward smooth blobs; the gaps and lines carry little weight.

---

## 9. Must-reproduce phenomena, ranked by how many of the six clips show them

| rank | phenomenon | clips | concrete numbers |
|---|---|---|---|
| 1 | **White appears only after a delay**, then as a ring *around* the impact; the impact spot itself stays brown while the spout is low. | 6/6 (every pour stage: 9/9 stages) | delay 0.9-2.1 s (median ~1.3 s); brown core 0.05-0.12 D; C-shape opening toward the spout. |
| 2 | **Crisp interfaces that persist**: white/brown boundary <=0.01 D wide, unchanged for >10 s; stripes 0.02-0.04 D wide stay separated by 0.01-0.02 D brown lines. | 6/6 | no measurable blur between t=4 s and t=20 s. |
| 3 | **Brown annulus at the wall** 0.10-0.25 D stays brown; total white fraction 0.23-0.42 at the end. | 6/6 | |
| 4 | **Final thin-stream cut** B->D: notch at entry, stem 0.1-0.2 D x 0.01-0.02 D to the rim, thin brown line along the path, lateral bulge/V-shaping of what it crosses; done in 0.7-1.2 s. | 6/6 | speed 0.3-0.6 D/s |
| 5 | **Radial growth of a poured blob** ~0.1 D/s in radius with the far edge nearly stationary once >0.3 D from the source (growth is on the source side). | 6/6 | heart: 0.3 D at 2.3 s -> 0.5 D at 4.6 s -> 0.65 D at 6.7 s. |
| 6 | **Wiggle stripes**: spout oscillation ~2.5-3 Hz, amplitude ~0.1 D, gives one arc per half-cycle, arcs convex away from the spout, spacing 0.025-0.04 D, tighter far from the source. | 5/6 (all but heart) | |
| 7 | **Push with brown gap**: a new pour displaces older white away, thins it into a crescent, and opens a brown gap of 0.04-0.13 D that is created *before* the new white appears and then persists. | 3/6 (push_heart x2, tulip x3, swan x1) | gap initially ~0.15 D, final 0.05-0.1 D |
| 8 | **Forward motion while wiggling / "回包"**: the stack extends toward D and closes into a disc; arcs wrap laterally. | 3/6 (layered, tulip, swan/leaf partially) | |
| 9 | **Fast retreat while wiggling** -> chevrons narrowing toward the tip. | 2/6 (leaf, swan x2) | retreat 0.16 D/s, 0.4-0.6 D |
| 10 | **Thin-stream drawing** of a white line along the pattern edge (swan neck) - a high thin stream *does* deposit a narrow white band ~0.05 D when moved slowly. | 1/6 | |
| 11 | Cup tilt at start, righted while pouring. | 6/6 | 15-25 deg, over 1-2.5 s |

Items 1, 4 and 7 are the ones neither closure reproduces according to the task brief; note that 1 and 7 are the same mechanism seen twice (brown is pushed ahead of the source for ~1 s before white emerges, so any new source first clears a brown zone of ~0.15 D around itself), and 4 requires a thin high stream to drag a 0.01-0.02 D line through existing white without whitening it.
