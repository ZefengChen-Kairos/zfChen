# Skeptic check of A18 — "动作空间错：V0.5 脚本翻译×(h,q,s)，推心脚本画的是另一种图，每图各拟动作=标定非预测"

Verdict: **confirmed with a factual correction**. The code side of the claim (what the action space is, that
(h,q,s) cannot change the script structure, that every fit used per-pattern actions and never the senior's
calibrate-A+B/freeze protocol) is fully traceable. The *description of the video* that the claim inherits from
conv [545] is wrong on one point that it treats as decisive: the push_heart video **does** end with a
through-cut / stem drawn from the inner block through both crescents to the rim. The V0.5 script differs from
the video, but not in the way [545] said.

Checks: `v05_controls/push_heart.json` dumped and translated with `pitcher.v05_moves` (a `python -c` call,
no simulation); `action_opt.scaled_moves`, `joint_cma`, `v2_actions`, `stage_actions`,
`push_heart_video_script.py` read; `video_obs.md` §2 and the frames `review/frames/z_push_cut.png` (t = 13.3 s)
and `z_push_final.png` (t = 16.2 s) inspected; `notion_senior.md` §1.3 and §4.1.

## 1. What the code DOES

**a. Six-pattern action space = V0.5 control file → `v05_moves` → three global factors.**
`action_opt.py` docstring L4–8 and `scaled_moves` L29–45: `s` scales every start/end/Bezier/wobble about the
cup centre (clip ±0.45); `h` rescales `z` and `q` rescales `Q` only for moves with `Q0 > 0 and z0 < 0.05`
(clip 0.8–5 cm); the move-away keeps its geometry. `joint_cma.py` L6/L27/L44–45: x ∈ [0,1]^25 = 7 (or 14 for
v2) physics constants + 6 × (h,q,s) with log bounds h 0.3–2, q 0.6–2, s 0.8–1.3. `v2_actions.py`: per-pattern
coordinate polish of (h,q,s) with the physics frozen. LATTE_MODEL.md §6.3 (L276–277) documents exactly this:
"动作参数（每个图案三个，与物理参数联合优化）… 用 Point2Pose / AprilTag + 电子秤实测动作后，这三个量不再是自由度".
So "V0.5 脚本翻译 × (h,q,s)" is literally what the fits used. Confirmed.

**b. The V0.5 push_heart script** (`v05_controls/push_heart.json`, `source: manual_synthetic_observed_inlet`,
i.e. a Codex guess, not a measurement; total duration 5.09 s). Translated moves (my dump):

| # | dur s | path (cup units, −y = forward/D side) | z cm | Q mL/s | kind |
|---|---|---|---|---|---|
| 0 | 1.25 | (0,0.14)→(0,0.015) | 2.0 | 25.5 | pour |
| 1 | 0.18 | (0,0.015)→(0,0.20) | 3.0 | 0 | zero-flow bridge |
| 2 | 1.10 | (0,0.20)→(0,0.104) | 2.1 | 22.5 | pour |
| 3 | 0.18 | →(0,0.26) | 3.0 | 0 | bridge |
| 4 | 0.75 | (0,0.26)→(0,0.20) | 2.4 | 19.5 | pour |
| 5 | 0.18 | →(0,0.26) | 3.0 | 0 | bridge |
| 6 | 1.10 | (0,0.26)→(0,−0.30) | 7.1→9.0 | 5.4→0.5 | fine cut through everything |
| 7 | 0.35 | hold | 9.0 | 0 | stop |

Three pushes on one line whose y-intervals [0.015,0.14], [0.104,0.20], [0.20,0.26] overlap/abut (stage k+1
starts *behind* stage k, each one shorter), no per-stage lift (bridges are 0.18 s at 3 cm, `v05_moves`
L800–801 "hold at least 3 cm"), then one cut from behind the last block to the far rim. "三段重叠 + 贯穿切" is
an accurate description of the file. Confirmed.

**c. (h,q,s) cannot change that structure.** `s` is a uniform scale of all points on the same line (overlap
and ordering preserved); the cut starts at z = `height_from_radius(0.0038)` = 7.1 cm ≥ 5 cm, so neither `h`
nor `q` touches it (L39–43 condition `m.z0 < 0.05`); no knob inserts a lift or moves a start point back to B.
Confirmed by reading; no numeric check needed.

**d. Per-pattern actions, no freeze protocol.** `joint_cma.py` L102–108: each candidate physics is run on all
six `PATTERNS` with that candidate's own six (h,q,s) triples and J = mean over the six — physics and actions
are fitted on the six targets simultaneously. `joint_sweep.py` L175–176/188: candidate ranking by
`tulip.loss + heart.loss`, not tulip + leaf; `results.json[9]` was "picked by eye" (code_map). `action_opt`
search/refine (L101–144): per (model, pattern) grid on (h,q,s) with the physics frozen — the closest thing to
the protocol, but the actions are fitted to the very target they are scored on. `stage_actions.py`: 20 free
action parameters on push_heart alone under J_A. Nothing in the repo or the notes runs "calibrate on 压纹郁金香 +
树叶, freeze, score 大白心/推推乐/千层心/天鹅 with actions taken from the video" (notion_senior §1.3 L36–38;
notion_z1 L144/L251 "项目主流程 … 尚未开始"; conv part2 O-7: heart-fit/tulip-hold-out "promised at least five
times, not implemented"). Confirmed.

## 2. Where the claim's video description is wrong

The claim (and docs_map C18, diagnosis A18) copies [545]: "视频三次独立推、各自抬缸、**无贯穿**". Independent
frame reading says otherwise:

* `video_obs.md` §2, 13.3–14.5 s: "小流量拉高收尾 — lift; **thin stream drawn from the inner blob through both
  crescents to the D rim** (stem at lower-left, `z_push_cut.png` t = 13.3, `z_push_final.png` t = 16.2)". I
  looked at both frames: at 13.3 s the pitcher is high and a thin stream is being drawn from the inner element
  toward the lower-left rim; at 16.2 s a thin white stem runs from the inner heart through the brown gap and
  the outer crescent to the rim.
* The senior's own segmentation (notion_senior §4.1, row 02 推推乐): "58.0–60.3 **小心形与贯穿收尾**".
* Stage 3 is also not "一小块 at B, 抬缸": video_obs 11.2–13.3 s = small blob at B (left behind as the detached
  little heart), then pitcher moves B→C with big flow building a third blob inside the inner crescent, then
  the lift-and-draw.

So the single structural element that [545] singled out as the thing V0.5 gets wrong — the final through-cut
— is present in **both** the V0.5 script (move 6, 0.26 → −0.30 at 7–9 cm, 5 → 0.5 mL/s) and the video. The
repo's "video script" (`push_heart_video_script.py`: stage 3 = 0.9 s at B then lift, no cut; docstring "no
through-cut at the end") and `stage_actions.SPEC["push_heart"]` (docstring L8–9 "no through-cut") encode the
same misreading, so the post-[545] push_heart runs are *also* not the video's actions (positions "从视频叠加的
标注上估的，不是测量值", [545]).

The real differences between the V0.5 file and the video are instead:
1. start points recede (0.14 / 0.20 / 0.26) vs always back to B (0.22) — in the video the new jet lands
   *behind* the old block at the same spot each time, which is what opens the gap (video_obs §2 7.2–8.0 s);
2. no per-stage lift: 0.18 s zero-flow bridge at 3 cm vs 1.5 s (4.4–5.9 s) / 0.2 s lifts with a trickle;
3. timing: stages 1.25 / 1.10 / 0.75 s, total 5.1 s, vs ≈4.3 / 3.7 / 3.3 s, total 17 s, each pour with
   0.9–2.1 s of white-free impact (video_obs §1 "delay before white");
4. flow: 25.5 / 22.5 / 19.5 mL/s constant vs "大流量" on stage 2 and a 1–2 s low-flow start.
These are enough to justify "画的是另一种图" without the through-cut argument.

## 3. Sub-claim "所有推心评估都在错误动作上"

Strictly true in the weak sense that no push_heart evaluation used actions measured from the video: V0.5 ×
(h,q,s) before [545]; a hand-written approximation that drops the final cut after [545]; a 20-parameter free
search (θ = −23°) under J_A. It is **not** true that the pre-[545] runs are wrong *because of a through-cut
the video lacks*. Also note the pitcher chain adds its own distortion on top of the script (code_map §5 item
12: 30–57 mL/s bursts during the 0.18 s repositions, then 1–2 mL/s for the last 0.5 s of stage 2), so even a
correct intent is not what the solver received.

## 4. Sub-claim "每图各拟动作 = 标定非预测"

Confirmed as a statement about what was done (§1d). Nuance in fairness: the actions were never measured
(V0.5 controls are guesses), so *some* per-pattern action freedom is unavoidable, and LATTE_MODEL §6.3 states
openly that (h,q,s) are temporary free parameters. The defect is that the **physics** was fitted at the same
time on all six targets (`joint_cma` mean-J) and the resulting six losses were discussed as if they measured
the model (conv [528]/[530]), while the protocol the senior wrote (calibrate on 2, freeze, test 4 with video
actions) was never run. With 3 free action scalars per pattern and a loss that is area/overlap-dominated
(A17), the per-pattern (h,q,s) absorb area errors (tulip h = 2.0 at the bound, code_map L142; layered-heart
gaps filled by h×0.41, q×1.5 — skeptic_A17).

## 5. Corrected statement

The fits optimise V0.5's guessed control files scaled by three global factors (h,q,s) per pattern (log bounds
0.3–2 / 0.6–2 / 0.8–1.3); the physics is co-fitted on all six targets (`joint_cma`) or frozen from a sweep
ranked by tulip+heart (`joint_sweep`), so every reported loss is a calibration score and the senior's
calibrate-(郁金香+树叶)/freeze/test-4 protocol has never been executed. For push_heart the V0.5 file
(5.1 s; pushes 0.14→0.015, 0.20→0.104, 0.26→0.20 on one line; 0.18 s/3 cm bridges; cut 0.26→−0.30 at
7–9 cm) differs from the 17 s video in receding start points (video: always back to B), absent per-stage
lifts, and 3× shorter stages with no white-free impact phase; (h,q,s) cannot change any of these. **Both**
script and video end with a thin draw-through from the last block to the rim ("贯穿收尾" in the senior's
segmentation; `z_push_cut.png`/`z_push_final.png`), so [545]'s "最后没有一刀贯穿" and the repo's video script /
stage spec that omit the cut are a misreading. Consequently *no* push_heart run (V0.5-scaled, video script or
stage search) has used the video's actions, and the three-component claim should be re-tested with a script
that includes the final cut and the B→C stage-3 push.

---

## 6. Independent re-verification (second skeptic pass, 2026-10-11)

Everything above was re-checked from the primary sources rather than taken on trust:

* `action_opt.scaled_moves` (L29–45) re-read: `s` is a uniform scale about the centre with clip ±0.45; `h`/`q`
  apply only when `m.Q0 > 0 and m.z0 < 0.05`. `python -I -c` dump of `pitcher.v05_moves('push_heart')` gives
  exactly the table in §1b (cut = move 6, z 0.071→0.090 m, Q 5.4→0.54 mL/s; bridges Q = 0 at 0.03 m), so the
  cut and bridges are untouched by any of the three factors — verified numerically, not just by reading.
* `joint_cma.py` L27 `ACT = dict(h=(0.3,2.0), q=(0.6,2.0), s=(0.8,1.3))`, L44–47 `decode` → 6×(h,q,s), L102–108
  mean-J over `PATTERNS` (all six) per candidate physics; `v2_actions.py` STEP coordinate polish per pattern with
  physics frozen; `joint_sweep.py` L175–176 ranks by `tulip.loss + heart.loss`. No script in the repo calibrates
  on (tulip, leaf) and freezes for the other four; notion_z1 L144/L251 "项目主流程 … 留出检验 … 尚未开始".
* `LATTE_MODEL.md` §6.3 (L276–277) states the (h,q,s) design and that they are a stop-gap until actions are
  measured — so the "action space" is documented as provisional, not hidden.
* conv [545] (conversation_part3.md L2274) literally says "整个过程最后没有一刀贯穿". Frames `z_push_cut.png`
  (t = 13.3 s: pitcher lifted high, thin stream at the inner heart) and `z_push_final.png` (t = 16.2 s: thin
  white stem from the inner heart through the outer crescent to the lower-left rim) show the opposite;
  notion_senior §4.1 row 02 "58.0–60.3 小心形与贯穿收尾" agrees. `push_heart_video_script.py` (docstring "no
  through-cut at the end") and `stage_actions.SPEC["push_heart"]` (L8–9) carry the misreading into the
  post-[545] experiments.
* One softening of §2 item 1: V0.5's receding start points (0.14/0.20/0.26) are a crude stand-in for "return to
  B but the old block has already moved toward D" (video_obs §2, 6.0–7.2 s: old blob displaced ~0.1 D before
  the new white appears), so the start-point difference is less decisive than the missing per-stage lifts,
  constant 20–26 mL/s flow, and 3× compressed timing.

Net: A18's code-side mechanism (V0.5 × 3 global factors, per-pattern co-fit, no freeze protocol) is confirmed;
its video-side evidence inherits one factual error from [545] (the video does have a final draw-through). The
corrected statement in §5 stands.
