# Skeptic notes — Claim A20

**Claim A20:** 所有拟合在 96²；学长的收敛/多初始化/不按测试图选参协议未执行；咖啡师 bug 曾让两轮扫描全部郁金香在"细切没奶"下评分。
**Evidence given:** notion_z1 §2.5；docs_map C20.

Verdict: **confirmed** (three sub-claims; each traced to records, code or a reproduction). Nuances below.

---

## 1. "所有拟合在 96²"

Checked the run records on disk (not the notes).

| fit | script / default N | record | N found in record |
|---|---|---|---|
| E5 single-pattern CMA (V0.5 physics, actions only) | `optimize.py` (`--N` default **128**, L195) | `runs/opt/v05phys_baseline/heart.log`, `push_heart.log` | log line `[heart] n=13 popsize=11 N=96` |
| E6 joint 41-dim CMA | `optimize.py --joint` | `runs/opt/joint_v1.log` | `[joint ['heart','push_heart']] n=41 popsize=15 N=96 workers=4` |
| E17 sweep round 1 (100 LHS) | `joint_sweep.py` (`--N` default 96, L163) | `runs/joint_sweep/results.json` | `N: 96` (all 100) |
| E18 sweep round 2 (60) | same | `runs/joint_round2/results.json` | 96 (all 60) |
| E20 round-2 rerun after barista fix | same | `runs/joint_sweep_r2fix/results.json` | 96 (all 60) |
| E28 two-layer constants LHS | `twolayer_fit.py` (default 96, L33) | `runs/action_opt/twolayer_fit_N96.json` | 96 |
| E28 action re-opt (old vs twolayer) | `action_opt.py` (default 96, L204) | `search_N96.json`, `refine_N96.json`, `fit_search_N96.json`, `fit_refine_N96.json` | 96 |
| E29/E30 cell refit (gap-aware) | `cell_test.py --fit` (default 96, L106; reads `twolayer_fit_N96.json`, `fit_refine_N96.json` hard-coded, L109–110) | `cell_fit_N96.json`, `cell_test_N96.json` | 96 |
| E31 25-dim joint CMA (cell) | `joint_cma.py` (default 96, L60) | `runs/action_opt/joint_cma/history.json` (src `fit_refine_N96.json`) + `jcGG_KK/*_N96.json` | 96 |
| v2 32-dim joint CMA | `joint_cma.py --spec v2` | `joint_cma_v2`, `joint_cma_v2w`, `joint_cma_v2s` dirs: every per-run file is `*_N96.json` | 96 |
| v2 per-pattern (h,q,s) polish | `v2_actions.py` (default 96) | `v2_actions_N96.json`, `v2s_actions_N96.json`, `v2best/*_N96.json` | 96 |
| push-heart staged action CMA | `stage_actions.py` (default 96, L86) | `runs/action_opt/stage_old_push_heart_cma` | (default) 96 |

`action_opt.py` has a `--verify` path for re-running the winner at `--N 128` (L4, L211), but `find runs -name '*N128*' -path '*action_opt*'` returns nothing — never used.

The only N ≠ 96 records are **forward re-checks, not fits**: `runs/codexmap/r2fix_{3,5,7,10}_N256` (E21 top-3 at 256²), `runs/codexmap/gapdiag/*_N256`, `base_N512.json` (E23, 512², loss 0.5516, 1434 s), `runs/codexmap/delay/*_N128|N256` (E27 patch trials), `runs/final/*/metrics.json` N=256 (rendering), `runs/cp_sweep.json` N=128 (E1/E7 sensitivity, not a fit).

The repo's own doc states it outright: **LATTE_MODEL.md L269** `| N | 96²（拟合）/ 256²（出图） | 4 mm 褐缝在 96² 上只有 5 格 |`. And `action_opt.py:167` labels the two-layer constants as "fitted on the six V0.5 intents (LHS, 96^2)".

→ Sub-claim 1 holds without exception. (At 96², D_L = 8 cm, one cell = 0.83 mm; the senior's own resolvability threshold is 4/384 D ≈ 0.83 mm, i.e. exactly one 96² cell.)

## 2. 学长协议未执行（收敛 / 多初始化 / 不按测试图选参）

Senior's protocol (notion_senior.md L35–43, L317):
- Calibration A = tulip, B = leaf; **frozen test** = heart, push_heart, layered_heart, swan. "不得用测试图案选择最佳参数初始化、追加物理模块、修改阈值或修正时间对应。"
- Convergence: freeze parameters/actions, refine space/time/sub-pixel in three tiers (128/256/512, dt/dt/2/dt/4, 1×1/2×2/4×4).
- Multi-init: "首轮至少记录 3 组确定性初始化及相同计算预算；按校准目标和预定并列规则选模型，不看测试挑解。"

What the code does:
- **Calibration/test split:** `action_opt.py:17` `PATTERNS = ["heart","push_heart","layered_heart","tulip","leaf","swan"]`; `cell_test.py` fit loop (L113–114) iterates `for n in PATTERNS` and takes `J = mean over all six` (L120); `joint_cma.py` jobs are `for n in PATTERNS` (L104) and `J = mean(per.values())` (L109). `twolayer_fit.py` likewise (six intents). So physics constants and the "winner" model were selected on the mean over all six patterns, including the senior's four frozen test patterns. The only two-pattern fits (E6 joint, E17/E18 sweeps: heart + tulip) use heart, which is a *test* pattern in the protocol. No held-out evaluation anywhere (`grep 留出|held` over conv_part*.md: no hits; notion_z1 L144/L251: 留出检验 "尚未开始").
- **Convergence:** no three-tier study exists. Records show at most two grids on v1 #9 (256 vs 512, E23 — `gapdiag/base_N256.json` vs `base_N512.json`: loss 0.555→0.552, stripe count 19→23, median width 2.5→2.0 mm — i.e. the fine-stripe statistics *did not* converge) and a 96→256 "复核" of top-3 (E21). No dt refinement, no sub-pixel tiers; E1 self-admits "没有时间步和网格收敛检查" (A:L133). For v2 there is **no** record at any N other than 96 (the scratchpad `v2cmp_*_96.npy` are also 96²). The repo docs quote the senior's 0/3/21 non-convergence result (POUR_MODEL_SUMMARY L55–59, TWOLAYER L87) but do not report one of their own.
- **Multi-init:** `joint_cma.py` takes a single `--seed` (default 5, L68) and a single `x0`. The three v2 runs are not three deterministic inits of the same problem with equal budget: `joint_cma_v2` (src `neutral`, 16 gens, best gen 14, J 0.551), `joint_cma_v2w` (src = `_stale/v2_nobulk/joint_cma_v2/history.json`, i.e. warm-started from a run on the bulk-flow-bug model, 12 gens, J 0.563), `joint_cma_v2s` (src `v2_struct_init.json`, different objective `--objective struct`, J_A 0.974). conv_part3 L271 self-assessment: v2 had ~220 candidates in 32 dims — "远远不够收敛". v1 "#9" closure was "picked by eye for structure" from two sweeps (conv_part3 L271), not by a pre-registered tie rule.

→ Sub-claim 2 holds. Nuance: it is not that *nothing* multi-grid was ever done — E21/E23 are two-grid checks of one v1 configuration — but none of the three protocol items (three-tier convergence, ≥3 equal-budget inits with a pre-set selection rule, frozen test set) was executed for either closure.

## 3. 咖啡师 bug → 两轮扫描全部郁金香在"细切没奶"下评分

Timeline from disk and git (`latte_imex` repo):
- `runs/joint_sweep/results.json` written 10-08 22:45 (round 1, 100 samples); `runs/joint_round2/results.json` 10-08 23:05 (round 2, 60 samples).
- Fix commit `7e0bc51` 10-08 **23:16** "Barista: hold the tilt just below onset while stopped (short petals and the cut were starved)". Second fix `37d1c95` 10-09 10:20 (feed-forward wrist + height rate limit; swan head/beak).
- `runs/joint_sweep_r2fix/results.json` 10-08 23:21 (round 2 rerun after fix).

So both rounds ran on the pre-fix barista and only round 2 was rerun. Current code: `pitcher.py:205` `flowing: bool`, L454–458 hysteresis on `h_max` vs `h_on/h_off`, L704–707 `rate = -self.swing_rate if st.flowing else -self.hold_rate` (hold_rate 4°/s, L623), L711–713 feed-forward comment "Without it a 0.25 s head or beak pour never reaches its flow (PI alone moves 6 deg/s)".

The sweep records do **not** store per-phase arrival flow (keys of `patterns.tulip`: loss, ok, elapsed, retries, white, l1, iou, area_diff, c), so "全部" cannot be read off the results files. I therefore reproduced it: extracted the pre-fix tree (`git archive 513b1c3`, the commit the sweeps ran on) and the post-fix tree (`7e0bc51`), and drove **only the pitcher/barista chain** (`BaristaScript` + `joint_sweep.tulip_moves`, kQ from each sample, fixed 1/120 s clock, no solver) on the actual round-1 and round-2 tulip actions, averaging `Q_hit` over each named phase (script: `scratchpad/a20/check_barista.py`).

Pre-fix (first 3 of each round):
```
joint_sweep  0  base hold 13.0  petal2  7.5  petal3 0.77  cut 0.00 mL/s
joint_sweep  1            13.2         15.7         1.22      0.00
joint_sweep  2            15.7          4.9         0.69      0.00
joint_round2 0            10.9          9.7         0.96      0.00
joint_round2 1             9.9          8.6         1.15      0.00
joint_round2 2             9.9         11.6         1.19      0.00
```
Post-fix (same samples): petal3 5.1–12.0, cut 2.0–7.1 mL/s. (Base-hold also changed for some samples, 13→26 mL/s, so commit 7e0bc51 altered more than the restart branch — the "fix" is not a pure bug fix.)

Larger pre-fix batch (15 per round = 30 samples): see §3b below (filled from `scratchpad/a20/pre30.log`).

These numbers match the Notion figures (petal 3 ≈ 1.1 mL/s, cut 0.0 → 9.9 / 5.4 after fix; here 1.0 / 0.0 → 8.3 / 5.0 mean) and the mechanism is deterministic in the controller (restart after every 0.18 s zero-flow reposition went through the PI branch because Q never read exactly zero), so every tulip in both rounds had a cut phase with 0.0 mL/s arriving on the coffee.

→ Sub-claim 3 holds, with the caveat that the claim's "曾" is correct: round 2 was rerun post-fix (E20, `joint_sweep_r2fix`) and the later #9 closure and all E21+ results come from the rerun. Round 1 (100 samples) was never rerun. Note also E20's finding that the fix made tulip end-frame loss *worse* (median 0.58→0.72) — the scoring, not just the barista, was at fault.

### 3b. 30-sample pre-fix batch result
(appended after the background run finished — see bottom of file)

## 4. What the evidence given does and does not support

- notion_z1 §2.5 is a secondary summary of conv [239]/[243] (conv_part2 L99), where the assistant itself wrote "两轮扫描的所有郁金香都是在'细切没奶'的条件下评的". The Notion page does not contain a per-sample proof; the reproduction above supplies it.
- docs_map C20 ("全部拟合限于 96²…从未在 256² 上拟合") is confirmed by every fit record on disk and by LATTE_MODEL.md L269 itself.
- The "learned-protocol not executed" part is not in §2.5/C20 at all; it is supported by notion_senior.md L35–43/L317 vs the code's `PATTERNS`/`seed` handling and the absence of convergence/held-out records (notion_z1 L144, L251 "尚未开始").

## 5. Severity note

The diagnosis files A20 as [minor]. Two of its pieces are not minor for the project's conclusions: (i) all closure comparisons (old #9 vs two-layer vs v2) were ranked on a grid where one cell equals the senior's resolvability limit, and E21 showed structure "96² 看不到的在 256² 都出来"; (ii) all closure constants were selected on the senior's test patterns, so no reported loss is a test loss.

---
### 3b (result). 30 pre-fix samples (15 from round 1, 15 from round 2), pitcher chain at commit 513b1c3
`SUMMARY n=30  cut Q_hit mL/s: mean 0.01, max 0.04, n(<0.5 mL/s)=30/30;  petal3: mean 1.06, min 0.50`
Every one of the 30 tulips had an effectively dry cut (≤0.04 mL/s arriving on the coffee) and a starved petal 3 (≈1 mL/s against a commanded 0.6·petal_Q ≈ 6–14 mL/s). Same samples post-fix (7e0bc51): cut 2.0–7.1 mL/s, petal 3 5.1–12.0. The mechanism is in the controller, not the sampled actions, so extrapolating to all 160 samples is safe. Log: `scratchpad/a20/pre30.log`; script: `scratchpad/a20/check_barista.py`.
