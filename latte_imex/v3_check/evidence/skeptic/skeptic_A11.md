# Skeptic check of CLAIM A11

**Claim (diagnosis.md L71):** 牵引目标 v* = κ_t u_in 把推力全部挂在"奶柱水平速度"上，而这个量在 V0.5 时代是 ±1.4 D/s 的合成摆动，物理壶链只有它的 1/7；整套 (B_dep, κ_t, κ_r, kQ) 是围着 Codex 不可实现的入口（0.6 mm 奶柱需 48 m/s、85% 锁相脉动）调出来的，换物理入口后花瓣消失。

**Verdict: refuted** (as stated, for the current code). The historical core is true for θ05 at [174]–[219]; the three load-bearing assertions about the *current* closure are wrong, and the "1/7" is an artefact of a coupling constant that was removed on 10-08.

---

## 1. What the code does (mechanism check)

- `solver.py` L325–349 (closure `"chi"`): target velocity
  `vsx = P.kappa_t * inlet.u_in[0] + rad*dx/den`, `vsy = P.kappa_t * inlet.u_in[1] + rad*dy/den`, with
  `rad = chi*kappa_r*U_perp` (v1) or `kappa_r*V_R` capped (v05). Traction weight `Lam = kappa_c*(1-chi)*S_eff*K` (L336).
  So yes: the only non-radial push is κ_t·u_in inside the footprint kernel. Mechanism part of the claim holds.
- `Inlet.u_in` (L101) = "horizontal material arrival velocity [D_L/s]".
- V0.5 replay (`v05_program.py` L103–116): `jet_x = p["jet"][0] + jet_amplitude_x*sin(2π f t + φ)`; tulip base has
  `jet=[0,-1.2]`, `jet_amplitude_x=1.4`, `jet_frequency=2.5` (`v05_controls/tulip.json`), then capped at
  `U_perp*tan(60°)` with `U_perp = Q/(π r²)`, Q = flow/κ_Q. The "±1.4 D/s 合成摆动" is literally in the control file. Confirmed.
- Physical chain (`pitcher.py` L294–331): `u_h = v0[:2]`, v0 = Torricelli exit speed along the spout **plus the pitcher's own velocity** (`carry_velocity=True` default, L186/L304). The coupling then does `u_in = c_u * u_h`.

## 2. Where "物理壶链只有它的 1/7" comes from — a retired coupling constant, not physics

`pitcher.py` L537–544, old `Coupling()` defaults: `c_S = 0.05/15e-6`, `c_U = 0.263/0.70`, **`c_u = 0.5/0.15 = 3.33 /m`** ("0.5 D_L/s at 0.15 m/s"). Physical conversion is `1/D_L = 12.5 /m`, i.e. the old coupling divided u_in by **3.75**.

Numerical check (script `scratchpad/a11_check.py`, V0.5 layered-heart intent `layered_heart_moves(axis="x", amp=1, freq=2.5)` = 3.6 mm lateral wiggle @ 2.5 Hz, sampled at 240 Hz over the two wiggle moves):

| coupling | u_in lateral amp (p97.5) | u_in fore-aft mean | |u_in| mean | U_perp |
|---|---|---|---|---|
| old `Coupling()` (c_u = 3.33) | **0.18 D/s** | −0.40 D/s | 0.41 | 0.24 D/s |
| physical (c_u = 1/D_L) | **0.68 D/s** | −1.55 D/s | 1.59 | 8.06 D/s |
| sweep-winner wiggle 6.5 mm @ 3.5 Hz, physical | **1.83 D/s** | −1.48 D/s | 1.83 | 8.13 D/s |

1.4 / 0.19 = **7.4** — that is exactly the "七分之一" of conversation [174]/[183]/[185] (conversation.md L2057, L2150, L2175), computed with the old c_u. With the dimensional coupling the same wiggle gives 1/2 of Codex's ±1.4, and the round-2 winners' 5–8 mm / 3–4 Hz wiggle gives ±1.8 D/s, *more* than Codex. The steady component |u_h| of the physical chain is 1.2–2.0 D/s (`codex_phases.py` run: tulip base 1.90/1.97, petal 2 1.56, cut 1.42 D/s; heart disc 1.64), comparable to Codex's κ_t|v∥| targets 1.0–1.3 D/s (PITCHER_MODEL §13 table, needs κ_t 0.5–0.9 to match).

The old c_u was dropped at Z1 §8.10 ("耦合改成纯量纲换算（杯径 8 cm，去掉 c_S、c_U、c_u 三个倍率）"). Every current driver uses the physical coupling: `joint_sweep.py` L89, `joint_run.py` L26, `action_opt.py` L74, `delay_test.py` L16, `topview_movie.py` L28, `codex_phases.py` L21 all pass `c_u=1.0/D_L, footprint="physical"`. The only remaining `Coupling()` default is the fallback in `pitcher.py` L848 (`coupling or Coupling()`), and `run.py` L152 builds from CLI kwargs. docs_map L106 / PITCHER L134–135, L155–157 still carry the "1/7" sentence — stale doc text from 10-08, not what the code does now.

## 3. "整套闭合是围着 Codex 入口调的" — true for θ05, false for the closure actually in use

- θ05 (`v05_controls/common_parameters.json`, `Params` defaults `solver.py` L23–32: κ_Q 125, κ_t 0.7, κ_r 0.3, B_dep 240) is Codex's set, tuned to Codex's effective inlet. That part is right, and the `Params` dataclass defaults still are θ05.
- But the closure the project runs with since 10-08 is **#9** (`action_opt.py` L23–26 `physics9()` reads `runs/joint_sweep_r2fix/results.json[9]`): cp 0.97, β 2.41, ν 1.44e-3, κ_c 0.66, **κ_t 0.905, κ_r 0.0085, B_dep 1719, p_dep 1.56, kQ 1.33**. It was selected by a 100 + 60 sample Latin-hypercube sweep in which *every* inlet came from the physical pitcher chain (`joint_sweep.py` L5–9, L22–27, L89; ranges B_dep 300–10000, κ_t 0.1–1, κ_r 0.003–0.1, kQ 0.8–3; wiggle 2–8 mm, 2–5 Hz, direction lateral→fore-aft). Codex's inlet never enters that sweep. Round2fix tulip top-8 (results.json): B_dep 1160–3350, κ_t 0.61–0.91, κ_r 0.003–0.018, kQ 0.98–1.45, wiggle 5.2–7.5 mm @ 2.5–3.8 Hz.
- `codex_phases.py` / PITCHER §13 is a *post-hoc* consistency check (dimensionless groups U²/(B d), κ_Q Q, κ_r V_R, κ_t|v∥| mapped onto the physical inlet → B_dep 1200–7100 GM ≈ 2100–2800, kQ 1.0–2.6, κ_t 0.5–0.9), not the source of the sweep values; κ_r disagrees by 5–10× (0.03–0.07 mapped vs 0.004–0.013 sweep; conv_part2 C-4), which is itself evidence that the sweep was *not* anchored to Codex.

## 4. "换物理入口后花瓣消失" — true at [183]–[219] with θ05 + old coupling, false afterwards

- At [219] (Z1 §8.9) the physical chain was run with θ05 (B_dep 240) and the old c_u, and gave "只是一个盘". With B_dep 240 and the physical U²/d ≈ 1000–1200 (codex_phases output), χ = 1/(1+(U²/(240 d))²) ≈ 1/(1+(4.5)²) ≈ 0.05 even in the base: almost nothing deposits. That is a parameter mismatch (B_dep = g′·Fr_c² is a dimensional constant tied to the inlet scale, Z1 §8.11 / PITCHER §13), not a closure "built around" an unrealisable inlet.
- After re-scaling + the barista starvation fix (`pitcher.py` hold_rate 4°/s, restart on `flowing`), `results/joint_sweep/round2fix_top3_N256.png` (viewed) shows, for #9/#5/#7 with the physical inlet at 256²: stacked tulip petals separated by brown seams, a top heart and a stem; hearts with notch and tail. Petals did not "disappear" with the physical inlet; they disappeared with the physical inlet **plus Codex's numbers plus the old coupling**.
- What stays missing with any inlet is the ≥4 mm brown band (PITCHER §13 last paragraph, gap_diagnosis: physical #9 median 2.2 mm; Codex 768 1.5–4 mm; footprint ×0.3 and 85 % pulsation tested, "脉动几乎无影响 2.2→2.8 mm"). So the Codex-only inlet features (sub-mm footprint, locked pulsation) are *not* what the closure's output depends on — the opposite of the claim.

## 5. The "0.6 mm 奶柱需 48 m/s / 85 % 锁相脉动" evidence

- Arithmetic is right (13.5 mL/s through 0.6 mm → 47.7 m/s; 5 mL/s through 0.56 mm → 20 m/s; ±1.4 D/s at 2.5 Hz needs ±7.1 mm) but it mixes units: it pushes the *real* 13.5 mL/s through Codex's footprint. Codex's own flow field is F = κ_Q·Q with κ_Q = 125 (`v05_program.py` L102 `Q = flow/kappa_Q`; `phase_flow_units: effective_input_rate_before_chi`), so its tulip base receives Q = 0.045/125 D³/s = **0.18 mL/s**, which through r = 0.64 mm is **0.14 m/s** — self-consistent, just 100×/10×/10× smaller than real (PITCHER §13 L176–179). E-8 [239] explicitly supersedes E-6 [219] on this ("Codex 猜的不是物理入口，而是四个无量纲组的工作点"; `unified.retarget` rule a⁵ = k² B_old/B_new). The diagnosis cites E-6 and E-8 side by side as if they agree; conv_part2 C-3/C-15 already flag that they do not.
- 85 % pulsation: tested, no effect on gaps (PITCHER §13; conv_part2 L201 "tested, no effect on gaps"). It is not something the closure was tuned around.

## 6. What survives of A11 (for the corrected statement)

1. The traction form v* = κ_t u_in (+ radial κ_r term) is Codex's and is the only lateral push in the chi closure (`solver.py` L348–349); its *form* was never re-derived — that is a legitimate (A3/A4-type) complaint, but not "tuned around an impossible inlet".
2. `Params` defaults are still θ05 (B_dep 240 etc.), and PITCHER §6 L81–83 / §11 L134–135, L155–157 still document the old c_S/c_U/c_u coupling and the "1/7" sentence — stale docs vs code.
3. B_dep is dimensional and inlet-scale-dependent, so "B_dep = 240 只在 Codex 的缩小入口下成立" (F7) is correct; the fix was done.

## 7. Corrected statement

θ05 (B_dep 240, κ_t 0.7, κ_r 0.3, κ_Q 125) 是 Codex 围着自己按量纲缩小 100×/10×/10× 的入口定的，直接配物理壶链 + 旧耦合 c_u=3.33 时 χ≈0.05、u_in 被压到 0.2 D/s，所以 [219] 时"没有花瓣"。但"物理壶速只有 ±1.4 D/s 的 1/7"是旧 Coupling 常数 c_u = 0.5/0.15 的产物：改成量纲耦合后同一 3.6 mm@2.5 Hz 抖动给 ±0.68 D/s，扫描赢家 5–8 mm@3–4 Hz 给 ±1.8 D/s，稳态 |u_h| 1.2–2.0 D/s，与 Codex 的 κ_t|v∥| 1.0–1.3 同量级或更大。现用闭合 #9（B_dep 1719、κ_t 0.90、κ_r 0.0085、kQ 1.33）是由 160 组全物理入口扫描选出的，不是围着 Codex 入口调的，且在 256² 上给出叠瓣/褐缝/顶心/茎（round2fix_top3_N256.png）。仍成立的是：牵引律 v* = κ_t u_in 的形式来自 Codex、从未重推导；`Params` 默认值与 PITCHER §6/§11 文档仍停在 θ05 和 c_u 时代；而宽褐带缺失与入口形态无关（足迹 ×0.3、85 % 脉动都不改变 2–3 mm 缝）。

## Files
- Code: `/home/user/zfChen/latte_imex/solver.py` (L23–32, L325–349), `pitcher.py` (L186, L294–331, L537–544, L848), `v05_program.py` (L100–127), `joint_sweep.py` (L5–27, L89), `action_opt.py` (L23–26, L74), `codex_phases.py`.
- Data: `/home/user/zfChen/runs/joint_sweep_r2fix/results.json` (#9), `/home/user/zfChen/results/joint_sweep/round2fix_results.json`, `round2fix_top3_N256.png`.
- Check script: `/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/a11_check.py`.
