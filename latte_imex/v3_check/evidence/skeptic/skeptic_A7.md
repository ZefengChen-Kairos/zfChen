# Skeptic check of A7 — "χ 开关 B_dep 是效果参数；'切'只是 Λ 的 1–2 mm 拖带而非穿透分断；联合搜索退化到切刀也沉积（χ 0.7）"

Verdict: **confirmed with nuance**. The mechanism (no penetration / exposure process in the `chi` closure; the cut is a traction band inside the footprint kernel) is traceable in the code. Two of the four pieces of evidence are weaker than the claim makes them: the B_dep "drift" is mostly an inlet-units artefact that the docs themselves explain, and the "χ ≈ 0.7" number belongs to an abandoned Codex-scaled search, not to the current physics #9, where the cut is 94 % non-depositing.

## 1. What the code does (solver.py, `closure="chi"`)

`inlet_fields`, L331–350:
```
zc  = U_perp**2 / (B_dep * d)                     # L333
chi = 1/(1 + zc**p_dep)                            # L334
s   = chi * S_eff * K                              # L335  deposit (adds l and m, c = 1)
Lam = kappa_c * (1 - chi) * S_eff * K              # L336  traction weight
v*  = kappa_t * u_in + kappa_r * V_R * e_r         # L341–349 (V_R = min(U chi^{1/2p}, U, sqrt(B d)) for "v05")
```
Momentum step (`kernels.stepB_prepare`, L165–175): θ = 1 + dt(β + Λ/ℓ), a = (q* + dt(s+Λ)v*)/θ, i.e. Λ relaxes the film momentum toward ℓ·v* **inside the kernel K only**. Mass: `rhs = l + dt*s_l − dt*div(G)` (L460) — there is **no sink** of film under the jet in the chi closure; (1−χ) of the poured volume simply vanishes (it is neither added to ℓ nor removes ℓ).

Visible whiteness for chi: `concentration()` L836 returns `m/l` — a passive tracer (step C advects m with the same flux as ℓ). Hence a film that is pushed aside or thinned along the cut path keeps c unchanged; brown can only appear on the path if **brown film is transported there** (from the disc edge → the notch), never by exposure. This is the same defect as A1 and is the real reason the three observed cut features — thin brown line 0.01 D inside the white, lateral bulging of both lobes, stem to the rim (video_obs §0 "final cut") — cannot be produced: (i) the line needs exposure (impossible, tracer); (ii) bulging needs a lateral push (the code *has* a radial term κ_r V_R e_r, see §3); (iii) the stem is a traction effect the model does produce.

Kernel width: `kernel()` L261 floors r1, r2 at 1.5 h → at N = 96 the 0.93 mm cut footprint becomes 1.25 mm (2.5 mm diameter); at 256² it is the physical 0.93 mm. The brown lines the old model produces are 2–3 mm (PITCHER L262 "细线 2–3 mm"). So "1–2 mm 拖带" is the right order (the band is the kernel), the resulting line is 2–3 mm.

Conclusion for the mechanism part: **confirmed.** The only cutting action is Λ(v* − u) inside K; there is no process in which the thin stream pushes foam under, separates the two sides, exposes coffee and lets the foam partly heal. Docs agree: PITCHER L257 (twolayer section, but the same Λ term) "这一项决定心形的凹口和细切，去掉以后心形没有缺口"; physics_prior §3d (L161–163) describes the real process as drag + push-under + resurfacing behind the point, 3–6 mm trail healing to a thin line — the Λ term reproduces only the "drag" half.

## 2. `stage_actions.py` lift does not cut through — confirmed, and by construction

`moves_of` L64–67: `step = ±0.06`, the lift Move goes from the stage end point `b` to `b + 0.06` along the push axis at `z_lift`, `Q_lift → 0.25 Q_lift`. The traverse length is **hard-coded 0.06 D ≈ 5 mm** (not a search parameter: GLOBAL L26–27 only exposes theta, u_B, z_lift, dur_lift, Q_lift). The video cut traverses 0.6–0.7 D at 0.3–0.6 D/s (video_obs §0). Whatever the blob position, a 5 mm traverse cannot be a through-cut; [555] "没有切穿白团" is a property of the parameterisation, not of the optimiser.

## 3. Where the evidence is weaker than the claim

### 3a. "B_dep 4 → 2.5 → 240 → 1719 → 2800" is mostly an inlet-units artefact, and the docs say so
χ depends only on the dimensionless group U⊥²/(B_dep d). Each value belongs to a different convention for U⊥ and d:

| value | where | inlet convention | χ (low / cut), computed |
|---|---|---|---|
| 240 | Codex θ05, `Params` default L32 | Codex-scaled: U 0.8–1.8 D/s, d 0.6 mm, U²/d ≈ 940–1320 at the cut | 0.97 / 0.03–0.06 (PITCHER L193) |
| 4 | PITCHER L88 | c_U-scaled U⊥ (0.70 m/s ↦ 0.263 D/s), continuity d 5.3 / 2.2 mm | 0.94 / 0.14 (my calc, p = 2) |
| 2.5 | PITCHER L170, `pitcher.PHYSICAL_PARAMS` L586 (dead code, code_map item 13) | same U, physical footprint d 5 / 2.6 mm | 0.84 / 0.08 |
| 1719 | POUR L34, #9 fit, `runs/joint_sweep_r2fix/results.json[9]` | physical D/s: U 8 / 16 D/s, d 4.5 / 2 mm | **0.655 / 0.059** (my calc with p = 1.556) |
| ≈2800 | PITCHER L200 | geometric mean of back-solving Codex's χ per segment through the physical inlet (1210–7100) | — |

PITCHER L176–178 states explicitly: "沉积阈值和入口定义绑在一起，换入口就得换尺度"; L200–204 and POUR L36–37: "B_dep = 240 只在 Codex 的缩小入口下成立，换物理入口必须换 B_dep". So the sequence is not a free-floating effect parameter being retuned to taste; it is one group U²/(B_dep d) kept at a design point (low χ ≈ 0.6–0.95, cut χ ≈ 0.03–0.14) under four inlet conventions. The one genuine change is 240 → 1719 when the physical pitcher chain replaced the Codex-scaled inlet, and the fit landed where the back-calculation says it should (1200–4600).

The docs also give B_dep a physical reading (PITCHER L256): B_dep = g′ Fr_c² with g′ = 75 D/s², Fr_c² = 23 → 1725, i.e. within 0.4 % of #9's 1718.6. This is clearly post-hoc (Fr_c² was read off from #9; docs_map C2 lists four inconsistent Fr_c² values, 9.5–41 across fits), so "effect parameter" in the sense of *fitted, not measured* is fair (PITCHER §13 O5 says so itself). But "effect parameter" in the sense of *physically meaningless / drifting* is not supported by the listed numbers.

### 3b. "联合搜索退化到切刀也沉积（χ 0.7）" refers to the Codex-scaled inlet, not the current model
conversation_part1 L918 / conv_part1 B-4 [95]: the 41-dim search used **V0.5 controls** (Codex-scaled inlet, U²/d ≈ 940–1320 at the cut) with B_dep bounded 50–1500 (L407, L1445); it hit 1470 (near the upper bound). With that inlet, B_dep = 1470 gives χ_cut = 0.54–0.71 (p 1.5–2) — the number is reproducible. But in the **current** chain (physical inlet, #9), the cut has U²/d = 10240, zc = 5.96, **χ = 0.059**: 94 % of the cut stream is non-depositing, s_peak ≈ 1.3 /s vs Λ_peak 14.4 /s (code_map L74). The "degenerate" search was abandoned at U-31 and never feeds the current results. So this item is a true historical fact but **not evidence about model A as it now stands**, and it does not show that χ is unidentifiable: the later 160-sample sweeps and the Codex back-calculation converge on the same group value.

Residual point that does survive: the loss (end-frame, area-dominated, B-3/B-9) never penalises a depositing cut, so nothing except the inlet magnitudes keeps the cut non-depositing. That is a loss defect (A17), not a χ-switch defect.

### 3c. The code has a lateral-push term that the fit switched off
`inlet_fields` L341–349: v* includes κ_r V_R e_r, a radial push away from the hit point — the one term that could give "两侧鼓起". With #9: V_R(cut) = min(16·0.059^{0.32}, 16, √(1719·0.025)) = 6.4 D/s, κ_r = 0.0085 → **0.054 D/s**, against κ_t|u_in| ≈ 1.5 D/s (ratio 1:28). Same at the low pour (0.059 D/s). So "切的唯一机制是 Λ 拖带" is true of the *fitted* model; the *code* offers a second mechanism that the end-frame fit drove to zero (docs_map O4 also notes κ_r "不受体积守恒约束"). Even if it were on, it could not expose brown (c tracer), only thin the film.

### 3d. "穿透分断" is not what physics_prior describes either
physics_prior §3d says the real cut is: stream pierces the raft, entrains raft material within a few mm **toward and under** the impact point, leaves a 3–6 mm brown trail that partly heals to a thin line as pushed-under foam resurfaces behind. That is "drag + sink + delayed re-surfacing", not a clean geometric split. The old model's Λ is the drag part; what is missing is (i) a film **sink** under a penetrating stream (there is none, L460), (ii) re-surfacing behind the point (the §14 delay patch has m_sub but only for the χ-deposited part, L367–379, and it is off, tau_d = 0, in all fits), (iii) exposure in the whiteness observable.

## 4. Corrected statement

The χ switch is a fitted dimensionless group U⊥²/(B_dep d) whose numeric B_dep is inlet-convention dependent (4 / 2.5 / 240 / 1719 are the same design point χ_low ≈ 0.65–0.95, χ_cut ≈ 0.03–0.14 under four conventions; the docs state this); it is "effect" in the sense that nothing but the end-frame fit fixes it and the end-frame loss does not penalise a depositing cut. In the current physical chain (#9) the cut *is* non-depositing (χ = 0.059); the "χ ≈ 0.7" degeneration belongs to the abandoned Codex-scaled 41-dim search. The structural defect stands: the only cutting action is the traction Λ = κ_c(1−χ)S K relaxing the film toward κ_t u_in inside the 1–2.5 mm kernel (plus a radial push κ_r V_R e_r that the fit set to 0.05 D/s, 1/28 of the tangential target); there is no film sink under a penetrating stream, no delayed re-surfacing behind it, and — because whiteness is the tracer m/ℓ — no way for pushing foam aside to expose brown, so the observed thin brown line + lateral bulging cannot arise; `stage_actions.py` additionally hard-codes a 0.06 D (5 mm) lift traverse versus the 0.6–0.7 D cut in the video.

## 5. Experiment that would settle the residual
Run the heart V0.5-translated script at 256² with #9 and (a) κ_r = 0.3 (Codex value) vs 0.0085, (b) a one-line film sink −(1−χ)S K·ℓ/ℓ_ref under the jet; compare the cut path profile (c and ℓ across the path). If (b) with an exposure-capable whiteness (A1 fix) yields a brown line with bulging, the "missing sink + exposure" reading is right; if only κ_r matters, the lateral-push reading dominates. ~2 × 1 min at 128².

Files: /home/user/zfChen/latte_imex/solver.py (L261, L331–350, L353–460, L836), /home/user/zfChen/latte_imex/kernels.py (L165–175), /home/user/zfChen/latte_imex/stage_actions.py (L24–31, L53–70), /home/user/zfChen/latte_imex/pitcher.py (L586), /home/user/zfChen/latte_imex/PITCHER_MODEL.md (L88, L170–178, L190–204, L256–257), /home/user/zfChen/latte_imex/POUR_MODEL_SUMMARY.md (L34–37), /home/user/zfChen/runs/joint_sweep_r2fix/results.json[9].
