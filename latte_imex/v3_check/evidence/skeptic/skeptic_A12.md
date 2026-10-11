# Skeptic notes — CLAIM A12 (κ_r 径向返流 / V_R v05 封顶律 / 拟合 κ_r≈0 vs Codex 反算 0.03–0.07)

Verdict: **confirmed with nuance**. The three factual sub-claims hold in the code and the records, but one of them
(the v05 "cap") is weaker than stated (the cap is mathematically inert), and the "3–10×" discrepancy is not evidence
of a wrong fit: κ_r is unidentifiable from the end-frame loss, so the mismatch carries almost no information.

## 1. What the code does (solver.py, `closure="chi"`)

`solver.py` L31–34 (Params): `kappa_r = 0.3`, `return_law = "v1"`; every fit forces `return_law="v05"`
(`action_opt.py` L76, `joint_sweep.py` L92, `joint_run.py` L29, `optimize.py` L19–20).

`solver.py` L333–350 (`inlet_fields`, old closure):
```
zc  = U⊥² / (B_dep d);  chi = 1/(1 + zc^p)
s   = chi S_eff K;      Lam = kappa_c (1 − chi) S_eff K
VR  = U⊥ chi^(1/(2p));  VR = min(VR, U⊥, sqrt(B_dep d))        # L343–344, return_law="v05"
rad = kappa_r VR                                                 # L345
vs  = kappa_t u_in + rad (x − x_hit)/sqrt(R_dep² + |x − x_hit|²) # L348–349
```
The target `vs` enters only through the relaxation/source momentum in step B (`kernels.py` L165–175,
`stepB_prepare`): θ = 1 + dt(β + Λ/ℓ), a = (q* + dt (s+Λ) v*)/θ. There is **no** divergence condition on q linked to
the poured volume: the outward speed is a free constant `kappa_r·VR` that depends on U⊥, χ, B_dep, d only — not on Q
(S_eff), not on ℓ, not on any sub-layer. Q enters only via the relaxation *rate* (s + Λ ∝ S_eff), i.e. how fast the
film is pulled toward the target, not how far it must move. The film's own volume **is** conserved
(`solver.py` L461: `l_new = l + dt*s_l − dt*div(G)`, conservative update; conservation error ≤1e-14 in E3), so
"不受体积守恒约束" is precise only in the sense of docs_map O4 / conv [4746]: the push speed is not derived from the
volume that has to be displaced (contrast twolayer/v2: ∇·U_v = SK − ⟨SK⟩, conv [4749]).

Magnitude check (python, #9 physics B_dep 1718.6, p 1.556, κ_r 0.00847, κ_t 0.905; inlets from PITCHER §13):
- low pour U⊥ 8 D/s, d 4.5 mm: zc 0.66, χ 0.655, V_R 6.98 D/s → κ_r V_R = **0.059 D/s** vs κ_t|u_in| 1.70 D/s
  (code_map §1.1 quotes 0.07 D/s; same order). With κ_r 0.04 (Codex 反算): 0.28 D/s; with θ05's 0.3: 2.1 D/s.
- cut U⊥ 16, d 2 mm: zc 5.96, χ 0.059, V_R 6.43 → 0.054 D/s.
- For comparison, a volume-conserving radial surface speed Q/(2π r H) at Q 10–15 mL/s, r = 2.5–5 mm (footprint
  edge), H 0.5–1 cm is 0.4–2.4 D/s; at r = 1 cm, 0.2–0.6 D/s. The fitted radial target is 10–40× below that, and
  it is applied only inside the compact kernel K (footprint), not as a 1/r field.

## 2. The v05 "cap law" (`VR = min(VR, U⊥, sqrt(B_dep d))`, L344)

- Source: Codex `experiments/v1_unified/RETURN_SPEED.md` (scratchpad copy). It says explicitly: "本文件记录一个尚未实测
  标定的有效返流假设"; the only literature cited (Xue, Khodaparast, Stone 2019, fountain mixing) "不直接给出下面的
  速度公式、当前杯内量纲或参数值". The motivation is purely effect-driven: with the v1 law `chi·kappa_r·U⊥` a thin fast
  jet (χ→0) could not push, so χ's role as "deposition fraction" and as "radial efficiency" was decoupled by
  V_R = U χ^{1/(2p)}.
- The bound `min(U, sqrt(B d))` is **not an independent cap law**: V_R^{-2p} = U^{-2p} + (B d)^{-p}, so
  V_R ≤ min(U, √(Bd)) identically (RETURN_SPEED.md itself states this as a property of the formula). Numerical check
  over 2·10⁵ random (U, B, d, p): max V_R / min(U, √(Bd)) = 1.000000000000001 → the `min()` at L344 never binds.
  Codex kept the bound only as a stable log-space evaluation for U→∞ underflow.
- So "封顶律无依据" is true in the strong sense: the "cap" is an inert line of code and the underlying law is an
  uncalibrated heuristic by its author's own admission. The diagnosis should not call it a "cap law" at all.

## 3. "拟合 κ_r≈0 vs Codex 反算 0.03–0.07, 差 3–10 倍"

Numbers check out:
- #9 (`runs/joint_sweep_r2fix/results.json[9]`): κ_r = 0.008466 (also 0.0085 in code_map §2; POUR_MODEL_SUMMARY L33
  "0.008").
- Codex 反算 (`codex_phases.py`, `kr = vr/(U·chi_codex^0.25)`; PITCHER §13 table): 0.030 / 0.070 / 0.069 / 0.057,
  geometric mean 0.04. Ratio to #9: 3.5–8×; to the round-2 "winner range" 0.004–0.013: 2.3–17×. "3–10×" is fair.
- PITCHER §13 L200 indeed writes "扫描靠末帧评分自己找到了 Codex 的工作点（κ_r 再小一些）" and never discusses the
  κ_r mismatch (notion_z1 G11 is correct).

But the mismatch is **not informative**, which the claim's "差 3–10 倍" phrasing hides:
- Round 1 (100 LHS, κ_r log-uniform 0.003–0.1): Spearman(log κ_r, total loss) = **+0.14** (tulip 0.14, heart 0.10),
  vs B_dep −0.73. Top-10 samples include κ_r = 0.0548 (#71, loss 0.909), 0.077 (#60, 0.914), 0.0565 (#81, 0.920)
  against the best 0.901 (#37, κ_r 0.0075). Binned minima: [0.003,0.006) 0.907, [0.006,0.013) 0.901,
  [0.03,0.07) **0.909**, [0.07,0.11) 0.914 — differences of 0.01 at 96², which the project itself says "只能排序、
  不能定论" (C:L575) and "0.55 vs 0.60 肉眼差别很小" (E18).
- Round 2 (60 samples resampled ±25 % around #37/#83/#71 and clipped to the bounds): κ_r p10 = median-of-low =
  **0.0030, i.e. 22/60 samples sit exactly on the lower bound**; Spearman +0.03 (r2fix) / +0.06 (round2). The
  "winner range 0.004–0.013" is largely a consequence of seeding round 2 from two low-κ_r seeds, not an independent
  finding. In round2_results the [0.03,0.07) bin reaches 0.892 vs best 0.859.
- Brown-gap count is also insensitive to κ_r (E25: "ν、c_p、κ_r、p_dep 几乎无关").
- The project itself anticipated this (conv L2989): "之前扫描说 κ_r 越小越好，是因为没有时延时径向推挤只会把白摊大。
  有了时延，κ_r 应该能回到 Codex 那种 0.3 的量级，这是一个内在一致性的检查" — but the check was never run:
  `delay_test.py` loads #9 (κ_r 0.0085) and only overrides τ_d/w/α; `gap_diagnosis.py` likewise.
- Matching Codex's κ_r V_R is itself a questionable target: Codex's footprint is 10× smaller (0.64 mm vs 3–5 mm), so
  equal integrated momentum source ∫(s+Λ)v* over a 100× smaller area means ~100× larger local forcing; and Codex's
  own patterns do not reproduce real gap widths (1.5–4 mm vs 4–9 mm, E22).

## 4. What would settle the interpretive part

A 1-D experiment: fix #9 and the heart action, sweep κ_r ∈ {0.0085, 0.04, 0.1, 0.3} with and without τ_d = 0.2 s
at N ≥ 192, scoring process frames (white-disc expansion speed) rather than the end frame. If the loss stays flat
(expected from the ±0.01 spread above), κ_r is a dead parameter and A12's third sentence should be reworded as
"unidentifiable", not "wrong by 3–10×".

## 5. Corrected statement

A12 (corrected): In `closure="chi"` the radial push is a kernel-local relaxation toward a free target speed
κ_r V_R(U⊥, χ, B_dep, d) that is independent of the poured volume Q and of the film thickness (Q only sets the
relaxation rate through s + Λ); the film's own mass is conserved. V_R = U χ^{1/(2p)} is, by its author's own
RETURN_SPEED.md, an uncalibrated heuristic chosen so thin fast jets can still push; the `min(U, √(B_dep d))` at
solver.py L344 is an identity of that formula and never binds. The fitted κ_r = 0.0085 gives 0.06 D/s versus
κ_t|u_in| 1.5–1.7 D/s and versus 0.03–0.07 from the Codex invariants, but the 96² end-frame loss is flat in κ_r
(Spearman +0.14 / +0.03; κ_r = 0.05–0.08 samples within 0.01 of the best; 22/60 round-2 samples clipped at the
0.003 bound), so the discrepancy shows the parameter is unidentified, not that it was fitted wrong. Severity: minor.
