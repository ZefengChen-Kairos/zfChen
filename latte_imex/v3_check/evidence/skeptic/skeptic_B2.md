# Skeptic check of CLAIM B2 (v2: "唯一非局部推力是 1/r 势流 U_v … −S_v/A 均匀汇项让整杯膜以 9%/s 压缩")

Verdict: **refuted as stated** (half of it is confirmed, the two load-bearing words "唯一" and "9%/s 压缩" are wrong in the code's actual dynamics). Corrected statement at the end.

Scripts (read-only, N = 96, v2F physics `results/joint_sweep/codexmap/v2_final_fit_N96.json`, low pour 23.3 mL/s @ 2 cm, hit point (0, 0.1)):
`scratchpad/review/skeptic_B2_check.py` (static fields + 3 × 0.4 s runs, log `skeptic_B2_run.log`), `skeptic_B2_check2.py` (1 s runs with c_p, c_p/10, c_p/100).

## 1. What the code does (solver.py, verified by reading)

`_bulk_flow` L590–626:
- L610–613 `potential_velocity(div)`: subtracts the cup mean of `div`, solves the constant-coefficient Neumann Laplacian (`_factor_bulk` L713–727, face conductivity `kx, ky = g.fx, g.fy` = **1**, i.e. "depth = 1"), returns `cell_grad(phi)`.
- L615 `vx, vy, div_v = potential_velocity(S * self._K)` with `S = self._S = S_v = S_eff·D_L/H` (L304). So ∇·U_v = S_v(K − ⟨K⟩), ⟨K⟩ = 1/A_cup, exactly as LATTE_MODEL.md §2.3 L86 states ("深度平均，表面和深层相同").
- L617–621 cell flow U_c from `SE (Kc − K)`; **U_s = `Ub` = U_v + U_c** (L621), U_d = U_v − U_c.
- L624 `divUs = div_v + Ds`, i.e. the uniform −S_v/A is included in the surface divergence used for `s_skin` (L414–416: `s_skin = l_skin·max(divUs, 0)`); because it is negative outside the footprint it only lowers the positive part by 0.086/s (negligible against the +29/s ring and −126/s core).
- L626 `Ds = None` for v2 → no D_s mass sources; U_s acts on the film only through the drag target in step B (L419–421: `vsx = (β U_s + Λ v_jet + s_l U_s)/(β + Λ + s_l)`, `Lam = β + Λ`, `beta_B = 0`).

Static numbers (my run; agree with `closure_numbers.py` / code_map §1.3):
- S_eff = 0.0455/s, **S_v = 0.0649/s**, S_E = 0.325/s, A_cup = 0.7548 D² (cup_radius 0.49, L79) ⇒ **S_v/A = 0.0860/s**; `divUs` for r > 0.2 D is −0.0860 to four digits everywhere (uniform sink confirmed in the field).
- |U_s| median at r = 0.10 D: **0.0965** (cell on) / 0.0995 (ent = 0); at 0.20 D: **0.0433** / 0.0442; at 0.30 D: 0.023. Centred analytic 2-D potential S_v/(2πr)(1 − r²/R²) gives 0.099 / 0.043 / 0.0215 — the field is the 1/r potential flow to within the off-centre correction. Outside R_c = 4.75 mm the cell's dipole contribution is < 3 %.
- So the numbers "0.1 D/s @ 0.1 D, 0.04 @ 0.2 D, 1/r, no front" are **correct** (PITCHER_MODEL.md L270 "落点 1 cm 处约 0.08 杯径/s" is the same number for the twolayer S).

## 2. "整杯膜以 9%/s 压缩" — refuted numerically

code_map item 8 derived 9 %/s kinematically (∂ℓ/∂t = −ℓ∇·U_s if u = U_s). The film in v2 does **not** follow U_s in the far field because it is pressure-stiff: c_p = √(g′H_f/D_L) = 2.487 D/s (L253), so the pressure −c_p²ℓ∇ℓ balances the drag β(U_s − u) with a static tilt ∇ℓ ≈ βU_s/(c_p²ℓ) ≈ 8×0.05/6.2 ≈ 0.06/D (Δℓ ≈ 0.01 across the far field) instead of a steady convergence.

Run A — v2F with φ_foam = 0, κ_c = 0, ent = 0 (nothing but the U_v drag acts on the film), far = r > 0.25 D from the hit (5146 cells):

| t | ⟨ℓ⟩_far | [min, max] | median |u|/|U_v| (far) | total ∫ℓ |
|---|---|---|---|---|
| 0.2 s | 1.0036 | [0.998, 1.008] | 0.43 | 0.7552 |
| 0.4 s | 1.0036 | [0.995, 1.010] | 0.078 | 0.7555 |
| 1.0 s | 1.0052 | [0.998, 1.010] | **0.036** | 0.7567 |

Far-field thickening after 1 s: **0.5 %** (≈ 0.005/s, decaying), and that 0.5 % is mostly the s_skin mass added at the footprint (∫ℓ grew 0.0017 = 0.2 %) spreading by pressure. The film velocity relaxes to 3.6 % of U_v, i.e. the far-field film is essentially at rest; the uniform sink is absorbed by a 1 % tilt. ℓ at the footprint cell stays 0.965 (pressure refills what U_v's +38/s divergence stretches).

Run B — same with g′/100 (c_p = 0.249): |u|/|U_v| → 1.00 by 0.6 s, ⟨ℓ⟩_far = 1.077 at 1 s, slope 0.08/s ≈ S_v/A, footprint ℓ → 0.037 (= l_skin). So the 9 %/s is exactly what the sink *would* do to a pressure-free film; at the fitted c_p it is suppressed ~20×. Run C (c_p/100) same growth, with sloshing.

Run D — full v2F (foam, traction, cell on), 0.4 s: ⟨ℓ⟩_far 1.0418, far-field max|q| = 0.12 D/s = 4–6× |U_v| there, ∫ℓ +4.4 % (deposited foam). The far-field thickening here is the **pile spreading outward by pressure**, not the sink (Run A isolates the sink at 0.4 % over the same interval).

So "ℓ、m 以 9%/s 增长" / "整杯膜被压缩" does not describe v2F. (The E-18 end state ℓ ≡ 1.25 is deposited foam homogenised by c_p — B3's mechanism — not compression by the sink.)

## 3. "唯一的非局部推力是 U_v" — overstated

Terms acting on film outside the footprint in v2 (L395–470, L679–712): −c_p²ℓ∇ℓ (gravity-current speed √(c_p²ℓ) = 2.5–3.5 D/s, D_eff = c_p²ℓ²/β = 0.77ℓ² D²/s), ∇·(ℓT) with ν_eff up to ν_max = 0.5 D²/s, Rusanov-advected inertia, and the drag toward U_s. code_map §3 (same reviewer) itself lists the pressure as "non-local, isotropic". In Run D the pressure-driven far-field |q| (0.12 D/s) exceeds |U_v| (0.02–0.03 D/s) by 4–6×. What is true: U_v is the only non-local **coffee-flow** (external) forcing outside R_c; it is radial from the hit point (isotropic, no front, like the pressure); the only *directed* force in v2 is Λ(v_jet − u) inside K (κ_c = 0.29, Λ_peak 104/s) and, inside R_c, U_c converges (−126/s) against it. "无局部前沿" is correct for U_v (Laplace solution) — and the thing B2 wants, a block-wide directed push, is indeed absent, but not because U_v is the only non-local term.

## 4. Physics side-note on the sink (not load-bearing)

"物理上应是液面上升而非表面汇聚" is a modelling judgement, not something the code settles: for a depth-uniform spreading flow a floating material surface *is* compressed at S/(AH) (new surface is created at the source); it is zero only if the jet's volume spreads at depth (plunge depth 2.32 Fr d ≈ 5 cm ≈ H, so plausibly). Either way Run A shows the term is dynamically irrelevant at the fitted c_p.

## 5. Corrected statement

v2's only non-local *coffee-flow* forcing outside R_c is the depth-uniform 1/r potential U_v (|U_v| ≈ 0.10 D/s at 0.1 D, 0.04 at 0.2 D, radial from the hit point, no front); inside R_c the surface converges (−126/s). The uniform sink −S_v/A = −0.086/s is in `divUs` (L615, L624) but, with c_p = 2.49 D/s, the film does not follow it: the far-field film stays at rest (|u|/|U_v| → 0.04) with a ≈1 % static tilt, thickening ≈ 0.5 %/s, not 9 %/s (9 %/s only appears with c_p ≲ 0.25). The dominant non-local push on existing film is the foam pile's own pressure −c_p²ℓ∇ℓ (far-field |q| 0.12 D/s, 4–6× U_v), which is isotropic (B3); no term in v2 is a directed block-wide push except Λ inside the footprint.
