# Skeptic B5 — "v2 的褐色只能靠胞环运动学稀释；环宽 1–2 mm；v2s R_c < 足迹 ⇒ 零环格，优化器在关掉胞"

Verdict: **confirmed with nuance**. Every number in the evidence reproduces from the code; the only wording that
over-reaches is "褐色只能靠 s_skin" (s_skin is not the brown source — foam thinning m→0 is; s_skin is the ℓ-floor that
also caps the whiteness in the ring) and "关掉胞" (the fit removes the *exterior upwelling ring*, not the entrainment
sink at the jet).

## 1. What the code does (solver.py, v2 branch)

- L415–418: `s_skin = P.l_skin * max(divUs, 0)`; `s_l = s + s_skin`; `brown_rate = Σ s_skin · area`.
  `divUs` (L624) = `div_v + Ds` = `S_v(K − ⟨K⟩) + S_E(K_c − K)`; in v2 `self.Ds` is then set to `None` (L626) so the
  L424–431 up/down-welling source branch is **not** taken — brown enters only through film kinematics + s_skin.
- Mass: `rhs = l + dt*s_l − dt*div(G)` (L448), foam: `m_new = m + dt*s − dt*div(G c)` (L507) — s_skin adds ℓ, never m.
- Whiteness (L831–833, `concentration()`): `W = 1 − exp(−m/m_o)` — a function of **m only**, not of c = m/ℓ.
  ⇒ in v2 "brown" = thin foam. s_skin changes the colour only indirectly: `m_new = clip(m_new, 0, l_new)` (L511) and
  in the ring ℓ → l_skin exactly (balance `l_skin·divU_s = ℓ·divU_s`), so m ≤ 0.03 ⇒ W ≤ 1−e^{−0.03/0.229} = 0.12 (v2s),
  ≤ 0.03 (v2F, m_o = 0.934). So l_skin does fix the gap's residual whiteness — item 10 of code_map is right.
- l_skin = 0.03 is in `BASE_V2` (joint_cma.py L22), not in `SPACE_V2` (L19–21): **fixed, never fitted**. LATTE_MODEL.md
  L108 calls it a regularisation ("结果应对它不敏感"); L312–313 (§7.7) admits the sensitivity is unchecked.
- `_bulk_flow` L617–618: `cell = hypot(x−x0, y−y0) < max(R_c, 2h)`; `R_c = min(cell_frac·2.32·Fr·d, 0.45)` (L312).

## 2. Numbers (closure_numbers.py re-run + my own map, N = 96, h = 0.83 mm, low pour 23 mL/s @ 2 cm)

| | cell_frac | ent | R_c | footprint (r1×r2) | cells divU_s>0 outside K | divU_s max | |U_c| at r=5 mm |
|---|---|---|---|---|---|---|---|
| v2F | 0.0898 | 0.688 | 4.75 mm | 3.86×2.54 mm | 48 (r = 3.28–4.73 mm) | 28.7/s | 0.057 D/s |
| v2s | **0.0528** (bound 0.05, log) | 0.369 (bound 0.2) | **2.82 mm** | 3.86×2.54 mm | **0** | 38.5/s | 0.024 D/s |

- Ring width v2F: R_c − r1 = 0.89 mm (major axis), R_c − r2 = 2.21 mm (minor) ⇒ "1–2 mm" is right; e-fold thinning
  1/divU_s = 35 ms; equilibrium ℓ = l_skin. At the high cut the ring is 4.39 − 1.25 = 3.1 mm (kernel floored to 1.5h).
- v2s: closure_numbers' `ring` mask uses a *circle* of max(r1, r2) = 3.86 mm, which could have overstated "0 cells" for an
  ellipse; but with the true elliptical footprint (K > 0) the count is still **0** (minor-axis excess 2.82 − 2.54 =
  0.28 mm < h). Integrated positive divergence outside the footprint = 0.0000/s. So the evidence holds strictly.
- v2s is **not** "cell off": 28 of the 56 footprint cells still have divU_s > 0 (max 38.5/s, higher than v2F) because
  K_c (top-hat, 256/D²) exceeds K (peak 594) in the outer ~58 % of the footprint; the jet sink remains (divU_s min −18/s).
  What is gone is the exterior upwelling annulus and most of the exterior converging surface flow (|U_c| halves at 5 mm
  and is pure quadrupole residue beyond). The cell now "only reshapes the footprint source" — code_map item 4's phrase
  is the accurate one; "关掉胞" is a fair gloss only if read as "关掉外环".
- "优化器在关掉胞": cell_frac = 0.0528 vs log-bound 0.05 in v2s; 0.0898 in v2F; 0.122 vs bound 0.1 in cell_fit. Three
  fits, three times at/near the lower bound ⇒ the objective does reward shrinking R_c. ent_coef is interior (0.37, 0.69).

## 3. Where the claim over-reaches

1. "褐色只能靠胞环运动学稀释". In the equations, ℓ and m thin wherever the **film's own** velocity diverges
   (`∂ℓ/∂t + ∇·(ℓu) = …`), e.g. the trailing edge of the footprint where Λ_jet (peak 362/s in v2s) pins film to
   v_jet = 1.73 D/s, or a pressure-spread mound thinning gradually. These are kinematic exposures not tied to the
   cell ring; only there is no s_skin (divU_s ≤ 0), so ℓ can fall to `l_floor = 1e-3` (L247, L576) and trigger a
   step retry. Whether they ever make a visible *gap* (vs. a gradient) is untested — a run with `ent_coef=0` /
   `cell_frac` tiny and κ_c large would show it. The ring is the only *designed* brown-creation mechanism (LATTE L110:
   "不再有'生成褐色表面'…的源项"), which is what the claim means in practice.
2. "s_skin 正则化" is the floor, not the brown source: brown colour comes from m → 0 under stretching; s_skin bounds how
   brown (W ≤ 0.03–0.12) and how thin the gap film is. Both are regularisation-defined, so the thrust stands.
3. Ring width is resolution-bound: 0.9 mm (major axis) is ≈ 1 cell at N = 96, and the kernel floor 1.5h changes the
   high-cut ring; fitted cell_frac is resolution-dependent (code_map item 3).

## 4. Corrected statement

v2 creates brown only by kinematic thinning of the floating layer (W = 1−e^{−m/m_o} depends on m alone); the only
designed divergent surface flow is the entrainment-cell annulus footprint < r < R_c, where ℓ relaxes to the fixed,
unfitted l_skin = 0.03 (W ≤ 0.03–0.12) in ~35 ms. At the low pour that annulus is 0.9–2.2 mm wide in v2F and
non-existent in v2s (R_c = 2.82 mm < r1 = 3.86 mm; zero cells with divU_s > 0 outside the elliptical footprint),
because the joint fit pushed cell_frac to its lower bound (0.0528 vs 0.05); the entrainment sink itself (ent = 0.37,
divU_s = −18/s at the jet) is kept, so the fit removes the exterior ring and its converging surface flow, not the
cell. Film stretching by jet traction is an untested second route to brown.
