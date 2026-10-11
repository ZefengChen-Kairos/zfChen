# Skeptic check of B1 — "v2 makes the surface converge on the jet and sucks the old white block toward the new milk"

Verdict: **refuted as stated** (the U_s field does converge, but the claimed consequence — the old block being pulled toward the new milk, i.e. a *reversed* push — does not occur in the code; the cell has no measurable effect on the old block). The sign statement and the "prediction never tested" part are correct and should survive as a *minor* model-consistency item, not a fatal one.

Scripts (read-only on the repo): `skeptic_B1_check.py` (part A field diagnostics; part B block runs), `skeptic_B1_check2.py` (block runs with cleaner metrics, argv: gaps, L0s, phi_foam). Logs: `skeptic_B1_gap06.log`. All at N = 96, v2F physics (`results/joint_sweep/codexmap/v2_final_fit_N96.json`), low-pour inlet numbers from `closure_numbers.py` (Q 23.3 mL/s, U⊥ 9.27 D/s, d 6.3 mm, footprint 3.86 × 2.54 mm; cut: 3.3 mL/s, d 1.9 mm).

## 1. What the code does (traced)

`solver.py`
- L303–305 (v2): `S_v = S_eff·D_L/H`, `SE = 2·ent_coef·sqrt(Fr2)·S_v`. v2F: ent 0.688, Fr 3.64 ⇒ S_E/S_v = **5.01** (claim says "5 S_v" — correct; at the cut Fr 11.2 ⇒ 15.5).
- L312: `R_c = min(cell_frac·2.32·Fr·d, 0.45)` = 4.75 mm (low), 4.39 mm (cut).
- L609–624 `_bulk_flow`: `div U_v = S_v(K − ⟨K⟩)`, `div U_c = S_E(K_c − K)` (mean-subtracted), `U_b = U_v + U_c`, `divUs = div_v + Ds`. So at the hit point div U_s = S_v K (1 − 2 ent Fr) + S_E K_c. With K_peak = 563/D² (low) ≫ K_c = 1/(πR_c²) = 90/D², the footprint is convergent. **Confirmed.**
- L625–626: in v2 `Ds` is set to None after the divergence is stored, so up/down-welling act only through U_s (drag/source momentum) and `s_skin = l_skin·max(divUs, 0)` (L414–415). Only *positive* divergence creates skin; the convergent core creates nothing.
- L420–423: v2 step-B target `v* = (β U_s + Λ v_jet + s_ℓ U_s)/(β + Λ + s_ℓ)` — this is the "sources arrive with U_s" term the claim cites.

## 2. Part A — the U_s field at the fitted v2F constants (hit at the origin)

| | low pour | cut |
|---|---|---|
| div U_s at the K peak | **−115 /s** (min −115, max +31) | **−264 /s** (max +15) |
| radial U_s along +x, mm → D/s | 1: −0.67, 2: −0.77, 3: −0.59, 4: −0.26, **5: +0.15**, 6: +0.13, 8: +0.10, 10: +0.07, 20: +0.03 | 1: −1.12, 2: −0.63, 3: −0.31, 4: −0.10, **5: +0.012**, 8: +0.013 |
| inward region | **r ≤ 3.75 mm** (= footprint semi-axis 3.86 mm) | r ≤ 3.75 mm |
| \|U_s\| max | 0.88 D/s at r = 1.3 mm (inside the footprint, not at its edge) | 1.13 D/s at 1.3 mm |
| step-B target v* inside K (median) | \|v*\| 0.87 D/s; along-jet +0.81 (v_jet +1.84); radial **−0.33**; weights at the K peak β 8, Λ 98, s_ℓ 80 | \|v*\| 1.26; along-jet +1.25; radial −0.20; weights β 8, Λ 52, s_ℓ 5 |
| ring (footprint < r < R_c) | 28 cells, div +31 /s, u_r +0.05 D/s | 84 cells, div +15 /s, u_r −0.24 D/s |
| far field | mean div −0.086 /s (uniform sink of U_v, code_map item 8) | −0.012 /s |

Numbers vs the claim: the claim's −126 / −341 /s come from `closure_numbers.py` with the hit at (0, 0.1) and `dv.min()`; on-grid at the origin I get −115 / −264. Same order, same sign; the exact value is grid-alignment dependent (K_peak 563/D² is set by the 1.5h kernel floor at the cut). "|U_s| 0.94 D/s at the footprint edge" is slightly off: the maximum (0.88) is at r ≈ 1.3 mm; at the footprint edge (3.9 mm) it is −0.26 and it changes sign at ≈ 4 mm.

Key geometric fact the claim omits: **the convergent zone is the footprint itself.** Outside r ≈ 4–5 mm the surface flow is *outward* (U_v, 1/r, +0.15 D/s at 5 mm, +0.07 at 10 mm) — the same sign as the observed push, only weak. An old white block that the jet lands *behind* (push_heart: `actions.py` L109–116 starts blob 2 at y = −0.08 while blob 1 was poured at 0.17–0.23, i.e. ≈ 0.1 D ≈ 8 mm from the old centre's near side; video_obs §7: gap born ≈ 0.15 D) lies outside the convergent zone unless it has already spread onto the hit point.

## 3. Part B — does the old block move toward the jet? (ablation: cell on/off)

Set-up: white disc ℓ = L0, m = L0 − 1 (W = 0.88 for L0 = 3), centre (0, 0.10), radius 0.165 D (back edge y = −0.065); low-pour jet stationary at B = back edge − gap, pointing +y (toward the block); v2F; 0.3–0.5 s. "no cell" = `ent_coef = 1e-9` (S_E → 0, U_s = U_v only). Film velocity v_y = q_y/ℓ on the axis at 3.3 / 5 / 8 mm ahead of the jet (positive = away from the jet, toward the block).

gap 2.4 mm (block edge **inside** the convergent zone), L0 = 3, φ_foam = 0 (no new white, flow only):

| t | full: v_y 3.3/5/8 mm | no cell: v_y 3.3/5/8 mm |
|---|---|---|
| 0.05 | −0.66 −0.83 −0.90 | −0.67 −0.84 −0.90 |
| 0.10 | +0.44 +0.31 +0.20 | +0.44 +0.30 +0.20 |
| 0.20 | +0.60 +0.28 +0.03 | +0.61 +0.26 +0.03 |
| 0.30 | +0.68 +0.54 +0.25 | +0.67 +0.53 +0.24 |

Same with φ_foam = 0.337 (fitted; the s_ℓ U_s source-momentum term active, surfacing foam ≈ 80 /s at the K peak): full +0.45/+0.35/+0.24 vs no cell +0.48/+0.37/+0.24 at t = 0.1 s; +0.45/+0.39/+0.28 vs +0.48/+0.41/+0.29 at 0.3 s. L0 = 1.5: identical within 0.02 D/s.

gap 5.4 mm, L0 = 3, φ fitted, 0.5 s (`skeptic_B1_gap06.log`): far-block centroid y and white area identical to 3–4 decimals with and without the cell at every sampled time (0.1207/0.1207, 0.1382/0.1384, 0.1116/0.1110, 0.1077/0.1078, 0.1158/0.1158).

Reading: (i) the only *inward* film motion (t = 0.05 s, −0.6 to −0.9 D/s) is the block's own gravity-current spreading toward the jet (c_p = 2.49 D/s, Δℓ = 2) — it is identical without the cell; (ii) after 0.1 s the film ahead of the jet moves **away** from the jet at 0.3–0.85 D/s in both cases; (iii) the cell changes these velocities by ≤ 0.03 D/s (if anything it *reduces* the outward speed by a few %). The convergent U_s does not reach the film because inside the footprint Λ (98 /s) + s_ℓ pin v* to the forward jet velocity (+0.8 of 1.84 D/s; the inward radial part −0.33 is small and confined to r < 4 mm), and outside the footprint U_s is already outward. **The "old block sucked toward the new milk" consequence is not produced by the code.**

Caveats: synthetic stationary jet, N = 96, ≤ 0.5 s, not the full push_heart script; the measure is the axis column only. The cut phase ([541] "ball at the bottom of v2 hearts") was not simulated: there the ring is inward (u_r −0.24 D/s over 1.25–4.4 mm) and s_ℓ is small (χ = 0.09) so the U_s weight in v* is 0.2 — a ~1–2 mm inward displacement of the film across the cut line is plausible but untested (it would need a run of the cut segment with the cell on/off; ≈ 2 min at N = 96).

Side note: the jet-at-edge configuration (gap 1.2 mm, L0 1.5) runs at 0.9 s per sub-step (PCG not converging quickly; ≈ 50× slower) — a numerics observation, not part of B1.

## 4. The documentary part of the claim

- `LATTE_MODEL.md` §7.6 (L309): "射流足迹内 ∇·U_s ≈ K S_v (1 − 2·ent·Fr) … **射流处表面总是汇聚的** … 白盘扩张速度可以直接从俯视过程视频量出来，用来检验 g′、H_f、τ̂_foam". Correctly quoted. The threshold Fr < 1/(2 ent) = 0.73 is arithmetic ✓.
- No measurement of the disc expansion speed exists in conv_part1–3, notion_z1 or docs (grep "扩张速度/expansion"): only conv [297] (recommend checking c_p against it) and docs_map L222 ("视频可量，未量"). video_obs §1 gives the data it would need: white radius grows ≈ 0.1 D/s (2.5–4.6 s), impact core stays brown ≈ 0.1 D, "the hit point is the last place to turn white". conv [541] labels the ball-at-the-bottom attribution "unverified"; O-10 remains open. **"从未检验" confirmed.**
- physics_prior §2.4 (near field r ≲ 2d outward sheet, far field inward 1–3 cm/s) vs model: model near field (r < 4 mm = 0.6 d) inward up to 0.88 D/s = 7 cm/s, far field outward. The *signs* are indeed swapped relative to the prior in both zones — that is a real model-consistency objection to V4 (docs_map §4.2 row "§7.6"), and the brown impact core in video_obs §1 does not discriminate between the two pictures (both give a brown core). But it is a statement about U_s, not about what the film does.

## 5. Corrected statement

B1 should read: "v2's surface flow converges on the jet inside r ≲ 4 mm (the footprint; div U_s ≈ −115…−126 /s low, −264…−341 /s cut), opposite in sign to the near-field outward sheet of physics_prior §2.4, and LATTE §7.6 states this as an untested prediction. It is *not* what prevents the push: the convergent zone does not reach an old block the jet lands behind, outside it U_s is outward (+0.15 D/s at 5 mm, 1/r), the film inside the footprint is pinned by Λ v_jet, and ablation runs (cell on/off, with and without surfacing foam) show the old block and the film ahead of the jet moving *away* from the jet at 0.3–0.85 D/s identically (Δ ≤ 0.03 D/s). The missing push is B2's problem (no directed near-field momentum beyond Λ in K), not a reversed one. Severity: minor (model-consistency / untested prediction); the [541] ball-at-the-cut-exit hypothesis stays open and needs a cut-segment run with S_E on/off."
