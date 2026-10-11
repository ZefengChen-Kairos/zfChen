# Skeptic check of CLAIM B3 (v2: c_p² = g′H_f/D_L → c_p ≈ 2.5, D_eff = 0.77ℓ² fills the brown ring in ms; pressure on total ℓ, yield only in foam ⇒ crema is the only mobile phase and is squeezed out of the gap)

Verdict: **confirmed in its conclusion, refuted in two of its three mechanism statements.** The diagnostic bottom line — a v2 film cannot hold a thickness contrast between a foam block and the crema gap, the gap is closed by pressure well inside one pour stage, and no fitted parameter prevents it — holds and is even stronger than stated (it does not depend on c_p being 2.5, nor on τ_y). But (i) "crema 成唯一可动相被挤出缝" is wrong in the code's kinematics (foam and crema share one velocity; the blocks' foam slumps; the crema mass in the gap *increases* while the gap fills), (ii) the attribution to H_f = 6 mm / c_p = 2.5 is wrong (c_p = 0.97 still fills the gap; the old closure with the same thickness step collapses just as fast; c_p = √(g′H_f/D_L) is the physically correct buoyancy wave speed), and (iii) the E-18 "c 0.95–0.98" is the opacity W, not the foam fraction, and the evidence numbers contradict "no crema left".

Scripts (read-only): `scratchpad/review/skeptic_B3_check.py`; logs `skeptic_B3_run1..4.log`. Setup: N = 96, no pour (`Inlet(active=False)` ⇒ `_bulk_flow` L601–603 sets U_s = None, drag toward rest), two foam blocks |x| < 0.12, 0.025 < |y| < 0.20 with ℓ = 2, m = 1 (6 mm foam on 6 mm crema in v2F units) separated by a 4 mm (0.05 D, 5-cell) crema strip ℓ = 1, m = 0; relax 0.3 s. "gap" = strip centre |x| < 0.02; "blk" = |x| < 0.02, 0.08 < |y| < 0.12; "strip crema" = ∫(ℓ − m) over the whole strip, normalised by its initial value.

## 1. What the code does (verified by reading)

- `solver.py` L250–253: for `closure == "v2"`, `cp = sqrt(g_red * H_f / D_L)` — overrides `Params.cp`. v2F: g_red = 82.83 D/s², H_f = 5.97 mm ⇒ c_p = **2.487 D/s**; v2s (the physics of the E-18 run, `ph3_v2s` = `BASE_V2 + joint_cma_v2s best.phys`): g_red 81.34, H_f 5.77 mm ⇒ c_p = **2.422**. (The user's "c_p ≈ 2.1" in [549]/[551] is not what the code uses.) The comment on L251 says "/D_L²" but the code divides by D_L once; LATTE_MODEL L289 shows this is the correct non-dimensionalisation (g′ in D/s², c_p² = g′_model·H_f/D_L).
- Physically: g′_phys = 82.8 × 0.08 = 6.6 m/s² (ρ_f/ρ = 0.32), √(g′H_f) = **0.20 m/s** = 2.49 D/s. physics_prior §2.5 gives √(g′h) ≈ 0.13 m/s for 3 mm. So **c_p = 2.5 is the correct buoyancy-wave speed of a 6 mm floating foam, not an error introduced by the H_f bound**; the physically missing piece is the arrest (yield stress, h_stop), not a smaller c_p.
- Pressure on the total thickness: `stepB_prepare(..., lsafe, beta_B, P.cp ** 2, ...)` L446, `coef = c_p² ℓⁿ/θ`, `q⁺ = a − dt·coef·∇ℓ⁺` (L465). ℓ = crema + foam. ✓ as the claim says.
- Yield weighted by c only: `_ssa_step` L691–693 `tau = cr * P.tau_y + (1 - cr) * P.tau_y_crema`, `tau_y_crema = 0` fixed (BASE_V2, both fits); `nue = min(nu + tau/(2 De + eps), nu_max)`, `nu_max = 0.5`, `yield_eps = 0.02`. ✓ as the claim says. But it is a **viscosity on velocity gradients**, applied once per frame (L803–809, `ssa_per_frame=True`), after the sub-steps in which pressure already moved the film. There is one velocity field u = q/ℓ for crema and foam (step C transports m with the same flux G, L476–511); there is no phase-selective mobility anywhere.
- Drag in v2: `beta_B = 0`, `Lam_eff = β + Λ` (L419–423) ⇒ −β(u − U_s), relaxation time ℓ/β = **0.125 s** (v2F) / 0.143 s (v2s). With no pour, U_s = 0.
- D_eff = c_p²ℓ²/β: 6.19/8.0 = **0.77 ℓ² D²/s** (v2F), 0.84 (v2s) — arithmetic ✓. But this is the *drag-limited* (t ≫ ℓ/β = 125 ms) spreading coefficient; the claim's "R_c²/D_eff ≈ 5 ms" uses it on a timescale 25× shorter than the drag time, where the dynamics are inertial gravity waves at c_p√ℓ = 2.5–3.5 D/s (4 mm ⇒ ~15 ms) — and in the code, neither regime is what sets the fill time (see §2).

## 2. Numerical test: does the gap fill, how fast, and what moves?

v2F as fitted (τ_y 0.72, ν_max 0.5, frame 1/120 s). First frame has u_max = 0 ⇒ nsub = 1, dt = 8.3 ms (CFL blind to c_p, code_map item 11); later frames nsub 2–3:

| t | ℓ_gap | m_gap | crema_gap | W_gap (m_o 0.93) | ℓ_blk | m_blk | strip crema/crema0 |
|---|---|---|---|---|---|---|---|
| 8.3 ms | 1.475 | 0.239 | 1.236 | 0.23 | 1.951 | 0.976 | 1.20 |
| 25 ms | 1.798 | 0.418 | 1.380 | 0.36 | 1.677 | 0.839 | 1.29 |
| 50 ms | 1.394 | 0.300 | 1.094 | 0.27 | 1.260 | 0.630 | 1.09 |
| 100 ms | 0.952 | 0.158 | 0.794 | 0.16 | 1.049 | 0.525 | 0.81 |
| 300 ms | 1.132 | 0.221 | 0.911 | 0.21 | 1.112 | 0.556 | 0.91 |

v2s (E-18 physics, m_o 0.229): same ℓ, m to 2 digits; W_gap = **0.64 at 8 ms, 0.85 at 25 ms, 0.68 at 300 ms** — the gap is white after one frame. τ_y = 0 gives the same table to 3 digits at 8 ms and within 0.05 afterwards; τ_y = 3 and τ_y = 10 (20 Pa) identical at 8 ms (the SSA has not acted yet) and only slightly slower afterwards (gap ℓ 1.14–1.17 at 300 ms instead of 1.13); ν_max = 50 with τ_y = 10 changes nothing (ℓ_gap 1.170 vs 1.135 at 300 ms). SSA per sub-step (`ssa_per_frame=False`): identical to per-frame (sub-steps are 2.8–8.3 ms anyway).

Time-resolved (frame_dt = 1/1200 ⇒ dt = 0.83 ms, SSA every 0.83 ms):

| variant | ℓ_gap 25 / 50 / 100 / 200 / 300 ms | ℓ_blk 300 ms | strip crema/crema0 (max, 300 ms) |
|---|---|---|---|
| v2F | 1.11 / 1.28 / 1.58 / 1.87 / 1.81 | 1.59 | 1.33, 1.27 |
| v2F τ_y = 0 | 1.12 / 1.30 / 1.65 / 1.97 / 1.90 | 1.52 | 1.36, 1.28 |
| v2s | 1.11 / 1.27 / 1.59 / 1.92 / 1.88 (W_gap 0.72 at 100 ms, 0.87 at 200 ms) | 1.55 | 1.35, 1.29 |
| v2F τ_y = 3, ν_max = 50 | 1.10 / 1.22 / 1.42 / 1.59 / 1.58 | 1.78 | 1.24, 1.22 |
| v2F τ_y = 10 (20 Pa), ν_max = 50 | 1.09 / 1.16 / 1.27 / 1.37 / 1.41 (W 0.20 at m_o 0.93; 0.60 at m_o 0.23) | 1.98 | 1.17, 1.17 |

Readings:
1. **The gap closes and the blocks slump in every fitted variant.** With the production time step the gap thickness equals the block thickness after 1–3 frames (10–25 ms) and the blocks are flat (ℓ ≈ 1.0–1.2) by 100 ms; the whole film then sloshes. With a 10× smaller step the fill takes 100–200 ms and the blocks are half gone at 300 ms. Either way it is far inside a 0.9–1.6 s pour stage, so "褐缝必然被填" holds. The "毫秒内" figure is specifically the production-dt behaviour: one implicit-Euler pressure step with dt = 8.3 ms smears ℓ over √(dt² c_p² ℓ) = 0.025 D = 2 mm (exactly the half-gap) — a numerical smearing, not the D_eff diffusion; the resolved fill time is set by implicit-Euler/Rusanov damping of the 4 mm gravity wave, not by drag. So "D_eff = 0.77ℓ²…≈5 ms" is the wrong estimate for the right conclusion.
2. **τ_y is inert at the fitted values and at the fit's upper bound, and the yield formulation cannot arrest a 6 mm step even at 20 Pa with the cap lifted and the step resolved** (gap still creeps to ℓ 1.4, W 0.2–0.6, in 0.3 s). Reasons visible in the code: the stress is a capped viscosity (ν_eff ≤ ν_max; τ̂/ε = 36 (v2F) or 500 (τ_y 10) ≫ ν_max) so "unyielded" foam is a Newtonian film that creeps; c̄ is face/corner-averaged (`S["Avg"]`, L690) so the block edge, where the pressure gradient lives, has τ̂ ≈ ½τ_y; the crema half of the film has τ̂ = 0; and the stress acts after the pressure sub-steps. The structural defect is therefore "no mechanism arrests a thickness step" (the physics_prior h_stop / Bn ~ 1 raft does not exist in v2), not "c_p too strong".
3. **Crema is not squeezed out of the gap.** In every v2 variant the crema mass in the strip rises to 1.2–1.4× its initial value while the gap fills (the blocks' own crema sub-layer, ℓ − m = 1, moves with the foam), and at 300 ms it is 0.9–1.3×. The gap centre's crema thickness is 1.1–1.5 H_f throughout. The gap whitens because **foam (and crema) from the blocks flows in** — m_gap 0 → 0.2–0.5 — while the blocks' foam thins (m_blk 1 → 0.5–0.9). A single-velocity mixture cannot have a "唯一可动相"; the pressure flattens ℓ by moving everything, and c is then mixed by van Leer transport.
4. **Lowering c_p to the old value does not save the gap**: v2F with g_red/6.4 (c_p = 0.98): ℓ_gap 1.21 / 1.48 / 1.61 at 8 / 25 / 50 ms, blocks ℓ 1.43 at 300 ms. And the **old closure (#9, c_p 0.97, β 2.4) with the same thickness step collapses just as fast**: ℓ_gap 1.93 at 50 ms, ℓ_blk 0.94 at 300 ms, W_gap = c/0.6 ≈ 0.3–0.4. The old model keeps gaps in push_heart not because its pressure is weaker but because its gaps are **tracer stripes at near-uniform ℓ** (old units: "1" = 60 mm; E-18 old blocks c 0.63, m 0.7 ⇒ ℓ ≈ 1.1, i.e. a 10 % step), on which pressure does nothing (A1/A3). The [551] sentence "旧模型的膜压力弱，crema 被夹在块与块之间" is a misattribution that B3 inherited.

## 3. The E-18 evidence re-read

- "ℓ ≡ 1.25": `runs/action_opt/ph3_v2s/push_heart_h1.000_q1.000_s1.000_N96.json` has `deposited = 0.2018` (H_f·D² units); cup mask area 0.7548 D² ⇒ 1 + 0.2018/0.7548 = **1.267**. So ℓ ≡ 1.25 means *all* deposited foam was spread uniformly over the whole cup — consistent with the pressure flattening of §2 (and with skeptic_B2's finding that it is not the uniform sink).
- "c 0.95–0.98 in the white, m 0.7 in blocks": with ℓ = 1.25 the foam fraction is m/ℓ = **0.56**, not 0.95; 1 − exp(−0.7/m_o) with the v2s m_o = 0.229 is **0.953**. The table's "c" is the opacity W (`concentration()` L832–833), mislabeled as "奶泡占比". Hence "白区最后几乎全是奶泡，里面没剩 crema" is contradicted by the same numbers: 1.25 − 0.7 = **0.55 H_f (3.2 mm) of crema under every block** and 0.7–0.8 H_f in the gap — more crema in the gap than under the blocks. The gap is "filled" because m_gap = 0.44–0.56 gives W = 0.85–0.91 at m_o = 0.229; with the v2F m_o = 0.93 the same m would be W = 0.38–0.45 (grey).
- The E-18 run used **v2s**, whose τ_y = 0.055 D²/s² = 0.12 Pa at ρ_f = 330 — i.e. effectively no yield stress — so it says nothing about the Bingham term; §2 shows v2F's 0.72 (1.5 Pa) behaves the same.
- "H_f 变薄反而更差": consistent with the code, D_eff = g_red·h²/(D_L·H_f·β) ∝ 1/H_f at fixed fitted β (and ℓ = h/H_f larger) — but this is a parameterisation artefact (β should scale as 1/H_f for a fixed basal stress, LATTE L290), not evidence about the mechanism.
- The `.npy` saved by `run_one` holds only W (`action_opt.py` L85–89); ℓ and m of the E-18 run are not on disk, so the profile numbers can only be checked for internal consistency (done above), not re-read.

## 4. Bottom line

The sentence to keep in the diagnosis: *v2 represents a brown gap as a real thickness contrast (foam on crema vs bare crema) and has no mechanism that can hold such a contrast: the buoyancy pressure c_p²ℓ∇ℓ (c_p = √(g′H_f/D_L) = 2.4–2.5 D/s, the physically correct wave speed) flattens a 6 mm foam step into a 4 mm gap within 1–3 frames (10–25 ms, production dt; 0.1–0.2 s when time-resolved), carrying foam and crema together into the gap; the only resistances are drag β/ℓ = 7–8 /s and a capped (ν_max = 0.5), per-frame, c-weighted viscosity that is inert at τ_y = 0–3 and still creeps at τ_y = 10; the end state is all deposited foam spread uniformly (ℓ = 1 + deposited/A_cup = 1.27).*

The sentences to drop: "crema 成唯一可动相被挤出缝" (crema in the gap increases), "c_p ≈ 2.5 because H_f at the upper bound makes the film incompressible" (c_p = 0.97 fills the gap too; the old closure's advantage is the tracer, not c_p), "D_eff … 5 ms" (wrong regime), and "c 0.95–0.98" as a foam fraction (it is W).

Experiment that would settle the physical side (not the code side, which is settled): measure the white-disc rim speed and the raft thickness after a single low pour (LATTE L309, physics_prior §3 item 4) — if rafts stop at h ≈ 2–5 mm with sharp edges, an arrest mechanism (yield/h_stop) must be added; no refit of c_p/β/τ_y within the current formulation can produce it.
