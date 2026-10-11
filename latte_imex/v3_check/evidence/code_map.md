# Code map: what `latte_imex/solver.py` actually computes (old / twolayer / v2)

Reviewed files (read-only, nothing modified): `latte_imex/solver.py` (843 lines, HEAD after commit 258de63),
`kernels.py`, `pitcher.py`, `action_opt.py`, `joint_cma.py`, `v2_actions.py`, `optimize.py`, `cell_test.py`,
`LATTE_MODEL.md`, `TWOLAYER_CELL_MODEL.md`, `YIELD_FILM_DERIVATION.md`, `PITCHER_MODEL.md` §12–15, `POUR_MODEL_SUMMARY.md`,
fit files `runs/joint_sweep_r2fix/results.json[9]`, `runs/action_opt/cell_fit_N96.json`,
`results/joint_sweep/codexmap/v2_final_fit_N96.json`, `runs/action_opt/joint_cma_v2s/history.json`.
Numbers quoted below were evaluated by calling `Solver.inlet_fields` / `_bulk_flow` at N = 96 with the fitted constants
(script: `scratchpad/review/closure_numbers.py`) on two inlets measured from the pitcher chain on `push_heart`
(NX pitcher, 250 mL): **low pour** Q = 23 mL/s at 2 cm (U⊥ = 9.27 D/s = 0.74 m/s, d = 6.3 mm, footprint 3.9 × 2.5 mm,
|u_in| = 1.88 D/s) and **high cut** Q = 3.3 mL/s at 7–9 cm (U⊥ = 15.7 D/s = 1.25 m/s, d = 1.9 mm, footprint 0.94 mm,
|u_in| = 1.61 D/s). D ≡ D_L = 0.08 m.

一句话结论（中文）：三个闭合里"推开旧膜"的力都只有两种：落点核内的牵引 Λ(v*−u)（局部、速度≈κ_t u_in≈1.7 D/s）和落点堆高后的膜压力 c_p²ℓ∇ℓ（各向同性的重力流扩散）。旧闭合的白度 c = m/ℓ 是被动示踪，拉伸永远不出褐色，褐缝只能是被压缩的原有褐条；twolayer 的体流 |U_s| ≤ 0.2 D/s 基本不起作用、浮上来的奶泡动量为零；v2 的卷吸胞在落点处让表面**向射流汇聚**（div U_s ≈ −126/s），把旧白块往新奶里吸而不是推开，露出的褐环（宽 ≈ R_c ≈ 2–5 mm）又被 c_p = 2.5 D/s 的压力扩散在几毫秒内填回，拟合出的屈服应力 τ̂ = 0.72（或 0.055）在 Δℓ > 0.12 H_f（或 0.01）时就屈服，Bingham 膜实际上是一个 ν_max = 0.5 的牛顿厚膜，每帧只施加一次。所以 v2 出的是软边团块。

---

## 0. Which closure is which (strings, commits, where used)

| `Params.closure` | code path | fitted constants in use | used by |
|---|---|---|---|
| `"chi"` (default) = **old / v1** | `inlet_fields` L331–350; delay patch L367–379, L447–451, L471–474 (only if `tau_d > 0`) | `physics9()` = `runs/joint_sweep_r2fix/results.json[9]` (`action_opt.py` L23–26); `tau_d = 0` in every fit and in the web presets (`export_web` L156, `web/scripts.json` `_default_physics.tau_d = 0.0`) | `action_opt` model `old`, `stage_actions --model old`, `push_heart_video_script ph3_old` |
| `"skin"` | L318–330, L479–503, `_entrainment_flow` L730–754 | none (abandoned 10-09) | nothing current |
| `"twolayer"` | L298–317 (not v2 branch), L384–409, L424–431, L439–442, `_bulk_flow` L590–626 | `cell_fit_N96.json` best (ent 0.369, cell_frac 0.122, …) and `joint_cma/history.json` | `action_opt` model `new` (`phi_foam=1, ent_coef=0`), `cell_test`, `joint_cma --spec cell` |
| `"v2"` | L301–305, L410–423, L521–530, `_ssa_operator/_ssa_step` L628–712, L803–809, L833; `cp` overridden L250–254 | `v2_final_fit_N96.json` (= `joint_cma_v2` gen 14 + per-pattern polish) and `joint_cma_v2s` (structure objective, gen 13) | `joint_cma --spec v2`, `v2_actions`, `stage_actions --model <run>` |

Commit 258de63 (2026-10-10 00:30) changed exactly one line, L766 `if self.P.closure == "twolayer":` → `in ("twolayer", "v2")`. Before it the v2 closure ran with `Ub = Ud = Ds = divUs = None`: drag toward coffee at rest, sources at rest, no skin source, submerged foam not carried, `ent_coef/cell_frac/l_skin` inert. `joint_cma_v2` was restarted after the fix (its history best is gen 14, J 0.551, consistent with commit ea7975b "16 gens"); `runs/action_opt/_stale` holds the bulk-less run. `v2_final` and `v2s` are post-fix.

---

## 1. Governing equations as implemented (one notation)

### 1.0 Common skeleton (all closures)

State per cell: ℓ (film thickness), **q** = ℓ**u** (film momentum), m (white material in the film), plus per closure: m_sub (old delay), f, f_t (submerged foam and its rise-time moment). Grid N×N on [−½, ½]², h = 1/N, cup = disc r ≤ 0.49 (L119); faces open only if both cells are inside. Initial ℓ = 1, q = 0, m = 0 (L233–236). Cells outside the cup are reset to ℓ = 1, q = 0, m = 0 every sub-step (L514–517).

**Meaning of "thickness 1"**
- old and twolayer: ℓ, m are in units of the initial layer; the inlet rate is `S_eff = kQ·Q/D_L³` (`action_opt.py` L74: `Coupling(c_S = kQ/D_L**3)`), so "1" ↔ D_L/kQ = **60 mm** with kQ = 1.326 — i.e. a coffee-depth-like scale, not a film thickness (LATTE_MODEL §0 修正 2). `Params.kappa_Q = 125` is **never read by the solver** (only `Coupling.kappa_Q` for `footprint="v05"`, `pitcher.py` L565); the V0.5 control files use `c_S = 0.05/15 mL/s` ⇒ "1" ↔ 47 mm. The `Inlet.S_eff` docstring (L100, "kappa_Q * Q_received") is stale.
- v2: ℓ, m, f, f_t in units of **H_f** (fitted 5.8–6.0 mm; initial crema = 1); the coffee potential flows use the depth **H** (56–65 mm). `S_eff = Q/D_L³` (kQ forced to 1, `action_opt.py` L72–73), `S_v = S_eff·D_L/H`, `S_f = S_eff·D_L/H_f` (L304). At 23 mL/s: S_eff = 0.0455/s, S_v = 0.065/s, S_f = 0.61/s.

**Control frame** (`advance_frame` L757–812): Δt_f = 1/120 s, shortened to 0.5·min(h, r₁, r₂)/v_scan when the hit point moves (L759–761); inlet fields, Λ, v*, and the bulk flows are frozen in the frame. Sub-steps n = ⌈Δt_f / (0.5·h/(2·u_max))⌉ ≤ 64, u_max = max(|q/ℓ|, |v*| where s+Λ>0, |U_s|, |U_d|) (L768–780). A frame is rolled back and retried with 2n if any field is non-finite or min ℓ ≤ 10⁻³ (L574–577, L783–801).

**One sub-step** (`substep` L353–588), dt:

Step 0 (closure sources, §1.1–1.3) gives: s_ℓ (film mass source), s_m (white mass source), s_mom (mass that carries momentum), Λ, v*, β_B.

Step A, explicit Rusanov (L433–438, `kernels.rusanov_fluxes` L58–78):
  F_f = ½(q_L u_{n,L} + q_R u_{n,R}) − ½ α_f (q_R − q_L), α_f = 2 max(|u_{n,L}|, |u_{n,R}|); q* = qⁿ − dt ∇_h·F.
  (twolayer only, L439–442: q* += dt·β·ℓⁿ·U_b.)

Step B, implicit mass + pressure + drag + traction (L444–474, `stepB_prepare` L165–175, `mass_fluxes` L82–98, `face_coef_and_diag` L179–199, `finish_B` L203–211, `q_from_gradient` L215–233):
  θ = 1 + dt(β_B + Λ/ℓⁿ); **a** = (q* + dt (s_mom + Λ) v*)/θ; coef = c_p² ℓⁿ/θ;
  explicit face flux Ge_f = avg(**a**)·n − ½ w_f (ℓ_R − ℓ_L), w_f = max(|a_L/ℓ_L|, |a_R/ℓ_R|) (a Rusanov-type term on ℓ);
  k_f = dt·avg(coef); solve  ℓ⁺ − dt ∇_h·(k ∇_h ℓ⁺) = ℓⁿ + dt s_ℓ − dt ∇_h·Ge  (Jacobi-PCG, rel. tol 10⁻¹⁰);
  G_f = Ge_f − k_f (ℓ⁺_R − ℓ⁺_L)/h; ℓ⁺ = ℓⁿ + dt s_ℓ − dt ∇_h·G (conservative); q⁺ = **a** − dt·coef·∇_cell ℓ⁺.
  Continuous equivalent: ∂q/∂t = −c_p² ℓ∇ℓ − β_B q − Λ(u − v*) + s_mom v*,  ∂ℓ/∂t + ∇·q = s_ℓ, with pressure, −β_B q and −Λu implicit.
  (old delay, L447–451: **a** −= dt·coef·w·∇_cell m_sub⁺, w = `sub_pressure`; L471–474: q⁺ += dt·s_surf·uⁿ.)

Step C, white transport (L476–511): c = m/ℓⁿ (L360, old ℓ), c_f = van Leer limited upwind value on the sign of G_f (`limited_face_x/y` L107–152, clipped to [c_L, c_R]); m⁺ = mⁿ + dt s_m − dt ∇_h·(G c_f); then m⁺ = clip(m⁺, 0, ℓ⁺) (violation logged in `stats["clip_mass"]`).

Step D (L520–572): old/twolayer: q⁺ += dt ∇_h·(ν ℓ⁺ ∇u) per component, explicit if ν dt/h² < 0.2 else PCG; mixing D (10⁻⁷) explicit. v2: nothing per sub-step when `ssa_per_frame=True` (default); instead once per frame (L803–809) (M + Δt_f K(ν_eff(u⁻))) u⁺ = M u⁻, M = diag(ℓ), K = Gᵀ diag(w) G (§1.3).

Ledger (L585–586, L818–826): `deposited` += ∫s dt (= surfaced white; for twolayer/v2 this is the *surfacing foam*, not the poured volume); `brown_net` += ∫(s_ℓ − s) dt.

### 1.1 Old closure (`"chi"`, physics #9) — L331–350

  z_c = U⊥² / (B_dep d_jet), χ = 1/(1 + z_c^{p}),
  s = χ S_eff K,  Λ = κ_c (1 − χ) S_eff K,
  v* = κ_t u_in + κ_r V_R ê_r,  ê_r = (x − x_hit)/√(R_dep² + |x − x_hit|²), R_dep = √(r₁r₂),
  V_R = min(U⊥ χ^{1/(2p)}, U⊥, √(B_dep d)) (`return_law="v05"`),
  s_ℓ = s_m = s_mom = s, β_B = β. Visible whiteness c = m/ℓ (L836), compared as clip(c/0.6, 0, 1) (`optimize.visible` L39–44).

K is the compact ellipse kernel (1 − a² − b²)² normalised to ∫K = 1 (L257–287) with semi-axes floored at 1.5h (L261).
Numbers (#9, N = 96): low pour χ = 0.67, s_peak = 23.9/s, Λ_peak = 7.9/s, κ_t|u_in| = 1.70 D/s, radial target κ_r V_R = 0.07 D/s (negligible); high cut χ = 0.057, s_peak = 1.3/s, Λ_peak = 14.4/s, κ_t|u_in| = 1.46 D/s. K_peak = 594 (low) / 2684 (high, footprint floored 0.93 → 1.25 mm).

Delay patch (`tau_d > 0`, L367–379; not used in any fit): m_sub⁺ = (m_sub + dt s)/(1 + dt/τ_d); s_surf = (m_sub + dt s − m_sub⁺)/dt; m_sub advected by α u with first-order upwind (α = `sub_advect`); s_ℓ = s_m = s_surf, **s_mom = s** (impact momentum immediate, L363 before the reassignment at L371); extra pressure −c_p² ℓ ∇(w m_sub) (L447–451); surfaced mass gets the local velocity (L471–474).

### 1.2 Twolayer + entrainment cell — L298–317, L384–409, L424–431, L439–442, L590–626

Inlet: Fr² = U⊥²/(g′ d), χ = 1/(1 + (Fr²/Fr_c²)^{p}), F = χ φ S_eff K (`foam_src`, L309), τ_new = max(2U⊥/g′, 10⁻³ s), S_E = ent·Fr·S_eff (L308), R_c = min(cell_frac·2.32·Fr·d, 0.45) (L312), Λ = κ_c S_eff K (L314), v_jet = κ_t u_in (L315). `inlet_fields` returns s ≡ 0 (L316–317): the film mass source is produced inside `substep`.

Coffee (quasi-steady potential flows, Neumann walls, ε_φ = 10⁻⁶, sparse LU once, L604–621):
  ∇·U_v = S_eff(K − ⟨K⟩),  ∇·U_c = S_E(K_c − K) − ⟨·⟩ =: D_s (K_c uniform on the disc r < max(R_c, 2h)),
  U_s = U_v + U_c (surface, `Ub`), U_d = U_v − U_c (deep, `Ud`).

Submerged foam (L386–401): f* = f − dt ∇_h·(U_d f)_upwind + dt F; f_t* likewise with F τ_new; τ_f = f_t*/f* (floor 10⁻³); σ = f*(1 − e^{−dt/τ_f}); f⁺ = f* − σ, f_t⁺ = τ_f f⁺; **s = σ/dt**.

Film sources: s_ℓ = s + D_s⁺ − min(D_s⁻, 0.9(ℓ − m)/dt) (L428–430), s_m = s, **s_mom = 0** (L363 binds `s_mom` to the zero array returned by `inlet_fields`; never reassigned on this path), v* = (s U_s + Λ v_jet)/(s + Λ) (L407–409), drag −β(q − ℓU_s) split as +dt β ℓ U_s explicit (L441) and −β q implicit (β_B = β). Continuous:
  ∂q/∂t + ∇·(q⊗u) = −c_p² ℓ∇ℓ − β(q − ℓU_s) + Λ(v* − u) + ∇·(νℓ∇u)  (no momentum with the surfacing foam),
  ∂ℓ/∂t + ∇·q = s + D_s⁺ − D_s⁻|_capped,  ∂m/∂t + ∇·(c q) = s.
Whiteness c = m/ℓ → clip(c/0.6). "Thickness 1" = 60 mm (same S_eff as old).
Numbers (cell_fit, low pour): χ = 0.92, Fr = 2.97, τ_new = 149 ms, S_E = 0.066/s ≈ S_eff = 0.060/s ⇒ **div U_s at the jet ≈ 0** (1 − ent·Fr = −0.1), ring upwelling +4.7/s over 60 cells, |U_s|_max = 0.14 D/s, |U_d|_max = 0.57 D/s, R_c = 5.3 mm, Λ_peak = 28/s, F_peak = 31/s. High cut: χ = 0.25, div U_s(jet) = −52/s, |U_s|_max = 0.18 D/s.

### 1.3 v2 — L250–254, L301–317, L410–423, L624–626, L628–712, L803–809, L833

Inlet as twolayer but with S_v, S_f (L304): F = χ φ S_f K, S_E = 2·ent·Fr·S_v (L305), Λ = κ_c S_f K, c_p² = g′ H_f/D_L (L253; not fitted). Potential flows with S_v; `divUs = ∇·U_v + D_s` stored (L624) and `Ds = None` (L626) so there are no brown sources.

Film sources (L414–423): s = σ/dt as above (f carried by U_d); s_skin = l_skin·max(∇·U_s, 0); s_ℓ = s + s_skin; s_m = s; s_mom = s_ℓ; Λ_eff = β + Λ; v* = (β U_s + Λ v_jet + s_ℓ U_s)/(β + Λ + s_ℓ); β_B = 0 ⇒ θ = 1 + dt(β + Λ)/ℓⁿ. Continuous:
  ∂q/∂t + ∇·(q⊗u) = −c_p² ℓ∇ℓ − β(u − U_s) − Λ(u − v_jet) + (s + s_skin) U_s + ∇·(ℓ T/ρ),
  ∂ℓ/∂t + ∇·q = s + l_skin max(∇·U_s, 0),  ∂m/∂t + ∇·(c q) = s.
In-plane stress (`_ssa_step` L679–712, once per frame with Δt_f): (M + Δt_f K) u⁺ = M u⁻, K = Gᵀ diag(w) G with rows u_x on x-faces, v_y on y-faces, divergence at cells, shear at corners (L664); w = 2 ν_eff ℓ̄ (×½ on corner rows); ν_eff = min(ν + τ̂(c̄)/(2|D|_e + ε), ν_max), |D|_e² = u_x² + v_y² + u_x v_y + ¼(u_y + v_x)², τ̂(c) = c τ_y + (1 − c) τ_y,crema, evaluated from u⁻ at each row's own location; Jacobi-PCG rel. tol 10⁻⁴.
Whiteness W = 1 − exp(−m/m_o) (L833), compared directly (`visible`, closure == "v2").
Numbers (v2_final, low pour): c_p = 2.49 D/s, χ = 0.70, Fr = 3.64, τ_new = 224 ms, S_E = 0.325/s, R_c = 4.75 mm, F_peak = 85/s, Λ_peak = 104/s, v_jet = 1.84 D/s, |U_s|_max = 0.94 D/s, |U_d|_max = 1.58 D/s, **div U_s(jet) = −126/s**, ring +29/s (38 cells), |U_s| at r = 0.1 D: 0.096 D/s. High cut: χ = 0.088, τ_new = 378 ms, div U_s(jet) = −341/s, Λ_peak = 66/s. v2s (low pour): χ = 0.36, Λ_peak = 362/s, **R_c = 2.8 mm < footprint semi-axis 3.9 mm ⇒ 0 ring cells**; high cut R_c = 2.6 mm, ring +22/s.

---

## 2. Parameter table

Columns: #9 = old closure physics (`results.json[9]`), cell = `cell_fit_N96` (twolayer; inherits #9 for cp, ν, κ_t, p_dep, kQ), v2F = `v2_final_fit_N96.phys`, v2s = `joint_cma_v2s best.phys`. "bound" = `joint_cma.SPACE_V2` range; ⚠ = at/near a bound.

| parameter | unit | used by | role in code | #9 (old) | cell (twolayer) | v2F | v2s | v2 bound |
|---|---|---|---|---|---|---|---|---|
| c_p | D/s | all | pressure −c_p² ℓ∇ℓ (step B coef) | 0.973 | 0.973 | 2.487 (derived) | 2.422 (derived) | = √(g′H_f/D_L) |
| β | 1/s | all | old/twolayer: −βq (β_B); v2: −β(u − U_s) via θ, β/ℓ on u | 2.407 | 3.258 | 8.00 | 7.00 | 0.5–20 log |
| ν | D²/s | all | old/twolayer Laplace viscosity; v2 Newtonian part of ν_eff | 1.44e-3 | 1.44e-3 | 8.88e-4 | 3.72e-3 | 1e-4–1e-2 log |
| D | D²/s | all | mixing of c (step D) | 1e-7 fixed | 1e-7 | 1e-7 | 1e-7 | fixed |
| κ_Q | 1/D | **none** (dead in solver; `Coupling.kappa_Q` only for footprint "v05") | — | 125 | 125 | 125 | 125 | — |
| kQ | — | old, twolayer | S_eff = kQ Q/D_L³ (coupling, not `Params`) | 1.326 | 1.326 | 1 (forced) | 1 | — |
| κ_c | — | all | traction strength Λ = κ_c(1−χ)S K (old) / κ_c S K (twolayer) / κ_c S_f K (v2) | 0.665 | 0.784 | 0.287 | 0.966 | 0.1–3 log |
| κ_t | — | all | tangential target κ_t u_in | 0.905 | 0.905 | 0.977 ⚠ | 0.919 | 0.3–1.0 |
| κ_r | — | old | radial return target κ_r V_R | 0.00847 (≈0) | — | — | — | — |
| B_dep | D/s² | old | deposition switch z_c = U²/(B d) | 1718.6 | — | — | — | — |
| p_dep | — | all | exponent of χ | 1.556 | 1.556 | 1.407 | 1.641 | 1–4 |
| return_law | — | old | "v05" cap on V_R | v05 | — | — | — | — |
| τ_d, sub_pressure, sub_advect | s, —, — | old (only if τ_d>0) | delay reservoir | 0 (off), 0.5, 0.3 | off | off | off | — |
| g′ | D/s² | twolayer, v2 | Fr², τ_new, (v2) c_p | — | 124.7 | 82.8 | 81.3 | 40–110 log |
| Fr_c² | — | twolayer, v2 | χ = 1/(1+(Fr²/Fr_c²)^p) | — | 40.75 | 23.8 | 9.49 | 5–100 log |
| φ_foam | — | twolayer, v2 | foam fraction of the poured volume | — | 0.939 | 0.337 | 0.317 | 0.2–0.7 |
| ent | — | twolayer, v2 | S_E = ent·Fr·S (v2: 2·ent·Fr·S_v) | — | 0.369 | 0.688 | 0.369 | 0.2–1.5 log |
| cell_frac | — | twolayer, v2 | R_c = cell_frac·2.32·Fr·d | — | 0.122 (bound 0.1) | 0.0898 | 0.0528 ⚠ (bound 0.05) | 0.05–1 log |
| H | m | v2 | S_v = Q/(H D_L²) | — | — | 0.0561 | 0.0647 | 0.03–0.07 |
| H_f | m | v2 | film unit; S_f; c_p | — | — | 0.00597 ⚠ | 0.00577 ⚠ | 0.001–0.006 log |
| m_o | H_f | v2 | W = 1 − e^{−m/m_o} | — | — | 0.934 ⚠ (= 5.6 mm) | 0.229 (= 1.3 mm) | 0.05–1 log |
| τ_y | D²/s² | v2 | foam yield stress/ρ in ν_eff | — | — | 0.718 | 0.0551 | 0.01–3 log |
| τ_y,crema | D²/s² | v2 | crema yield | — | — | 0 fixed | 0 | fixed |
| l_skin | H_f | v2 | s_skin = l_skin max(∇·U_s,0) | — | — | 0.03 fixed | 0.03 | fixed |
| yield_eps | 1/s | v2 | regularisation ε | — | — | 0.02 fixed | 0.02 | fixed |
| ν_max | D²/s | v2 | cap of ν_eff | — | — | 0.5 fixed | 0.5 | fixed |
| D_L | m | v2 | unit conversions | — | — | 0.08 | 0.08 | fixed |
| m_opaque | — | skin only | — | 0.02 | — | — | — | — |

Actions (h, q, s) fitted jointly (bounds h 0.3–2, q 0.6–2, s 0.8–1.3, log): v2F tulip h = 2.0 ⚠, swan 1.84, heart 1.61; v2s tulip 1.97 ⚠. Numerics: N = 96 for all fits (h = 0.83 mm; kernel floor 1.5h = 1.25 mm; cell-disc floor 2h = 1.67 mm), cfl 0.5, Δt_f 1/120 s, cg_tol 1e-10, ssa_tol 1e-4, `ssa_per_frame=True`, l_floor 1e-3, ε_φ 1e-6.

Loss (`optimize.loss_fields` L47–53): final frame only; L1 of Gaussian-blurred whiteness (σ = N/48 cells = 1.5 % D = 1.2 mm) + 0.5(1 − IoU at threshold 0.5) + 0.5|ΔA|; whiteness = clip(c/0.6) (old, twolayer) or W (v2). `gap_stats` (L56–69) counts brown only inside a 6 mm binary closing of the white set, so gaps wider than ≈ 6 mm are not counted as gaps.

---

## 3. How EXISTING film is moved when new milk arrives

Common to all three: the only terms that act on film outside the footprint are the pressure −c_p² ℓ∇ℓ (non-local, isotropic, transmitted as a gravity current of speed ≈ √(c_p² ℓ) and, against drag, as a diffusion with D_eff = c_p² ℓ²/β) and the Rusanov-advected momentum ∇·(q⊗u) of film that is already moving. Everything else is local to the footprint kernel K (support ≈ 2r₁ × 2r₂ ≈ 8 × 5 mm at the low pour, 2.5 mm at the cut).

| | old (#9) | twolayer (cell_fit) | v2 (v2F) |
|---|---|---|---|
| direct momentum into the film at the footprint | s v* (s_peak 24/s × 1.7 D/s) + Λ(v* − u), Λ_peak 8/s (low) 14/s (high); target v* = κ_t u_in + 0.07 ê_r ≈ 1.5–1.7 D/s along the jet's horizontal velocity | Λ(v* − u), Λ_peak 28/s, but v* = (s U_s + Λ v_jet)/(s+Λ) ≈ ½ v_jet at the low pour (s ≈ Λ) ⇒ ≈ 0.9 D/s; surfacing foam enters with **zero** momentum (s_mom = 0) ⇒ a brake −s u | Λ(v_jet − u), Λ_peak 104/s (v2s 362/s): film inside K pinned to v_jet ≈ 1.8 D/s within ~10 ms; sources enter with U_s (converging at the jet) |
| drag reference | rest (−βq, β = 2.4/s; coasting distance u/β ≈ 0.7 D for u = 1.7) | coffee U_s: \|U_s\| ≤ 0.14 D/s, 0.09 D/s at r = 0.1 D — negligible against traction/pressure | coffee U_s: inside R_c ≈ 4.8 mm the surface **converges** to the jet (div −126/s, \|U_s\| up to 0.9 D/s at the footprint edge); outside R_c only U_v (0.1 D/s at r = 0.1, 0.04 at 0.2); β/ℓ = 8/ℓ /s |
| pressure push | pile grows at s_peak ≈ 24 "units"/s (= 2.4 units ≈ 14 cm of coffee-depth equivalent per 0.1 s dwell): c_p² ℓ∇ℓ with c_p = 0.97 ⇒ gravity speed √(c_p²ℓ) ≈ 1.8 D/s at ℓ ≈ 3.4; D_eff = 0.39 ℓ² D²/s | same c_p, pile from s ≈ 31/s after τ_new = 0.15 s, D_eff = 0.29 ℓ² | c_p = 2.49: gravity speed 3.5 D/s at ℓ = 2, D_eff = 0.77 ℓ² D²/s (3.1 at ℓ = 2) — a 1 cm step relaxes in ~5 ms |
| in-plane stress | ν = 1.4e-3 (Laplace), negligible | same | ν_eff ≤ ν_max = 0.5 D²/s applied once per 8.3 ms frame ⇒ smoothing length √(ν_max Δt_f) = 5.2 mm per frame; yield only resists if c_p²Δℓ < τ_y ⇒ Δℓ < 0.116 H_f = 0.7 mm (v2s: 0.009 H_f = 0.05 mm) — every real pile is yielded; τ_y/ε = 36 ≫ ν_max so unyielded cells just have ν = 0.5 |
| what sets the push speed | κ_t·u_in (jet horizontal velocity from the pitcher: exit-speed component + pitcher velocity, capped at U⊥ tan 60°), then c_p and β | same, halved by the blended v* and the zero-momentum foam | κ_t·u_in inside K; c_p (= √(g′H_f/D_L)) and β/ℓ outside; U_s pulls inward |
| non-local reach | pressure + inertia only | pressure + inertia; U_s too weak | pressure (fast, isotropic) + U_s (inward within R_c) |

Note on the delay patch (old, τ_d > 0): the mound term −c_p² ℓ∇(w m_sub) is the only term in any closure that pushes the *existing* film outward from the footprint **before** white appears there; this is why PITCHER_MODEL §14 round 2 was the only run with 6–10 mm bands. It is off in every fit.

---

## 4. The brown between two white blocks, step by step (push_heart: stage k+1 starts at B behind block k, pushes forward)

The measured chain gives for stage 2: 30–57 mL/s bursts during the 0.18 s backward reposition and the first 0.3 s at B, then 1–2 mL/s for the last 0.5 s of the stage (§5 item 12) — so "new milk" arrives mostly as a burst at B.

**Old closure.**
1. The jet lands on brown film (ℓ = 1, c = 0). s = χSK adds white at c = 1: c at the kernel peak reaches 0.7 in 0.1 s (s = 24/s) — the footprint is white almost immediately.
2. s v* and Λ(v* − u) accelerate the film in K forward at ≈ 1.7 D/s; the pile (ℓ↑) adds an isotropic pressure push. The brown strip between B and block k's back edge (initially ≈ 0.06 D ≈ 5 mm) is pushed forward and **compressed** against block k (ℓ↑, c unchanged = brown): its width can only shrink.
3. c = m/ℓ is a passive tracer (step C uses the same flux as ℓ): stretching white film changes ℓ, never c ⇒ **no brown can ever be created by thinning**; D = 10⁻⁷ is inert. Brown appears only where originally-brown surface fluid is transported (the strip ahead of the jet, brown from the pattern's sides folded in by the pressure/traction flow).
4. New white deposited along B → C overwrites everything the kernel passes over; the surviving brown is the compressed strip squeezed between the new trail and block k, with edges smeared over ~2–3 cells (2 mm) by van Leer and further softened because Rusanov diffuses the velocity (ν_num = |u|h ≈ 0.02 D²/s at u = 2) that shears the strip. Result: 1–3 mm stripes, never wider than the strip the jet started on.

**Twolayer.**
1. Jet at B: foam goes to f, surfaces after τ_new ≈ 0.15 s carried outward at depth by U_d (≤ 0.57 D/s inside R_c = 5.3 mm): white appears as a diffuse disc of radius ≈ R_c around the (moving) hit point, with zero momentum.
2. Meanwhile Λ(v* − u) pushes the brown strip and block k forward at ≈ 0.9 D/s (blended target), and the pile pressure spreads things isotropically.
3. The ring r_K < r < R_c gets brown film from D_s⁺ at 4.7 "units"/s (vs ℓ = 1): in 0.2 s c in the ring halves ⇒ a grey ring (visible = clip(c/0.6)). This is the only brown-creation mechanism; it is a source added to ℓ (no exposure by stretching — c is still a tracer here).
4. D_s⁻ at the jet ≈ 0 at the low pour (S_E ≈ S_eff), so nothing is subducted. Net: compressed original strip + a grey ring ≈ 1 footprint wide, mixed into c by sources ⇒ "变糊", still thin.

**v2.**
1. Jet at B on crema (ℓ = 1, m = 0, W = 0). F = χφS_f K → f; surfaces after τ_new = 0.22 s (0.38 s at the cut) while U_d carries it outward to r ≈ R_c (|U_d| ≤ 1.6 D/s); first-order upwind spreads it a further ≈ 4.5 mm. White appears 0.2–0.4 s later as a soft disc 2–5 mm around where the jet *was* (the hit point moves ≈ 0.09 D/s during the stage).
2. Film in K is pinned to v_jet (1.8 D/s forward) by Λ; film in the ring is stretched by ∇·U_s = +29/s (e-fold 35 ms) and relaxes to ℓ = l_skin = 0.03, m → 0, W → 0: a brown ring of width R_c − r_K ≈ 1–2 mm (v2F) forms within ≈ 50 ms by kinematic exposure — this is the correct physical mechanism in principle. (v2s: R_c = 2.8 mm < footprint 3.9 mm ⇒ no ring at the low pour at all; the fit drove cell_frac to its lower bound.)
3. Inside R_c the surface **converges** on the jet (div −126/s): block k's near edge is pulled *toward* the new milk at up to ~1 D/s — the opposite of the push that makes the real gap — and the sources arrive with that inward U_s.
4. The pile (new foam + sucked-in old foam) spreads by pressure with D_eff = 0.77 ℓ² D²/s; the brown ring (≈ R_c) is refilled in ≈ R_c²/D_eff ≈ 5 ms/ℓ². Nothing resists: yield needs Δℓ < 0.7 mm; ν_max = 0.5 only smooths velocity differences over 5 mm per frame and is applied *after* the sub-steps in which the spreading already happened (Lie splitting at frame level).
5. W = 1 − e^{−m/m_o} with m_o = 0.93 H_f = 5.6 mm (v2F; W = 0.5 needs 3.6 mm of foam) renders moderate foam as grey, so edges are soft gradients. Result: brown exists only transiently in the ring and at the far edge; the blocks merge into one smooth blob. With v2s (m_o = 1.3 mm) the edge is sharper but the cell is inside the footprint and τ_y is 13× smaller.
6. No term in v2 transmits a *directed, block-wide* forward push to block k; the only directed force is Λ inside K.

---

## 5. Bugs, inconsistencies, hidden assumptions (line numbers in `solver.py` unless stated)

1. **twolayer: surfacing foam enters with zero momentum and the traction target is blended.** L363 `s_mom = s` binds the zero array from `inlet_fields` (L316); L402 rebinds `s` but not `s_mom`; `stepB_prepare` therefore uses f = dt·Λ only, while v* (L407–409) is (s U_s + Λ v_jet)/(s+Λ). Known (LATTE_MODEL §0 修正 1) but still live; every twolayer/cell fit (cell_fit, joint_cma) was done with it.
2. **v2 bulk flow wired only at 258de63** (L766). Confirmed by `git show`: before it `Ub/Ud/Ds/divUs` were None in v2 (L386, L404, L414). `joint_cma_v2` gens 0–5 pre-fix are archived (`runs/action_opt/_stale`); v2F/v2s are post-fix.
3. **Kernel radius floored to 1.5h, d_jet not** (L261 vs L297): at N = 96 the high-cut footprint 0.93 mm becomes 1.25 mm (area ×1.8, K_peak 2684 instead of ≈ 4800) while χ uses the physical d. Source peak densities, Λ_peak, the ring cell count (38 vs more at 256) and therefore the fitted κ_c, Fr_c², cell_frac are resolution-dependent beyond ordinary discretisation error. 96² fits ≠ 256² renders.
4. **Entrainment cell at/under the footprint scale.** cell_frac is at the lower bound in v2s (0.0528 vs 0.05) and near it in v2F (0.09) and cell_fit (0.122 vs 0.1): R_c = 2.6–5.3 mm. In v2s at the low pour R_c (2.8 mm) < r₁ (3.9 mm): the upwelling "ring" has zero cells (`closure_numbers.py`), K_c sits inside K, and the cell only reshapes the footprint source. The optimiser is switching the cell off.
5. **Other active bounds in the v2 fits**: H_f = 5.97/5.77 mm (bound 6 mm) in both; m_o = 0.934 (bound 1.0) in v2F; κ_t 0.977 (bound 1.0); tulip h = 2.0 (bound) in both. Fitted "physics" is partly bound-defined; m_o = 5.6 mm of foam for 63 % whiteness contradicts the document's own 0.3–1 mm estimate (LATTE_MODEL §2.7).
6. **Yield stress cannot hold a pile at the fitted values.** Pressure force c_p²ℓ∇ℓ vs yield force ∼ τ_y ℓ/L ⇒ yielded when c_p²Δℓ > τ_y: Δℓ > 0.116 H_f (v2F) / 0.009 H_f (v2s). Combined with ν_max = 0.5 ≪ τ_y/ε = 36 (v2F) the "Bingham film" is a Newtonian film of viscosity 0.5 D²/s in unyielded cells; and per-frame application (L803–809, `ssa_per_frame=True`) lets the pressure/drag sub-steps spread the pile freely for 8.3 ms before any stress acts. LATTE_MODEL §5.3 quotes a mean whiteness difference 0.023 vs per-substep — that is a statement about the fitted blob, not about edge preservation.
7. **Surface convergence at the jet (v2) is the opposite of a push.** `_bulk_flow` L615–624: div U_s = S_v K − S_E K + S_E K_c; with S_E = 2·ent·Fr·S_v = 5·S_v (v2F, Fr 3.6) the surface inside the footprint converges at −126/s (−341/s at the cut). LATTE_MODEL §7.6 states this as a prediction ("射流处表面总是汇聚的"); for a push pattern it means the old block is pulled into the new milk, and the surfacing foam inherits that inward velocity through s_ℓ U_s (L420–421).
8. **Uniform sink of U_v applied to the surface** (L610, L615, L624): ∇·U_v = S_v(K − ⟨K⟩) includes −S_v/A_cup ≈ −0.086/s everywhere (23 mL/s, v2F). As a *surface* velocity this compresses the whole film (ℓ and m grow ≈ 9 %/s far from the jet) and biases s_skin; physically the level rises, the surface does not converge. (twolayer uses only D_s for sources, so it affects only its drag target.)
9. **Coffee has no inertia or memory** (`_bulk_flow` L600–603): U_s, U_d, D_s, divUs are recomputed from the current inlet each frame and are None the frame the inlet stops (S ≤ 0) — then drag is toward rest. No post-pour drift, no circulation persists through the 0.18 s repositions; the ε_φ = 10⁻⁶ Neumann potential flow is irrotational and depth-uniform (no boundary layer under the film).
10. **Skin source fixes the gap's thickness, not the physics** (L415–418): in the ring ℓ → l_skin = 0.03 exactly (balance l_skin·divU_s = ℓ·divU_s), m → 0. The brown is a regularisation state; sensitivity to l_skin is untested (LATTE_MODEL §7.7).
11. **CFL does not see c_p or the velocities generated inside a frame** (L768–780): u_max is taken at frame start; the gravity-current speed √(c_p²ℓ) ≈ 3.5 D/s (v2, ℓ = 2) exceeds any u_max seen at the start of a pour; violations are only caught by the non-finite/ℓ-floor retry (L574–577) and otherwise show up as extra Rusanov dissipation (α = 2 max|u|).
12. **Action-chain artefacts in push_heart** (measured with `pitcher.BaristaScript`, NX geometry, kQ 1.326; `records_table`): during each 0.18 s zero-flow reposition 30–34 mL/s (stage 1→2) / 14–26 mL/s (3→4) still *arrive* while the hit point sweeps backwards across the block (lip buffer τ_stop = 0.15 s + 80 ms flight); after the reposition the feed-forward wrist overshoots (Q_feed 99.6 mL/s at t = 1.66 s, Q_hit 51–57 mL/s), then the flow collapses to 1–2 mL/s for the last 0.5 s of stage 2 (tilt stuck at 54°, `flowing=1`), so stage 2 (wanted 22.5 mL/s) delivers a burst at B and almost nothing along B → C. All closure comparisons on push_heart are made under this input; it is not a solver property.
13. **Stale constants/docstrings**: `pitcher.PHYSICAL_PARAMS` (L586, B_dep = 2.5) assumes the old c_U = 0.263/0.70 scaling and is exported as `BaristaScript.params` (L856) but never used by `action_opt`; `Params.kappa_Q` dead; `Inlet.S_eff` docstring (L100); comment L251 says c_p² = g′H_f/D_L² while the code (correctly for g′ in D/s²) divides by D_L once; `Params.cp` default 0.3 survives in `as_dict()` for v2 even though the solver uses 2.4–2.5.
14. **Step C uses c = m/ℓⁿ (L360) with the new flux G** (L477–507); the clip at L511 (`clip_mass`) hides the inconsistency. `m` can also be clipped *up* by ℓ⁺ only through the 0 bound; fine but means m is not exactly conserved.
15. **Delay-closure momentum**: the surfaced mass gets the pre-step velocity uⁿ (L471–474) after q⁺ was already computed from **a** — an O(dt) inconsistency; irrelevant now (τ_d = 0 everywhere).
16. **Gap metric blind spot**: `gap_stats` closes with a 6 mm disc, so a brown band wider than ≈ 6 mm between two blocks is not "inside the outline" and is not scored as a gap; the objective therefore cannot reward the 4–9 mm push_heart gaps it was built for when they exceed 6 mm.
17. **Loss uses the final frame only** (`run_one` L85–88); c_p, τ_new, β — the parameters that govern *motion* — are only constrained through end-state shape (POUR_MODEL_SUMMARY §3 says the same for c_p).

---

## 6. Numerical dissipation (N = 96, h = 0.83 mm; values from `closure_numbers.py`)

| source | where | magnitude |
|---|---|---|
| Rusanov momentum flux, α = 2 max\|u\| | step A (`rusanov_fluxes`) | ν_num ≈ \|u\|·h = 0.005 D²/s at u = 0.5, 0.021 at u = 2 — 4–15× the physical ν (1–4e-3); acts on q wherever the film moves (jet trail, spreading fronts) |
| Rusanov-type term on ℓ, w = max\|a/ℓ\| | step B explicit flux (`mass_fluxes`) | ν_num,ℓ ≈ \|u\|h/2 = 0.003–0.010 D²/s: thins/rounds every ℓ step that moves |
| implicit Euler on pressure (k_f = dt c_p² ℓ/θ) | step B | gravity waves damped per step by 1/√(1 + (c_p k dt)²): c_p = 2.45, λ = 4 mm, dt = 1.3 ms: 0.93/step, 3e-3 per 0.1 s; λ = 10 mm: 0.99/step, 0.38 per 0.1 s; c_p = 0.97, λ = 4 mm: 0.99/step. The implicit solve also acts as a diffusion of ℓ with coefficient dt·c_p²ℓ/θ (0.008 D²/s at c_p = 2.5, dt = 1.3 ms) — on top of the physical drag-limited spreading c_p²ℓ²/β |
| van Leer limited upwind on c | step C | TVD; interfaces smeared over 2–3 cells (≈ 2 mm); corners of stripes rounded; stripes < 2 cells cannot be represented — a 4 mm gap is 5 cells |
| first-order upwind on f, f_t (twolayer/v2) and m_sub (delay) | L389–394, L374–379 | ν_num = \|U_d\|h/2 = 0.005 D²/s at 1 D/s ⇒ the submerged foam spreads ≈ 4.5 mm during a 0.3 s rise even without U_d; white always surfaces blurred |
| explicit viscosity (old) | step D | ν dt/h² ≈ 0.013 ≪ 0.2 ⇒ explicit, additional O(dt) error only |
| per-frame SSA (v2) | L803–809 | Lie splitting at Δt_f = 8.3 ms: velocity differences smoothed over √(ν_eff Δt_f) ≤ 5.2 mm once per frame; Picard lag of ν_eff on u⁻; PCG tol 1e-4 (velocity error ~6e-4 per the commit note) |
| frame freezing | `advance_frame` | inlet, Λ, v*, U_s, U_d constant for 8.3 ms (or shorter when scanning); u_max from frame start (item 11) |
| kernel floor 1.5h, cell floor 2h | L261, L618 | sources widened at 96² (item 3) |
| sub-step count | L778–780 | u_max = 2 D/s ⇒ dt = 1.3 ms, 7 sub-steps/frame; max 64 |

---

## 7. Pitcher → inlet → run_one (for completeness)

- `Coupling` (`pitcher.py` L533–579): x = x_hit/D_L; S_eff = c_S·Q; U⊥ = c_U·U⊥_SI; u_in = c_u·u_h, capped so |u_in| ≤ U⊥ tan 60° (L561–564); `footprint="physical"`: r₁, r₂ = J["r1"], J["r2"]/D_L from the lip-flux second moment with capillary rounding (`jet` L315–324), φ = lip tangent, d_jet = 2√(r₁r₂); `use_arrivals=True` feeds what lands this frame (parcels with flight windows, L479–504). Class defaults c_S = 0.05/15e-6, c_U = 0.263/0.70, c_u = 0.5/0.15 are the V0.5 effective units; every fit instead builds `Coupling(c_S = kQ/D_L³, c_U = c_u = 1/D_L)` (`action_opt.py` L74, `joint_run.py` L26, `joint_sweep.py` L89), with kQ = 1.326 (old/twolayer) or 1 (v2).
- `Inlet` (`solver.py` L95–107): active, x_hit, S_eff, u_in, U⊥, d_jet, r₁, r₂, φ, scan_speed. The solver never sees Q, H or the pitcher.
- `Move`/`Barista` (L591–741): intent (path in cup units, spout height z, wanted Q, wobble, Bezier); feed-forward tilt from the weir law + PI trim, 60°/s swing when not flowing, hold-back 4°/s at Q_want = 0; spout height rate-limited 0.25 m/s; cup-collision lift; stream outside r > 0.47 deactivated (L739). `v05_moves` (L791–811): Q = S·(15e-6/0.05), z from the footprint-radius table `_R_Z` (0.035 → 1.5 cm … 0.0015 → 9 cm), zero-flow bridges at ≥ 3 cm, "move away" appended. `scaled_moves` (`action_opt.py` L29–45): h scales z of pouring moves below 5 cm (clip 0.8–5 cm), q scales their Q, s scales positions/Bezier/wobble (clip ±0.45).
- `run_one` (`action_opt.py` L52–98): cached per `<out>/<model>/<pattern>_<tag>_N<N>.{npy,json}`; `physics9()` gives ph (cp, β, ν, κ_c, κ_t, κ_r, B_dep, p_dep) and kQ; `pd = dict(ph, D=1e-7, kappa_Q=125, return_law="v05"); pd.update(over)` so a v2 `over` inherits #9's κ_t etc. only where SPACE_V2 does not override them (it overrides all 14); frames advanced with `script.sample(sol.t + 0.5·fdt)`; result = `concentration()` → `visible` → `loss_fields` against `target_at(name, N)`.
