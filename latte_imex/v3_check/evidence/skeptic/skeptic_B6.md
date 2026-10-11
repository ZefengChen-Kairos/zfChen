# Skeptic B6 — "水下奶泡是被动示踪；浮出率处处相同还填新缝；上浮时间 0.15–0.38 s 比首白延迟 0.9–2.1 s 小 4–10 倍；无团块/单泡区分；§14 水下施压被删且无替代"

Verdict: **confirmed with nuance**. Four of the five sub-claims trace to the code and reproduce numerically; one
("无团块/单泡区分") is wrong as worded, and the "4–10× too fast" item is a true model-vs-video discrepancy whose
*cause* is not settled (the physics prior itself puts cluster rise at 0.05–0.3 s, i.e. agrees with the model).

Script: `skeptic_B6_check.py` (read-only; N = 96, stationary low-pour jet at the origin, v2F and cell_fit physics,
0.6 s; `argv[4] == "ud0"` monkey-patches `Solver._bulk_flow` so `Ud = 0`, foam surfaces where it was poured).

## 1. What the code does (solver.py, closure in ("twolayer", "v2"))

- **State**: `self.f` (submerged foam, L240) and `self.ft = f·τ` (L241). `inlet_fields` L309–310:
  `foam_src = χ φ_foam S_f K`, `tau_new = max(2 U⊥/g′, 1e-3)`. L384–401 (substep): f, ft advected first-order upwind
  with `Ud = U_v − U_c`; `+dt·foam_src`; `τ_f = ft/f`; `surf = f(1 − e^{−dt/τ_f})`; `s = surf/dt` (L402).
- **No volume**: ℓ's mass equation `rhs = l + dt·s_l − dt·div(G)` (L458) and `l_new = l + dt·s_l − …` (L466) contain
  only the *surfaced* s (plus s_skin / D_s); f never enters ℓ. No pressure: step B (`stepB_prepare` L445) sees
  `lsafe`, `c_p²`; the only sub-surface pressure term is L447–451 `a −= dt·coef·sub_pressure·∇m_sub`, gated by
  `msub_new is not None`, which requires `P.tau_d > 0 and self.tau_cur is not None` (L370); `tau_cur` is set **only**
  on the chi path (L331–332, after the twolayer/v2 `return` at L317). Runs confirm `tau_cur=None, msub max=0` for v2F
  and cell_fit. No momentum of its own: the surfacing foam arrives with U_s (v2, L420–421 `s_l·ubx`) or with zero
  momentum (twolayer, `s_mom` bound to the zero array, code_map item 1). **Sub-claim 1 confirmed.**
- **Uniform surfacing rate**: τ_new depends only on the inlet (U⊥, g′), so every parcel poured by a given inlet has
  the same τ; surfacing removes f and ft in the same proportion (`tl_new = tau_f·f_tl`, L399) so τ_f never ages.
  Measured: f-weighted mean τ_f = 224 ms = τ_new at every frame (v2F); min 224 ms; the "max" up to 0.96 s occurs only
  in cells with f ≈ 1e-14 (upwind tail noise). So the per-unit-f surfacing rate 1/τ_f is spatially uniform and the
  surfacing *flux* f/τ_f is simply "wherever U_d has carried the foam". **Confirmed.**
- **"还填新缝"**: at depth U_d = U_v − U_c is *outward* inside R_c (U_c inward at the surface, mirror at depth), so the
  plume crosses the upwelling ring (footprint < r < R_c, the only designed brown-making zone, B5) before most of it
  surfaces. v2F low pour (χ 0.70, τ_new 224 ms, R_c 4.75 mm; 56 footprint cells, 40 ring cells), steady split of the
  surfacing flux: **12 % footprint / 17 % ring / 71 % outside R_c**, mean surfacing radius 5.8 mm > R_c. Per cell the
  ring receives *more* new white than the footprint (0.17/40 vs 0.12/56). Cumulative in-ring surfacing 0.0118 D²·H_f
  by 0.6 s vs ring foam mass 0.0038 (the rest exported by the +31 /s divergence). Ring whiteness (median W): 0.19 at
  0.10 s, 0.33 at 0.2 s, 0.45 at 0.3 s, 0.61 at 0.6 s — the "gap" is white. cell_fit (twolayer): 33/30/37 %, ring W
  median 0.34 at 0.6 s (max 0.94). **Confirmed** — with a nuance from the ablation (§2).

## 2. Ablation: is the ring filled by in-ring surfacing or by the pile spreading?

`Ud = 0` (foam surfaces where poured, 100 % in the footprint, 0 % in the ring), v2F low pour:
ring median W = 0.00 (0.1 s), 0.04 (0.2 s), 0.18 (0.3 s), 0.44 (0.4 s), 0.62 (0.5 s), 0.68 (0.6 s).
So the uniform surfacing whitens the ring ~0.15–0.2 s earlier, but the pile's own pressure spreading
(c_p = 2.49 D/s, Δℓ ≈ 1–2) fills the ring anyway within 0.4–0.5 s. **Corrected wording**: the passive, uniformly
surfacing foam is *one* of two fillers; removing it alone would not keep a gap open in v2.

## 3. Rise time vs first-white delay

- Numbers: τ_new = 149 ms (cell_fit), 224 ms (v2F low), 378 ms (v2F cut) — the claim's 0.15–0.38 s is right.
- Model first-white (stationary low pour, W = 1 − e^{−m/m_o}, m_o 0.934): W > 0.1 at **0.05 s**, > 0.3 at 0.13 s,
  > 0.5 at **0.25 s**; twolayer (W = clip(c/0.6)): W > 0.5 at **0.05 s**, > 0.8 at 0.07 s. Video (video_obs §0):
  0.9–2.1 s, median 1.3 s. Ratio 4–8× (v2F), 18–40× (twolayer). **The discrepancy is confirmed and if anything
  understated for twolayer.** (High cut: v2F W_max 0.06 after 0.6 s, χ 0.09 — no white, consistent with the video.)
- But the *attribution* "上浮时间太短" is not supported by the prior the claim cites: physics_prior §2.3 / §3f gives
  intact-cluster return in **0.05–0.3 s** (U ≈ √(g′a) 0.15–0.25 m/s), the same order as 2U⊥/g′. The 6.5 s Stokes
  figure (docs_map §4.2) is for *dispersed* 50 µm bubbles, which the prior says "never return / turn the canvas tan",
  i.e. the fraction the model throws away as (1 − χ). So slowing τ_new to ~1 s would contradict the prior. The 0.9–2.1 s
  must come from something the model lacks (cup tilt + initial pour geometry, foam fraction of the first milk out of
  the pitcher — `pitcher.py` has no foam field, φ_foam is a constant —, opacity threshold after dispersal, …) —
  **uncertain; needs the side-view clear-cup pour (conv O-10/O-11) or first-white vs height/flow from the top-view clips.**

## 4. "无团块/单泡区分" — refuted as worded

χ = 1/(1 + (Fr²/Fr_c²)^p) (L302, L322) *is* a survive-as-cluster vs. destroyed switch on U⊥: the destroyed
fraction never surfaces, which is exactly the prior's "single bubbles take 3–30 s, many never". Threshold speed
U_c = √(Fr_c² g′ d): v2F (Fr_c² 23.8, g′ 6.6 m/s², d 6.3 mm) **1.0 m/s**; v2s (9.49, 6.5 m/s²) 0.62 m/s; cell_fit
(40.75, 10 m/s²) 1.6 m/s — bracketing the prior's v_crit 0.7–0.9 m/s. Low pour χ = 0.70–0.92, cut χ = 0.09 (v2F):
the model already gives "低位出白 / 高位不出白". What *is* missing: (i) the threshold scales with jet diameter d
instead of bubble radius R (We_R = ρv²R/σ); (ii) the broken fraction vanishes instead of lightening the canvas later;
(iii) nothing distinguishes cluster size (a 3–10 mm cluster vs. foam that re-aggregates), and the jet momentum of the
broken fraction is still delivered through Λ. Keep as a minor item, not as "no distinction".

## 5. "§14 水下施压被删且无替代"

Deleted: yes, in code (dead path, §1) and in doc (PITCHER L259 "去掉的：…sub_pressure / sub_advect / tau_d"; LATTE
§0/§2 give no mound term). "无替代" needs qualification: v2 still has two pre-white pushes — the jet traction
Λ(v_jet − u) (κ_c S_f K, peak ≈ 98–104 /s, directed along u_in, same mechanism as the old closure that gave 1–3 mm
bands) and the basal drag toward U_v (≈ 0.06 D/s at 1 cm, 10× weaker than the old κ_r V_R). B1's φ_foam = 0 run shows
the film ahead of the jet moving away at 0.3–0.7 D/s from Λ alone. What has no replacement is the **isotropic,
volume-proportional** push of a growing submerged mound (−c_p² ℓ ∇(w m_sub)), the only term that ever produced the
6–10 mm bands (PITCHER §14 round 2; code_map §3 note). With f carrying no volume, a v2 push can only be as wide as the
jet's traction footprint.

## 6. Corrected statement

"In twolayer/v2 the submerged foam f is a pure tracer: it adds no thickness, no pressure (the L447–451 mound term is a
dead path, `tau_cur` is never set outside closure='chi') and no momentum of its own; it surfaces with a uniform rate
constant 1/τ_new (f-weighted τ_f = τ_new to the frame) wherever U_d has carried it, so at the low pour 17 % of the
flux surfaces inside the upwelling ring (more per cell than in the footprint) and the ring is W ≈ 0.6 by 0.6 s —
though the pile's pressure spreading fills it by 0.5 s even with no in-ring surfacing. First white appears at 0.25 s
(v2F) / 0.05 s (twolayer) vs 0.9–2.1 s in every clip; the rise time 2U⊥/g′ = 0.15–0.38 s is however consistent with
the physics prior's cluster return (0.05–0.3 s), so the delay is a missing mechanism, not a wrong τ. χ(Fr) already
separates surviving clusters from destroyed foam with a threshold 0.6–1.6 m/s (prior 0.7–0.9); what is missing is a
bubble-scale threshold and the fate of the broken fraction. The §14 submerged-mound pressure is gone with no isotropic
replacement; only the directed jet traction Λ and the weak U_v drag push before white."
