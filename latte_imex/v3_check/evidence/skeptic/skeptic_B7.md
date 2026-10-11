# Skeptic check of CLAIM B7 — entrainment-cell coefficients and scales have no physical basis

Verdict: **confirmed with nuance**. The three factual sub-claims trace to the code and docs; the rhetoric ("no physical basis", "a second jet") overstates in two places.

## 1. What the code does (solver.py, closure "twolayer"/"v2")

- L45–49 `Params`: `ent_coef = 0.74` with the comment *"Ricou-Spalding jet entrainment 0.32 z/d × Turner fountain depth z/d = 2.32 Fr"*; `cell_frac = 1.0`, "radius of the entrainment cell / fountain depth".
- L298–312 `inlet_fields`: `Fr2 = U_perp²/(g_red d)`; v2: `S_v = S_eff·D_L/H`, `SE = 2·ent_coef·√Fr2·S_v` (twolayer: `SE = ent_coef·√Fr2·S_eff`); `_Rc = min(cell_frac·2.32·√Fr2·d, 0.45)`.
- L590–624 `_bulk_flow`: `cell = hypot(x−x0, y−y0) < max(_Rc, 2h)`; `Kc = uniform on the disc`; `div U_c = SE·(Kc − K)` (potential flow); `Ub = U_v + U_c`, `Ud = U_v − U_c`; `divUs = div_v + Ds`. In v2 `Ds` is then set to None: the cell acts on the film only through the kinematic stretching `∇·U_s` (skin source `l_skin·max(divUs,0)` at L415–418).
- Docs repeat the same derivation verbatim: TWOLAYER L29–30, LATTE L79–80, L88–92 ("设环流的上、下两支各占一半深度 h_s = H/2 … S_E = 2·ent·Fr·S_v"), PITCHER L254.

So the attribution in the claim is exact: ent 0.74 = 0.32 × 2.32 and R_c = cell_frac × (vertical fountain depth) are literally what the code and all three docs say.

## 2. Sub-claim (a): 0.74 is a Re > 10⁴ turbulent-jet number, the jet is Re ≈ 2–4×10³ — CONFIRMED

- Ricou & Spalding (1961) 0.32 z/d is their asymptotic value for Re ≳ 2.5×10⁴; their own data show the coefficient falling below that at lower Re. Turner's 2.32 Fr d (actually a Fr-linear penetration for *turbulent* fountains, Fr ≳ 3) likewise assumes a turbulent, unconfined fountain.
- Recomputed Re (ν_milk 1–2×10⁻⁶ m²/s): low pour U 0.68 m/s, d 5.3 mm → Re 1800–3600; high cut 1.2 m/s, 2.2 mm → Re 2640. physics_prior.md §2.1 gives 1.6×10³ with μ = 2 mPa·s. All ≥ 7× below the Ricou–Spalding range.
- Order-of-magnitude bracket I computed: turbulent E/Q = 0.32·z_f/d = **2.2–2.8** (low pour, g′ 10–6) and 6–7.8 (high); fully laminar round jet (Schlichting, Q(z) = 8πνz) gives E/Q = **0.06–0.08** (low) and 0.23–0.29 (high). The real transitional plunging jet lies somewhere in a 30× window; the project's own physics_prior.md L163 quotes "entrained-to-jet ratio O(0.1–1) at Re ≈ 2–3×10³ [Biń 1993]". Fitted values ent 0.37–0.69 give E/Q = 1.1–2.5, i.e. at or above the top of that window, and the CMA bounds (joint_cma.py L19: ent_coef 0.2–1.5; cell_test.py L36: 0.3–1.5) **exclude the laminar/transitional values** (ent ≈ 0.02–0.3).
- Additional regime violation the claim mentions: fountain depth 2.32 Fr d = 3.6–4.7 cm (low) and 4.1–5.3 cm (high) vs coffee depth H = 5 cm; Turner's unconfined scaling does not apply when z_f ≈ H and the cup wall is 4 cm away. Neither doc has a bottom.

## 3. Sub-claim (b): R_c = cell_frac × vertical penetration, cell_frac 0.05–0.12 "has no physical basis" — CONFIRMED as stated in the docs, but the wording is too strong

- Code: R_c = cell_frac·2.32·Fr·d, floored at 2h (1.7 mm at 96²), capped 0.45 D. Fitted cell_frac: 0.122 (cell_fit, lower bound 0.1), 0.0898 (v2F), 0.0528 (v2s, bound 0.05) → R_c 2.6–5.3 mm (code_map table). The docs give no argument for the 0.1 factor: LATTE L253 just lists "cell_frac 0.1–1, from single-point pour clearing radius" (never measured, B24). conv [473] states the author's *own* physical expectation is R_c ≈ penetration depth 2–3 cm (cell_frac ≈ 1) and admits "R_c 正好等于缝宽 … 讲过头了"; yet TWOLAYER L80–81 and LATTE §7.3 L300 keep R_c ≈ 3–4 mm as the gap-width mechanism (docs_map C13). All true.
- Nuance: a horizontal radius ∝ a fraction of the fountain height is not intrinsically unphysical — for a turbulent fountain the return annulus reaching the source level has radius ≈ 0.1–0.25 z_m, so cell_frac ≈ 0.1–0.2 could be defended as "the fountain's own annular return", and in that picture the upwelling should be an **annulus around the jet, not a disc containing the jet core** (= B13). The docs never make this argument; they call the cell "where the entrained coffee wells up again", which in the real flow is the cup-scale toroidal recirculation (physics_prior §2.4 region 2, r ≳ 2–3 cm). The claim is right that the docs supply no basis; it would be wrong to say no basis *could* exist.
- At cell_frac 0.05 the fitted R_c is at the 2h grid floor and in v2s (R_c 2.8 mm < footprint semi-axis 3.9 mm) the "ring" has zero cells (code_map item 4) — the fit has effectively switched the cell off, which is the stronger criticism.

## 4. Sub-claim (c): the cell's upwelling ≈ 66 cm/s "a second jet" — arithmetic CONFIRMED, interpretation partly rhetorical

- Recomputed (Q 15 mL/s, H 5 cm, D_L 8 cm, h_s = H/2 as LATTE L88): doc example ent 0.74, Fr 3, R_c 4 mm → E = 33 mL/s = 2.2 Q, ∇·U_s in cell = S_E K_c = 26.5 /s (LATTE L300's "≈ 27 /s" ✓), w = E/(πR_c²) = ∇·U_s·H/2 = **66 cm/s**. The number comes from the doc's own worked example, not from a fit; with fitted parameters: cell_fit 19 cm/s, v2F 53 cm/s (E = 2.5 Q), v2s 82 cm/s (E = 1.3 Q through 0.25 cm²). Jet impact speed 68 cm/s through 0.22 cm². So "comparable to the jet" holds for every v2 fit.
- Nuance 1: in v2 this is *not* a volume or momentum source (Ds = None, L622–623); the quantity the film actually feels is a surface strain rate of 21–33 /s over R_c plus a convergence of −126 /s at the footprint (code_map). Surface strain rates of order U/d ≈ 130 /s do exist within a few mm of a 0.7 m/s jet (the outward impact sheet, physics_prior §2.4 region 1), so "27 /s next to the jet" is not absurd by magnitude. What is unphysical is (i) that this flux is attributed to *entrained coffee returning* (E/Q = 2.2 is the turbulent-jet value, see §2) and (ii) its sign/location — converging at the jet, uniform disc including the core — which belong to B13 and the "surface always converges" critique, not to B7.
- Nuance 2: "equal to a second jet" conflates flux and momentum; the cell carries no momentum of its own, it is a potential flow.

## 5. Bottom line

The evidence cited (docs_map §4.2 rows "ent", "R_c", "胞内散度", "喷泉深度"; conv [473]; TWOLAYER L80 / LATTE §7.3) is accurate and the mechanism traces exactly to solver.py L45–49, L298–312, L590–624. Confirmed, with the correction that (1) "no physical basis" should read "a turbulent, unconfined-fountain estimate applied outside its Re/confinement regime, with the entrainment ratio uncertain by ~30× (laminar 0.06 ↔ turbulent 2.2) and the fit bounds excluding the low end"; (2) the 0.1 fraction of fountain height is at best the fountain's own return-annulus radius — a defence the docs do not make and which would require an annular, not disc, upwelling; (3) "66 cm/s" is the doc's example (fitted: 19–82 cm/s) and is a kinematic stretching rate ~27 /s in v2 rather than a momentum-carrying jet.

Experiment that would settle the physical cell: TWOLAYER L94's own proposal — a single stationary pour at fixed height/flow on plain crema, measuring the cleared-circle radius and clearing speed vs height; never done (B24).
