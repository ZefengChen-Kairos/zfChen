# Skeptic notes on claim A5 (old `chi` closure, model A): "膜是单相牛顿流体，无屈服应力、无自选厚度，堆高后以 √(c_p²ℓ)≈1.8 D/s 摊开，条纹合并、边缘变软；优化器把 c_p 压到下界是此缺陷的症状"

Files read (nothing modified): `/home/user/zfChen/latte_imex/solver.py` (Params L24–72, v2 c_p override L250–254, step B L444–474, step D L520–553, `_ssa_step` L676–695, `energy` L839–843), `TWOLAYER_CELL_MODEL.md` L45–79, `YIELD_FILM_DERIVATION.md` §0–1, `optimize.py`, `joint_cma.py` L21–24, `cp_sweep.py`; review notes `code_map.md` §1.0, §3, §5; `conv_part1.md` E-3, E-9, E-10; `notion_z1.md` E1, E6, E7, E17, E25; `physics_prior.md` §1.1–1.3, §2.5, §3c; `video_obs.md` §2, §9; `diagnosis.md` A1/A4/A5/A6; `skeptic_A4.md`.
Data re-read: `runs/joint_sweep_r2fix/results.json` (60 samples, #9 = idx 9), `runs/opt/joint_v1_heart_push_heart/{history,best}.json` (the 41-dim search of E-3), `runs/cp_sweep.json` (E1).
Numerical checks: `scratchpad/skeptic/a5_test.py`, logs `a5_disc.log`, `a5_stripes.log`, `a5_pour.log`, `a5_disc192.log`, `a5_stripes192.log` — old closure with the #9 constants (c_p 0.973, β 2.407, ν 1.44e-3, κ_c 0.665, κ_t 0.905, κ_r 0.0085, B_dep 1719, p_dep 1.56, kQ 1.326), N = 96 (and 192), same stationary low-pour inlet as `skeptic_A4.md` (Q 23 mL/s, u_in = (0, −1.88) D/s, U⊥ 9.27, footprint 0.049 × 0.031 D).

## 1. What the code does — the structural half of the claim is TRUE

* Pressure: `stepB_prepare(..., P.cp ** 2, ...)` L446, `coef = c_p² ℓⁿ/θ`, `q⁺ = a − dt·coef·∇ℓ⁺` (kernels L215–233); `energy()` L842 `pe = ½ c_p² ℓ²`. So p = ½c_p²ℓ², −c_p²ℓ∇ℓ, isotropic, acts on the **total** film thickness, white or brown alike. ✓
* In-plane stress for `closure="chi"`: step D L531–553 is the Laplace form ∇·(νℓ∇u) with ν = 1.44e-3 (explicit when ν dt/h² < 0.2, else PCG). The Bingham/SSA operator (`_ssa_operator` L628, `_ssa_step` L676) is reached **only** through `elif v2:` at L521 (per sub-step) or `advance_frame` L803–809 (per frame, v2 only). `Params.tau_y` (L67) exists but is dead for the chi closure. ✓ "无屈服应力".
* No thickness selection: ℓ obeys ∂ℓ/∂t + ∇·q = s with the pressure above and drag −βq; nothing stops a bump from relaxing to the flat state; there is no foam-thickness variable at all (white is the fraction c = m/ℓ of a film whose unit "1" is D_L/kQ = 60 mm of coffee-depth, code_map §1.0 / A6). ✓ "无自选厚度" — if anything the claim understates it: model A has no quantity that *could* select a foam thickness.
* Drag −βq towards **rest** (β_B = β, L383/L446): the film stops by drag, not by yield.

## 2. What the model actually does with a pile (a5_test.py)

### 2.1 Isolated white disc (R = 0.1 D, c = 1), no inlet
| ℓ in the disc | c_p | ℓ_max at 0.1 / 0.3 / 1.0 s | max\|u\| | white radius r(c = 0.6) 0 → 0.1 → 0.3 → 1.0 s | white area ×(1.0 s) | edge width c 0.9→0.1 (N = 96) |
|---|---|---|---|---|---|---|
| 3.4 | 0.973 | 1.53 / 1.27 / 1.12 | 0.73 D/s (0.05 s) | 0.103 → 0.167 → 0.200 → 0.182 | ×3.1 | 0.008 → 0.037 D |
| 3.4 | 0.110 | 3.40 / 3.18 / 1.53 | 0.067 D/s | 0.103 → 0.110 → 0.123 → 0.151 | ×2.2 (still spreading) | 0.008 → 0.030 D |
| 2.0 | 0.973 | 1.25 / 1.11 / — | 0.37 D/s | 0.103 → 0.134 → 0.147 | ×1.9 (0.5 s) | 0.008 → 0.029 D |
| **1.2** | 0.973 | 1.06 / 1.02 / 1.02 | 0.086 D/s | 0.103 → 0.110 → 0.112 → 0.110 | **×1.2** | 0.008 → 0.023 D |

* A real pile (ℓ = 3.4) does spread as an under-damped gravity current: front speed ≈ 0.7 D/s in the first 50 ms (not 1.8 D/s — that number is c_p√ℓ with ℓ = 3.4 taken as the *base* depth; the current rides on a base of ℓ = 1 and the measured front/|u| are 0.7 D/s), ℓ_max 3.4 → 1.5 in 0.1 s, centre overshoots to ℓ = 0.57 at 0.2 s and sloshes back. The same at N = 192 (r 0.098 → 0.164 at 0.1 s, area ×2.7), so the spreading is physical, not grid. ✓ mechanism exists *if a pile exists*.
* The white area ends at ≈ ℓ₀ × initial area (3.4 × 0.031 = 0.105 vs 0.096–0.116): a pile relaxes to ℓ ≈ 1 and its white area is set by **volume / 1 unit**, i.e. the "no stopping thickness" property is real.
* Edge width: 1 cell → 4.4 cells at N = 96 but only → 2.3 cells (0.019 D) at N = 192 at the same time: the softening is van Leer numerical diffusion of the passive c under a moving interface, **not** a physical consequence of the Newtonian film (c is passive, D = 1e-7, ν acts on u only — A1).
* At the working-point thickness ℓ = 1.2 the pile does essentially nothing: radius +7 %, area +20 % (mass conservation), max|u| 0.09 D/s.

### 2.2 Three 3 mm white stripes with 2 mm brown gaps, no inlet
| ℓ in white | c_p | gap (c < 0.3) at 0.1 / 0.5 / 1.5 s | min c in gap at 0.1 / 1.5 s | ℓ_max 0.1 s |
|---|---|---|---|---|
| **1.2**, N = 96 | 0.973 | 1.67 / 1.67 / 0.83 mm | 0.13 / 0.19 | 1.06 |
| **1.2**, N = 192 | 0.973 | 1.25 / 1.25 / — mm | **0.002 / 0.006 (0.5 s)** | 1.06 |
| 3.4, N = 96 | 0.973 | 0 / 0 / 0 mm | 0.67 / 0.71 | 1.59 |
| 3.4, N = 96 | 0.110 | 0 / 0 / 0 mm | 0.52 / 0.67 | 2.88 |

* With ℓ_white = 3.4 the gaps are gone within 50–100 ms at **either** c_p (the brown film between the piles is squeezed below one cell; with c_p = 0.11 it takes 0.1 s instead of 0.05 s). So "small c_p keeps the piles from spreading" is not even true for a 2 mm gap: a factor-9 lower c_p buys ≈ 50 ms.
* With ℓ_white = 1.2 (the thickness the fitted model actually produces, §2.3) the gaps **survive** for the whole 1.5 s: they narrow from 2.0 to ≈ 1.3–1.7 mm (the 20 % excess volume relaxes), and the rise of c in the gap at N = 96 (0 → 0.19) disappears at N = 192 (0.006), i.e. it is numerical. The film is frozen by drag: max|u| 0.067 → 0.003 D/s. **No merging, no physical softening** at the working point.

### 2.3 Real deposition: #9 low pour 0.3 s, then inlet off (a5_pour.log; cf. skeptic_A4 §2.2)
* c_p = 0.973: ℓ_max 1.27 (0.1 s) → 1.15 (0.3 s) → 1.03 (1.3 s). **There is no pile to spread**: the deposited film is flung along the jet direction at ≈ 1 D/s by the traction/source momentum (A4) and leaves the footprint at ℓ ≈ 1.15. After the stop: max|u| 0.61 (0.35 s) → 0.04 (1.0 s) → 0.01 (1.8 s); white area 0.0106 → 0.0113 → 0.0087 (no growth); c_max 0.774 → 0.758 (constant); the front coasts 0.155 → 0.201 D, almost all of it by 0.7 s. The pattern is frozen within ≈ 0.5 s of the stop by −βq, which is the role the claim assigns to the missing yield stress.
* c_p = 0.110 with the same inlet: ℓ_max 3.5 (0.1 s) → 4.6 (0.3 s), max|u| 1.3 → 2.1 D/s, c_max 0.94, and the solver **fails at t = 0.43 s** (`frame ... failed after retries (nsub=4608)`, solver.py L801). The "lower-bound" c_p is not a viable operating point of model A with physical inlets; the pile it creates is numerically unresolvable, not "堆高不摊开的实白锐边".
* All earlier records agree that the fitted/used model A never piles: E-10 (θ05, V0.5 controls, N = 128) ℓ_max 1.23–1.46 at c_p 0.3 and 1.14–1.33 at c_p 4; E2 (256²) 1.29–2.06; skeptic_A4 (#9) 1.15–1.19, 1.42 at 46 mL/s. The ℓ ≈ 3.4 in the claim is code_map §3's *thought experiment* ("pile grows at s_peak ≈ 24 units/s … 2.4 units per 0.1 s dwell" if nothing moved); the 1.8 D/s follows from it.

## 3. The cited evidence

1. **"conv part1 E-3 c_p → 0.11"**: E-3 is the 41-dim joint search (physics bounds c_p 0.1–1.5 log). `history.json` (52 gens) and `best.json`: the 0.11 reading is the **gen-12** snapshot ([52]); by gen 24 c_p = 0.56, [72] "c_p wandering 0.1–0.6", and the search's **final best has c_p = 0.89** (β 8.9, ν 0.01, κ_Q 65, B_dep 1470). The run was abandoned because the cut stopped cutting and the area-dominated loss preferred discs (notion_z1 E6, C:L12), not because of c_p. "优化器把 c_p 压到下界" is a transient misread.
2. **The loss is flat in c_p**: `joint_sweep_r2fix` (60 samples, c_p 0.3–4): corr(log c_p, mean loss) = **−0.06**; the ten best samples have c_p 0.3–2.6 (median 1.3); bin means 0.560 / 0.503 / 0.546 for c_p [0.3,0.7) / [0.7,1.5) / [1.5,4]. E-10: IoU 0.80–0.96 between c_p 0.3 and 2/4 on all six patterns; E17 "c_p、β 几乎不影响"; E25 gap counts "ν、c_p … 无关". The working point #9 has c_p = 0.973 — mid-range, at no bound. A non-identifiable parameter cannot be "pressed to a bound" as a symptom of anything.
3. **"ℓ_max 2.87"** is E1 (`cp_sweep.json`): c_p = **0.25**, Codex inlet (κ_Q 125, 0.6 mm stream), heart 2.5 s, N = 128; at c_p = 1 the same sweep gives 1.90. It is a different inlet, a different c_p and a different experiment from the joint search; it does not show what the fit did.
4. **E-9 [76] "small c_p … sharper edges that the loss rewards"** is the assistant's hypothesis (marked CLAIM in conv_part1), never tested; the observed effect of larger c_p in E-10 was "shorter heart tip, tulip base arcs more open, layered layers fainter" with IoU ≥ 0.80 — a small shape effect, and the only "sharp edges" model A is credited with come from the hard threshold c_sat = 0.6 (A6, conv part3 [537]), not from piles. Model A is, in the project's own comparison, the *sharp* model and v2 the soft one — the opposite of "边缘必然变软".
5. **physics_prior §1.1 (Bn ~ 1, h_stop 2–5 mm)**: a computed prior with τ_y = 1–5 Pa *assumed* and R = 2 cm, marked [E, computed]/[I]; no measurement. It is a fair reason to want a yield stress in a *foam-thickness* model, but model A has no foam thickness variable for h_stop to act on (§1 above) — the correct form of this criticism is A6/C4 (units, no h_f), not "Newtonian spreading".
6. **video V2 (10 s 不糊)**: true of the videos (video_obs L28, L178), but model A also does not blur after the pour (§2.2, §2.3; D = 1e-7; drag freezes the film in 0.5 s). The 2–4-cell numerical edge at N = 96 is a grid issue (conv [84]: "褐缝宽度只有 2–3 格时会被数值扩散抹淡"), not the closure. Note also video_obs L61: the real first push-heart blob after the lift is itself "soft-edged rounded".

## 4. What survives (for the correction)

* Model A's film is a single-phase Newtonian film with isotropic buoyancy-like pressure and no yield stress or stopping thickness — true, and a genuine pile would flatten to ℓ ≈ 1 in 0.1–0.2 s at ≈ 0.7 D/s (physical, grid-independent).
* But at the fitted working point the film never piles (ℓ_max 1.15–1.27), the pattern is a passive c on a nearly flat ℓ, motion is stopped by β within 0.5 s, 2 mm gaps survive ≥ 1.5 s at N ≥ 192, and the only edge softening is van Leer diffusion. Stripe narrowing in model A is compression by traction kinematics during the pour (A1/A4), not post-pour spreading.
* Consequences of the missing yield stress that *do* hold for model A: (i) white area = deposited volume / (ℓ ≈ 1 unit) — no h_stop, so area is not set by foam thickness (and the unit is 60 mm, A6); (ii) during a pour the old block is a liquid pushed by traction and deforms (near edge > far edge, A4 §2.4 — which is also what the video shows, V7 crescent); (iii) the "frozen pattern" is produced by drag to rest rather than by yield, which is adequate after the lift but gives no rigid-block translation during the push.
* The c_p story is the reverse of the claim: c_p is unidentifiable from end frames (flat loss), the 41-dim search ended at 0.89, #9 is 0.97, and c_p = 0.11 with physical inlets piles to ℓ = 4.6 and crashes the solver.

## 5. Verdict
**Refuted** as stated. The structural premise (Newtonian, no τ_y, no self-selected thickness) is confirmed in the code (L446/L842 pressure; L521 gates the Bingham step to v2 only; L531–553 Laplace ν), but the claimed consequence chain and its evidence do not hold for model A: there is no ℓ ≈ 3.4 pile at the working point (ℓ_max 1.15–1.27 measured; 2.87 is a c_p = 0.25 Codex-inlet sweep), piles at ℓ ≈ 1.2 neither merge 2 mm gaps nor soften edges (grid-converged), drag freezes the pattern within 0.5 s, and "c_p → 0.11" was a gen-12 transient of a search whose final best is c_p = 0.89 in a loss that is flat in c_p (corr −0.06, IoU 0.80–0.96 over 0.3–4). The real yield-stress-related defects of model A are the absence of any foam-thickness variable (so no h_stop, white area = volume/60 mm-unit) and liquid-like deformation of old blocks during a push — not spreading, merging or soft edges.
