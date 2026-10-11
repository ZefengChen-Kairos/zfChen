# Skeptic notes on claim A4 (old `chi` closure): "推挤速度量级与衰减错：足迹内膜被钉到 κ_t u_in ≈ 1.7 D/s 后以 u/β ≈ 0.7 D 滑行，而真实位移 0.05–0.15 D 且随距离快速衰减（> 0.3 D 不动）"

Files read (nothing modified): `/home/user/zfChen/latte_imex/solver.py` (inlet_fields L288–350, substep L353–480, advance_frame L757–812),
`kernels.py` L165–233 (`stepB_prepare`, `finish_B`, `q_from_gradient`), `pitcher.py` L533–580 (`Coupling.inlet`, u_in and the 60° cap),
`PITCHER_MODEL.md` §13–15, review notes `code_map.md` §1.1, §3, §5; `video_obs.md` §7, §9; `physics_prior.md` §0, §2.4, §3b;
`diagnosis.md` A3/A4; `skeptic_A1.md` §2.2 (independent moving-hit run, same closure).
Numerical check: `scratchpad/skeptic/a4_test.py` (logs `a4_log.txt`, `a4_log2.txt`): old closure with the #9 constants
(`runs/joint_sweep_r2fix/results.json[9]`: c_p 0.973, β 2.407, ν 1.4e-3, κ_c 0.665, κ_t 0.905, κ_r 0.0085, B_dep 1719, p_dep 1.56, kQ 1.326),
N = 96, **stationary** low pour at (0, +0.10): Q = 23 mL/s (S_eff 0.0596), u_in = (0, −1.88) D/s, U⊥ 9.27, d 0.079, footprint 0.049 × 0.031 D —
the same inlet `code_map.md` used for its numbers. Lagrangian particles seeded on the push axis at r = 0–0.4 D ahead of the hit and
laterally, advected with u = q/ℓ (bilinear, RK2, once per 1/120 s frame); |u| profiles read directly from the solver state.
In the chi closure c = m/ℓ is passive (skeptic_A1 §1), so particle displacement = displacement of an old white block; a disc run
(A1 geometry, d0 = 0.15 and 0.30 D) confirmed this to ±0.01 D.

## 1. What the code does (mechanism as claimed, partly)

* `inlet_fields` L331–350: s = χ S_eff K, Λ = κ_c(1−χ) S_eff K, v* = κ_t u_in + κ_r V_R ê_r. At this inlet χ = 0.67,
  s_peak = 23.9/s, Λ_peak = 7.9/s, |κ_t u_in| = 1.70 D/s, κ_r V_R = 0.07 D/s (code_map §1.1). ✓ the *target* is 1.7 D/s.
* `stepB_prepare` (kernels L165–175): a = (q* + dt (s+Λ) v*)/θ, θ = 1 + dt(β + Λ/ℓ) ⇒ continuous ℓ ∂u/∂t = (s+Λ)(v*−u) − βℓu − c_p²ℓ∇ℓ − (advection).
  Relaxation rate at the kernel peak (ℓ ≈ 1): (s+Λ)/ℓ + β ≈ 34/s ⇒ **e-fold ≈ 30 ms, not "~10 ms"** (the 10 ms figure in code_map §3 is the
  v2 column, Λ_peak 104/s; the diagnosis copied it into the old-closure claim).
* Drag is −βq toward rest (β_B = β, L446/L455) ✓, and u/β = 1.7/2.407 = 0.71 D is the arithmetic the claim quotes ✓ — but it is the
  coasting distance of an *isolated* parcel with no neighbours, no pressure and no momentum exchange. The code has all three
  (Rusanov ∇·(q⊗u) L433–438; −c_p²ℓ∇ℓ L444–474), so that arithmetic is not what the model does (see §2.2).

## 2. What the model actually produces (a4_test.py, Q = 23 mL/s unless stated)

### 2.1 The film is NOT pinned to κ_t u_in
|u| at the kernel centre / max |u| anywhere (D/s): t = 8 ms 0.33/0.33; 33 ms 0.64/0.66; 0.1 s 0.84/0.87; 0.3 s 0.97/1.07;
0.5–1.3 s **0.95–0.97 / 1.08–1.10** (max sits at r ≈ 0.03 D, just ahead of the footprint). Steady footprint speed = **0.56–0.65 × v***
(7.6–8.8 cm/s, not 13 cm/s). ℓ_max stays 1.15–1.19: the pile does not build because the film is flung forward and the momentum injected
in K is immediately shared with the film it runs into (momentum, not velocity, is conserved by the Rusanov flux) and partly converted to
pressure. Scaling: u_in halved (0.94) ⇒ centre 0.52, max 0.65 (∝ u_in, same 0.55–0.69 fraction); Q doubled (46 mL/s, the chain's burst
level, code_map §5 item 12) ⇒ centre 1.12, max 1.42 (+15–30 %, ℓ_max 1.42).

### 2.2 Coasting after the pour stops is 0.06 D, not 0.7 D
Pour 0.3 s (film at 0.97–1.07 D/s), then `Inlet(active=False)` for 1.0 s: |u| at the hit 0.97 → 0.23 (0.2 s later) → 0.07 (0.5 s) → 0.03 (0.7 s)
→ 0.005 (1.0 s). Pure −βq would give 0.97·e^{−0.48} = 0.60 at 0.2 s; the actual decay is 3× faster because the moving film hands its
momentum to stationary film ahead and the pressure converts it. Extra displacement after the stop: seed r = 0 **+0.060 D**, 0.1 D +0.049,
0.2 D +0.027, 0.3 D +0.015, 0.4 D +0.009. **The "u/β ≈ 0.7 D" coasting is wrong by an order of magnitude (12×).**

### 2.3 The velocity field DOES decay with distance, with an e-fold length ≈ 0.1 D
Steady axis profile (t = 0.5–1.3 s), |u| in D/s vs r ahead of the hit:
r = 0.03: 1.09 · 0.05: 1.05 · 0.08: 0.87 · 0.10: 0.75 · 0.15: 0.54 · 0.20: 0.35 · 0.30: 0.11 · 0.40: 0.04. Lateral (same y as the hit):
0.1 D: 0.12, 0.2 D: 0.06–0.09, 0.3 D: 0.03–0.06. So the push is a forward lobe ~0.1 D wide; it is not isotropic diffusion and not a
non-decaying field. The front does creep outward slowly (|u| at 0.4 D 0.007 → 0.04 over 0.3 → 1.0 s).

### 2.4 Lagrangian displacement of old material vs. the video
Displacement along the push (D) after 0.5 / 1.0 / 1.3 s, by initial distance r from the hit:

| r (D) | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|---|
| 0.5 s | 0.156 | 0.113 | 0.077 | 0.052 | 0.036 | 0.025 | 0.012 |
| 1.0 s | 0.244 | 0.196 | 0.152 | 0.114 | 0.083 | 0.060 | 0.030 |
| 1.3 s | 0.277 | 0.229 | 0.183 | 0.141 | 0.106 | 0.077 | 0.039 |

Disc check (A1 geometry): rear edge d0 = 0.15 D from the hit moves 0.010/0.031/0.051/0.083 D at 0.1/0.2/0.3/0.5 s (then the white
merges and the edge detector fails); d0 = 0.30 D: 0.006/0.016/0.026/0.058/0.068 D at 0.1/0.2/0.3/0.8/1.0 s; the far edge (0.48–0.63 D
away) moves ≤ 0.008 D. skeptic_A1 §2.2 (hit moving at 0.09 D/s, d0 = 0.15 D) found 0.11 D in 0.6 s — consistent.

Video (`video_obs.md` §7 items 1, 4; §9 rank 7): the old element, whose near edge sits ≈ 0.15–0.2 D from the new impact, is displaced
0.05–0.15 D during the ~1 s white-free start; near edge moves more than far edge (crescent); elements > 0.3 D away "barely move".
Model at the same geometry: **0.11–0.15 D in 1 s at r = 0.15–0.2 D** (upper end of the real range, within ×1–1.5), 0.06 D in 1 s at 0.3 D
(5 mm; "barely" in the video is unquantified, so ×1.5–2 at most), near edge > far edge ✓ (0.15 vs 0.06 at 1 s: the block thins, as observed).
At the chain's burst flow (46 mL/s, 0.5 s): 0.126 / 0.093 D at r = 0.15 / 0.20 — above the real band, but that excess comes from the
action-chain burst (A18), not from the closure's speed law.

### 2.5 The physics_prior evidence does not say what the claim uses it for
`physics_prior.md` §0's "4–5 cm/s" is u ~ √(M/ρA) with A = the whole cup area — a cup-scale bulk estimate marked [I]; the same page §0
says "features move at 1–5 cm/s" (= 0.12–0.6 D/s) and §2.4 says the near-field surface sheet at r ≲ 2d runs at **u ≈ v ≈ 0.7 m/s**,
decaying with r, with 1–3 cm/s in the far field. The model's 8 cm/s inside a 1 cm² footprint and 6 / 2.8 / 0.9 cm/s at 0.1 / 0.2 / 0.3 D
sit inside that envelope. "量级错" is not supported by this prior. (The diagnosis's "图案漂移 0.1–0.16 D/s" is the *pitcher* drift speed
in video_obs L44/L60/L126, not a measured pattern speed.)

## 3. What survives of A4

* The speed law is v* = κ_t u_in: the push scales with the jet's horizontal velocity (and, weaker, with Q), with no saturation and no
  dependence on the film's yield state. In the video the push stops when the pour stops; the model's 0.06 D over-run after a stop
  is small but not zero.
* At 23 mL/s the model's displacement is at/above the upper edge of the real 0.05–0.15 D band and its reach is perhaps ×1.5–2 too long
  (0.06 D at r = 0.3 D). A calibration-level discrepancy (κ_t ≈ 0.5–0.6 or β ≈ 4–5 would bring both into the band), not an order of magnitude.
* The real defect in the push is ordering, not speed: the deposited white (c = 1 at once) moves at the same 1 D/s and overruns the gap
  (skeptic_A1 §2.2, A2/A3), and the chain's bursts (A18) push harder than the barista's 10–20 mL/s. Those are separate claims.

## 4. Verdict
**Refuted as stated.** Each quantitative element of the mechanism is contradicted by the code's own behaviour: the film reaches
0.95–1.1 D/s (0.55–0.65 v*), not 1.7; it coasts 0.06 D after the pour stops, not 0.7 D; the velocity decays with an e-fold of ≈ 0.1 D
(0.75/0.35/0.11 D/s at 0.1/0.2/0.3 D) and old material at 0.15–0.2 D moves 0.11–0.15 D in 1 s, matching the video's 0.05–0.15 D to within
×1.5. The cited physics_prior number is a cup-average, not the near-field speed. What remains is a ≤ ×2 over-push/over-reach at the
real flow and a speed law tied to u_in — worth noting as calibration, not as a "magnitude and decay are wrong" structural failure.
Caveats: stationary hit, N = 96, old #9 constants only; a moving hit (A1 run) gave the same displacement rate; v2 was not tested here.
