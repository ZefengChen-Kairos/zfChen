# Skeptic notes — CLAIM A16 (帧级冻结 / CFL 取帧首 u_max 不含重力波速 / 细切 ROI 时间步不收敛)

Claim text (diagnosis.md L86): 入口、Λ、v* 在 8.3 ms 内常数；u_max 取帧首值，不含帧内产生的重力波速 √(c_p²ℓ)；违例只靠非有限/ℓ 下限重试捕获，否则表现为额外 Rusanov 耗散（code_map item 11）。细切阶段 ROI 时间步不收敛（notion_senior §3.2）。

Model A = family A = old χ closure (`closure="chi"`, physics #9: c_p = 0.973, N = 96, cfl 0.5, Δt_f = 1/120 s), as used by `action_opt.run_one`.

**Verdict: REFUTED as a shortcoming.** The frame freeze and the frame-start CFL bound exist exactly as described (the *description* is accurate, see §1), but (i) the implied consequences — CFL violations, retries, "extra Rusanov dissipation" — do not occur in a full pipeline heart run (§2); (ii) the omission of √(c_p²ℓ) from the CFL is the documented IMEX design (pressure is implicit) and for model A √(c_p²ℓ) < u_max anyway (§3); (iii) the end state is insensitive to the time step at the level 1e-4 in L1, 4e-3 pointwise — three orders of magnitude below the model–target mismatch (§4); (iv) the cited "细切 ROI 时间步不收敛" evidence is from the senior's V0.5 HLLC code at 384–1536², not from model A, and the senior himself attributed it to inlet-kernel under-resolution, not frame freezing (§5). One genuinely untested residual remains (§6): the pitcher clock is a fixed 1/120 s zero-order hold that the solver's scan-shortening cannot refine; its effect on swan/tulip is unmeasured.

---

## 1. What the code does (solver.py, pitcher.py, action_opt.py)

- `advance_frame` (solver.py L757–812): `frame_dt = self.num.frame_dt` (1/120, L758); if the inlet is active and `scan_speed > 0`, `frame_dt = min(frame_dt, max(0.5·min(h, r1, r2)/scan_speed, 1e-4))` (L759–761). `s, Lam, vsx, vsy, chi = self.inlet_fields(inlet)` is called **once per frame** (L762) and passed unchanged to every `substep` (L786). So inside a frame s, Λ, v* are constant. Module docstring L5 states this explicitly: "inlet fields s, Lambda, v* frozen within a control frame".
- u_max (L768–776) = max over cells of |q|/ℓ at frame start, **plus** the inlet target speed |v*| where s+Λ>0 ("it is reached quickly", L770–772), plus |G_e| and |U_b|,|U_d| when they exist (not for chi). `dt_cfl = cfl·h/(2·u_max)` (L778), `nsub = ceil(frame_dt/dt_cfl)` capped at 64 (L779–780). A frame is rolled back and retried with 2·nsub when a sub-step returns False, i.e. non-finite fields or min ℓ ≤ l_floor (L574–577, L783–801).
- Pressure/gravity-wave terms are **implicit**: step B "implicit mass + pressure + drag/traction → SPD elliptic equation for ℓ^{n+1}, PCG" (docstring L8, code L445–462). The explicit part is only the momentum advection with Rusanov flux α = 2·max|u| (L433–436), a Burgers-type flux with f′ = 2u — hence the factor 2 in dt_cfl. README.md L14–15: "the frame is split into sub-steps by the material CFL Δt ≤ CFL·h/(2 max|u|) only (no c_p restriction)". LATTE_MODEL.md L185–188 says the same for v2.
- Pipeline loop (`action_opt.run_one` L80–84, identical in optimize.py L27–28, joint_run.py L35–36, run.py L72–73): `probe = script.sample(sol.t)` → scan-shortened fdt → `inl = script.sample(sol.t + 0.5·fdt)` → `advance_frame(inl, frame_dt=fdt)`.
- **`BaristaScript.sample` (pitcher.py L859–866)** advances the pitcher "on its own fixed clock (dt = 1/120 s) up to t and returns the latest record". The docstring says irregular steps were deliberately removed ("irregular steps … made the acceleration estimate jump"). Consequence: the inlet record is a zero-order hold at exactly 1/120 s **independent of the solver frame length**; a scan-shortened solver frame and its neighbours all receive the same record (checked analytically for Δt_f = 1/120, 1/240, 1/480: the record index is ⌊t·120⌋ in all three cases). The scan guard `scan_safety` (hit point moves ≤ 0.5·min(h,r1,r2) per *solver* frame) therefore does not limit the hit-point jump per *pitcher* step.

So "入口/Λ/v* 8.3 ms 常数" is correct, and in fact stronger than the claim (and than code_map L41 "shortened when the hit point moves"): the shortening never refines the inlet signal.

## 2. Numerical check: does the frame-start u_max ever get exceeded? (scratchpad/a16/probe.py)

Pipeline heart (BaristaScript, physics #9, old closure, N = 96, T = 8.925 s), per-frame record of u_max at frame start (as in L768–772) vs max |q|/ℓ reached after each sub-step inside the frame:

| run | frames (pouring) | retries | clip_mass | nsub max / mean | max u_sub/u_max0 over pouring frames (u_max0 > 0.1) | actual max Courant 2·u·dt/h |
|---|---|---|---|---|---|---|
| cfl 0.5, Δt_f 1/120 | 1144 (917) | 0 | 0.0 | 6 / 4.55 | **0.560** (t = 0.77 s: bound 1.593 → reached 0.893) | **0.272** |
| cfl 0.25, 1/120 | 1144 (917) | 0 | 0.0 | 12 / 8.54 | 0.56 | — |
| cfl 0.125, 1/120 | 1144 (917) | 0 | 0.0 | 23 / 16.7 | 0.56 | — |
| cfl 0.5, 1/240 | 2187 (1734) | 0 | 0.0 | 3 / 2.73 | 0.561 | 0.272 |
| cfl 0.5, 1/480 | 4292 (3388) | 0 | 0.0 | 2 / 1.81 | 0.561 | 0.272 |

- Over the whole run the only frames with u_sub/u_max0 > 1 (29–129 of them, ratio ≤ 1.05) occur at t > 7.8 s after the pour, where u_max0 ≤ 1.4e-3 D/s — the ratio of two numbers ≈ 0 (Courant ≈ 1e-3). During pouring the velocity reached inside a frame is at most **56 %** of the frame-start bound, because the bound includes |v*| (L770–772) and the traction only partially realises it in one frame. The effective Courant number is 0.27, not 0.5.
- `retries` = 0 and `clip_mass` = 0 (boundedness guard L497–500 inactive) in all five runs: nothing is "caught only by the non-finite/ℓ-floor retry" because nothing needs catching.
- Side remark on "额外 Rusanov 耗散": Rusanov's α = 2·max(|u_L|,|u_R|) is dt-independent; an advective CFL violation in an explicit Rusanov step produces instability (growth), not extra dissipation. The modified-equation diffusion of a Lax–Friedrichs/Rusanov flux *decreases* with Courant number. So this part of the claim is mis-stated regardless of whether violations happen.

## 3. "不含重力波速 √(c_p²ℓ)"

- For model A: c_p = 0.973 (physics #9), ℓ_max ≈ 1.3 ⇒ √(c_p²ℓ)_max = **1.11 D/s** (probe column `cpl`) while u_max0 reaches **1.78 D/s**. The gravity-wave speed is *below* the material speed throughout the heart; code_map item 11's "3.5 D/s" is the v2 number (ℓ = 2, c_p² = g′H_f/D_L) and does not belong to family A.
- More fundamentally the pressure is implicit (step B), so the gravity-wave speed does not enter a stability bound at all; that is the point of IMEX and is documented (README L14–15; conv_part1 E-6/C-5 derive the IMEX/HLLC step ratio 2‖u‖/(‖u‖+a) from exactly this). Omitting it from dt_cfl is design, not an oversight. Implicit treatment damps fast gravity waves at large dt — an accuracy question, which §4 measures.

## 4. Time-step sensitivity of the end state (scratchpad/a16/cmp.py)

End-state whiteness c = m/ℓ at t = 8.925 s, N = 96. ROI = cut strip |x| < 0.08, y ∈ [−0.30, 0.03] (hit point of the heart's cut runs y: 0 → −0.27, 512 cells). Reference loss of this very run vs the video target (runs/action_opt/old/heart_h1.000_q1.000_s1.000_N96.json): **L1 = 0.165, IoU = 0.58**.

| pair | L1 cup | L1 cut ROI | max |Δc| (location) | IoU(c>0.4) | white cells |
|---|---|---|---|---|---|
| cfl 0.5 vs 0.25 (Δt_f 1/120) | 2.01e-4 | 8.98e-4 | 3.8e-3 (x 0.02, y −0.15) | 0.9993 | 1468 / 1467 |
| cfl 0.25 vs 0.125 | 1.11e-4 | 5.05e-4 | 2.1e-3 | 0.9993 | 1467 / 1466 |
| Δt_f 1/120 vs 1/240 (cfl 0.5) | 6.42e-5 | 2.80e-4 | 8.6e-4 | 1.0000 | 1468 / 1468 |
| Δt_f 1/240 vs 1/480 | 8.60e-5 | 3.61e-4 | 1.1e-3 | 0.9993 | 1468 / 1467 |

- Sub-step refinement converges roughly first order (ratios 1.8 in cup and ROI). Frame refinement is not monotone (6.4e-5 → 8.6e-5), but note (a) the inlet record is identical in all three frame lengths (§1, pitcher clock), so this sequence does *not* test the inlet freeze, only the frame-level splitting; (b) the ceil-rounded sub-step changes with the frame (mean dt 1.78 → 1.53 → 1.24 ms), so the sub-step error (≈2e-4 at cfl 0.5, larger than the frame-level differences) is mixed in. Either way every difference is ≤ 9e-4 in the cut ROI and ≤ 3.8e-3 pointwise, i.e. **2–3 orders of magnitude below the model–target L1 of 0.165**, and the white set is unchanged to 1 cell in 1468.
- The largest pointwise differences sit in the cut/notch (x ≈ 0.02, y ≈ −0.15), as the senior found for his code, but at 2e-3–4e-3 instead of his 0.04–0.07 — at N = 96 the cut footprint (min(r1,r2)/h ≥ 0.6, kernel floor 1.5h) is already smeared spatially, so there is little fine structure left for the time step to act on. This is a statement about N = 96 (every fit in the project); at 384+ it is untested here.

## 5. The cited evidence does not concern model A

- notion_senior §3.2 (L168–169) reports *the senior's* V0.5 tests: 384/768/1536 grids, heart at t = 6.425 s, ROI L1 3.477e-4 → 4.648e-4, max 0.0377 → 0.0687. notion_z1 E25′ (L48) files the same result under solver "**Codex HLLC**" (the senior's explicit HLLC code), not latte_imex IMEX+Rusanov. Model A (latte_imex) has had **no** time-step convergence test before this one — notion_z1 E1 quotes A:L133 "没有…时间步和网格收敛检查".
- The senior's own clue for the non-convergence (§3.2) is "细切阶段的入口短半轴 min(r1,r2)/Δx 从约 2.653 降至约 0.997，自动求积密度仅在末端约 6.4241 s 从 4 切至 5" — a spatial inlet-resolution / quadrature-switch issue (notion_senior L204: "与入口核宽/格距比 ~1 相关；源项求积密度要固定"), not frame freezing or a CFL bound. In latte_imex the quadrature is fixed (`kernel_quadrature = 3`, Numerics L85) and the kernel is floored at 1.5h (A15), so even the senior's mechanism is not transferable one-to-one.
- The claim therefore stitches a code description of latte_imex (code_map 11) to a convergence failure of a different solver and different grid.

## 6. What is genuinely untested (residual uncertainty, not part of the claim as stated)

- The pitcher-clock hold: in push_heart / layered_heart / tulip / leaf / swan the scan guard shortens 86 / 173 / 228 / 266 / 288 pouring frames (swan scan up to 3.24 D/s ⇒ hit point jumps 2.2 mm = 2.6 h per 8.3 ms pitcher step), but those sub-frames all reuse the same inlet record, so the hit point still moves in 8.3 ms jumps. For the heart the guard never triggers while pouring (cut scan 0.51 D/s ⇒ limit 9.1 ms > 8.3 ms). Settling whether the 8.3 ms hold matters for the fast-scan patterns needs runs with `BaristaScript(dt=1/240, 1/480)` (which also changes the pitcher's finite-difference kinematics — the docstring warns of this) and a comparison of end states; expected magnitude by analogy with §4: ≲1e-3, but unmeasured.
- Finer grids (≥ 384) where the cut is resolved: the time-step sensitivity could grow toward the senior's 0.04–0.07 pointwise; not run (>2 min).

## 7. Bottom line

The frame freeze and the frame-start CFL are real and documented; nothing in the measured behaviour of model A at the grid used for every fit (N = 96) makes them a shortcoming: no CFL violations (effective Courant 0.27), no retries, no clipping, and an end-state time-step sensitivity of ~1e-4 (L1) / ≤4e-3 (pointwise) against a model–target mismatch of 0.165. The gravity-wave omission is IMEX design and for model A √(c_p²ℓ) (1.11 D/s) is below u_max (1.78 D/s). The "细切 ROI 时间步不收敛" evidence belongs to the senior's HLLC V0.5 code at 384–1536² and was attributed by him to inlet-kernel resolution. A16 should be dropped from the list of reasons model A fails, or reduced to the one open item in §6.

Files: scratchpad/a16/probe.py (runner), scratchpad/a16/cmp.py (comparison), scratchpad/a16/heart_old_cfl*_f*.npz/.log (five runs, 21–100 s each).
