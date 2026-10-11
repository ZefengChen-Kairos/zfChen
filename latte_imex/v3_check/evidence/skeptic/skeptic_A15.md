# Skeptic check of CLAIM A15 — 核半轴下限 1.5h 与分辨率耦合

CLAIM A15: 核半轴下限 1.5h 把 96² 上细切足迹 0.93 mm 抬到 1.25 mm（面积×1.8），拟合常数与分辨率耦合，96² 拟合≠256² 出图。
EVIDENCE GIVEN: solver.py L261；code_map item 3。

Verdict: **confirmed with nuance** — the mechanism and every number in the first clause are exactly what the code does (re-derived below, including K_peak = 2684). The second clause ("拟合常数与分辨率耦合，96² 拟合≠256² 出图") is only *partly* traceable: (i) the floor touches only the thin-cut phase (0.6–0.9 s of a 7–12 s pour); the low-pour footprint (3.9 × 2.5 mm) is never floored at 96², so it cannot explain resolution dependence of the bulk of the pattern; (ii) χ (Fr_c², B_dep, p_dep) and R_c (cell_frac) use the un-floored `d_jet`, so those "fitted constants" are not coupled *through the floor* — only κ_c (Λ = κ_c S K) and the local deposition density are; (iii) even without the floor, a 0.93 mm kernel on 0.83 mm cells is itself badly resolved (K_peak 3536 vs 6379 at 256²), so the floor is the larger but not the only part of the 96² widening; (iv) the "96² ≠ 256²" practice is real and documented, but no experiment in the record isolates the floor's share of it (see §4 for the one I ran).

---

## 1. What the code does (solver.py)

- `Grid.__init__` L114: `self.h = 1.0 / N` — cell size in cup diameters, D_L = 0.08 m (`Params.D_L` L64). So h = 0.833 mm at N = 96, 0.625 mm at 128, 0.3125 mm at 256.
- `Solver.kernel` L261: `r1, r2, phi = max(inlet.r1, 1.5 * g.h), max(inlet.r2, 1.5 * g.h), inlet.phi` — both semi-axes floored at 1.5h = **1.25 mm (96²)**, 0.94 mm (128²), **0.47 mm (256²)**.
- L263–264: quadrature `q = min(12, max(3, ceil(4h / min(r1, r2))))` is computed from the *floored* radii, so with the floor active q = ceil(4/1.5) = 3 (the base 3×3). The adaptive sub-cell quadrature (up to 12×12) that LATTE_MODEL L60 advertises for "薄足迹" therefore never engages while the floor is active: the floor pre-empts it.
- L286–287: `K / Z` with `Z = K.sum() * area` — the kernel is normalised, so the *integral* of every source (s = χSK, Λ = κ_c S K, foam_src, `_Lam_jet`) is resolution-independent; only the peak density and the support change.
- L297: `d = max(inlet.d_jet, 1e-6)` — **d_jet is not floored**. Fr² = U⊥²/(g′ d) (L299, L311), χ = 1/(1 + (Fr²/Fr_c²)^p) and `_Rc = cell_frac · 2.32 · Fr · d` (L307) all use the physical diameter. Old closure: `zc = U²/(B_dep d)` likewise (closure "chi" branch, L338 ff.).
- L618: the entrainment cell disc has its own floor `max(self._Rc, 2 * h)` (1.67 mm at 96²); inactive for both reference cases (R_c = 2.6–5 mm).
- LATTE_MODEL.md L60 states the floor openly ("半轴下限 1.5h，网格内自适应子格求积"); L269 table: "N | 96²（拟合）/ 256²（出图） | 4 mm 褐缝在 96² 上只有 5 格". So the doc admits the fit/render split but does not mention that the floor makes the inlet itself grid-dependent.

## 2. The numbers in the claim

High-cut footprint from the pitcher chain (`closure_numbers.py` CASES, taken from `Pitcher.jet` L315–324: continuity ellipse A_hit = A0 u0/|v| with capillary rounding; `milk.r_min` = 0.5 mm, L169, does not bind): r1 = 0.94 mm, r2 = 0.93 mm. Re-computed with `Solver.kernel` at x_hit = (0, 0.1), S_eff = 1:

| N | h (mm) | 1.5h (mm) | floor active? | K_peak [1/D²] | cells with K>0 | rms radius of K (mm) |
|---|---|---|---|---|---|---|
| 96 (repo) | 0.833 | 1.25 | yes (both axes) | **2684** | 12 | 0.702 |
| 96, floor removed (my re-implementation of L257–287 with floor 0) | 0.833 | — | — | 3536 | 8 | 0.572 |
| 128 | 0.625 | 0.94 | marginal (0.93→0.9375) | 4403 | 12 | 0.519 |
| 256 | 0.3125 | 0.47 | no | 6379 | 36 | 0.483 |
| analytic (1−ρ²)² on 0.94 × 0.93 mm | — | — | — | 6991 | — | 0.467 |

- Area factor 1.25² / (0.94 · 0.93) = **1.79** ✓ (the claim's ×1.8). Analytic peak of a floored kernel 3/(π·1.25²) = 3911/D², the discrete 2684 is lower only because 3×3 cell averaging of a 1.5-cell kernel clips the peak.
- K_peak 2684 ✓ matches code_map item 3 / §1.2 "K_peak = 594 (low) / 2684 (high)".
- So at the cut the per-cell deposition rate s_peak and the traction rate Λ_peak at 96² are 2684/6379 = **0.42×** their 256² values (and 0.38× the continuum value); integrated over the footprint they are identical.
- Decomposition of the 96² second-moment widening: floor 0.702² − 0.572² = 0.165 mm², cell averaging without floor 0.572² − 0.467² = 0.11 mm². The floor is the larger part (≈60 %) but a floor-free 96² kernel is still ×1.5 too wide in second moment and ×0.55 too low in peak. "Beyond ordinary discretisation error" is therefore half right: ordinary discretisation error at 96² for a 1.1-cell footprint is already large; the floor adds a comparable amount on top of it and — more importantly — makes it *deterministic* (a 2.5 mm minimum trail width instead of a noisy 1-cell line).
- Low pour (3.86 × 2.54 mm): both semi-axes > 1.25 mm, floor inactive at 96². The low pour deposits ≈ 90 % of the milk and forms the blocks/arcs; the floor only touches the cut line (and the V0.5 "cut" moves at z ≥ 7 cm in `v05_moves` / `_R_Z`).

## 3. Which fitted constants are actually coupled through the floor

Tracing the inlet fields (`inlet_fields` L289–352, `_bulk_flow` L605–626):

| quantity | depends on floored K? | depends on un-floored d_jet? | fitted constant exposed |
|---|---|---|---|
| χ (deposition / foam survival) | no | yes | Fr_c² (B_dep, p_dep) — **not** coupled via the floor |
| τ_new = 2U⊥/g′ | no | no | — |
| R_c = cell_frac·2.32·Fr·d (L307), cell disc L618 | no (own 2h floor) | yes | cell_frac — not directly coupled; but the "ring" footprint<r<R_c shrinks from 1.67 mm (256²) to 1.35 mm (96²) at the cut because its inner edge is the floored r_K |
| s = χ S K, foam_src, s_peak | **yes** (peak ×0.42) | — | the dwell time to whiten a cell at the cut line; the cut trail minimum width 2·1.25 = 2.5 mm vs 1.9 mm |
| Λ = κ_c S K, `_Lam_jet` | **yes** (peak ×0.42) | — | **κ_c**: pinning time 1/Λ_peak at the cut 2.4× longer at 96² for the same κ_c; integrated traction force ∫Λ(v*−u) unchanged |
| v* = κ_t u_in (+ κ_r term) | no | no | κ_t, κ_r not coupled |
| div U_s = S K − S_E K + S_E K_c (v2) | yes through K | yes through R_c | surface convergence peak at the jet ×0.42 at 96² |

So the only fitted constants whose *meaning* changes with N through L261 are κ_c (and in v2 the per-cell source/convergence densities that the Bingham/SSA film responds to nonlinearly). code_map item 3's list "κ_c, Fr_c², cell_frac" over-states it for Fr_c² and cell_frac: those are coupled to N only indirectly (through what the rest of the pattern does at the cut), which is ordinary resolution dependence, not the floor.

## 4. "96² 拟合 ≠ 256² 出图" — what the record shows

- The split is documented practice: LATTE_MODEL L269; `action_opt.py` default `--N 96` (L204) and `--verify` at 128 (L4); `cell_test.py` N = 96 (L17, L106, L136); all four reported fits (#9, cell_fit_N96, v2_final_fit_N96, v2s) are 96² (code_map L142); the 256² reruns are E21 (top-3 v1) and the `prof256*` folders.
- Same parameters at two grids, v1 #9 (`runs/joint_sweep_r2fix/results.json[9]` vs E21 / `codexmap/base_N256.json`): heart loss 0.386 (96²) → 0.401 (256²), tulip 0.551 → 0.555. Loss barely moves, but E21 reports qualitatively new structure at 256² (arcs, three nested petals, stem). None of this isolates the floor: the 96²→256² change bundles van Leer/Rusanov smearing (ν_num = |u|h), the time step, and the floor.
- E25′ (senior's 384/768/1536 grid study on V0.5): "窄入口求积在 128 失败；细切 ROI 时间步不收敛" — independent evidence that the thin cut is the resolution-sensitive phase, consistent with where the floor acts.
- Isolation experiment (this check): v1 #9 heart, V0.5 moves, N = 96, `Solver.kernel` monkey-patched (`skeptic/a15/floor_test.py`, repo untouched) with floor 1.5h vs floor 0. Result: see §4.1.

### 4.1 Floor-isolation run (v1 #9 heart, 96², floor 1.5h vs 0)

**Not completed.** Both runs (`floor_test.py 1.5 96 heart` and `floor_test.py 0.0 96 heart`, `action_opt.run_one` on the #9 physics with the V0.5 heart moves, outputs to `skeptic/a15/out_f*/`) were started in parallel expecting ≈ 65 s each (`tm.new96.log`), but the host was at load 11–21 on 4 cores (other checks running) and they had not produced a final frame after 11 min, so I killed them to stay within the 2-minute budget. Nothing was written to the repo. The experiment to settle the share of L261 in "96² ≠ 256²" is exactly this pair plus the same at 256² (floor inactive there, so one run): compare whiteness IoU / loss / cut-line width (`cmp.py` in the same folder is ready: IoU of w > 0.5, mean |Δw|, white run-length across x at y = ±0.05, ±0.15). Expected from §2: the cut line should be ≈ 0.6 mm (≈ 0.7 cell) wider and its peak whiteness lower with the floor; if the 96²-vs-256² difference of the cut is dominated by that, the claim's attribution holds; if the floor-free 96² cut is already as blurred as the floored one (van Leer / Rusanov smearing over 2–3 cells ≈ 2 mm > 0.6 mm), the floor is a minor contributor and the coupling is ordinary under-resolution.

## 5. Corrected statement

`solver.py` L261 floors both kernel semi-axes at 1.5h while L297 leaves d_jet physical. At N = 96 (h = 0.833 mm) the high-cut footprint 0.94 × 0.93 mm becomes 1.25 × 1.25 mm (area ×1.79; K_peak 2684 vs 6379 at 256², 6991 continuum), so per-cell deposition and traction densities at the cut are 0.42× their 256² values at identical integrals, and the cut trail has a 2.5 mm minimum width. This couples κ_c (and any nonlinear response to local source density) to N; χ/Fr_c²/B_dep and R_c/cell_frac use the un-floored d and are not coupled through this line. The low pour (3.9 × 2.5 mm) is unaffected at 96², so the floor cannot account for resolution dependence of the main blocks. Without the floor a 96² cut kernel is still ×1.5 too wide (rms) and ×0.55 too low in peak, so roughly 60 % of the cut-kernel widening is the floor and 40 % plain under-resolution. "96² fit ≠ 256² render" is documented practice (LATTE_MODEL L269, E21) but its attribution to L261 is not established by any run in the record, and my isolation run could not finish within budget (§4.1 names the experiment). Expected size of the floor-only effect: ≈ 0.6 mm extra cut-line width and 0.42× peak density at the cut, against 2–3 cells (≈ 2 mm) of van Leer / Rusanov smearing that exists at 96² with or without the floor.
