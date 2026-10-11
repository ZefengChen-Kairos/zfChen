# Skeptic check of CLAIM A13 — 参数冗余 / 死参数

CLAIM A13: `Params.kappa_Q` 从未被 solver 读取；c_p、β、ν 对末帧相关系数 ≈ 0；真正起作用只有 4 个无量纲组。
EVIDENCE GIVEN: code_map §1.0、§2；notion_z1 E17/E25；PITCHER §13。

Verdict: **confirmed with corrections** — the "dead κ_Q" and "c_p, β ≈ 0 for the end-frame loss" parts hold (and I re-derived the numbers from the raw sweep files), but (i) κ_Q is only dead in the *pitcher* chain, not globally; (ii) ν is weakly but not negligibly correlated and its range sits below the numerical viscosity, so "≈ 0" is a resolution artefact rather than model redundancy; (iii) β is the *strongest* predictor in E25's gap-count score, so E25 contradicts the β clause instead of supporting it; (iv) "only 4 dimensionless groups act" is a misreading of PITCHER §13.

---

## 1. `Params.kappa_Q` — where it is and is not read

### What `solver.py` does
- `solver.py` L28: `kappa_Q: float = 125.0  # received volume -> layer [1/D_L]` — the dataclass field.
- L61 (comment only): "The inlet must be S_eff = Q / D_L^3 (k_Q = 1)".
- L100 (comment only, `Inlet.S_eff`): "kappa_Q * Q_received".
- `grep -n kappa_Q solver.py` gives exactly these three lines. **No `P.kappa_Q` / `self.P.kappa_Q` anywhere in `Solver`.** The solver consumes `Inlet.S_eff` as given. So "从未被 solver 读取" is literally true.

### Where it IS read (the claim and code_map §1.0 miss one path)
1. `pitcher.py` L545 `Coupling.kappa_Q = 125.0`, L565 `Q_rec = S / self.kappa_Q`. `Q_rec` is used **only** in the `else` branch (`footprint != "physical"`, L574-577) to build the continuity-ellipse footprint. Every official fit builds `Coupling(c_S=kQ/D_L**3, c_U=1/D_L, c_u=1/D_L, footprint="physical")` (`action_opt.py` L74, `joint_sweep.py` L89, `joint_run.py` L26), so there `Coupling.kappa_Q` is dead too, and `Params.kappa_Q` is pinned: `pd = dict(ph, D=1e-7, kappa_Q=125.0, return_law="v05")` (`action_opt.py` L76, `joint_sweep.py` L92). The flow scale is the *separate* scalar `kQ` → `c_S = kQ/D_L³` (#9: kQ = 1.326; v2: forced to 1 at `action_opt.py` L72-73). So in the pitcher chain (physics #9, cell fit, v2F, v2s) the claim holds: κ_Q is dead and the thickness unit is set by kQ/c_S.
2. **`v05_program.py` L58 `self.kappa_Q = self.config["parameters"]["kappa_Q"]`, L102 `Q = flow / self.kappa_Q`, L113 `U_perp = Q/(π r²)`, L121-122 `area = Q/speed_material`, `d = 2 sqrt(area/π)`.** This is the `controls`/`run.py`/`optimize.py` chain: `controls.build()` L212 copies the physics dict into `cfg["parameters"]`, so `physics["kappa_Q"]` sets U⊥, the footprint aspect, d_jet and the tan 60° cap of u_in — i.e. the deposition switch χ = 1/(1+(U⊥²/(B_dep d))^p). With U⊥ ∝ κ_Q⁻¹ and d ∝ κ_Q^{-1/2} this is U⊥²/(B_dep d) ∝ κ_Q^{-3/2}: κ_Q in that chain is (almost) a re-parameterisation of B_dep, exactly the `unified.retarget` degeneracy PITCHER §13 describes.
   - It is a *live search variable* there: `controls.PHYSICS_BOUNDS` L237 `kappa_Q=(20, 400, "log")`, and the E6 joint CMA-ES result `runs/opt/joint_v1_heart_push_heart/best.json` has `kappa_Q = 64.81` (moved from 125; `B_dep = 1470`, `nu = 0.01` at the bound, `D = 0.0`).
   - `optimize.py` L173 prints `kQ={ph['kappa_Q']:.0f}` — note it calls `Params.kappa_Q` "kQ", a different quantity from `joint_sweep`'s `kQ`. Two things named kQ/κ_Q with different meanings; `Inlet.S_eff` docstring (L100) is stale for the pitcher chain.

**Corrected statement for part 1**: `Params.kappa_Q` is never read by `Solver`; in the pitcher chain used for all reported fits it is pinned at 125 and inert (the inlet scale is `kQ`/`c_S` instead), and even `Coupling.kappa_Q` is bypassed by `footprint="physical"`. In the V0.5-controls chain (`run.py`, `optimize.py`, E6) it is live through `Program` and degenerate with B_dep (U⊥²/(B_dep d) ∝ κ_Q^{-3/2}).

## 2. "c_p、β、ν 对末帧相关系数 ≈ 0"

### Recomputed from the raw sweep files (Spearman; log of scale parameters; all samples `ok`)
`runs/joint_sweep/results.json` (E17, n = 100, LHS over c_p 0.3-4, β 1-10, ν 1e-4-3e-3, …, B_dep 300-9915, kQ 0.8-3) and `runs/joint_sweep_r2fix/results.json` (n = 60, B_dep 1160-1e4, β 1.2-6.3):

| param | round 1 tulip loss | round 1 heart loss | round 1 mean loss | r2fix tulip loss | r2fix heart loss | r2fix mean loss | pooled n=160 mean loss (ρ, p) |
|---|---|---|---|---|---|---|---|
| c_p | −0.01 | −0.03 | −0.02 | +0.17 | **−0.24** | +0.03 | −0.04 (0.66) |
| β | +0.02 | −0.05 | −0.06 | +0.10 | **−0.31** | −0.05 | −0.03 (0.73) |
| ν | −0.12 | −0.19 | −0.19 | −0.20 | +0.02 | −0.19 | **−0.16 (0.039)** |
| κ_c | +0.01 | −0.05 | −0.08 | +0.19 | +0.16 | +0.28 | −0.12 (0.12) |
| κ_t | −0.11 | −0.05 | −0.06 | +0.07 | −0.22 | −0.10 | −0.15 (0.06) |
| κ_r | +0.14 | +0.10 | +0.14 | −0.06 | +0.13 | +0.02 | +0.16 (0.04) |
| B_dep | **−0.54** | **−0.76** | **−0.73** | +0.55 | −0.26 | +0.33 | −0.49 (<0.001), Pearson −0.71 |
| p_dep | +0.19 | +0.22 | +0.22 | +0.26 | −0.10 | +0.19 | +0.19 (0.016) |
| kQ | +0.18 | −0.35 | −0.30 | **+0.74** | **−0.55** | +0.36 | −0.21 (0.008) |

(Loss = `optimize.loss_fields`: blurred-L1 + 0.5(1−IoU) + 0.5|ΔA| on the final frame, 96².) This matches E17's text (B_dep −0.62/−0.78 in the page vs −0.54/−0.76 here; sign conventions and sample filtering differ slightly).

Reading:
- **c_p and β: ≈ 0 for the pooled end-frame loss — confirmed** (|ρ| ≤ 0.06, p > 0.6). Consistent with `runs/cp_sweep.json` (heart script, 128², T = 2.5 s): c_p 0.25 → 4 changes `white_area` 0.239 → 0.293 (+22 %) and `l_max` 2.87 → 1.64, while notion_z1 L66 reports IoU 0.80-0.96 and "2 vs 4 no difference". So c_p does change the fields (mound height, pressure energy 0.065 → 15.8) but not the thresholded whiteness pattern much — plausibly because whiteness c = m/ℓ is a ratio and step C advects m with the same face flux G that spreads ℓ (`code_map` §1.0 step C), so the pressure term moves ℓ and m together. (Mechanism plausible, not separately tested here.)
- **β: ≈ 0 only for *this* score.** The claim cites E25 as evidence, but E25 (notion_z1 L47) is the gap-count score over the same 160 samples and there **β is the strongest predictor (−0.39 tulip, −0.51 heart)**; the page itself (notion_z1 L234) notes the two scores disagree on which parameter matters (B_dep large vs β small). In the r2fix subset β is −0.31 on heart loss / +0.35 on heart IoU. So "β 对末帧相关系数 ≈ 0" is true of the L1/IoU loss only; it is false of the gap statistic, which is also a final-frame quantity and the one the project actually cares about.
- **ν: not ≈ 0** (−0.16 pooled, p = 0.04; −0.19 in each round separately). More importantly the swept range 1e-4-3e-3 D²/s lies *below* the Rusanov numerical viscosity ν_num ≈ |u| h ≈ 0.005-0.021 D²/s at 96² (code_map §6, claim A14). ν is masked by the scheme, not redundant in the model — the correct conclusion is "ν is unidentifiable at 96²", which is a numerics statement, not a parameter-redundancy statement. (conv_part2 L201: "ν-gaps −0.1… 缝不是被黏性抹掉的，是被填掉的".)
- Caveat on the method: every sample also varied the action vector (height, flow, wiggle…), so these are *marginal* rank correlations in a 9+13-dim LHS; |ρ| ≈ 0 means "no monotone marginal effect that beats the action-induced variance", not "no effect". E17's own caveat (conv_part2 L97): "This is a sensitivity map, not a calibration."

## 3. "真正起作用只有 4 个无量纲组（PITCHER §13）"

PITCHER_MODEL.md L174-220 says something narrower: Codex's V0.5 *inlet* is dimensionally rescaled (κ_Q = 125 ⇒ 0.1-0.2 mL/s, r 0.64 mm, U 0.06-0.14 m/s vs real 10-20 mL/s, 3-5 mm, 0.7-1.2 m/s), `unified.retarget` keeps U²/(B_dep d) and the jet angle invariant under a κ_Q change, so what Codex "guessed" is the working point of four groups **of the inlet/source closure**: U²/(B_dep d), κ_Q Q, κ_r V_R, κ_t|v∥|. It then back-solves B_dep ≈ 2800, kQ ≈ 1.8, κ_r ≈ 0.04, κ_t ≈ 0.6 for the physical inlet. It does **not** say c_p, β, ν are inert; those are film-dynamics constants outside the inlet retargeting and are simply not discussed there. docs_map O5 paraphrases this as "有意义的只是 4 个无量纲组" and A13 inherits the over-reach.

The data do not support "4 groups" either: of the four, κ_r is ≈ 0 in the fits (#9: 0.0085) and weak in the sweep (+0.14/+0.16); the dominant one is U²/(B_dep d) via B_dep (−0.73 round 1), with kQ (−0.30 / +0.74 & −0.55 pattern-dependent) and p_dep (+0.22) secondary, κ_t marginal. So the honest sensitivity structure for the end-frame loss is "one dominant group (χ), two secondary (kQ, p_dep), the rest marginal", and for the gap score "β first". That is a *different* claim from "only 4 groups act".

## 4. What the evidence given actually supports vs not

| clause | holds? | note |
|---|---|---|
| `Params.kappa_Q` never read by `Solver` | yes | `solver.py` L28/L61/L100 only; dead in all pitcher-chain fits |
| "只在 footprint='v05' 的 Coupling 用" (diagnosis parenthetical) | incomplete | also live in `v05_program.Program` (`optimize.py`/`run.py`/E6, searched 20-400, E6 best 64.8), degenerate with B_dep |
| c_p ≈ 0 on end-frame loss | yes | pooled ρ −0.04; cp_sweep IoU 0.80-0.96 |
| β ≈ 0 on end-frame loss | yes for L1/IoU loss; **no** for gap count (E25: −0.39/−0.51, strongest) | cited evidence E25 contradicts the clause |
| ν ≈ 0 | weakly no (−0.16, p 0.04) and masked by ν_num at 96² | a numerics/identifiability issue (A14), not redundancy |
| only 4 dimensionless groups act | not supported | PITCHER §13 is about inlet retargeting; sweep shows 1 dominant + 2 secondary |

## 5. Corrected statement (proposed replacement for A13)

A13 [minor] 参数冗余 / 不可辨识：`Params.kappa_Q = 125` 从未被 `Solver` 读取；在所有正式拟合走的 pitcher 链里它被钉死、`Coupling.footprint="physical"` 也绕过了 `Coupling.kappa_Q`，入口尺度另由 `kQ`（`c_S = kQ/D_L³`）给出（`Inlet.S_eff` 注释已过时）；只在 V0.5 控制链（`run.py`/`optimize.py`/E6）里它经 `v05_program.Program` 决定 U⊥、足迹与 d_jet，与 B_dep 简并（U⊥²/(B_dep d) ∝ κ_Q^{-3/2}）。在 160 组 96² LHS 上，末帧 L1/IoU 损失对 c_p、β 的 Spearman ≈ 0（−0.04、−0.03），对 ν −0.16，且 ν 的扫描范围（1e-4-3e-3）低于 Rusanov 数值黏性 |u|h ≈ 0.005-0.02，故 ν 在 96² 上不可辨识；但 **褐缝计数对 β 相关 −0.39/−0.51，是最强因子**（E25），两种评分给出的"重要参数"不同。末帧损失实际由 χ 组 U⊥²/(B_dep d) 主导（B_dep −0.73），kQ、p_dep 次之；PITCHER §13 的"四个无量纲组"只是 Codex 入口重标的不变量，不是求解器只有四个有效参数。

## 6. Experiment that would sharpen it (optional)
One-at-a-time sweeps of c_p, β, ν at fixed #9 physics and fixed actions at 256² (not 96²), scored by both the loss and the ≥1.5 mm gap count; that separates "no effect" from "effect buried under action variance / numerical viscosity". ~6 runs × 3 params × 2 patterns at 256² — longer than the 2-min budget here, so not run.

Files read: `/home/user/zfChen/latte_imex/solver.py`, `pitcher.py`, `action_opt.py`, `optimize.py`, `controls.py`, `v05_program.py`, `joint_sweep.py`, `cp_sweep.py`, `run.py`, `joint_cma.py`, `PITCHER_MODEL.md` §13; review notes `code_map.md` §1.0/§2/§7, `notion_z1.md` E6/E17/E19/E25, `docs_map.md` O5, `conv_part2.md`; data `/home/user/zfChen/runs/joint_sweep/results.json`, `runs/joint_sweep_r2fix/results.json`, `runs/cp_sweep.json`, `runs/opt/joint_v1_heart_push_heart/best.json`.
