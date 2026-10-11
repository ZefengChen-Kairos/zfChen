# Skeptic check of B8 — "咖啡无惯性、无记忆、无涡量；停倒即 None；拖曳瞬间指向静止；β₀ 的停倒测法量不到"

Verdict: **confirmed, with three nuances** (the film keeps sliding ℓ/β₀; the repositions are not actually zero-flow in the simulated push_heart; "量不到" should read "only a lower bound / confounded with τ_b").

Scripts: `skeptic_B8_check.py` (v2, N = 48, pour 0.4 s then stop), `skeptic_B8_script.py` (push_heart inlet timing).

## 1. What the code does (solver.py, commit state as read)

**Coffee flow = quasi-steady potential flow, re-solved every control frame, zero the frame the inlet stops.**

- `advance_frame` L757–767: `s, Lam, vsx, vsy, chi = self.inlet_fields(inlet)` then `if closure in ("twolayer","v2"): self._bulk_flow()`. Both run once per 1/120 s frame; nothing is carried over.
- `inlet_fields` L292–296: `if not inlet.active or inlet.S_eff <= 0: ... self._SE = 0.0; self._S = 0.0; self._Lam_jet = None`.
- `_bulk_flow` L600–603:
  ```python
  S, SE = getattr(self, "_S", 0.0), getattr(self, "_SE", 0.0)
  if S <= 0:
      self.Ub = None; self.Ud = None; self.Ds = None; self.divUs = None
      return
  ```
- L611–621: `U = grad(phi)` with `phi = LU^{-1}(-div)`, div = `S K − <S K>` and `SE (K_c − K)`. The solve has no time derivative; `U_b` is a function of the *current* (S, SE, hit point, R_c) only. Being a gradient, it is irrotational: measured `|curl U_b|_max / (|U_b|_max/h) = 5.8e-3` (discretisation noise) during the pour.

**Consequence for the film drag (v2 branch of `substep`):**
- L404: `ubx, uby = Ub if Ub is not None else (zeros, zeros)`.
- L403: `Lam = self._Lam_jet if ... is not None else zeros` → after the stop Λ = 0.
- L414–423: `s_skin = 0` (divUs None); `vsx = (β·ubx + Λ·jx + s_l·ubx)/(β + Λ + s_l) = 0`; `Lam_eff = β + 0`, `beta_B = 0`.
- `kernels.stepB_prepare` L165–175: `theta = 1 + dt(beta_B + Lam/l)`, `a = (q* + dt(s + Lam) v*)/theta` → with v* = 0 the film momentum is relaxed toward **rest** at rate β₀/ℓ. So "拖曳瞬间指向静止" is exactly what the code does.
- The "twolayer" (v1) branch L439–442 has the same structure: the explicit `+dt β ℓ U_b` term disappears when `Ub is None` while the implicit `−β q` stays → target = rest.

**Numerical check (v2, X0_V2 physics, Q = 15 mL/s, off-centre, u_in = 0.6 D/s):**
```
during pour: Ub not None; |Ub|max 0.872 D/s; |curl Ub| ratio 5.8e-3
first frame after stop: Ub None, Ud None, divUs None, Ds None, foam_src None; submerged f still 4.8e-2 (surfaces in place, Ud=None → not advected)
film mean |u|: 0.0706 (t+8 ms) → 0.0478 (t+0.11 s) → 0.0247 (t+0.21 s) → 0.0134 (t+0.41 s)
fitted decay rate of mean film speed, first 0.1 s: 3.77 /s  (β₀ = 3, ℓ ≈ 1–1.4)
```
So: the coffee stops instantly; the **film** keeps sliding with e-folding time ≈ ℓ/β₀ (0.27 s here; the extra 0.8/s over β₀ is the SSA ν_max stress and pressure).

## 2. What the docs claim

- LATTE_MODEL.md L69: "咖啡……水平流动由两个准稳态势流给出（第 2.3 节），不单独求动量"; L84 "### 2.3 咖啡：两个准稳态势流（每帧解一次）". TWOLAYER_CELL_MODEL.md L12 says the same. The docs therefore **do not claim** coffee inertia; code and doc agree. B8 is a physics-shortcoming claim, not a code/doc mismatch.
- LATTE_MODEL.md L254 (parameter table): `β₀ | 膜底剪切的速率 | 1–10 /s | 停倒后表面停下来的时间` — this is the measurement method B8 attacks.
- LATTE_MODEL.md L313 lists only the h_s = H/2 assumption of the potential flows as open; nothing about memory/τ_b.

## 3. Conversation / prior evidence

- conv [457] (conversation.md L5170 ff., the assistant's own words): "**膜下的咖啡层没有惯性**：U_v 和 U_c 是根据当前这一帧的倒入量算出的准稳态势流，停倒的瞬间就变成零，没有任何记忆，也没有涡量。" and "表面膜有惯性：…停倒后膜还会滑一段，大约在 1/β ≈ 0.3 s 内被拖停". It proposes a depth-averaged coffee momentum equation or a relaxation `U_b → potential solution` with spin-down time τ_b "可以从视频里'停倒后图案还漂多久'直接量出来". conv_part3.md L358 (O-4): "deferred to process video; not done". So the author already conceded the mechanism; the evidence pointer is accurate.
- physics_prior.md §3f: "cup recirculation turnover 1–2 s; decay after pour ends 1–3 s [I]"; "crater/near-field sheet decay after the stream stops 0.1–0.3 s [I]". Both rows are tagged **[I] = inferred**, not measured — the 1–3 s figure is an estimate, not video data. video_obs.md has no measured post-lift drift speed/duration either (the lift rows only describe shape changes).

## 4. The β₀ measurement sub-claim

In the model the "停倒后表面停下来的时间" **is** ℓ/β₀ (verified above: decay rate ≈ β₀ for ℓ ≈ 1), so the L254 method is internally consistent with the model. The diagnosis' point is that in the real cup the surface keeps moving as long as the coffee underneath does; the observable is ≈ max(τ_b, ℓ/β₀). If τ_b (1–3 s, inferred) exceeds ℓ/β₀ (0.1–1 s for β₀ 1–10, ℓ = 1; up to 2 s for a ℓ = 6 block at β₀ = 3), the measurement returns τ_b, and fitting it as β₀ gives β₀ ≈ ℓ/τ_b — a **lower bound**, biased low, and the model then "explains" the drift with a slow film instead of a moving coffee. "量不到" is therefore right in substance but overstated: the method gives a bound and confounds two time constants; it does not give nothing. Whether the confound bites depends on τ_b vs ℓ/β₀, which nobody has measured (physics_prior's 1–3 s is [I]).

## 5. Nuance on "没有 0.18 s 重定位期间的持续环流"

`skeptic_B8_script.py` (stage_actions `SPEC["push_heart"]["x0"]`, BaristaScript with the NX geometry): the reposition moves are specified with Q = 0, but the **delivered** inlet `S_eff` stays > 0 continuously from t = 0.225 s to t = 4.483 s, i.e. through both lifts and both repositions (lip buffer τ_stop + flight, cf. code_map item 12: 30–34 mL/s still arrive during the reposition). `S_eff = 0` occurs only for 215 of 966 frames, essentially after t = 4.48 s (final lift/stop). So in the simulated push_heart `Ub` is **not** None during the repositions; the coffee flow is nonzero but slaved frame-by-frame to the sweeping hit point. The missing thing during repositions is *memory* (the flow pattern jumps with the hit point), not flow per se. This sub-sentence of B8 should be reworded.

Side observations (do not change the verdict): after the stop the submerged foam `f` still surfaces (`tau_f` decay, L396–400) but is no longer advected (`Ud` None), so it surfaces exactly where it was; and the potential flow is depth-uniform (no Ekman/bottom layer), so even during the pour there is no mechanism for the coffee flow to lag the input.

## 6. Verdict

**Confirmed (with nuance).** Mechanism traced: L292–296 → L600–603 → L403–404, L414–423 → `stepB_prepare` L171–174; gradient flow ⇒ no vorticity; no state carried between frames ⇒ no memory; tested numerically. Corrected statement:

> 咖啡层是每帧按当前入口重解的准稳态势流（无旋、深度均匀、无状态），停倒的那一帧 `Ub/Ud/divUs = None`，膜的拖曳目标瞬间变为静止；**膜本身有动量**，以 ℓ/β₀（β₀ = 3 时 ≈ 0.3 s，ℓ = 6 的厚块 ≈ 2 s）衰减。模拟 push_heart 的重定位期间入口实际上并未归零（S_eff 连续 > 0 至 4.48 s），缺的是流场记忆而非流场本身。L254 的"停倒后表面停下来的时间"在模型内确实等于 ℓ/β₀，但真实量到的是 max(τ_b, ℓ/β₀)：若咖啡自旋衰减 τ_b（physics_prior 估 1–3 s，[I] 未实测）大于 ℓ/β₀，该测法只给 β₀ 的下界并把 τ_b 错记进 β₀。作者在 conv [457] 已承认此简化并提出 τ_b 弛豫方案，未实施（O-4）。

Experiment that would sharpen it (not needed for the verdict): pepper tracers on the canvas, stop the pour, time the tracer and the foam-block velocities separately; τ_b from the tracers, β₀ from the block's lag.
