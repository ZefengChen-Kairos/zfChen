# Skeptic check of A10 — "先推后白" patch: right ingredient, three eyeballed numbers, τ_d 6× too short, off everywhere

Verdict: **confirmed with nuance** (one quantitative refinement, one wording fix).

## What the code does (solver.py, verified 2026-10-11)
- L331–332: `if P.tau_d > 0: self.tau_cur = P.tau_d` — τ is a **constant**, no dependence on U_perp / z (the τ∝U variant was deleted in commit 17daddf; PITCHER L247 says it was "试过没用").
- L367–371: `tot = msub + dt*s; msub_new = tot/(1+dt/tau); s = (tot-msub_new)/dt` — first-order (exponential) reservoir; `s_mom = s` is bound at L363 **before** the reassignment, so impact momentum is immediate while the mass source is delayed.
- L374–379: msub advected by `sub_advect`·u (first-order upwind).
- L447–451: `a -= dt*coef*sub_pressure*grad(msub_new)` — the only term in any closure that pushes the existing film before white appears.
- L471–474: `q_new += dt*s*u^n` after `q_from_gradient` — O(dt) inconsistency (code_map item 15).
- Params defaults (L50–56): `tau_d = 0.0` (off), `sub_advect = 0.3`, `sub_pressure = 0.5`.

## Is it off in every fit and web preset?
- `runs/joint_sweep_r2fix/results.json[9]` physics keys: cp, beta, nu, kappa_c, kappa_t, kappa_r, B_dep, p_dep, kQ — **no tau_d** → Params default 0. action_opt.physics9() (L23–26) therefore runs τ_d = 0; `export_web` L156 forces `prm["tau_d"] = 0.0` in every preset.
- `web/scripts.json`: `_default_physics.params.tau_d = 0.0` and all 14 scripts (6 originals + sweep9 + 6 `_opt`) have tau_d 0.0. `web/latte2d.js` L14 default `tau_d: 0`.
- optimize.py, twolayer_fit.py, joint_sweep.py, joint_cma.py, cell_test.py, pitcher.py: no mention of tau_d/sub_pressure/sub_advect → off.
- Only `runs/codexmap/delay/*` (delay_test.py experiments, 128²/256²) ran τ_d > 0. Git: 9b9c8ea web v20 default **on**, 74f0bdb web v21 default **off**.

## "三个肉眼折中数" and "2 条 7.2 mm"
- PITCHER L247: "这是三个图案上肉眼折中的结果，不是标定". Confirmed.
- §14 table round 5 (L243): push_heart **2 / 7.2 mm**, tulip 9 / 3.8; round 2 (w=1, τ∝U) gave 11 / 6.2 (max 10.6) but hollowed the heart (conv C-8). Confirmed.

## τ_d 0.2 s vs video 1.3 s
- video_obs §0: first-white 0.9–2.1 s, median ≈1.3 s; mid-pattern stages (push_heart stage 2/3 = 1.2 s, tulip stage 2 = 1.3 s) are inside the modelled phase, so the comparison is fair. 1.3/0.2 = 6.5. Confirmed.
- PITCHER L256 defends 0.2 s as the fountain rise time 2U⊥/g′; the doc never checked it against video although L247 says it is directly measurable.
- **Nuance (python check)**: the reservoir is a first-order filter, not a dead time. With constant deposition, 10 % of the deposited mass has surfaced by t ≈ 0.21·τ (0.04 s at τ=0.2; 0.28 s at τ=1.3) and 50 % by ≈1.6·τ. So the model's "first white" appears almost immediately for any τ_d; raising τ_d to 1.3 s would only stretch the whitening, not reproduce the 1–2 s brown dead time. The discrepancy is structural (filter vs lag) as well as 6× in magnitude.
- Height dependence: physics_prior §4 item 1 argues the delay should grow with impact velocity/depth; the video gives no clean height series (stage-1 2.1 s vs stage-2/3 1.2 s differs in pre-pattern state, not height). The code's constant τ is a fact; whether real delay is height-dependent is a prior, not an observation.

## Wording fix
"上线工作点" is the §14 working point (τ 0.2, w 0.5, α 0.3; commit faa491e; web v20 default on) — it is no longer online: v21 (74f0bdb) turned it off. Better: "文档工作点".
