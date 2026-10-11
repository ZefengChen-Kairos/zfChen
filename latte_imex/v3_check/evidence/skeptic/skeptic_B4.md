# Skeptic check of CLAIM B4 — "奶泡与 crema 在面内永远同速；没有筏相对画布滑动、整体平移、画布被推到筏前的机制；v2 未修 [429] 关键问题 3"

Verdict: **confirmed with nuance**. The code fact (one face flux for m and ℓ, no relative in-plane transport of foam vs crema, v1's brown-only subduction switched off in v2) is traced line by line. Three of the four sub-phrases of the claim are over-stated and need rewording; see §4.

## 1. What the code DOES (solver.py, closure "v2" and "twolayer")

| Item | Line | Code | Meaning |
|---|---|---|---|
| Foam flux = layer flux | `solver.py:503,506,507` | `Gx_m, Gy_m = Gx, Gy` (for every closure except `skin`); `m_new = m + dt*s_m - dt*g.div(Gx_m*cfx, Gy_m*cfy)` | m is transported with exactly the face flux G that transports ℓ (step B final flux, `kernels.finish_B` L203). c_f is a van Leer limited face value of c = m/ℓ clipped to [c_L, c_R] (`kernels.limited_face_x` L107–128). So c is a passive contact discontinuity carried by the single film velocity u = q/ℓ. |
| Only relative transport | `solver.py:27`, `joint_cma.py:22` | `D: float = 1e-7`, `BASE_V2 = dict(..., D=1e-7)` | Mixing ℓ D ∇c with D = 10⁻⁷ D²/s: diffusion length in 10 s ≈ 1 mm·0.0125 = √(10⁻⁶) = 10⁻³ D = 0.08 mm. Inert. Not in the CMA search space (`joint_cma.py:16–21` lists no D). |
| Brown-only sources | `solver.py:412–418` (v2), `423–430` (twolayer) | v2: `s_skin = l_skin*max(divUs,0)`, `s_l = s + s_skin`, `s_m = s`; twolayer: `s_up = max(Ds,0)`, `s_dn = min(max(-Ds,0), 0.9(l-m)/dt)`, `s_l = s + s_up - s_dn` | v1 has a brown-only *sink* at the jet (foam stays, crema is subducted) — this is exactly the fix [429] proposed for its "problem 3". v2 keeps only a brown-only *source* s_skin with l_skin = 0.03 (a regularisation, LATTE L108) and has **no** brown-only sink. |
| v1 fix disabled in v2 | `solver.py:625–626` | `if self.P.closure == "v2": self.Ds = None` | The s_up/s_dn branch at L424 is skipped for v2 (`Ds is None`). LATTE L28 states this deliberately: "v1 的'上涌生成褐色表面、只卷下褐色液'两个源项删除". |
| Rigid translation allowed | `solver.py:525`, LATTE L213 | "K = ½ GᵀWG, symmetric positive semi-definite, rigid motions free"; `_ssa_operator` builds G from u_x, v_y, div, shear only | A Bingham plug can translate/rotate as a block without any stress; τ̂(c) = c τ̂_foam + (1−c) τ̂_crema with τ̂_crema = 0 (`solver.py:693`). |
| Film slides on the coffee | `solver.py:412–421` + `kernels.stepB_prepare` | drag target `vsx = (β U_s + Λ v_jet + s_l U_s)/(β + Λ + s_l)`, implicit θ = 1 + dt(β+Λ)/ℓ | u ≠ U_s is allowed; the film (foam **and** crema together) slips over the coffee surface flow with relaxation time ℓ/β₀. But U_s is a prescribed potential flow (`_bulk_flow`), it never reacts to the film. |

So, literally: foam and crema inside the floating layer share one velocity field everywhere; the only in-plane differential transport is D = 10⁻⁷; the only vertical exchange that would let crema leave the film is v1's s_dn and it is off in v2.

## 2. What the docs CLAIM

- LATTE L113: `∂m/∂t + ∇·(c q) = s_f + ∇·(ℓ D_m ∇c)` — single velocity, explicitly. L257: D_m = 10⁻⁷ "不相溶，近零". L221: D′ step "可忽略".
- LATTE L28/L110: white/brown separation in v2 is to come from **kinematic thinning** ("上涌处表面散开，漂浮层被拉薄，奶泡变薄就透出褐色") plus opacity W = 1 − exp(−m/m_o) (L141) — i.e. by *thickness*, not by relative motion. This is a real (if different) separation mechanism, which the claim's wording ignores.
- docs_map V1 (L63) and C8 (L253) say the same thing as the claim; C8's "v2 没有修 conv [429] 认定的'关键'问题 3" is accurate in that v2 still has one velocity, but note that [429]'s *own* proposed fix (brown-only source/sink) was implemented in v1 (`solver.py:423–430`, commit era E-6/E-7) and then **removed** by v2, not "never done".
- conv [429] problem 3 (conversation_part3.md ~L540): "奶泡和褐色液体用同一个速度运动，无法互相分离（我认为这是关键）" — the author's proposed remedy was the upwelling/subduction source pair, i.e. a *vertical* exchange, never an in-plane relative velocity. So even v1 never had "筏相对画布滑动"; the claim's "v2 未修" should not imply v1 had it.

## 3. Numerical check (`skeptic_B4_raft2.py`, N = 64, closure v2, no inlet, β = 0.5, c_p = 1.677 from g′ = 75, H_f = 3 mm)

Equal-thickness raft (ℓ = 1 everywhere, m = 1 in a disc of radius 0.08 D at x = −0.10) given u₀ = 0.4 D/s; crema at rest; 0.3 s.

| | τ̂_foam = 0.3 | τ̂_foam = 0 |
|---|---|---|
| u at the front, foam side / crema side, t = 0.05 s | 0.1505 / 0.1527 | 0.2324 / 0.2070 |
| ℓ just ahead of the front, t = 0.05 s | 1.066 (compression), 0.93 behind (rarefaction) | 1.079 / 0.92 |
| crema 0.05 D ahead of the front, u at t = 0.05 s | 0.13 | 0.13 |
| lateral v at y = 0.11 (crema flowing around), max | 0.107 | 0.128 |
| u across the raft at t = 0.3 s | 0.0664–0.0666 (**plug**) | 0.10–0.16 (deforms) |
| raft centroid displacement in 0.3 s | 0.031 D | 0.042 D |
| free-coasting displacement against drag only, u₀/β(1−e^{−βt}) | 0.111 D | 0.111 D |
| cells with 0.1 < c < 0.9 on the mid-row | 4 | 4 |

Reading: (i) the contact velocity is continuous — the crema touching the raft moves at the raft speed, so the raft can advance only by shoving the crema film (ℓ-compression wave at c_p ≈ 1.7 D/s, sideways flow around the disc); (ii) ~70 % of the raft's displacement is lost because its momentum is shared with the whole crema film within ~0.05 s (sound crossing time of the raft); (iii) with τ̂_foam = 0.3 the raft does move as a rigid plug, so "没有整体平移机制" is false as a statement about the film rheology; (iv) the foam/crema interface stays 2–4 cells sharp (van Leer), consistent with video_obs §7 item 3's "sharp interfaces kept sharp" — the single-velocity film does *not* by itself smear interfaces.

First test (`skeptic_B4_raft.py`, ℓ = m = 4 in the raft) collapsed to ℓ ≈ 0.9 within 0.05 s under c_p = 1.68 with or without τ̂ — that is claim B3, not B4, and is noted only to show B4's test must use an equal-thickness raft.

## 4. Verdict and corrected statement

**Confirmed (core):** in v2 the foam m and the brown material ℓ − m of the floating layer have one in-plane velocity (`solver.py:503–507`; LATTE L113); the only relative in-plane transport is D = 10⁻⁷ (inert); v2 removed v1's brown-only subduction s_dn (`solver.py:625–626`, LATTE L28) and kept only the l_skin = 0.03 brown-only source, so no material can leave the film under a raft. Consequently a raft cannot glide over the canvas: to move at all it must set the crema film in motion at the same contact speed, which (numerically) dumps ~70 % of its momentum into the film in 0.05 s.

**Over-stated parts to fix:**
1. "没有整体平移机制" — wrong: the SSA/Bingham operator has rigid motions in its null space (`solver.py:525`, LATTE L213) and a τ̂_foam = 0.3 raft translates as a plug (u uniform to 3×10⁻⁴). What is missing is a *low-resistance substrate*, not block kinematics.
2. "没有画布被推到筏前的机制" — partly wrong: crema *is* pushed ahead (ℓ 1.07 compression, u 0.13 D/s at 0.05 D ahead) and flows around the raft (v up to 0.1 D/s). What is missing is the canvas being displaced *under* the raft (3-D subduction) and any slip at the foam/crema contact; in v2 the crema between two rafts is volume-conserved and can only be squeezed sideways (→ B3).
3. "筏相对画布滑动" — ambiguous: the film does slide relative to the *coffee* (drag β₀(u − U_s), implicit θ), so if 画布 = coffee surface the statement is false; if 画布 = the crema part of the floating layer it is true. The diagnosis should say "相对 crema 膜".
4. "v2 未修 [429] 问题 3" — accurate for the velocity sharing, but [429]'s own remedy (brown-only source + subduction) *was* implemented in v1 and v2 **reverted** it, replacing it with thickness-based separation (kinematic thinning + opacity W(m)). The claim should say "v2 把 v1 对问题 3 的修法撤掉，换成按厚度分辨白褐；在面内速度上仍未修".

**Suggested corrected B4:** 在 v2 的漂浮层里奶泡 m 与褐色部分 ℓ−m 共用同一面通量 G（solver.py L503–507，LATTE L113），唯一的相对输运是 D_m = 10⁻⁷（惰性）；v1 对 [429] 问题 3 的修法（射流处只卷下褐色液 s_dn，L423–430）在 v2 被关闭（L625–626，LATTE L28），只剩 l_skin = 0.03 的褐色源。于是筏不能在画布上滑行：它前缘的 crema 必须以同样的接触速度被推走并绕流（数值检验：0.05 s 内筏的动量摊给整层，位移只有自由滑行的 30 %），crema 也不能潜入筏下。Bingham 膜本身允许筏整体平移（算子零空间 = 刚体运动，检验中 τ̂ = 0.3 的筏速度均匀到 3×10⁻⁴），膜也能相对咖啡 U_s 滑动（β₀ 拖曳）；缺的是筏相对 **crema 膜** 的滑动和 crema 的垂向让位，不是刚体运动学本身。v2 用"厚度→不透明度"代替了相对运动来分辨白褐（LATTE L28/L141），这是另一机制，其是否够用见 B3/B5。

## 5. Files
- Notes: `/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review/skeptic_B4.md`
- Checks: `.../review/skeptic_B4_raft.py` (thick raft, shows B3 collapse), `.../review/skeptic_B4_raft2.py` (equal-thickness raft, the B4 test; 36 frames in < 1 s at N = 64).
- Code read: `/home/user/zfChen/latte_imex/solver.py` L353–626, 679–713, 757–813; `kernels.py` L82–152, 203–233; `joint_cma.py` L16–24.
