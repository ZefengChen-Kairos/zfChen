# Skeptic check of A14 — "细纹宽度由网格和数值扩散决定且不收敛"

Verdict: **confirmed with nuance** (the mechanism and the grid-dependence of fine lines hold and are measurable on the stored fields; three of the quoted numbers are off, and the importance is overstated: at 256² the macro pattern and the wide gaps are converged, so this is not why wide brown gaps are missing).

All line numbers: `/home/user/zfChen/latte_imex/solver.py` (S), `/home/user/zfChen/latte_imex/kernels.py` (K). New numbers come from
`/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review/skeptic_A14_run.py` (a 1.5 s old-closure push_heart at 96²) and from re-measuring the stored E23 fields `/home/user/zfChen/runs/codexmap/gapdiag/base_N256.npy` / `base_N512.npy`.

## 1. What the code does (traced)

| claimed term | code | verified? |
|---|---|---|
| Rusanov α = 2 max\|u\| on momentum | K L59–78 `rusanov_fluxes`: `a = 2.0*max(abs(uL),abs(uR))`, flux `0.5(qL uL + qR uR) − 0.5 a (qR − qL)` | yes. Modified equation ν_num = αh/2 − u²dt/2 = \|u\|h(1 − Co/2); at Co ≤ 0.25 this is 0.88–1.0 × \|u\|h. Acts on **q**, not on c. |
| "Rusanov-type" term on ℓ, w = max\|a/ℓ\| | K L82–96 `mass_fluxes`: `Gex = 0.5(aL+aR) − 0.5 w (lR − lL)`, w = max\|a/lsafe\| | yes, ν_num,ℓ ≈ \|u\|h/2. Acts on ℓ only. |
| van Leer limited upwind on c | K L104–151 `limited_face_x/y`: upwind MUSCL, r = d_upwind/d_face, van Leer φ, clipped to [min,max] of the two cells; forward Euler, same face flux G as ℓ (S L480–505) | yes. This is the **only** term that smears the visible field c = m/ℓ. |
| physical mixing D | `Params.D` default 1e-7, every fit keeps 1e-7 (`action_opt.run_one` L78 `D=1e-7`) | diffusion length √(2·1e-7·7 s) = 1.2e-3 D = 0.1 mm → no physical floor on stripe width |
| time step | S L778–780: dt = cfl·h/(2 u_max), cfl 0.5 → global material Courant ≤ 0.25; u_max is the **global** max | local Courant numbers at interfaces are far smaller (measured below) |
| first-order upwind on f, f_t (twolayer/v2) and m_sub | S L389–394, L374–379 | yes, ν_num = \|U_d\|h/2 |

So the mechanism exists in the code exactly as code_map §6 describes it. Nothing in any closure (chi / twolayer / v2) introduces a length scale for a stripe: D ≈ 0, no surface tension, no bubble size. TWOLAYER L84–88 states this openly ("细纹宽度由网格和数值扩散决定"), POUR L52–58 repeats it.

## 2. Numbers that are wrong or misleading in the claim

### 2.1 "ν_num ≈ |u|h 为物理 ν 的 4–15 倍" — a jet-footprint number, not an interface number

Measured in the 96² old-closure (#9, ν = 1.44e-3) push_heart run, on cells where |∇c|h > 0.05 (i.e. where the pattern edges are):

| t | u_max (global) | \|u\| at interfaces p10 / p50 / p90 [D/s] | ν_num = \|u\|h / ν | local Courant p50 / p90 |
|---|---|---|---|---|
| 0.50 s | 0.86 | 0.163 / 0.311 / 0.633 | 1.2 / **2.2** / 4.6 | 0.041 / 0.084 |
| 1.01 s | 1.04 | 0.133 / 0.191 / 0.588 | 1.0 / **1.4** / 4.3 | 0.025 / 0.078 |

The 4–15× range in the claim is obtained with u = 0.5–2 D/s, which is the jet-vicinity speed (max_u over the run 2.18). At the interfaces that carry the pattern, numerical and physical momentum viscosity are of the **same order** (1–5×). This is consistent with E-18 ([318], 128²): switching ν from 1.4e-3 to 0 changed the push_heart gap count 11 → 5 and the median width 3.1 → 3.8 mm — impossible if ν_num dominated ν everywhere that matters. Corrected statement: ν_num/ν ≈ 1–5× on the pattern edges, ≈ 7–13× inside the jet footprint.

Also: this viscosity acts on q, i.e. it sets the smallest velocity-shear scale, √(ν_num/β) ≈ 2–4 mm at 96² (vs √(ν/β) = 1.9 mm physical). It thins stripes only indirectly; the direct smearing of c is item 2.2.

### 2.2 "van Leer 抹 2–3 格" — it is 4–9 cells at the Courant numbers the solver actually runs at

1-D test with the repo kernel `K.limited_face_x` (square arrays, forward Euler, as in step C):

| local Courant | step 0.1→0.9 width after 30–300 cells of travel | stripe of w cells after 30 cells: peak c |
|---|---|---|
| 0.25 (the global cap) | 4 cells (steady) | w=2: 0.42, w=4: 0.79, w=6: 0.97 |
| 0.10 | 7–8 cells | — |
| 0.05 | 8 cells | w=2: 0.31, w=4: 0.61, w=6: 0.83, w=8: 0.95 |
| 0.02 | 9 cells | w=2: 0.30, w=4: 0.59, w=6: 0.81, w=8: 0.93 |

Because dt is set by the **global** u_max (S L769–778) and the measured local Courant at interfaces is 0.03–0.08, the relevant row is 0.02–0.05: a c-interface occupies **8–9 cells**, a stripe needs **≥ 4 cells to keep c > 0.5** (the whiteness threshold used by `gap_stats` and the cut-line counts) and **≥ 8 cells to keep c > 0.9**. In millimetres:

| N | h | ≥4 cells (contrast 0.5) | ≥8 cells (contrast 0.9) | 0.1–0.9 interface |
|---|---|---|---|---|
| 96 | 0.83 mm | 3.3 mm | 6.7 mm | 6.7–7.5 mm |
| 256 | 0.31 mm | 1.25 mm | 2.5 mm | 2.5–2.8 mm |
| 512 | 0.16 mm | 0.6 mm | 1.25 mm | 1.25–1.4 mm |

This is worse than the claim says and it matters for 96²: the minimum stripe/gap that shows with contrast 0.5 at 96² (3.3 mm) is inside the range of real gap widths (4–9 mm), so every 96² fit is grid-contaminated (that is A15/B21 territory; the claim's "96² 一格 0.83 mm" understates it — the senior's own 4-cell rule gives 3.3 mm at 96², not 0.83 mm; 0.83 mm is 4/384 D, i.e. 4 cells of **his** 384 grid).

In the 96² run the number of transitional cells (0.1 < c < 0.9) was 0.9–1.3× the number of interface cells and at t = 1 s there were 453 transitional cells against 270 white (c > 0.5) cells — the early pattern is mostly smear at 96².

### 2.3 E23 (256² → 512²) re-measured on the stored fields

| measure | 256² | 512² |
|---|---|---|
| loss / IoU vs target (from json) | 0.5547 / 0.354 | 0.5516 / 0.347 |
| `optimize.gap_stats` (thr 0.5, close 6 mm): gap_frac / mean width | 0.332 / 2.74 mm | 0.391 / 2.54 mm |
| gap components ≥ 1 mm²: count; width p10/p50/p90 (2·max EDT) | 10; 0.88 / 1.77 / 4.0 mm | 14; 1.03 / 1.40 / 4.14 mm |
| of which width < 1.5 mm / ≥ 3 mm / ≥ 4 mm | 4 / 4 / 1 | 8 / 4 / 2 |
| widest gap | 4.42 mm | 4.86 mm |
| five vertical cut lines, brown runs ≥ 1 mm: total / median / max | 11 / 2.81 / 6.56 mm | 16 / 2.50 / 7.50 mm |
| white-set (c > 0.5) IoU between the two grids (512 box-averaged to 256) | 0.862 | |
| L1 of c between the grids / after the objective's blur (σ = 1.5 % D) | 0.0176 / 0.0053 | |
| fraction of cells with 0.1 < c < 0.9, relative to white cells | 2.46 | 2.34 |

(The chat's "19 → 23, 2.5 → 2.0 mm" used a different line/threshold recipe; the direction is identical.)

Reading: refinement **doubles the number of sub-1.5 mm lines** (4 → 8) and lowers the median, while the number of gaps ≥ 3 mm is unchanged (4 → 4) and the widest gap moves by 0.4 mm. The white set agrees to IoU 0.86, the blurred L1 (what the loss sees) is 0.005 and the loss moves by 0.003. So:

- fine lines (< ~1.5 mm at 256²) are grid-set and not converged — the claim is right;
- everything the objective and the wide-gap question depend on **is** converged at 256²: the grid is not the reason the model has no 4–9 mm gaps, and 768² would not change that (as [257]/[258] themselves said);
- the low contrast of the gaps ("缝只是浅褐", [249]) does not change with grid (transitional fraction 2.46 → 2.34), so it is a property of the closure (thin sheared layers with intermediate c), not of numerical smearing.

### 2.4 The senior's 0/3/21

`notion_senior.md` L168: the 384/768/1536 dark-seam count is from the **Codex V0.5 HLLC code** (tracer + second-order limiter), not from `latte_imex`. Same class of numerics and the same absence of a physical width, so it is evidence by analogy, and the senior himself only concludes "不能宣称…收敛" and marks seams < 4/384 D unresolved. The 0/3/21 does not measure this solver; the 256/512 pair in 2.3 does, and it shows the same thing.

## 3. What is not a shortcoming of the numerics

- The solver is TVD/bounded on c (clip to neighbours, K L120–122; mass guard S L507–511; `clip_mass` ledger). The limiter is the standard one; the kernel is correct against the NumPy reference (`limited_face_value`, max diff 0.0).
- The time step is independent of c_p (pressure implicit) — the only Courant constraint is material, so refining dt alone would not sharpen c (smearing per cell travelled is *worse* at lower Courant, §2.2).
- Sub-grid stripes do not "fail"; they lose contrast monotonically (w = 2 cells → peak 0.3–0.4), which is what the thin-line counts in E23 show.

## 4. Corrected statement

细纹（256² 上 < 1.5 mm、96² 上 < 3.3 mm）的宽度和条数由网格决定、不收敛：c 的 van Leer 迎风在求解器实际的局部 Courant 数 0.03–0.08 下把界面抹成 8–9 格，一条纹需要 ≥ 4 格才能保持 c > 0.5（96² 3.3 mm、256² 1.25 mm）；D = 1e-7 和所有闭合都没有给纹一个物理宽度，TWOLAYER L84–88 自认。但：(i) "ν_num 为 ν 的 4–15 倍"是射流足迹内的数字，在图案边缘处 |u| 中位 0.2–0.3 D/s，ν_num/ν ≈ 1–5（E-18 ν→0 使缝数 11→5 也说明物理 ν 不可忽略）；(ii) 256²→512² 的白区 IoU 0.86、模糊 L1 0.005、≥3 mm 缝数 4→4、最宽缝 4.4→4.9 mm、损失 0.555→0.552——宏观图案和宽缝在 256² 已收敛，褐缝浅、无 4–9 mm 宽缝是闭合的性质而非网格；(iii) 0/3/21 是 Codex HLLC 代码的结果，对本求解器只是类比。因此 A14 的后果是：96² 上的一切拟合被网格污染（A15/B21），细纹指标不能跨网格比较；它**不**解释宽褐缝的缺失。建议把它从"major（模型失败原因）"改为"major（对 96² 拟合与细纹计分）/ minor（对宽缝问题）"。

## 5. Experiment that would settle the remaining gap

A 128/256/512 triple of the **same** action with the senior's protocol (conservative coarsening, counts only ≥ 4 cells of the coarsest grid) on push_heart for the old and the v2 closures: if the ≥ 3 mm gap count and the transitional fraction stay flat as they do here for the tulip, the "not converged" label applies to sub-resolution lines only. The 96² runs used for every fit should additionally be re-scored at 256² (`optimize.gap_stats` has no width threshold; `gap_diagnosis` counts ≥ 1 mm = 1.2 cells at 96², below the 4-cell contrast floor).
