# Skeptic check of CLAIM A6 (model A whiteness = mixing ratio, hard saturation, "thickness 1" = 60 mm)

Verdict: **confirmed with nuance** — the mechanism is real and code-traceable, but two of the cited numbers are wrong
(file/line of the 0.6, and the ">70 %"), and the "renderer gets no physical quantity" part is partly a misreading of the
senior's render spec, which *asks* for exactly the quantity model A delivers.

## 1. What the code does (traced)

### 1.1 Whiteness of the old ("chi") closure is the mixing ratio c = m/ℓ
`solver.py` L828–836 `concentration()`:
```
if closure == "v2":   return 1 - exp(-max(m,0)/m_o)
if closure == "skin": return clip(m/m_opaque, 0, 1)
return m / l                       # chi closure (model A) and twolayer
```
So for model A the "visible" field is literally m/ℓ. Confirmed.

### 1.2 Where the 0.6 saturation really lives
- `render.py` L14: `def c_to_rgb(c, mask, c_sat=0.8)`, L16: `v = clip(c / c_sat, 0, 1)`, L17: `v = v ** 0.8`. The movie overlay
  (L63) even says "display saturates at c = 0.8". **render.py saturates at 0.8 (with gamma 0.8), not 0.6.**
- `optimize.py` L34: `def whiteness_sim(c, mask, c_sat=0.6)`; L39–44 `visible()`: v2 → c itself (already an opacity),
  every other closure → `whiteness_sim` (c/0.6 clipped). `visible` is what every fit/score uses: `action_opt.run_one` L87,
  `optimize` L87/L185, `compare.py`, `joint_sweep.py` L115, `gap_diagnosis.py` L41. The IoU/area terms of `loss_fields`
  (L47–53) threshold at s > 0.5 ⇔ **c > 0.3**; `gap_stats` thr = 0.5 ⇔ c > 0.3; the run metrics use c > 0.4
  (`run.py` L99, `cp_sweep`, `finalize`, `tests`).
- So the evidence line "render.py:16 c_sat 0.6" (copied from docs_map O1) is wrong in file; the correct citation is
  `optimize.py:34` (loss) and `render.py:14` (diagnostic, 0.8). The *mechanism* (hard clip, white at c ≥ 0.6, "not white"
  at c < 0.3) is correct for everything that was optimised.

### 1.3 "Thickness 1" = D_L/kQ ≈ 60 mm — confirmed numerically
- `action_opt.py` L72–74: `ph, kQ = physics9()` (kQ popped from `runs/joint_sweep_r2fix/results.json[9]`), then
  `Coupling(c_S = kQ / D_L**3, ...)`; `pitcher.Coupling.inlet` L554: `S = c_S * Q` ⇒ S_eff = kQ·Q/D_L³.
- results.json[9]: kQ = 1.32600…, D_L = 0.08 ⇒ one unit of ℓ ⇔ D_L/kQ = **60.33 mm** (my run printed it).
- `Params.kappa_Q = 125` is passed (`pd = dict(ph, D=1e-7, kappa_Q=125.0, ...)` L76) but the solver never reads it;
  only `Coupling.kappa_Q` for `footprint="v05"` (pitcher L565). Confirmed dead.
- Whether that 60 mm "is the coffee depth" is an *interpretation* (LATTE_MODEL §0 修正 2: "这个 1 ≈ D_L/k_Q ≈ 6 cm，
  是咖啡深度"); the senior's spec (notion_senior §2.1) says ℓ is "不是实际液面高度", i.e. the old model never claimed ℓ
  is a thickness. But the physical reading is forced by the mass budget, see 1.4.

### 1.4 Short run (old closure, push_heart video-scaled V0.5 moves, N = 64, 8 s) — `skeptic_A6_run.py`
```
poured 57.8 mL;  ledger deposited 0.0881 D_L² (= 0.0881·D_L³/kQ = 34 mL deposited, rest is χ<1 traction only)
final ℓ in cup: 1.1164 … 1.1173     ← flat to 1e-3: pressure (c_p 0.97) has relaxed ℓ completely
white (c>0.3): ℓ 1.12, m med 0.54 max 0.84, c med 0.49 max 0.75
sat   (c≥0.6): m med 0.71,  c med 0.64 max 0.75
brown (c<0.1): m ≤ 0.11
fraction of local m that must leave (ℓ fixed) to reach c<0.3: median 0.53, max 0.60
```
Cached fit result `runs/action_opt/ph3_old/push_heart_h1.000_q1.000_s1.000_N96.npy`: c max 0.765, c median inside c>0.3
= 0.61, only 10 % of the cup has c > 0.6. conv_part3 E-18 ([549]) agrees: "old c 0.6–0.65 in white, m in the gap
0.05–0.09, in blocks 0.7".

Consequences:
- At rest ℓ is uniform, so the final pattern is **m alone** (deposited milk column per area) and whiteness is a hard
  ramp in m: white at m ≳ 0.67 units (≈ 40 mm-equivalent of milk, 60 mm × 0.67), "not white" at m < 0.34 (≈ 20 mm-eq.).
  34 mL deposited over the white area (0.18 × 0.754 × D_L² = 8.7 cm²) is indeed 39 mm thick: the model has no bulk
  (A8), so the whole poured volume stays in the surface layer. That is the quantitative core of "白度与光学无关".
- The blob is only *marginally* white (c_max 0.75 vs saturation 0.6; c cannot reach 1 because deposit c=1 mixes with the
  ℓ=1 brown base). To open a brown line (c < 0.3) you must remove **≈ 50–60 % of the local milk m**, not ">70 %" (70 % would
  need c = 1). And "膜质量" in the claim should read 奶质量 m, not film mass ℓ.
- Step C (`solver.py` L476–511) transports m with the **same face flux G_f** as ℓ and an upwind, van-Leer-limited c_f;
  the only non-advective changes of c are mixing D = 1e-7 and the clip m ≤ ℓ. So **c is a passive tracer: stretching or
  thinning a white film never changes its whiteness**; brown appears at a point only if c≈0 base material is advected in
  (or the white material is advected out and replaced). This is the precise sense in which "露褐是搬运问题而不是薄层问题":
  in the optical picture (physics_prior §1.5, R ≈ L/(L+0.2–0.35 mm), saturation ≈ 1 mm) a raft thinned to < 0.3 mm turns
  brown; in model A thinning is invisible and is anyway refilled by pressure on ~0.1–0.3 s.

## 2. What the claim gets wrong or overstates
1. **File/line**: 0.6 is `optimize.py:34` (`whiteness_sim`), used through `visible()`; `render.py:14/16` uses 0.8 with
   gamma 0.8. docs_map O1 carries the same error.
2. **">70 % of the film mass"**: measured 53–60 % of the *milk* m (c 0.64–0.75 → 0.3), ℓ unchanged. Also, "mass removal at
   fixed ℓ" is not something the dynamics can do — c only changes by replacing material (passive tracer); the correct
   statement is "brown requires importing c≈0 base material / exporting the white column, since thinning does not change c".
3. **"渲染器拿不到物理量" + citing notion_senior §5 as support**: the senior's render spec (§5, V1 §4) demands exactly a
   bounded component c that linearly mixes materials, says ℓ "不是自由液面高度", and *forbids* a thickness–alpha mapping
   ("不加入厚度—alpha 映射…避免按有效层厚加权白度"). Model A's c = m/ℓ ∈ [0,1] is the quantity the spec asks for; it is
   v2's W = 1 − exp(−m/m_o) that conflicts with the spec (notion_senior §5 last paragraph). So, relative to the senior's
   spec, model A's *interface* is correct; what is missing is that c has no optical/thickness meaning (a 60 mm-column
   mixing ratio, not foam coverage) and no dynamics make it behave like a coverage (no thinning → brown, no "bare canvas").
   The claim's conclusion ("the renderer gets no physical quantity") is right in substance, but §5 is evidence against the
   claimed *requirement*, not for it.
4. "与光学无关" is true for the *scale* (40 mm-equiv. vs 1 mm) and for the *mechanism* (ratio vs thickness), but the
   *shape* of the mapping (fast saturation, hard edge) is the part conv_part3 [537] pt. 3 credits for the old model's sharp
   edges; S-A6 in diagnosis.md already notes this.

## 3. Corrected statement
Model A's visible field is the mixing ratio c = m/ℓ (`solver.concentration()` L836); all fits score clip(c/0.6)
(`optimize.whiteness_sim` L34 via `visible`; `render.py` L14 shows clip(c/0.8)^0.8 for pictures). The unit layer is
D_L/kQ = 60.3 mm (kQ = 1.326 fitted, `action_opt` L74), and because the closure has no bulk, the whole deposited volume
(34 mL in push_heart) sits in the surface layer; pressure flattens ℓ to 1.117, so the final pattern is m alone with
"white" ⇔ m ≳ 0.67 units ≈ 40 mm-equivalent milk and "not white" (c < 0.3) ⇔ m ≲ 0.34 ≈ 20 mm-equivalent — unrelated to
the ≈ 1 mm optical saturation / < 0.2–0.3 mm brown threshold. Opening a brown line needs ≈ 50–60 % of the local milk
replaced by c ≈ 0 base material; since c is a passive tracer (same flux G_f for ℓ and m, step C), thinning a film never
exposes brown. ℓ and m are therefore neither a film thickness nor a coverage. The renderer *does* receive the bounded
component c the senior's spec (§5) asks for — the defect is that this c carries no optical/thickness physics, not that the
interface is missing.

## 4. No experiment needed
The mechanism is fully traceable in code; the numbers above come from an 8 s run (`skeptic_A6_run.py`) and a cached fit.
