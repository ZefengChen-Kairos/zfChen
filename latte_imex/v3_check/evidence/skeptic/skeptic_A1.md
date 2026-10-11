# Skeptic notes on claim A1 (old `chi` closure): "c = m/ℓ is a passive tracer; stretching never yields brown; the brown gap can only be the compressed original brown strip and can never be wider than the strip at the start point"

Files read (nothing modified): `/home/user/zfChen/latte_imex/solver.py` (HEAD, 843 lines), `kernels.py` L107–150, `action_opt.py` L1–100,
`stage_actions.py`, `gap_diagnosis.py`, `optimize.py` `gap_stats`, `PITCHER_MODEL.md` §14–15 (L236–272), review notes
`code_map.md` §4, `conv_part2.md` §8–9 (E-11…E-24), `notion_z1.md` F10/F11, `video_obs.md` §2, §7, §9, `diagnosis.md` A1–A4, A10.
Numerical check: `scratchpad/skeptic/a1_test2.py` (N = 96 and 128, chi closure with the #9 constants from
`runs/joint_sweep_r2fix/results.json[9]`, each run 1–3 s).

## 1. What the code does (the kinematic part of A1 is correct)

* `solver.py` L360 `c = m/lsafe`; step C L476–478 + L511 `m_new = m + dt*s_m − dt*div(G·c_f)` with the **same** face flux `G`
  (`Gx, Gy` from `finish_B`, L461–466) that updates ℓ (L467). `kernels.limited_face_x/y` (L107–150) clamp the face value to
  `[min(cL,cR), max(cL,cR)]`, so the update is a convex combination: c obeys a discrete maximum principle.
* Sources: old closure `s_l = s` (L457–458) and `s_m = s` (L508): the deposit raises ℓ and m by the same amount ⇒ new material
  arrives with c = 1. Λ traction adds momentum only. Hence **brown mass B = ∫(ℓ − m) dA is exactly conserved** (ledger
  `brown = L − M`, L820; in my runs 0.6695 D² to 4 digits for all 1.2 s). D = 1e-7 (`action_opt.py` L76) is inert.
* Consequence: a parcel's c never changes; thinning a white parcel (ℓ↓) leaves it white; no brown can appear inside a connected
  white region. **This part of A1 is confirmed from the code.**

## 2. What the claim over-states

### 2.1 "宽度不可能超过起笔处的褐条" is not a theorem, only an observation
Brown *area* = brown mass / local ℓ. Brown fluid at ℓ < 1 occupies more area than at ℓ = 1, and brown from the sides can be
funnelled into the gap. The code does produce ℓ < 1 regions: in the delay-patch run below ℓ_min = 0.38–0.47 in the wake behind
the hit; without the patch ℓ_min = 0.82–0.93. So widening by stretching of *brown* is allowed by the model. In none of my runs
did the gap exceed d0, so empirically the bound holds — but it holds for a different reason (below).

### 2.2 The gap is not mainly "compressed"; it is overrun and squeezed out sideways
Test: old white disk (r = 0.165 D, c = 1) on the surface; #9 low pour (Q = 23 mL/s, S_eff = 0.0596, U⊥ = 9.27 D/s, d = 0.079 D,
footprint 0.049 × 0.031 D, u_in = 1.88 D/s forward) starts d0 = 0.15 D = 12 mm behind the disk's rear edge (the real
push-heart geometry: video_obs L63 "new white ring appears at B ~0.15–0.2 D from the old blob") and moves forward at 0.09 D/s.
Centre-line result (N = 96; N = 128 identical to 0.5 mm):

| t (s) | gap (mm) | ℓ in gap | old block rear edge (D) | new white front (D) |
|---|---|---|---|---|
| 0.2 | 5.0 | 1.18–1.20 | −0.016 | +0.057 |
| 0.4 | 2.5 | 1.13–1.14 | −0.057 | −0.016 |
| 0.6 | 1.7 | 1.12 | −0.099 | −0.068 |
| 0.8 | merged | — | — | — |

The brown strip is compressed by only 12–20 % (ℓ 1.12–1.2) yet shrinks from 12 mm to 1.7 mm in 0.6 s: on the centre line
12 mm × 1.0 of brown became 1.7 mm × 1.12 — the brown was displaced **laterally** around the expanding new disc. Meanwhile the
old block **is** pushed: its rear edge moves 0.11 D in 0.6 s (0.18 D/s; real: 0.05–0.15 D during the first ~1 s, video_obs
§7). The gap closes because the *new white front* advances at 0.36 D/s — the deposited white (c = 1 at once, s_peak ≈ 24/s)
is slung forward by s·v* + Λ(v* − u) with v* = κ_t u_in = 1.7 D/s. With Q × 0.3 the collapse is slower (7.5 → 5.0 → 3.3 →
2.5 → 1.7 → 0.8 mm over 1.2 s) but monotonic. So the limiting mechanism is "white deposited at the hit, immediately visible
and faster than the block it pushes" (A2/A3/A4), not the tracer property.

### 2.3 The real gap is itself a swept original brown strip narrower than its initial width
video_obs L63/L161: the new pour starts 0.15–0.2 D from the old blob, the brown between is "initially ~0.15 D, final
0.04–0.13 D"; §7 item 3: the push is "a smooth, nearly area-preserving deformation" (0.33 D disc → 0.10 D × 0.5 D crescent,
area ≈ conserved). A passive-tracer surface with the right push kinematics therefore reproduces the observed gap widths; the
bound "gap ≤ initial brown strip" is satisfied by the real video too. A1 cannot be what makes 4–9 mm gaps unreachable.

### 2.4 Direct counter-example inside the same tracer: the §14 delay/mound patch
`PITCHER_MODEL.md` L238–245 (table verified): with step C unchanged (c still m/ℓ), round 2 gives push_heart 11 gaps /
median 6.2 mm / max 10.6 mm, round 3 6 / 7.2 mm, round 5 (shipped working point) 2 / 7.2 mm. `conv_part2` E-17/E-20 agree.
So the passive tracer does not forbid ≥ 4 mm bands; what changed was the *ordering* (push before whitening) and a sub-surface
pressure. The diagnosis lists this under A10 but omits it from A1's evidence.

## 3. The evidence given for A1 does not test A1
* 160-sample sweep, footprint × 0.3, 85 % pulsation, ν → 0, 256² → 512² (E-11, E-12, E-18; notion_z1 F10/F11): all are
  runs of the no-delay chi closure on V0.5-translated actions. They show that *this closure with these actions* gives
  1–3 mm stripes; none varies the tracer property or the push/whiten ordering, so they cannot attribute the failure to c = m/ℓ.
* "均无 ≥ 4 mm 褐带" is loosely stated: E-12 reports widest gap 9.1 → 9.5 mm (stem–top-heart), E-18 max 5.0 / 9.4 mm; those
  are pattern-exterior brown, not inter-petal bands, which is consistent with §1 but shows the threshold statement needs the
  qualifier "between stacked elements".
* The V0.5 push_heart script overlaps its three segments (diagnosis A18; `stage_actions.py` docstring), so in those runs the
  "initial brown strip at the start point" is ≈ 0, and the 1–3 mm stripes that *do* appear are brown folded in from the sides —
  the model's gaps there already come from somewhere other than "the strip at the start point".

## 4. Verdict
Kinematics confirmed (c is a Lagrangian invariant, brown mass conserved, no brown by thinning; D inert). The diagnostic
inference — that this is why wide gaps are unreachable and that it is fatal — is refuted: the real gap is also a conserved
brown strip narrower than its start width; the same tracer produced 6–10 mm bands once the push preceded whitening; and in the
model the old block is pushed as far as in the video but the immediately-white deposit outruns it and squeezes the strip out
sideways. Whether a thickness-based whiteness (exposure by thinning) is *needed* for other features (crisp 0.01 D interfaces,
thin stripe lines, the brown impact hole) is a separate question not settled by A1's evidence.
