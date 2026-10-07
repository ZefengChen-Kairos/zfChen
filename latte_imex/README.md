# latte_imex — V1 surface-layer latte-art model, IMEX + Rusanov

NumPy implementation of the V1 two-dimensional effective-surface-layer model
(state `l, qx, qy, m`; equations and inlet closure as in the Notion V1 page) with the
IMEX splitting described in "V1 表层模型：方程与 IMEX 分裂":

| step | terms | treatment |
|---|---|---|
| A | momentum advection `∇·(q⊗u)` | explicit, Rusanov flux, `α_f = 2 max(|u_n,L|, |u_n,R|)` |
| B | mass flux `∇·q`, pressure `∇p`, drag `−βq`, traction `Λ(v*−u)` | implicit; one SPD elliptic solve for `l^{n+1}` (matrix-free PCG, Jacobi) |
| C | milk flux `∇·(c q)` | same face flux `G_f` as the layer, upwind `c_f` with van Leer limiter |
| D | viscosity, mixing | implicit (per component), `l` frozen at `l^{n+1}` |

Sources `s, Λ, v*` are frozen inside a control frame (1/120 s); the frame is split into
sub-steps by the material CFL `Δt ≤ CFL·h/(2 max|u|)` only (no `c_p` restriction).

```
python -m latte_imex.tests                       # static stripe, conservation, timing
python -m latte_imex.run --pattern heart --N 256  # one pattern -> runs/heart_N256/{final.png,movie.mp4,metrics.json}
python -m latte_imex.peek tulip s.fwd=0.6 cp=0.3  # quick low-res look with script/param overrides
python -m latte_imex.cp_sweep                     # sub-step count vs c_p
```

Files: `solver.py` (grid, PCG, sub-step), `actions.py` (guessed pouring scripts for six patterns),
`run.py` / `peek.py` (drivers), `render.py` (diagnostic brown/white images and movies), `cp_sweep.py`.

## Controls

Three control sources are selectable with `--control`:

| control | what it is |
|---|---|
| `guess` | `actions.py`: first-round hand-guessed scripts (kept for the record; inlet scale was 5–10× off) |
| `v05` | `v05_program.py`: the V0.5 (Codex) phase lists ported verbatim — only used to check that IMEX reproduces HLLC |
| `zefeng` | `controls.py` + `optimize.py`: own phase structures whose parameter vectors are **optimised by CMA-ES** against targets extracted from the reference clips (`targets.py`) |

```
python -m latte_imex.targets                      # reference frames -> latte_imex/targets/<name>.npz (+ preview.png)
python -m latte_imex.optimize heart --N 96 --gens 60 --workers 2   # -> runs/opt/heart/{best.json,best.png,history.json}
python -m latte_imex.run --pattern heart --N 256 --control zefeng  # render the optimised programme with video
python -m latte_imex.compare runs/zefeng_256 results/zefeng_opt/compare.png
```

`targets.py` fits an ellipse to the convex hull of the brown crema region of one hand-picked end frame,
maps it affinely to the unit cup (weak-perspective view of a circle), rotates so the cut exit is at +y,
and converts saturation to a whiteness field `w ∈ [0,1]` by Otsu split.  The loss in `optimize.py` is
`mean|G_σ(s) − G_σ(w)| + 0.5(1 − IoU) + 0.5|ΔA|` with `s = clip(c/0.6, 0, 1)`, `σ = 1.5 %` of the diameter.
Physical parameters are the shared V0.5 set (`controls.V05_PHYSICS`) — still guesses, nothing fitted from video yet.

## Performance

The stencil operations run as parallel numba kernels (`kernels.py`, NumPy fallback when numba is
missing; first call compiles and caches, ~1 min). Viscosity is explicit whenever `nu*dt/h^2 < 0.2`
(the 2-D explicit limit is 0.25), which removes two PCG solves per sub-step at N ≤ 256; above that the
implicit solve is used automatically. BLAS is pinned to one thread (`__init__.py`): multi-threaded
BLAS thrashes on the small dot products and made N ≥ 128 runs 40× slower.

Heart with the V0.5 controls, 1 s of model time at N=256, 4 CPU cores:

| version | wall | note |
|---|---|---|
| NumPy, implicit viscosity (PCG 1e-10) | 45 s | 21 viscosity PCG iterations per step |
| explicit viscosity, fewer temporaries | 15 s | states differ by O(dt), ~1 % in m |
| + numba kernels, fused step B, numba PCG vector ops | 4.4 s | identical to the previous row to 1e-15 |

Set `NUMBA_NUM_THREADS=1` when running several simulations in parallel on one machine.

## CUDA back-end

`cuda_solver.py` is a numba.cuda port of the same scheme (same discrete formulas; results agree with the CPU
solver to ~1e-12). The whole state lives on the GPU; one thread per cell; the explicit momentum step, the
layer right-hand side, the final flux + conservative update + milk transport are each one fused kernel, the
PCG runs matrix-free on the device with block-reduced dot products.

```
bash scripts/autodl_setup.sh                                   # installs numba-cuda[cu12], runs the CPU-vs-CUDA smoke test
python -m latte_imex.run --pattern heart --N 768 --control v05 --backend cuda
python -m latte_imex.bench_cuda --N 768 --video                # six patterns, wall times, finalize overview
NUMBA_ENABLE_CUDASIM=1 python -m latte_imex.test_cuda --N 24 --T 0.05   # no GPU: simulator (slow)
```

Differences from the CPU solver: viscosity and mixing are always explicit (sub-cycled when `nu*dt/h^2` or
`D*dt/h^2` exceeds 0.2), the deposited-milk ledger uses the exact kernel normalisation (`chi*S*dt`).
Per sub-step there are ~10 kernel launches plus 2 host reads per PCG iteration; at N=768 this is launch-bound
(~0.5 ms per sub-step), so a 10 000-step pattern is expected to take 5–10 s on an RTX 4090 in FP64.
