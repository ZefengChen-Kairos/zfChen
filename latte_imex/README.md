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
