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

`cuda_solver.py` is a numba.cuda port of the same scheme (same discrete formulas; agrees with the CPU solver to
rounding, ~1e-12 in l and m, ~1e-9 in q where the viscous PCG tolerance 1e-8 sets the level). The state lives on the
GPU as `[B, N, N]` arrays: `CudaSolver` is the single-run drop-in (B = 1), `BatchCudaSolver` advances B independent
simulations in one launch set (each member: own inlet, frame length, sub-steps, retries, ledger; a member reproduces
its B = 1 run). Per sub-step 5 fused kernels + 2 per PCG iteration, PCG scalars and the convergence test on the
device, ~2 host polls per sub-step; inlet kernel, off-cup fraction Z and CFL maxima are computed on the device from
the inlet scalars. Details and numbers: `CUDA_PERF.md`. The GPU forward pitcher is `cuda_pitcher.py`
(batched `ForwardPitcher.run`, same results to rounding).

```
bash scripts/autodl_setup.sh                                   # install + GPU self-tests + short benchmark
python -m latte_imex.run --pattern heart --N 768 --control v05 --backend cuda
python -m latte_imex.bench_cuda --quick --N 256 --B 1,8,32      # launches / syncs per sub-step, batched throughput
python -m latte_imex.bench_cuda --N 768 --video                # six patterns (V0.5 controls), wall times
NUMBA_ENABLE_CUDASIM=1 python -m latte_imex.test_cuda --N 20 --T 0.02 --batchN 16 --frames 2   # no GPU: simulator
```

Differences from the CPU solver (unchanged from before): componentwise viscosity and mixing are explicit (sub-cycled
when `nu*dt/h^2` or `D*dt/h^2` exceeds 0.2), the deposited-milk ledger uses the exact kernel normalisation
(`chi*S*dt`), q^{n+1} uses the gradient of the PCG solution (CPU: of the conservative l^{n+1}; they differ by the
PCG tolerance), and the retry guard is evaluated once per frame (a failing sub-step fails the frame in both).

## AutoDL 使用说明（GPU：对齐模型 + 前向壶 + 手误差稳健性）

**要求**：一块 CUDA 12 的 NVIDIA GPU（RTX 4090 / A100 一类；全程 FP64，A100 的 FP64 快得多，4090 也能跑），
驱动 ≥ 525，Python 3.9–3.12，显存 ≤ 2 GB（(b) 60 个成员一批 256² 约 1.3 GB；(a) 512² 六个一批约 0.6 GB）。不需要本仓库以外的数据：S1 理想手的姿态和设计程序
在 `latte_imex/data/s1_forward/`（1.6 MB，`s1_forward.export` 从复核目录导出，导出时已逐位核对与
`v05_forward.forward_program` 给出的入口记录相同）。

**一行命令**（在仓库根目录）：

```
bash scripts/autodl_setup.sh                 # 装环境；真 GPU 上自检：CPU 与 GPU 求解器对比、批量 = 单跑、GPU 壶与 CPU 壶对比；短基准
bash scripts/autodl_run_v05_aligned.sh       # (a) 六图案 256²/512² 终图+视频+指标 (b) 手误差网格 256² (c) 汇总图和 results.json
```

可用环境变量改：`OUT=results/xxx`（输出目录）、`NS=256,512`（(a) 的分辨率）、`NR=256`（(b) 的分辨率）、
`CHUNK=30`（(b) 每批成员数，显存不够时用；0 = 60 个一批）。分步跑：

```
python -m latte_imex.autodl_v05 aligned --N 256,512 --video
python -m latte_imex.autodl_v05 robust --N 256
python -m latte_imex.autodl_v05 figures
```

**做了什么**
- 模型：`Params` 默认值，即对称黏性（耦合隐式，GPU 上无矩阵 CG）+ 杯外入流记账（`offcup="account"`）。
- 入口：前向壶 `ForwardPitcher`（V1 模式 B）在 GPU 上批量回放 S1 理想手姿态 → 到达记录 → 设计程序的入口规则
  （`s1_forward.FwdProgram`，与 `v05_pourable.pl_program` 的 'pitcher' 路径逐位一致）。
- 手误差（`v05_pourable/data/robust_prereg.json` 的 P1–P6、R1–R3，逐出流笔）：高度 ±3 mm 加在壶嘴高度上；
  流量 ×0.8/×1.2 用每笔一个倾角偏置实现，偏置由前向壶批量割线迭代标定到该笔倒出量 = f × 无流量误差时的量（容差 0.3%）；
  ±50 ms 把倾角时间历程相对路径平移。路径、偏航、侧倾不变。与 §3.3 的不同：那里扰动的是代理模型的设计入口并重算壶表，
  这里扰动的是姿态、入口来自前向壶；壶嘴不低于壶下限的截断这里没有做，只在 `metrics.json → extra.tip_z_min_mm` 里记录。
- 稳健性图的标签是与名义运行的 c > 0.5 白区 IoU 和白区面积比，是指标不是判定（判定仍要看图）。

**输出**（默认 `results/autodl_v05/`）
- `aligned/N256|N512/<图案>/`：`final.png`、`movie.mp4`（V0.5 样式的视频，`render.make_video`）、`c.npz`、`final_state.npz`、
  `snapshots_c.npz`、`metrics.json`（账本、统计、壶的数）、`trajectory.json`
- `robust/N256/<扰动>/<图案>/`：`final.png`、`c.npz`、`metrics.json`（`extra` 里有每笔的倾角偏置、倒出量比、最低壶嘴高度）
- `pitcher/<图案>_<扰动>.npz`：到达记录和所用的姿态扰动
- `fig_aligned_six.png`、`fig_robust_grid.png`、`results.json`；日志 `log_*.txt`；自检日志 `results/autodl_selftest/`

**预计用时**（未在真机上测过，按每个子步的启动数和带宽估算；以 `bench_cuda --quick` 的实测为准）
- `autodl_setup.sh`：安装 2–5 min；自检 3–6 min（其中 CPU 参考的 256² 心形 1 s 和 CPU 壶 9 段各 1500 步占大头）；基准 1–2 min。
- (a) 六图案：壶 < 1 min；256² 一批 6 个约 2–5 min，512² 约 5–15 min；视频编码（CPU）每个 0.5–1 min。
- (b) 60 次运行（6 图案 × 10）：壶标定 1–3 min；256² 一批 60 个约 10–25 min。
- (c) 几秒。

## Pitcher model (level 0)

`pitcher.py` closes the chain upstream of the inlet record: pitcher pose (tip position, yaw, tilt) and
remaining volume -> horizontal free surface (weighted quantile over interior sample points) -> sharp-crested
weir outflow over the rim (`Q = C_d (2/3) sqrt(2g) ∮ h^{3/2} dl`, Kindsvater head correction, start/stop
hysteresis, spout lag) -> free jet (Torricelli exit speed, ballistic flight, continuity thinning) -> the
six impact quantities in SI -> `Coupling` -> `solver.Inlet` in cup units.  Derivation and defaults:
`PITCHER_MODEL.md`.  A `Barista` turns intent (stream path, tip height, wanted flow) into poses with a
rate-limited wrist controller, so "pour longer -> tilt more" appears by itself.

```
python -m latte_imex.test_pitcher                               # capacity, V-notch law, bookkeeping, ballistics, inversion
python -m latte_imex.run --pattern heart --control pitcher --N 256 --out runs/pitcher_heart_256
python -m latte_imex.pitcher_demo --out results/pitcher_demo    # inputs.png, side_view.png, tilt_flow.png, records.csv
```

`web/pitcher_playground.html` is the same pitcher model ported to JavaScript with a three.js scene (drag the
spout, hold to pour, replay the heart script), feeding `web/latte2d.js`, a port of the 2-D solver (same sub-steps;
explicit viscosity/mixing only).  The coffee surface shows its milk fraction and a double-click on the cup opens
the 2-D field.  Both ports were checked against Python: same Q within 0.3 % at 60–70° tilt, and the heart at
N=96 agrees to 3e-8 in c frame by frame (Node: about 2x real time at 96², 1x at 128²).

Level 1 (`OutflowLaw.slosh`, on by default): the free surface follows the effective gravity `g - a` of the
moving pitcher through its first sloshing mode (damped oscillator, `zeta` 0.08), the jet carries the pitcher
velocity, and the barista reacts to low-passed flow and aim.  `pitcher_cad.py` turns a STEP pitcher into the
(z, phi) cavity grid used by `GridPitcherGeometry` (`--pitcher nx`) and by the web page (`web/pitcher_nx.json`).
See `PITCHER_MODEL.md` sections 9 and 10.  Conservation ledger (bulk, lip buffer, in-flight parcels with
arrival windows, cup; `Q_feed`, `Q_out`, `Q_hit`) and the lip-flux second-moment footprint: section 12;
`PITCHER_MODEL_V12_PLAN.md` is the reviewed V1.2 plan (its F1 and flux-moment items are the ones implemented).
The web page runs the 2-D solver in a Web Worker (grids 64²–768², the model clock waits for the solver).
