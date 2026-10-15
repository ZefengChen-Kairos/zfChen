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

Design / calibration on the GPU: `gpu_calib.py` (closure calibration against the videos, the `v05_calibrate` setup with
every CMA-ES generation as one batch; pitcher-constant fit hook), `gpu_design.py` (per-pattern action re-design with the
forward pitcher in the loop, CMA-ES, 128^2 screen / 256^2 confirm, resumable), both on `gpu_batch.BatchEngine`; one-command
script `scripts/autodl_run_design.sh`, tests `test_gpu_design` (Chinese section below).

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

## AutoDL 标定与动作优化（GPU：闭合标定 → 前向壶上重新设计动作 → 手误差稳健性 → 终图/视频）

**为什么要做**：对齐到 V1 之后（对称黏性、出射时刻入口速度 + 杯外记账、前向姿态驱动壶 + 投影堰宽），按旧壶模型设计的 S1
动作在前向壶上多倒 4–6 mL；大白心 / 推推乐几乎不变（IoU 0.95 / 0.87），千层心、郁金香、树叶、天鹅白区多 20–35%、线变粗。
所以动作要在前向壶上重新设计，闭合参数也可能要重新标定。顺序固定为：**① 闭合标定 → ② 用标定后的闭合重新设计动作 →
③ 手误差稳健性检查 → ④ 终图和视频**。

**一行命令**（在仓库根目录，先跑过 `bash scripts/autodl_setup.sh`；中断后原命令重跑即可续算，每一代都存盘）：

```
bash scripts/autodl_run_design.sh
```

环境变量（括号内为默认值）：`OUT`（results/autodl_design）、`PATTERNS`（六个）、`STEPS`（selftest,calib,predict,design,robust,final，可只跑其中几步；selftest = `test_gpu_design` 在真 GPU 上，2–4 min）、
标定 `CAL_GENS`（20）`CAL_POP`（12）`CAL_N`（96）`CAL_INITS`（0,1,2）`PROCS`（4，CPU 上算 calib 指标的进程数）、
设计 `GENS`（20）`POP`（12）`N_LO`（128，筛选分辨率）`N_HI`（256，复核分辨率）`TOPK`（2）`CONFIRM_EVERY`（2）
`ROB`（R1,R2,R3；设为空串则不算稳健项，约快一倍）`VARS`（tilt,z,shift,fill）`CHUNK`（0 = 一代一个 GPU 批；显存不够时设 96 之类）、
`CLOSURE`（calib = 用 ① 的 best.json；s1 = S1 程序自带闭合；或一个 JSON 路径）、`PITCHER_LAW`（壶常数拟合结果，可选）、
`NS`（256,512）`VIDEO`（1）、`T`（只用于冒烟测试：每个运行截断在 T 秒）、`BACKENDS=cpu`（CPU 参考实现，很慢）。

分步命令：

```
python -m latte_imex.gpu_calib run     --out results/autodl_design/calib            # ① 标定（可续算）
python -m latte_imex.gpu_calib predict --out results/autodl_design/calib            #    冻结参数预测其余四个
python -m latte_imex.gpu_design run    --out results/autodl_design/design --closure results/autodl_design/calib/best.json   # ②
python -m latte_imex.gpu_design status --out results/autodl_design/design
python -m latte_imex.gpu_design robust --out results/autodl_design/design --N 256    # ③
python -m latte_imex.gpu_design final  --out results/autodl_design/design --N 256,512 --video   # ④
python -m latte_imex.gpu_calib fit_pitcher --scale a.npz,b.npz --out results/pitcher_fit     # 有秤数据时：拟合壶常数
NUMBA_ENABLE_CUDASIM=1 python -m latte_imex.test_gpu_design --quick                  # 无 GPU 的自检（模拟器 + CPU，约 30 min）
python -m latte_imex.test_gpu_design                                               # GPU 上的自检（较大网格）
```

**① 闭合标定**（`gpu_calib.py`，就是 `v05_calibrate` 的预注册设置，只是把前向计算批量放到 GPU 上）：模型 = `V05Model`（咖啡师的
真实动作，物理入口），求解器用 `Params` 默认值即对齐后的模型；参数 = `v05_adapter.SPACE` 的 8 个闭合参数（B_dep、p_dep、κ_r、
κ_Q、c_p、β、κ_t、κ_c，物理边界）+ 2 个输入扰动参数（u_exit、Q_scale）；损失 = `calib.loss_of(delay=False)`（⟨1−Dice⟩ + ⟨边界距离⟩/0.05 D
+ 阶段末细线数误差），郁金香 + 树叶等权平均；CMA-ES（N 96、种群 12、σ0 0.25、20 代、3 个起点 FSp2 / FS / 盒中心，种子 1+起点号），
3 个起点同步推进，一代 3×12×2 = 72 次前向是一个 GPU 批，calib 指标在 CPU 进程池里算。最优 = 所有 CMA 样本里平均损失最低者
（与 `calib.calibrate` 相同），写 `best.json`（给 ② 用）。`predict`：最优参数和 FSp2 起点在六个视频上 128² 一批跑，
分开报告“标定拟合”（郁金香、树叶）和“测试预测”（大白心、推推乐、千层心、天鹅），外加过程对照图和 256² 末帧对照。

**② 动作重新设计**（`gpu_design.py`，在前向壶上闭环）：
- 决策变量（逐出流笔；路径、偏航、侧倾、时间轴保持 S1 不变，与 S1 预注册“不许改路径/时间/摆动”一致）：倾角偏置 tilt_k
  ∈ [−6, 6]°（前向壶里流量由倾角决定，对应 S1 设计的流量系数 s_k；`s1_forward.tilt_offset` 的剖面）、壶嘴高度偏置 z_k ∈ [−4, 8] mm
  （对应 S1 的设计高度 z_k，0.11 m/s 高度过渡）、出流时间平移 shift_k ∈ [−80, 80] ms（倾角历程相对路径平移，同 P5/P6 的机制）；
  每个图案一个起始装奶量 fill ∈ [100, 125] mL。
- 目标 J = 10·J_S1 + 1·L_shape + 0.5·R + 1·P（权重可改，`--w-*`）：J_S1 = `v05_pourable.finalize` 在实际入口上的 S1 设计目标
  （逐笔 χ 对 V0.5 目标、相对 V0.5 笔量的流量系数、水平动量对 FS）；L_shape = 2 − IoU(c>0.4) − r_hp，对 V0.5 参考图
  （`allphys.rank` 的筛选分数）；R = 手误差抽样（默认预注册的 R1–R3，每笔同时有高度 ±3 mm、流量 ×0.8–1.2、±50 ms）相对本候选
  名义运行的 (1 − IoU(c>0.5)) + |ln 白区面积比|（`autodl_v05.compare_c`），扰动用 autodl_v05 的同一套（姿态上加，流量误差用 GPU 壶
  批量割线标定倾角偏置）；P = 约束罚：逐笔流量加权倾角在 63–81°、流量加权壶嘴高度在 S1 的范围（low 3–25 mm、cut 12–70 mm）、
  倒奶时壶嘴 ≥ 3 mm、壶里最后 ≥ 20 mL、理想手的速率/加速度上限（只罚超过 max(上限, S1 原姿态值) 的部分）。关键结构
  （`design_metrics.STRUCT`，即 ref2/structm 的心形缺口、推纹、千层、郁金香轴线、叶脉、天鹅颈）只报告，不进目标。
- 优化器：CMA-ES（单位盒，起点 = S1 原动作，σ0 0.2，种群 12，20 代），所有图案同步；一代 = 所有候选 ×（1 + 扰动数）条姿态
  一次 GPU 壶批量 + 一次 `BatchCudaSolver` 批（默认 6×12×4 = 288 个成员，128²）。多保真：每 2 代把每个图案筛选最好的 2 个
  在 256² 复核（壶的结果直接复用），最后一代再复核 CMA 均值；报告的设计 = 256² 复核最好的。每代存 `state.pkl`（含随机数状态，
  续算与不中断结果逐位相同，已测）。S1 原动作作为参照在两种分辨率各算一次。

**壶常数拟合钩子**（暂无数据）：`gpu_calib fit_pitcher` 用秤测的质量曲线拟合 `ForwardLaw` 的 C_d、h_k（可加 C_v，但秤上几乎
不可辨识）。输入每次倒奶一个 `.npz`：`t`（姿态时间，等步长）、`tip`（n×3 壶嘴尖位置 m，杯坐标、液面 z = 0）、`yaw`、`tilt`、`roll`
（rad）、`fill_mL`、`alpha_deg`、`m_t`（秤读数时间 s，同一时钟）、`m_g`（读数 g）、`scale_on`（'cup' 杯在秤上 = 到杯量，或
'pitcher' 壶在秤上）、可选 `rho`（1.03 g/mL）、`tau_scale`（秤的一阶滞后 s）。输出 `pitcher_law.json`，`gpu_design --pitcher-law`
（或 `PITCHER_LAW=`）直接读。`gpu_calib make_scale` 生成同格式的合成文件（测试里用它验证能找回 C_d = 0.55）。

**输出**（默认 `results/autodl_design/`）
- `calib/`：`calib.json`（`calib.calibrate` 同格式：每个起点的历史和最优、参照点）、`best.json`、`state.pkl`、`log.txt`、
  `predict/predict.json`、`predict/process_<图案>.png`、`predict/final_N256.png`
- `design/`：`design.json`（每个图案：S1 原动作参照、筛选最优、复核最优、解码后的动作、各项目标和约束、每代历史）、`state.pkl`、
  `log.txt`、`data/`（重新设计的姿态和程序，s1_forward 格式，autodl_v05 `--data` 可直接用）、`robust_check/`（P1–P6、R1–R3 网格、
  `fig_robust_grid.png`）、`final/`（256²/512² 终图、`movie.mp4`、`fig_aligned_six.png`）、`final_s1/`（S1 原动作同一路径对照）、
  `fig_design_compare.png` + `design_compare.json`（V0.5 参考 / S1 原动作 / 新设计，IoU、白区、倒奶量）

**预计用时**（**估计值，未在真机上测过**；按每个子步的启动数和带宽、每帧的主机开销推算，以 `bench_cuda --quick` 和日志里每代的
实际秒数为准）

| 步骤 | A100 | RTX 4090 |
|---|---|---|
| ① 标定：20 代 × 72 次 96² 前向 + 参照点 | 20–40 min | 30–60 min |
| ① predict：12 次 128² + 12 次 256² 末帧 | 3–6 min | 5–10 min |
| ② 设计：6 图案 × 20 代 × 288 次 128² + 每 2 代 48 次 256² 复核 + 壶（含流量标定割线） | 1.5–3 h（每代 4–9 min） | 3–6 h（每代 8–18 min） |
| ③ 稳健性：60 次 256² | 10–25 min | 20–40 min |
| ④ 终图：6+6 次 256²、6 次 512²、视频编码 | 15–30 min | 25–50 min |

省时：`ROB=` （不算稳健项，约一半）、`POP=8`、`GENS=12`、`PATTERNS=layered_heart,tulip,leaf,swan`（变化大的四个）。

**要发回来的**：`results/autodl_design/` 下除 `*/state.pkl`、`final*/**/final_state.npz`、`snapshots_c.npz` 以外的全部（尤其
`calib/calib.json`、`calib/best.json`、`calib/predict/`、`design/design.json`、`design/log.txt`、`design/data/`、
`design/fig_design_compare.png`、`design/design_compare.json`、`design/final/results.json`、`design/final/fig_aligned_six.png`、
`design/robust_check/fig_robust_grid.png`、`design/robust_check/results.json`、所有 `log_*.txt`），视频可选；以及
`results/autodl_selftest/` 的日志和 `nvidia-smi` 截图（看 GPU 利用率）。

**只有真 GPU 能验证的**：用时和 GPU 利用率（主机端每帧的程序计算可能成为瓶颈）、288 成员批的显存（估计 ~1.8 GB）、
GPU 壶在完整姿态上与 CPU 壶的一致性（已有 `test_cuda_pitcher`）、优化结果本身（模拟器上只跑了极小网格和截断时间）。

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
