# 拉花模拟器：现用的两个模型（记录）

本文记录 2026-10-24 时模拟器实际使用的两个模型：**倒奶模型**（拉花缸 → 杯面上的奶流）和**拉花模型**（杯面表层 → 白色图案）。公式和数值都对照分支 `claude/adoring-fermi-yt9iu6` 的 HEAD 逐行核对过（含 8d96389 壁面出口、4f0adf8 三视角视频）。代码与文档不一致时以代码为准，见 §6。`SC` 指会话草稿目录 `/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad`，证据文件都在那里，不在仓库里。

状态标记：**测** = 测量；**推** = 独立来源或物理推得，未拟合；**拟** = 在 tulip、leaf 上拟合；**拟(t)** = 从目标视频读出的起点值；**设** = 声明的假设或离散选择。

---

## 1 总览

```
手部姿态 pose v2（两相机标签 + CAD 对深度的连续拟合）
 └ real_program.cup_poses：1 ms 网格，杯坐标系 F，平滑
   └ 倒奶模型 pitcher.ForwardPitcher（CAD pitcher_kona700，TIP_KONA，装奶量按称重 165 g 反解）
     ├ 第 1–6 步：准静态液面 → 逐段堰流 Q_i → 体积账本
     ├ 出口：record = 唇法向 + 弹道 | law = 壁面出口 + 毛细飞行 | replay = law + 窗口内实测修正
     └ arrivals：按到达时间分到 1 ms 箱
       └ 入口记录 (x, y, Q, u_h, U_perp, d, r1, r2, phi, V_j)，每控制帧一行（schedule.build）
         └ 拉花模型 diff/wsolver.WSolver（Warp float64，与 solver.Solver 逐步相同）
           └ c = m/l → 评分（J、SEP）、三视角视频（viz/，viz21）
```

| 项 | 定义 |
|---|---|
| 时钟 | `cup_poses` 的 t = 0 是第一个同时有姿态和液面圆盘的样本（tulip 原始时钟 98.9101 s，leaf 11.0818 s，v1/v2 相同）。出流步 $t_k=(k+\tfrac12)\Delta t$，Δt = 1 ms。求解器时间与之相同，不缩放。时长 = $T_{\rm pour}$ + 1 s（`TAIL`） |
| 杯坐标系 F | 原点在液面圆盘中心（`cup_register.py` 的刚体杯轨迹），轴为 $(x_b,-y_b,-z_b)$（`S_FLIP`，标定板向上为 −z），z 向上，咖啡面在 z = 0 |
| 壶体系 | CAD X 指向壶嘴（真实壶的 +y），Z 为壶轴；$R=R_z(\text{yaw})R_y(\text{tilt})R_x(\text{roll})$，tilt > 0 时壶嘴向下 |
| 长度单位 | $L=R_{\rm disc}/0.49$：求解器杯半径 0.49 对应实测圆盘半径（tulip 85.5 mm，leaf 86.3 mm）。代码注释里的“D”就是 L |
| 求解器坐标 | $x=X_F/L$，$y=Y_F/L=-(y_b-c_y)/L$；速度 /L，流量 /L³。网格 $[-\tfrac12,\tfrac12]^2$，杯为 $r\le0.49$ |
| 圆盘坐标 | $(U,V)=(x/0.49,-y/0.49)$，以圆盘半径为单位，行方向为 +板 y；视频与目标图用它 |

倒奶模型内部全用 SI。`scaled_record` 先乘 0.08/L，`FwdProgram` 再除以 0.08，结果就是上面的求解器单位。

---

## 2 倒奶模型（拉花缸）

### 2.1 输入（`tulip_feas.Pour(flow='kona')`）

- **几何**：`pitcher_geom/pitcher_kona700.json`（KONA 700 mL，按目录图描线，卷尺核对），由 `GridPitcherGeometry` 读入，唇斜为 0。
- **壶嘴尖**：`TIP_KONA` = (0, 50.2, 87.9) mm。pose v2 独立给出 (−0.2, 50.1, 88.3) mm。没有 DTILT 修正。
- **姿态**：Slerp 到 1 ms，四元数与位置都做高斯平滑（σ = 0.04 s）。ω 由 $\dot RR^{\mathsf T}$ 的反对称部分得到。
- **装奶量**：`full_pour_kona_poured165.json` 的 `fill_at_cut_mL`：tulip 159.8、leaf 183.6、heart 156.5、push_heart 165.6、swan 162.5 mL。
  - 来源：整次倒奶称得 165 g，按 0.8 g/mL 合 206.25 mL；用整段录像反解初始量，取图案段开始时的余量。
  - pose v2 另加一条：本段送出的体积与 v1 链相同。
  - `calibrate_fill` 不在这条链上。

### 2.2 第 1–6 步（三种模式共用，逐位相同）

1. **前沿**：$r_{\rm rim}>\mathrm{median}+0.5$ mm 且与鸟嘴连通的沿口弧。段宽 $\Delta w_i$ 取垂直于主方向的投影宽。主方向为水平唇法向的平均，权重为 $h_i^{3/2}$ × 弧长。
2. **唇速度**：$\mathbf v_{{\rm lip},i}=\mathbf v_{\rm tip}+\boldsymbol\omega\times(\mathbf m_i-\mathbf x_{\rm tip})$。
3. **液面**：水平液面 $z_f$ 由“液面下体积 = $V_P$”定出（`FastPitcher.free_surface`，无晃荡）。水头 $h_i=[z_f-z_{{\rm lip},i}]_+$，无死水头、无起停滞回。
4. **堰流**：$Q_i=\tfrac23C_d\sqrt{2g}\,\Delta w_i h_i^{3/2}$，$C_d$ = 0.60。前沿以外被淹的沿口按同式计为侧溢，不进杯。
5. 惯性状态关闭。
6. **账本**：$\dot V_P=-(Q_o+Q_{\rm spill})$，且 $V_P+V_{\rm jet}+V_{\rm cup}+V_{\rm spill}=V_P(0)$。

### 2.3 新出口 `ForwardLaw(exit='wall', capillary='round')`（模式 law / replay）

这一节替换第 7、7b、8 步，不是叠加（`exit_lift`/`anchor` ≠ 0 时报错，EXIT_CAL 被拒）。它是 M5 `exitlaw.py`（sha256 b7c3a8ba…）的逐行移植，T2 与哈希预测相差 ≤ 1e-9。

**P1 方向（假设的规则）。** 由 CAD 内壁 $r_{\rm in}(z,\varphi)$ 求外法向 $\mathbf n_w=\mathbf P_\varphi\times\mathbf P_z/|\cdot|$，取 $\mathbf d_i=\mathbf n_w\times\mathbf s_i/|\cdot|$（壁切平面内、垂直于沿口、指向开口外），再流量加权：$\mathbf e_o=R\sum Q_i\mathbf d_i/|\cdot|$。`clip_rise`：$e_{o,z}>0$ 时仰角置 0，tulip / leaf 分别有 58 % / 99 % 的出流步被截。这个方向没有从动量平衡推出，测量也检验不到。

**P2 速度。** $H=z_f-z_{\rm low}$。沿口廓线在主方向上做条带离散：

$$A=\sum\Delta w_i(y-b_i)_+,\ \ B=\sum_{b_i<y}\Delta w_i,\ \ S_1=\tfrac12\sum\Delta w_i(y-b_i)_+^2,\ \ P=\sum_{b_i<y}\ell_i$$

由 $H=y_c+A/(2B)$ 二分求 $y_c$，再算

$$Q_c=\sqrt{gA^3/B},\quad u_j=\frac{Q_c}{A}+\frac{gS_1}{Q_c},\quad k_M=\frac{u_j}{\sqrt{2gH}}\le\frac{\sqrt3}{2}$$

$$\theta_m=\sqrt{\tfrac{0.45}{6}}\sqrt{\frac{\nu c_LH}{k_M\sqrt{2gH}}},\quad C_{\rm visc}=\max\!\Big(0.5,\,1-\frac{\theta_mP_b}{A_b}\Big),\quad U_o=C_{\rm visc}k_M\sqrt{2gH}$$

- 下标 b 指深度 $\kappa_by_c$ 处的 brink 断面。
- 典型值：CAD 鸟嘴上 $k_M$ 0.766–0.777；出流步上 $C_{\rm visc}\ge0.876$。
- 质量按锐缘堰、速度按宽顶断面，是两种堰顶理想化并用（已声明）；$Q_w/Q_c\approx1.10$，见 `E['Qc']`。

**P3 出流点**：brink 断面形心，高度 $z_{\rm low}+y_b-S_1/A$；该点的唇速度为 $\mathbf v_{\rm tip}+\boldsymbol\omega\times(\mathbf x_o-\mathbf x_{\rm tip})$。

**P4 毛细飞行 `jet_flight`**（参考系以出流时的 $\mathbf v_{\rm lip}$ 平移）：

$$\mathbf M=\rho Q\mathbf v_{\rm rel}-T\hat{\mathbf t},\quad \dot{\mathbf M}=-\rho Qg\mathbf e_z,\quad \mathbf M_0=\max(\rho QU_o-T_0,0)\,\mathbf e_o,\quad |\mathbf v_{\rm rel}|=\frac{|\mathbf M|+T}{\rho Q},\quad T=\pi\sigma\sqrt{\frac{Q}{\pi|\mathbf v|}}$$

- 0.25 ms 中点步积分到 z = 0；σ = 0 时与闭式弹道相差 ≤ 2e-7 s（T3）。
- 'sheet' 与 'none' 两种变体保留作对照。
- 输出：$x_{\rm hit}$、$u_h$、$U_\perp=-v_z$、$t_{\rm arr}=t+t_f$。`wall_miss`（有流量却没有 brink 断面）实际发生 0 次。

### 2.4 分箱、足迹、送入求解器

- **分箱**（`arrivals`，两种出口共用）：每个奶包的到达窗口介于相邻奶包到达时刻的中点之间（流的两端取 ±Δt/2），体积按重叠分到 1 ms 箱，箱内量取体积加权平均。飞行时间变化会拉伸或压缩到达流量。
- **记录 → Inlet**（`s1_forward.FwdProgram` → `drvf_record`，L 单位）：
  - Q 夹在 [1, 32] mL/s。区间按 0.08 m 缩放记录计，SI 下 ×(L/0.08)³ ≈ ×1.22。
  - 低于下限时：相位类别不是 'low'/'cut' 则关闭，否则抬到下限。'low' = 观察到在流，即轴倾在 1 s 滑动最大值的 6° 内。
  - $|u_h|\le\min(5,U_\perp)$ L/s（tulip 合 0.4275 m/s，MI 下不触及）。
- **足迹**（圆射流斜截面），$s=\sqrt{U_\perp^2+u_h^2}$：

$$d=2\sqrt{\frac{Q}{\pi s}},\qquad r_1=\frac d2\frac{s}{U_\perp},\qquad r_2=\frac d2,\qquad \varphi=\operatorname{atan2}(u_y,u_x)$$

  `schedule.build` 写入 $Q=S_{\rm eff}/k_Q$，所以求解器里起作用的是闭包的 $\kappa_Q$。`arrivals` 自己的 $r_1,r_2$ 和 `pitcher.Coupling` 不在这条链上。
- **帧**：1/120 s；有扫描时再限为 $0.5\min(h,r_1,r_2)/v_{\rm scan}$，$v_{\rm scan}\le0.5/(N\cdot1\,\text{ms})$。入口在帧中点取值，帧内冻结。
- **$V_j$**（`foot_velocity`）：落点位移的指数滑动平均，τ = 0.15 s，断流时清零。

### 2.5 旧出口（record，默认）与 EXIT_CAL（已退役）

**record**：$U=C_v\sqrt{2gH}$，$C_v$ = 1，H 为最大水头；方向为外唇法向，之后走闭式弹道。68–80° 轴倾下方向比水平低 70–80°，$u_{h,\rm rel}\approx0.24U$。错在方向，不在参数。在验证过的柱脚上，u_h 低 2.1–2.4 倍，落点短 11.3 / 10.8 mm。

**EXIT_CAL**（$C_v$ 1.5，抬 9.5°，前移 3 mm）：在多为高位、斜率无符号的 E0 帧上标定。$C_v$ 1.5 要求 2.25 gH 的动能，可用的只有 gH。u_h 对，$U_\perp$ 错（0.73 / 0.68 m/s），落点短 5–6 mm。law/replay 下拒用；record 下只为复现旧结果保留（闭包 `exit_cal: 1`）。

### 2.6 入口模式与开关

| 模式（闭包 `inlet`） | 内容 | E0 |
|---|---|---|
| record（0，默认） | 第 1–8 步；与 fc9f270 逐位相同（T1） | 按闭包 `e0`，柱脚文件由 `LATTE_FOOT` 选 |
| law（1） | 第 1–6 步 + `EXIT_LAW`；五个图案 | 强制关 |
| replay（2） | law + 窗口内实测；仅 tulip、leaf | 'world'，`LATTE_FOOT=mi`（未设则就地设，设成别的报错） |

**优先级**（`resolve_inlet`）：`INLET_OVERRIDE[p]`（diag 对留出图案设 'law'）> 闭包 `inlet` > `LATTE_INLET` > 'record'。

**replay**（`rig/measured_inlet.py`）：
1. **窗口**：tulip [0, 1.702] s 与 [3.377, 4.144] s，leaf [0, 4.122] s。权重 $w(t)$ 窗内为 1，窗外 0.25 s 内按 cos² 降到 0。
2. **门控帧**：F0 为 `use_kin` 且壶嘴高 < 15 mm；E1 为出流速度 ≤ $\sqrt{2g(H_{\rm head}+3\,\text{mm})}$。tulip 35 帧，leaf 52 帧。
3. **残差**：$r_q=q_{\rm meas}-q_{\rm law}$，$q\in\{u_x,u_y,w_f,a_f\}$；$q_{\rm law}$ 是 law 到达记录在 ±20 ms 内的 Q 加权值。
4. **GP**：每个 q 一个零均值 Matérn-3/2 + 白噪声 + η·sd68 的 GP。超参数用最大边际似然求得后冻结在 `stream_kin.json`（LOO z-sd 0.98–1.02）。运行时若配对帧或 $\hat r$ 与冻结值不符（容差 1e-9）就报错。
5. **逐奶包修正**：$u_h\mathrel{+}=w\hat r_u$，$U_\perp\mathrel{+}=w\hat r_{w_f}$，$t_{\rm arr}=t_{\rm em}+t_{f,\rm law}+w\hat r_{a_f}$（2 次不动点迭代）。Q、出流点、$x_{\rm hit}$ 都是律的；支撑区外与 law 完全相同。
6. **落点**：E0 'world'，对“柱脚 − 模型落点”做 Tukey 稳健时间加权（σ 0.10 s）。用 `stream_foot_mi.json`：只保留 `stream_foot_col.json` 中 A/B 级、w > 0 且门控通过的行（35 / 52 行）。窗口外退回中位偏移。

### 2.7 参数

| 参数 | 值 | 来源 | 状态 |
|---|---|---|---|
| 几何、壶嘴尖 | kona700，TIP_KONA | 目录图 + 卷尺；pose v2 核对 0.5 mm 内 | 测 |
| 姿态 | pose v2，σ 0.04 s | 约 0.2° / 1 mm | 测（平滑为设） |
| V0 | 156.5–183.6 mL | 称重 165 g，0.8 g/mL | 推 |
| $C_d$ | 0.60 | V 形堰文献 0.58–0.60 | 推 |
| ρ / σ / ν | 800 kg/m³ / 0.045 N/m / 1.5e-6 m²/s | 微泡奶；50–65 °C 牛奶；含气修正（范围 600–1030 / 0.040–0.052 / 1e-6–5e-6） | 推 |
| $c_L$ / $\kappa_b$ | 4（2–8）/ 0.75（0.70–0.80） | 堰前加速段；Rouse 0.715 至三角形 0.795 | 推 |
| clip_rise / capillary | True / 'round' | 复现 M5；规则 2 选出 | 设（离散拟合选择） |
| GP、窗口、门 | `stream_kin.json` | 测量残差的最大边际似然；门在看结果前声明 | 测（冻结） |
| GAP_S、PAIR_S、V_FOOT_TAU | 0.25 s、20 ms、0.15 s | SPEC27、规则 2、E0 抖动 | 设 |

除了 'round' 这个离散选择，倒奶模型没有一个数是在拉花图案上拟合的。

### 2.8 对照实测奶柱

测量来自 M1：在 col25 rev-2 验证过的奶柱上反演，出流点为先验 $d_a\in[-5,0]$、$d_z\in[0,3]$ mm。表中为门控帧中位数，误差依次为 ± 统计 ± 系统 ± 结构（SPEC27 §1.1、§2.3）。

| 量 | tulip 测 | leaf 测 | law（phys_round）tulip / leaf | record |
|---|---|---|---|---|
| \|u_h\| m/s | 0.255 ±0.042 ±0.020 ±0.04 | 0.240 ±0.040 ±0.016 ±0.04 | 0.285 / 0.230 | 0.116 / 0.100 |
| $U_\perp$ m/s | 0.52 ±0.027 ±0.020 ±0.04 | 0.50 ±0.025 ±0.020 ±0.04 | 0.536 / 0.519 | 0.62 / 0.59 |
| 出流速度 / $\sqrt{2gH}$ | 0.68 | 0.72 | 0.743 / 0.737 | 1 |
| 落点（沿 / 横，相对柱脚） | | | +0.3/−1.8；+0.1/+1.1 mm | −11.4 / −10.3 mm（沿） |

求解器内落点（T8，N48）：tulip +0.37 / −1.94 mm，leaf +0.14 / +1.11 mm，限值 3 mm。

**规则 2**（(a) |Δu| ≤ 0.05；(b) ≥ 70 % 的帧在 2σ 内；(c) |ΔU| ≤ 0.05；(d) 落点 ≤ 3 mm；(e) M5 证伪条件）：

| 律 | tulip Δu / (b) / ΔU | leaf Δu / (b) / ΔU | 结论 |
|---|---|---|---|
| phys（液片张力） | −0.033 / 0.92 / +0.023 | **−0.098 / 0.51** / +0.010（落点 −4.3 mm） | 不过 |
| **phys_round** | +0.033 / 0.89 / +0.021 | −0.003 / 0.84 / +0.017 | **过（勉强）** |
| phys_nocap | **+0.057** / 0.76 / +0.016 | +0.033 / 0.80 / +0.011 | 不过 (a) |
| record | −0.14 / 0.27–0.29 / +0.09–0.11 | 同左 | 不过 |
| EXIT_CAL | +0.02 / 0.82–0.89 / **+0.18–0.21** | 同左 | 不过 (c, d) |

tulip 的 (a) 余量 0.017 m/s，小于共模系统误差 0.020。u_meas 减 1σ_sys，或出流点放在壶嘴尖，都会导致不过，所以状态是**暂定**。

**留出 heart**（只报告）：phys_round 的 $U_\perp$（0.468 对 0.478）和落点（+1.8 / −1.4 mm）都对，但 u_h 为 0.281 对 0.185，偏高 0.10 m/s。heart 的测量是未验证的 Method-A 重跑，这可能是律的错，也可能是测量的错（R2）。

### 2.9 已知限制

- **方向是假设**：3–4 mm 深的 brink 跟不上最后 21° 的外张，动量积分给出上仰 +5…+13°。圆射流减速与方向可能互相混淆；飞行时间由先验主导，测不出来（R7）。
- **R1**：自由落体柱比实测柱弯（约 10σ，两台仪器同号），成因未知，只在结构项内影响 u_h。
- **起流**：0–0.45 s 实测流量只有堰流的 0.4–0.74；流在 t = 0 前 0.3–0.5 s 已重启。没有起流律，Q 仍是堰流值（R3）。
- **高位倒奶**（壶嘴高 ≥ 15 mm）：没有测量，只用律（R4）。
- 其他：留出图案只有 law；没有晃荡；咖啡面固定；杯是替身。

---

## 3 拉花模型（液面）

### 3.1 方程（closure 'chi'；`diff/wsolver.py` = `solver.Solver`，`test_parity.py`）

状态为 ℓ（有效表层量，静止 = 1）、$\mathbf q=\ell\mathbf u$、新奶 m；白度 $c=m/\ell$，棕色 $b=\ell-m$。

$$\partial_t\ell+\nabla\!\cdot\mathbf G=s,\qquad \partial_tm+\nabla\!\cdot(\mathbf Gc_f)=s_m+\nabla\!\cdot(\ell D\nabla c)$$

$$\partial_t\mathbf q+\nabla\!\cdot(\mathbf q\otimes\mathbf u)=-\nabla\Pi+\nabla\!\cdot(2\nu\ell\,\mathsf S(\mathbf u))-\beta\mathbf q+(\tilde s_m+\Lambda)\mathbf v_{\rm tgt}-\Lambda\mathbf u+(s-\tilde s_m)\mathbf u$$

- **压力**：$\Pi=\tfrac12c_p^2[g(c)\ell^2+1-g(c)]$，$g=1-\xi_b(1-c)$，所以 $-\nabla\Pi=-c_p^2g\ell\nabla\ell$（隐式）$-\tfrac12c_p^2(\ell^2-1)\xi_b\nabla c$（显式）。
- **黏性**：对称形式，隐式 PCG：$\ell\mathbf u^+-\Delta t\nabla\!\cdot(2\nu\ell\mathsf S(\mathbf u^+))=\ell\mathbf u^-$，`visc_tr` = 0。
- **源**：$\tilde s_m=(1-\mu_s)s_m$；两套闭包都是 $\mu_s=0$、upwell = 0、$e_F=0$，于是动量源就是 $s\mathbf v_{\rm tgt}+\Lambda(\mathbf v_{\rm tgt}-\mathbf u)$。
- **开放棕色面**（$\lambda_b>0$）：每子步乘 $e^{-\Delta tW_o}$，$W_o=\lambda_b(\ell-1)/\ell$，只作用于 b，即 $db/dt=-\lambda_b(\ell-1)(1-c)$。$\kappa_e$、$\kappa_M$ 两套都是 0。

**数值**：每子步依次为 A（Rusanov 显式动量对流）→ B（质量 + 压力 + 拖曳 + 牵引，隐式 SPD，PCG 1e-10）→ C（奶用同一面通量 G）→ D（黏性、混合）。$\Delta t\le0.25h/\max|\mathbf u|$，每帧 ≤ 64 子步。N 256（AutoDL），本地 N 96 / 128。

- **足迹核**：$(1-|\mathbf y|^2)_+^2$，半轴下限 1.5h。
- **杯外**（`offcup` 'account'）：$Z$ 为核在杯内的份额，$(1-Z)$ 记为溢出。
- **奶面值 $c_f$**（闭包 `milk`）：0 为 van Leer（v05n）；1 为 mp5fct，即 MP5 + Zalesak FCT，界 [0, ℓ]（S1kq）。

### 3.2 落点闭合（`wsolver.k_fields`）

$$Z_c=\frac{U_\perp^2}{B_{\rm dep}d},\quad \chi=\frac1{1+Z_c^{p_{\rm dep}}},\quad S_{\rm rec}=\kappa_QQZ,\quad s_m=\chi S_{\rm rec}K_d,\quad \Lambda=\kappa_c(1-\chi)S_{\rm rec}K$$

$$V_R=\min(U_\perp\chi^{1/(2p_{\rm dep})},U_\perp,\sqrt{B_{\rm dep}d}),\qquad \mathbf v_*=\kappa_t\mathbf u_{\rm in}+\kappa_rV_R\frac{\mathbf x-\mathbf x_{\rm hit}}{\sqrt{r_1r_2+|\mathbf x-\mathbf x_{\rm hit}|^2}}$$

- $K_d$ 以 $\mathbf x_{\rm hit}+\tau_s\mathbf u_{\rm in}$ 为中心（偏出杯外时退回 K）。
- $Z_c$ 用 $U_\perp$ 而不用 s，是声明的选择；MI 下 $s/U_\perp$ = 1.10–1.12。
- vj 开（S1kq）时：$\mathbf v_{\rm tgt}=[\tilde s_m\mathbf v_*+\Lambda(\mathbf V_j+\kappa_{tc}(\mathbf u_{\rm in}-\mathbf V_j))]/(\tilde s_m+\Lambda)$；关时 $\mathbf v_{\rm tgt}=\mathbf v_*$。
- 单位：$c_p$ L/s，β 1/s，ν、D L²/s，$B_{\rm dep}$ L/s²，$\tau_s$ s。

### 3.3 两个版本

| 键 | v05n A0（`bp256_trained` = `bp256_mi`） | v05n update21 拟合（`res21/.../v05n/params/best_closure.json`） | S1kq（`mcd_S1kq_N96`） | S1kq_mi |
|---|---|---|---|---|
| cp / β | 0.68 / 0.5784 | 0.6247 / 0.8622 | 1.7712 / 0.7017 | 同左 |
| ν / D | 0.001 / 1e-7 | **0.0199** / 1e-7 | 0.01 / 1e-7 | 同左 |
| κ_Q | 2.5937 | 2.4151 | 2.06 | 2.06 |
| κ_c / κ_t / κ_r | 0 / 0.1964 / 0.2 | 0 / 0.3360 / 0.1145 | 0.3465 / 0 / 0 | 同左 |
| B_dep / p_dep | 2665.2 / 8 | 1698.2 / 12（箱顶） | 1267.4 / 7.150 | 同左 |
| τ_s [s] | 0 | 0 | 0.075 | 0.03478 |
| vj、κ_tc / ξ_b、λ_b / milk | 0、0 / 0、0 / vl | 同左 | 1、0 / 0.5、10 / mp5fct | 同左 |
| e0 / inlet | 2 / record（bp256_mi 为 2） | 2 / 2 | 2 / — | 2 / 2 |

**状态**（SPEC27 §4，闭包 `_note`）：
- **v05n**：全部是**拟**，而且是在 record 入口（u_h 低 2.2 倍）上拟合的，κ_t 等吸收了入口误差。
- **S1kq**：
  - κ_Q 为**拟(t)**（白面积预算，D3c AR̄ 0.9877，保留 2.06）。
  - B_dep 为 C：在独立视频阈值范围 1230–2210 内，按 MI 的 $f_z$ 0.64–0.94 映射后仍在范围内。
  - p_dep、κ_c 为**拟**。
  - τ_s：0.075 s 来自视频白色位置（8.5 mm），已被激发律（3.2–5.6 mm）排除，只是保留未改；_mi 版取 $f_u\times0.075$（$f_u$ = 0.4637），状态**拟(t)**。
  - cp、β、ν、ξ_b、λ_b 为 C/D，与入口无耦合；κ_t、κ_r、μ_s、κ_tc 已移除。
- **update21 的 S1kq 拟合**：τ_s 0.0534、κ_Q 1.736、B_dep 1639、p_dep 8.26、κ_c → 0。

### 3.4 对在哪里、错在哪里

**v05n 起点**（MI-replay，低 ν）：
- 对的：新入口下出现分层的形（心形 + 嵌套弯月 + 棕线），tulip SEP 0.91（N256）、0.96–0.97（N96/128）。按 SKEL、LKEEP，它在两个图案、两种参考下都比拟合结果更接近视频（`SC/tgt30/C`，N256）。
- 错的：**白色前移约 5 mm**。tgt30/B 测得 3–6 s 质心 +5.0 mm；off34 测得 tulip 稳定在约 +5.5 mm，leaf 3–6 s +11、6–9 s +14 mm。前 1.5 s 反而偏后 1.6–2.0 mm，前沿扩展是视频的 1.5 倍。
- 原因（off34）：偏移由 κ_t 决定。κ_t = 0 时白色后移 17 mm，扩展速度与视频一致，但浮现点差 6–8.6 mm（视频的新白在柱脚前 9.4 / 7.8 mm 浮出）。修正所需的 κ_t 倍数随时间变化（前 1.5 s 为 1.2–1.3，之后 tulip 约 0.7、leaf 约 0.35）。白色是被“推”过去的，应改为“放”在柱前：降低动量转移，同时让新白在柱前浮出。
- κ_c = 0，没有奶柱拖穿表层（拉出茎）的机制。

**v05n update21 拟合**：
- ν ×20（0.0199 L²/s = 1.45e-4 m²/s，是热牛奶的 150–300 倍），条纹周期内的扩散长度从 1.3 mm 变成 5.7 mm，分层被抹成“球 + 弯月”。
- 只换 ν 一项就让 SEP 归零，并占损失下降的 65–68 %，所以 ν 在这里是模糊旋钮（tgt30/B）。
- J 从 0.543 升到 0.752，主要来自填棕线和挪位置。

**S1kq**：
- 没有自由的动量参数；κ_Q 由白面积预算重测，B_dep 在独立范围内。
- 求解器内有三项相对奶柱的检验不过（`SC/src26/spec/SPEC4.md` §0）：
  1. **奶柱被包住**：tulip T9 = 0.94（允许 [0, 0.15]），0.95 s 已包住，视频要到 1.79 s 以后。
  2. **后方 crema 向外走**：leaf +3.24 mm/s，视频 10 个读数全部向内（−1.35 至 −2.44）。“包柱定理”指出：在无旋层、白色为被动示踪的假设下，这两点是必然结果。
  3. **leaf 环心**：+9.25 mm，视频 +3.17 mm。
- **缺浮现延迟**：白色早出约 0.27 s。M4 的结论是首批 6–10 mL 奶先钻到 crema 下约 0.25–0.3 s，入口本身没错。Stage 1 的 T8 起白（t50）为 0.12 / 0.14 s，早于视频区间（约从 0.37 / 0.40 s 开始）。
- update21 中 J 0.35–0.45，SEP 0（无形团块）。
- 两个模型都**没有**推导出的拖穿 / 茎机制：v05n 的 Λ ≡ 0；S1kq 的 Λ 只拖向 $V_j$（κ_tc = 0），update21 拟合还把 κ_c 压到 0。C 的 CUT 指标在真视频性质检验中不过，所以这一点也测不了。

---

## 4 拟合（反演）

### 4.1 设置

- **标定**：tulip、leaf（有测量入口，用 MI-replay）。
- **留出**：heart、push_heart、swan。参数冻结后用 MI-law 跑（e0 0），只报告；跑失败即中止。
- **自由参数**：
  - v05n 为 `PARAM_WIDE` 九个数：cp 0.1–6、β 0.02–30、ν 1e-4–0.05、κ_Q 0.2–15、κ_c 0–3、κ_t 0–2、κ_r 1e-3–1、B_dep 30–3e4、p_dep 0.5–12。
  - S1kq 为 pset S1 五个数：κ_c、p_dep、τ_s 0–0.12、κ_Q 0.5–6、B_dep。
- **模式**：params 只放开参数；inlet / both / both_wide 在 'meas' 盒内修正入口。后三种是 **oracle**，只作诊断，不进模型。
- **评分**：J = 0.25 viou_n + 0.75 struct_n（N48 等效）；SEP 为最后 30 % 帧中成对分开的白色部件在模拟里也分开的比例。“打开” = tulip J ≥ 0.80 且 SEP ≥ 0.50。

### 4.2 旧损失与失败原因

```
L = Σ_j wt_j [ (1/3) Σ v(p−w)²/Σ v + Σ_{s=0.15,0.3} coef_s Σ_keep (G_s*(vp) − G_s*(vw))² / G_s*v ]
p = sigmoid((c−0.5)/τ)，τ 0.10 → 0.03；Adam lr 0.03；最优迭代按硬 IoU 选
```

w 为旧目标 W（`video_obs`），两个模糊尺度为 6.3 / 12.6 mm。失败原因（tgt30/B、C）：
1. **目标没有细结构**：W 只找回真实棕线的 0.08（精确率 0.47），因为分类器是在填满条纹的 Otsu 掩模上训练的。
2. **像素项占 83–87 %**：填掉起点自己的棕线，损失降 10 %。在真视频上，硬 IoU 给填满版 0.96–0.98、4 mm 团块 0.78–0.87，都高于真图案平移 2 mm 的 0.70–0.78。
3. **ν 自由且只按 IoU 选**：J 在第 1–7 次迭代就到 0.70–0.71，那时形还在；之后 J 只涨 0.02，ν 却涨了 11 倍。

### 4.3 新目标

- **N4**（`SC/tgt30/A`，`w_gated`，256 px）：
  - 构造：每台相机在线性光下按局部端元解混出泡沫份额，跨相机去高光，ECC 配准到相机 1，按 (px/mm)⁴ 融合，限制在 W > 0.5 外扩 2 mm 内。
  - 结果：召回 / 精确率 / F1 为 0.74 / 0.90 / 0.81（W 为 0.08 / 0.47 / 0.14）；留出 0.79 / 0.91；独立原始横截线上 10/11（W 为 3/11）。
  - 局限：线深被夸大 1.2–1.4 倍，位置跟随相机 1（0.4–1.1 mm），严格精确率 0.59。
- **M2_multicam**（`SC/tex31/final`）：
  - 构造：三相机矫正，两端元解混，配准融合，保边时序滤波，刻入棕线，另附 `line` 通道。
  - 盲标 TEST：F1*@1.5 mm 0.686（旧 0.574），细线召回 0.84（旧 0.20）；tulip 0.905，leaf 0.335（弱）。
  - 接口：`tfinal.on_solver_grid` 可直接替换 `run.video_on_grid`。
- **现状**：两者都还没接进仓库，`diff/run.py` 仍读 `video_obs`。SPEC30 v2 用 N4；换 M2 要按 SPEC30 §2.7 修订。

### 4.4 新目标函数 `video30c`（`SC/tgt30/spec/SPEC30.md` rev 2）

$$T=\tfrac12\sum_{p}\Big[\frac{\mathrm{COV}_p}{u_{{\rm cov},p}}+\max\Big(0,\frac{\mathrm{LINE}_p(\theta)-\mathrm{LINE}_p(\theta_0)}{u_{{\rm line},p}}\Big)\Big]+\tfrac12\sum z_i^2$$

- **COV**：6.3 / 12.6 mm 两个尺度上的覆盖残差，p = clip(c)，无 sigmoid、无像素项。
- **LINE**：在 N4 棕线上的光滑双侧线保持，加单侧屏障：丢线付代价，多画线不得分。
- **单位**：$u$ 是视频自身图案刚性平移 2 mm 的代价。
- **先验**：$z_i=\ln(\theta_i/\theta_{i0})/\ln2$；κ_c、κ_t 用 $(\theta-\theta_0)/0.25$。
- **自由参数**：cp、β、κ_Q、κ_c、κ_t、κ_r、B_dep，在物理箱 `v05_adapter.SPACE` 内。**ν = 1e-3、p_dep = 8 固定**；起点 bp256_mi。
- **形是硬约束**：每个提议步都要通过 C 预注册的指标（N256 用 SKEL + LKEEP，N96 用 LAYERS + LKEEP；参考 R 与 A；两个图案；共 8 项）：$C_m(\theta)\ge\min(C_m(\theta_0),C_m(\text{video}))-\tfrac12\mathrm{thr0}_m$，且 T 上升 ≤ 0.1。不通过就回退并把步长减半，连续 4 次拒绝即停。
- **步长**：Adam 在先验单位下 0.045σ / 步，前 3 步 lr/3，信赖域 0.15。
- **选择**：在被接受的迭代中取 T 最小者，J、SEP 只报告。
- **保证**：形“在噪声内不比起点差”；不保证会有改进。注意这条规则也会拒绝 update21 的三个 oracle。
- **状态**：BUILD GO；AutoDL update22 要等 G2、G3 通过。本地 G3 L1（`SC/tgt30/G3/G3_verdicts.json`）：10 个提议接受 4 个，ΔT −0.16，COV −5 至 −7 %，参数移动 ≤ 0.04σ，主要卡在 leaf/R/LKEEP，判为“informative pass”。去掉约束的 L1f 下降更快，但迭代全被约束拒绝。

### 4.5 当前结果：update21（N 256，pose v2；`SC/res21/tar/results_diag21/summary.md`）

| 试验 | tulip J | tulip SEP | leaf J | 打开？ |
|---|---|---|---|---|
| S1kq / params（MI-R） | 0.348 → 0.381 | 0.00 → 0.00 | 0.449 → 0.437 | 否 |
| v05n / params（MI-R） | 0.543 → 0.752 | 0.91 → 0.68 | 0.559 → 0.733 | 否 |
| 对照 v05n / params（旧入口 R1） | 0.619 → 0.585 | 0.24 → 0.21 | 0.700 → 0.699 | 否 |
| S1kq / inlet（oracle） | 0.348 → 0.556 | 0.00 → 0.00 | 0.449 → 0.668 | 否 |
| v05n / inlet（oracle） | 0.543 → 0.840 | 0.91 → 0.88 | 0.559 → 0.788 | 是 |
| v05n / both（oracle） | 0.543 → 0.882 | 0.91 → 0.82 | 0.559 → 0.844 | 是 |
| v05n / both_wide（oracle） | 0.543 → 0.876 | 0.91 → 0.88 | 0.559 → 0.834 | 是 |

- **v05n 留出**（MI-L）：heart 0.416 → 0.579，push_heart 0.555 → 0.490，swan 0.402 → 0.611，SEP 全为 0。
- **读法**：
  - S1kq 打不开，是模型本身的问题。
  - v05n 只放开参数打不开；在测量 2σ 盒内修正入口就能打开（oracle），窗口内落点平移 p50 为 3.5 / 4.1 mm，量级与 §3.4 的位置误差相当。
  - J 的提升伴随形的丢失，所以 J 本身不能证明“更好”。

---

## 5 文件与开关速查

| 内容 | 位置 |
|---|---|
| 倒奶模型 | `latte_imex/pitcher.py`（`ForwardLaw`、`ForwardPitcher`、`jet_flight`、`arrivals`） |
| 姿态 → 入口 | `rig/real_program.py`（`cup_poses`、`build_inlet`、`resolve_inlet`、`EXIT_LAW`、`EXIT_CAL`）、`rig/tulip_feas.py:Pour`、`rig/full_pour.py:TIP_KONA` |
| 实测回放 | `rig/measured_inlet.py`；测试 `rig/test_measured_inlet.py`（T1–T9） |
| 调度 | `s1_forward.FwdProgram`、`v05_pitcher_check.drvf_record`、`diff/schedule.py` |
| 求解器 | `diff/wsolver.py`（生产）、`solver.py`（参考）、`kernels.py` |
| 拟合 | `diff/run.py:DiffPour`、`diff/diag.py`、`diff/render512.py` |
| 闭包 | `data/closures/bp256_mi.json`、`mcd_S1kq_N96.json`、`mcd_S1kq_mi_N96.json`、`data/mi27/bp256_trained.json` |
| 数据 | `raw_data/rig/results/<p>/{pitcher_pose.npz, disc.json, stream_kin.json, stream_foot{,_col,_mi}.json}`、`full_pour_kona_poured165.json` |
| 视频 | `viz/`（viz21：倒奶 / 客人 / 俯视；`forward_fields.py` 按 diag 方式重跑） |
| 规格与证据 | `V05_MODEL_SPEC.md` §3.11；`SC/mi27/spec/SPEC27.md`、`SC/mi27/M1–M5/REPORT.md`、`SC/tgt30/{A,B,C,spec}`、`SC/tex31/final`、`SC/off34/MEASURE.md`、`SC/src26/spec/SPEC4.md` |

| 开关 | 取值（默认在前） | 作用 |
|---|---|---|
| 闭包 `inlet` / `LATTE_INLET` | record / law / replay | 入口模式 |
| 闭包 `e0` | 0 / 1 jet / 2 world | 落点改到测得的柱脚 |
| `LATTE_FOOT` | v1 / col / mi | E0 读的柱脚文件 |
| `LATTE_POSE` | v2 / v1 | 姿态轨迹 |
| 闭包 `exit_cal` | 0 / 1 | 旧出口（只在 record 下） |
| 闭包 `milk` | 0 vl / 1 mp5fct / 2 thincfct / 3 mp5fct_nb | 奶面值格式 |
| 闭包 `vj`、`upwell`、`circ`、`sink`、`visc_tr` | 0 / 1 | 拖向 $V_j$、全量上浮、表面流、柱汇、迹应力 |
| `PITCHER_GEOM`、`PITCHER_LAW` | 路径、JSON | 壶几何、律字段 |

---

## 6 代码与文档不一致处（以代码为准）

1. `V05_MODEL_SPEC.md` §1.2–1.3 写的是逐分量黏性和 $s=\chi S_{\rm eff}K$（S1/V0.5 的设置）。现行代码用对称黏性 $\nabla\cdot(2\nu\ell\mathsf S)$，源为 $\chi\kappa_QQZK_d$（杯外记账 + $\tau_s$ 偏移核）。
2. SPEC27 §3.3 说窗口外 E0 回落值与律落点相差 ≤ 0.5 mm，N96 实测为 1.9 / 1.1 mm（V05 §3.11.5 已改）。
3. SPEC27 §3.1 说 $C_{\rm visc}$ 下限从不触发。实际在 Q > 1 mL/s 时确实不触发，但在 Q < 1 mL/s 的收尾细流步上会触发：tulip 299 步中 35 步，leaf 35 步中 4 步（`Cvis_guard_hits`）。
4. `pitcher.py` 的模块文档描述的是旧 `Pitcher`（Kindsvater、滞回、ρ 1030）。生产用 `ForwardPitcher`（h_k 0，无滞回，ρ 800）；`Coupling` 不在链上。
5. “没有拖穿是因为 κ_c = 0”只对 v05n 成立（以及 update21 的 S1kq 拟合）。`mcd_S1kq*_N96.json` 中 κ_c = 0.3465，Λ 存在，只是它拖向 $V_j$。
