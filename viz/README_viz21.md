# viz21：拟合后的三视角视频（update21 附加包，第 3 版：起点参数 + 拟合后参数）

这个附加包放进 update21 的同一棵目录树（和 `latte_imex/`、`scripts/` 同级）。对每个**已经完成**的 params 拟合，它渲染**两套**闭包，先起点、后拟合后：

| 套 | 闭包 | 字幕里写 |
|---|---|---|
| `start/` | 这次拟合的**起点闭包**，原样不动 | 起点参数（未重新拟合，<文件名>） |
| `best/` | 这次拟合的 `best_closure.json` | 拟合后参数（best_closure.json） |

为什么也渲染起点：在 v05n 的郁金香上，起点参数（`bp256_mi.json`，即 `bp256_trained` + 新的实测入口 `"inlet": 2`）的形态更对——心形 + 一层层的花瓣，树叶也是分层的——虽然它的 IoU 比拟合后的低；拟合后的 IoU 高，形态却不对。两套用同一个倒奶、同一个入口、同一个时钟，只差参数，可以并排比较。

每套、每个倒奶各输出四段视频和一张总览图：

| 文件 | 内容 |
|---|---|
| `<倒奶>_pour.mp4` | **倒奶视角**：从侧面看，壶是半透明的，能看到壶里剩多少奶、奶面怎么斜、奶柱怎么出来。左上角是时间、壶口流量 Q [mL/s]、已倒出 / 壶内剩余 / 已落入杯中 [mL]、液头、壶轴倾角、落点速度；左下角是 Q(t) 曲线和当前时刻光标。 |
| `<倒奶>_guest.mp4` | **客人视角**：坐在拉花师对面看杯子（40° 俯角），液面贴的是求解器的白度场，壶和奶柱在上方。 |
| `<倒奶>_top.mp4` | **俯视二维**：二维液面上的图案（求解器白度场，方向与真实俯视视频一致），红色十字 = 求解器的落点，红圈 = 求解器用的落点足迹椭圆，蓝色菱形 = 壶嘴尖的投影。 |
| `<倒奶>_combo.mp4` | 2×2 对比：倒奶视角 / 客人视角 / 俯视模拟 / **真实俯视视频**（`raw_data/rig/video_obs`，校正后），同一时钟。 |
| `<倒奶>_sheet.jpg` | 6 个时刻 × 4 个视角的总览图。 |

位置：`results_diag21/<模型>/params/viz/start/` 和 `results_diag21/<模型>/params/viz/best/`，模型 = `s1kq`、`v05n`、`v05n_ctrl`。倒奶 = tulip、leaf（标定）；两个实测入口模型（s1kq、v05n）的两套闭包另加 heart、push_heart、swan（留出，入口 MI-L、e0 关，只做报告）。加 `VIZ_HELD_RECORD=1` 时 v05n_ctrl 的两套也渲染这三个留出倒奶（旧入口 R1 + E0，和对照拟合评估留出倒奶时一样）。

| 模型 | 起点参数（`start/`） | 入口 |
|---|---|---|
| s1kq | `latte_imex/data/closures/mcd_S1kq_mi_N96.json` | 实测入口 MI-R（回放）；留出倒奶 MI-L、e0 关 |
| v05n | `latte_imex/data/closures/bp256_mi.json`（= `bp256_trained` + `"inlet": 2`） | 同上 |
| v05n_ctrl | `closures/bp256_trained.json` | 旧入口 R1（记录 + E0）+ 位姿 v2，和对照拟合时一样 |

起点闭包不是写死的路径，而是按这个顺序找，用第一个找到且对得上的：① 拟合的 `done.json` 里的 `init`（目录树搬过家，比如 u20 → u21，就在当前目录树下找同一个相对路径）；② 拟合的 `run.log` 第一行的 `start closure ...`；③ `scripts/autodl_diag21.sh` 里的 INIT 表；④ 仓库里的上表文件。“对得上”= 除了这次拟合放开的那几个数，其余每个数都和 `best_closure.json` 相同（`best_closure.json` 就是起点闭包把拟合的数替换掉）。入口用这次拟合自己的标定入口（`done.json` 的 `inlet.cal`；没有这一项就是旧入口 R1）。`viz.log` 里每个模型有一行写明用了哪个文件、从哪里找到的。

## 怎么跑（拟合已经跑完）

```bash
cd /root/autodl-tmp/u21                       # update21 的目录树
tar -xzf viz21_addon.tar.gz                   # 加入/更新 viz/、scripts/autodl_viz21.sh、README_viz21.md；update21 本身的文件一个不动
NOWAIT=1 bash scripts/autodl_viz21.sh         # 后台启动后立即返回
tail -f results_diag21/viz.log                # 看进度
```

- 拟合已经结束，所以用 `NOWAIT=1`：不再等拟合的锁，也不等显存（拟合不在跑时，不加也只是多做一次检查）。如果装过第 1 / 2 版附加包，直接解压覆盖即可（只替换附加包自己的文件）。
- 实例必须带 GPU 开机：无卡模式下没有 `cuda:0`，`viz.log` 里只有一行 `FAILED: device cuda:0 is not available ...`，什么也不渲染（`DEVICE=cpu` 可以在 CPU 上跑，但非常慢）。
- 默认任务：3 个模型 × 2 套 × (tulip, leaf) = 12 个，加 2 个实测入口模型 × 2 套 × (heart, push_heart, swan) = 12 个，共 **24 个**，按 tulip → leaf → heart → push_heart → swan 的顺序，同一模型先起点后拟合后。
- 再运行一次会续跑：每个（模型, 起点/拟合后, 倒奶）有自己的 `<viz 目录>/<倒奶>.done`，里面记着闭包文件的 sha256、入口和渲染设置（`N`、`VIZ_FPS`、`VIZ_T`、`VIZ_SIZE`、`VIZ_SS`、`VIZ_CUP`）；全都没变就跳过，改了其中任何一个就重渲（`viz.log` 写明哪一项变了），`VIZ_FORCE=1` 全部重渲。缓存的稠密场只在闭包、入口、`N`、`VIZ_FPS`、`VIZ_T` 都相同时复用，否则重算。第 1 版放在 `viz/<倒奶>_*` 的视频（拟合后参数）会自动移到 `viz/best/`，算作完成。
- 跑完后把 `results_diag21_viz.tar.gz`（视频 + 总览图 + 日志，不含 npz）发回。总表：`results_diag21/viz_summary.md`（每个倒奶一组，起点和拟合后挨着）。
- 任何失败只写一行日志（详细的错误连同 Python 的出错代码行在 `<viz 目录>/<倒奶>.log`），退出码总是 0。

### 时间估计

单个任务（RTX 4090，1280×720、2×2 超采样、30 fps，N 256 正向重跑 + 渲染）：tulip 约 4–5 分钟，leaf 约 3 分钟，heart 约 3 分钟，push_heart、swan 各约 4 分钟（正向重跑每个约 1–1.3 分钟，其余是渲染）。这是按本地 CPU 实测的单帧耗时和 GPU 加速 50–200 倍推算的，不是 4090 上的实测。

| 选择 | 任务数 | 串行 | `VIZ_PAR=2`（默认） |
|---|---|---|---|
| 默认（起点 + 拟合后，全部倒奶） | 24 | 约 85–95 分钟 | 约 1–1.5 小时 |
| `VIZ_POURS="tulip leaf"`（只要标定倒奶） | 12 | 约 40–50 分钟 | 约 25–35 分钟 |
| `VIZ_WHICH=start`（只要起点） | 12 | 约 40–50 分钟 | 约 25–35 分钟 |
| `VIZ_WHICH=start VIZ_POURS="tulip leaf"` | 6 | 约 20–25 分钟 | 约 15 分钟 |
| 默认 + `VIZ_HELD_RECORD=1`（加 v05n_ctrl 的 3 个留出倒奶 × 2 套） | 30 | 约 110–120 分钟 | 约 1.25–1.75 小时 |

`VIZ_MAX_H=4`（小时）是总时限，足够默认的 24 个任务（也够 30 个）；到时限后不再开始新任务，下次运行续跑。

### 选择渲染什么

```bash
NOWAIT=1 VIZ_WHICH=start bash scripts/autodl_viz21.sh            # 只要起点参数（未重新拟合）
NOWAIT=1 VIZ_WHICH=best  bash scripts/autodl_viz21.sh            # 只要拟合后参数
NOWAIT=1 VIZ_MODELS="v05n" VIZ_POURS="tulip leaf" bash scripts/autodl_viz21.sh
# 任意闭包：VIZ_CLOSURES="名字=路径[,入口] ..."（路径相对于目录树；入口 record | law | replay | 0 | 1 | 2）
NOWAIT=1 VIZ_MODELS=none VIZ_CLOSURES="bp256_R1=latte_imex/data/closures/bp256_mi.json,record" bash scripts/autodl_viz21.sh
# 对照模型的留出倒奶（旧入口 R1 + E0，起点和拟合后两套，6 个任务）
NOWAIT=1 VIZ_MODELS=v05n_ctrl VIZ_HELD_RECORD=1 VIZ_POURS="heart push_heart swan" bash scripts/autodl_viz21.sh
# 拟合的桥接闭包（bridge_closures/：mil.json = 拟合后参数 + MI-L，r1.json = 拟合后参数 + 旧入口 R1）
NOWAIT=1 VIZ_MODELS=none VIZ_POURS="tulip leaf" VIZ_CLOSURES="v05n_mil=results_diag21/v05n/bridge_closures/mil.json" bash scripts/autodl_viz21.sh
NOWAIT=1 VIZ_MODELS=none VIZ_HELD_RECORD=1 VIZ_POURS="heart push_heart swan" VIZ_CLOSURES="v05n_r1=results_diag21/v05n/bridge_closures/r1.json" bash scripts/autodl_viz21.sh
```

- `VIZ_CLOSURES` 的输出在 `results_diag21/viz_extra/<名字>/`，字幕写“指定闭包（<文件名>）”。不写入口时按闭包自己的 `"inlet"` 键 → 同目录的 `done.json` → 旧入口 R1 的顺序决定（和拟合时一样）；实测入口（law / replay）的闭包另加三个留出倒奶（MI-L、e0 关）；旧入口（record）的闭包只有加 `VIZ_HELD_RECORD=1` 才渲染留出倒奶（旧入口 R1 + 闭包自己的 E0）。
- `VIZ_MODELS=none`：只渲染 `VIZ_CLOSURES`。

常用开关：`FG=1`（前台）、`VIZ_PAR=2`（同时几个任务）、`VIZ_SIZE=1920x1080`、`VIZ_SS=3`（更细的抗锯齿）、`VIZ_CUP=white`（白色示意杯）、`N=256`、`DEVICE=cuda:0`、`VIZ_JOB_MIN=45`（单个任务的硬时限，分钟，正向重跑 + 渲染合计）、`VIZ_FORCE=1`（忽略 `.done`，全部重渲）、`VIZ_HELD_RECORD=1`（旧入口闭包也渲染留出倒奶）。改这些开关后直接再运行即可，用了别的设置的任务会自动重渲。拟合还在跑时不加 `NOWAIT=1` 启动，它会等拟合结束（最长 `VIZ_WAIT_H=36` 小时；拟合占显存到约 19 GB，两者不同时用 GPU）。本地 CPU 彩排：`LOCAL_SMOKE=1 bash scripts/autodl_viz21.sh`（N 32、每个倒奶 1 秒、320×180、10 fps、tulip + heart，默认 10 个任务，约 10–15 分钟；读 `results_diag21` 里完成的拟合（只复制小文件），输出在 `results_diag21_localsmoke/`）。

## 字幕

- 第一行：模型（说明）· 起点参数（未重新拟合，<起点文件名>）或 拟合后参数（best_closure.json）· 入口（实测入口 MI-R 回放 / MI-L 出口定律 / 旧入口 R1）· N=256（网格只写一次，取自场文件）。留出倒奶在入口后面注明：实测入口模型是“（留出，e0 关，仅报告）”，旧入口闭包（`VIZ_HELD_RECORD=1`）是“（留出，仅报告）”——那时 E0 照常开着，和对照拟合的留出评估一样。
- 第二行：杯子为示意（只测了液面圆盘）· 壶 = CAD + 实测位姿 · 奶量/奶柱 = 倒奶模型 · 拉花 = 求解器白度场 · 奶柱下端被拉到求解器落点的校正量中位数。
- 每个视角左上角：`v05n · 起点参数 · 郁金香 tulip（标定）` 这样的标题，截图时也分得清是哪一套。
- 对照模型 v05n_ctrl（旧入口 R1）的落点校正中位约 8 mm（倒奶开始的前 2 s 约 14 mm，最大约 16 mm），所以它的奶柱看起来是斜的——那正是旧入口的落点修正，不是渲染错误，字幕会注明（字幕和 `viz_summary.md` 里是实测的中位数）。

## 画面里每样东西从哪里来（没有手工动画）

| 东西 | 来源 |
|---|---|
| 壶的位置和姿态 | 实测位姿 v2（`real_program.cup_poses`：壶嘴尖、yaw/tilt/roll，杯坐标系 F），与 `ForwardPitcher` 用的同一个旋转 |
| 壶的形状 | CAD `pitcher_kona700.stl`（壶嘴尖放在实测壶嘴尖上，TIP_KONA） |
| 壶里剩的奶、奶面 | 倒奶模型的账本 `V_P`、`z_f`（壶腔按奶面截取的封闭体；倒奶时奶面高出壶口一个液头 H） |
| 奶柱 | 倒奶模型发射的流体微团（壁面出流 + 毛细飞行 `jet_flight` 的同一算术；旧入口是抛物线），半径 = √(Q / (π·速度))；下端对到求解器当时用的落点 |
| 拉花图案 | 求解器白度场 C（N 256，每 1/30 秒一帧），解场坐标 x = X/L、y = Y/L，L = R_disc / 0.49 |
| 时钟 | 一个时钟：`cup_poses` 的 t（t = 0 为图案开始），位姿、倒奶记录和场都用它 |
| 杯子 | **示意**：只测了液面圆盘（圆心、平面、半径）。示意杯按实测液面半径做成深色马克杯（真实杯子是黑色的），杯沿高度随杯中液面上升而变化 |

## 文件

- `viz/jobs21.py`：任务表（不跑求解器）：找起点闭包并核对、定入口、列出任务、`.done` 标记、旧版目录迁移、`viz_summary.md`。`python viz/jobs21.py start results_diag21/v05n/params v05n --root .` 打印起点闭包是怎么找到的。
- `viz/forward_fields.py`：按拟合时的方式重跑闭包（`diag.render` 的 `render512.make_pour + run_pour`），每 1/30 秒存一帧白度场（uint8），入口的解析与拟合完全一致（闭包的 `inlet` 键 / 同目录 `done.json` / `--inlet`；MI 模型 `LATTE_FOOT=mi`；留出倒奶 MI-L、e0 关）。
- `viz/scene.py`：每一帧的场景（纯 numpy）：壶位姿、奶体网格、奶柱、示意杯、场的插值。
- `viz/raytrace.py`：Warp 光线追踪（GPU 上用 `wp.Mesh` BVH + `mesh_query_ray`）：壶、杯、奶体 + 奶柱、液面（解析平面 + 白度纹理）、桌面；主光 + 补光、阴影光线、超采样、gamma。
- `viz/render3.py`：命令行，输出四段视频 + 总览图 + `<倒奶>_viz.json`。
- `viz/test_viz.py`、`viz/test_scene.py`：单元测试（渲染部分需要本地的 testdata，不随包发送；`python viz/test_viz.py --jobs-only` 只测任务表，不需要 testdata）；`python viz/test_viz.py --device-check cuda:0` 是启动脚本用的 GPU/CPU 一致性检查。
- `viz/assets/pitcher_kona700.stl`、`.json`：CAD 壶（update21 补丁里没有 STL，所以随本包发送）。
- 稠密场存在 `results_diag21_vizfields/<模型>/<start|best>/<倒奶>_fields_N256.npz`（`VIZ_CLOSURES` 的在 `results_diag21_vizfields/extra/<名字>/`），在 results_diag21 之外，不进入任何压缩包。只重渲染（例如换分辨率）：

```bash
python viz/render3.py --pour tulip --fields results_diag21_vizfields/v05n/start/tulip_fields_N256.npz \
    --out results_diag21/v05n/params/viz/start --model v05n --model-desc "v05n 基线" --which start --root .
```
