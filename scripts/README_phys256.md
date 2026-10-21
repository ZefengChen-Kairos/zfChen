# update19：物理版模型在 N=256 上的对比（不训练，只跑和出视频）

## 跑什么
在 N=256（每格约 0.33 mm）上跑三组参数、五杯拉花，每组都和上次的"反传训练 N=256"结果并排出视频：

1. **物理版 第一阶段**（`mcd_S1_N96.json`）
   - 新浮上来的奶泡不带奶流的水平速度；
   - 奶泡在落点前方 0.075 秒处浮出（实测）；
   - 奶泡粘度取物理值 0.01；
   - 用更锐利的输运格式；
   - 删掉径向推力和表面流。
2. **物理版 第二阶段**（`mcd_S2_N96.json`）：第一阶段，再加上奶流扎下去时把落点附近的表面吸进液体。
3. **新基线参数，未训练**（`v05n`）：用来看分辨率本身带来的变化。

这一轮不训练，所有参数固定。物理版的参数都有独立估计，每一项的依据写在 json 的 `_note` 里。

## 运行
把 `phys256_update19.tar.gz` 上传到 `/root/autodl-tmp/`，然后：
```bash
mkdir -p /root/autodl-tmp/u19 && tar xzf /root/autodl-tmp/phys256_update19.tar.gz -C /root/autodl-tmp/u19 \
  && cd /root/autodl-tmp/u19 && bash scripts/autodl_phys256.sh
```
- 看进度：`tail -f /root/autodl-tmp/u19/results_phys/phys.log`
  - 每杯先跑一次参考结果（反传训练 N=256），再依次跑三组参数；
  - 每组算完会打印一行分数 J。
- 预计 40–70 分钟。中断后重新运行同一条命令即可，已完成的杯会跳过。
- 跑完把 `/root/autodl-tmp/u19/results_phys256.tar.gz` 发给我，约 60–80 MB。如果太大，只发：
  - `results_phys/summary.txt`
  - 三个 `*/tulip_compare.mp4`
  - 所有 `*_c128.npz`（我用它们在本地计算视频运动学检验）

## 输出
`results_phys/<组名>/` 下：
- `<杯>_compare.mp4`：四联视频，依次是真实视频、视频白度目标、这一组、反传训练 N=256。
- `<杯>_sim256.mp4`：这一组本身，256×256，一格一个像素。
- `<杯>_c128.npz`：每个观测帧的白色比例（128 px），用来在本地做运动学检验。
- `eval.json`：五杯的 J 分数（换算到 N=48 后的标定目标）。

`results_phys/summary.txt` 是三组五杯分数的汇总表。

## 先在 CPU 上自检（可选，几分钟）
```bash
SMOKE=1 bash scripts/autodl_phys256.sh
```
