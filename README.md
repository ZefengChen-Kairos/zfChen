# 拉花实拍数据：只保留出图案阶段

来源：Release `raw-pours-v1`（三台 RealSense D435/D435i，848×480，实际约 16 帧/秒）。
每段剪掉了开头的高位融合段和结尾的收壶、展示段，只保留**出图案阶段**。

## 剪切点

时间是相对于该段录制第一帧的秒数（`metadata.json → cut` 里另有绝对时间戳）。

| 图案 | 开始 | 结束 | 时长 | 每台相机帧数 | 开始依据 | 结束依据 |
|---|---|---|---|---|---|---|
| heart 爱心 | 10.4 | 18.4 | 8.0 s | 127 | 白色在 10.8 s 首次出现，提前 0.4 s | 18.4 s 奶柱断开、开始收壶 |
| push_heart 推推乐 | 12.6 | 24.0 | 11.4 s | 175 | 白色在 13.0 s 首次出现，提前 0.4 s | 24.0 s 奶柱断开 |
| leaf 树叶 | 11.0 | 20.0 | 9.0 s | 135–136 | 白色在 11.4 s 首次出现，提前 0.4 s | 20.0 s 奶柱断开 |
| swan 天鹅 | 10.4 | 22.0 | 11.6 s | 197–198 | 白色在 10.8 s 首次出现，提前 0.4 s | 22.0 s 奶柱断开 |
| tulip 郁金香 | 98.8 | 112.2 | 13.4 s | 190 | 白色在 99.2 s 首次出现，提前 0.4 s | 112.2 s 奶柱断开 |

开始点统一取"杯面第一次出现白色"之前 0.4 s，这样能包含壶压低、贴近液面的过程。
`previews/` 里有每个切点前后 2.4 s 的抽帧图（cam1，每 0.2 s 一张），可以直接核对。
注：郁金香原始录制的前约 87 s 是空镜头。

## 目录

```
<图案>/
  color/<相机>/<帧号>.webp   彩色，无损 WebP（与原 PNG 逐像素相同），BGR
  depth/<相机>/<帧号>.png    深度，16 位 PNG，单位 mm，已对齐到彩色
  metadata.json              内参、序列号、每帧时间戳（只含保留的帧）、cut 信息
  calibrate.pkl              三台相机到标定板的 4×4 位姿（原文件）
previews/                    切点抽帧图
```

## 读取

```python
import cv2, json
m = json.load(open("heart/metadata.json"))
color = cv2.imread("heart/color/0/1234.webp")                  # BGR uint8
depth = cv2.imread("heart/depth/0/1234.png", cv2.IMREAD_UNCHANGED)  # uint16, mm
t = m["recording"]["0"]["1234"]                                # 时间戳 (s)
```

`calibrate.pkl` 里只有 numpy 数组。读取工具见代码分支的 `latte_imex/rig/`（`safepkl.py` 只允许读 numpy 数组）。
