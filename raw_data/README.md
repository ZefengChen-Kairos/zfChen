# 实验数据放这里

| 目录 | 放什么 |
|---|---|
| `pitcher/` | 拉花缸尺寸（`pitcher_dims.yaml`）、起流实验（`onset.csv`）、液面高度（`level.csv`）、照片（侧面/正上方，带尺子） |
| `rig/` | 三相机标定文件（内参、外参，原格式即可）、相机型号说明 `rig.md`、静物场景的短录制或导出的帧 |
| `pours/<日期_图案_序号>/` | 倒奶实验：三相机录制（或导出的帧）、电子秤记录、`notes.md`（装量、对时方式、谁倒的） |

单个文件 ≤ 100 MB（网页上传 ≤ 25 MB）。`.bag` 一类大文件先截成几秒，或导出成 png 帧（彩色 jpg、深度 16 位 png）再放。

## 已裁剪的拉花实拍数据

五个图案（heart、push_heart、leaf、swan、tulip）只保留出图案阶段，放在单独的数据分支 `pours-trimmed`（约 1.3 GB），
不在本分支里。只取数据：

    git clone --single-branch -b pours-trimmed https://github.com/ZefengChen-Kairos/zfChen.git latte_pours

原始完整录制在 Release `raw-pours-v1`。剪切点和格式说明见数据分支的 README。
