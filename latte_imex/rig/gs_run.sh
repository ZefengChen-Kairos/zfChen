#!/bin/bash
# 3D Gaussian splatting on the AutoDL GPU (background; re-running skips finished objects):
#   bash run.sh            -> log: gs.log ; results: out/<object>/{point_cloud.ply, turntable.mp4, views.jpg, metrics.json}
#   tail -f gs.log
cd "$(dirname "$0")"
if [ "${_GS_BG:-}" != "1" ]; then
  _GS_BG=1 nohup bash "$0" "$@" >> gs.log 2>&1 &
  echo "started in the background (pid $!); watch: tail -f $(pwd)/gs.log"; exit 0
fi
set -eo pipefail
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
echo "===== $(date '+%F %T') setup ====="
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || python -m pip install -q "torch==2.4.1"
python -m pip install -q ninja imageio imageio-ffmpeg scipy "numpy<2" "opencv-python-headless==4.10.0.84"
if ! python -c "import gsplat" 2>/dev/null; then
  TAG=$(python -c "import torch;v=torch.__version__.split('+')[0].split('.');c=(torch.version.cuda or '').replace('.','');print(f'pt{v[0]}{v[1]}cu{c}')")
  echo "gsplat wheel index: $TAG"
  python -m pip install -q gsplat --index-url "https://docs.gsplat.studio/whl/$TAG" || python -m pip install -q gsplat
fi
python - <<'PY'
import torch, gsplat
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0), "gsplat", gsplat.__version__)
from gsplat import rasterization                       # first call compiles the CUDA kernels if no wheel matched
m = torch.zeros(1, 3, device="cuda"); m[0, 2] = 1
out, a, _ = rasterization(m, torch.tensor([[1., 0, 0, 0]], device="cuda"), torch.full((1, 3), 0.1, device="cuda"),
                          torch.ones(1, device="cuda"), torch.ones(1, 3, device="cuda"), torch.eye(4, device="cuda")[None],
                          torch.tensor([[[100., 0, 32], [0, 100., 32], [0, 0, 1]]], device="cuda"), 64, 64)
print("gsplat self-test ok, alpha max", float(a.max()))
PY
for obj in coffee_cup milk_cup pitcher_pours; do
  [ -d "data/$obj" ] || continue
  if [ -f "out/$obj/metrics.json" ]; then echo "$obj done, skipping"; continue; fi
  IT=15000; [ "$obj" = pitcher_pours ] && IT=30000
  case $obj in                                             # object volume rmax,zmin,zmax [m] in the object frame
    coffee_cup) VOL=0.075,0.002,0.075 ;;
    milk_cup) VOL=0.090,0.002,0.135 ;;
    pitcher_pours) VOL=0.090,-0.036,0.098 ;;               # pitcher tag frame: bottom -33 mm, rim +93 mm
  esac
  echo "===== $(date '+%F %T') $obj ($IT iterations) ====="
  python gs_train.py --data "data/$obj" --out "out/$obj" --iters $IT --volume $VOL
done
tar -czf gs_results.tar.gz out
ls -lh gs_results.tar.gz
echo "===== $(date '+%F %T') DONE ====="
