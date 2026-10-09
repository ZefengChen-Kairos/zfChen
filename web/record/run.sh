#!/bin/bash
# usage: run.sh <script> <grid> <outname> [vw vh extra]
set -e; cd "$(dirname "$0")"
S=$1; G=$2; O=$3; VW=${4:-960}; VH=${5:-680}; EX=${6:-4}
rm -rf "rec_$O"; mkdir -p "rec_$O"
timeout 2400 node rec.mjs http://localhost:8765/local.html "$S" "$G" "$PWD/rec_$O" 2000 "$VW" "$VH" "$EX" | tail -4
python3 - "$O" <<'PY'
import json, sys, subprocess
o=sys.argv[1]; j=json.load(open(f"rec_{o}/box.json")); b=j["box"]; r=j["realPerModel"] or 1.0
f=max(1.0, r)  # speed factor -> video plays at model time
cmd=["ffmpeg","-y","-loglevel","error","-i",j["webm"],"-vf",f"crop={b['w']}:{b['h']}:{b['x']}:{b['y']},setpts=PTS/{f:.4f}","-r","25","-an","-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",f"rec_{o}/{o}.mp4"]
subprocess.run(cmd, check=True); print("speed factor", round(f,2), "->", f"rec_{o}/{o}.mp4")
PY
ffprobe -v error -show_entries format=duration -of csv=p=0 "rec_$O/$O.mp4"
