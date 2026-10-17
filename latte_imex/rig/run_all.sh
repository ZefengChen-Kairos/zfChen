#!/bin/bash
set -e
cd "$(dirname "$0")"
for p in "$@"; do
  echo "== $p $(date +%T)"
  python3 -I run_ba.py $p
  python3 -I pitcher_model.py $p
  TIP=0,53,93 python3 -I tip_track.py $p
  python3 -I cup_track.py $p
  python3 -I topview.py $p 2>/dev/null
  python3 -I disc_fit.py $p
  python3 -I action_summary.py $p 2>/dev/null
  echo "== done $p $(date +%T)"
done
