#!/bin/bash
# Everything on one AutoDL GPU machine, in the background (survives closing the browser / JupyterLab):
#   bash scripts/autodl_all.sh            -> starts, prints where the log is, returns at once
#   tail -f results/autodl_all.log        -> watch progress
# Re-running the same command after an interruption resumes (finished stages are skipped; the design step resumes from its checkpoints).
# Order: install + GPU self-tests -> aligned model (6 patterns, videos, hand-error grid) -> closure calibration ->
# action re-design -> robustness of the new actions -> final videos -> one archive results/autodl_all_results.tar.gz
cd "$(dirname "$0")/.."
LOG=results/autodl_all.log
mkdir -p results
if [ "${_LATTE_BG:-}" != "1" ]; then
  _LATTE_BG=1 nohup bash "$0" "$@" >> "$LOG" 2>&1 &
  echo "started in the background (pid $!); log: $(pwd)/$LOG"
  echo "watch: tail -f $(pwd)/$LOG"
  exit 0
fi
set -eo pipefail
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
stage() { echo; echo "===== $(date '+%F %T') $1 ====="; }
done_mark() { touch "results/.done_$1"; }
is_done() { [ -f "results/.done_$1" ]; }

if ! is_done setup; then stage "1/4 install + GPU self-tests"; bash scripts/autodl_setup.sh; done_mark setup; fi
if ! is_done aligned; then stage "2/4 aligned model: six patterns + videos + hand-error grid"; bash scripts/autodl_run_v05_aligned.sh; done_mark aligned; fi
stage "3/4 calibration -> action re-design -> robustness -> final videos"
bash scripts/autodl_run_design.sh
done_mark design
stage "4/4 packing results"
tar -czf results/autodl_all_results.tar.gz --exclude='*.npz' results/autodl_selftest results/autodl_v05 results/autodl_design results/autodl_all.log
ls -lh results/autodl_all_results.tar.gz
stage "ALL DONE"
