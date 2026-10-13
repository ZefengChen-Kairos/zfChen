"""Reproduce the figures of 'V0.5 with the barista's real actions' (REAL_ACTIONS_RESULT.md) in this folder.

    python latte_imex/v05_real_actions/run.py --out WORKDIR [--procs 3] [--copy] [--review REVIEW_DIR]

Steps (each run is skipped when WORKDIR/runs/<tag>_<name>_N<N>/eval.json exists; all through latte_imex.v05_real_run):
  1  unit tests of the adapter (python -m latte_imex.v05_adapter test)
  2  closure sets FS and FSp2 (unchanged, zero tuning), six patterns, N = 128 and N = 256 (full runs: c at the clip's
     5 fps analysis frames and 15 fps video frames; calib evaluation)              ~1 min / 10-13 min per run
  3  u_exit 0.10 and 0.27 m/s (FS, push_heart and tulip, N = 128)
  4  DIAGNOSTIC runs (not fits): chi045, foot05, chi045foot05 (FS, FSp2, N = 128; FSp2 also N = 256 on the four
     wiggle patterns); wanted flow x0.5 and x1.5 (FS, FSp2, N = 128)
  5  figures: final_<name>.png and final_vs_real.png (real photo | real whiteness | model FSp2 | model FS, 256^2),
     process_<name>_<set>.png (calib contact sheets, physical top view), diag_<set>.png
  6  videos/<name>.mp4 (FSp2, N = 128, k = 3)
  7  with --review: codex_vs_real_actions.png (needs the review folder with v05real/compare.py, the Codex-action runs
     of v05allphys/final and v05video/codexcmp.py, figcodex.py)
  --copy copies the figures (< 1 MB each) into this folder.
At most --procs worker processes (default 3).  Nothing in the solver is changed.
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
FIGS = (["final_vs_real.png"] + [f"final_{n}.png" for n in ("heart", "push_heart", "layered_heart", "tulip", "leaf", "swan")]
        + ["diag_FS.png", "diag_FSp2.png", "diag_FSp2_N256.png", "codex_vs_real_actions.png"])


def rr(*args):
    cmd = [PY, "-m", "latte_imex.v05_real_run", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True, env=dict(os.environ, NUMBA_NUM_THREADS="1"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", default="3")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--skip_tests", action="store_true")
    ap.add_argument("--no_videos", action="store_true")
    ap.add_argument("--review", default=None)
    a = ap.parse_args()
    o, P = os.path.abspath(a.out), a.procs
    if not a.skip_tests:
        subprocess.run([PY, "-m", "latte_imex.v05_adapter", "test"], cwd=REPO, check=True)
    for N in ("128", "256"):
        rr("run", "--sets", "FS,FSp2", "--N", N, "--procs", P, "--out", o, "--skip_done")
    rr("run", "--sets", "FS", "--u_exit", "0.10,0.27", "--names", "push_heart,tulip", "--procs", P, "--out", o, "--skip_done")
    rr("run", "--sets", "FS,FSp2", "--diag", "chi045,foot05,chi045foot05", "--procs", P, "--out", o, "--skip_done")
    rr("run", "--sets", "FS,FSp2", "--q_scale", "0.5", "--procs", P, "--out", o, "--skip_done")
    rr("run", "--sets", "FS,FSp2", "--q_scale", "1.5", "--procs", P, "--out", o, "--skip_done")
    rr("run", "--sets", "FSp2", "--N", "256", "--diag", "chi045,foot05,chi045foot05", "--names",
       "layered_heart,tulip,leaf,swan", "--procs", P, "--out", o, "--skip_done")
    rr("figs", "--best", "FSp2", "--other", "FS", "--N_hi", "256", "--out", o)
    rr("diagfig", "--sets", "FS,FSp2", "--out", o)
    rr("diagfig", "--sets", "FSp2", "--N", "256", "--out", o)
    if not a.no_videos:
        rr("videos", "--sets", "FSp2", "--N", "128", "--k", "3", "--out", o)
    if a.review:
        v = os.path.join(os.path.abspath(a.review), "v05video")
        for args in (["codexcmp.py"], ["codexcmp.py", "merge"], ["figcodex.py"]):
            subprocess.run([PY, "-I", os.path.join(v, args[0]), *args[1:]], check=True)
        shutil.copy(os.path.join(v, "codex_vs_real_actions.png"), os.path.join(o, "codex_vs_real_actions.png"))
    if a.copy:
        for f in FIGS:
            src = os.path.join(o, f)
            if os.path.exists(src):
                assert os.path.getsize(src) < 1_000_000, (f, os.path.getsize(src))
                shutil.copy(src, os.path.join(HERE, f))
                print("copied", f, os.path.getsize(src), flush=True)


if __name__ == "__main__":
    main()
