"""push_heart written from the reference video: three stages, each starts at B and pushes forward, each closed by its
own lift + flow cut (no through-cut at the end)."""
import json, os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latte_imex import action_opt as ao
from latte_imex.pitcher import v05_moves
from latte_imex.joint_cma import BASE_V2
proto = v05_moves("push_heart")
pour, repo, cut, stop, away = proto[0], proto[1], proto[6], proto[7], proto[8]
def mv(base, **kw):
    m = copy.deepcopy(base)
    for k, v in kw.items(): setattr(m, k, v)
    return m
B, C, D = 0.22, 0.05, -0.12
L = 1e-6
def video_moves(name, h=1.0, q=1.0, s=1.0):
    seq = [mv(pour, dur=1.6, p0=(0, B), p1=(0, D), z0=0.02, z1=0.02, Q0=15 * L, Q1=15 * L, name="stage1 B->D"),
           mv(cut, dur=0.4, p0=(0, D), p1=(0, D - 0.06), z0=0.06, z1=0.07, Q0=4 * L, Q1=1 * L, name="lift 1"),
           mv(repo, dur=0.35, p0=(0, D - 0.06), p1=(0, B), name="back to B"),
           mv(pour, dur=1.4, p0=(0, B), p1=(0, C), z0=0.018, z1=0.018, Q0=22 * L, Q1=22 * L, name="stage2 B->C big flow"),
           mv(cut, dur=0.4, p0=(0, C), p1=(0, C - 0.06), z0=0.06, z1=0.07, Q0=4 * L, Q1=1 * L, name="lift 2"),
           mv(repo, dur=0.35, p0=(0, C - 0.06), p1=(0, B), name="back to B"),
           mv(pour, dur=0.9, p0=(0, B), p1=(0, B - 0.06), z0=0.02, z1=0.02, Q0=15 * L, Q1=15 * L, name="stage3 at B"),
           mv(cut, dur=0.4, p0=(0, B - 0.06), p1=(0, B - 0.12), z0=0.06, z1=0.07, Q0=4 * L, Q1=1 * L, name="lift 3"),
           mv(stop, p0=(0, B - 0.12), p1=(0, B - 0.12)), mv(away, p0=(0, B - 0.12), p1=(0, -1.8))]
    for m in seq[:-1]:
        m.p0 = (m.p0[0], m.p0[1] * s); m.p1 = (m.p1[0], m.p1[1] * s)
        m.z0 = min(max(m.z0 * h, 0.008), 0.05) if m.z0 < 0.05 else m.z0; m.z1 = min(max(m.z1 * h, 0.008), 0.05) if m.z1 < 0.05 else m.z1
        m.Q0 *= q; m.Q1 *= q
    return seq
ao.scaled_moves = video_moves
H = json.load(open("runs/action_opt/joint_cma_v2s/history.json"))["best"]["best"]
ao.MODELS["ph3_old"] = {}
ao.MODELS["ph3_v2s"] = dict(BASE_V2, **H["phys"])
which = sys.argv[1]
for h in (1.0, 0.75):
    ao.run_one((which, "push_heart", h, 1.0, 1.0, 96, "runs/action_opt"))
