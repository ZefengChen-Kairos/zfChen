"""viz29: tests of scene.py on forward_fields outputs (viz/testdata, tulip N 48: MI-R s1kq whole pour, record control
0-4 s, record + diag20 oracle; heart MI-L inlet-only).  python test_scene.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scene as S  # noqa: E402

TD = os.environ.get("VIZ_TESTDATA") or os.path.join(HERE, "testdata")


def run(fn, snap_p90_max, times=(1.0, 2.5, 3.9)):
    r = S.check(os.path.join(TD, fn), times=times)
    assert r["stl_tip_anchor_err_m"] < 1e-9, r
    for t, led, mesh, k in r["milk_mL_ledger_vs_mesh_capk"]:
        assert abs(mesh - led) / led < 0.01, (fn, t, led, mesh)
    assert r["jet_xhit_vs_model_max_mm"] < 1e-6, r
    assert r.get("jet_tf_vs_law_max_s", r.get("jet_tf_vs_model_max_s")) < 1e-9, r
    assert r["field_blend_exact"], r
    assert r["snap_mm_p50_p90_max"][1] < snap_p90_max, (fn, r["snap_mm_p50_p90_max"])
    print(f"ok {fn}: snap p50/p90/max {np.round(r['snap_mm_p50_p90_max'], 2).tolist()} mm, "
          f"{r['frames_with_stream']}/{r['frames']} frames with a stream")
    return r


def frame_shapes():
    sc = S.Scene(os.path.join(TD, "ff_s1kq_tulip_N48.npz"))
    f = sc.frame(0.9)
    V, F = sc.stl()
    X = V @ f["pitcher"]["R"].T + f["pitcher"]["t"]
    assert X[:, 2].min() > -0.02, "pitcher far below the liquid surface"
    assert f["milk"]["cap"] is not None and np.allclose(f["milk"]["cap"][:, 2], f["milk"]["z_f"], atol=1e-9)
    seg = f["stream"]["segments"]
    assert seg and np.allclose(seg[0]["P"][-1, :2], sc.solver_foot(0.9)[:2], atol=1e-9) and abs(seg[0]["P"][-1, 2]) < 1e-12
    assert np.linalg.norm(seg[0]["P"][0] - f["pitcher"]["tip"]) < 0.02, "stream top far from the spout tip"
    assert f["field"]["C"].shape == (48, 48) and 0 <= f["field"]["C"].min() and f["field"]["C"].max() <= 1
    Vm, Fm, cap = sc.milk_mesh(0.9, inset=2e-4)
    assert abs(S.mesh_volume(Vm, Fm) / f["milk"]["V_P"] - 1) < 0.03
    sec = f["milk"]["section"]
    assert len(sec) >= 4 and sec[:, 2].max() <= f["milk"]["z_f"] + 1e-12
    print("ok frame(0.9): pitcher, milk cap at z_f, stream foot on the solver landing, field, inset mesh, section")


if __name__ == "__main__":
    run("ff_s1kq_tulip_N48.npz", 5.0)
    # record inlet: the E0 landing correction of the lip exit, median ~8 mm over the whole tulip, ~14 mm in its first
    # 2 s, max ~16 mm (this file holds the first 4 s, hence the 20 mm bound)
    run("ff_ctrl_tulip_N48_T4.npz", 20.0)
    run("ff_oracle20_tulip_N48.npz", 25.0)
    Z = dict(np.load(os.path.join(TD, "inl_v05n_heart_N48.npz")))
    Z["C"] = np.zeros((2, 48, 48), np.uint8); Z["t"] = np.array([0.0, 7.9])
    np.savez(os.path.join(TD, "_heart_tmp.npz"), **Z)
    try:
        run("_heart_tmp.npz", 1.5, times=(1.0, 3.0, 6.0))   # MI-L: the landing is the law's
    finally:
        os.remove(os.path.join(TD, "_heart_tmp.npz"))
    frame_shapes()
