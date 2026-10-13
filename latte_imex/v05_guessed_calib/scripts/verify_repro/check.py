"""check.py -- compare a fresh run (fresh_run.py) of the frozen calibrated closure with the calibration / evaluate records:
field (float32 as stored, bitwise), mask, ledger / stats, inlet log summary; L_fin by the driver's own lfin (no video
cache) and by the independent implementation indep.py; for N = 256 also the eval.json table numbers (compare.metrics /
compare.best_align, recomputed independently).  Differences > TOL between identical runs are flagged.

    python3 check.py <pattern> <N> OUT.json
"""
import json
import os
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, "/home/user/zfChen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                            # noqa: E402
from latte_imex import v05_guessed_calib as G                # noqa: E402
from latte_imex import v05_calibrate as VC                   # noqa: E402
import indep as I                                             # noqa: E402

ROOT = os.path.join(os.environ["V05GCAL_REVIEW"], "v05gcal")   # review dir (copy: path from the environment)
TOL = 1e-6
FLAGS = []


def cmp(tag, a, b, tol=TOL):
    if isinstance(a, (list, tuple)):
        ok = len(a) == len(b) and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))
        d = max([abs(float(x) - float(y)) for x, y in zip(a, b)] or [0.0])
    else:
        d = abs(float(a) - float(b))
        ok = d <= tol
    if not ok:
        FLAGS.append(dict(tag=tag, ours=a, ref=b, diff=d))
    return dict(ours=a, ref=b, absdiff=d, ok=bool(ok))


def main():
    p, N, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    pre = json.load(open(os.path.join(ROOT, "prereg.json")))
    rec = json.load(open(os.path.join(ROOT, "calib", "calib.json")))
    params = {k: float(rec["best"]["params"][k]) for k in G.CLOSURE8}
    key = VC.cache_key(params, p, N, pre["spec"])
    cr = json.load(open(os.path.join(ROOT, "cache", f"{key}.json")))
    ev = cr["eval"]
    z = np.load(os.path.join(HERE, "runs", f"{p}_N{N}.npz"))
    meta = json.load(open(os.path.join(HERE, "runs", f"{p}_N{N}.json")))
    c, m = z["c"], z["mask"]
    zc = np.load(os.path.join(ROOT, "cache", f"{key}.npz"))
    res = dict(pattern=p, N=N, cache_key=key, cache_tag=cr["tag"], fresh_wall_s=meta["wall_s"], cache_wall_s=cr["wall_s"])
    # ---- field / mask / ledger / inlet
    c32 = np.where(m, c, 0.0).astype(np.float32)
    ref32 = np.where(zc["mask"], zc["c"], 0.0).astype(np.float32)
    res["field"] = dict(mask_equal=bool(np.array_equal(m, zc["mask"])), bitwise_equal_f32=bool(np.array_equal(c32, ref32)),
                        max_abs_diff=float(np.max(np.abs(c32.astype(np.float64) - ref32))),
                        finite=bool(np.all(np.isfinite(c[m]))))
    if N == 256:
        e = np.load(os.path.join(ROOT, "evaluate", f"best_{p}_N256.npy"))
        res["field"]["evaluate_npy_bitwise_equal"] = bool(np.array_equal(np.where(m, c, np.nan).astype(np.float32), e,
                                                                          equal_nan=True))
    res["ledger"] = {k: cmp(f"ledger.{k}", meta["ledger"][k], ev["ledger"][k], 0.0)
                     for k in ("layer_error", "milk_error", "deposited")}
    res["ledger"]["T"] = cmp("ledger.T", meta["t_end"], ev["ledger"]["T"], 0.0)
    res["stats"] = {k: dict(ours=meta["stats"][k], ref=ev["stats"][k], ok=meta["stats"][k] == ev["stats"][k])
                    for k in ev["stats"]}
    res["inlet_equal"] = meta["inlet"] == ev["inlet"]
    res["solver_params_equal"] = meta["solver_params"] == {**ev["solver_params"], "closure": "chi", "return_law": "v05"} or \
        all(meta["solver_params"][k] == ev["solver_params"][k] for k in G.CLOSURE_KEYS)
    res["solver_closure_return"] = [meta["solver_params"].get("closure"), meta["solver_params"].get("return_law")]
    res["kQ_equal"] = meta["kQ"] == ev["kQ"]
    # ---- driver's own loss, fresh process, no video cache
    lf = G.lfin(c, m, p, vdir=None)
    res["driver_lfin"] = dict(loss=cmp("driver.loss", lf["loss"], ev["loss"], 0.0),
                              transform_equal=lf["transform"] == ev["transform"],
                              n_model_equal=lf["aligned"]["n_model"] == ev["n_model"])
    # ---- independent implementation
    ind = I.evaluate(c, m, p, compare_align=(N == 256))
    L = ind["lfin"]
    t = ev["transform"]
    res["indep_lfin"] = dict(
        L_fin=cmp("indep.L_fin", L["L"], ev["loss"]), dice=cmp("indep.dice", L["dice"], ev["dice"]),
        iou=cmp("indep.iou", L["iou"], ev["iou"]), bnd_D=cmp("indep.bnd_D", L["bnd_D"], ev["bnd_D"]),
        S=cmp("indep.S", L["S"], ev["S"]), n_model=dict(ours=L["n_model"], ref=ev["n_model"], ok=L["n_model"] == ev["n_model"]),
        n_video=dict(ours=ind["video"]["n"], ref=ev["n_video"], ok=ind["video"]["n"] == ev["n_video"]),
        transform=dict(ours=L["transform"], ref=t, ok=(L["transform"]["theta_deg"], L["transform"]["dx_D"],
                                                        L["transform"]["dy_D"]) == (t["theta_deg"], t["dx_D"], t["dy_D"])),
        white_frac=cmp("indep.white_frac", L["white_frac"], ev["white_frac"]),
        video_white_frac=cmp("indep.video_white_frac", ind["video"]["white_frac"], ev["video_white_frac"]))
    rw = ev["raw"]
    res["indep_raw"] = dict(L=cmp("indep.raw.L", ind["raw"]["L"], rw["L"]), iou=cmp("indep.raw.iou", ind["raw"]["iou"], rw["iou"]),
                            dice=cmp("indep.raw.dice", ind["raw"]["dice"], rw["dice"]),
                            bnd_D=cmp("indep.raw.bnd_D", ind["raw"]["bnd_D"], rw["bnd_D"]),
                            S=cmp("indep.raw.S", ind["raw"]["S"], rw["S"]),
                            n_model=dict(ours=ind["raw"]["n_model"], ref=rw["n_model"], ok=ind["raw"]["n_model"] == rw["n_model"]))
    # ---- calibration loss of the best sample (tulip + leaf at the search N): our tulip + the recorded leaf
    if N == int(rec["N"]) and p in ("tulip", "leaf"):
        other = "leaf" if p == "tulip" else "tulip"
        ko = VC.cache_key(params, other, N, pre["spec"])
        eo = json.load(open(os.path.join(ROOT, "cache", f"{ko}.json")))["eval"]
        res["calib_loss"] = cmp("calib.loss", 0.5 * (L["L"] + eo["loss"]), rec["best"]["loss"])
        res["calib_loss"]["other_pattern_recorded"] = {other: eo["loss"]}
    # ---- N = 256: the eval.json table numbers (compare.py definitions), recomputed independently
    if N == 256:
        E = json.load(open(os.path.join(ROOT, "eval", "eval.json")))["N256"][p]["CAL"]
        er, eb, el = E["compare"]["raw"], E["compare"]["best"], E["lfin"]
        b = ind["best"]
        res["eval_table"] = dict(
            iou_raw_c04=cmp("eval.iou_raw_c04", ind["raw"]["iou"], er["iou_c04"]),
            iou_raw_c05=cmp("eval.iou_raw_c05", ind["raw"]["iou_c05"], er["iou_c05"]),
            iou_best_c04=cmp("eval.iou_best_c04", b["iou"], eb["iou_c04"]),
            iou_best_c05=cmp("eval.iou_best_c05", b["iou_c05"], eb["iou_c05"]),
            bnd_raw_mm=cmp("eval.bnd_raw_mm", ind["raw"]["bnd_mm"], er["bdist_c04_mm"]),
            bnd_best_mm=cmp("eval.bnd_best_mm", b["bnd_mm"], eb["bdist_c04_mm"]),
            white_ratio_raw=cmp("eval.white_ratio_raw", ind["raw"]["white_ratio"], er["white_ratio"]),
            white_ratio_best=cmp("eval.white_ratio_best", b["white_ratio"], eb["white_ratio"]),
            best_transform=dict(ours=b["transform"], ref=eb["transform"],
                                ok=(b["transform"]["theta_deg"], b["transform"]["dx_D"], b["transform"]["dy_D"]) ==
                                   (eb["transform"]["theta_deg"], eb["transform"]["dx_D"], eb["transform"]["dy_D"])),
            seams_best=dict(ours=b["n_model"], ref=eb["seams"], ok=b["n_model"] == eb["seams"]),
            seams_lfin=dict(ours=L["n_model"], ref=el["seams_model"], ok=L["n_model"] == el["seams_model"]),
            L_fin=cmp("eval.L_fin", L["L"], el["L_fin"]),
            bnd_lfin_mm=cmp("eval.bnd_lfin_mm", L["bnd_mm"], el["bnd_mm"]))
    res["flags"] = FLAGS
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    print(json.dumps(res, default=float, indent=1))


if __name__ == "__main__":
    main()
