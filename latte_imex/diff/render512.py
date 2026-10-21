"""Evaluation and movies of a closure run at a fine grid (diff/fit.py trained at N = 256 / 512), on the calibration's
own scale: the objective of the N = 48 calibration is recomputed from the fine run.

    python3 -m latte_imex.diff.render512 --closure C.json --N 512 --out DIR [--pours tulip,leaf,...] [--label TEXT]
        [--baseline B.json] [--device cuda:0] [--every 4] [--T seconds] [--no-movie]
-> DIR/eval.json: per pour J = 0.25 viou_n + 0.75 struct_n (cmafit.objective) and its parts, computed at the N = 48
   equivalent: the fine run's m and l summed conservatively onto the 48 x 48 cells (exact 1-D overlap weights; both
   grids are cell-centred on the same box [-0.5, 0.5]^2), c = sum m / sum l, then scored exactly as cmafit.evaluate scores
   an N = 48 run (score.set_grid(48), the N = 48 video targets of run.video_on_grid, the video's own 0.25 s lag
   reference).  At N = 48 the remap is the identity, so a closure scores what cmafit gave it.
   DIR/<p>_simN.mp4: the fine field, one pixel per cell (N x N), latte palette; DIR/<p>_compare.mp4: real top view |
   video white target | the closure at N | the baseline closure at N = 48, on one clock (30 fps).
Forward run as the calibration's (cmafit._init: tau 0.05, scales (0.3, 0.15, 0), e0 / exit_cal / upwell / circ / milk
switches from the closure json, parameters = DiffPour.s1 + surface-current defaults + the closure's numbers)."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import schedule as SC
from . import score as SS

PATTERNS = ("tulip", "leaf", "heart", "push_heart", "swan")
CUP_R, EXT, FPS = 0.49, 1.05, 30
CREMA = np.array([150, 96, 62], float)       # [RGB] latte palette: crema / brown
FOAM = np.array([246, 243, 236], float)      #       milk foam / white
CUP = np.array([28, 24, 26], float)          #       black mug outside the disc
HIDDEN = np.array([70, 52, 78], float)       #       not observed in the video (pitcher / stream / outside the views)
NAMES = dict(tulip="郁金香 tulip（标定）", leaf="树叶 leaf（标定）", heart="爱心 heart（留出）",
             push_heart="推推乐 push_heart（留出）", swan="天鹅 swan（留出）")
NAMES_EN = dict(tulip="tulip (calibration)", leaf="leaf (calibration)", heart="heart (held out)",
                push_heart="push_heart (held out)", swan="swan (held out)")


def load_closure(fn):
    """(numbers, switches) of a closure json (fit.py / cmafit best_closure.json or latte_imex/data/closures/*.json)."""
    from .wsolver import milk_code
    J = json.load(open(fn))
    num = {k: float(v) for k, v in J.items() if isinstance(v, (int, float)) and not k.startswith("_")}
    sw = dict(e0=SC.e0_mode(num.get("e0", 0.0)), exit_cal=bool(num.get("exit_cal", 0.0)), upwell=bool(num.get("upwell", 0.0)),
              circ=bool(num.get("circ", 0.0)), circ_shape=dict(L=num.get("circ_L", 0.4), r_f=num.get("circ_rf", 0.06),
                                                               r=num.get("circ_r", 1.0)),
              milk=milk_code(num.get("milk", 0.0)))
    return num, sw


def make_pour(p, sw, N, device, T=None):
    from .run import DiffPour
    return DiffPour(p, N=N, device=device, T_max=T, tau=0.05, scales=(0.3, 0.15, 0.0), e0=sw["e0"], exit_cal=sw["exit_cal"],
                    upwell=sw["upwell"], circ=sw["circ"], circ_shape=sw["circ_shape"], milk=sw.get("milk", 0),
                    log=lambda *a: None)


def run_pour(D, num, sw, every):
    """Forward run of DiffPour D with the closure; snapshots of every `every`-th frame and of the observed frames.
    Returns t_end, {k: c (float32)}, {observed k: (m, l)} (float64)."""
    from .fit import CIRC_START
    P = dict(D.s1)
    P.update(CIRC_START if sw["circ"] else {"kappa_M": 0.0, "eps_f": 0.0, "kappa_w": 0.0})
    P.update(num)
    D._apply_foot(P)
    sol = D.sol
    sol.rec, sol.tape = False, None
    Pnp, Pw = D._P(P, False)
    nf = len(D.S["fdt"])
    kobs = {int(o["k"]) for o in D.obs}
    snap = sorted(set(range(every - 1, nf, every)) | kobs | {nf - 1})
    sol.fast_forward(Pw, Pnp, D.Sw, D.S, keep=False, snap=snap)
    msk = D.geo.mask_np
    C, ML = {}, {}
    for k in snap:
        st = sol.snaps[k]
        m, l = st["m"].numpy(), st["l"].numpy()
        C[k] = np.where(msk, m / np.where(msk, l, 1.0), 0.0).astype(np.float32)
        if k in kobs:
            ML[k] = (m, l)
    sol.snaps = {}
    return np.cumsum(D.S["fdt"]), C, ML


def overlap(n_from, n_to):
    """1-D conservative remap weights A (n_to x n_from) between cell-centred uniform grids on the same interval:
    A[I, i] = |coarse cell I intersect fine cell i| / |fine cell i|  (coarse sum = A @ fine)."""
    e_f = np.linspace(0.0, 1.0, n_from + 1)
    e_t = np.linspace(0.0, 1.0, n_to + 1)
    lo = np.maximum(e_t[:-1, None], e_f[None, :-1])
    hi = np.minimum(e_t[1:, None], e_f[None, 1:])
    return np.clip(hi - lo, 0.0, None) * n_from


def to48(m, l, mask, A):
    mm = A @ np.where(mask, m, 0.0) @ A.T
    ll = A @ np.where(mask, l, 0.0) @ A.T
    return np.where(ll > 1e-12, mm / np.where(ll > 1e-12, ll, 1.0), 0.0)


def score48(p, sw, D, ML, T=None, D48=None):
    """J and its parts of the fine run D at the N = 48 equivalent (module docstring)."""
    D48 = D48 or make_pour(p, sw, 48, "cpu", T)
    m48 = D48.geo.mask_np
    A = overlap(D.N, 48)
    tf = np.array([o["t"] for o in D.obs]); kf = [int(o["k"]) for o in D.obs]
    C, W, V, wt = [], [], [], []
    for o in D48.obs:
        i = int(np.argmin(np.abs(tf - o["t"])))
        m, l = ML[kf[i]]
        C.append(to48(m, l, D.geo.mask_np, A) if D.N != 48 else np.where(m48, m / np.where(m48, l, 1.0), 0.0))
        W.append(o["wo_np"]); V.append(o["vo_np"]); wt.append(o["wt"])
    SS.set_grid(48)
    ref = SS.lag_reference([o["t"] for o in D48.obs], W, V, m48)
    r = SS.pour_scores(C, W, V, wt, m48, ref)
    r["J"] = 0.25 * r["viou_n"] + 0.75 * r["struct_n"]
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()}


# ------------------------------------------------------------------------------------------------ movies
def latte(c):
    c = np.clip(c, 0.0, 1.0)[..., None]
    return CREMA * (1 - c) + FOAM * c


def to_H(c, keep, H):
    """Cell field c (N x N, wsolver.Geo layout) on the H x H video grid (pixel centres linspace(-EXT, EXT, H), U = x / CUP_R,
    V = -y / CUP_R as run.video_on_grid): cells outside `keep` take the nearest kept cell, then bilinear."""
    from scipy import ndimage as ndi
    N = c.shape[0]
    idx = ndi.distance_transform_edt(~keep, return_distances=False, return_indices=True)
    cf = np.asarray(c, float)[tuple(idx)]
    s = np.linspace(-EXT, EXT, H)
    U, V = np.meshgrid(s, s)
    return ndi.map_coordinates(cf, [(-V * CUP_R + 0.5) * N - 0.5, (U * CUP_R + 0.5) * N - 0.5], order=1, mode="nearest")


def native(c, mask):
    """One pixel per cell, image rows = -y (the video's orientation), black mug outside the cup."""
    out = latte(c)
    out[~mask] = CUP
    return out[::-1]


def _fonts():
    from PIL import ImageFont
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in (os.path.join(here, "..", "..", "fonts", "wqy-zenhei.ttc"), "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(fn):
            return ImageFont.truetype(fn, 20), ImageFont.truetype(fn, 16), True
    try:
        return ImageFont.load_default(size=20), ImageFont.load_default(size=16), False
    except TypeError:
        return ImageFont.load_default(), ImageFont.load_default(), False


def _writer(fn):
    import imageio
    return imageio.get_writer(fn, fps=FPS, codec="libx264", quality=8, macro_block_size=16, pixelformat="yuv420p")


def movies(p, out, run, base, label, blabel, every):
    """run / base: (t_end, C, mask, N) of the closure at N and of the baseline (or None)."""
    from PIL import Image, ImageDraw
    from ..rig.real_program import REPO
    F, Fs, cjk = _fonts()
    Z = np.load(os.path.join(REPO, "raw_data", "rig", "video_obs", f"{p}.npz"))
    tv, img, Wv, Vd = Z["t"], Z["img"], Z["w"].astype(float) / 255.0, Z["valid"]
    t_end, C, msk, N = run
    ks = np.array(sorted(C)); tk = t_end[ks]
    T = min(tv[-1], tk[-1])
    H = 512
    s = np.linspace(-EXT, EXT, H); U, V = np.meshgrid(s, s); disc = U ** 2 + V ** 2 <= 1.0
    # 1. the fine field alone, one pixel per cell
    w1 = _writer(os.path.join(out, f"{p}_sim{N}.mp4"))
    for t in np.arange(max(tv[0], 0.0), T, 1.0 / FPS):
        k = ks[int(np.argmin(np.abs(tk - t)))]
        fr = np.clip(native(C[k], msk), 0, 255).astype(np.uint8)
        pad = (-fr.shape[0]) % 16
        w1.append_data(np.pad(fr, ((0, pad), (0, pad), (0, 0)), constant_values=18) if pad else fr)
    w1.close()
    # 2. comparison: real | target | closure at N | baseline at N = 48
    names = NAMES if cjk else NAMES_EN
    heads = (["真实视频（俯视校正）", "视频白度目标", label, blabel] if cjk else
             ["real (rectified top view)", "video white target", label, blabel])
    npan = 4 if base is not None else 3
    pad, top = 12, 64
    Wd = npan * H + (npan + 1) * pad; Wd += (-Wd) % 16
    Ht = top + H + 34; Ht += (-Ht) % 16
    w2 = _writer(os.path.join(out, f"{p}_compare.mp4"))
    for t in np.arange(max(tv[0], 0.0), T, 1.0 / FPS):
        j = int(np.searchsorted(tv, t, side="right") - 1)
        a = np.asarray(Image.fromarray(img[j]).resize((H, H), Image.BICUBIC), float); a[~disc] = CUP
        wv = np.asarray(Image.fromarray((Wv[j] * 255).astype(np.uint8)).resize((H, H), Image.BILINEAR), float) / 255
        vv = np.asarray(Image.fromarray(Vd[j].astype(np.uint8) * 255).resize((H, H), Image.NEAREST)) > 127
        b = latte(wv); b[~vv] = HIDDEN; b[~disc] = CUP
        k = ks[int(np.argmin(np.abs(tk - t)))]
        c1 = latte(to_H(C[k], msk, H)); c1[~disc] = CUP
        panels = [a, b, c1]
        if base is not None:
            bt, bC, bm, _ = base
            bk = np.array(sorted(bC))
            kb = bk[int(np.argmin(np.abs(bt[bk] - t)))]
            c2 = latte(to_H(bC[kb], bm, H)); c2[~disc] = CUP
            panels.append(c2)
        canvas = np.full((Ht, Wd, 3), 18, np.uint8)
        for i, pa in enumerate(panels):
            x = pad + i * (H + pad)
            canvas[top:top + H, x:x + H] = np.clip(pa, 0, 255).astype(np.uint8)
        im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
        d.text((pad, 8), f"{names.get(p, p)}    t = {t:5.2f} s", fill=(255, 255, 255), font=F)
        for i, h in enumerate(heads[:npan]):
            d.text((pad + i * (H + pad), 38), h, fill=(255, 230, 150), font=Fs)
        d.text((pad, top + H + 8), ("紫色 = 视频里被壶/奶流/视角挡住、不计分的区域" if cjk else
                                    "purple = not observed in the video (pitcher / stream / outside the views), not scored"),
               fill=(200, 200, 200), font=Fs)
        w2.append_data(np.asarray(im))
    w2.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--closure", required=True)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pours", default=",".join(PATTERNS))
    ap.add_argument("--label", default=None, help="panel label of the closure")
    ap.add_argument("--baseline", default=None, help="closure json shown at N = 48 next to it (movies)")
    ap.add_argument("--blabel", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--every", type=int, default=4, help="movie frames: every k-th solver frame (127 fps -> ~32 fps)")
    ap.add_argument("--T", type=float, default=None, help="simulate only the first T seconds (tests)")
    ap.add_argument("--no-movie", dest="movie", action="store_false")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    num, sw = load_closure(a.closure)
    bnum, bsw = load_closure(a.baseline) if a.baseline else (None, None)
    label = a.label or f"{os.path.basename(os.path.dirname(os.path.abspath(a.closure)))} N={a.N}"
    blabel = a.blabel or "baseline N=48"
    fn = os.path.join(a.out, "eval.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    res["_meta"] = dict(closure=os.path.abspath(a.closure), N=a.N, baseline=a.baseline, T=a.T,
                        note="J = 0.25 viou_n + 0.75 struct_n at the N = 48 equivalent (render512 docstring)")
    for p in a.pours.split(","):
        t0 = time.time()
        D = make_pour(p, sw, a.N, a.device, a.T)
        t_end, C, ML = run_pour(D, num, sw, a.every)
        t1 = time.time()
        r = score48(p, sw, D, ML, a.T)
        res[p] = dict(r, forward_s=t1 - t0)
        print(f"{p}: N {a.N}  J {r['J']:.3f}  IoU {r['viou']:.3f} ({r['viou_n']:.2f})  structure {r['struct']:.3f} "
              f"({r['struct_n']:.2f})  forward {t1 - t0:.0f} s", flush=True)
        json.dump(res, open(fn, "w"), indent=1, default=float)
        if a.movie:
            base = None
            if bnum is not None:
                Db = make_pour(p, bsw, 48, a.device, a.T)
                bt, bC, _ = run_pour(Db, bnum, bsw, 1)
                base = (bt, bC, Db.geo.mask_np, 48)
            movies(p, a.out, (t_end, C, D.geo.mask_np, a.N), base, label, blabel, a.every)
            print(f"{p}: movies written ({time.time() - t1:.0f} s)", flush=True)
        del D, C, ML
    cal = [res[p]["J"] for p in ("tulip", "leaf") if p in res]
    tst = [res[p]["J"] for p in ("heart", "push_heart", "swan") if p in res]
    res["_summary"] = dict(cal_mean=float(np.mean(cal)) if cal else None, test_mean=float(np.mean(tst)) if tst else None)
    json.dump(res, open(fn, "w"), indent=1, default=float)
    print(f"-> {fn}: calibration mean J {res['_summary']['cal_mean']}, held-out mean J {res['_summary']['test_mean']}")


if __name__ == "__main__":
    main()
