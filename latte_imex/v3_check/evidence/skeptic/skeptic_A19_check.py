import numpy as np, json, glob, os, sys
sys.path.insert(0, "/home/user/zfChen")
from scipy import ndimage
from latte_imex import targets
from latte_imex.optimize import target_at, loss_fields, visible, gap_stats
N = 96
mask = np.hypot(*np.meshgrid(*(2*[(np.arange(N)+0.5)/N-0.5]))) < 0.49
print("=== target w statistics")
for name in ["heart","push_heart","layered_heart","tulip","leaf","swan"]:
    w, m = targets.load(name); M = w.shape[0]
    v = w[m]
    h, _ = np.histogram(v, bins=[0,0.1,0.2,0.3,0.5,0.7,0.9,1.01])
    wN = target_at(name, N)
    vN = wN[mask]; hN, _ = np.histogram(vN, bins=[0,0.1,0.2,0.3,0.5,0.7,0.9,1.01])
    gs = gap_stats(wN, mask, N)
    print(f"{name:14s} M={M} frac in [0,.1,.2,.3,.5,.7,.9,1]: {np.round(h/h.sum(),3)}  | N96: {np.round(hN/hN.sum(),3)} gap_frac={gs['gap_frac']:.3f} gapw={gs.get('gap_mm', gs.get('mean_width_mm', float('nan'))):.2f}")
# push_heart centre column profile
w,m = targets.load("push_heart"); M=w.shape[0]
col = w[:, M//2]
print("push_heart centre column (y up, every 4th):", np.round(col[::4],2))
w,m = targets.load("leaf"); M=w.shape[0]
row = w[M//2, :]
print("leaf centre row (every 4th):", np.round(row[::4],2))
wN = target_at("leaf", N); print("leaf N96 centre row (every 2nd):", np.round(wN[N//2, ::2],2))
w,m = targets.load("swan"); M=w.shape[0]
print("swan col M//2 (every 4th):", np.round(w[::4, M//2],2))

print("=== best cached runs & loss sensitivity")
sig = N/48
for name in ["push_heart","leaf","swan","tulip","heart"]:
    best=None
    for f in glob.glob(f"runs/action_opt/*/{name}_*_N{N}.json"):
        r=json.load(open(f))
        if r.get("ok") and (best is None or r["loss"]<best[0]["loss"]): best=(r,f)
    r,f = best if best else (None,None)
    wN = target_at(name, N)
    L_t,_ = loss_fields(wN, wN, mask, sig)
    # gap-closed target
    white = (wN>0.5)&mask
    rad = int(round(3.0/80*N))  # 3 mm radius -> 6 mm disc
    yy,xx=np.mgrid[-rad:rad+1,-rad:rad+1]; disc=(xx**2+yy**2)<=rad**2
    closed = ndimage.binary_closing(white, disc) | white
    closed = ndimage.binary_fill_holes(closed) & mask
    L_c,p_c = loss_fields(closed.astype(float), wN, mask, sig)
    blur = np.clip(ndimage.gaussian_filter(wN, sig)*1.3,0,1)*mask
    L_b,p_b = loss_fields(blur, wN, mask, sig)
    # pure brightness check: target scaled to 0.7 max
    L_s,p_s = loss_fields(np.clip(wN*0.7,0,1), wN, mask, sig)
    line = f"{name:11s} target-vs-self {L_t:.3f}; gap-closed {L_c:.3f} iou={p_c['iou']:.2f} l1={p_c['l1']:.3f}; blurred*1.3 {L_b:.3f}; target*0.7 {L_s:.3f} l1={p_s['l1']:.3f}"
    if r:
        c = np.load(f.replace(".json",".npy")); s = visible(c, mask, r.get("closure"))
        gs = gap_stats(s, mask, N); gt = gap_stats(wN, mask, N)
        hs,_ = np.histogram(s[mask], bins=[0,0.1,0.3,0.5,0.7,0.9,1.01])
        line += f" | best {os.path.basename(os.path.dirname(f))} closure={r.get('closure')} loss={r['loss']:.3f} iou={r['iou']:.2f} l1={r['l1']:.3f} gap_frac sim/tgt={gs['gap_frac']:.3f}/{gt['gap_frac']:.3f}; s hist [0,.1,.3,.5,.7,.9,1]={np.round(hs/hs.sum(),3)} max s={s[mask].max():.2f}"
    print(line)
