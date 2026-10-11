import sys, json, os, glob, numpy as np
sys.path.insert(0, '/home/user/zfChen'); os.chdir('/home/user/zfChen')
from latte_imex.optimize import loss_fields, target_at, whiteness_sim, gap_stats, visible
from latte_imex.cell_test import struct_stats
from scipy import ndimage
def mask_of(N): return np.hypot(*np.meshgrid(*(2 * [(np.arange(N) + 0.5) / N - 0.5]))) < 0.49
def dec(tag, s, w, mask, sigma, N):
    L, p = loss_fields(s, w, mask, sigma)
    return f"{tag:34s} L={L:.3f} l1={p['l1']:.3f} iou={p['iou']:.3f} 0.5(1-iou)={0.5*(1-p['iou']):.3f} 0.5area={0.5*p['area_diff']:.3f} white={((s>0.5)&mask).sum()/mask.sum():.3f} gap={gap_stats(s,mask,N)['gap_frac']:.3f}"

print("=== targets: white fraction / gap_frac / edge density (N=96)")
N=96; mask=mask_of(N)
for n in ["heart","push_heart","layered_heart","tulip","leaf","swan"]:
    w=target_at(n,N); ed,q,gf=struct_stats(w,mask,N)
    wb=(w>0.5)&mask; yy,xx=np.mgrid[0:N,0:N]; cy,cx=yy[wb].mean(),xx[wb].mean(); R=np.sqrt(wb.sum()/np.pi)
    disc=((np.hypot(yy-cy,xx-cx)<R)&mask).astype(float)
    filled=ndimage.binary_fill_holes(wb).astype(float)
    print(f"{n:14s} white={wb.sum()/mask.sum():.3f} gap_frac={gf:.3f} edge={ed:.3f}  | same-area disc: {dec('',disc,w,mask,N/48,N)[36:]}")
    print(f"{'':14s} filled outline: {dec('',filled,w,mask,N/48,N)[36:]}")

print("\n=== Codex-style heart, runs/final/heart (N=256) vs heart target")
z=np.load('runs/final/heart/final_state.npz'); l,m,mk=z['l'],z['m'],z['mask']
c=np.where(mk, m/np.maximum(l,1e-9),0)
for Nn in (256,128):
    if Nn==256: cc,mm=c,mk
    else: cc=ndimage.zoom(c,0.5,order=1); mm=mask_of(128)
    s=whiteness_sim(cc,mm); w=target_at('heart',Nn)
    print(dec(f'final/heart @{Nn}', s, w, mm, 0.015*Nn, Nn))

print("\n=== action_opt old model: base vs best per pattern (N=96)")
for f in ("runs/action_opt/search_N96.json","runs/action_opt/refine_N96.json"):
    if not os.path.exists(f): print('missing',f); continue
    S=json.load(open(f)); print('--',f)
    for k,v in S.items():
        b,z=v['best'],v['base']
        def ld(r):
            p=f"runs/action_opt/{r['model']}/{r['name']}_h{r['h']:.3f}_q{r['q']:.3f}_s{r['s']:.3f}_N96.npy"
            if not os.path.exists(p): return None
            c=np.load(p); s=visible(c,mask,r.get('closure')); ed,q,gf=struct_stats(s,mask,96)
            return f"loss={r['loss']:.3f} iou={r['iou']:.2f} white={((s>0.5)&mask).sum()/mask.sum():.3f} gap={gf:.3f} edge={ed:.3f}"
        print(f"{k:22s} base: {ld(z)}\n{'':22s} best: {ld(b)}  (h{b['h']} q{b['q']} s{b['s']})")
