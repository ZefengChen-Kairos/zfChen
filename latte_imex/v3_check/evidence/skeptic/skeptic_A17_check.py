import sys, json, os, numpy as np
sys.path.insert(0, '/home/user/zfChen')
os.chdir('/home/user/zfChen')
from latte_imex.optimize import loss_fields, target_at, whiteness_sim, gap_stats
from scipy import ndimage

def mask_of(N): 
    return np.hypot(*np.meshgrid(*(2 * [(np.arange(N) + 0.5) / N - 0.5]))) < 0.49

def report(tag, s, w, mask, sigma, N):
    L, p = loss_fields(s, w, mask, sigma)
    print(f"{tag:40s} L={L:.3f}  l1={p['l1']:.3f} 0.5(1-iou)={0.5*(1-p['iou']):.3f} 0.5area={0.5*p['area_diff']:.3f}  white={((s>0.5)&mask).sum()/mask.sum():.3f} gap={gap_stats(s,mask,N)['gap_frac']:.3f}")

for N, sigma in [(128, 0.015*128), (96, 96/48)]:
    print(f"\n=== N={N} sigma={sigma:.2f}")
    mask = mask_of(N); w = target_at('heart', N)
    wb = (w > 0.5) & mask
    print(f"target heart white frac {wb.sum()/mask.sum():.3f}, gap_frac target {gap_stats(w,mask,N)['gap_frac']:.3f}")
    report('target vs itself', w, w, mask, sigma, N)
    report('whole cup white', np.where(mask,1.0,0.0), w, mask, sigma, N)
    # disc with the same area at the white centroid
    yy, xx = np.mgrid[0:N,0:N]
    cy, cx = yy[wb].mean(), xx[wb].mean()
    R = np.sqrt(wb.sum()/np.pi)
    disc = (np.hypot(yy-cy, xx-cx) < R) & mask
    report('disc same area @ centroid', disc.astype(float), w, mask, sigma, N)
    for f in (0.8, 1.2):
        d2 = (np.hypot(yy-cy, xx-cx) < R*f) & mask
        report(f'disc area x{f*f:.2f}', d2.astype(float), w, mask, sigma, N)
    # convex hull / filled target (fill holes)
    filled = ndimage.binary_fill_holes(wb)
    report('target white with holes filled', filled.astype(float), w, mask, sigma, N)
    # target thresholded (binary)
    report('target binarised', wb.astype(float), w, mask, sigma, N)

# Codex heart (runs/final/heart) 
z = np.load('runs/final/heart/final_state.npz'); print('\nfinal_state keys', list(z.keys()), {k: z[k].shape for k in z})
