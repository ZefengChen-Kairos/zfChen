import numpy as np
from PIL import Image
# Use the 256x256 rectified web targets (cup fills the circle). Measure runs of brown/white along horizontal and vertical centre lines & a few chords.
import glob,os
for p in ['heart','push_heart','layered_heart','tulip','leaf','swan']:
    im=np.asarray(Image.open(f'/home/user/zfChen/web/targets/{p}.png').convert('RGB')).astype(float)
    # whiteness: distance from brown (approx 90,50,30) vs cream (240,230,210)
    L=im.mean(axis=2)
    H,W=L.shape
    yy,xx=np.mgrid[:H,:W]; r=np.hypot(yy-H/2,xx-W/2)
    inside=r<0.47*W
    thr=(L[inside].max()+L[inside].min())/2
    white=(L>thr)&inside
    frac=white.sum()/inside.sum()
    # run-lengths along horizontal lines at several y
    print(f'== {p}: white fraction inside cup={frac:.2f}, thr={thr:.0f}, Lmin={L[inside].min():.0f} Lmax={L[inside].max():.0f}')
    for y in [int(H*f) for f in (0.3,0.4,0.5,0.6,0.7)]:
        row=white[y]; xs=np.where(inside[y])[0]
        if len(xs)==0: continue
        seg=row[xs[0]:xs[-1]+1]
        runs=[];cur=seg[0];n=0
        for v in seg:
            if v==cur: n+=1
            else: runs.append(('W' if cur else 'b',n)); cur=v;n=1
        runs.append(('W' if cur else 'b',n))
        D=xs[-1]-xs[0]+1
        print(f'  y={y/H:.1f}D: '+' '.join(f'{c}{n/D*100:.0f}%' for c,n in runs))
    for x in [int(W*f) for f in (0.4,0.5,0.6)]:
        col=white[:,x]; ys=np.where(inside[:,x])[0]
        seg=col[ys[0]:ys[-1]+1]
        runs=[];cur=seg[0];n=0
        for v in seg:
            if v==cur: n+=1
            else: runs.append(('W' if cur else 'b',n)); cur=v;n=1
        runs.append(('W' if cur else 'b',n))
        D=ys[-1]-ys[0]+1
        print(f'  x={x/W:.1f}D (top->bottom): '+' '.join(f'{c}{n/D*100:.0f}%' for c,n in runs))
