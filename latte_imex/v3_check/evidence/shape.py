import sys, numpy as np
from scipy import ndimage as ndi
exec(open('seg2.py').read().split("pat=sys.argv[1]")[0])
def blobmask(pat,t,thr=0.80):
    n=int(round(t*15))+1
    a=load(pat,n); r,g,b=a[...,0],a[...,1],a[...,2]
    mx=a.max(-1); mn=a.min(-1); sat=(mx-mn)/(mx+1e-6)
    brown=(sat>0.35)&(mx>0.25)&(mx<0.95)&(r>g)&(g>b)
    overlay=((mx>0.93)&(sat<0.06))|((r>0.75)&(g<0.35)&(b<0.35))
    white=(mx>thr)&(sat<0.35)&(sat>0.02)&(r>=b)&~overlay
    conn=ndi.binary_opening(brown,iterations=2)
    lab,nn=ndi.label(conn); sizes=ndi.sum(np.ones_like(lab),lab,range(1,nn+1)); k=np.argmax(sizes)+1
    br=lab==k; br2=br|(overlay&ndi.binary_dilation(br,iterations=6))
    filled=ndi.binary_fill_holes(ndi.binary_closing(br2,iterations=4))
    wm=filled&white&~br
    wl,wn=ndi.label(wm); s=ndi.sum(np.ones_like(wl),wl,range(1,wn+1)); kk=np.argmax(s)+1
    blob=wl==kk
    yy,xx=np.nonzero(filled); cx,cy=xx.mean(),yy.mean(); L=np.sqrt(filled.sum())
    # fit ellipse of surface via second moments -> affine normalise to unit disc
    X=np.stack([xx-cx,yy-cy]); C=np.cov(X); w,V=np.linalg.eigh(C)
    W=V@np.diag(1/np.sqrt(w))@V.T*0.5  # maps surface ellipse to disc of radius ~1 (uniform disc var=R^2/4)
    by,bx=np.nonzero(blob); P=W@np.stack([bx-cx,by-cy])
    return P, blob.sum()/filled.sum()
def raster(P,N=200):
    M=np.zeros((N,N),bool); ij=np.clip(((P+1.1)/2.2*N).astype(int),0,N-1); M[ij[1],ij[0]]=True
    return ndi.binary_closing(M,iterations=1)
ref=float(sys.argv[1])
P0,f0=blobmask('push_heart',ref); M0=raster(P0)
for t in [float(x) for x in sys.argv[2].split(',')]:
    P,f=blobmask('push_heart',t); M=raster(P)
    iou=(M&M0).sum()/(M|M0).sum()
    # best translation
    best=(iou,0,0)
    c0=P0.mean(1); c=P.mean(1)
    for dx in range(-12,13,2):
        for dy in range(-12,13,2):
            Ms=np.roll(np.roll(M,dy,0),dx,1); v=(Ms&M0).sum()/(Ms|M0).sum()
            if v>best[0]: best=(v,dx,dy)
    d=c-c0
    # extent along direction from blob centroid toward upper-right (jet side) in disc units (R=1 -> D/2)
    print('t=%.2f frac=%.3f IoU=%.2f bestIoU=%.2f (shift %d,%d px of 200) centroid shift=(%+.3f,%+.3f) [disc radius units; x0.5 -> D]'%(t,f,iou,best[0],best[1],best[2],d[0],d[1]))
