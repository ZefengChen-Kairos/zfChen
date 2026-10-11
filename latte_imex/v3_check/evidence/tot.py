import sys, numpy as np
from scipy import ndimage as ndi
exec(open('seg2.py').read().split("pat=sys.argv[1]")[0])
def tot(pat,t,thr=0.80):
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
    yy,xx=np.nonzero(filled); cx,cy=xx.mean(),yy.mean()
    X=np.stack([xx-cx,yy-cy]); C=np.cov(X); w,V=np.linalg.eigh(C); W=V@np.diag(1/np.sqrt(w))@V.T*0.5
    by,bx=np.nonzero(wm)
    if len(bx)==0: return filled.sum(),0,(0,0),(0,0)
    P=W@np.stack([bx-cx,by-cy])
    # extreme points along axis from upper-right (B) to lower-left (D): direction (-1,1)/sqrt2 in disc coords
    u=(-P[0]+P[1])/np.sqrt(2)
    return filled.sum(), wm.sum()/filled.sum(), P.mean(1), (np.percentile(u,2),np.percentile(u,98))
pat=sys.argv[1]
for t in [float(x) for x in sys.argv[2].split(',')]:
    A,f,c,(u0,u1)=tot(pat,t)
    print('t=%5.2f A=%6d whitefrac=%.3f centroid=(%+.3f,%+.3f) B-side edge u=%+.3f  D-side edge u=%+.3f (disc radius units)'%(t,A,f,c[0],c[1],u0,u1))
