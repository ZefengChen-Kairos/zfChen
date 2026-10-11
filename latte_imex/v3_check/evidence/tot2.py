import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
R='/tmp/claude-0/-home-user-zfChen/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/ref/full_%s/f%04d.png'
def load(pat,n): return np.asarray(Image.open(R%(pat,n)).convert('RGB')).astype(float)/255
def surf(a):
    r,g,b=a[...,0],a[...,1],a[...,2]
    mx=a.max(-1); mn=a.min(-1); sat=(mx-mn)/(mx+1e-6)
    brown=(sat>0.35)&(mx>0.25)&(mx<0.95)&(r>g)&(g>b)
    overlay=((mx>0.93)&(sat<0.08))|((r>0.7)&(g<0.4)&(b<0.4))
    conn=ndi.binary_opening(brown,iterations=2)|(overlay&ndi.binary_dilation(brown,iterations=8))
    lab,nn=ndi.label(conn); sizes=ndi.sum(np.ones_like(lab),lab,range(1,nn+1)); k=np.argmax(sizes)+1
    S=ndi.binary_fill_holes(ndi.binary_closing(lab==k,iterations=4))
    return S,brown,overlay,sat,mx,r,b
def run(pat,t,thr=0.80,save=None):
    n=int(round(t*15))+1; a=load(pat,n)
    S,brown,overlay,sat,mx,r,b=surf(a)
    white=(mx>thr)&(sat<0.35)&(sat>0.02)&(r>=b)&~overlay&S&~brown
    yy,xx=np.nonzero(S); cx,cy=xx.mean(),yy.mean()
    C=np.cov(np.stack([xx-cx,yy-cy])); w,V=np.linalg.eigh(C); W=V@np.diag(1/np.sqrt(w))@V.T*0.5
    wl,wn=ndi.label(white); out=[]
    for i in range(1,wn+1):
        m=wl==i; s=m.sum()
        if s<60: continue
        by,bx=np.nonzero(m); P=W@np.stack([bx-cx,by-cy]); u=(-P[0]+P[1])/np.sqrt(2)
        out.append((s/S.sum(),P.mean(1)[0],P.mean(1)[1],np.percentile(u,3),np.percentile(u,97)))
    out.sort(reverse=True)
    if save is not None:
        v=(a*255).astype(np.uint8).copy(); v[white]=[0,255,0]; e=S&~ndi.binary_erosion(S,iterations=2); v[e]=[255,0,255]
        Image.fromarray(v).save(save)
    return S.sum(),out
pat=sys.argv[1]
for t in [float(x) for x in sys.argv[2].split(',')]:
    A,o=run(pat,t,save='t2_%s_%.2f.png'%(pat,t))
    print('t=%5.2f A=%6d '%(t,A)+' '.join('[f=%.3f c=(%+.2f,%+.2f) uB=%+.2f uD=%+.2f]'%x for x in o[:2]))
