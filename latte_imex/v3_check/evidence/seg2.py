import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
R='/tmp/claude-0/-home-user-zfChen/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/ref/full_%s/f%04d.png'
def load(pat,n):
    return np.asarray(Image.open(R%(pat,n)).convert('RGB')).astype(float)/255
def analyse(pat,n,wthr=0.80,save=None):
    a=load(pat,n); r,g,b=a[...,0],a[...,1],a[...,2]
    mx=a.max(-1); mn=a.min(-1); sat=(mx-mn)/(mx+1e-6)
    brown=(sat>0.35)&(mx>0.25)&(mx<0.95)&(r>g)&(g>b)
    overlay=((mx>0.93)&(sat<0.06))|((r>0.75)&(g<0.35)&(b<0.35))
    white=(mx>wthr)&(sat<0.35)&(sat>0.02)&(r>=b)&~overlay
    conn=ndi.binary_opening(brown,iterations=2)
    lab,nn=ndi.label(conn)
    sizes=ndi.sum(np.ones_like(lab),lab,range(1,nn+1)); k=np.argmax(sizes)+1
    br=lab==k
    # allow overlay pixels adjacent to brown to join
    br2=br|(overlay&ndi.binary_dilation(br,iterations=6))
    filled=ndi.binary_fill_holes(ndi.binary_closing(br2,iterations=4))
    inner=filled&~br
    wm=inner&white
    wl,wn=ndi.label(wm)
    A=filled.sum(); yy,xx=np.nonzero(filled); cx,cy=xx.mean(),yy.mean(); L=np.sqrt(A)
    comps=[]
    for i in range(1,wn+1):
        m=wl==i; s=m.sum()
        if s<40: continue
        y,x=np.nonzero(m); comps.append((s/A,(x.mean()-cx)/L,(y.mean()-cy)/L))
    comps.sort(reverse=True)
    if save:
        vis=(a*255).astype(np.uint8).copy(); vis[wm]=[0,255,0]
        Image.fromarray(vis).save(save)
    return A,comps
pat=sys.argv[1]; t0,t1,dt=[float(x) for x in sys.argv[2].split(',')]
thr=float(sys.argv[3]) if len(sys.argv)>3 else 0.80
t=t0
while t<=t1+1e-9:
    n=int(round(t*15))+1
    A,c=analyse(pat,n,thr)
    print('t=%5.2f A=%6d '%(t,A)+'  '.join('[%.3f (%+.2f,%+.2f)]'%x for x in c[:3]))
    t+=dt
