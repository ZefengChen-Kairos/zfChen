import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
R='/tmp/claude-0/-home-user-zfChen/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/ref/full_%s/f%04d.png'
def load(pat,t):
    n=int(round(t*15))+1
    return np.asarray(Image.open(R%(pat,n)).convert('RGB')).astype(float)/255
def masks(a):
    r,g,b=a[...,0],a[...,1],a[...,2]
    mx=a.max(-1); mn=a.min(-1); sat=(mx-mn)/(mx+1e-6)
    brown=(sat>0.35)&(mx>0.25)&(mx<0.95)&(r>g)&(g>b)
    # white foam: bright, low-mid sat, warm
    white=(mx>0.80)&(sat<0.30)&(r>=b)
    return brown,white
def analyse(pat,t,save=None):
    a=load(pat,t); brown,white=masks(a)
    # largest brown component
    lab,n=ndi.label(ndi.binary_opening(brown,iterations=2))
    if n==0: return None
    sizes=ndi.sum(np.ones_like(lab),lab,range(1,n+1)); k=np.argmax(sizes)+1
    br=lab==k
    filled=ndi.binary_fill_holes(ndi.binary_closing(br,iterations=3))
    inner=filled&~br
    wl,wn=ndi.label(inner&white)
    comps=[]
    for i in range(1,wn+1):
        m=wl==i; s=m.sum()
        if s<30: continue
        yy,xx=np.nonzero(m); comps.append((s,xx.mean(),yy.mean()))
    comps.sort(reverse=True)
    A=filled.sum()
    if save:
        vis=(a*255).astype(np.uint8).copy()
        vis[inner&white]=[0,255,0]; vis[br]=(vis[br]*0.5).astype(np.uint8)
        Image.fromarray(vis).save(save)
    return A,[(s/A,round(x),round(y)) for s,x,y in comps[:4]]
if __name__=='__main__':
    pat=sys.argv[1]
    for t in [float(x) for x in sys.argv[2].split(',')]:
        r=analyse(pat,t,save='seg_%s_%.2f.png'%(pat,t))
        if r is None: print(t,'none'); continue
        A,c=r
        print('t=%.2f surfpx=%d  white comps (frac,x,y): %s  total=%.3f'%(t,A,c,sum(x[0] for x in c)))
