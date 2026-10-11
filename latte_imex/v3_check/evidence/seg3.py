import sys, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
sys.argv=[sys.argv[0]]+sys.argv[1:]
exec(open('seg2.py').read().split("pat=sys.argv[1]")[0])
def vis(pat,t,thr=0.80):
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
    edge=filled&~ndi.binary_erosion(filled,iterations=2)
    v=(a*255).astype(np.uint8).copy(); v[filled&white]=[0,255,0]; v[edge]=[255,0,255]
    im=Image.fromarray(v).crop((40,140,440,520)); ImageDraw.Draw(im).text((5,5),'%.2f'%t,fill=(255,255,0))
    return im
ts=[float(x) for x in sys.argv[2].split(',')]
ims=[vis(sys.argv[1],t) for t in ts]
w,h=ims[0].size; cols=4; rows=(len(ims)+3)//4
M=Image.new('RGB',(w*cols,h*rows))
for i,im in enumerate(ims): M.paste(im,((i%cols)*w,(i//cols)*h))
M.save(sys.argv[3])
