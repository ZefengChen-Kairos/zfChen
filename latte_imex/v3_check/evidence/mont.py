import sys
from PIL import Image, ImageDraw
R='/tmp/claude-0/-home-user-zfChen/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/ref/full_%s/f%04d.png'
pat=sys.argv[1]; times=[float(x) for x in sys.argv[2].split(',')]; out=sys.argv[3]
crop=(0,110,480,570) if len(sys.argv)<5 else tuple(int(v) for v in sys.argv[4].split(','))
ims=[]
for t in times:
    n=int(round(t*15))+1
    im=Image.open(R%(pat,n)).convert('RGB').crop(crop)
    d=ImageDraw.Draw(im); d.text((5,5),'t=%.2f'%t,fill=(255,255,0))
    ims.append(im)
w,h=ims[0].size; cols=min(4,len(ims)); rows=(len(ims)+cols-1)//cols
M=Image.new('RGB',(w*cols,h*rows))
for i,im in enumerate(ims): M.paste(im,((i%cols)*w,(i//cols)*h))
M.save(out)
