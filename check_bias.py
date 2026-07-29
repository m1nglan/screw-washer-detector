"""验证推理"""
import numpy as np
from PIL import Image
import onnxruntime as ort

M=[0.485,0.456,0.406]; S=[0.229,0.224,0.225]; sc=0.03125
def pp(p):
    img=Image.open(p).convert('RGB'); w,h=img.size
    if w<h: nw,nh=256,int(256*h/w)
    else: nw,nh=int(256*w/h),256
    img=img.resize((nw,nh),Image.LANCZOS)
    l,t=(nw-224)//2,(nh-224)//2; img=img.crop((l,t,l+224,t+224))
    raw=np.array(img,dtype=np.float32)/255.0
    for c in range(3): raw[:,:,c]=(raw[:,:,c]-M[c])/S[c]
    q=np.round(raw/sc).clip(-128,127).astype(np.int8)
    return q.transpose(2,0,1)[np.newaxis,:,:,:]

sess=ort.InferenceSession('output/model_espdl.onnx')
name=sess.get_inputs()[0].name
for f,lbl in [('t1.jpg','washer'),('t2.jpg','screw')]:
    x=pp(f).astype(np.float32)*sc
    out=sess.run(None,{name:x})[0][0,:,0,0]
    pred='washer' if out[1]>out[0] else 'screw'
    ok='OK' if pred==lbl else 'XX'
    print(f'{f}: [{out[0]:.2f},{out[1]:.2f}] -> {pred} {ok}')
