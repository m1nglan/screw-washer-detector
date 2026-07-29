"""
ONNX 推理验证脚本
用法: python check_bias.py
需要: 图片 t1.jpg t2.jpg 放在当前目录下

功能:
  1. 用和训练完全相同的预处理 (Resize256→CenterCrop224→ToTensor→Normalize)
     处理 t1.jpg 和 t2.jpg
  2. 喂给 output/model_espdl.onnx 做 FP32 推理
  3. 打印原始 logit 值并判断分类是否正确

期望输出:
  t1.jpg: 垫圈 (washer)  →  class 1 分数 > class 0 分数
  t2.jpg: 螺丝 (screw)   →  class 0 分数 > class 1 分数
"""
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
