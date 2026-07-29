"""处理后图片保存为 PNG"""
import numpy as np
from PIL import Image

mean_255 = [123.675, 116.28, 103.53]
std_255  = [58.395, 57.12, 57.375]
exponent = -5
scale = 2**exponent

def preprocess(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    print(f"  原图尺寸: {w}x{h}")
    if w < h:
        nw, nh = 256, int(256 * h / w)
    else:
        nw, nh = int(256 * w / h), 256
    img = img.resize((nw, nh), Image.BILINEAR)
    print(f"  缩放后: {nw}x{nh}")
    l = (nw - 224) // 2
    t = (nh - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    print(f"  CenterCrop: ({l},{t}) -> 224x224")
    
    # 保存 crop 后的图
    crop_path = path.replace('.jpg', '_crop.png')
    img.save(crop_path)
    print(f"  已保存: {crop_path}")
    
    raw = np.array(img, dtype=np.uint8)
    float_img = raw.astype(np.float32)
    for c in range(3):
        float_img[:, :, c] = (float_img[:, :, c] - mean_255[c]) / std_255[c]
    quantized = np.round(float_img / scale).clip(-128, 127).astype(np.int8)
    return raw, quantized

for f in ['t1.jpg', 't2.jpg']:
    print(f'\n--- {f} ---')
    raw, q = preprocess(f)
    p = q.flatten()
    rp = raw.flatten()
    print(f"  raw ch0[0..3]: {rp[0]} {rp[3]} {rp[6]} {rp[9]}")
    print(f"  raw ch1[0..3]: {rp[1]} {rp[4]} {rp[7]} {rp[10]}")
    print(f"  raw ch2[0..3]: {rp[2]} {rp[5]} {rp[8]} {rp[11]}")
    print(f"  NHWC input ch0[0..3]: [{p[0]}, {p[3]}, {p[6]}, {p[9]}]")
    print(f"  NHWC input ch1[0..3]: [{p[1]}, {p[4]}, {p[7]}, {p[10]}]")
    print(f"  NHWC input ch2[0..3]: [{p[2]}, {p[5]}, {p[8]}, {p[11]}]")
