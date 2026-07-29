"""模拟 ESP32 预处理流程测试模型推理"""
import numpy as np
from PIL import Image
from torchvision import transforms
import onnxruntime as ort
import os

# ESP32 量化参数
INPUT_SCALE = 0.03125
INPUT_ZP = 0.0
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def esp32_preprocess(img_path):
    """模拟 ESP32 上的预处理：resize→crop→norm→quant→INT8"""
    img = Image.open(img_path).convert('RGB')
    img = transforms.functional.resize(img, 256)
    img = transforms.functional.center_crop(img, 224)
    raw = np.array(img, dtype=np.float32) / 255.0  # [H,W,C]
    for ch in range(3):
        raw[:, :, ch] = (raw[:, :, ch] - MEAN[ch]) / STD[ch]
    q = np.round(raw / INPUT_SCALE).clip(-128, 127).astype(np.int8)
    return q.transpose(2, 0, 1)[np.newaxis, :, :, :]  # [1,3,224,224]

print("=" * 60)
print("1️⃣  ESP32 风格预处理 (INT8 量化) + ONNX FP32 推理")
print("=" * 60)

sess = ort.InferenceSession('output/model_espdl.onnx')
input_name = sess.get_inputs()[0].name

for img_name in ['t1.jpg', 't2.jpg']:
    int8_x = esp32_preprocess(img_name)
    fp32_x = int8_x.astype(np.float32)
    out = sess.run(None, {input_name: fp32_x})[0][0, :, 0, 0]
    exp = np.exp(out - out.max())
    prob = exp / exp.sum()
    label = '螺丝' if out[0] > out[1] else '螺母'
    print(f"  {img_name}:  raw=[{out[0]:6.2f}, {out[1]:6.2f}]  "
          f"softmax=[{prob[0]:.4f}, {prob[1]:.4f}]  -> {label}")

print()
print("=" * 60)
print("2️⃣  标准预处理 (FP32 Normalize) 对比")
print("=" * 60)

t = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

for img_name in ['t1.jpg', 't2.jpg']:
    img = Image.open(img_name).convert('RGB')
    x = t(img).unsqueeze(0).numpy()
    out = sess.run(None, {input_name: x})[0][0, :, 0, 0]
    exp = np.exp(out - out.max())
    prob = exp / exp.sum()
    label = '螺丝' if out[0] > out[1] else '螺母'
    print(f"  {img_name}:  raw=[{out[0]:6.2f}, {out[1]:6.2f}]  "
          f"softmax=[{prob[0]:.4f}, {prob[1]:.4f}]  -> {label}")
