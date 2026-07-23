"""ONNX Runtime 静态量化"""
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
from onnxruntime.quantization.preprocess import quant_pre_process
import onnxruntime as ort
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(root='dataset', transform=transform)
targets = [s for _, s in dataset.samples]
val_idx = []
for c in range(len(dataset.classes)):
    ci = [i for i, t in enumerate(targets) if t == c]
    np.random.shuffle(ci)
    val_idx.extend(ci[:max(1, int(len(ci)*0.15))])
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)

class CalibDataReader:
    def __init__(self, n=150):
        self.n = n; self.i = 0
        idx = np.random.choice(len(dataset), n, replace=False)
        self.loader = iter(DataLoader(Subset(dataset, idx), batch_size=1, shuffle=False))
    def get_next(self):
        if self.i >= self.n: return None
        self.i += 1
        images, _ = next(self.loader)
        return {'input': images.numpy()}

# FP32
sess = ort.InferenceSession('output/model_fixed.onnx')
correct = total = 0
for images, labels in val_loader:
    out = sess.run(None, {'input': images.numpy()})[0]
    total += len(labels)
    correct += (np.argmax(out, axis=1) == labels.numpy()).sum()
print(f'FP32: {100*correct/total:.2f}%')

# Quant
quant_pre_process('output/model_fixed.onnx', 'output/model_pp.onnx')
quantize_static('output/model_pp.onnx', 'output/model_int8.onnx',
    CalibDataReader(150), quant_format=QuantFormat.QOperator,
    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)

# INT8
sess = ort.InferenceSession('output/model_int8.onnx')
correct = total = 0
for images, labels in val_loader:
    out = sess.run(None, {'input': images.numpy()})[0]
    total += len(labels)
    correct += (np.argmax(out, axis=1) == labels.numpy()).sum()
print(f'INT8: {100*correct/total:.2f}%')

print(f'FP32: {os.path.getsize("output/model_fixed.onnx")/1024:.0f} KB')
print(f'INT8: {os.path.getsize("output/model_int8.onnx")/1024:.0f} KB')
ratio = os.path.getsize("output/model_fixed.onnx")/os.path.getsize("output/model_int8.onnx")
print(f'压缩: {ratio:.1f}x')
