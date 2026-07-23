"""用 ESP-DL 兼容模型 + PPQ 生成 .espdl"""
import torch, numpy as np, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from esp_ppq.api import espdl_quantize_onnx, QuantizationSetting

t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
ds = datasets.ImageFolder(root='dataset', transform=t)
loader = DataLoader(Subset(ds, np.random.choice(len(ds), 150, replace=False)),
    batch_size=16, shuffle=False, collate_fn=lambda b: torch.stack([i[0] for i in b]))

print("PPQ 量化 model_espdl.onnx (无 Flatten/Gemm)...")
s = QuantizationSetting()
for a in ['fusion','equalization','lsq_optimization','blockwise_reconstruction','bias_correct']:
    setattr(s, a, False)

try:
    r = espdl_quantize_onnx(onnx_import_file="output/model_espdl.onnx",
        espdl_export_file="output/model.espdl", calib_dataloader=loader,
        calib_steps=9, input_shape=[3, 224, 224], target="esp32s3",
        num_of_bits=8, device="cpu", verbose=1, setting=s)
    print("✅ 成功!")
    for f in os.listdir("output"):
        if ".espdl" in f:
            print(f"  {f}: {os.path.getsize(os.path.join('output', f))/1024:.1f} KB")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()

