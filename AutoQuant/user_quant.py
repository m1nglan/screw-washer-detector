"""user_quant.py — ESP-DL PPQ 量化契约文件

使用 espdl-quantize skill 对 model_espdl.onnx 进行量化调参。
- 校准集: check/（无标签扁平 JPG）
- 评估集: dataset/（螺丝/螺母 两个子目录，有标签）
- 评估指标: 分类准确率 (accuracy)
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# QUANT_CONFIG
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

QUANT_CONFIG = {
    "model_type": "onnx",
    "onnx_path": os.path.join(_HERE, "model_espdl.onnx"),
    "input_shape": [3, 224, 224],
    "batch_size": 16,
    "target": "esp32s3",
    "num_of_bits": 8,
    "device": "cpu",
    "calib_steps": 18,
    "analyse_steps": 8,
    "top_k_layers": 20,
    "primary_metric": "accuracy",
    "metric_direction": "max",
    "target_metric": 0.95,
}

CALIB_DIR = os.path.join(_HERE, "check")        # 校准集（扁平 JPG）
DATASET_DIR = os.path.join(_HERE, "dataset")     # 评估集（子目录 = 类别）

# 与训练一致的预处理
_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# 校准数据集（check/ — 扁平目录）
# ---------------------------------------------------------------------------
class FlatImageDataset(Dataset):
    """读取扁平目录下的所有图片文件。"""

    def __init__(self, root: str, transform) -> None:
        self.transform = transform
        self.paths = [
            os.path.join(root, name)
            for name in sorted(os.listdir(root))
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        if not self.paths:
            raise FileNotFoundError(f"在 {root} 中未找到任何图片")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def create_calib_dataloader() -> DataLoader:
    """从 check/ 加载校准数据。"""
    ds = FlatImageDataset(CALIB_DIR, transform=_EVAL_TRANSFORM)
    # 随机采样 300 张
    indices = random.Random(42).sample(range(len(ds)), min(300, len(ds)))
    subset = Subset(ds, indices)
    return DataLoader(
        subset,
        batch_size=QUANT_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=lambda b: torch.stack(b),
    )


# ---------------------------------------------------------------------------
# 评估函数 — 分类准确率
# ---------------------------------------------------------------------------
def _run_accuracy(
    quant_graph, eval_loader, device: str
) -> dict:
    """计算量化模型在带标签评估集上的准确率。

    模型输出形状 [N, 2, 1, 1] → squeeze → argmax → 预测类别。
    """
    from esp_ppq.executor.torch import TorchExecutor

    executor = TorchExecutor(graph=quant_graph, device=device)

    correct = 0
    total = 0
    for images, labels in eval_loader:
        images = images.to(device)
        out = executor(images)
        # out 可能是 tuple 或 tensor
        if isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out
        # [N, 2, 1, 1] → [N, 2]
        logits = logits.squeeze(dim=(2, 3))
        preds = logits.argmax(dim=1)
        correct += (preds.cpu() == labels).sum().item()
        total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    return {"accuracy": float(acc), "total_samples": total, "correct": correct}


def evaluate(quant_graph) -> dict:
    """在完整 dataset/ 上评估准确率。"""
    ds = datasets.ImageFolder(root=DATASET_DIR, transform=_EVAL_TRANSFORM)
    loader = DataLoader(
        ds, batch_size=QUANT_CONFIG["batch_size"], shuffle=False
    )
    return _run_accuracy(quant_graph, loader, QUANT_CONFIG["device"])


def evaluate_fast(quant_graph) -> dict:
    """快速评估 — 在 dataset/ 的子集上评估（每类取 30 张）。"""
    ds = datasets.ImageFolder(root=DATASET_DIR, transform=_EVAL_TRANSFORM)
    # 从每个类别中取固定数量，保证类平衡
    samples_per_class = 30
    indices = []
    for cls_idx in range(len(ds.classes)):
        cls_indices = [i for i, (_, lbl) in enumerate(ds.samples) if lbl == cls_idx]
        sampled = random.Random(42).sample(
            cls_indices, min(samples_per_class, len(cls_indices))
        )
        indices.extend(sampled)
    subset = Subset(ds, indices)
    loader = DataLoader(
        subset, batch_size=QUANT_CONFIG["batch_size"], shuffle=False
    )
    return _run_accuracy(quant_graph, loader, QUANT_CONFIG["device"])
