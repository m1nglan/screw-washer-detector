# model_espdl — 量化模型说明

## 概述

`output/iter_0/model.espdl` 是由 `esp-ppq` 将 `model_espdl.onnx` 量化后生成的 ESP-DL 部署格式模型，目标芯片为 **ESP32-S3**，位宽 **8-bit INT8**。

| 属性 | 值 |
|------|-----|
| 原始格式 | ONNX (FP32) |
| 部署格式 | `.espdl` (EDL2) |
| 文件大小 | **2.49 MB** |
| 目标芯片 | ESP32-S3 |
| 量化精度 | 8-bit INT8 (对称 + 线性 + Per-Tensor + Power-of-2) |
| 评估准确率 | **96.67%** (58/60) |

---

## 模型架构

该模型为 **MobileNetV2** 风格的轻量级卷积神经网络，包含 18 个倒残差块（Inverted Residual Block），每个块由以下结构组成：

```
输入 → Conv1×1 (扩展) → Clip (ReLU6) → Conv3×3 (深度可分离) → Clip → Conv1×1 (压缩) → + (shortcut, 可选)
```

### 算子分布

| 算子类型 | 数量 | 说明 |
|----------|------|------|
| **Conv** | 53 | 标准卷积 + 深度可分离卷积 + 逐点卷积 |
| **Clip** | 35 | ReLU6 激活函数（通过查找表 LUT 实现） |
| **Add** | 10 | 残差连接（shortcut） |
| **合计** | **98** | |

### 网络流水线

```
features.0:   Conv 3×3  (3→32)   → Clip                    # 输入层
features.1:   Conv 1×1, DW 3×3                              # 倒残差块 1 (无 shortcut)
features.2:   Conv 1×1, DW 3×3, Conv 1×1                    # 倒残差块 2
features.3:   Conv 1×1, DW 3×3, Conv 1×1  → Add            # 倒残差块 3 (+ shortcut)
features.4:   Conv 1×1, DW 3×3, Conv 1×1                    # 倒残差块 4
features.5:   Conv 1×1, DW 3×3, Conv 1×1  → Add            # 倒残差块 5 (+ shortcut)
features.6:   Conv 1×1, DW 3×3, Conv 1×1  → Add            # 倒残差块 6 (+ shortcut)
features.7:   Conv 1×1, DW 3×3, Conv 1×1                    # 倒残差块 7
features.8:   Conv 1×1, DW 3×3, Conv 1×1  → Add            # 倒残差块 8 (+ shortcut)
features.9:   Conv 1×1, DW 3×3, Conv 1×1  → Add            # 倒残差块 9 (+ shortcut)
features.10:  Conv 1×1, DW 3×3, Conv 1×1  → Add            # 倒残差块 10 (+ shortcut)
features.11:  Conv 1×1, DW 3×3, Conv 1×1                    # 倒残差块 11
features.12:  Conv 1×1, DW 3×3, Conv 1×1  → Add            # 倒残差块 12 (+ shortcut)
features.13:  Conv 1×1, DW 3×3, Conv 1×1  → Add            # 倒残差块 13 (+ shortcut)
features.14:  Conv 1×1, DW 3×3, Conv 1×1                    # 倒残差块 14
features.15:  Conv 1×1, DW 3×3, Conv 1×1  → Add            # 倒残差块 15 (+ shortcut)
features.16:  Conv 1×1, DW 3×3, Conv 1×1  → Add            # 倒残差块 16 (+ shortcut)
features.17:  Conv 1×1, DW 3×3, Conv 1×1                    # 倒残差块 17
features.18:  Conv 1×1 (1280→1280) → Clip                   # 最后一层
classifier:   Conv 1×1 (1280→2)                             # 分类头
```

---

## 输入/输出

### 输入

| 属性 | 值 |
|------|-----|
| 张量名称 | `input` |
| 数据格式 | **NHWC** (ESP-DL 原生格式) |
| 形状 | `1 × 224 × 224 × 3` |
| 数据类型 | `INT8` |
| 指数 (exponent) | `-6`（缩放因子 $scale = 2^{-6} = 0.015625$） |

预处理（量化前 FP32 输入需匹配）：
```python
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

量化后输入值范围：$[-128 \times 0.015625, 127 \times 0.015625] = [-2.0, 1.984]$（对应于 FP32 归一化后的值）。

### 输出

| 属性 | 值 |
|------|-----|
| 来源算子 | `/classifier/Conv` |
| 形状 | `1 × 2 × 1 × 1` |
| 数据类型 | INT8 |
| 含义 | 二分类 logits（screw / washer） |

输出后处理：
```python
logits = output.squeeze()          # [2]
pred_class = logits.argmax()       # 0 = screw, 1 = washer
```

### 类别映射

参见 `labels.json`：

| 索引 | 名称 | 中文 |
|------|------|------|
| 0 | `screw` | 螺丝 |
| 1 | `washer` | 螺母 |

---

## 量化配置

### 量化策略

所有张量采用统一的量化策略：

| 属性 | 值 |
|------|-----|
| 对称性 | **SYMMETRICAL**（对称量化） |
| 线性 | **LINEAR**（线性量化） |
| 粒度 | **PER_TENSOR**（逐张量） |
| 缩放因子 | **POWER_OF_2**（2 的幂次） |

量化公式：$Q = \text{round}(x \cdot 2^{-e})$，其中 $e$ 为每张量的指数。

### 张量状态分布

| 状态 | 数量 | 说明 |
|------|------|------|
| **ACTIVATED** | 107 | 激活值，含独立的 scale |
| **OVERLAPPED** | 122 | 与其他张量共享 scale（由 dominator 决定） |
| **PASSIVE** | 153 | 被动张量，scale 由上下游推导 |
| **合计** | **382** | |

### 位宽分布

| 位宽 | 数量 | 用途 |
|------|------|------|
| **8-bit** | 329 | 权重 + 激活值 |
| **20-bit** | 53 | 卷积累加器中间结果 |

---

## 文件结构

`.espdl` 文件是 ESP-DL 的二进制部署格式，魔数为 `EDL2`。其伴随文件包括：

| 文件 | 大小 | 说明 |
|------|------|------|
| `model.espdl` | 2.49 MB | ESP-DL 部署二进制（核心文件） |
| `model.info` | 15.1 MB | 模型结构文本描述（调试用） |
| `model.json` | 294 KB | 量化配置 JSON（每张量的 scale/bit 设置） |
| `_input.onnx` | 9.37 MB | 原始 ONNX 模型副本 |

---

## 通道数变化

网络各阶段的特征图通道数（从 model.info 中的权重形状提取）：

| 阶段 | 输入 → 中间 → 输出通道 |
|------|----------------------|
| features.0 | 3 → 32 |
| features.1 | 32 → 32 → 16 |
| features.2 | 16 → 96 → 24 |
| features.3 | 24 → 144 → 24 |
| features.4 | 24 → 144 → 32 |
| features.5 | 32 → 192 → 32 |
| features.6 | 32 → 192 → 64 |
| features.7 | 64 → 384 → 64 |
| features.8 | 64 → 384 → 96 |
| features.9 | 96 → 576 → 96 |
| features.10 | 96 → 576 → 160 |
| features.11 | 160 → 960 → 160 |
| features.12 | 160 → 960 → 160 |
| features.13 | 160 → 960 → 320 |
| features.14 | 320 → 1280 → 320? |
| features.15 | ... → ... |
| features.18 | 1280 → 1280 |
| classifier | 1280 → **2** |

---

## 量化误差分析

基线量化结果（iter-0）中误差最高的 5 层：

| 排名 | 层路径 | SNR (信噪比) |
|------|--------|:------------:|
| 1 | `features.1/conv/conv.0/conv.0.0/Conv` | 0.687 |
| 2 | `features.3/conv/conv.1/conv.1.0/Conv` | 0.142 |
| 3 | `features.1/conv/conv.1/Conv` | 0.034 |
| 4 | `features.2/conv/conv.1/conv.1.0/Conv` | 0.031 |
| 5 | `classifier/Conv` | 0.017 |

整体准确率：**96.67%** — 量化损失极小。

---

## 部署说明

1. 将 `output/iter_0/model.espdl` 复制到 ESP-DL 工程的模型目录
2. 在 ESP32-S3 固件中加载模型：
   ```cpp
   #include "esp_dl_model.h"
   Model *model = model_from_espdl(model_espdl_data);
   ```
3. 输入需为 INT8 NHWC `[1, 224, 224, 3]` 格式
4. 输出为 INT8 `[1, 2, 1, 1]`，squeeze 后 argmax 取类别
