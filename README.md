# 螺丝/螺母 分类器 — MobileNetV2 → ESP32-S3

轻量级图像二分类项目，使用 **MobileNetV2** 实现螺丝 (screw/class 0) 与螺母/垫圈 (washer/class 1) 的实时识别，部署到 **ESP32-S3**。

---

## 📋 项目状态

| 环节 | 状态 | 备注 |
|------|------|------|
| 数据集整理 | ✅ | 螺丝 453 张，螺母 744 张 |
| 模型训练 | ✅ | MobileNetV2, 50 epochs, 100% 验证精度 |
| PC ONNX 推理 | ✅ | FP32/INT8 模拟均正确 |
| PPQ 量化 → .espdl | ✅ | 分类器噪声 0.148% |
| **ESP32-S3 部署** | ❌ | **推理结果始终偏 class 0，排查中** |

### 当前问题

模型在 PC 上（ONNX Runtime FP32 + INT8 模拟）推理完全正确，但在 ESP32-S3 上所有输入均输出 class 0 > class 1。已排除：
- ✅ 模型架构问题（3 种架构均测试）
- ✅ 量化噪声过高（bias_correct 后 0.148%）
- ✅ Transpose 算子问题（已移除）
- ✅ AveragePool 算子问题（已移除）
- ❌ **ImagePreprocessor 预处理不一致**（板子输出 ch0=[6,6,6,6], PC=[18,18,19,19]）
- ❌ **ESP-DL Conv INT8 计算**（怀疑有运行时 Bug）

**排查方向**：IDF 侧修复预处理一致性 → 若仍偏 class 0，则向乐鑫提 ESP-DL issue。

---

## 📁 文件说明

### 🔵 PC 侧 — 训练 & 导出

| 文件 | 作用 |
|------|------|
| `main.py` | **训练脚本** — MobileNetV2 迁移学习，50 epochs，保存 `best_model.pth` |
| `demo.py` | **实时检测** — 摄像头 (-c N) 或屏幕选区 (-s) 实时推理 |
| `export_espdl_compat.py` | **ONNX 导出** — 加载 `best_model.pth` → 转 ESP-DL 兼容结构 → 输出 `model_espdl.onnx` |
| `make_espdl.py` | **PPQ 量化** — ONNX → INT8 `.espdl`（EDL2 格式，含 bias_correct + equalization） |
| `quantize.py` | ONNX Runtime 静态 INT8 量化（生成 `model_int8.onnx`，仅供参考） |

### 🔵 PC 侧 — 验证 & 调试

| 文件 | 作用 |
|------|------|
| `verify_model.py` | **ESP32 预处理一致性验证** — 完全模拟板端预处理公式，打印 INT8 量化值 |
| `verify_ppq.py` | **PPQ 量化后验证** — 对比 FP32 ONNX vs INT8 .espdl 推理误差 |
| `test_esp32.py` | **板端预处理模拟** — 用 BILINEAR resize + [0,255] mean/std 处理图片，保存 crop 结果 |
| `test_normal.py` | 标准 ToTensor+Normalize 预处理验证 |
| `check_bias.py` | 快速推理验证（检查 bias 和权重范围） |
| `check_quant.py` | 查看 model.json 中的量化参数（scale/zero_point） |

### 🔵 输出文件

| 文件 | 说明 |
|------|------|
| `output/best_model.pth` | **训练好的权重**（9.1 MB） |
| `output/model_espdl.onnx` | **ESP-DL 兼容 ONNX**（8.9 MB，算子: Conv+Clip+Add，无 Transpose/AveragePool） |
| `output/model.espdl` | **🏆 部署文件**（2.4 MB，INT8 量化，在板子上运行） |
| `output/model.info` | PPQ 量化详情（网络结构、每层量化参数、测试值） |
| `output/model.json` | 量化参数 JSON（scale/zero_point 等） |
| `output/model_fixed.onnx` | 原始 FP32 ONNX（8.7 MB） |
| `output/model_int8.onnx` | ONNX Runtime INT8 量化（2.3 MB，opset 兼容性问题） |

### 🟢 ESP-IDF 侧 — 部署

| 文件 | 说明 |
|------|------|
| `deploy/CMakeLists.txt` | IDF 项目配置 |
| `deploy/main/main.cpp` | **主程序** — JPEG 加载 + ImagePreprocessor + 模型推理 + 结果输出 |
| `deploy/main/model_data.h/cpp` | `.espdl` 的 C 数组嵌入 |
| `deploy/main/tensors/` | 各层 scale/zero_point 的 .npy 文件（344 个） |

---

## 🛠️ PPQ 源码补丁

因 ESP-PPQ 1.3.6 存在多个 Bug，修改了以下文件（`py310_env/Lib/site-packages/esp_ppq/`）：

| 文件 | 修改 | 原因 |
|------|------|------|
| `executor/op/torch/default.py` | AveragePool_forward: 3D→4D unsqueeze | graph pass 压掉 batch 维导致 1D/2D 误判 |
| `executor/op/torch/default.py` | Transpose_forward: 3D→4D unsqueeze | 同上，batch 维被压没 |
| `parser/espdl/espdl_graph_utils.py` | transpose_shape: 补 batch 维 | export 时 shape 维度不匹配 |
| `parser/espdl_exporter.py` | quantize_and_transpose: 补 batch 维 | test value 导出时 permute 维度不匹配 |

---

## 🔄 完整流程

```bash
# 1. 训练
python main.py

# 2. 导出 ESP-DL 兼容 ONNX
python export_espdl_compat.py

# 3. PPQ 量化
python make_espdl.py        # → output/model.espdl

# 4. PC 验证
python check_bias.py        # ONNX FP32 推理验证
python test_esp32.py        # 模拟板端预处理

# 5. 部署到 ESP32-S3
#    - 将 output/model.espdl 转为 C 数组 → deploy/main/model_data.h/cpp
#    - 重新编译烧录
```

---

## ⚡ 预处理公式（板端与 PC 保持一致）

```
resize:      短边缩放到 256（保持宽高比，BILINEAR）
center crop: 224×224
normalize:   (pixel / 255 - mean) / std
             其中 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
quantize:    round(normalized / 0.03125) → clamp[-128, 127]
layout:      NHWC [1, 224, 224, 3]
input scale: 0.03125 (exponent=-5), zero_point=0
output scale: 0.125 (exponent=-3), zero_point=0
```

---

## 📊 模型架构

```
MobileNetV2 backbone (features)
    → AvgPool2d(kernel_size=7)  [合并为 Conv2d(1280,2,7) 的版本也测试过]
    → Conv2d(1280 → 2, kernel_size=1)
    → output [1,2,1,1] NCHW → PPQ 内部转 NHWC
```

---

## 🔗 参考

- [ESP-DL](https://github.com/espressif/esp-dl)
- [ESP-PPQ](https://github.com/espressif/esp-ppq)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)

### 训练模型

```bash
python main.py
```

### 实时检测

```bash
# 摄像头模式
python demo.py -c 0

# 屏幕区域检测模式
python demo.py -s
```

### 量化模型

```bash
# ONNX Runtime 量化 (Python 3.12)
python quantize.py

# PPQ → .espdl (需 Python 3.10)
python make_espdl.py
```

---

## 📊 训练结果

| 指标 | 值 |
|------|-----|
| **模型** | MobileNetV2 |
| **参数量** | 2,226,434 |
| **训练数据** | 1,019 张 (386 screw + 633 washer) |
| **验证数据** | 178 张 (67 screw + 111 washer) |
| **FP32 准确率** | **98.88%** |
| **INT8 准确率** | **98.31%** |
| **ESP-DL 大小** | **2.3 MB** |
| **训练耗时** | ~38 分钟 (CPU) |

---

## 🏆 模型文件说明

| 文件 | 格式 | 大小 | 用途 |
|------|------|------|------|
| `model.espdl` | ESP-DL | 2.3 MB | 🏆 **ESP32-S3 部署** |
| `model.info` | 文本 | — | 模型结构/输入输出信息、算子列表 |
| `model.json` | JSON | — | 量化配置文件（每层 scale/zero_point） |
| `model_int8.onnx` | INT8 ONNX | 2.3 MB | ONNX Runtime 部署 |
| `model_fixed.onnx` | FP32 ONNX | 8.5 MB | 精度验证/调试 |
| `best_model.pth` | PyTorch | 9.1 MB | 重新训练/微调 |

---

## 🛠️ 技术栈

- **框架**: PyTorch 2.x, torchvision
- **量化**: ESP-PPQ, ONNX Runtime
- **部署**: ESP-DL, ESP-IDF
- **网络**: MobileNetV2 (Conv 52, Clip 35, Add 10, GlobalAveragePool 1, Conv 1)
