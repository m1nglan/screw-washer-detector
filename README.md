# 螺丝/螺母 分类器 — MobileNetV2 → ESP32-S3

轻量级图像二分类项目，使用 MobileNetV2 实现螺丝 (screw) 与螺母 (washer) 的实时识别，支持部署到 **ESP32-S3** 嵌入式设备。

---

## 📋 项目结构

```
├── dataset/                          ← 训练数据
│   ├── 螺丝/                         ← screw (453张)
│   └── 螺母/                         ← washer (744张)
│
├── output/                           ← 模型输出
│   ├── model.espdl                   ← 🏆 ESP-DL 格式 (2.3MB, 可部署)
│   ├── model_int8.onnx               ← INT8 ONNX (2.3MB, 98.31%)
│   ├── model_fixed.onnx              ← FP32 ONNX (8.5MB)
│   ├── best_model.pth                ← 最佳训练权重
│   └── training_curves.png           ← 训练曲线
│
├── deploy/                           ← ESP-IDF 部署模板
│   ├── CMakeLists.txt
│   └── main/
│       ├── main.cpp                  ← 主程序（需补充推理引擎）
│       ├── model_data.h/cpp          ← 模型二进制数据
│       └── tensors/                  ← INT8 权重文件 (344个)
│
├── main.py                           ← 训练脚本
├── demo.py                           ← 实时检测（摄像头/屏幕）
├── export_espdl_compat.py            ← ESP-DL 兼容模型导出
├── make_espdl.py                     ← PPQ 量化 → .espdl
├── quantize.py                       ← ONNX Runtime 量化
├── requirements.txt                  ← Python 依赖
├── agent.md                          ← 原始需求文档
├── ppq.md                            ← 量化任务文档
└── README.md
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：PPQ 量化需要 **Python 3.10** + PyTorch 2.6.0

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
