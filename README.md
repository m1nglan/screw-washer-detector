# 螺丝/螺母 分类器 — MobileNetV2 → ESP32-S3

轻量级图像二分类项目，使用 **MobileNetV2** 实现螺丝 (screw/class 0) 与垫圈 (washer/class 1) 的实时识别，部署到 **ESP32-S3**。

---

## 📋 项目状态

| 环节 | 状态 | 备注 |
|------|------|------|
| 数据集整理 | ✅ | 螺丝 453 张，垫圈 744 张 |
| 模型训练 | ✅ | MobileNetV2, 50 epochs, 100% 验证精度 |
| PC ONNX 推理 | ✅ | FP32/INT8 模拟均正确 |
| **PPQ 量化 → .espdl** | ⚠️ | **改用 AutoQuant 自动化量化** |
| **ESP32-S3 部署** | ✅ | **AutoQuant 导出的模型可正常工作** |

### 关键发现

手动 PPQ 量化的模型在板子上始终偏 class 0，但 **AutoQuant 自动量化的模型正常工作**。对比分析：

| 参数 | 手动 PPQ | AutoQuant | 说明 |
|------|---------|-----------|------|
| 输入 scale | 0.03125 (2⁻⁵) | **0.015625 (2⁻⁶)** | 量化步长减半，精度更高 |
| 输出 scale | 0.125 (2⁻³) | **0.0625 (2⁻⁴)** | 输出精度更高 |
| 权重 scale | 7.629e-06 | 7.629e-06 ✅ 相同 |
| 偏置 scale | 4.768e-07 | 4.768e-07 ✅ 相同 |
| 校准步数 | 9 | **18** | 更多数据，估计更准 |
| 调参方式 | 手动 | **自动化搜索** | AutoQuant 自动调优 |

**原因**：AutoQuant 通过自动化参数搜索找到了更优的量化配置（尤其是输入/输出 scale），我们手写的 PPQ 脚本量化参数不够精细。

---

## 📁 项目结构

```
├── dataset/                          ← 训练数据
│   ├── 螺丝/                         ← screw (453张)
│   └── 螺母/                         ← washer (744张)
│
├── AutoQuant/                        ← 🏆 ESP-DL 自动化量化工具
│   ├── user_quant.py                 ← 量化契约文件（校准/评估/目标）
│   ├── model_espdl.onnx              ← 待量化的 ONNX 模型
│   ├── check/                        ← 校准集（扁平 JPG）
│   ├── dataset/                      ← 评估集（子目录=类别）
│   ├── SKILL/espdl-quantize/         ← 量化技能（迭代调参）
│   └── output/iter_0/
│       ├── model.espdl               ← 🏆 最终部署模型
│       ├── model.info / model.json   ← 量化详情
│       └── metrics.json              ← 量化精度指标
│
├── output/                           ← 训练 & 中间产物
│   ├── model_espdl.onnx              ← ESP-DL 兼容 ONNX（供 AutoQuant 使用）
│   ├── best_model.pth                ← 最佳训练权重
│   └── training_curves.png           ← 训练曲线
│
├── main.py                           ← 训练脚本
├── export_espdl_compat.py            ← ESP-DL 兼容 ONNX 导出
├── verify_model.py                   ← ESP32 预处理一致性验证
├── check_bias.py                     ← 快速推理验证
├── requirements.txt                  ← Python 依赖
├── agent.md                          ← 原始需求文档
└── README.md
```

---

## 🚀 完整流程

```bash
# 1. 训练
python main.py                              # → output/best_model.pth

# 2. 导出 ESP-DL 兼容 ONNX
python export_espdl_compat.py               # → output/model_espdl.onnx

# 3. 复制 ONNX 到 AutoQuant 目录
copy output\model_espdl.onnx AutoQuant\

# 4. 用 AutoQuant 量化（两种方式）
#
#    方式 A — VS Code 触发 skill（推荐）：
#      打开 AutoQuant/user_quant.py
#      运行 "ESP-DL Quantization Tuning" skill
#      AutoQuant 会自动迭代调参 → AutoQuant/output/iter_0/model.espdl
#
#    方式 B — 命令行直接跑：
#      cd AutoQuant
#      python -m esp_ppq.samples.espdl_quantize_skill
#      # 或按 SKILL/espdl-quantize/SKILL.md 的指引运行

# 5. PC 验证
python check_bias.py                        # ONNX FP32 推理
python verify_model.py                      # ESP32 预处理一致性

# 6. 部署到 ESP32-S3
#    AutoQuant/output/iter_0/model.espdl → C 数组 → deploy/main/
#    idf.py build flash monitor
```

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
