# ESP-DL PPQ 量化项目

将 ONNX 模型量化为 ESP32-S3 可部署的 `.espdl` 格式，使用 `esp-ppq` 工具链。

## 项目结构

```
├── model_espdl.onnx          # 输入 ONNX 模型（二分类：螺丝/螺母）
├── quantize_espdl_fixed.py   # 原始量化脚本（单次运行）
├── user_quant.py             # espdl-quantize skill 契约文件（迭代调参用）
├── dataset/                  # 带标签评估集
│   ├── 螺丝/                 # class 0 — screw
│   └── 螺母/                 # class 1 — washer
├── check/                    # 无标签校准集（扁平 JPG）
├── labels.json               # 类别映射
├── output/                   # 量化输出目录
│   └── iter_0/               # 基线量化结果
│       ├── model.espdl       # ESP-DL 部署格式
│       ├── iteration_index.json
│       └── ...
├── SKILL/espdl-quantize/     # 量化调参 skill（高级工具）
├── AGENTS.md                 # AI 代理项目说明
└── README.md                 # 本文件
```

## 模型信息

| 属性 | 值 |
|------|-----|
| 输入 | `[1, 3, 224, 224]` |
| 输出 | `[1, 2, 1, 1]`（二分类 logits） |
| 算子 | 98 个（Conv / Add / Clip） |
| 目标芯片 | ESP32-S3 |
| 位宽 | 8-bit INT8 |

## 快速开始

```bash
# 1. 单次量化（使用原始脚本）
python quantize_espdl_fixed.py

# 2. 使用 skill 框架进行基线量化 + 分析
python SKILL/espdl-quantize/scripts/run_iteration.py \
    --user-quant user_quant.py \
    --baseline \
    --output-dir output/iter_0

# 3. 验证契约文件
python SKILL/espdl-quantize/scripts/run_iteration.py \
    --user-quant user_quant.py \
    --check-contract \
    --output-dir output/check
```

## 依赖

- `esp-ppq` >= 1.3.6
- `torch` / `torchvision`
- `onnx` / `onnxruntime` / `onnxsim-prebuilt`
- `numpy` / `pandas` / `scipy` / `tqdm`

## 量化设置

| 设置项 | 基线值 | 说明 |
|--------|--------|------|
| `fusion` | True | 算子融合 |
| `equalization` | True | 跨层均衡化 |
| `bias_correct` | True | 偏置校正 |
| `blockwise_reconstruction` | True | 逐块重建 |
| `lsq_optimization` | False | LSQ 优化（POWER_OF_2 目标下自动禁用） |

## 数据预处理

与训练时一致的 ImageNet 预处理：

```python
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

- 校准：从 `check/` 随机采样 300 张，`batch_size=16`，`calib_steps=18`
- 评估：`dataset/` 下按子目录分类，计算准确率

## 基线结果

| 指标 | 值 |
|------|-----|
| 准确率 | **96.67%** (58/60) |
| `.espdl` 大小 | **2.49 MB** |
| 总耗时 | ~10.6 分钟 |

## 高级调参

`SKILL/espdl-quantize/` 目录包含完整的迭代式量化调优工具链，支持：

- 校准算法搜索（KL / Percentile / MSE / MinMax / Isotone）
- TQT（训练后量化训练）
- 混合精度（dispatching_table int16 提升）
- 逐层误差分析与分布诊断

详见 `SKILL/espdl-quantize/SKILL.md`。
