# AGENTS.md — ESP-DL PPQ 量化项目

## 项目概述

本项目是一个轻量级的单脚本项目，用于将 ONNX 模型通过 `esp_ppq`（PPQ 的 ESP-DL 分支）量化为 ESP32-S3 可部署的 `.espdl` 格式。项目核心仅包含一个 Python 脚本，没有复杂的包结构或构建系统。

当前根目录结构：

```
.
├── AGENTS.md                 # 本文件
├── quantize_espdl_fixed.py   # 主量化脚本
├── model_espdl.onnx          # 实际存在的输入 ONNX 模型
├── dataset/                  # 校准图片集合（约 358 张 JPG）
├── output/                   # 量化输出目录（当前为空）
└── SKILL/espdl-quantize/     # 可选的高级量化调参 Skill（独立工具集）
```

模型基本信息（通过 `onnx.load` 检查）：

- 文件：`model_espdl.onnx`
- 输入：`input` → `[1, 3, 224, 224]`
- 输出：`output` → `[1, 2, 1, 1]`（二分类模型）
- 算子：98 个节点，仅含 `Conv`、`Add`、`Clip`

## 核心技术栈

- **esp_ppq / ppq**：英飞凌 ESP-DL 神经网络量化工具包
- **TargetPlatform.ESP_DL_INT8_TORCH**：ESP-DL INT8 目标平台
- **torch / torchvision**：校准数据加载与预处理
- **numpy**：随机采样与数值操作
- **ONNX**：输入模型格式
- **目标芯片**：`esp32s3`
- **位宽**：8-bit INT8

## 量化流程

```
ONNX 模型 → 校准数据集（300 张随机子集，ImageNet 预处理） → PPQ 量化设置 → .espdl + INT8 ONNX
```

## 关键命令

```bash
# 运行主量化脚本
python quantize_espdl_fixed.py
```

## 文件说明

| 文件/目录 | 作用 |
|-----------|------|
| `quantize_espdl_fixed.py` | 主入口脚本：加载 ONNX、构建校准 dataloader、调用 `espdl_quantize_onnx` |
| `model_espdl.onnx` | 项目实际存放的 ONNX 模型文件 |
| `dataset/` | 校准图片目录（当前为扁平结构，所有 `.jpg` 直接放在该目录下） |
| `output/` | 量化产物输出目录，脚本运行后应生成 `model.espdl` 与 `model_int8.onnx` |
| `SKILL/espdl-quantize/` | 可选的高级量化调参 Skill，包含迭代式精度恢复、错误分析、参数搜索等工具 |

## 依赖环境

脚本依赖以下 Python 包（已确认存在，但分散在不同 Python 环境中）：

- `esp-ppq`（安装在 `C:\Users\30709\esp-ppq`，版本 1.3.6）
- `torch` 2.12.0+cpu
- `torchvision` 0.27.0+cpu
- `torchaudio` 2.11.0+cpu
- `onnx` 1.16.1
- `onnxruntime` 1.27.0
- `onnxsim-prebuilt` 0.4.39.post2
- `numpy`

注意：当前环境存在 Python 解释器与 `pip` 不一致的情况——默认 `python` 是 Python 3.10，而 `pip` 指向 Python 3.12。运行脚本前请确保目标 Python 环境已安装上述依赖。

## 代码组织

项目没有采用包结构：

- 根目录下仅有一个可执行脚本 `quantize_espdl_fixed.py`，所有逻辑都写在该文件中。
- 无 `__init__.py`、无 `pyproject.toml`、无 `setup.py`、无 `requirements.txt`、无 `package.json`、无 `Cargo.toml`。
- `SKILL/espdl-quantize/` 是一套独立的“技能”工具，用于更复杂的迭代式量化调优，但根脚本目前并未按 Skill 的 `user_quant.py` 契约编写。

## 主脚本详细说明

`quantize_espdl_fixed.py` 的关键行为：

1. **数据预处理**（与训练时一致）：
   ```python
   transforms.Resize(256)
   transforms.CenterCrop(224)
   transforms.ToTensor()
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ```

2. **校准数据加载**：
   - 使用 `torchvision.datasets.ImageFolder(root='dataset', transform=t)`
   - 从全部数据集中随机采样 300 张作为校准集
   - `batch_size=16`，`calib_steps=18`

3. **量化设置**：
   | 设置项 | 当前值 | 说明 |
   |--------|--------|------|
   | `fusion` | `True` | 算子融合优化 |
   | `equalization` | `True` | 跨层均衡化 |
   | `bias_correct` | `True` | 偏置校正 |
   | `blockwise_reconstruction` | `True` | 逐块重建优化 |
   | `lsq_optimization` | `False` | LSQ 优化（关闭） |

4. **量化调用**：
   - `espdl_quantize_onnx(...)`
   - `input_shape=[3, 224, 224]`
   - `target="esp32s3"`
   - `num_of_bits=8`
   - `device="cpu"`

5. **输出**：打印 `output/` 目录下各文件大小。

## 已知问题与注意事项

1. **输入模型文件名不一致**：
   - 脚本中写的是 `onnx_import_file="model_fixed.onnx"`
   - 项目实际存在的文件是 `model_espdl.onnx`
   - 运行前需要修改脚本中的文件名，或将模型重命名。

2. **数据集目录结构问题**：
   - `datasets.ImageFolder` 要求 `dataset/` 下按类别分子目录存放图片。
   - 当前 `dataset/` 下所有图片为扁平存放，直接运行脚本会报错。
   - 修复方式：在 `dataset/` 下创建类别子目录（如 `dataset/class_a/`、`dataset/class_b/`）并将图片移入，或改用自定义 `Dataset`。

3. **输出文件名**：
   - 脚本当前输出 `output/model.espdl`，但 AGENTS.md 历史版本提到同时输出 `output/model_int8.onnx`。
   - 实际是否生成 INT8 ONNX 取决于 `espdl_quantize_onnx` 的内部实现，脚本代码中未显式导出 INT8 ONNX 路径。

4. **环境一致性**：
   - 默认 `python` 与 `pip` 指向不同 Python 版本，建议在运行前使用对应环境的 `python -m pip` 安装/检查依赖。

## 测试策略

- 项目根目录没有单元测试、没有 CI/CD 配置。
- `SKILL/espdl-quantize/tests/` 下包含针对 Skill 自身脚本（`apply_setting.py`、`compare_iterations.py`、`run_iteration.py` 等）的测试，但这些测试与根目录主脚本无直接关联。
- 对主脚本最简单的验证方式是：修复数据集结构、修正模型文件名后执行 `python quantize_espdl_fixed.py`，观察是否成功生成 `output/model.espdl`。

## 开发约定

- 项目语言风格：脚本注释与文档使用中文。
- 保持脚本为单一入口，修改量化参数直接编辑 `quantize_espdl_fixed.py` 中的 `QuantizationSetting()` 配置。
- 数据预处理必须与训练时完全一致，任何改动都会影响量化精度。
- 如需进行更复杂的量化调参（混合精度、TQT、错误分析、迭代搜索），可参考并使用 `SKILL/espdl-quantize/` 中的高级工具；使用该 Skill 时需要按照 `SKILL/espdl-quantize/references/contract.md` 编写 `user_quant.py` 契约文件。

## 部署流程

1. 准备并整理好 `dataset/` 目录结构（ImageFolder 格式）。
2. 确认 `model_fixed.onnx` 存在，或修改脚本中的 `onnx_import_file` 为实际模型文件名。
3. 在正确安装依赖的 Python 环境中运行：
   ```bash
   python quantize_espdl_fixed.py
   ```
4. 检查 `output/` 目录下是否生成 `model.espdl`。
5. 将 `model.espdl` 部署到 ESP32-S3 设备对应的 ESP-DL 工程中使用。

## 安全与合规

- 脚本仅读取本地文件，不访问网络。
- 量化过程可能消耗较多 CPU 内存；若内存不足，可降低 `batch_size` 或 `calib_steps`。
- 当前 `device="cpu"`，如需使用 CUDA 加速，可将 `device` 改为 `"cuda"` 并确保 PyTorch CUDA 版本可用。
