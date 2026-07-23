【任务要求】
请帮我使用ESP-PPQ工具，完成以下所有步骤：

1. **环境准备**：
   - 检查并安装 `esp-ppq` 库[reference:3][reference:4]。
   - 确保Python版本为3.10或以下（ESP-PPQ对更高版本兼容性不佳）。

2. **准备校准数据集 (Calibration Dataset)**：
   - 从我的训练数据中，取一小部分（大约100-200张）图片作为校准数据集[reference:5]。
   - 校准数据的预处理必须与训练时完全一致：`Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`[reference:6]。
   - 创建一个DataLoader来加载这些校准数据。

3. **模型量化 (Quantization)**：
   - 使用 `esp_ppq.api` 中的 `espdl_quantize_onnx` 接口[reference:7]。
   - 将 `target` 参数设置为 `"esp32s3"`[reference:8][reference:9]。
   - 量化位宽设置为 8 位（INT8）[reference:10]。
   - 编写一个量化脚本 `quantize.py`，并使用 `espdl_quantize_onnx` 函数将 `model.onnx` 量化为 ESP-DL 专用的 `.espdl` 格式[reference:11]。

4. **精度验证 (Accuracy Validation)**：
   - 在量化前后，分别对模型进行推理，并对比其在验证集上的准确率。
   - 生成一份简短的报告，说明量化带来的精度损失（如果存在）。

5. **输出文件**：
   - 运行 `quantize.py` 脚本，并确保成功生成 `model.espdl` 文件。
   - 提供 `quantize.py` 的完整源代码。
   - 给出量化前后的模型大小和推理速度对比。