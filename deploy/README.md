# Screw/Washer Detector - ESP32-S3 Deployment

## 文件说明

```
deploy/
├── CMakeLists.txt          # ESP-IDF 项目文件
├── main/
│   ├── CMakeLists.txt      # 组件编译配置
│   ├── Kconfig.projbuild   # 菜单配置 (阈值/摄像头引脚)
│   ├── main.cpp            # 主程序模板
│   ├── model_data.h        # 模型数据结构定义
│   ├── model_data.cpp      # 模型二进制数据 (2354.4 KB)
│   └── labels.json         # 标签映射
```

## 模型信息

| 项目 | 值 |
|------|-----|
| 网络 | MobileNetV2 (INT8) |
| 输入 | 224×224×3 (RGB) |
| 输出 | 2 类 (screw, washer) |
| 模型大小 | 2354.4 KB |
| 参数量 | 2,226,434 |
| FP32 准确率 | 98.88% |
| INT8 准确率 | 98.31% |

## 部署步骤

### 1. 安装 ESP-IDF

参考: https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/

### 2. 配置项目

```bash
cd deploy
idf.py set-target esp32s3
idf.py menuconfig
# -> Screw/Washer Detector -> 配置摄像头引脚和置信度阈值
```

### 3. 实现推理引擎

当前 `main.cpp` 为模板代码，需要根据实际使用的推理引擎实现：

**方案 A: ESP-DL (推荐)**
- 将 INT8 权重转换为 ESP-DL 格式
- 参考: https://github.com/espressif/esp-dl

**方案 B: ONNX Runtime for ESP**
- 使用 onnxruntime 嵌入式库
- 直接加载 model.onnx 进行推理

**方案 C: 自定义推理**
- 使用 ESP-NN 加速库手动实现 MobileNetV2 推理
- 参考: https://github.com/espressif/esp-nn

### 4. 编译烧录

```bash
idf.py build
idf.py -p PORT flash monitor
```

## 摄像头连接

推荐 OV2640 摄像头模块，引脚在 `menuconfig` 中配置。
