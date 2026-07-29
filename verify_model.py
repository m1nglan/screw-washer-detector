"""
verify_model.py
===============
在 PC 上使用与 ESP32 驱动完全相同的预处理逻辑，验证 INT8 ONNX 模型
是否能正确区分 t1.jpg（螺丝）和 t2.jpg（垫圈）。

预处理公式（与 classifier_driver.cpp 完全一致）：
    quantized = clamp(round(((pixel/255 - mean) / std) / scale), -128, 127)

其中:
    mean = [0.485, 0.456, 0.406]     (ImageNet 标准)
    std  = [0.229, 0.224, 0.225]     (ImageNet 标准)
    scale = 0.03125 = 2^(-5)          (模型 exponent=-5)

Resize 策略（与驱动一致）：
    1) 短边缩放到 256（保持宽高比）
    2) CenterCrop 到 224×224

布局：NCHW [3, 224, 224]

用法:
    python verify_model.py

依赖:
    pip install onnxruntime pillow numpy
"""

import numpy as np
from PIL import Image
import os

# ====== 配置 ======
MODEL_PATH = "output/model_espdl.onnx"  # FP32 ONNX
IMG_DIR = "."
IMG1 = "t1.jpg"                       # 螺丝
IMG2 = "t2.jpg"                       # 垫圈

# 与 C++ 驱动完全一致的参数
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SCALE = 0.03125                       # 2^(-5)
TARGET_SHORT = 256                    # 短边缩放目标
TARGET_SIZE = 224                     # 最终裁剪尺寸
INPUT_SHAPE = (1, 224, 224, 3)        # ONNX 输入 [N, H, W, C] (NHWC)


def preprocess(image_path: str) -> np.ndarray:
    """
    与 ESP32 driver 完全相同的预处理逻辑。
    返回: 量化后的 numpy 数组，shape=(3, 224, 224), dtype=np.int8
    """
    # 1) 加载图片 → RGB
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    print(f"  原图尺寸: {w}x{h}")

    # 2) Resize: 短边 → 256，保持宽高比
    if w < h:
        new_w = TARGET_SHORT
        new_h = int(h * (TARGET_SHORT / w))
    else:
        new_w = int(w * (TARGET_SHORT / h))
        new_h = TARGET_SHORT
    img = img.resize((new_w, new_h), Image.BILINEAR)
    print(f"  缩放后: {new_w}x{new_h}")

    # 3) CenterCrop → 224×224
    left = (new_w - TARGET_SIZE) // 2
    top = (new_h - TARGET_SIZE) // 2
    img = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))
    print(f"  CenterCrop: ({left},{top}) → {TARGET_SIZE}x{TARGET_SIZE}")

    # 4) 转 numpy → [H, W, C], [0, 255], float32
    pixels = np.array(img, dtype=np.float32)  # shape: (224, 224, 3)

    # 调试：打印前 4 像素（与 ESP32 日志对比）
    r = pixels[:2, :2, 0].flatten()
    g = pixels[:2, :2, 1].flatten()
    b = pixels[:2, :2, 2].flatten()
    print(f"  前4像素 R={r[0]:.0f} {r[1]:.0f} {r[2]:.0f} {r[3]:.0f} "
          f"G={g[0]:.0f} {g[1]:.0f} {g[2]:.0f} {g[3]:.0f} "
          f"B={b[0]:.0f} {b[1]:.0f} {b[2]:.0f} {b[3]:.0f}")

    # 5) 归一化: pixel/255
    pixels /= 255.0

    # 6) 标准化: (x - mean) / std
    pixels = (pixels - MEAN) / STD

    # 7) 量化: / scale → round → clamp [-128, 127]
    pixels = pixels / SCALE
    pixels = np.round(pixels)
    pixels = np.clip(pixels, -128, 127)

    # 8) 保持 HWC 布局（不转置）
    quantized = pixels.astype(np.int8)  # shape: (224, 224, 3) NHWC

    print(f"  量化后 ch0[0..3]: {quantized[0, 0, 0]} {quantized[0, 1, 0]} {quantized[0, 2, 0]} {quantized[0, 3, 0]}")
    print(f"  量化后 ch1[0..3]: {quantized[0, 0, 1]} {quantized[0, 1, 1]} {quantized[0, 2, 1]} {quantized[0, 3, 1]}")
    print(f"  量化后 ch2[0..3]: {quantized[0, 0, 2]} {quantized[0, 1, 2]} {quantized[0, 2, 2]} {quantized[0, 3, 2]}")

    return quantized


def softmax(x):
    """数值稳定的 softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def run_inference(model_path, quantized_input):
    """用 ONNX Runtime 推理（INT8→反量化→喂 FP32 模型）"""
    import onnxruntime as ort

    print(f"\n加载模型: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    input_type = session.get_inputs()[0].type
    print(f"  输入节点: {input_name}, 类型: {input_type}")

    # INT8 预处理结果 → 反量化回 FP32（模型内部采用 NHWC）
    input_data = quantized_input[np.newaxis, ...].astype(np.float32) * SCALE  # [1,224,224,3]

    result = session.run([output_name], {input_name: input_data})[0]
    return result


def main():
    """主测试流程"""
    print("=" * 60)
    print("ESP32 预处理一致性验证脚本")
    print("=" * 60)

    image_paths = [os.path.join(IMG_DIR, IMG1), os.path.join(IMG_DIR, IMG2)]
    labels = ["screw (螺丝)", "washer (垫圈)"]

    # ===== 预处理两张图片 =====
    quantized_inputs = []
    for path, label in zip(image_paths, labels):
        print(f"\n--- 预处理: {os.path.basename(path)} ({label}) ---")
        q = preprocess(path)
        quantized_inputs.append(q)

    # 验证两张图的输入是否不同
    diff = np.abs(quantized_inputs[0].astype(int) - quantized_inputs[1].astype(int))
    print(f"\n两图输入差异: max={diff.max()}, mean={diff.mean():.2f}, "
          f"pixels_different={np.count_nonzero(diff)}/{224*224*3}")

    # ===== ONNX 推理 =====
    if not os.path.exists(MODEL_PATH):
        print(f"\n[跳过推理] ONNX 模型不存在: {MODEL_PATH}")
        print("请将 model.onnx 放到项目根目录，或修改 MODEL_PATH。\n")
        print("对比 ESP32 日志，验证预处理是否一致：")
        print("  ESP32 t1 ch0[0..3]: 11 11 12 12  (有 CenterCrop)")
        print("  ESP32 t1 ch1[0..3]: 14 14 16 16")
        print("  ESP32 t2 ch0[0..3]: 17 17 17 17")
        print("  ESP32 t2 ch1[0..3]: 21 21 21 21")
        return

    print("\n" + "=" * 60)
    print("ONNX 推理结果")
    print("=" * 60)

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        print(f"\n--- {os.path.basename(path)} ({label}) ---")
        output = run_inference(MODEL_PATH, quantized_inputs[i])
        output = output.flatten()

        print(f"  原始输出: {output.tolist()}")

        # Softmax 概率
        probs = softmax(output)
        for j, p in enumerate(probs):
            cls_name = "screw" if j == 0 else "washer"
            print(f"  {cls_name}: score={output[j]:.4f}, prob={p*100:.4f}%")

        pred = np.argmax(probs)
        pred_label = "screw" if pred == 0 else "washer"
        print(f"  >>> 预测: {pred_label} (class_id={pred}, prob={probs[pred]*100:.2f}%)")


if __name__ == "__main__":
    main()
