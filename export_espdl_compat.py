"""
ESP-DL 兼容的 ONNX 导出脚本
在保存 best_model.pth 后运行此脚本，将权重转换后导出
不需要重新训练！

用法: python export_espdl_compat.py
"""
import torch
import torch.nn as nn
import onnx
import os
import json

# 配置
IMG_SIZE = 224    # 可以先保持224，测试通过再改小
NUM_CLASSES = 2
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
ONNX_OUT = os.path.join(OUTPUT_DIR, "model_espdl.onnx")

def build_espdl_model(num_classes):
    """构建和训练时一样的backbone，但classifier换成Conv2d"""
    from torchvision.models import mobilenet_v2
    
    model = mobilenet_v2(weights=None)  # 不加载预训练权重
    model.classifier[1] = nn.Linear(1280, num_classes)  # 先和训练时保持一致
    
    return model

def convert_linear_to_conv(model):
    """
    把训练好的 Linear 分类器权重转成 Conv2d
    Linear(1280, 2) → Conv2d(1280, 2, kernel_size=1)
    """
    # 获取训练好的Linear层权重
    old_fc = model.classifier[1]
    weight = old_fc.weight.data  # shape: [2, 1280]
    bias = old_fc.bias.data      # shape: [2]
    
    # 创建Conv2d替代
    conv = nn.Conv2d(1280, NUM_CLASSES, kernel_size=1)
    
    # 转换权重: [out, in] → [out, in, 1, 1]
    conv.weight.data = weight.unsqueeze(-1).unsqueeze(-1)
    conv.bias.data = bias
    
    # 自定义 forward：把 features 的输出保持 4D，不 flatten
    class EspdlMobileNet(nn.Module):
        def __init__(self, features, classifier_conv):
            super().__init__()
            self.features = features
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # [1,1280,7,7] -> [1,1280,1,1]
                classifier_conv,           # [1,1280,1,1] -> [1,2,1,1]
            )
            # 用 Reshape 替代 Squeeze/Flatten（通过 view 实现）
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)  # [1,2,1,1]
            return x  # 保持 4D 输出
    
    return EspdlMobileNet(model.features, conv)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 构建和训练时一样的模型并加载权重
    print("加载训练好的模型权重...")
    model = build_espdl_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ 权重加载成功: {MODEL_PATH}")
    
    # 2. 转换classifier为Conv2d（不丢失权重信息）
    model = convert_linear_to_conv(model)
    model.eval().to("cpu")
    
    # 3. 验证输出是否一致（忽略形状，只验证数值）
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        output = model(dummy)
    print(f"模型输出形状: {output.shape}")
    
    # 4. 导出ONNX
    torch.onnx.export(
        model,
        dummy,
        ONNX_OUT,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"✅ ONNX 导出成功: {ONNX_OUT}")
    
    # 5. 验证
    onnx_model = onnx.load(ONNX_OUT)
    onnx.checker.check_model(onnx_model)
    
    # 6. 检查算子
    ops = {}
    for node in onnx_model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"\n算子统计: {ops}")
    
    unsupported = {'Flatten', 'Gemm', 'BatchNormalization', 'Softmax'}
    found = [op for op in ops if op in unsupported]
    if found:
        print(f"⚠️ 仍不兼容: {found}")
    else:
        print("✅ 全部算子 ESP-DL 兼容！可以直接用 ESP-PPQ 转换！")
    
    # 打印模型信息
    print(f"\n输入: 1x3x{IMG_SIZE}x{IMG_SIZE}")
    print(f"输出: {[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")
    size_mb = os.path.getsize(ONNX_OUT) / (1024 * 1024)
    print(f"大小: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
