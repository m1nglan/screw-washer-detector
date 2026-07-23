"""将 INT8 ONNX 权重提取为 ESP-DL C++ 张量定义"""
import onnx
import numpy as np
import os

ONNX_PATH = "output/model_int8.onnx"
OUT_DIR = "deploy/main/tensors"

model = onnx.load(ONNX_PATH)
os.makedirs(OUT_DIR, exist_ok=True)

# 读取所有权重
weights = {}
for init in model.graph.initializer:
    arr = onnx.numpy_helper.to_array(init)
    weights[init.name] = arr

# 提取算子结构
print(f"模型: {ONNX_PATH}")
print(f"权重数量: {len(weights)}")

# 统计算子
ops = {}
for node in model.graph.node:
    t = node.op_type
    ops[t] = ops.get(t, 0) + 1

for op, cnt in sorted(ops.items()):
    print(f"  {op}: {cnt}")

# 生成权重文件清单
print(f"\n权重导出到: {OUT_DIR}/")
for name, arr in sorted(weights.items()):
    fname = name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".npy"
    np.save(os.path.join(OUT_DIR, fname), arr)

print(f"完成! 共 {len(weights)} 个权重文件")
