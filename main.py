import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, mobilenet_v2
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from tqdm import tqdm

# ========== 配置参数 ==========
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005
IMG_SIZE = 224
NUM_CLASSES = 2
VAL_SPLIT = 0.15  # 15% 验证集
UNFREEZE_LAST_N_BLOCKS = 3  # 解冻最后 N 个 Bottleneck 块进行微调
DATASET_ROOT = "dataset"
OUTPUT_DIR = "output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 中文文件夹名 → 英文标签映射
CLASS_NAME_MAP = {
    "螺母": "washer",
    "螺丝": "screw",
}

# ========== 工具函数 ==========
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_class_mapping(dataset):
    """获取 ImageFolder 类别索引到英文标签的映射"""
    idx_to_class = {}
    for cls_name, cls_idx in dataset.class_to_idx.items():
        english_name = CLASS_NAME_MAP.get(cls_name, cls_name)
        idx_to_class[cls_idx] = english_name
    return idx_to_class

# ========== 数据加载 ==========
def load_data():
    """加载全部数据进行分层抽样"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载原始数据集获取索引
    raw_dataset = datasets.ImageFolder(root=DATASET_ROOT)
    idx_to_class = get_class_mapping(raw_dataset)
    print(f"检测到类别: {raw_dataset.class_to_idx}")
    print(f"标签映射: {idx_to_class}")

    targets = [s for _, s in raw_dataset.samples]

    # 分层抽样：每类按 VAL_SPLIT 比例划分
    train_indices = []
    val_indices = []

    for cls_idx in range(len(raw_dataset.classes)):
        cls_indices = [i for i, t in enumerate(targets) if t == cls_idx]
        np.random.shuffle(cls_indices)

        n_val = max(1, int(len(cls_indices) * VAL_SPLIT))
        n_train = len(cls_indices) - n_val

        actual_train = cls_indices[:n_train]
        actual_val = cls_indices[n_train:]

        train_indices.extend(actual_train)
        val_indices.extend(actual_val)

        english_name = idx_to_class[cls_idx]
        print(f"  {english_name}: 训练 {len(actual_train)} 张, 验证 {len(actual_val)} 张 (共 {len(cls_indices)} 张)")

    # 创建带数据增强的数据集
    full_train_dataset = datasets.ImageFolder(root=DATASET_ROOT, transform=train_transform)
    full_val_dataset = datasets.ImageFolder(root=DATASET_ROOT, transform=val_transform)

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")
    return train_loader, val_loader, idx_to_class

# ========== 模型构建 ==========
def build_model(num_classes):
    """构建 MobileNetV2 迁移学习模型，解冻最后若干层微调"""
    print("\n正在加载 MobileNetV2 预训练模型...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后 UNFREEZE_LAST_N_BLOCKS 个 Bottleneck 块
    if UNFREEZE_LAST_N_BLOCKS > 0:
        blocks = list(model.features.children())
        for block in blocks[-UNFREEZE_LAST_N_BLOCKS:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"  解冻最后 {UNFREEZE_LAST_N_BLOCKS} 个 Bottleneck 块进行微调")

    # 替换最后的全连接层
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # 全连接层可训练
    for param in model.classifier[1].parameters():
        param.requires_grad = True

    return model

# ========== 训练与验证 ==========
def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{total_epochs}] 训练", leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="验证", leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# ========== 绘图 ==========
def plot_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax1.plot(epochs, train_losses, "b-", label="训练损失")
    ax1.plot(epochs, val_losses, "r-", label="验证损失")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("训练与验证损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, train_accs, "b-", label="训练准确率")
    ax2.plot(epochs, val_accs, "r-", label="验证准确率")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("训练与验证准确率曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"\n训练曲线已保存到: {curve_path}")

# ========== ONNX 导出 ==========
def export_onnx(model, output_dir, idx_to_class):
    model.eval().to("cpu")
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    onnx_path = os.path.join(output_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"ONNX 模型已导出到: {onnx_path}")

    # 写入标签映射文件
    label_path = os.path.join(output_dir, "labels.json")
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)
    print(f"标签映射已保存到: {label_path}")

    # 将模型移回设备
    model.to(DEVICE)

# ========== 主函数 ==========
def main():
    set_seed()

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 设备信息
    print(f"使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    train_loader, val_loader, idx_to_class = load_data()

    # 构建模型
    model = build_model(NUM_CLASSES)
    model = model.to(DEVICE)

    # 打印模型参数量
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 估算模型大小（float32）
    model_size_mb = total_params * 4 / (1024 * 1024)
    print(f"模型预估大小: {model_size_mb:.2f} MB")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    # 学习率调度：每 10 个 epoch 衰减为 0.5 倍
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练记录
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    start_time = time.time()

    print(f"\n{'='*50}")
    print(f"开始训练 (共 {EPOCHS} 个 epoch)")
    print(f"{'='*50}\n")

    for epoch in range(EPOCHS):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, EPOCHS)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 打印本 epoch 结果
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] 完成 | "
              f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% | "
              f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}% | "
              f"最佳验证准确率: {best_val_acc:.2f}% | "
              f"LR: {current_lr:.6f}\n")

    total_time = time.time() - start_time
    print(f"{'='*50}")
    print(f"训练完成! 总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"{'='*50}")

    # 绘制曲线
    plot_curves(train_losses, val_losses, train_accs, val_accs, OUTPUT_DIR)

    # 导出 ONNX
    export_onnx(model, OUTPUT_DIR, idx_to_class)

    # 最终模型保存
    final_path = os.path.join(OUTPUT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"最终模型已保存到: {final_path}")

    # ========== 训练报告 ==========
    print(f"\n{'='*50}")
    print("训练报告")
    print(f"{'='*50}")
    print(f"  模型架构: MobileNetV2 (迁移学习)")
    print(f"  分类类别: {NUM_CLASSES} 类 ({', '.join(idx_to_class.values())})")
    print(f"  训练设备: {DEVICE}")
    print(f"  批量大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  模型大小: {model_size_mb:.2f} MB")
    print(f"  总训练耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  最终验证准确率: {val_acc:.2f}%")
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  输出目录: {OUTPUT_DIR}/")
    print(f"    - best_model.pth (最佳权重)")
    print(f"    - final_model.pth (最终权重)")
    print(f"    - model.onnx (ONNX 格式)")
    print(f"    - labels.json (标签映射)")
    print(f"    - training_curves.png (训练曲线)")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
