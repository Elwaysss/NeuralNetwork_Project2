"""
VGG-A 训练与损失景观相关的工具函数。

这个文件被 bn_compare.py / loss_landscape.py 调用，提供：
  - set_random_seeds  固定随机种子
  - get_accuracy      在某个 loader 上算分类准确率
  - train             训练一个模型，按 step 记录 loss、梯度变化、beta-smoothness

为了有/无 BN 的对比公平，train 里按"训练步(step)"而不是"epoch"记录 loss，
这样画损失景观时横轴就是优化步数。
"""
import os
import random

import numpy as np
import torch
from torch import nn

from device_utils import get_device


def set_random_seeds(seed_value=0, device="cpu"):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if "npu" in str(device):
        torch.npu.manual_seed(seed_value)
        torch.npu.manual_seed_all(seed_value)
    elif "cuda" in str(device):
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def train(model, optimizer, criterion, train_loader, val_loader, device,
          scheduler=None, epochs_n=20, best_model_path=None):
    """训练模型并按 step 记录用于损失景观分析的量。

    返回一个 dict：
      loss_steps   每个训练步的 loss（用于损失景观 / max-min 曲线）
      grad_diff    相邻两步 classifier[4].weight 梯度的 L2 距离（梯度可预测性）
      beta_steps   grad_diff / 权重变化距离（beta-smoothness 的近似）
      train_curve  每个 epoch 的平均训练 loss
      val_acc      每个 epoch 的验证准确率
    """
    model.to(device)

    loss_steps = []
    grad_diff = []
    beta_steps = []
    train_curve = []
    val_acc = []

    prev_grad = None
    prev_weight = None
    best_acc = 0.0

    for epoch in range(epochs_n):
        model.train()
        running_loss = 0.0
        n_batch = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            # 记录最后一层全连接的梯度，用来衡量梯度的"可预测性"
            grad = model.classifier[4].weight.grad.detach().clone().flatten()
            weight = model.classifier[4].weight.detach().clone().flatten()
            if prev_grad is not None:
                gd = torch.norm(grad - prev_grad).item()
                wd = torch.norm(weight - prev_weight).item()
                grad_diff.append(gd)
                beta_steps.append(gd / (wd + 1e-12))
            prev_grad = grad
            prev_weight = weight

            optimizer.step()

            loss_steps.append(loss.item())
            running_loss += loss.item()
            n_batch += 1

        if scheduler is not None:
            scheduler.step()

        train_curve.append(running_loss / max(n_batch, 1))
        acc = get_accuracy(model, val_loader, device)
        val_acc.append(acc)

        if best_model_path is not None and acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)

        print(f"  epoch {epoch + 1:2d}/{epochs_n}  "
              f"train_loss={train_curve[-1]:.4f}  val_acc={acc:.4f}")

    return {
        "loss_steps": loss_steps,
        "grad_diff": grad_diff,
        "beta_steps": beta_steps,
        "train_curve": train_curve,
        "val_acc": val_acc,
        "best_acc": best_acc,
    }


if __name__ == "__main__":
    # 单模型冒烟测试：本地 CPU 上用很小的数据跑一两轮，确认能跑通
    from models.vgg import VGG_A
    from data.loaders import get_cifar_loader

    device = get_device()
    print("device:", device)
    set_random_seeds(2020, device)

    train_loader = get_cifar_loader(root="./data", train=True, n_items=256, num_workers=0)
    val_loader = get_cifar_loader(root="./data", train=False, n_items=256, num_workers=0)

    model = VGG_A()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    out = train(model, optimizer, criterion, train_loader, val_loader, device, epochs_n=1)
    print("steps recorded:", len(out["loss_steps"]))
