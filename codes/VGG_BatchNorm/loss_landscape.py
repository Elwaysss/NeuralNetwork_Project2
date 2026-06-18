"""
任务 2.3：BN 如何让优化更稳定 —— 损失景观。

做法（对应 PDF 2.3.1）：
  1. 选一组学习率，分别训练 VGG-A 与 VGG-A+BN；
  2. 记录每个训练步的 loss；
  3. 同一步上，对所有学习率取 loss 的最大/最小值，得到 max_curve / min_curve；
  4. 用 fill_between 把两条曲线之间的区域填上，有 BN / 无 BN 画在同一张图。
额外还画了梯度可预测性（相邻步梯度差）和 beta-smoothness 的对比。

用法：
  python loss_landscape.py                 # 完整（云端 NPU）
  python loss_landscape.py --quick         # 本地 CPU 冒烟测试
"""
import npu_compat  # noqa: F401  必须在 import torch 之前
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from device_utils import get_device
from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm
from VGG_Loss_Landscape import set_random_seeds, train

FIG_DIR = "figures"
RES_DIR = "results"


def min_max_curve(step_lists):
    """输入若干条等长曲线，逐步取最小/最大。长度不一时按最短对齐。"""
    n = min(len(s) for s in step_lists)
    arr = np.array([s[:n] for s in step_lists])
    return arr.min(axis=0), arr.max(axis=0)


def train_over_lrs(builder, lrs, loaders, device, epochs, seed):
    """对一组学习率训练同一种网络，收集每个 lr 的 step 级 loss / 梯度差。"""
    train_loader, val_loader = loaders
    criterion = nn.CrossEntropyLoss()
    loss_runs, grad_runs, beta_runs = [], [], []
    for lr in lrs:
        print(f"    lr = {lr}")
        set_random_seeds(seed, device)
        model = builder()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        h = train(model, optimizer, criterion, train_loader, val_loader,
                  device, epochs_n=epochs)
        loss_runs.append(h["loss_steps"])
        grad_runs.append(h["grad_diff"])
        beta_runs.append(h["beta_steps"])
    return loss_runs, grad_runs, beta_runs


def fill_plot(ax, runs, color, label):
    lo, hi = min_max_curve(runs)
    steps = np.arange(len(lo))
    ax.fill_between(steps, lo, hi, color=color, alpha=0.35, label=label)
    ax.plot(steps, lo, color=color, lw=0.8)
    ax.plot(steps, hi, color=color, lw=0.8)


def run(args):
    device = get_device()
    print("device:", device)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    lrs = [1e-3, 2e-3, 1e-4, 5e-4]
    n_items = 512 if args.quick else -1
    epochs = 1 if args.quick else args.epochs
    workers = 0 if args.quick else args.num_workers
    if args.quick:
        lrs = [1e-3, 2e-3]

    train_loader = get_cifar_loader(root="./data", train=True,
                                    batch_size=args.batch_size,
                                    num_workers=workers, n_items=n_items)
    val_loader = get_cifar_loader(root="./data", train=False,
                                  batch_size=args.batch_size,
                                  num_workers=workers, n_items=n_items)
    loaders = (train_loader, val_loader)

    print("learning rates:", lrs)
    print("\n==== VGG-A (no BN) ====")
    std_loss, std_grad, std_beta = train_over_lrs(VGG_A, lrs, loaders, device, epochs, args.seed)
    print("\n==== VGG-A + BN ====")
    bn_loss, bn_grad, bn_beta = train_over_lrs(VGG_A_BatchNorm, lrs, loaders, device, epochs, args.seed)

    # 损失景观对比
    fig, ax = plt.subplots(figsize=(8, 5))
    fill_plot(ax, std_loss, "tab:green", "Standard VGG")
    fill_plot(ax, bn_loss, "tab:red", "Standard VGG + BatchNorm")
    ax.set_xlabel("steps")
    ax.set_ylabel("loss landscape")
    ax.set_title("Loss landscape: BN smooths the optimization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, "loss_landscape.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 梯度可预测性对比
    fig, ax = plt.subplots(figsize=(8, 5))
    fill_plot(ax, std_grad, "tab:green", "Standard VGG")
    fill_plot(ax, bn_grad, "tab:red", "Standard VGG + BatchNorm")
    ax.set_xlabel("steps")
    ax.set_ylabel("gradient difference (L2)")
    ax.set_title("Gradient predictiveness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, "gradient_predictiveness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # beta-smoothness 对比
    fig, ax = plt.subplots(figsize=(8, 5))
    fill_plot(ax, std_beta, "tab:green", "Standard VGG")
    fill_plot(ax, bn_beta, "tab:red", "Standard VGG + BatchNorm")
    ax.set_xlabel("steps")
    ax.set_ylabel("effective beta-smoothness")
    ax.set_title('"Effective" beta-smoothness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, "beta_smoothness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(os.path.join(RES_DIR, "loss_landscape.npz"),
             lrs=np.array(lrs),
             std_loss=np.array([r[:min(len(x) for x in std_loss)] for r in std_loss]),
             bn_loss=np.array([r[:min(len(x) for x in bn_loss)] for r in bn_loss]))
    print("figures saved to", FIG_DIR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--quick", action="store_true", help="本地 CPU 小数据冒烟测试")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
