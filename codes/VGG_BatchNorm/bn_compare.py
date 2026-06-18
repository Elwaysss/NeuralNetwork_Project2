"""
任务 2.2：对比 VGG-A 有 BN / 无 BN 的训练表现。

跑完会在 figures/ 下生成两张图：
  bn_compare_loss.png  训练 loss 曲线对比
  bn_compare_acc.png   验证准确率曲线对比
同时把曲线数据存到 results/bn_compare.npz，方便报告里复现。

用法：
  python bn_compare.py                # 完整训练（云端 NPU）
  python bn_compare.py --quick        # 本地 CPU 冒烟测试
"""
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


def run(args):
    device = get_device()
    print("device:", device)

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    n_items = 512 if args.quick else -1
    epochs = 2 if args.quick else args.epochs
    workers = 0 if args.quick else args.num_workers

    train_loader = get_cifar_loader(root="./data", train=True,
                                    batch_size=args.batch_size,
                                    num_workers=workers, n_items=n_items)
    val_loader = get_cifar_loader(root="./data", train=False,
                                  batch_size=args.batch_size,
                                  num_workers=workers, n_items=n_items)

    criterion = nn.CrossEntropyLoss()
    histories = {}

    for name, builder in [("VGG-A", VGG_A), ("VGG-A + BN", VGG_A_BatchNorm)]:
        print(f"\n==== training {name} ====")
        set_random_seeds(args.seed, device)
        model = builder()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        ckpt = os.path.join(RES_DIR, f"{name.replace(' ', '').replace('+', '_')}.pth")
        histories[name] = train(model, optimizer, criterion, train_loader,
                                val_loader, device, epochs_n=epochs,
                                best_model_path=ckpt)

    # 训练 loss 曲线
    plt.figure(figsize=(7, 5))
    for name, h in histories.items():
        plt.plot(range(1, len(h["train_curve"]) + 1), h["train_curve"], label=name)
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("VGG-A: with vs without BatchNorm (training loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "bn_compare_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 验证准确率曲线
    plt.figure(figsize=(7, 5))
    for name, h in histories.items():
        plt.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=name)
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")
    plt.title("VGG-A: with vs without BatchNorm (val accuracy)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "bn_compare_acc.png"), dpi=150, bbox_inches="tight")
    plt.close()

    np.savez(os.path.join(RES_DIR, "bn_compare.npz"),
             **{f"{k}_loss": np.array(v["train_curve"]) for k, v in histories.items()},
             **{f"{k}_acc": np.array(v["val_acc"]) for k, v in histories.items()})

    for name, h in histories.items():
        print(f"{name}: best val acc = {h['best_acc']:.4f}")
    print("figures saved to", FIG_DIR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--quick", action="store_true", help="本地 CPU 小数据冒烟测试")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
