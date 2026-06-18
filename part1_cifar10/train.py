"""
第一部分主训练脚本，所有消融都通过命令行参数控制，方便用 run.sh 批量跑。

例子：
  python train.py --tag baseline
  python train.py --tag width32 --width 32
  python train.py --tag gelu --activation gelu
  python train.py --tag ls --loss ce_ls
  python train.py --tag sgd --optimizer sgd
  python train.py --tag myadam --optimizer myadam
  python train.py --quick                      # 本地 CPU 冒烟
"""
import npu_compat  # noqa: F401  必须在 import torch 之前
import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import get_loaders
from device_utils import get_device
from engine import train_one_epoch, evaluate, measure_throughput
from models import MyCNN, count_parameters
from optimizers import MySGD, MyAdam

FIG_DIR = "figures"
RES_DIR = "results"


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "mysgd":
        return MySGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "myadam":
        return MyAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"未知优化器: {name}")


def build_criterion(name):
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "ce_ls":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    raise ValueError(f"未知损失: {name}")


def run(args):
    device = get_device()
    print("device:", device, "| tag:", args.tag)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    set_seed(args.seed, device)

    epochs = 2 if args.quick else args.epochs
    n_items = 512 if args.quick else -1
    workers = 0 if args.quick else args.num_workers

    train_loader, test_loader = get_loaders(
        root="./data", batch_size=args.batch_size, num_workers=workers,
        augment=not args.no_augment, n_items=n_items)

    model = MyCNN(width=args.width, activation=args.activation,
                  use_bn=not args.no_bn, use_residual=not args.no_residual,
                  use_dropout=not args.no_dropout).to(device)
    n_params = count_parameters(model)
    print(f"参数量: {n_params:,}")

    criterion = build_criterion(args.loss)
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = None
    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    sec_per_batch = measure_throughput(model, train_loader, criterion, optimizer, device)
    print(f"训练速度: {sec_per_batch * 1000:.1f} ms/batch")

    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["test_loss"].append(te_loss)
        hist["test_acc"].append(te_acc)
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), os.path.join(RES_DIR, f"{args.tag}_best.pth"))
        print(f"epoch {epoch + 1:3d}/{epochs}  "
              f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}  best={best_acc:.4f}")

    # 训练曲线
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, epochs + 1)
    ax[0].plot(ep, hist["train_loss"], label="train")
    ax[0].plot(ep, hist["test_loss"], label="test")
    ax[0].set_title(f"loss ({args.tag})"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, hist["train_acc"], label="train")
    ax[1].plot(ep, hist["test_acc"], label="test")
    ax[1].set_title(f"accuracy ({args.tag})"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, f"{args.tag}_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "tag": args.tag,
        "config": dict(width=args.width, activation=args.activation,
                       use_bn=not args.no_bn, use_residual=not args.no_residual,
                       use_dropout=not args.no_dropout, loss=args.loss,
                       optimizer=args.optimizer, lr=args.lr,
                       weight_decay=args.weight_decay, epochs=epochs),
        "n_params": n_params,
        "sec_per_batch": sec_per_batch,
        "best_test_acc": best_acc,
        "best_test_err": 1 - best_acc,
        "history": hist,
    }
    with open(os.path.join(RES_DIR, f"{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{args.tag}] best test acc = {best_acc:.4f}  (err = {1 - best_acc:.4f})")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--activation", type=str, default="relu",
                   choices=["relu", "leaky_relu", "gelu"])
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "ce_ls"])
    p.add_argument("--optimizer", type=str, default="sgd",
                   choices=["sgd", "adam", "adamw", "mysgd", "myadam"])
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--no_bn", action="store_true")
    p.add_argument("--no_residual", action="store_true")
    p.add_argument("--no_dropout", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
