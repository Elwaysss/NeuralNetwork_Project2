"""
任务 1 第 6 条：网络的可视化与可解释性。
包含三块：
  1. 第一层卷积核可视化
  2. 损失景观（filter-normalized 二维切片，参考 Li et al. 2018）
  3. Grad-CAM 看模型在图像哪块区域做决策

用法（需要先有训练好的权重）：
  python visualize.py --ckpt results/baseline_best.pth
  python visualize.py --ckpt results/baseline_best.pth --quick
"""
import npu_compat  # noqa: F401  必须在 import torch 之前
import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_loaders, MEAN, STD
from device_utils import get_device
from models import MyCNN

FIG_DIR = "figures"
CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_model(ckpt, device):
    model = MyCNN().to(device)
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("loaded:", ckpt)
    else:
        print("没有找到权重，使用随机初始化的模型（仅用于本地跑通流程）")
    model.eval()
    return model


def first_conv(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return m
    return None


def plot_filters(model, path):
    conv = first_conv(model)
    w = conv.weight.detach().cpu()
    n = min(64, w.size(0))
    w = w[:n]
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)  # 归一化到 0-1 好显示
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(w[i].permute(1, 2, 0).numpy())
    fig.suptitle("First conv-layer filters")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- 损失景观 ----------
def rand_like_params(params):
    return [torch.randn_like(p) for p in params]


def filter_normalize(direction, params):
    """按 Li et al. 的做法：让随机方向的每个滤波器范数对齐到权重的范数。"""
    for d, p in zip(direction, params):
        if p.dim() >= 2:  # conv / linear 权重，按输出维逐 filter 归一化
            for i in range(p.size(0)):
                d[i] *= p[i].norm() / (d[i].norm() + 1e-10)
        else:
            d.mul_(p.norm() / (d.norm() + 1e-10))


@torch.no_grad()
def loss_at(model, params0, d1, d2, a, b, batch, criterion, device):
    for p, p0, x1, x2 in zip(model.parameters(), params0, d1, d2):
        p.copy_(p0 + a * x1 + b * x2)
    x, y = batch
    return criterion(model(x), y).item()


def plot_loss_landscape(model, loader, device, path, grid=21, span=1.0):
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters()]
    params0 = [p.detach().clone() for p in params]
    d1, d2 = rand_like_params(params), rand_like_params(params)
    filter_normalize(d1, params0)
    filter_normalize(d2, params0)

    x, y = next(iter(loader))
    batch = (x.to(device), y.to(device))

    coords = np.linspace(-span, span, grid)
    Z = np.zeros((grid, grid))
    for i, a in enumerate(coords):
        for j, b in enumerate(coords):
            Z[i, j] = loss_at(model, params0, d1, d2, a, b, batch, criterion, device)
    # 还原参数
    with torch.no_grad():
        for p, p0 in zip(params, params0):
            p.copy_(p0)

    A, B = np.meshgrid(coords, coords)
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(A, B, Z.T, levels=30, cmap="viridis")
    ax.contour(A, B, Z.T, levels=15, colors="k", linewidths=0.3)
    fig.colorbar(cs, ax=ax, label="loss")
    ax.set_title("Loss landscape (filter-normalized)")
    ax.set_xlabel("direction 1"); ax.set_ylabel("direction 2")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- Grad-CAM ----------
def last_conv(model):
    conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv = m
    return conv


def grad_cam(model, loader, device, path, n_show=6):
    target_layer = last_conv(model)
    feats, grads = {}, {}

    def fwd_hook(_, __, out):
        feats["v"] = out.detach()

    def bwd_hook(_, gin, gout):
        grads["v"] = gout[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x, y = next(iter(loader))
    x, y = x[:n_show].to(device), y[:n_show].to(device)
    model.zero_grad()
    out = model(x)
    pred = out.argmax(1)
    out.gather(1, pred.view(-1, 1)).sum().backward()

    fmap = feats["v"]
    grad = grads["v"]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * fmap).sum(dim=1))
    cam = F.interpolate(cam.unsqueeze(1), size=(32, 32), mode="bilinear",
                        align_corners=False).squeeze(1)
    cam = cam.cpu().numpy()

    mean = np.array(MEAN).reshape(3, 1, 1)
    std = np.array(STD).reshape(3, 1, 1)
    imgs = (x.cpu().numpy() * std + mean).clip(0, 1)

    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
    for i in range(n_show):
        axes[0, i].imshow(imgs[i].transpose(1, 2, 0))
        axes[0, i].set_title(f"{CLASSES[pred[i]]}", fontsize=9)
        axes[0, i].axis("off")
        c = cam[i]
        c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        axes[1, i].imshow(imgs[i].transpose(1, 2, 0))
        axes[1, i].imshow(c, cmap="jet", alpha=0.5)
        axes[1, i].axis("off")
    fig.suptitle("Grad-CAM (top: input, bottom: attention)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    h1.remove(); h2.remove()


def run(args):
    device = get_device()
    print("device:", device)
    os.makedirs(FIG_DIR, exist_ok=True)
    workers = 0 if args.quick else args.num_workers
    n_items = 256 if args.quick else 1024
    _, test_loader = get_loaders(root="./data", batch_size=128,
                                 num_workers=workers, augment=False, n_items=n_items)
    model = load_model(args.ckpt, device)

    plot_filters(model, os.path.join(FIG_DIR, "filters.png"))
    print("filters.png done")
    grad_cam(model, test_loader, device, os.path.join(FIG_DIR, "gradcam.png"))
    print("gradcam.png done")
    grid = 9 if args.quick else 25
    plot_loss_landscape(model, test_loader, device,
                        os.path.join(FIG_DIR, "loss_landscape_2d.png"), grid=grid)
    print("loss_landscape_2d.png done")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="results/baseline_best.pth")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
