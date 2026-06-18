#!/bin/bash
# 把所有消融重跑到与基线相同的 epoch（默认 25），使对比完全同条件。
# 基线 baseline 与可视化已有，无需重跑；本脚本只覆盖 13 个消融的 results/figures。
#   用法（在 NeuralNetwork_Project2 目录下）：
#     EP=25 bash part1_cifar10/rerun_ablation.sh
set -e
cd "$(dirname "$0")"
EP=${EP:-25}

echo "==== 检查依赖 ===="
python - <<'PY'
import importlib, subprocess, sys
import npu_compat  # import torch 之前打兼容补丁
for pkg in ["torchvision", "matplotlib"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
PY

echo "==== 消融统一 $EP epoch（与基线同条件）===="

echo "---- width ----"
python train.py --tag width32 --width 32 --epochs $EP --optimizer sgd --lr 0.1 --cosine
python train.py --tag width96 --width 96 --epochs $EP --optimizer sgd --lr 0.1 --cosine

echo "---- activation ----"
python train.py --tag act_leaky --activation leaky_relu --epochs $EP --optimizer sgd --lr 0.1 --cosine
python train.py --tag act_gelu  --activation gelu       --epochs $EP --optimizer sgd --lr 0.1 --cosine

echo "---- loss / reg ----"
python train.py --tag loss_nowd --weight_decay 0 --epochs $EP --optimizer sgd --lr 0.1 --cosine
python train.py --tag loss_ls   --loss ce_ls     --epochs $EP --optimizer sgd --lr 0.1 --cosine

echo "---- optimizers ----"
python train.py --tag opt_adam   --optimizer adam   --lr 0.001 --epochs $EP --cosine
python train.py --tag opt_adamw  --optimizer adamw  --lr 0.001 --epochs $EP --cosine
python train.py --tag opt_mysgd  --optimizer mysgd  --lr 0.1   --epochs $EP --cosine
python train.py --tag opt_myadam --optimizer myadam --lr 0.001 --epochs $EP --cosine

echo "---- components ----"
python train.py --tag no_bn       --no_bn       --epochs $EP --optimizer sgd --lr 0.1 --cosine
python train.py --tag no_residual --no_residual --epochs $EP --optimizer sgd --lr 0.1 --cosine
python train.py --tag no_dropout  --no_dropout  --epochs $EP --optimizer sgd --lr 0.1 --cosine

echo "==== 完成，结果已覆盖 results/ 与 figures/ ===="
