#!/bin/bash
# 任务一（CIFAR-10 分类）一键脚本：华为云昇腾 NPU 的 JupyterLab 终端里跑
#   cd 到本目录后执行：bash run.sh
# 想省时间可以把 BASE_EPOCHS / ABL_EPOCHS 改小。
set -e
cd "$(dirname "$0")"

BASE_EPOCHS=${BASE_EPOCHS:-50}
ABL_EPOCHS=${ABL_EPOCHS:-30}

echo "==== 检查依赖 ===="
python - <<'PY'
import importlib, subprocess, sys
import npu_compat  # 在 import torch/torchvision 之前打兼容补丁
for pkg in ["torchvision", "matplotlib"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        print("安装", pkg)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
PY

echo "==== 基线模型 ===="
python train.py --tag baseline --epochs $BASE_EPOCHS --optimizer sgd --lr 0.1 --cosine

echo "==== 消融：滤波器数量（width） ===="
python train.py --tag width32 --width 32 --epochs $ABL_EPOCHS --optimizer sgd --lr 0.1 --cosine
python train.py --tag width96 --width 96 --epochs $ABL_EPOCHS --optimizer sgd --lr 0.1 --cosine

echo "==== 消融：激活函数 ===="
python train.py --tag act_leaky --activation leaky_relu --epochs $ABL_EPOCHS --optimizer sgd --lr 0.1 --cosine
python train.py --tag act_gelu  --activation gelu       --epochs $ABL_EPOCHS --optimizer sgd --lr 0.1 --cosine

echo "==== 消融：损失函数 / 正则 ===="
python train.py --tag loss_nowd --weight_decay 0   --epochs $ABL_EPOCHS --optimizer sgd --lr 0.1 --cosine
python train.py --tag loss_ls   --loss ce_ls        --epochs $ABL_EPOCHS --optimizer sgd --lr 0.1 --cosine

echo "==== 消融：优化器（torch.optim 多个 + 自实现） ===="
python train.py --tag opt_adam   --optimizer adam   --lr 0.001 --epochs $ABL_EPOCHS --cosine
python train.py --tag opt_adamw  --optimizer adamw  --lr 0.001 --epochs $ABL_EPOCHS --cosine
python train.py --tag opt_mysgd  --optimizer mysgd  --lr 0.1   --epochs $ABL_EPOCHS --cosine
python train.py --tag opt_myadam --optimizer myadam --lr 0.001 --epochs $ABL_EPOCHS --cosine

echo "==== 消融：结构组件（去掉 BN / 残差 / dropout） ===="
python train.py --tag no_bn       --no_bn       --epochs $ABL_EPOCHS --optimizer sgd --lr 0.1 --cosine
python train.py --tag no_residual --no_residual --epochs $ABL_EPOCHS --optimizer sgd --lr 0.1 --cosine
python train.py --tag no_dropout  --no_dropout  --epochs $ABL_EPOCHS --optimizer sgd --lr 0.1 --cosine

echo "==== 可视化（用基线最优权重） ===="
python visualize.py --ckpt results/baseline_best.pth

echo "==== 全部完成，结果在 figures/ 与 results/ ===="
