#!/bin/bash
# 任务二（BatchNorm）一键脚本：在华为云昇腾 NPU 的 JupyterLab 终端里跑
#   cd 到本目录后执行：bash run.sh
set -e

cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-20}

echo "==== 检查依赖 ===="
python - <<'PY'
import importlib, subprocess, sys
import npu_compat  # 在 import torch/torchvision 之前打兼容补丁
for pkg in ["torchvision", "tqdm", "matplotlib"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"安装 {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
PY

echo "==== 2.2 有/无 BN 对比 ===="
python bn_compare.py --epochs $EPOCHS

echo "==== 2.3 损失景观 ===="
python loss_landscape.py --epochs $EPOCHS

echo "==== 完成，结果在 figures/ 与 results/ ===="
