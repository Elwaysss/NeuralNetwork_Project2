# 神经网络与深度学习 Project-2

马静 23300180134

CIFAR-10 图像分类 + Batch Normalization 实验。

## 目录结构

```
part1_cifar10/          任务一：CIFAR-10 分类
  models.py             自建 CNN（卷积/池化/全连接/激活 + BN/Dropout/残差）
  data.py               数据加载与增强
  engine.py             训练/评估/计速
  optimizers.py         自己实现的优化器（SGD、Adam）
  train.py              训练入口，消融实验靠命令行参数控制
  visualize.py          卷积核 / 损失景观 / Grad-CAM 可视化
  run.sh                一键跑完所有实验

codes/VGG_BatchNorm/    任务二：Batch Normalization
  models/vgg.py         VGG_A 与 VGG_A_BatchNorm
  VGG_Loss_Landscape.py 训练与按 step 记录 loss/梯度的工具
  bn_compare.py         2.2 有/无 BN 对比
  loss_landscape.py     2.3 多学习率损失景观
  run.sh                一键跑完任务二

test_ascend_npu.py      上云第一步：检测昇腾 NPU 是否可用
华为云操作指南.md        从零开始的云端操作步骤
report/                 实验报告（LaTeX）
```

## 运行环境

- 华为云 ModelArts，昇腾 NPU（snt9b），镜像 pytorch 2.6.0 + cann 8.2 + python 3.11
- 代码会自动选设备：检测到 `torch_npu` 用 `npu:0`，否则退回 CPU（本地调试）

## 怎么跑

```bash
# 1. 确认 NPU
python test_ascend_npu.py

# 2. 任务一
cd part1_cifar10 && bash run.sh && cd ..

# 3. 任务二
cd codes/VGG_BatchNorm && bash run.sh && cd ..
```

数据集 CIFAR-10 会自动下载，不放在仓库里。

## 数据集与模型权重

- 数据集：CIFAR-10，运行脚本时自动下载，或见官网 https://www.cs.toronto.edu/~kriz/cifar.html
- 模型权重（基线 CNN + VGG-A 有/无 BN）：https://pan.quark.cn/s/3c01505088ab（提取码：8jBk）
