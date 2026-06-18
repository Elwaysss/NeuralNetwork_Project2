"""
第一部分自建的 CNN。

一个 ResNet 风格的小网络，刻意做得参数量不大但精度够看（CIFAR-10）。
它把作业要求的组件都覆盖了：
  - 2D 卷积、2D 池化（MaxPool + 全局平均池化）、全连接层、激活函数（必选项）
  - BatchNorm、Dropout、残差连接（可选项里我三个都用了）

通过构造参数可以做消融：
  width       基础通道数，控制"滤波器数量"
  activation  relu / leaky_relu / gelu
  use_bn / use_residual / use_dropout  开关，方便对比
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"未知激活函数: {name}")


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, activation="relu", use_bn=True, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = make_activation(activation)

        # 维度对不上时（通道数变化或下采样）用 1x1 卷积把 shortcut 投影过去
        self.shortcut = nn.Identity()
        if use_residual and (stride != 1 or in_ch != out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_residual:
            out = out + self.shortcut(x)
        return self.act(out)


class MyCNN(nn.Module):
    def __init__(self, num_classes=10, width=64, activation="relu",
                 use_bn=True, use_residual=True, use_dropout=True, blocks_per_stage=2):
        super().__init__()
        self.cfg = dict(width=width, activation=activation, use_bn=use_bn,
                        use_residual=use_residual, use_dropout=use_dropout)

        c1, c2, c3 = width, width * 2, width * 4

        stem = [nn.Conv2d(3, c1, 3, stride=1, padding=1, bias=not use_bn)]
        if use_bn:
            stem.append(nn.BatchNorm2d(c1))
        stem.append(make_activation(activation))
        stem.append(nn.MaxPool2d(2))  # 32 -> 16，这里是 2D 池化层
        self.stem = nn.Sequential(*stem)

        self.stage1 = self._make_stage(c1, c1, blocks_per_stage, 1, activation, use_bn, use_residual)
        self.stage2 = self._make_stage(c1, c2, blocks_per_stage, 2, activation, use_bn, use_residual)
        self.stage3 = self._make_stage(c2, c3, blocks_per_stage, 2, activation, use_bn, use_residual)

        self.gap = nn.AdaptiveAvgPool2d(1)
        head = [nn.Flatten(), nn.Linear(c3, c3), make_activation(activation)]
        if use_dropout:
            head.append(nn.Dropout(0.5))
        head.append(nn.Linear(c3, num_classes))
        self.classifier = nn.Sequential(*head)

        self._init_weights()

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride, activation, use_bn, use_residual):
        blocks = [BasicBlock(in_ch, out_ch, stride, activation, use_bn, use_residual)]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock(out_ch, out_ch, 1, activation, use_bn, use_residual))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = MyCNN()
    x = torch.randn(2, 3, 32, 32)
    print("output:", net(x).shape)
    print("params:", count_parameters(net))
