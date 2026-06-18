"""CIFAR-10 数据加载，带标准的数据增强。"""
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# CIFAR-10 常用的逐通道均值方差
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_loaders(root="./data", batch_size=128, num_workers=4, augment=True, n_items=-1):
    train_tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []
    train_tf += [T.ToTensor(), T.Normalize(MEAN, STD)]
    test_tf = [T.ToTensor(), T.Normalize(MEAN, STD)]

    train_set = torchvision.datasets.CIFAR10(root, train=True, download=True,
                                             transform=T.Compose(train_tf))
    test_set = torchvision.datasets.CIFAR10(root, train=False, download=True,
                                            transform=T.Compose(test_tf))

    if n_items > 0:
        train_set = Subset(train_set, range(min(n_items, len(train_set))))
        test_set = Subset(test_set, range(min(n_items, len(test_set))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader
