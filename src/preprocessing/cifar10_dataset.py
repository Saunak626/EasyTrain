"""
CIFAR-10数据集模块
简化的CIFAR-10数据集加载
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_dataloaders(root='./data', batch_size=128, num_workers=4,
                           augment=True, download=True):
    """
    创建CIFAR-10数据加载器

    Args:
        root: 数据根目录
        batch_size: 批大小
        num_workers: 工作进程数
        augment: 是否使用数据增强
        download: 是否下载数据

    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    # 数据变换
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 创建数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=download, transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=download, transform=test_transform
    )

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_dataloader, test_dataloader
