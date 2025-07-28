"""CIFAR-10数据集定义"""

import torchvision
import torchvision.transforms as transforms


# CIFAR-10数据集常量
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class CIFAR10Dataset:
    """CIFAR-10数据集包装器"""
    
    def __init__(self, data_dir, augment=True, download=True):
        self.data_dir = data_dir
        self.augment = augment
        self.download = download
        self.num_classes = 10
        
    def get_transforms(self):
        """获取数据变换"""
        if self.augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
        
        return train_transform, test_transform
    
    def get_datasets(self):
        """获取训练和测试数据集"""
        train_transform, test_transform = self.get_transforms()
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=self.download,
            transform=train_transform
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=self.download,
            transform=test_transform
        )
        
        return train_dataset, test_dataset