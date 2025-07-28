"""简化的数据加载器模块

提供CIFAR-10和自定义数据集的数据加载器创建功能。
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd


# 数据集常量
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CustomDataset(Dataset):
    """自定义数据集类"""
    
    def __init__(self, data_dir, csv_file=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        if csv_file and os.path.exists(csv_file):
            self.data_info = pd.read_csv(csv_file)
            self.image_paths = self.data_info.iloc[:, 0].tolist()
            self.labels = self.data_info.iloc[:, 1].tolist()
        else:
            self.image_paths, self.labels = self._load_from_directory()
    
    def _load_from_directory(self):
        """从目录结构加载数据"""
        image_paths = []
        labels = []
        
        class_names = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        
        for class_name in class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        image_paths.append(os.path.join(class_dir, img_name))
                        labels.append(self.class_to_idx[class_name])
        
        return image_paths, labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloaders(dataset_name, data_dir, batch_size, num_workers=4, **kwargs):
    """
    创建数据加载器
    
    Args:
        dataset_name: 数据集名称 ('cifar10' 或 'custom')
        data_dir: 数据目录
        batch_size: 批大小
        num_workers: 工作进程数
        **kwargs: 其他参数
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        # CIFAR-10数据变换
        augment = kwargs.get('augment', True)
        
        if augment:
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

        # 创建CIFAR-10数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=kwargs.get('download', True),
            transform=train_transform
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=kwargs.get('download', True),
            transform=test_transform
        )

        num_classes = 10

    elif dataset_name == "custom":
        # 自定义数据集变换
        image_size = kwargs.get('image_size', 224)
        augment = kwargs.get('augment', True)
        
        if augment:
            train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        # 创建自定义数据集
        full_dataset = CustomDataset(
            data_dir=data_dir,
            csv_file=kwargs.get('csv_file', None),
            transform=train_transform
        )

        # 分割训练集和验证集
        train_split = kwargs.get('train_split', 0.8)
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size

        train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size])
        
        # 为测试集设置不同的变换
        test_dataset.dataset.transform = test_transform
        
        num_classes = len(full_dataset.class_to_idx) if hasattr(full_dataset, 'class_to_idx') else None

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}。支持的数据集: cifar10, custom")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, num_classes


def get_dataset_info(dataset_name):
    """
    获取数据集信息
    
    Args:
        dataset_name: 数据集名称
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        return {
            "num_classes": 10,
            "input_size": (3, 32, 32),
            "classes": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        }
    elif dataset_name == "custom":
        return {
            "num_classes": None,
            "input_size": (3, 224, 224),
            "classes": None
        }
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")