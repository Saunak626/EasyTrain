"""
自定义数据集定义模块
包含通用的自定义数据集类
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
import numpy as np


class CustomDataset(Dataset):
    """
    自定义数据集模板
    用户可以参照此模板创建自己的数据集
    """
    
    def __init__(self, data_dir, csv_file=None, transform=None, target_transform=None):
        """
        初始化自定义数据集
        
        Args:
            data_dir (str): 数据目录路径
            csv_file (str): CSV文件路径，包含图片路径和标签
            transform (callable, optional): 图像变换
            target_transform (callable, optional): 标签变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # 如果提供了CSV文件，从CSV加载数据
        if csv_file and os.path.exists(csv_file):
            self.data_info = pd.read_csv(csv_file)
            self.image_paths = self.data_info.iloc[:, 0].tolist()  # 第一列为图片路径
            self.labels = self.data_info.iloc[:, 1].tolist()      # 第二列为标签
        else:
            # 否则从目录结构自动加载（假设按类别分文件夹）
            self.image_paths, self.labels = self._load_from_directory()
    
    def _load_from_directory(self):
        """
        从目录结构加载数据
        假设目录结构为：
        data_dir/
        ├── class1/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── class2/
            ├── img3.jpg
            └── img4.jpg
        """
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
        # 加载图像
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个默认图像
            image = Image.new('RGB', (224, 224), color='black')
        
        # 获取标签
        label = self.labels[idx]
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    






def get_custom_transforms(augment=True, image_size=224):
    """
    获取自定义数据集的数据变换

    Args:
        augment: 是否使用数据增强
        image_size: 图像大小

    Returns:
        tuple: (train_transform, test_transform)
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def get_custom_dataloaders(data_dir, csv_file=None, batch_size=32, num_workers=4,
                          augment=True, image_size=224, train_split=0.8):
    """
    创建自定义数据集的数据加载器

    Args:
        data_dir: 数据目录路径
        csv_file: CSV文件路径（可选）
        batch_size: 批大小
        num_workers: 工作进程数
        augment: 是否使用数据增强
        image_size: 图像大小
        train_split: 训练集比例

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    from torch.utils.data import DataLoader, random_split

    # 获取数据变换
    train_transform, test_transform = get_custom_transforms(augment, image_size)

    # 创建完整数据集
    full_dataset = CustomDataset(
        data_dir=data_dir,
        csv_file=csv_file,
        transform=train_transform
    )

    # 分割训练集和验证集
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 为验证集设置不同的变换
    val_dataset.dataset.transform = test_transform

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_dataloader, val_dataloader
