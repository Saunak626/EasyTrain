"""自定义数据集定义"""

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
from PIL import Image
import pandas as pd


# ImageNet数据集常量
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


class CustomDatasetWrapper:
    """自定义数据集包装器"""
    
    def __init__(self, data_dir, csv_file=None, image_size=224, augment=True, train_split=0.8):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.image_size = image_size
        self.augment = augment
        self.train_split = train_split
        
        # 创建完整数据集以获取类别信息
        temp_dataset = CustomDataset(data_dir, csv_file)
        self.num_classes = len(temp_dataset.class_to_idx) if hasattr(temp_dataset, 'class_to_idx') else None
        
    def get_transforms(self):
        """获取数据变换"""
        if self.augment:
            train_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

        test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        return train_transform, test_transform
    
    def get_datasets(self):
        """获取训练和测试数据集"""
        train_transform, test_transform = self.get_transforms()
        
        # 创建完整数据集
        full_dataset = CustomDataset(
            data_dir=self.data_dir,
            csv_file=self.csv_file,
            transform=train_transform
        )

        # 分割训练集和验证集
        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = total_size - train_size

        train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size])
        
        # 为测试集设置不同的变换
        test_dataset.dataset.transform = test_transform
        
        return train_dataset, test_dataset