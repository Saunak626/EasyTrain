"""自定义数据集定义

该模块提供了灵活的自定义数据集加载功能，支持：
- 目录结构分类（ImageFolder风格）
- CSV文件标注
- 自动训练/验证集分割
- 数据增强和预处理

支持的数据格式：
- 图像格式：.png, .jpg, .jpeg, .bmp, .tiff
- 目录结构：data/class1/img1.jpg, data/class2/img2.jpg
- CSV标注：image_path,label 或 image_path,class_name

标准化参数使用ImageNet预训练模型的统计量：
    mean = [0.485, 0.456, 0.406] (RGB)
    std = [0.229, 0.224, 0.225] (RGB)
"""

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
from PIL import Image
import pandas as pd


# ImageNet预训练模型的标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB通道均值
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB通道标准差


class CustomDataset(Dataset):
    """自定义数据集类
    
    支持两种数据加载方式：
    1. 目录结构：根据子目录名自动分类
    2. CSV标注：通过CSV文件指定图像路径和标签
    
    特性：
    - 自动图像格式验证和错误处理
    - RGB格式统一转换
    - 内存高效的按需加载
    - 灵活的标签映射
    
    属性：
        data_dir (str): 数据根目录
        csv_file (str, optional): CSV标注文件路径
        transform (callable, optional): 数据变换函数
        image_paths (list): 图像路径列表
        labels (list): 标签列表
        class_to_idx (dict): 类别名到索引的映射
    
    示例：
        >>> # 目录结构方式
        >>> dataset = CustomDataset('./data/my_dataset')
        >>> # CSV标注方式
        >>> dataset = CustomDataset('./data', csv_file='./labels.csv')
    """
    
    def __init__(self, data_dir, csv_file=None, transform=None):
        """初始化自定义数据集
        
        Args:
            data_dir (str): 数据根目录路径
            csv_file (str, optional): CSV标注文件路径。如果提供，将使用CSV加载方式
            transform (callable, optional): 数据预处理变换函数
        
        Raises:
            FileNotFoundError: 当数据目录或CSV文件不存在时
            ValueError: 当CSV文件格式不正确时
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 选择数据加载方式：CSV文件或目录结构
        if csv_file and os.path.exists(csv_file):
            self._load_from_csv(csv_file)
        else:
            self._load_from_directory()
    
    def _load_from_csv(self, csv_file):
        """从CSV文件加载数据
        
        CSV文件格式要求：
        - 第一列：图像路径（相对于data_dir或绝对路径）
        - 第二列：标签（可以是数字或字符串）
        
        Args:
            csv_file (str): CSV文件路径
        """
        self.data_info = pd.read_csv(csv_file)
        
        # 验证CSV格式
        if len(self.data_info.columns) < 2:
            raise ValueError("CSV文件至少需要包含两列：图像路径和标签")
        
        # 获取图像路径和标签
        self.image_paths = self.data_info.iloc[:, 0].tolist()
        raw_labels = self.data_info.iloc[:, 1].tolist()
        
        # 处理字符串标签，转换为数字索引
        if all(isinstance(label, str) for label in raw_labels):
            unique_labels = sorted(set(raw_labels))
            self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.labels = [self.class_to_idx[label] for label in raw_labels]
        else:
            self.labels = [int(label) for label in raw_labels]
            # 为数字标签创建类别映射
            unique_labels = sorted(set(self.labels))
            self.class_to_idx = {str(label): label for label in unique_labels}
    
    def _load_from_directory(self):
        """从目录结构加载数据（ImageFolder风格）
        
        期望的目录结构：
            data_dir/
            ├── class1/
            │   ├── img1.jpg
            │   └── img2.png
            └── class2/
                ├── img3.jpg
                └── img4.jpeg
        
        自动为每个类别分配数字索引，按类别名排序。
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
        """返回数据集大小
        
        Returns:
            int: 数据集中样本的总数
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (image, label) 图像和对应的标签
            
        Note:
            - 自动处理图像加载失败的情况，返回黑色占位图像
            - 统一转换为RGB格式
            - 应用数据变换（如果提供）
        """
        img_path = self.image_paths[idx]
        
        # 安全地加载图像，处理可能的异常
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  加载图像失败 {img_path}: {e}")
            # 创建黑色占位图像，避免训练中断
            image = Image.new('RGB', (224, 224), color='black')
        
        label = self.labels[idx]
        
        # 应用数据变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CustomDatasetWrapper:
    """自定义数据集包装器
    
    提供高级数据集管理功能：
    - 自动训练/验证集分割
    - 灵活的数据增强配置
    - 可配置的图像尺寸
    - ImageNet标准化处理
    
    适用于自定义数据集的完整训练流程，支持各种预处理需求。
    
    属性：
        data_dir (str): 数据根目录
        csv_file (str, optional): CSV标注文件路径
        image_size (int): 目标图像尺寸
        augment (bool): 是否启用数据增强
        train_split (float): 训练集比例
        num_classes (int): 数据集类别数
        
    示例：
        >>> wrapper = CustomDatasetWrapper(
        ...     data_dir='./data/my_dataset',
        ...     image_size=224,
        ...     augment=True,
        ...     train_split=0.8
        ... )
        >>> train_set, val_set = wrapper.get_datasets()
    """
    
    def __init__(self, data_dir, csv_file=None, image_size=224, augment=True, train_split=0.8):
        """初始化自定义数据集包装器
        
        Args:
            data_dir (str): 数据根目录路径
            csv_file (str, optional): CSV标注文件路径
            image_size (int, optional): 目标图像尺寸，默认224
            augment (bool, optional): 是否启用数据增强，默认True
            train_split (float, optional): 训练集分割比例，默认0.8
            
        Raises:
            ValueError: 当train_split不在(0,1)范围内时
        """
        if not 0 < train_split < 1:
            raise ValueError("train_split必须在0和1之间")
            
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.image_size = image_size
        self.augment = augment
        self.train_split = train_split
        
        # 创建临时数据集以获取类别信息
        temp_dataset = CustomDataset(data_dir, csv_file)
        self.num_classes = len(temp_dataset.class_to_idx) if hasattr(temp_dataset, 'class_to_idx') else None
        
    def get_transforms(self):
        """获取数据预处理变换
        
        根据配置返回不同的变换组合：
        - 训练集（augment=True）：包含数据增强的变换
        - 测试集（augment=False）：仅包含基本预处理
        
        训练增强包括：
            - 随机水平翻转（p=0.5）
            - 随机旋转（±10度）
            - 颜色抖动（亮度、对比度、饱和度、色调）
        
        所有变换都包含：
            - 图像尺寸调整
            - ToTensor转换
            - ImageNet标准化
            
        Returns:
            tuple: (train_transform, test_transform) 训练和测试的变换
        """
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
        """获取训练和测试数据集
        
        创建并分割数据集，应用相应的数据变换：
        1. 创建完整数据集
        2. 按train_split比例分割为训练集和验证集
        3. 为不同数据集应用不同的变换
        
        Returns:
            tuple: (train_dataset, test_dataset) 训练集和测试集
            
        Note:
            - 使用random_split进行随机分割
            - 测试集应用不包含增强的变换
            - 确保分割后的数据集没有重叠
        """
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