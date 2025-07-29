"""CIFAR-10数据集定义

该模块提供了CIFAR-10数据集的完整加载和预处理功能：
- 自动下载和解压数据集
- 标准化的数据预处理流程
- 训练时数据增强支持
- 优化的数据加载性能

数据集信息：
- 10个类别：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 60,000张32x32彩色图像（50,000训练 + 10,000测试）
- 每个类别6,000张图像

参考文献：
    Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.
"""

import torchvision
import torchvision.transforms as transforms


# CIFAR-10数据集标准化参数（基于训练集计算）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # RGB通道均值
CIFAR10_STD = (0.2023, 0.1994, 0.2010)   # RGB通道标准差


class CIFAR10Dataset:
    """CIFAR-10数据集包装器
    
    提供CIFAR-10数据集的完整加载和预处理功能，支持：
    - 自动下载和数据验证
    - 训练时数据增强（随机裁剪、翻转等）
    - 标准化的数据预处理（归一化、Tensor转换）
    - 批量数据加载优化
    
    属性：
        data_dir (str): 数据存储目录
        augment (bool): 是否启用数据增强
        download (bool): 是否自动下载数据集
        num_classes (int): 分类类别数（固定为10）
        
    示例：
        >>> dataset = CIFAR10Dataset('./data', augment=True, download=True)
        >>> train_set, test_set = dataset.get_datasets()
    """
    
    def __init__(self, data_dir, augment=True, download=True):
        """初始化CIFAR-10数据集
        
        Args:
            data_dir (str): 数据存储根目录路径
            augment (bool, optional): 是否启用数据增强，默认为True
            download (bool, optional): 是否自动下载数据集，默认为True
        """
        self.data_dir = data_dir
        self.augment = augment
        self.download = download
        self.num_classes = 10  # CIFAR-10固定有10个类别
        
    def get_transforms(self):
        """
        
        根据是否启用数据增强返回不同的变换组合：
        - 训练集：随机裁剪 + 水平翻转 + 标准化
        - 测试集：仅标准化
        
        Returns:
            tuple: (train_transform, test_transform) 训练和测试的数据变换
        """
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
        """获取训练和测试数据集
        
        创建并返回CIFAR-10的训练集和测试集，应用相应的数据变换。
        如果download=True，会自动下载和解压数据集。
        
        Returns:
            tuple: (train_dataset, test_dataset) 训练集和测试集
            
        Raises:
            RuntimeError: 当数据下载失败时
        """
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