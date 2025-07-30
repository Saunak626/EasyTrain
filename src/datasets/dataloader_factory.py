"""统一数据加载器工厂

该模块提供统一的数据加载器创建接口，使用src/datasets中定义的数据集类。
"""

import os
from torch.utils.data import DataLoader
from .cifar10_dataset import CIFAR10Dataset
from .custom_dataset import CustomDatasetWrapper


def create_dataloaders(dataset_name, data_dir, batch_size, num_workers=4, **kwargs):
    """
    统一的数据加载器创建函数
    
    Args:
        dataset_name (str): 数据集名称，支持'cifar10'或'custom'
        data_dir (str): 数据存储根目录路径
        batch_size (int): 批大小
        num_workers (int, optional): 数据加载的工作进程数，默认为4
        **kwargs: 其他数据集特定参数，如augment, download, csv_file等
        
    Returns:
        tuple: (train_loader, test_loader, num_classes) 训练和测试数据加载器及类别数
        
    Raises:
        ValueError: 当指定的数据集名称不支持时
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        # 创建CIFAR-10数据集
        cifar10_dataset = CIFAR10Dataset(
            data_dir=data_dir,
            augment=kwargs.get('augment', True),
            download=kwargs.get('download', True)
        )
        
        train_dataset, test_dataset = cifar10_dataset.get_datasets()
        num_classes = cifar10_dataset.num_classes

    elif dataset_name == "custom":
        # 创建自定义数据集
        custom_dataset = CustomDatasetWrapper(
            data_dir=data_dir,
            csv_file=kwargs.get('csv_file', None),
            image_size=kwargs.get('image_size', 224),
            augment=kwargs.get('augment', True),
            train_split=kwargs.get('train_split', 0.8)
        )
        
        train_dataset, test_dataset = custom_dataset.get_datasets()
        num_classes = custom_dataset.num_classes

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
    获取数据集基本信息
    
    Args:
        dataset_name (str): 数据集名称，支持'cifar10'或'custom'
        
    Returns:
        dict: 包含数据集名称、类别数、输入尺寸和类别列表的字典
        
    Raises:
        ValueError: 当指定的数据集名称不支持时
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "cifar10":
        return {
            "name": "CIFAR-10",
            "num_classes": 10,
            "input_size": (3, 32, 32),
            "classes": ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        }
    elif dataset_name == "custom":
        return {
            "name": "Custom Dataset",
            "num_classes": None,  # 需要运行时确定
            "input_size": (3, 224, 224),  # 默认大小
            "classes": None  # 需要运行时确定
        }
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")