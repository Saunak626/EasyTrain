"""统一数据加载器工厂

该模块提供统一的数据加载器创建接口，使用src/datasets中定义的数据集类。
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from .cifar10_dataset import CIFAR10Dataset
from .custom_dataset import CustomDatasetWrapper
from .video_dataset import VideoDataset, CombinedVideoDataset


def create_dataloaders(dataset_name, data_dir, batch_size, num_workers=4, model_type=None, **kwargs):
    """
    统一的数据加载器创建函数

    Args:
        dataset_name (str): 数据集名称，支持'cifar10'、'custom'或'ucf101'
        data_dir (str): 数据存储根目录路径
        batch_size (int): 批大小
        num_workers (int, optional): 数据加载的工作进程数，默认为4
        model_type (str, optional): 模型类型，用于视频数据集的动态transforms
        **kwargs: 其他数据集特定参数，如augment, download, csv_file等

    Returns:
        tuple: (train_loader, test_loader, num_classes) 训练和测试数据加载器及类别数

    Raises:
        ValueError: 当指定的数据集名称不支持时
    """
    dataset_name = dataset_name.lower()
    # 数据子采样比例（0-1），1.0表示使用全部数据
    data_percentage = float(kwargs.get('data_percentage', 1.0))

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

    elif dataset_name in ["ucf101", "ucf101_video"]:
        # 统一使用VideoDataset处理UCF-101视频数据（从预处理帧图像加载）
        clip_len = kwargs.get('clip_len', kwargs.get('frames_per_clip', 16))  # 兼容两种参数名

        train_dataset = VideoDataset(
            dataset_path=data_dir,
            images_path='train',
            clip_len=clip_len,
            model_type=model_type  # 传递模型类型用于动态transforms
        )

        # 将val和test合并作为测试集
        test_dataset = CombinedVideoDataset(
            dataset_path=data_dir,
            clip_len=clip_len,
            model_type=model_type  # 传递模型类型用于动态transforms
        )

        num_classes = 101  # UCF-101固定为101个类别

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}。支持的数据集: cifar10, custom, ucf101, ucf101_video")

    # 按比例随机抽样数据子集（支持快速实验）
    if 0 < data_percentage < 1.0:
        def _sample_subset(dataset):
            total = len(dataset)
            sample_size = max(1, int(total * data_percentage))
            indices = torch.randperm(total)[:sample_size]
            return Subset(dataset, indices)
        train_dataset = _sample_subset(train_dataset)
        test_dataset = _sample_subset(test_dataset)

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
        dataset_name (str): 数据集名称，支持'cifar10'、'custom'或'ucf101'
        
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
    elif dataset_name in ["ucf101", "ucf101_video"]:
        return {
            "name": "UCF-101 Video",
            "num_classes": 101,
            "input_size": (3, 16, 112, 112),  # (C, T, H, W)
            "classes": None  # 需要运行时确定
        }
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")