"""数据加载器工厂模块

提供统一的数据加载器创建接口，支持多种数据集。
"""

from src.datasets.cifar10_dataset import CIFAR10Dataset
from src.datasets.custom_dataset import CustomDatasetWrapper


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
        dataset = CIFAR10Dataset(
            root=data_dir,
            download=kwargs.get('download', True)
        )
        train_loader, test_loader = dataset.get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            augment=kwargs.get('augment', True)
        )
        num_classes = dataset.num_classes

    elif dataset_name == "custom":
        dataset = CustomDatasetWrapper(
            data_dir=data_dir,
            csv_file=kwargs.get('csv_file', None),
            image_size=kwargs.get('image_size', 224)
        )
        train_loader, test_loader = dataset.get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=kwargs.get('train_split', 0.8),
            augment=kwargs.get('augment', True)
        )
        num_classes = dataset.num_classes

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}。支持的数据集: cifar10, custom")

    return train_loader, test_loader, num_classes


def get_dataset_info(dataset_name, data_dir=None, **kwargs):
    """
    获取数据集信息
    
    Args:
        dataset_name: 数据集名称
        data_dir: 数据目录（自定义数据集需要）
        **kwargs: 其他参数
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        dataset = CIFAR10Dataset()
        return dataset.get_info()
    elif dataset_name == "custom":
        if data_dir is None:
            return {
                "num_classes": None,
                "input_size": (3, 224, 224),
                "classes": None
            }
        dataset = CustomDatasetWrapper(
            data_dir=data_dir,
            csv_file=kwargs.get('csv_file', None),
            image_size=kwargs.get('image_size', 224)
        )
        return dataset.get_info()
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")