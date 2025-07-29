"""数据集模块

该模块提供了统一的数据集加载接口，支持多种数据集类型：
- CIFAR-10: 内置支持，自动下载和数据增强
- 自定义数据集: 支持目录结构和CSV标注文件

主要功能：
1. 提供标准化的数据加载接口
2. 支持数据增强和预处理
3. 自动处理训练/测试集分割
4. 统一的数据格式转换

使用示例：
    >>> from src.datasets import create_dataloaders
    >>> train_loader, test_loader, num_classes = create_dataloaders(
    ...     dataset_name='cifar10',
    ...     data_dir='./data',
    ...     batch_size=128
    ... )
"""

from .cifar10_dataset import CIFAR10Dataset
from .custom_dataset import CustomDatasetWrapper
from .dataloader_factory import create_dataloaders, get_dataset_info

__all__ = ['CIFAR10Dataset', 'CustomDatasetWrapper', 'create_dataloaders', 'get_dataset_info']