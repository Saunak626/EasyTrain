"""数据集模块

该模块提供了统一的数据集加载接口，支持视频分类数据集：
- UCF-101: 视频动作识别数据集，支持多种视频模型
- 新生儿多标签: 新生儿行为多标签识别数据集

主要功能：
1. 提供标准化的数据加载接口
2. 支持数据增强和预处理
3. 自动处理训练/测试集分割
4. 统一的数据格式转换

使用示例：
    >>> from src.datasets import create_dataloaders
    >>> train_loader, test_loader, num_classes = create_dataloaders(
    ...     dataset_name='ucf101_video',
    ...     data_dir='./data',
    ...     batch_size=32
    ... )
"""

from .video_dataset import VideoDataset, CombinedVideoDataset
from .neonatal_multilabel_dataset import NeonatalMultilabelDataset
from .neonatal_multilabel_simple import NeonatalMultilabelSimple
from .dataloader_factory import create_dataloaders, get_dataset_info

__all__ = [
    'VideoDataset',
    'CombinedVideoDataset',
    'NeonatalMultilabelDataset',
    'NeonatalMultilabelSimple',
    'create_dataloaders',
    'get_dataset_info'
]