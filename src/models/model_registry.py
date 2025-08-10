"""统一模型注册表

提供统一的模型创建接口，简化模型工厂函数的实现。
支持图像分类和视频分类模型的统一管理。
"""

import torch.nn as nn
import timm
from torchvision import models

# 模型注册表：统一管理所有支持的模型
# 每个模型配置包含以下字段：
# - library: 模型来源库 ('timm', 'torchvision', 'torchvision.video')
# - task_type: 适用的任务类型 ('image_classification', 'video_classification')
# - adapt_cifar: 是否需要CIFAR-10适配 (仅图像模型)
# - model_func: 模型函数名 (仅视频模型)
# - classifier_attr: 分类器属性路径 (部分torchvision模型)
# - classifier_in_features: 分类器输入特征数 (部分torchvision模型)
MODEL_REGISTRY = {
    # 图像分类模型
    'resnet18': {
        'library': 'timm',
        'adapt_cifar': True,  # 需要CIFAR-10适配
        'task_type': 'image_classification'
    },
    'resnet50': {
        'library': 'timm', 
        'adapt_cifar': True,
        'task_type': 'image_classification'
    },
    'efficientnet_b0': {
        'library': 'timm',
        'adapt_cifar': True,
        'task_type': 'image_classification'
    },
    'mobilenet_v2': {
        'library': 'torchvision',
        'adapt_cifar': False,
        'task_type': 'image_classification',
        'classifier_attr': 'classifier.1',
        'classifier_in_features': 1280
    },
    'densenet121': {
        'library': 'torchvision',
        'adapt_cifar': False,
        'task_type': 'image_classification',
        'classifier_attr': 'classifier',
        'classifier_in_features': 1024
    },
    
    # 视频分类模型
    'r3d_18': {
        'library': 'torchvision.video',
        'task_type': 'video_classification',
        'model_func': 'r3d_18'
    },
    'mc3_18': {
        'library': 'torchvision.video',
        'task_type': 'video_classification',
        'model_func': 'mc3_18'
    },
    'r2plus1d_18': {
        'library': 'torchvision.video',
        'task_type': 'video_classification',
        'model_func': 'r2plus1d_18'
    },
    's3d': {
        'library': 'torchvision.video',
        'task_type': 'video_classification',
        'model_func': 's3d'
    },
    'mvit_v1_b': {
        'library': 'torchvision.video',
        'task_type': 'video_classification',
        'model_func': 'mvit_v1_b'
    },
    'mvit_v2_s': {
        'library': 'torchvision.video',
        'task_type': 'video_classification',
        'model_func': 'mvit_v2_s'
    },
    'swin3d_b': {
        'library': 'torchvision.video',
        'task_type': 'video_classification',
        'model_func': 'swin3d_b'
    },
    'swin3d_s': {
        'library': 'torchvision.video',
        'task_type': 'video_classification',
        'model_func': 'swin3d_s'
    },
    'swin3d_t': {
        'library': 'torchvision.video',
        'task_type': 'video_classification',
        'model_func': 'swin3d_t'
    }
}


def create_model_unified(model_name, num_classes=10, pretrained=True, **kwargs):
    """统一的模型创建接口
    
    Args:
        model_name (str): 模型名称
        num_classes (int): 分类类别数
        pretrained (bool): 是否使用预训练权重
        **kwargs: 其他模型参数
        
    Returns:
        torch.nn.Module: 创建的模型实例
        
    Raises:
        ValueError: 当模型名称不支持时
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"不支持的模型: {model_name}。支持的模型: {list(MODEL_REGISTRY.keys())}")
    
    config = MODEL_REGISTRY[model_name]
    library = config['library']
    
    if library == 'timm':
        # 使用timm库创建模型
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # CIFAR-10适配：修改网络结构以适应32x32小图像
        if config.get('adapt_cifar', False):
            # 将第一层卷积的kernel_size从7改为3，stride从2改为1，padding从3改为1
            if hasattr(model, 'conv1'):
                model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            # 移除最大池化层，避免特征图过小
            if hasattr(model, 'maxpool'):
                model.maxpool = nn.Identity()
                
    elif library == 'torchvision':
        # 使用torchvision创建图像模型
        if model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            model.classifier[1] = nn.Linear(config['classifier_in_features'], num_classes)
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
            model.classifier = nn.Linear(config['classifier_in_features'], num_classes)
        else:
            raise ValueError(f"未实现的torchvision模型: {model_name}")
            
    elif library == 'torchvision.video':
        # 使用torchvision创建视频模型
        weights = 'DEFAULT' if pretrained else None
        model_func = getattr(models.video, config['model_func'])
        model = model_func(weights=weights)
        
        # 修改分类头以适应目标类别数
        if hasattr(model, 'fc'):
            # 大多数视频模型使用fc作为分类头
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'head'):
            # 一些Transformer模型使用head作为分类头
            if hasattr(model.head, 'proj'):
                # MViT等模型的head包含proj层
                in_features = model.head.proj.in_features
                model.head.proj = nn.Linear(in_features, num_classes)
            else:
                # 简单的head结构
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"无法找到模型 {model_name} 的分类头")
            
    else:
        raise ValueError(f"不支持的模型库: {library}")
    
    return model


def get_supported_models(task_type=None):
    """获取支持的模型列表
    
    Args:
        task_type (str, optional): 任务类型过滤，'image_classification' 或 'video_classification'
        
    Returns:
        list: 支持的模型名称列表
    """
    if task_type is None:
        return list(MODEL_REGISTRY.keys())
    
    return [name for name, config in MODEL_REGISTRY.items() 
            if config['task_type'] == task_type]


def validate_model_for_task(model_name, task_type):
    """验证模型是否适用于指定任务
    
    Args:
        model_name (str): 模型名称
        task_type (str): 任务类型
        
    Returns:
        bool: 是否适用
    """
    if model_name not in MODEL_REGISTRY:
        return False
    
    return MODEL_REGISTRY[model_name]['task_type'] == task_type
