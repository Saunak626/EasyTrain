"""预训练图像分类模型
简化实现，专注于测试用途
"""

import torch
import torch.nn as nn
import timm
from torchvision import models
from .model_registry import create_model_unified, validate_model_for_task


class ImageNetModel(nn.Module):
    """预训练图像分类模型
    
    提供基于timm和torchvision的预训练模型，支持多种主流网络架构，
    针对小尺寸图像（如CIFAR-10）进行了优化适配。
    
    Attributes:
        model_name (str): 模型名称
        num_classes (int): 输出类别数
        backbone (nn.Module): 基础网络模型
    """
    
    def __init__(self, model_name='resnet18', num_classes=10, pretrained=True, **kwargs):
        """
        初始化图像分类模型
        
        Args:
            model_name (str, optional): 模型名称，支持'resnet18', 'resnet50', 
                'efficientnet_b0', 'mobilenet_v2', 'densenet121'等，默认为'resnet18'
            num_classes (int, optional): 输出类别数，默认为10
            pretrained (bool, optional): 是否使用预训练权重，默认为True
            **kwargs: 其他模型参数（目前未使用）
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        # 忽略不需要的参数如input_size, freeze_backbone等
        
        # 创建预训练模型
        if model_name in ['resnet18', 'resnet50', 'efficientnet_b0']:
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=num_classes
            )
            # 适配CIFAR-10小尺寸输入
            if hasattr(self.backbone, 'conv1'):
                self.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            if hasattr(self.backbone, 'maxpool'):
                self.backbone.maxpool = nn.Identity()
        else:
            # 使用torchvision模型
            if model_name == 'mobilenet_v2':
                self.backbone = models.mobilenet_v2(pretrained=pretrained)
                self.backbone.classifier[1] = nn.Linear(1280, num_classes)
            elif model_name == 'densenet121':
                self.backbone = models.densenet121(pretrained=pretrained)
                self.backbone.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为(batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: 输出 logits，形状为(batch_size, num_classes)
        """
        return self.backbone(x)


def get_model(model_name='resnet18', **kwargs):
    """
    模型工厂函数，创建预训练图像分类模型实例

    Args:
        model_name (str, optional): 模型名称，支持'resnet18', 'resnet50',
            'efficientnet_b0', 'mobilenet_v2', 'densenet121'等，默认为'resnet18'
        **kwargs: 传递给模型的其他参数，如num_classes, pretrained等

    Returns:
        torch.nn.Module: 配置好的模型实例
    """
    # 验证模型是否适用于图像分类任务
    if not validate_model_for_task(model_name, 'image_classification'):
        # 如果验证失败，回退到原有实现
        return ImageNetModel(model_name=model_name, **kwargs)

    # 使用统一的模型创建接口
    return create_model_unified(model_name, **kwargs)