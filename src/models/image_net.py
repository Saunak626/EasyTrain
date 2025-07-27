"""预训练图像分类模型
简化实现，专注于测试用途
"""

import torch
import torch.nn as nn
import timm
from torchvision import models


class ImageNetModel(nn.Module):
    """预训练图像分类模型"""
    
    def __init__(self, model_name='resnet18', num_classes=10, pretrained=True, **kwargs):
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
        return self.backbone(x)


def get_model(model_name='resnet18', **kwargs):
    """获取模型实例"""
    return ImageNetModel(model_name=model_name, **kwargs)