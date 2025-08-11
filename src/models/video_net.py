import torch
import torch.nn as nn
from torchvision import models
from .model_registry import create_model_unified, validate_model_for_task


class VideoNetModel(nn.Module):
    """
    视频分类模型包装器
    支持多种PyTorch预训练视频模型
    """
    
    def __init__(self, model_type='r3d_18', num_classes=101, pretrained=True):
        """
        初始化视频分类模型

        Args:
            model_type: 模型类型 ('r3d_18', 'mc3_18', 'r2plus1d_18', 's3d', 'mvit_v1_b', 'mvit_v2_s', 'swin3d_b', 'swin3d_s', 'swin3d_t')
            num_classes: 分类类别数
            pretrained: 是否使用预训练权重
        """
        super(VideoNetModel, self).__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # 创建基础模型
        self.backbone = self._create_backbone()
        
        # 替换分类头
        self._modify_classifier()
        
    def _create_backbone(self):
        """创建视频模型骨干网络"""
        # 使用新的weights参数替代deprecated的pretrained参数
        weights = 'DEFAULT' if self.pretrained else None
        
        if self.model_type == 'r3d_18':
            model = models.video.r3d_18(weights=weights)
        elif self.model_type == 'mc3_18':
            model = models.video.mc3_18(weights=weights)
        elif self.model_type == 'r2plus1d_18':
            model = models.video.r2plus1d_18(weights=weights)
        elif self.model_type == 's3d':
            model = models.video.s3d(weights=weights)
        elif self.model_type == 'mvit_v1_b':
            model = models.video.mvit_v1_b(weights=weights)
        elif self.model_type == 'mvit_v2_s':
            model = models.video.mvit_v2_s(weights=weights)
        elif self.model_type == 'swin3d_b':
            model = models.video.swin3d_b(weights=weights)
        elif self.model_type == 'swin3d_s':
            model = models.video.swin3d_s(weights=weights)
        elif self.model_type == 'swin3d_t':
            model = models.video.swin3d_t(weights=weights)
        else:
            raise ValueError(f"不支持的视频模型: {self.model_type}")
        
        return model
    
    def _modify_classifier(self):
        """修改分类器以适应目标类别数"""
        if self.model_type in ['r3d_18', 'mc3_18', 'r2plus1d_18']:
            # 对于ResNet3D系列模型，修改fc层
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, self.num_classes)
        elif self.model_type == 's3d':
            # 对于S3D模型，修改classifier的最后一层（Conv3d）
            # S3D的classifier[-1]是Conv3d(1024, 400, kernel_size=(1, 1, 1))
            in_channels = self.backbone.classifier[-1].in_channels
            self.backbone.classifier[-1] = nn.Conv3d(in_channels, self.num_classes, kernel_size=(1, 1, 1))
        elif self.model_type.startswith('mvit'):
            # 对于MViT系列模型，修改head的最后一层
            # MViT的head是Sequential，最后一层是Linear
            in_features = self.backbone.head[-1].in_features
            self.backbone.head[-1] = nn.Linear(in_features, self.num_classes)
        elif self.model_type.startswith('swin3d'):
            # 对于Swin3D系列模型，直接替换head（Linear层）
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def get_video_model(model_type, num_classes=101, **kwargs):
    """
    视频模型工厂函数

    Args:
        model_type: 模型类型
        num_classes: 分类类别数
        **kwargs: 其他模型参数

    Returns:
        torch.nn.Module: 配置好的视频模型实例
    """
    # 验证模型是否适用于视频分类任务
    if not validate_model_for_task(model_type, 'video_classification'):
        # 如果验证失败，回退到原有实现
        pretrained = kwargs.get('pretrained', True)
        return VideoNetModel(
            model_type=model_type,
            num_classes=num_classes,
            pretrained=pretrained
        )

    # 使用统一的模型创建接口
    return create_model_unified(model_type, num_classes=num_classes, **kwargs)