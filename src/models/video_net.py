"""视频分类模型模块

提供视频分类模型的创建，支持预训练权重加载和微调。
参考 tmp/model_registry.py 的 create_model_unified 实现。
"""

import torch.nn as nn
from torchvision import models
from torchvision.models.video import (
    R3D_18_Weights, MC3_18_Weights, R2Plus1D_18_Weights, S3D_Weights,
    MViT_V1_B_Weights, MViT_V2_S_Weights,
    Swin3D_B_Weights, Swin3D_S_Weights, Swin3D_T_Weights
)


def get_video_model(model_type, num_classes=101, pretrained=True, **kwargs):
    """视频模型工厂函数
    
    直接加载预训练模型并替换分类头，与 tmp/model_registry.py 实现一致。
    
    Args:
        model_type: 模型类型
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        配置好的视频模型
    """
    # 模型配置
    model_configs = {
        'r3d_18': (models.video.r3d_18, R3D_18_Weights.DEFAULT),
        'mc3_18': (models.video.mc3_18, MC3_18_Weights.DEFAULT),
        'r2plus1d_18': (models.video.r2plus1d_18, R2Plus1D_18_Weights.DEFAULT),
        's3d': (models.video.s3d, S3D_Weights.DEFAULT),
        'mvit_v1_b': (models.video.mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
        'mvit_v2_s': (models.video.mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
        'swin3d_b': (models.video.swin3d_b, Swin3D_B_Weights.DEFAULT),
        'swin3d_s': (models.video.swin3d_s, Swin3D_S_Weights.DEFAULT),
        'swin3d_t': (models.video.swin3d_t, Swin3D_T_Weights.DEFAULT),
    }
    
    if model_type not in model_configs:
        raise ValueError(f"不支持的视频模型: {model_type}。支持: {list(model_configs.keys())}")
    
    model_fn, weights = model_configs[model_type]
    
    # 加载模型
    if pretrained:
        model = model_fn(weights=weights)
        print(f"已加载 {model_type} 预训练权重 (Kinetics-400)")
    else:
        model = model_fn(weights=None)
    
    # 替换分类头并初始化权重
    if model_type in ['r3d_18', 'mc3_18', 'r2plus1d_18']:
        # ResNet3D系列 - 使用 fc 层
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # 权重初始化
        nn.init.normal_(model.fc.weight, 0, 0.01)
        nn.init.constant_(model.fc.bias, 0)
        
    elif model_type == 's3d':
        # S3D - classifier 是 Sequential: [Dropout, Conv3d]
        conv_layer = model.classifier[-1]
        in_channels = conv_layer.in_channels
        model.classifier[-1] = nn.Conv3d(
            in_channels, num_classes,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding
        )
        
    elif model_type.startswith('mvit'):
        # MViT系列 - head 是 Sequential: [Dropout, Linear]
        linear_layer = model.head[-1]
        in_features = linear_layer.in_features
        model.head[-1] = nn.Linear(in_features, num_classes)
        # 权重初始化
        nn.init.normal_(model.head[-1].weight, 0, 0.01)
        nn.init.constant_(model.head[-1].bias, 0)
        
    elif model_type.startswith('swin3d'):
        # Swin3D系列 - head 是 Linear
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        # 权重初始化
        nn.init.normal_(model.head.weight, 0, 0.01)
        nn.init.constant_(model.head.bias, 0)
    
    print(f"分类头已替换: {num_classes} 类")
    return model
