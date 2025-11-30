"""视频分类模型模块

提供视频分类模型的创建和管理，支持预训练权重加载和微调。
"""

import torch
import torch.nn as nn
from torchvision import models


class VideoNetModel(nn.Module):
    """视频分类模型 - 使用预训练骨干网络 + 新分类头"""

    def __init__(self, model_type='r3d_18', num_classes=101, pretrained=True):
        super(VideoNetModel, self).__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.pretrained = pretrained

        # 创建模型并替换分类头
        self.model = self._create_model()

    def _create_model(self):
        """创建模型，加载预训练权重并替换分类头"""
        from torchvision.models.video import (
            R3D_18_Weights, MC3_18_Weights, R2Plus1D_18_Weights, S3D_Weights,
            MViT_V1_B_Weights, MViT_V2_S_Weights,
            Swin3D_B_Weights, Swin3D_S_Weights, Swin3D_T_Weights
        )

        # 模型配置表
        model_configs = {
            'r3d_18': (models.video.r3d_18, R3D_18_Weights.DEFAULT, 'fc', 512),
            'mc3_18': (models.video.mc3_18, MC3_18_Weights.DEFAULT, 'fc', 512),
            'r2plus1d_18': (models.video.r2plus1d_18, R2Plus1D_18_Weights.DEFAULT, 'fc', 512),
            's3d': (models.video.s3d, S3D_Weights.DEFAULT, 'classifier', 1024),
            'mvit_v1_b': (models.video.mvit_v1_b, MViT_V1_B_Weights.DEFAULT, 'head', 768),
            'mvit_v2_s': (models.video.mvit_v2_s, MViT_V2_S_Weights.DEFAULT, 'head', 768),
            'swin3d_b': (models.video.swin3d_b, Swin3D_B_Weights.DEFAULT, 'head', 1024),
            'swin3d_s': (models.video.swin3d_s, Swin3D_S_Weights.DEFAULT, 'head', 768),
            'swin3d_t': (models.video.swin3d_t, Swin3D_T_Weights.DEFAULT, 'head', 768),
        }

        if self.model_type not in model_configs:
            raise ValueError(f"不支持的视频模型: {self.model_type}")

        model_fn, weights, head_name, in_features = model_configs[self.model_type]

        # 加载预训练模型
        if self.pretrained:
            model = model_fn(weights=weights)
            print(f"已加载 {self.model_type} 预训练权重 (Kinetics-400)")
        else:
            model = model_fn(weights=None)
            print(f"{self.model_type} 使用随机初始化权重")

        # 替换分类头
        if head_name == 'fc':
            # ResNet3D系列: model.fc
            model.fc = nn.Linear(in_features, self.num_classes)
            nn.init.normal_(model.fc.weight, 0, 0.01)
            nn.init.constant_(model.fc.bias, 0)
        elif head_name == 'classifier':
            # S3D: model.classifier
            if isinstance(model.classifier, nn.Sequential):
                # S3D的classifier是Sequential，最后一层是fc
                for i, layer in enumerate(model.classifier):
                    if isinstance(layer, nn.Linear):
                        model.classifier[i] = nn.Linear(layer.in_features, self.num_classes)
                        nn.init.normal_(model.classifier[i].weight, 0, 0.01)
                        nn.init.constant_(model.classifier[i].bias, 0)
            else:
                model.classifier = nn.Linear(in_features, self.num_classes)
        elif head_name == 'head':
            # MViT/Swin3D: model.head
            if isinstance(model.head, nn.Sequential):
                for i, layer in enumerate(model.head):
                    if isinstance(layer, nn.Linear):
                        model.head[i] = nn.Linear(layer.in_features, self.num_classes)
                        nn.init.normal_(model.head[i].weight, 0, 0.01)
                        nn.init.constant_(model.head[i].bias, 0)
            else:
                model.head = nn.Linear(in_features, self.num_classes)
                nn.init.normal_(model.head.weight, 0, 0.01)
                nn.init.constant_(model.head.bias, 0)

        print(f"分类头已替换为 {self.num_classes} 类输出")
        return model

    def forward(self, x):
        return self.model(x)


def get_video_model(model_type, num_classes=101, **kwargs):
    """视频模型工厂函数

    Args:
        model_type: 模型类型 (r3d_18, mc3_18, r2plus1d_18, s3d, mvit_v1_b, etc.)
        num_classes: 分类类别数
        **kwargs: 其他参数 (pretrained等)

    Returns:
        配置好的视频模型实例
    """
    pretrained = kwargs.get('pretrained', True)
    
    model = VideoNetModel(
        model_type=model_type,
        num_classes=num_classes,
        pretrained=pretrained
    )

    return model
