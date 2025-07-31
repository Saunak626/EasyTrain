import torch
import torch.nn as nn
from torchvision import models


class VideoNetModel(nn.Module):
    """
    视频分类模型包装器
    支持多种PyTorch预训练视频模型
    """
    
    def __init__(self, model_name='r3d_18', num_classes=101, pretrained=True):
        """
        初始化视频分类模型
        
        Args:
            model_name: 模型名称 ('r3d_18', 'mc3_18', 'r2plus1d_18', 's3d')
            num_classes: 分类类别数
            pretrained: 是否使用预训练权重
        """
        super(VideoNetModel, self).__init__()
        self.model_name = model_name
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
        
        if self.model_name == 'r3d_18':
            model = models.video.r3d_18(weights=weights)
        elif self.model_name == 'mc3_18':
            model = models.video.mc3_18(weights=weights)
        elif self.model_name == 'r2plus1d_18':
            model = models.video.r2plus1d_18(weights=weights)
        elif self.model_name == 's3d':
            model = models.video.s3d(weights=weights)
        else:
            raise ValueError(f"不支持的视频模型: {self.model_name}")
        
        return model
    
    def _modify_classifier(self):
        """修改分类器以适应目标类别数"""
        if hasattr(self.backbone, 'fc'):
            # 对于r3d_18, mc3_18, r2plus1d_18
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.backbone, 'classifier'):
            # 对于s3d
            if isinstance(self.backbone.classifier, nn.Sequential):
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier[-1] = nn.Linear(in_features, self.num_classes)
            else:
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, self.num_classes)
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def get_video_model(model_name, num_classes=101, **kwargs):
    """
    视频模型工厂函数
    
    Args:
        model_name: 模型名称
        num_classes: 分类类别数
        **kwargs: 其他模型参数
    
    Returns:
        VideoNetModel实例
    """
    # 从kwargs中提取参数
    pretrained = kwargs.get('pretrained', True)
    
    # 支持的视频模型列表
    supported_models = ['r3d_18', 'mc3_18', 'r2plus1d_18', 's3d']
    
    if model_name not in supported_models:
        raise ValueError(f"不支持的视频模型: {model_name}. 支持的模型: {supported_models}")
    
    model = VideoNetModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    return model