import torch
import torch.nn as nn
from torchvision import models
from .model_registry import validate_model_for_task


class MLPClassifier(nn.Module):
    """自定义MLP分类器，参考tmp/model.py的实现"""
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=101, dropout=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化分类器权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # 只在训练模式且batch_size > 1时使用BatchNorm
        if self.training and x.size(0) > 1:
            x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        # 只在训练模式且batch_size > 1时使用BatchNorm
        if self.training and x.size(0) > 1:
            x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x  # 不使用softmax，让CrossEntropyLoss处理


class VideoNetModel(nn.Module):
    """
    视频分类模型包装器 - 参考tmp/model.py的架构设计
    使用特征提取器 + 自定义分类器的方式，保持预训练特征的完整性
    """
    
    def __init__(self, model_type='r3d_18', num_classes=101, pretrained=True, feature_dim=512, freeze_backbone=False):
        """
        初始化视频分类模型

        Args:
            model_type: 模型类型
            num_classes: 分类类别数
            pretrained: 是否使用预训练权重
            feature_dim: 特征维度
            freeze_backbone: 是否冻结骨干网络
        """
        super(VideoNetModel, self).__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone
        
        # 创建特征提取器（保持预训练分类头）
        self.feature_extractor = self._create_feature_extractor()
        
        # 获取特征维度
        original_feature_dim = self._get_feature_dim()
        
        # 添加特征降维层
        self.pool = nn.AdaptiveAvgPool1d(self.feature_dim)
        
        # 自定义分类器
        self.classifier = MLPClassifier(
            input_dim=self.feature_dim,
            hidden_dim=self.feature_dim // 2,
            output_dim=self.num_classes
        )
        
        # 可选：冻结骨干网络
        if self.freeze_backbone:
            self._freeze_backbone()
    
    def _create_feature_extractor(self):
        """创建特征提取器，保持预训练模型完整性"""
        # 导入具体的权重类，确保正确加载预训练权重
        from torchvision.models.video import (
            R3D_18_Weights, MC3_18_Weights, R2Plus1D_18_Weights, S3D_Weights,
            MViT_V1_B_Weights, MViT_V2_S_Weights,
            Swin3D_B_Weights, Swin3D_S_Weights, Swin3D_T_Weights
        )
        
        if self.model_type == 'r3d_18':
            weights = R3D_18_Weights.DEFAULT if self.pretrained else None
            model = models.video.r3d_18(weights=weights)
        elif self.model_type == 'mc3_18':
            weights = MC3_18_Weights.DEFAULT if self.pretrained else None
            model = models.video.mc3_18(weights=weights)
        elif self.model_type == 'r2plus1d_18':
            weights = R2Plus1D_18_Weights.DEFAULT if self.pretrained else None
            model = models.video.r2plus1d_18(weights=weights)
        elif self.model_type == 's3d':
            weights = S3D_Weights.DEFAULT if self.pretrained else None
            model = models.video.s3d(weights=weights)
        elif self.model_type == 'mvit_v1_b':
            weights = MViT_V1_B_Weights.DEFAULT if self.pretrained else None
            model = models.video.mvit_v1_b(weights=weights)
        elif self.model_type == 'mvit_v2_s':
            weights = MViT_V2_S_Weights.DEFAULT if self.pretrained else None
            model = models.video.mvit_v2_s(weights=weights)
        elif self.model_type == 'swin3d_b':
            weights = Swin3D_B_Weights.DEFAULT if self.pretrained else None
            model = models.video.swin3d_b(weights=weights)
        elif self.model_type == 'swin3d_s':
            weights = Swin3D_S_Weights.DEFAULT if self.pretrained else None
            model = models.video.swin3d_s(weights=weights)
        elif self.model_type == 'swin3d_t':
            weights = Swin3D_T_Weights.DEFAULT if self.pretrained else None
            model = models.video.swin3d_t(weights=weights)
        else:
            raise ValueError(f"不支持的视频模型: {self.model_type}")
        
        # 打印权重加载信息和预训练数据集信息
        if self.pretrained:
            pretrain_info = self._get_pretrain_info()
            print(
                f"✅ 已加载 {self.model_type} 预训练权重 | "
                f"📊 数据集: {pretrain_info['dataset']} | "
                f"🎯 类别: {pretrain_info['classes']} | "
                f"🔧 策略: 保持预训练分类头 + 自定义分类器"
            )
            if pretrain_info['note']:
                print(f"   💡 注意: {pretrain_info['note']}")
        else:
            print(f"⚠️  {self.model_type} 使用随机初始化权重")
        
        return model
    
    def _get_feature_dim(self):
        """获取模型的原始特征维度"""
        feature_dims = {
            'r3d_18': 512,
            'mc3_18': 512, 
            'r2plus1d_18': 512,
            's3d': 1024,
            'mvit_v1_b': 768,
            'mvit_v2_s': 768,
            'swin3d_b': 1024,
            'swin3d_s': 768,
            'swin3d_t': 768
        }
        return feature_dims.get(self.model_type, 512)
    
    def _freeze_backbone(self):
        """冻结骨干网络，只训练分类器"""
        print(f"🧊 冻结 {self.model_type} 骨干网络，只训练自定义分类器")
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """前向传播 - 使用特征提取器 + 自定义分类器的方式"""
        # 通过预训练模型提取特征
        features = self.feature_extractor(x)
        
        # 特征降维 (从原始维度降到指定维度)
        features = self.pool(features)
        
        # 通过自定义分类器
        output = self.classifier(features)
        
        return output
    
    def get_transforms(self):
        """获取模型对应的预处理transforms"""
        if hasattr(self.feature_extractor, 'transforms'):
            return self.feature_extractor.transforms()
        return None
    
    def _get_pretrain_info(self):
        """获取预训练模型的数据集信息"""
        pretrain_datasets = {
            'r3d_18': {
                'dataset': 'Kinetics-400',
                'classes': 400,
                'note': 'Kinetics-400包含UCF-101的部分动作类别，迁移效果较好'
            },
            'mc3_18': {
                'dataset': 'Kinetics-400', 
                'classes': 400,
                'note': 'Kinetics-400包含UCF-101的部分动作类别，迁移效果较好'
            },
            'r2plus1d_18': {
                'dataset': 'Kinetics-400',
                'classes': 400, 
                'note': 'Kinetics-400包含UCF-101的部分动作类别，迁移效果较好'
            },
            's3d': {
                'dataset': 'Kinetics-400',
                'classes': 400,
                'note': 'S3D架构较复杂，可能需要更仔细的分类头处理'
            },
            'mvit_v1_b': {
                'dataset': 'Kinetics-400',
                'classes': 400,
                'note': 'MViT是Transformer架构，通常迁移效果很好'
            },
            'mvit_v2_s': {
                'dataset': 'Kinetics-400',
                'classes': 400,
                'note': 'MViT v2改进版本，通常迁移效果很好'
            },
            'swin3d_b': {
                'dataset': 'Kinetics-400',
                'classes': 400,
                'note': 'Swin3D是较新的架构，可能需要特殊的微调策略'
            },
            'swin3d_s': {
                'dataset': 'Kinetics-400', 
                'classes': 400,
                'note': 'Swin3D是较新的架构，可能需要特殊的微调策略'
            },
            'swin3d_t': {
                'dataset': 'Kinetics-400',
                'classes': 400,
                'note': 'Swin3D是较新的架构，可能需要特殊的微调策略'
            }
        }
        
        return pretrain_datasets.get(self.model_type, {
            'dataset': 'Unknown',
            'classes': 'Unknown', 
            'note': ''
        })
    
    # 旧的forward方法已删除，使用新的实现


def debug_model_structure(model, model_type):
    """调试模型结构，帮助理解分类头"""
    print(f"🔍 {model_type} 模型结构分析:")
    
    # 打印主要模块
    for name, module in model.named_children():
        print(f"  - {name}: {type(module).__name__}")
        
        # 特别关注分类相关的层
        if name in ['fc', 'classifier', 'head']:
            if isinstance(module, nn.Sequential):
                print(f"    Sequential包含:")
                for i, sub_module in enumerate(module):
                    print(f"      [{i}] {type(sub_module).__name__}: {sub_module}")
            else:
                print(f"    {type(module).__name__}: {module}")


def get_model_specific_config(model_type):
    """获取模型特定的训练配置"""
    configs = {
        # ResNet3D系列 - 相对稳定，使用标准配置
        'r3d_18': {
            'feature_dim': 512,
            'freeze_backbone': False,
            'suggested_lr': 0.001,
            'suggested_batch_size': 32
        },
        'mc3_18': {
            'feature_dim': 512,
            'freeze_backbone': False,
            'suggested_lr': 0.001,
            'suggested_batch_size': 32
        },
        'r2plus1d_18': {
            'feature_dim': 512,
            'freeze_backbone': False,
            'suggested_lr': 0.001,
            'suggested_batch_size': 32
        },
        # S3D - 复杂架构，需要更小学习率
        's3d': {
            'feature_dim': 512,  # 降维到512
            'freeze_backbone': True,  # 先冻结骨干网络
            'suggested_lr': 0.0001,
            'suggested_batch_size': 16
        },
        # MViT系列 - Transformer架构，通常效果好
        'mvit_v1_b': {
            'feature_dim': 512,
            'freeze_backbone': False,
            'suggested_lr': 0.0005,
            'suggested_batch_size': 16
        },
        'mvit_v2_s': {
            'feature_dim': 512,
            'freeze_backbone': False,
            'suggested_lr': 0.0005,
            'suggested_batch_size': 16
        },
        # Swin3D系列 - 新架构，需要特殊处理
        'swin3d_b': {
            'feature_dim': 256,  # 更小的特征维度
            'freeze_backbone': True,  # 冻结骨干网络
            'suggested_lr': 0.00001,
            'suggested_batch_size': 8
        },
        'swin3d_s': {
            'feature_dim': 256,
            'freeze_backbone': True,
            'suggested_lr': 0.00001,
            'suggested_batch_size': 8
        },
        'swin3d_t': {
            'feature_dim': 256,
            'freeze_backbone': True,
            'suggested_lr': 0.00001,
            'suggested_batch_size': 8
        }
    }
    
    return configs.get(model_type, {
        'feature_dim': 512,
        'freeze_backbone': False,
        'suggested_lr': 0.001,
        'suggested_batch_size': 32
    })


def get_video_model(model_type, num_classes=101, **kwargs):
    """
    视频模型工厂函数 - 使用模型特定配置

    Args:
        model_type: 模型类型
        num_classes: 分类类别数
        **kwargs: 其他模型参数

    Returns:
        torch.nn.Module: 配置好的视频模型实例
    """
    # 获取模型特定配置
    model_config = get_model_specific_config(model_type)
    
    # 合并用户参数和默认配置
    pretrained = kwargs.get('pretrained', True)
    feature_dim = kwargs.get('feature_dim', model_config['feature_dim'])
    freeze_backbone = kwargs.get('freeze_backbone', model_config['freeze_backbone'])
    debug = kwargs.get('debug', False)
    
    print(
        f"🏗️ 创建 {model_type} 模型 | "
        f"🎯 特征维度: {feature_dim} | "
        f"🧊 冻结骨干: {'是' if freeze_backbone else '否'} | "
        f"📚 建议学习率: {model_config['suggested_lr']} | "
        f"📦 建议批大小: {model_config['suggested_batch_size']}"
    )
    
    model = VideoNetModel(
        model_type=model_type,
        num_classes=num_classes,
        pretrained=pretrained,
        feature_dim=feature_dim,
        freeze_backbone=freeze_backbone
    )
    
    # 如果启用调试模式，打印模型结构
    if debug:
        debug_model_structure(model.feature_extractor, model_type)
    
    return model