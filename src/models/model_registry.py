"""统一模型注册表

提供视频分类模型的配置信息。
专注于视频分类任务，支持多种3D CNN和Transformer架构。

注意：模型创建请使用 video_net.get_video_model()
"""

from torchvision.models.video import (
    MC3_18_Weights, R3D_18_Weights, MViT_V1_B_Weights,
    MViT_V2_S_Weights, R2Plus1D_18_Weights, S3D_Weights,
    Swin3D_B_Weights, Swin3D_S_Weights, Swin3D_T_Weights
)

# 模型注册表：统一管理所有支持的视频模型
MODEL_REGISTRY = {
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

# 视频模型分类头配置映射
CLASSIFIER_CONFIG = {
    'fc': ['r3d_18', 'mc3_18', 'r2plus1d_18'],
    'classifier_conv3d': ['s3d'],
    'head_sequential': ['mvit_v1_b', 'mvit_v2_s'],
    'head_linear': ['swin3d_b', 'swin3d_s', 'swin3d_t'],
}

# 视频模型预训练权重映射
VIDEO_MODEL_WEIGHTS_MAP = {
    'r3d_18': R3D_18_Weights.DEFAULT,
    'mc3_18': MC3_18_Weights.DEFAULT,
    'r2plus1d_18': R2Plus1D_18_Weights.DEFAULT,
    's3d': S3D_Weights.DEFAULT,
    'mvit_v1_b': MViT_V1_B_Weights.DEFAULT,
    'mvit_v2_s': MViT_V2_S_Weights.DEFAULT,
    'swin3d_b': Swin3D_B_Weights.DEFAULT,
    'swin3d_s': Swin3D_S_Weights.DEFAULT,
    'swin3d_t': Swin3D_T_Weights.DEFAULT,
}


def get_supported_models():
    """获取支持的视频模型列表"""
    return list(MODEL_REGISTRY.keys())
