"""统一模型注册表

提供统一的模型创建接口，简化模型工厂函数的实现。
支持图像分类和视频分类模型的统一管理。
"""

import torch
import torch.nn as nn
import timm
from torchvision import models
from torchvision.models.video import (
    MC3_18_Weights, R3D_18_Weights, MViT_V1_B_Weights,
    MViT_V2_S_Weights, R2Plus1D_18_Weights, S3D_Weights,
    Swin3D_B_Weights, Swin3D_S_Weights, Swin3D_T_Weights
)

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

# 视频模型transforms映射表
# 提供每个视频模型对应的官方预训练权重transforms
VIDEO_MODEL_TRANSFORMS_MAP = {
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


def get_video_model_transforms(model_type):
    """获取视频模型的官方transforms

    Args:
        model_type (str): 视频模型类型

    Returns:
        callable: 对应的transforms函数，如果模型不支持则返回None
    """
    if model_type in VIDEO_MODEL_TRANSFORMS_MAP:
        weights = VIDEO_MODEL_TRANSFORMS_MAP[model_type]
        return weights.transforms()
    return None


def validate_model_transforms_compatibility(model_type):
    """验证模型transforms兼容性

    Args:
        model_type (str): 模型类型

    Returns:
        tuple: (is_compatible, message)
    """
    try:
        # 获取官方transforms
        transforms = get_video_model_transforms(model_type)
        if transforms is None:
            return False, "无法获取官方transforms"

        # 创建测试数据 (T, C, H, W)
        test_input = torch.randn(16, 3, 224, 224)

        # 测试transforms
        try:
            output = transforms(test_input)

            # 不同模型有不同的预训练分辨率，需要灵活处理
            expected_channels, expected_frames = 3, 16
            actual_c, actual_t, actual_h, actual_w = output.shape

            # 检查通道数和帧数是否正确
            if actual_c != expected_channels or actual_t != expected_frames:
                message = f"通道数或帧数不匹配: 期望({expected_channels}, {expected_frames}, H, W), 实际{output.shape}"
                return False, message

            # 对于空间分辨率，只要是合理范围内就接受
            if actual_h < 64 or actual_w < 64 or actual_h > 512 or actual_w > 512:
                message = f"空间分辨率超出合理范围: {actual_h}x{actual_w}"
                return False, message

            return True, "transforms兼容"

        except Exception as e:
            message = f"transforms执行失败: {str(e)}"
            return False, message

    except Exception as e:
        message = f"验证过程失败: {str(e)}"
        return False, message


def create_model_unified(model_type, num_classes=10, pretrained=True, debug=False, **kwargs):
    """统一的模型创建接口

    Args:
        model_type (str): 模型类型
        num_classes (int): 分类类别数
        pretrained (bool): 是否使用预训练权重
        debug (bool): 是否启用调试模式
        **kwargs: 其他模型参数

    Returns:
        torch.nn.Module: 创建的模型实例

    Raises:
        ValueError: 当模型类型不支持时
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"不支持的模型: {model_type}。支持的模型: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[model_type]
    library = config['library']
    
    if library == 'timm':
        # 使用timm库创建模型
        model = timm.create_model(
            model_type,
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
        if model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            model.classifier[1] = nn.Linear(config['classifier_in_features'], num_classes)
        elif model_type == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
            model.classifier = nn.Linear(config['classifier_in_features'], num_classes)
        else:
            raise ValueError(f"未实现的torchvision模型: {model_type}")
            
    elif library == 'torchvision.video':
        # 使用torchvision创建视频模型
        weights = 'DEFAULT' if pretrained else None
        model_func = getattr(models.video, config['model_func'])
        model = model_func(weights=weights)
        
        # 修改分类头以适应目标类别数 - 根据不同模型架构进行精确适配
        if model_type in ['r3d_18', 'mc3_18', 'r2plus1d_18']:
            # ResNet3D系列模型 - 使用fc层
            if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"ResNet3D模型 {model_type} 的fc层结构异常: {type(getattr(model, 'fc', None))}")

        elif model_type == 's3d':
            # S3D模型 - 使用classifier Sequential，最后一层是Conv3d
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
                # classifier是Sequential: [Dropout, Conv3d]
                conv_layer = model.classifier[-1]  # 最后一层Conv3d
                if isinstance(conv_layer, nn.Conv3d):
                    in_channels = conv_layer.in_channels
                    # 替换最后的Conv3d层
                    model.classifier[-1] = nn.Conv3d(
                        in_channels, num_classes,
                        kernel_size=conv_layer.kernel_size,
                        stride=conv_layer.stride,
                        padding=conv_layer.padding
                    )
                else:
                    raise ValueError(f"S3D模型的classifier最后一层不是Conv3d: {type(conv_layer)}")
            else:
                raise ValueError(f"S3D模型的classifier结构异常: {type(getattr(model, 'classifier', None))}")

        elif model_type.startswith('mvit'):
            # MViT系列模型 - 使用head Sequential，最后一层是Linear
            if hasattr(model, 'head') and isinstance(model.head, nn.Sequential):
                # head是Sequential: [Dropout, Linear]
                linear_layer = model.head[-1]  # 最后一层Linear
                if isinstance(linear_layer, nn.Linear):
                    in_features = linear_layer.in_features
                    # 替换最后的Linear层
                    model.head[-1] = nn.Linear(in_features, num_classes)
                else:
                    raise ValueError(f"MViT模型 {model_type} 的head最后一层不是Linear: {type(linear_layer)}")
            else:
                raise ValueError(f"MViT模型 {model_type} 的head结构异常: {type(getattr(model, 'head', None))}")

        elif model_type.startswith('swin3d'):
            # Swin3D系列模型 - 使用head Linear层
            if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"Swin3D模型 {model_type} 的head结构异常: {type(getattr(model, 'head', None))}")

        else:
            raise ValueError(f"不支持的视频模型类型: {model_type}。支持的模型: {list(VIDEO_MODEL_TRANSFORMS_MAP.keys())}")


        # 验证transforms兼容性（如果启用）
        if debug and model_type in VIDEO_MODEL_TRANSFORMS_MAP:
            is_compatible, message = validate_model_transforms_compatibility(model_type)
            if is_compatible:
                print(f"✓ {model_type} transforms兼容性验证通过")
            else:
                print(f"警告: {model_type} - {message}")

    else:
        raise ValueError(f"不支持的模型库: {library}")

    return model


def create_model_with_fallback(model_type, num_classes=10, pretrained=True, **kwargs):
    """带回退机制的模型创建函数

    Args:
        model_type (str): 模型类型名称
        num_classes (int): 分类类别数
        pretrained (bool): 是否使用预训练权重
        **kwargs: 其他模型参数

    Returns:
        torch.nn.Module: 创建的模型实例

    Raises:
        ValueError: 当所有创建方式都失败时
    """
    try:
        # 尝试使用统一接口创建模型
        return create_model_unified(model_type, num_classes, pretrained, **kwargs)
    except Exception as e:
        print(f"警告: 统一接口创建模型失败 ({e})，尝试回退方案")

        # 对于视频模型，尝试使用VideoNetModel作为回退
        if model_type in VIDEO_MODEL_TRANSFORMS_MAP:
            try:
                from .video_net import VideoNetModel
                print(f"使用VideoNetModel作为回退方案创建 {model_type}")
                return VideoNetModel(model_type=model_type, num_classes=num_classes, pretrained=pretrained)
            except Exception as fallback_e:
                print(f"回退方案也失败: {fallback_e}")
                raise ValueError(f"无法创建模型 {model_type}: 主要方式失败({e}), 回退方式失败({fallback_e})")
        else:
            raise ValueError(f"不支持的模型类型 {model_type}: {e}")


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


def validate_model_for_task(model_type, task_type):
    """验证模型是否适用于指定任务

    Args:
        model_type (str): 模型类型
        task_type (str): 任务类型

    Returns:
        bool: 是否适用
    """
    if model_type not in MODEL_REGISTRY:
        return False

    return MODEL_REGISTRY[model_type]['task_type'] == task_type
