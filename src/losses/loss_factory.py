"""
损失函数工厂模块
包含图像分类和视频分类任务的常用损失函数定义和工厂函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilabelBCELoss(nn.Module):
    """
    多标签二元交叉熵损失函数

    用于多标签分类任务，每个标签独立进行二元分类。
    支持类别权重和位置权重来处理类别不平衡问题。

    Args:
        pos_weight (torch.Tensor, optional): 正样本权重，用于处理正负样本不平衡
        weight (torch.Tensor, optional): 类别权重，用于处理类别不平衡
        reduction (str, optional): 损失聚合方式，'mean'、'sum'或'none'，默认为'mean'
    """

    def __init__(self, pos_weight=None, weight=None, reduction='mean'):
        super(MultilabelBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算多标签BCE损失

        Args:
            inputs (torch.Tensor): 模型输出的logits，形状为(batch_size, num_classes)
            targets (torch.Tensor): 真实标签，形状为(batch_size, num_classes)，值为0或1

        Returns:
            torch.Tensor: 计算得到的多标签BCE损失
        """
        # 确保pos_weight和weight在正确的设备上
        pos_weight = self.pos_weight
        weight = self.weight

        if pos_weight is not None:
            pos_weight = pos_weight.to(inputs.device)
        if weight is not None:
            weight = weight.to(inputs.device)

        # 使用sigmoid激活函数将logits转换为概率
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=pos_weight,
            weight=weight,
            reduction=self.reduction
        )
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss损失函数，用于解决类别不平衡问题
    
    通过降低易分类样本的权重，让模型更关注难分类样本。
    原论文：https://arxiv.org/abs/1708.02002
    
    Args:
        alpha (float, optional): 平衡因子，用于平衡正负样本，默认为1
        gamma (float, optional): 调制因子，用于调整难易样本的权重，默认为2
        reduction (str, optional): 损失聚合方式，'mean'或'sum'，默认为'mean'
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        计算Focal Loss
        
        Args:
            inputs (torch.Tensor): 模型输出的logits，形状为(batch_size, num_classes)
            targets (torch.Tensor): 真实标签，形状为(batch_size,)
            
        Returns:
            torch.Tensor: 计算得到的Focal Loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数，用于提高模型的泛化能力
    
    通过在真实标签中添加噪声，防止模型对训练数据过拟合。
    原论文：https://arxiv.org/abs/1512.00567
    
    Args:
        num_classes (int): 类别数量
        smoothing (float, optional): 平滑系数，取值范围[0,1)，默认为0.1
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        """
        计算标签平滑损失
        
        Args:
            inputs (torch.Tensor): 模型输出的logits，形状为(batch_size, num_classes)
            targets (torch.Tensor): 真实标签，形状为(batch_size,)
            
        Returns:
            torch.Tensor: 计算得到的标签平滑损失
        """
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * self.smoothing / (self.num_classes - 1)
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss


def get_loss_function(loss_config=None, loss_name=None, **kwargs):
    """
    损失函数工厂函数，创建并配置损失函数实例

    Args:
        loss_config (dict, optional): 损失函数配置字典
        loss_name (str, optional): 损失函数名称，用于向后兼容
        **kwargs: 损失函数参数，用于向后兼容

    Returns:
        torch.nn.Module: 配置好的损失函数实例

    示例：
        >>> loss_fn = get_loss_function({'type': 'crossentropy', 'label_smoothing': 0.1})
        >>> loss_fn = get_loss_function(loss_name='crossentropy', label_smoothing=0.1)
    """
    # 简化的配置解析
    if loss_config:
        loss_name = loss_config.get('type') or loss_config.get('name', 'crossentropy')
        params = loss_config.get('params', {}) if 'params' in loss_config else {k: v for k, v in loss_config.items() if k not in ['type', 'name']}
    else:
        loss_name = loss_name or 'crossentropy'
        params = kwargs

    loss_name = loss_name.lower()

    if loss_name == "crossentropy":
        return nn.CrossEntropyLoss(
            weight=params.get('weight', None),
            ignore_index=params.get('ignore_index', -100),
            reduction=params.get('reduction', 'mean'),
            label_smoothing=params.get('label_smoothing', 0.0)
        )
    elif loss_name == "focal":
        return FocalLoss(
            alpha=params.get('alpha', 1.0),
            gamma=params.get('gamma', 2.0),
            reduction=params.get('reduction', 'mean')
        )
    elif loss_name == "labelsmoothing":
        return LabelSmoothingLoss(
            num_classes=params.get('num_classes', 10),
            smoothing=params.get('smoothing', 0.1)
        )
    elif loss_name == "mse":
        return nn.MSELoss(
            reduction=params.get('reduction', 'mean')
        )
    elif loss_name == "l1":
        return nn.L1Loss(
            reduction=params.get('reduction', 'mean')
        )
    elif loss_name == "smoothl1":
        return nn.SmoothL1Loss(
            reduction=params.get('reduction', 'mean'),
            beta=params.get('beta', 1.0)
        )
    elif loss_name == "multilabel_bce":
        # 处理正样本权重
        pos_weight = params.get('pos_weight', None)
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            # 如果是标量，创建对应维度的tensor
            if isinstance(pos_weight, (int, float)):
                pos_weight = torch.full((24,), pos_weight)  # 新生儿数据有24个标签
            else:
                pos_weight = torch.tensor(pos_weight)

        return MultilabelBCELoss(
            pos_weight=pos_weight,
            weight=params.get('weight', None),
            reduction=params.get('reduction', 'mean')
        )
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}。支持的损失函数: crossentropy, focal, labelsmoothing, mse, l1, smoothl1, multilabel_bce")
