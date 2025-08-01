"""
图像任务损失函数模块
包含常用的图像分类和回归损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def get_loss_function(loss_name, **kwargs):
    """
    损失函数工厂函数，创建并配置损失函数实例
    
    Args:
        loss_name (str): 损失函数名称，支持'crossentropy', 'focal', 'labelsmoothing', 'mse'
        **kwargs: 损失函数参数，如weight, alpha, gamma, num_classes等
        
    Returns:
        torch.nn.Module: 配置好的损失函数实例
        
    Raises:
        ValueError: 当指定的损失函数名称不支持时
        
    示例：
        >>> loss_fn = get_loss_function('crossentropy')
        >>> loss_fn = get_loss_function('focal', alpha=1.0, gamma=2.0)
    """
    loss_name = loss_name.lower()

    if loss_name == "crossentropy":
        return nn.CrossEntropyLoss(
            weight=kwargs.get('weight', None),
            ignore_index=kwargs.get('ignore_index', -100),
            reduction=kwargs.get('reduction', 'mean'),
            label_smoothing=kwargs.get('label_smoothing', 0.0)
        )
    elif loss_name == "focal":
        return FocalLoss(
            alpha=kwargs.get('alpha', 1.0),
            gamma=kwargs.get('gamma', 2.0),
            reduction=kwargs.get('reduction', 'mean')
        )
    elif loss_name == "labelsmoothing":
        return LabelSmoothingLoss(
            num_classes=kwargs.get('num_classes', 10),
            smoothing=kwargs.get('smoothing', 0.1)
        )
    elif loss_name == "mse":
        return nn.MSELoss(
            reduction=kwargs.get('reduction', 'mean')
        )
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}。支持的损失函数: crossentropy, focal, labelsmoothing, mse")
