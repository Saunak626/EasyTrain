"""
图像任务损失函数模块
包含常用的图像分类和回归损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
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
    Label Smoothing Loss
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * self.smoothing / (self.num_classes - 1)
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss


def get_loss_function(loss_type="cross_entropy", **kwargs):
    """
    获取损失函数
    
    Args:
        loss_type: 损失函数类型
        **kwargs: 损失函数参数
    
    Returns:
        损失函数实例
    """
    loss_type = loss_type.lower()
    
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    
    elif loss_type == "focal_loss":
        return FocalLoss(
            alpha=kwargs.get('alpha', 1),
            gamma=kwargs.get('gamma', 2),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(
            num_classes=kwargs.get('num_classes', 10),
            smoothing=kwargs.get('smoothing', 0.1)
        )
    
    elif loss_type == "mse":
        return nn.MSELoss()
    
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}。支持的损失函数: cross_entropy, focal_loss, label_smoothing, mse")


# 损失函数注册表
LOSS_REGISTRY = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal_loss': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'mse': nn.MSELoss,
}
