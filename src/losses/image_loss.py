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


def get_loss_function(loss_name, **kwargs):
    """
    获取损失函数
    
    Args:
        loss_name: 损失函数名称
        **kwargs: 损失函数参数
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
