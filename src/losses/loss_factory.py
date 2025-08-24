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


class MultilabelFocalLoss(nn.Module):
    """
    多标签Focal Loss损失函数

    结合多标签二元交叉熵损失和Focal Loss机制，专门用于解决多标签分类中的类别不平衡问题。
    通过降低易分类样本的权重，让模型更专注于难分类的样本，特别是少数类别样本。

    核心特性：
    1. 支持多标签分类（每个样本可以有多个正标签）
    2. 自动降低易分类样本的权重（通过gamma参数控制）
    3. 支持类别平衡（通过alpha参数控制）
    4. 支持正样本权重（通过pos_weight参数处理正负样本不平衡）

    Args:
        alpha (float or torch.Tensor, optional): 类别平衡参数，用于平衡正负样本
            - 如果是float，所有类别使用相同的alpha值
            - 如果是Tensor，每个类别使用不同的alpha值
            - 默认为1.0（不进行类别平衡）
        gamma (float, optional): 聚焦参数，用于调整难易样本的权重
            - gamma=0时退化为标准BCE损失
            - gamma>0时降低易分类样本的权重
            - 通常取值2.0，默认为2.0
        pos_weight (torch.Tensor, optional): 正样本权重，用于处理正负样本不平衡
            - 形状为(num_classes,)，每个类别一个权重值
            - 默认为None（不使用正样本权重）
        reduction (str, optional): 损失聚合方式，'mean'、'sum'或'none'，默认为'mean'

    数学公式：
        对于每个类别c和样本i：
        FL(p_ic) = -α_c * (1 - p_ic)^γ * log(p_ic)

        其中：
        - p_ic 是样本i在类别c上的预测概率
        - α_c 是类别c的平衡参数
        - γ 是聚焦参数

    示例：
        >>> # 基本用法
        >>> loss_fn = MultilabelFocalLoss(alpha=1.0, gamma=2.0)
        >>>
        >>> # 使用类别平衡和正样本权重
        >>> alpha = torch.tensor([0.25, 0.75])  # 为两个类别设置不同的alpha
        >>> pos_weight = torch.tensor([2.0, 3.0])  # 为两个类别设置不同的正样本权重
        >>> loss_fn = MultilabelFocalLoss(alpha=alpha, gamma=2.0, pos_weight=pos_weight)
    """

    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None, reduction='mean'):
        super(MultilabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

        # 如果alpha是标量，在forward中会根据类别数量扩展
        if isinstance(alpha, (int, float)):
            self.alpha_scalar = alpha
            self.alpha_tensor = None
        else:
            self.alpha_scalar = None
            self.alpha_tensor = alpha

    def forward(self, inputs, targets):
        """
        计算多标签Focal Loss

        Args:
            inputs (torch.Tensor): 模型输出的logits，形状为(batch_size, num_classes)
            targets (torch.Tensor): 真实标签，形状为(batch_size, num_classes)，值为0或1

        Returns:
            torch.Tensor: 计算得到的多标签Focal Loss
        """
        # 确保输入在正确的设备上
        device = inputs.device
        batch_size, num_classes = inputs.shape

        # 将logits转换为概率
        probs = torch.sigmoid(inputs)

        # 处理alpha参数
        if self.alpha_scalar is not None:
            # 如果alpha是标量，为每个类别创建相同的alpha值
            alpha = torch.full((num_classes,), self.alpha_scalar, device=device)
        elif self.alpha_tensor is not None:
            # 如果alpha是tensor，确保在正确的设备上
            alpha = self.alpha_tensor.to(device)
            if alpha.shape[0] != num_classes:
                raise ValueError(f"alpha tensor的长度({alpha.shape[0]})必须等于类别数量({num_classes})")
        else:
            # 默认情况下，所有类别的alpha为1.0
            alpha = torch.ones(num_classes, device=device)

        # 处理pos_weight参数
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device)
            if pos_weight.shape[0] != num_classes:
                raise ValueError(f"pos_weight tensor的长度({pos_weight.shape[0]})必须等于类别数量({num_classes})")

        # 计算二元交叉熵损失（不进行reduction）
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=pos_weight,
            reduction='none'
        )

        # 计算pt（正确分类的概率）
        # 对于正样本：pt = p
        # 对于负样本：pt = 1 - p
        pt = torch.where(targets == 1, probs, 1 - probs)

        # 计算alpha权重（改进版本，避免与pos_weight冲突）
        # 在多标签场景中，如果已经使用了pos_weight，alpha权重应该更保守
        if pos_weight is not None:
            # 如果使用了pos_weight，alpha权重应该更平衡，避免双重加权
            alpha_weight = torch.where(targets == 1, alpha, alpha)  # 正负样本使用相同的alpha
        else:
            # 如果没有使用pos_weight，使用传统的alpha权重分配
            alpha_weight = torch.where(targets == 1, alpha, 1 - alpha)

        # 计算Focal Loss
        # FL = -α * (1 - pt)^γ * log(pt)
        # 由于bce_loss = -log(pt)，所以：
        # FL = α * (1 - pt)^γ * bce_loss
        focal_weight = alpha_weight * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss

        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"不支持的reduction方式: {self.reduction}")


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
        num_classes = params.get('num_classes', None)

        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            # 如果是标量，创建对应维度的tensor
            if isinstance(pos_weight, (int, float)):
                # 动态确定类别数量
                if num_classes is None:
                    # 向后兼容：如果没有指定num_classes，默认使用24（原始新生儿数据）
                    num_classes = 24
                pos_weight = torch.full((num_classes,), pos_weight)
            else:
                pos_weight = torch.tensor(pos_weight)

        return MultilabelBCELoss(
            pos_weight=pos_weight,
            weight=params.get('weight', None),
            reduction=params.get('reduction', 'mean')
        )
    elif loss_name == "focal_multilabel_bce" or loss_name == "multilabel_focal":
        # 处理alpha参数（类别平衡参数）
        alpha = params.get('alpha', 1.0)
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            # 如果是标量，保持为标量，在forward中处理

        # 处理pos_weight参数（正样本权重）
        pos_weight = params.get('pos_weight', None)
        num_classes = params.get('num_classes', None)

        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            # 如果是标量，创建对应维度的tensor
            if isinstance(pos_weight, (int, float)):
                # 动态确定类别数量
                if num_classes is None:
                    # 对于新生儿多标签数据，默认使用7个类别
                    num_classes = 7
                pos_weight = torch.full((num_classes,), pos_weight)
            elif isinstance(pos_weight, (list, tuple)):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

        return MultilabelFocalLoss(
            alpha=alpha,
            gamma=params.get('gamma', 2.0),
            pos_weight=pos_weight,
            reduction=params.get('reduction', 'mean')
        )
    elif loss_name == "focal_multilabel_balanced" or loss_name == "multilabel_focal_balanced":
        # 改进版多标签Focal Loss，专门为严重类别不平衡设计
        # 使用更保守的参数配置，避免过度预测问题

        # 处理pos_weight参数
        pos_weight = params.get('pos_weight', None)
        num_classes = params.get('num_classes', None)

        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            if isinstance(pos_weight, (int, float)):
                if num_classes is None:
                    num_classes = 7
                pos_weight = torch.full((num_classes,), pos_weight)
            elif isinstance(pos_weight, (list, tuple)):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

        # 使用更保守的默认参数
        return MultilabelFocalLoss(
            alpha=params.get('alpha', 1.0),  # 默认不使用alpha权重
            gamma=params.get('gamma', 1.0),  # 更保守的gamma值
            pos_weight=pos_weight,
            reduction=params.get('reduction', 'mean')
        )
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}。支持的损失函数: crossentropy, focal, labelsmoothing, mse, l1, smoothl1, multilabel_bce, focal_multilabel_bce, focal_multilabel_balanced")
