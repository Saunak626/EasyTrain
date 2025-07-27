# 自定义组件开发指南

## 📋 概述

本文档提供了如何扩展和自定义训练框架各个组件的详细指南，包括数据集、模型、优化器、调度器和损失函数。

## 🏗️ 项目架构

### 核心设计理念

本训练框架采用**模块化设计**，每个组件都是独立的、可替换的模块：

```
训练流程:
配置解析 → 数据加载 → 模型构建 → 训练器执行 → 结果记录
    ↓           ↓         ↓         ↓         ↓
 config/    data_preprocessing/  models/  trainers/  utils/
```

### 组件依赖关系

```
scripts/train.py (入口)
    ├── utils/config_parser.py (配置解析)
    ├── data_preprocessing/ (数据加载)
    │   ├── cifar10_dataset.py
    │   └── dataset.py
    ├── models/net.py (模型构建)
    ├── optimizers/optim.py (优化器)
    ├── schedules/scheduler.py (调度器)
    ├── losses/image_loss.py (损失函数)
    └── trainers/base_trainer.py (训练执行)
```

### 接口设计原则

1. **统一的工厂函数**：每个组件都有 `get_xxx()` 工厂函数
2. **配置驱动**：所有组件都通过配置文件参数化
3. **向后兼容**：新增组件不影响现有功能
4. **错误处理**：提供清晰的错误信息和降级机制

## 🗂️ 自定义数据集

### 1. 数据集准备

#### 方式1: 目录结构分类（推荐）
```
data/raw/my_dataset/
├── class1/
│   ├── img001.jpg
│   └── img002.jpg
├── class2/
│   ├── img003.jpg
│   └── img004.jpg
└── class3/
    └── img005.jpg
```

#### 方式2: CSV文件索引
```csv
image_path,label
class1/img001.jpg,0
class2/img003.jpg,1
class3/img005.jpg,2
```

### 2. 扩展CustomDataset类

```python
# src/data_preprocessing/my_dataset.py
from .dataset import CustomDataset
from PIL import Image
import torch

class MyCustomDataset(CustomDataset):
    """自定义数据集类示例"""

    def __init__(self, data_dir, transform=None, **kwargs):
        super().__init__(data_dir, transform=transform, **kwargs)
        # 添加自定义初始化逻辑
        self.custom_preprocessing = kwargs.get('custom_preprocessing', None)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 自定义图像加载逻辑
        image = Image.open(img_path).convert('RGB')

        # 自定义预处理
        if self.custom_preprocessing:
            image = self.custom_preprocessing(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self):
        """计算类别权重，用于处理不平衡数据集"""
        from collections import Counter
        label_counts = Counter(self.labels)
        total = len(self.labels)
        weights = {cls: total / count for cls, count in label_counts.items()}
        return weights
```

### 3. 创建专用数据集模块

```python
# src/data_preprocessing/my_special_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

class MySpecialDataset(Dataset):
    """完全自定义的数据集类"""

    def __init__(self, data_path, transform=None, mode='train'):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        # 加载数据索引
        self._load_data()

    def _load_data(self):
        """加载数据索引和标签"""
        # 示例：从CSV文件加载
        csv_file = f"{self.data_path}/{self.mode}.csv"
        self.data_df = pd.read_csv(csv_file)
        self.image_paths = self.data_df['image_path'].tolist()
        self.labels = self.data_df['label'].tolist()
        
        # 类别映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(set(self.labels))}
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_paths[idx])
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_my_special_dataloaders(data_path, batch_size=32, image_size=224, **kwargs):
    """创建专用数据加载器"""
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = MySpecialDataset(data_path, transform=train_transform, mode='train')
    val_dataset = MySpecialDataset(data_path, transform=val_transform, mode='val')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, val_loader
```

## 🧠 自定义模型

### 1. 添加新模型到现有模块

```python
# 在 src/models/net.py 中添加
import torch.nn as nn
import torch.nn.functional as F

class MyCustomModel(nn.Module):
    """自定义模型"""

    def __init__(self, num_classes=10, input_channels=3, **kwargs):
        super(MyCustomModel, self).__init__()
        self.num_classes = num_classes
        
        # 定义网络层
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(kwargs.get('dropout', 0.5)),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        """获取特征图，用于可视化"""
        features = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)
        return features

# 更新get_model函数
def get_model(model_name="simple_cnn", **kwargs):
    model_name = model_name.lower()

    if model_name == "my_custom_model":
        return MyCustomModel(**kwargs)
    # ... 其他模型
    else:
        raise ValueError(f"Unknown model: {model_name}")
```

### 2. 创建独立的模型模块

```python
# src/models/my_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class AdvancedCNN(nn.Module):
    """高级CNN模型"""

    def __init__(self, num_classes=10, dropout=0.5):
        super(AdvancedCNN, self).__init__()
        
        # 使用残差块
        self.conv1 = self._make_conv_block(3, 64)
        self.conv2 = self._make_conv_block(64, 128)
        self.conv3 = self._make_conv_block(128, 256)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256//16, 1),
            nn.ReLU(),
            nn.Conv2d(256//16, 256, 1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 应用注意力
        attention = self.attention(x)
        x = x * attention
        
        x = self.classifier(x)
        return x

class TransformerModel(nn.Module):
    """基于Transformer的模型"""

    def __init__(self, num_classes=10, d_model=512, nhead=8, num_layers=6, **kwargs):
        super(TransformerModel, self).__init__()
        
        # 图像到序列的转换
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 196, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # 图像分块
        x = self.patch_embed(x)  # [B, d_model, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer编码
        x = self.transformer(x)
        
        # 全局平均池化 + 分类
        x = x.mean(dim=1)
        x = self.classifier(x)
        
        return x
```

## ⚙️ 自定义优化器

### 1. 扩展现有优化器模块

```python
# 在 src/optimizers/optim.py 中添加
import torch.optim as optim
import math

def get_optimizer(model, optimizer_name, learning_rate, **kwargs):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "my_optimizer":
        return MyCustomOptimizer(
            model.parameters(),
            lr=learning_rate,
            **kwargs
        )
    elif optimizer_name == "lion":
        return LionOptimizer(
            model.parameters(),
            lr=learning_rate,
            **kwargs
        )
    # ... 其他优化器
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

class MyCustomOptimizer(optim.Optimizer):
    """自定义优化器示例"""

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0, **kwargs):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(MyCustomOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                param_state = self.state[p]

                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                p.data.add_(buf, alpha=-group['lr'])

        return loss

class LionOptimizer(optim.Optimizer):
    """Lion优化器实现"""
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(LionOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = param_state['exp_avg']
                beta1, beta2 = group['betas']

                # Lion update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                p.data.add_(torch.sign(update), alpha=-group['lr'])
                
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
```

## 📅 自定义学习率调度器

### 1. 扩展调度器模块

```python
# 在 src/schedules/scheduler.py 中添加
import math
from torch.optim.lr_scheduler import _LRScheduler

class MyCustomScheduler(_LRScheduler):
    """自定义学习率调度器"""

    def __init__(self, optimizer, warmup_epochs=5, max_epochs=100, min_lr=1e-6, **kwargs):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(MyCustomScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]

class CyclicLRScheduler(_LRScheduler):
    """循环学习率调度器"""
    
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, mode='triangular', **kwargs):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_up
        self.mode = mode
        super(CyclicLRScheduler, self).__init__(optimizer)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_fn = lambda x: 1.
        elif self.mode == 'triangular2':
            scale_fn = lambda x: 1 / (2. ** (cycle - 1))
        else:
            scale_fn = lambda x: 1.
            
        return [self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_fn(x)
                for _ in self.base_lrs]

def get_scheduler(optimizer, scheduler_name, **kwargs):
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "my_scheduler":
        return MyCustomScheduler(optimizer, **kwargs)
    elif scheduler_name == "cyclic":
        return CyclicLRScheduler(optimizer, **kwargs)
    # ... 其他调度器
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
```

## 💔 自定义损失函数

### 1. 扩展损失函数模块

```python
# 在 src/losses/image_loss.py 中添加
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomLoss(nn.Module):
    """自定义损失函数"""

    def __init__(self, alpha=1.0, beta=1.0, reduction='mean'):
        super(MyCustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 实现损失计算逻辑
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 添加自定义项
        custom_term = self._compute_custom_term(inputs, targets)
        
        loss = self.alpha * ce_loss + self.beta * custom_term

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _compute_custom_term(self, inputs, targets):
        # 具体的自定义损失计算
        # 例如：置信度惩罚
        probs = F.softmax(inputs, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        confidence_penalty = -torch.log(max_probs + 1e-8)
        return confidence_penalty

class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    
    def __init__(self, margin=1.0, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, output1, output2, label):
        # label: 1表示相似，0表示不相似
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

class DiceLoss(nn.Module):
    """Dice损失函数，常用于分割任务"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # inputs: [B, C, H, W]
        # targets: [B, H, W]
        
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def get_loss_function(loss_type="cross_entropy", **kwargs):
    loss_type = loss_type.lower()

    if loss_type == "my_custom_loss":
        return MyCustomLoss(**kwargs)
    elif loss_type == "contrastive":
        return ContrastiveLoss(**kwargs)
    elif loss_type == "dice":
        return DiceLoss(**kwargs)
    # ... 其他损失函数
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")
```

## 🔧 集成自定义组件

### 1. 更新训练器

```python
# 在 scripts/train.py 中集成自定义组件
def run_training(config, experiment_name=None):
    # ... 现有代码

    # 使用自定义数据集
    dataset_type = data_config.get('type', 'cifar10')
    if dataset_type == 'my_special':
        from src.data_preprocessing.my_special_dataset import create_my_special_dataloaders
        train_dataloader, test_dataloader = create_my_special_dataloaders(
            data_path=data_config.get('data_path'),
            batch_size=hyperparams['batch_size'],
            image_size=data_config.get('image_size', 224),
            num_workers=data_config.get('num_workers', 4)
        )
        num_classes = len(train_dataloader.dataset.class_to_idx)
    
    # 使用自定义模型
    model_type = model_config.get('type', 'simple_cnn')
    if model_type == 'my_custom_model':
        from src.models.my_models import AdvancedCNN
        model = AdvancedCNN(
            num_classes=num_classes,
            **model_config.get('params', {})
        )
    elif model_type == 'transformer':
        from src.models.my_models import TransformerModel
        model = TransformerModel(
            num_classes=num_classes,
            **model_config.get('params', {})
        )
    else:
        # 使用现有的get_model函数
        model = get_model(model_type, num_classes=num_classes, **model_config.get('params', {}))

    # ... 其余训练逻辑
```

### 2. 更新配置文件

```yaml
# config/custom_config.yaml
data:
  type: "my_special"  # 新的数据集类型
  data_path: "./data/raw/my_special_data"
  image_size: 224
  num_workers: 8

model:
  type: "my_custom_model"  # 新的模型类型
  params:
    dropout: 0.3
    input_channels: 3

optimizer:
  type: "lion"  # 新的优化器
  params:
    betas: [0.9, 0.99]
    weight_decay: 0.01

scheduler:
  type: "my_scheduler"  # 新的调度器
  params:
    warmup_epochs: 10
    max_epochs: 100
    min_lr: 1e-6

loss:
  type: "my_custom_loss"  # 新的损失函数
  params:
    alpha: 1.0
    beta: 0.5

hyperparameters:
  learning_rate: 1e-4
  batch_size: 32
  epochs: 100
```

## 🧪 测试自定义组件

### 1. 单元测试

```python
# tests/test_custom_components.py
import unittest
import torch
from src.data_preprocessing.my_dataset import MyCustomDataset
from src.models.my_models import AdvancedCNN, TransformerModel
from src.optimizers.optim import LionOptimizer
from src.losses.image_loss import MyCustomLoss

class TestCustomComponents(unittest.TestCase):

    def test_custom_dataset(self):
        """测试自定义数据集"""
        # 创建测试数据
        dataset = MyCustomDataset('./test_data')
        self.assertGreater(len(dataset), 0)
        
        # 测试数据加载
        sample = dataset[0]
        self.assertEqual(len(sample), 2)  # image, label

    def test_advanced_cnn(self):
        """测试高级CNN模型"""
        model = AdvancedCNN(num_classes=5)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (2, 5))

    def test_transformer_model(self):
        """测试Transformer模型"""
        model = TransformerModel(num_classes=10, d_model=256, nhead=8)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (1, 10))

    def test_lion_optimizer(self):
        """测试Lion优化器"""
        model = AdvancedCNN(num_classes=5)
        optimizer = LionOptimizer(model.parameters(), lr=1e-4)
        
        # 模拟一步优化
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 5, (2,))
        
        output = model(x)
        loss = torch.nn.CrossEntropyLoss()(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.assertTrue(True)  # 如果没有错误，测试通过

    def test_custom_loss(self):
        """测试自定义损失函数"""
        loss_fn = MyCustomLoss(alpha=1.0, beta=0.5)
        
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        
        loss = loss_fn(inputs, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # 标量

if __name__ == '__main__':
    unittest.main()
```

### 2. 集成测试

```python
# tests/test_integration.py
import unittest
import tempfile
import os
from scripts.train import run_training

class TestIntegration(unittest.TestCase):
    
    def test_custom_training_pipeline(self):
        """测试完整的自定义训练流程"""
        config = {
            'data': {
                'type': 'cifar10',  # 使用CIFAR-10进行快速测试
                'root': './data',
                'download': True,
                'batch_size': 16
            },
            'model': {
                'type': 'my_custom_model',
                'params': {'dropout': 0.3}
            },
            'optimizer': {
                'type': 'lion',
                'params': {'weight_decay': 0.01}
            },
            'scheduler': {
                'type': 'my_scheduler',
                'params': {'warmup_epochs': 1, 'max_epochs': 2}
            },
            'loss': {
                'type': 'my_custom_loss',
                'params': {'alpha': 1.0, 'beta': 0.1}
            },
            'hyperparameters': {
                'learning_rate': 1e-3,
                'batch_size': 16,
                'epochs': 2  # 快速测试
            }
        }

        # 运行训练
        result = run_training(config, experiment_name='test_custom')
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn('best_accuracy', result)
        self.assertGreater(result['best_accuracy'], 0)

if __name__ == '__main__':
    unittest.main()
```

## 📚 最佳实践

### 1. 代码组织
- **模块化设计**：每个自定义组件放在独立的文件中
- **接口一致性**：保持与现有组件相同的接口设计
- **文档完整性**：添加详细的文档字符串和类型注解
- **代码复用**：尽可能复用现有的工具函数和基类

### 2. 配置管理
- **参数验证**：在组件初始化时验证配置参数的有效性
- **默认值**：为所有参数提供合理的默认值
- **向后兼容**：确保新增参数不影响现有配置的使用
- **配置文档**：在配置文件中添加详细的参数说明

### 3. 错误处理
- **输入验证**：检查输入数据的格式和范围
- **异常处理**：提供有意义的错误信息和建议
- **降级机制**：在组件不可用时提供备选方案
- **日志记录**：记录关键操作和错误信息

### 4. 性能优化
- **内存效率**：避免不必要的内存分配和复制
- **计算优化**：使用向量化操作和GPU加速
- **缓存机制**：缓存重复计算的结果
- **批处理**：支持批量处理以提高效率

### 5. 测试策略
- **单元测试**：为每个组件编写独立的测试
- **集成测试**：测试组件之间的协作
- **性能测试**：验证组件的性能表现
- **回归测试**：确保修改不影响现有功能

## 🚀 扩展示例

### 添加新的数据增强策略

```python
# src/data_preprocessing/augmentations.py
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import random

class MixUp:
    """MixUp数据增强"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        
        # 生成混合权重
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        
        # 随机排列
        indices = torch.randperm(batch_size)
        
        # 混合图像
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # 混合标签
        mixed_labels = (labels, labels[indices], lam)
        
        return mixed_images, mixed_labels

class CutMix:
    """CutMix数据增强"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        
        # 生成混合比例
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        
        # 计算裁剪区域
        _, _, H, W = images.shape
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = torch.randint(0, W, (1,))
        cy = torch.randint(0, H, (1,))
        
        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)
        
        # 随机排列
        indices = torch.randperm(batch_size)
        
        # 应用CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        mixed_labels = (labels, labels[indices], lam)
        
        return mixed_images, mixed_labels
```

通过遵循这个开发指南，您可以轻松地扩展训练框架以满足特定需求，同时保持代码的质量和可维护性！
