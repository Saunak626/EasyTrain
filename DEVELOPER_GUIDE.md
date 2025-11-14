# EasyTrain 开发者指南

## 🎯 项目概述

EasyTrain是一个基于PyTorch的深度学习训练框架，支持图像分类和视频分类任务。项目采用模块化设计，支持多种模型、数据集和训练策略。

### 核心特性
- **任务支持**: 图像分类、视频分类
- **模型统一**: 统一的模型创建和管理接口
- **配置驱动**: 灵活的YAML配置系统
- **分布式训练**: 基于Accelerate的多GPU支持
- **实验追踪**: 集成SwanLab实验管理

## 📚 代码阅读顺序

### 🚀 快速入门路径 (30分钟)

#### 1. 项目结构概览
```
EasyTrain/
├── scripts/              # 入口脚本
│   ├── train.py         # 单实验训练入口 ⭐
│   └── grid_search.py   # 网格搜索入口
├── src/                 # 核心代码
│   ├── trainers/        # 训练器 ⭐
│   ├── models/          # 模型定义 ⭐
│   ├── datasets/        # 数据集 ⭐
│   ├── utils/           # 工具函数
│   ├── losses/          # 损失函数
│   ├── optimizers/      # 优化器
│   └── schedules/       # 学习率调度器
├── config/              # 配置文件 ⭐
└── data/               # 数据存储
```

#### 2. 核心文件阅读顺序 (⭐ 必读)
1. **config/grid.yaml** - 了解配置结构
2. **scripts/train.py** - 理解训练入口
3. **src/trainers/base_trainer.py** - 核心训练逻辑
4. **src/models/model_registry.py** - 模型管理系统
5. **src/datasets/dataloader_factory.py** - 数据加载

### 🔍 深入理解路径 (2-3小时)

#### 阶段1: 配置系统 (30分钟)
```
config/grid.yaml                    # 图像分类配置示例
config/ucf101_video.yaml           # 视频分类配置示例
src/utils/config_parser.py         # 配置解析逻辑
src/utils/config_parser_simplified.py  # 简化版解析器
```

#### 阶段2: 任务切换机制 (45分钟)
```
src/trainers/base_trainer.py:249-272   # 任务配置解析
src/trainers/base_trainer.py:32-65     # SUPPORTED_TASKS定义
src/trainers/base_trainer.py:67-89     # 向后兼容推断
```

### 🎬 任务配置与入口
- **单实验（UCF101）**：`python scripts/train.py --config config/ucf101_video.yaml`
- **网格搜索（UCF101）**：`python scripts/grid_search.py --config config/ucf101_video_grid.yaml`
- **单实验（Neonatal，多标签）**：`python scripts/train.py --config config/neonatal_multilabel.yaml`
- **网格搜索（Neonatal，多标签）**：`python scripts/grid_search.py --config config/neonatal_multilabel_grid.yaml`
- **配置示例**：`config/examples` 提供最小化模板，可复制后按需修改 `task.tag`、`data.type`、`model.type` 等字段。
- ✅ **唯一入口**：所有训练流程都通过 `scripts/train.py` / `scripts/grid_search.py` 加 YAML 配置驱动，旧版 `tmp/train_multilabel.py` 已移除，避免冗余参数和重复实现。

#### 阶段3: 模型系统 (60分钟)
```
src/models/model_registry.py:11-97     # 模型注册表
src/models/model_registry.py:100-178   # 统一创建接口
src/models/image_net.py:75-92          # 图像模型工厂
src/models/video_net.py:90-120         # 视频模型工厂
```

#### 阶段4: 数据系统 (45分钟)
```
src/datasets/dataloader_factory.py:15-110  # 数据加载器工厂
src/datasets/video_dataset.py:12-68        # 视频数据集基类
src/datasets/cifar10_dataset.py            # 图像数据集示例
```

## 🏗️ 架构设计理念

### 1. 任务驱动架构
```python
# 核心设计：基于task_tag的任务识别
task_tag = config.get('task', {}).get('tag')  # 'image_classification' 或 'video_classification'

# 任务信息查找
task_info = SUPPORTED_TASKS[task_tag]

# 基于任务选择对应组件
model_factory = globals()[task_info['model_factory']]  # get_model 或 get_video_model
```

### 2. 统一接口设计
```python
# 模型创建统一接口
model = create_model_unified(model_name, num_classes, **kwargs)

# 数据加载统一接口  
train_loader, test_loader, dataset_info = create_dataloaders(dataset_name, **params)
```

### 3. 配置驱动模式
```yaml
# 配置文件驱动所有训练行为
task:
  tag: "image_classification"  # 任务类型
  
model:
  type: "resnet18"            # 模型选择
  
data:
  type: "cifar10"             # 数据集选择
```

## 🔄 数据流分析

### 训练流程数据流
```
1. 配置解析
   config.yaml → parse_arguments() → config dict

2. 任务识别  
   config['task']['tag'] → SUPPORTED_TASKS → task_info

3. 组件创建
   task_info → model_factory → model
   data_config → create_dataloaders → dataloaders
   
4. 训练执行
   model + dataloaders → train_epoch() → metrics
   
5. 结果记录
   metrics → SwanLab → 实验追踪
```

### 模块依赖关系
```
scripts/train.py
├── src.utils.config_parser (配置解析)
├── src.trainers.base_trainer (训练逻辑)
│   ├── src.models.* (模型创建)
│   ├── src.datasets.* (数据加载)
│   ├── src.losses.* (损失函数)
│   ├── src.optimizers.* (优化器)
│   └── src.schedules.* (调度器)
└── accelerate (分布式训练)
```

## 🛠️ 修改代码最佳实践

### 1. 添加新模型

#### A. 图像分类模型
```python
# 1. 在 src/models/model_registry.py 中注册
MODEL_REGISTRY['new_model'] = {
    'library': 'timm',  # 或 'torchvision'
    'adapt_cifar': True,  # 是否需要CIFAR适配
    'task_type': 'image_classification'
}

# 2. 如果使用torchvision，需要添加创建逻辑
elif library == 'torchvision':
    if model_name == 'new_model':
        model = models.new_model(pretrained=pretrained)
        # 修改分类头...
```

#### B. 视频分类模型
```python
# 在 MODEL_REGISTRY 中添加
'new_video_model': {
    'library': 'torchvision.video',
    'task_type': 'video_classification',
    'model_func': 'new_video_model'  # torchvision.models.video中的函数名
}
```

### 2. 添加新数据集

#### A. 创建数据集类
```python
# src/datasets/new_dataset.py
class NewDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        # 实现初始化逻辑
        
    def __len__(self):
        # 返回数据集大小
        
    def __getitem__(self, idx):
        # 返回 (data, label)
```

#### B. 注册到工厂函数
```python
# src/datasets/dataloader_factory.py
def create_dataloaders(dataset_name, data_dir, batch_size, **kwargs):
    if dataset_name == "new_dataset":
        train_dataset = NewDataset(root=data_dir, train=True)
        test_dataset = NewDataset(root=data_dir, train=False)
        num_classes = 10  # 设置类别数
        # ...
```

#### C. 更新任务支持
```python
# src/trainers/base_trainer.py
SUPPORTED_TASKS = {
    'image_classification': {
        'supported_datasets': ['cifar10', 'custom', 'new_dataset'],  # 添加新数据集
        # ...
    }
}
```

### 3. 添加新任务类型

#### A. 定义任务配置
```python
# src/trainers/base_trainer.py
SUPPORTED_TASKS['new_task'] = {
    'description': '新任务类型',
    'supported_datasets': ['dataset1', 'dataset2'],
    'model_factory': 'get_new_task_model',
    'default_model': 'default_model_name'
}
```

#### B. 创建模型工厂
```python
# src/models/new_task_models.py
def get_new_task_model(model_name, **kwargs):
    """新任务的模型工厂函数"""
    # 实现模型创建逻辑
```

#### C. 更新配置文件
```yaml
# config/new_task.yaml
task:
  tag: "new_task"
  description: "新任务类型"
```

### 4. 修改训练逻辑

#### A. 自定义训练循环
```python
# 继承base_trainer或创建新的训练器
class CustomTrainer(BaseTrainer):
    def train_epoch(self, dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch):
        # 自定义训练逻辑
        pass
```

#### B. 添加新的损失函数
```python
# src/losses/custom_loss.py
class CustomLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, predictions, targets):
        # 实现损失计算
        pass

# src/losses/loss_factory.py
def get_loss_function(loss_name, **kwargs):
    if loss_name == "custom_loss":
        return CustomLoss(**kwargs)
```

## ⚠️ 常见陷阱和注意事项

### 1. 配置文件陷阱
```yaml
# ❌ 错误：任务类型与数据集不匹配
task:
  tag: "image_classification"
data:
  type: "ucf101_video"  # 视频数据集用于图像任务

# ✅ 正确：任务类型与数据集匹配
task:
  tag: "video_classification"
data:
  type: "ucf101_video"
```

### 2. 模型注册陷阱
```python
# ❌ 错误：忘记设置task_type
'new_model': {
    'library': 'timm',
    # 缺少 'task_type': 'image_classification'
}

# ✅ 正确：完整的模型配置
'new_model': {
    'library': 'timm',
    'task_type': 'image_classification',
    'adapt_cifar': True
}
```

### 3. 数据集接口陷阱
```python
# ❌ 错误：返回格式不一致
def __getitem__(self, idx):
    return data  # 缺少label

# ✅ 正确：统一返回格式
def __getitem__(self, idx):
    return data, label  # (input, target)
```

## 🧪 测试和验证

### 1. 功能测试
```bash
# 测试图像分类
python scripts/train.py --config config/grid.yaml --epochs 1

# 测试视频分类  
python scripts/train.py --config config/ucf101_video.yaml --epochs 1

# 测试网格搜索
python scripts/grid_search.py --config config/grid.yaml --max_experiments 2
```

### 2. 配置验证
```python
# 验证新配置文件
python -c "
from src.utils.config_parser import parse_arguments
args, config = parse_arguments('single_experiment')
print('配置验证成功')
"
```

### 3. 模型验证
```python
# 验证新模型
python -c "
from src.models.model_registry import create_model_unified
model = create_model_unified('new_model', num_classes=10)
print('模型创建成功')
"
```

## 🚀 快速开发模板

### 1. 新模型开发模板
```python
# 步骤1: 注册模型 (src/models/model_registry.py)
MODEL_REGISTRY['my_new_model'] = {
    'library': 'timm',
    'adapt_cifar': True,
    'task_type': 'image_classification'
}

# 步骤2: 测试模型创建
python -c "
from src.models.model_registry import create_model_unified
model = create_model_unified('my_new_model', num_classes=10)
print(f'模型参数量: {sum(p.numel() for p in model.parameters())}')
"

# 步骤3: 创建配置文件
# config/my_experiment.yaml
task:
  tag: "image_classification"
model:
  type: "my_new_model"
# ... 其他配置

# 步骤4: 运行测试
python scripts/train.py --config config/my_experiment.yaml --epochs 1
```

### 2. 新数据集开发模板
```python
# 步骤1: 创建数据集类 (src/datasets/my_dataset.py)
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        # 加载数据索引

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 加载数据和标签
        data, label = self.load_item(idx)
        if self.transform:
            data = self.transform(data)
        return data, label

# 步骤2: 注册到工厂 (src/datasets/dataloader_factory.py)
elif dataset_name == "my_dataset":
    train_dataset = MyDataset(root=data_dir, train=True)
    test_dataset = MyDataset(root=data_dir, train=False)
    num_classes = 10  # 设置正确的类别数

# 步骤3: 更新任务支持 (src/trainers/base_trainer.py)
SUPPORTED_TASKS['image_classification']['supported_datasets'].append('my_dataset')
```

### 3. 实验配置模板
```yaml
# 基础实验配置模板
task:
  tag: "image_classification"  # 或 "video_classification"
  description: "实验描述"

training:
  exp_name: "my_experiment"
  save_model: true
  model_save_path: "models/my_model.pth"

swanlab:
  project_name: "MyProject"
  description: "实验说明"

data:
  type: "cifar10"  # 数据集类型
  root: "./data"
  num_workers: 8

model:
  type: "resnet18"  # 模型类型
  params:
    pretrained: true

hp:
  batch_size: 128
  learning_rate: 0.001
  epochs: 10
  dropout: 0.1

optimizer:
  name: "adam"
  params:
    weight_decay: 0.0001

scheduler:
  name: "cosine"
  params:
    T_max: 10

loss:
  name: "crossentropy"
```

## 🔧 调试和故障排除

### 1. 常见错误及解决方案

#### 错误: "不支持的任务类型"
```python
# 原因：task_tag不在SUPPORTED_TASKS中
# 解决：检查配置文件中的task.tag字段
task:
  tag: "image_classification"  # 确保拼写正确
```

#### 错误: "任务不支持数据集"
```python
# 原因：数据集类型与任务类型不匹配
# 解决：确保数据集在任务的supported_datasets中
# 或者添加数据集支持
SUPPORTED_TASKS['image_classification']['supported_datasets'].append('new_dataset')
```

#### 错误: "不支持的模型"
```python
# 原因：模型未在MODEL_REGISTRY中注册
# 解决：添加模型注册或检查模型名称拼写
```

### 2. 性能调优建议

#### A. 数据加载优化
```python
# 增加数据加载工作进程
data:
  num_workers: 12  # 根据CPU核心数调整

# 使用数据预取
dataloader = DataLoader(dataset, num_workers=12, pin_memory=True)
```

#### B. 训练速度优化
```python
# 使用混合精度训练
accelerator = Accelerator(mixed_precision='fp16')

# 调整批大小
hp:
  batch_size: 256  # 根据GPU内存调整
```

## 📖 进阶学习资源

### 1. 项目完整文档
- `PROJECT_DOCUMENTATION.md` - 完整的项目文档，包含所有功能特性

### 2. 最佳实践示例
- `config/` - 配置文件示例
- `src/models/model_registry.py` - 统一接口设计示例
- `src/datasets/video_dataset.py` - 抽象基类设计示例

## 🤝 贡献指南

### 1. 代码提交规范
- 遵循现有的代码风格和注释规范
- 为新功能添加相应的测试
- 更新相关文档和配置示例

### 2. 测试要求
- 所有新功能必须通过基础功能测试
- 确保向后兼容性
- 提供使用示例和文档

### 3. 文档更新
- 更新DEVELOPER_GUIDE.md中的相关部分
- 添加配置文件示例
- 更新架构图和依赖关系

---

**更新时间**: 2025-01-10
**适用版本**: 重构后版本
**维护者**: EasyTrain开发团队
