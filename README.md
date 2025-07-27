# 深度学习训练框架

一个简洁、高效的深度学习训练框架，支持CIFAR-10、自定义数据集和预训练模型的训练。

## 🎯 核心特性

### 预训练模型支持
- **15+种主流预训练模型**：集成timm库，支持ResNet、EfficientNet、MobileNet、ViT、DenseNet、RegNet等系列
- **自动适配**：自动适配CIFAR-10的32x32输入尺寸和自定义数据集的任意尺寸
- **灵活配置**：支持冻结主干网络、只训练分类头等高级配置
- **显著性能提升**：预训练ResNet18在CIFAR-10上1个epoch达到86%准确率（vs 简单CNN的50%）

### 批量模型测试
- **预训练模型网格搜索**：支持批量测试多个预训练模型的性能
- **交互式选择**：用户可以选择要测试的具体模型
- **自动配置**：自动为每个模型生成临时配置文件
- **结果追踪**：每个模型的训练结果都会记录到SwanLab

### 简化的设计
- **函数式训练器**：只有 `train_epoch` 和 `test_epoch` 两个核心函数
- **参数化数据加载**：通过参数直接调用数据加载器，无需复杂的配置函数
- **独立数据集脚本**：每个数据集有独立的实现脚本

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### CIFAR-10训练（推荐入门）
```bash
# 单卡训练
python scripts/train.py --config config/config.yaml --epochs 10

# 多卡训练
python scripts/train.py --config config/config.yaml --multi_gpu --epochs 10

# 预训练模型训练
python scripts/train.py --config config/config.yaml --epochs 5
# 修改config.yaml中的model.type为预训练模型名称，如"resnet18"

# 网格搜索
python scripts/grid_search.py --search_type params  # 参数网格搜索
python scripts/grid_search.py --search_type models --models resnet18,efficientnet_b0  # 预训练模型搜索
```

### 预训练模型网格搜索
```bash
# 搜索指定的预训练模型
python scripts/grid_search.py --search_type models --models resnet18,efficientnet_b0

# 搜索所有预训练模型
python scripts/grid_search.py --search_type models --models all

# 查看帮助
python scripts/grid_search.py --help
```

## 📁 项目结构

```
training_framework/
├── config/
│   ├── base.yaml              # 基础配置模板（用于自定义修改）
│   ├── config.yaml            # CIFAR-10展示配置
│   └── grid.yaml              # 网格搜索配置
├── scripts/
│   ├── train.py               # 单次训练脚本
│   └── grid_search.py         # 网格搜索脚本（支持预训练模型搜索）
├── src/
│   ├── trainers/
│   │   └── base_trainer.py    # 简化的训练器（train_epoch + test_epoch）
│   ├── models/
│   │   └── net.py             # 网络模型集合（包含预训练模型支持）
│   ├── losses/
│   │   └── image_loss.py      # 图像任务损失函数
│   ├── optimizers/
│   │   └── optim.py           # 优化器（Adam, AdamW, SGD）
│   ├── schedules/
│   │   └── scheduler.py       # 学习率调度器
│   ├── data_preprocessing/
│   │   ├── cifar10_dataset.py # CIFAR-10数据集（简化版）
│   │   └── dataset.py         # 自定义数据集模板
│   └── utils/
│       └── data_utils.py      # 通用工具函数
├── data/
│   ├── README.md              # 数据说明文档
│   └── raw/                   # 原始数据（不上传Git）
├── example.py                 # 预训练模型示例脚本
├── DEVELOPMENT_GUIDE.md       # 自定义组件开发指南
└── requirements.txt           # 依赖（包含timm和torchinfo）
```

## ⚙️ 配置文件

### CIFAR-10展示配置 (`config/config.yaml`)
```yaml
data:
  type: "cifar10"
  root: "./data"
  download: true
  augment: true
  num_workers: 4

model:
  type: "simple_cnn"  # 或预训练模型名称如"resnet18"
  params:
    freeze_backbone: false  # 是否冻结预训练模型的主干网络

hyperparameters:
  learning_rate: 0.001
  batch_size: 128
  epochs: 10
  dropout: 0.1
```

### 基础配置模板 (`config/base.yaml`)
包含所有可配置选项和详细说明，用于指导自定义修改。

## 🔧 支持的组件

### 数据集
- ✅ **CIFAR-10**: 内置支持，自动下载
- ✅ **自定义数据集**: 支持目录结构分类和CSV索引

### 模型
- ✅ **SimpleNet**: 全连接网络
- ✅ **SimpleCNN**: 卷积网络
- ✅ **SimpleResNet**: 残差网络
- ✅ **预训练模型**: 支持timm库中的所有预训练模型
  - ResNet系列: `resnet18`, `resnet34`, `resnet50`, `resnet101`
  - EfficientNet系列: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
  - MobileNet系列: `mobilenetv3_small_100`, `mobilenetv3_large_100`
  - Vision Transformer: `vit_tiny_patch16_224`, `vit_small_patch16_224`
  - DenseNet系列: `densenet121`, `densenet169`
  - RegNet系列: `regnety_002`, `regnety_004`, `regnety_008`

### 优化器
- ✅ **Adam**: 自适应学习率优化器
- ✅ **AdamW**: 带权重衰减的Adam
- ✅ **SGD**: 随机梯度下降

### 学习率调度器
- ✅ **OneCycleLR**: 单周期学习率策略
- ✅ **StepLR**: 阶梯式学习率衰减
- ✅ **CosineAnnealingLR**: 余弦退火
- ✅ **ReduceLROnPlateau**: 基于指标的自适应调整

### 损失函数
- ✅ **CrossEntropyLoss**: 交叉熵损失
- ✅ **FocalLoss**: 处理类别不平衡
- ✅ **LabelSmoothingLoss**: 标签平滑
- ✅ **MSELoss**: 均方误差损失

## 🎯 使用场景

### 场景1: CIFAR-10基准测试
```bash
python scripts/train.py --config config/config.yaml --experiment_name cifar10_baseline
```

### 场景2: 预训练模型训练
```bash
# 修改config.yaml中的model.type为预训练模型名称
# model:
#   type: "resnet18"
#   params:
#     freeze_backbone: false

python scripts/train.py --config config/config.yaml --experiment_name resnet18_cifar10
```

### 场景3: 预训练模型网格搜索
```bash
# 测试指定的预训练模型
python scripts/grid_search.py --search_type models --models resnet18,efficientnet_b0 --epochs 5

# 测试所有预训练模型
python scripts/grid_search.py --search_type models --models all --epochs 3
```

### 场景4: 自定义数据集训练
1. **准备数据集**：
   ```
   data/raw/my_dataset/
   ├── class1/
   │   ├── img1.jpg
   │   └── img2.jpg
   └── class2/
       ├── img3.jpg
       └── img4.jpg
   ```

2. **修改配置**（参考`config/base.yaml`）：
   ```yaml
   data:
     type: "custom"
     data_dir: "./data/raw/my_dataset"
     image_size: 224
   
   model:
     type: "resnet18"  # 使用预训练模型
     params:
       freeze_backbone: true  # 冻结主干网络，只训练分类头
   ```

3. **开始训练**：
   ```bash
   python scripts/train.py --config config/base.yaml
   ```

### 场景5: 超参数调优
```bash
# 参数网格搜索
python scripts/grid_search.py --search_type params --max_experiments 10

# 多卡参数网格搜索
python scripts/grid_search.py --search_type params --multi_gpu

# 手动调参
python scripts/train.py --config config/config.yaml \
  --learning_rate 0.01 --batch_size 64 --epochs 50
```

## 🎛️ 命令行参数

```bash
python scripts/train.py \
  --config config/config.yaml \      # 配置文件路径
  --experiment_name my_exp \          # 实验名称
  --learning_rate 0.01 \              # 学习率
  --batch_size 64 \                   # 批大小
  --epochs 50 \                       # 训练轮数
  --multi_gpu \                       # 多卡训练
  --use_cpu                           # CPU训练
```

## 🚨 常见问题

### 1. GPU内存不足
```bash
# 减少批大小
python scripts/train.py --config config/config.yaml --batch_size 32

# 使用CPU训练
python scripts/train.py --config config/config.yaml --use_cpu

# 冻结预训练模型主干网络
# 在配置文件中设置 model.params.freeze_backbone: true
```

### 2. 预训练模型相关问题
```bash
# 安装timm库
pip install timm

# 查看可用的预训练模型
python -c "from src.models.net import list_pretrained_models; print(list_pretrained_models())"

# 使用较小的预训练模型
# 如 mobilenetv3_small_100 而不是 resnet101
```

### 3. 数据加载慢
```yaml
# 增加工作进程数
data:
  num_workers: 8  # 根据CPU核心数调整
```

### 4. 训练不收敛
```yaml
# 调整学习率
hyperparameters:
  learning_rate: 0.0001  # 降低学习率

# 或使用不同的优化器
optimizer:
  type: "adamw"
  params:
    weight_decay: 0.01

# 对于预训练模型，尝试冻结主干网络
model:
  params:
    freeze_backbone: true
```

## 📈 性能优化建议

### 1. 数据加载优化
- 使用SSD存储数据
- 适当增加`num_workers`
- 启用`pin_memory=True`

### 2. 训练加速
- 使用多卡训练：`--multi_gpu`
- 选择合适的批大小
- 使用预训练模型可以更快收敛

### 3. 内存优化
- 减少批大小
- 冻结预训练模型的主干网络
- 使用较小的预训练模型

### 4. 预训练模型优化
- 对于小数据集，建议冻结主干网络
- 对于大数据集，可以微调整个网络
- 根据任务选择合适的预训练模型

## 🔧 核心功能展示

### 预训练模型自动适配CIFAR-10
```python
# 自动调整第一个卷积层以适应32x32输入
if input_size == 32:  # CIFAR-10尺寸
    self.model.conv1 = nn.Conv2d(
        self.model.conv1.in_channels, 
        self.model.conv1.out_channels, 
        kernel_size=3, stride=1, padding=1, bias=False
    )
    # 移除最大池化层
    self.model.maxpool = nn.Identity()
```

### 性能对比
```
简单CNN (simple_cnn):
- 1 epoch: ~50% 准确率

预训练ResNet18 (resnet18):
- 1 epoch: 86.07% 准确率 🚀
```

### 训练器简化
```python
# 只有两个核心函数
def train_epoch(dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch):
    # 训练一个epoch的逻辑

def test_epoch(dataloader, model, loss_fn, accelerator, epoch):
    # 测试一个epoch的逻辑
```

### 数据加载简化
```python
# 直接参数化调用
if dataset_type == 'cifar10':
    train_loader, test_loader = get_cifar10_dataloaders(
        root='./data', batch_size=128, augment=True
    )
elif dataset_type == 'custom':
    train_loader, val_loader = get_custom_dataloaders(
        data_dir='./data/custom', batch_size=32, image_size=224
    )
```

## 🎉 开始你的第一个实验

```bash
# 1. 克隆项目
git clone <repository_url>
cd training_framework

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行CIFAR-10基准测试
python scripts/train.py --config config/config.yaml --epochs 5

# 4. 尝试预训练模型
# 修改config.yaml中的model.type为"resnet18"
python scripts/train.py --config config/config.yaml --epochs 5

# 5. 预训练模型网格搜索
python scripts/grid_search.py --search_type models --models resnet18,efficientnet_b0

# 6. 查看结果
# 训练完成后，检查SwanLab面板查看训练曲线和指标
```

## 📚 文档

- **DEVELOPMENT_GUIDE.md**: 自定义组件开发指南
- **data/README.md**: 数据说明文档

现在你已经准备好使用这个支持预训练模型的训练框架进行深度学习实验了！🚀
