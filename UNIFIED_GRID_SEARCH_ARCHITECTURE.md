# 统一网格搜索架构说明

## 概述

本项目已重构为统一的网格搜索架构，移除了单模型训练方式，所有训练都通过网格搜索模式进行。这种设计提供了更好的实验管理和参数优化能力。

## 架构变化

### 1. 配置文件结构 (`config/grid.yaml`)

配置文件现在分为两个主要部分：

#### 固定配置部分（不参与网格搜索）
- **训练配置**: 实验名称、模型保存路径等
- **SwanLab配置**: 项目名称、描述等
- **数据配置**: 数据集类型、路径、预处理参数等
- **GPU配置**: 设备ID、自动选择等
- **基础模型/优化器/调度器配置**: 框架结构（具体类型由网格搜索参数决定）

#### 网格搜索配置部分
```yaml
grid_search:
  # 网格搜索参数定义（所有超参数都在这里配置）
  grid:
    # 模型相关参数
    model_type: ["resnet18", "resnet50", "efficientnet_b0"]
    
    # 超参数
    learning_rate: [0.001, 0.01, 0.0001]
    batch_size: [128, 256, 512]
    dropout: [0.1, 0.2, 0.3]
    epochs: [5, 10, 15]
    
    # 优化器参数
    optimizer_type: ["adam", "sgd", "adamw"]
    weight_decay: [0, 0.0001, 0.001]
    
    # 调度器参数
    scheduler_type: ["onecycle", "step", "cosine"]
  
  # 固定参数（所有实验共享）
  fixed:
    # 可以在这里设置一些固定的参数
  
  # 网格搜索设置
  max_experiments: 50
  continue_on_error: true
  parallel_jobs: 1
  save_results: true
  results_file: "grid_search_results.csv"
```

### 2. 脚本重构

#### `scripts/train.py`
- 现在作为统一的训练入口点
- 自动检测是网格搜索调用还是单个实验调用
- 如果没有 `--experiment_name` 参数，则启动网格搜索
- 如果有 `--experiment_name` 参数，则作为单个实验运行

#### `scripts/grid_search.py`
- 保持原有的网格搜索逻辑
- 更新了参数处理，支持新的配置结构
- 处理 `model_type` 到 `model_name` 的参数映射

#### `src/utils/config_parser.py`
- 移除了单独的训练模式
- 统一使用网格搜索解析逻辑
- 支持两种模式：`grid_search` 和 `single_experiment`
- 增强了参数覆盖功能，支持优化器和调度器参数

## 使用方法

### 1. 启动网格搜索
```bash
# 使用默认配置
python scripts/train.py

# 使用自定义配置
python scripts/train.py --config config/grid.yaml

# 限制实验数量
python scripts/train.py --max_experiments 10

# 使用多卡训练
python scripts/train.py --multi_gpu
```

### 2. 直接调用网格搜索脚本
```bash
python scripts/grid_search.py --config config/grid.yaml
```

## 优势

1. **统一架构**: 所有训练都通过网格搜索，简化了代码维护
2. **灵活配置**: 可以轻松配置单个实验（网格中只有一个参数组合）
3. **参数管理**: 所有超参数都在配置文件中集中管理
4. **实验追踪**: 每个实验都有完整的参数记录和结果追踪
5. **扩展性**: 容易添加新的参数类型和优化策略

## 迁移指南

如果您之前使用单模型训练方式，现在需要：

1. 将您的参数配置移动到 `grid_search.grid` 部分
2. 如果只想运行单个实验，将参数设置为单元素列表
3. 使用 `python scripts/train.py` 替代之前的训练命令

例如，之前的单模型训练：
```bash
python scripts/train.py --learning_rate 0.001 --batch_size 256 --model_name resnet18
```

现在需要在配置文件中设置：
```yaml
grid_search:
  grid:
    learning_rate: [0.001]
    batch_size: [256]
    model_type: ["resnet18"]
```

然后运行：
```bash
python scripts/train.py
```

## 注意事项

1. 配置文件中的 `model.type`、`optimizer.type`、`scheduler.type` 现在由网格搜索参数动态设置
2. 所有超参数都应该在 `grid_search.grid` 中定义
3. 固定的配置（如数据集路径、GPU设置）保持在原有位置
4. 网格搜索会生成大量实验，请合理设置 `max_experiments` 参数