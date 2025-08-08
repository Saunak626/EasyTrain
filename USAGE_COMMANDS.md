# 训练框架使用指令

本文档提供了训练框架的各种使用场景的具体命令示例。

## 1. 单次训练 (scripts/train.py)

### 1.1 单卡训练

```bash
# 使用默认配置
python scripts/train.py --config config/base.yaml

# 指定实验名称
python scripts/train.py --config config/base.yaml --exp_name my_experiment

# 覆盖超参数
python scripts/train.py --config config/base.yaml --learning_rate 0.01 --batch_size 128 --epochs 10

# 指定GPU
python scripts/train.py --config config/base.yaml --gpu_ids "0"

# 强制使用CPU
python scripts/train.py --config config/base.yaml --use_cpu
```

### 1.2 多卡训练

```bash
# 使用多卡训练（自动调用accelerate launch）
python scripts/train.py --config config/base.yaml --multi_gpu

# 指定多个GPU
python scripts/train.py --config config/base.yaml --multi_gpu --gpu_ids "0,1,2,3"

# 多卡训练 + 自定义实验名称
python scripts/train.py --config config/base.yaml --multi_gpu --exp_name multi_gpu_test

# 多卡训练 + 超参数调整
python scripts/train.py --config config/base.yaml --multi_gpu --learning_rate 0.001 --batch_size 256
```

## 2. 网格搜索 (scripts/grid_search.py)

### 2.1 单卡网格搜索

```bash
# 使用默认配置进行网格搜索
python scripts/grid_search.py --config config/grid.yaml

# 限制最大实验数量
python scripts/grid_search.py --config config/grid.yaml --max_experiments 10

# 指定GPU
python scripts/grid_search.py --config config/grid.yaml --gpu_ids "0"

# 自定义结果保存文件
python scripts/grid_search.py --config config/grid.yaml --results_file my_grid_results.csv
```

### 2.2 多卡网格搜索

```bash
# 多卡网格搜索
python scripts/grid_search.py --config config/grid.yaml --multi_gpu

# 多卡 + 限制实验数量
python scripts/grid_search.py --config config/grid.yaml --multi_gpu --max_experiments 5

# 多卡 + 指定GPU
python scripts/grid_search.py --config config/grid.yaml --multi_gpu --gpu_ids "2,3"

# 多卡 + 自定义结果文件
python scripts/grid_search.py --config config/grid.yaml --multi_gpu --results_file grid_multi_gpu.csv
```

## 3. 配置文件说明

### 3.1 单次训练配置文件
- `config/base.yaml` - 基础训练配置
- `config/config.yaml` - 通用配置文件

### 3.2 网格搜索配置文件
- `config/grid.yaml` - 网格搜索配置，包含参数网格定义

## 4. 常用参数说明

### 4.1 通用参数
- `--config`: 配置文件路径
- `--exp_name`: 实验名称
- `--gpu_ids`: 指定GPU ID，如 "0,1,2"
- `--use_cpu`: 强制使用CPU训练
- `--multi_gpu`: 启用多卡训练

### 4.2 超参数覆盖
- `--learning_rate`: 学习率
- `--batch_size`: 批大小
- `--epochs`: 训练轮数
- `--dropout`: Dropout率

### 4.3 网格搜索专用参数
- `--max_experiments`: 最大实验数量限制
- `--save_results`: 是否保存结果表格（默认开启）
- `--results_file`: 结果保存文件名

## 5. 实际使用示例

### 5.1 快速测试
```bash
# 单卡快速测试（5个epoch）
python scripts/train.py --config config/base.yaml --epochs 5 --exp_name quick_test

# 多卡快速测试
python scripts/train.py --config config/base.yaml --multi_gpu --epochs 5 --exp_name quick_multi_test
```

### 5.2 完整训练
```bash
# 单卡完整训练
python scripts/train.py --config config/base.yaml --epochs 100 --exp_name full_training

# 多卡完整训练
python scripts/train.py --config config/base.yaml --multi_gpu --epochs 100 --exp_name full_multi_training
```

### 5.3 网格搜索示例
```bash
# 小规模网格搜索（测试）
python scripts/grid_search.py --config config/grid.yaml --max_experiments 3

# 完整网格搜索（多卡）
python scripts/grid_search.py --config config/grid.yaml --multi_gpu --max_experiments 20
```

## 6. 输出说明

### 6.1 单次训练输出
- 训练过程日志
- 每个epoch的损失和准确率
- 最终的最佳准确率
- SwanLab追踪链接

### 6.2 网格搜索输出
- 每个实验的参数和结果
- 最佳实验的详细信息
- 前3名实验排行榜
- CSV结果文件（包含所有实验的详细数据）

## 7. 故障排除

### 7.1 GPU相关问题
```bash
# 检查GPU可用性
nvidia-smi

# 如果GPU内存不足，减少batch_size
python scripts/train.py --config config/base.yaml --batch_size 32

# 强制使用CPU（如果GPU有问题）
python scripts/train.py --config config/base.yaml --use_cpu
```

### 7.2 多卡训练问题
```bash
# 检查accelerate配置
accelerate config

# 手动指定GPU
python scripts/train.py --config config/base.yaml --multi_gpu --gpu_ids "0,1"
```