"""

核心优化：
- 一次启动，串行执行所有实验
- 所有实验在同一进程中顺序运行
- 无需为每个实验创建子进程

启动方式：
- 单卡：python scripts/grid_search_unified.py --config ...
- 多卡：accelerate launch scripts/grid_search_unified.py --config ...

"""
import itertools
import yaml
import os
import sys
import csv
import json
import fcntl

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments

# ======================
# 模块级常量配置
# ======================

# 网格搜索相关常量
GRID_SEARCH_CONSTANTS = {
    'model_type_key': 'model.type',
    'batch_size_key': 'hp.batch_size',
    'group_key': 'group',
    'excluded_params': ['model.type', 'hp.batch_size'],
    'csv_base_columns': [
        'exp_name', 'model.type', 'group', 'success', 'trained_epochs',
        # 🎯 多标签分类关键指标（优先显示）
        'best_weighted_f1', 'best_weighted_accuracy', 'best_macro_accuracy', 'best_micro_accuracy',
        'best_macro_f1', 'best_micro_f1', 'best_macro_precision', 'best_macro_recall',
        'final_weighted_f1', 'final_weighted_accuracy', 'final_macro_accuracy', 'final_micro_accuracy',
        'final_macro_f1', 'final_micro_f1',
        # 传统字段（向后兼容）
        'best_accuracy', 'final_accuracy'
    ],
    'common_runtime_params': [
        'data_percentage',
        'optimizer.name', 'scheduler.name', 'loss.name'
    ],
    'excluded_csv_params': ['epochs', 'batch_size', 'learning_rate']
}

# ======================
# 参数组合生成器类
# ======================

class ParameterCombinationGenerator:
    """参数组合生成器
    
    负责处理网格搜索的参数组合生成逻辑，支持分组式配置和模型-batch_size智能配对。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化参数组合生成器
        
        Args:
            config: 网格搜索配置字典
        """
        self.config = config
        self.constants = GRID_SEARCH_CONSTANTS
    
    def generate_combinations(self) -> List[Dict[str, Any]]:
        """生成参数组合的主入口函数
        
        Returns:
            参数组合列表，每个字典代表一组实验参数
        """
        gs = (self.config or {}).get("grid_search", {}) or {}
        fixed = gs.get("fixed", {}) or {}
        models_to_train = self.config.get("models_to_train", [])
        
        # 分组式配置处理
        if "groups" in gs and gs["groups"]:
            print(f"📋 使用分组式网格搜索配置")
            return self._generate_combinations_by_groups(gs["groups"], fixed, models_to_train)
        
        # 边界情况：无搜索参数，从基础配置中提取信息
        else:
            print(f"⚠️  未找到groups配置，从基础配置中提取参数")
            base_params = {}
            
            # 从基础配置中提取模型类型
            if 'model' in self.config and 'type' in self.config['model']:
                base_params[self.constants['model_type_key']] = self.config['model']['type']
            
            # 从基础配置中提取其他参数
            if 'optimizer' in self.config and 'name' in self.config['optimizer']:
                base_params['optimizer.name'] = self.config['optimizer']['name']
            
            if 'scheduler' in self.config and 'name' in self.config['scheduler']:
                base_params['scheduler.name'] = self.config['scheduler']['name']
            
            if 'loss' in self.config and 'name' in self.config['loss']:
                base_params['loss.name'] = self.config['loss']['name']
            
            # 合并固定参数
            base_params.update(fixed)
            
            return [base_params] if base_params else []
    
    def _generate_combinations_by_groups(self, groups: Dict[str, Dict], fixed: Dict[str, Any],
                                        models_to_train: List[str]) -> List[Dict[str, Any]]:
        """分组式参数组合生成
        
        Args:
            groups: 分组配置字典
            fixed: 固定参数字典
            models_to_train: 要训练的模型列表
        
        Returns:
            参数组合列表
        """
        all_combinations = []
        print(f"🎯 发现 {len(groups)} 个模型组:")
        
        for group_name, group_config in groups.items():
            print(f"   - {group_name}: {group_config.get(self.constants['model_type_key'], [])}")
        
        print()  # 空行分隔
        
        for group_name, group_config in groups.items():
            print(f"🔧 处理模型组: {group_name}")
            group_combinations = self._process_single_group(
                group_name, group_config, fixed, models_to_train
            )

            if group_combinations:
                all_combinations.extend(group_combinations)
                print(f"   ✅ 组 {group_name} 生成 {len(group_combinations)} 个组合 "
                      f"({len(group_config.get(self.constants['model_type_key'], []))}模型 × "
                      f"{len(group_combinations) // max(1, len(group_config.get(self.constants['model_type_key'], [])))}参数组合)")
            else:
                print(f"   ⏭️  跳过组 {group_name}：无启用的模型")
            print()  # 空行分隔

        print(f"🎉 分组式搜索总计生成 {len(all_combinations)} 个组合")
        return all_combinations

    def _process_single_group(self, group_name: str, group_config: Dict[str, Any],
                             fixed: Dict[str, Any], models_to_train: List[str]) -> List[Dict[str, Any]]:
        """处理单个模型组的参数组合

        Args:
            group_name: 组名
            group_config: 组配置
            fixed: 固定参数
            models_to_train: 要训练的模型列表

        Returns:
            该组的参数组合列表
        """
        model_type_key = self.constants['model_type_key']
        batch_size_key = self.constants['batch_size_key']

        # 提取模型列表和batch_size列表
        models = group_config.get(model_type_key, [])
        batch_sizes = group_config.get(batch_size_key, [])

        # 过滤：只保留在 models_to_train 中的模型
        if models_to_train:
            enabled_models = [m for m in models if m in models_to_train]
            if not enabled_models:
                return []
            models = enabled_models

        # 打印组内配置
        print(f"   📋 组内配置:")
        print(f"      {model_type_key}: {models} (长度: {len(models)})")
        print(f"      {batch_size_key}: {batch_sizes} (长度: {len(batch_sizes)})")

        # 智能配对：模型和batch_size
        model_batch_pairs = self._pair_models_with_batch_sizes(models, batch_sizes)

        # 提取其他参数（排除model.type和hp.batch_size）
        other_params = {
            k: v for k, v in group_config.items()
            if k not in self.constants['excluded_params']
        }

        # 生成其他参数的笛卡尔积
        if other_params:
            other_keys = list(other_params.keys())
            other_values = [other_params[k] if isinstance(other_params[k], list) else [other_params[k]]
                          for k in other_keys]
            other_combinations = [dict(zip(other_keys, combo)) for combo in itertools.product(*other_values)]
        else:
            other_combinations = [{}]

        # 组合：(模型, batch_size) × 其他参数
        group_combinations = []
        for (model, batch_size), other_combo in itertools.product(model_batch_pairs, other_combinations):
            combo = {
                model_type_key: model,
                batch_size_key: batch_size,
                self.constants['group_key']: group_name
            }
            combo.update(other_combo)
            combo.update(fixed)
            group_combinations.append(combo)

        return group_combinations

    def _pair_models_with_batch_sizes(self, models: List[str], batch_sizes: List[int]) -> List[Tuple[str, int]]:
        """智能配对模型和batch_size

        Args:
            models: 模型列表
            batch_sizes: batch_size列表

        Returns:
            (模型, batch_size) 配对列表
        """
        if len(batch_sizes) == 1:
            # 情况1：只有一个batch_size，所有模型使用相同的batch_size
            return [(m, batch_sizes[0]) for m in models]

        elif len(batch_sizes) == len(models):
            # 情况2：batch_size数量与模型数量相同，按顺序配对
            print(f"   ✅ 长度匹配，将按顺序配对")
            return list(zip(models, batch_sizes))

        else:
            # 情况3：长度不匹配，扩充batch_size列表
            print(f"   🔄 扩充batch_size: {batch_sizes} (扩充到与model.type长度一致)")
            expanded_batch_sizes = []
            for i in range(len(models)):
                expanded_batch_sizes.append(batch_sizes[i % len(batch_sizes)])
            print(f"   🔄 扩充后: {expanded_batch_sizes}")

            # 打印配对结果
            pairs = list(zip(models, expanded_batch_sizes))
            print(f"   🎯 启用的模型配对: {dict(pairs)}")
            return pairs


# ======================
# 实验结果管理器类
# ======================

class ExperimentResultsManager:
    """实验结果管理器

    负责管理网格搜索的结果文件，包括主结果CSV、详情CSV和单实验文件。
    """

    def __init__(self, csv_filepath: str, details_filepath: str, grid_search_dir: str):
        """初始化结果管理器

        Args:
            csv_filepath: 主结果CSV文件路径
            details_filepath: 详情CSV文件路径
            grid_search_dir: 网格搜索目录
        """
        self.csv_filepath = csv_filepath
        self.details_filepath = details_filepath
        self.grid_search_dir = grid_search_dir
        self.experiments_dir = os.path.join(grid_search_dir, "experiments")
        self.fieldnames = []

        # 创建实验目录
        os.makedirs(self.experiments_dir, exist_ok=True)
        print(f"📁 创建实验文件夹结构: {self.experiments_dir}")

    def initialize_csv_file(self, fieldnames: List[str]) -> None:
        """初始化CSV文件

        Args:
            fieldnames: CSV字段名列表
        """
        self.fieldnames = fieldnames

        # 初始化主结果CSV
        os.makedirs(os.path.dirname(self.csv_filepath), exist_ok=True)
        with open(self.csv_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        # 初始化详情CSV
        print(f"📋 初始化详情表: {self.details_filepath}")
        with open(self.details_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def append_result_to_csv(self, result: Dict[str, Any]) -> None:
        """追加结果到CSV文件

        Args:
            result: 实验结果字典
        """
        # 写入主结果CSV
        with open(self.csv_filepath, 'a', newline='', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                row = self._prepare_csv_row(result)
                writer.writerow(row)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # 写入详情CSV
        with open(self.details_filepath, 'a', newline='', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # 保存单实验JSON文件
        self._save_single_experiment_file(result)

    def _prepare_csv_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """准备CSV行数据

        Args:
            result: 实验结果字典

        Returns:
            CSV行字典
        """
        row = {}
        params = result.get("params", {})

        for field in self.fieldnames:
            if field in result:
                row[field] = result[field]
            elif field in params:
                row[field] = params[field]
            else:
                row[field] = ""

        return row

    def _save_single_experiment_file(self, result: Dict[str, Any]) -> None:
        """保存单个实验的JSON文件

        Args:
            result: 实验结果字典
        """
        exp_name = result.get('exp_name', 'unknown')
        exp_dir = os.path.join(self.experiments_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        # 保存完整结果
        result_file = os.path.join(exp_dir, 'result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


# ======================
# 辅助函数
# ======================

def generate_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """生成参数组合（便捷函数）

    Args:
        config: 配置字典

    Returns:
        参数组合列表
    """
    generator = ParameterCombinationGenerator(config)
    return generator.generate_combinations()


def get_csv_fieldnames(all_params: List[Dict[str, Any]]) -> List[str]:
    """获取CSV字段名

    Args:
        all_params: 所有参数组合列表

    Returns:
        字段名列表
    """
    base_columns = GRID_SEARCH_CONSTANTS['csv_base_columns']
    common_runtime_params = GRID_SEARCH_CONSTANTS['common_runtime_params']
    excluded_csv_params = GRID_SEARCH_CONSTANTS['excluded_csv_params']

    # 收集所有参数键
    param_keys = set()
    for params in all_params:
        param_keys.update(params.keys())

    # 过滤掉排除的参数
    param_keys = [k for k in param_keys if k not in excluded_csv_params]

    # 组合字段名：基础列 + 运行时参数 + 其他参数
    fieldnames = base_columns.copy()

    # 添加常见运行时参数
    for param in common_runtime_params:
        if param not in fieldnames:
            fieldnames.append(param)

    # 添加其他参数
    for key in sorted(param_keys):
        if key not in fieldnames and key != GRID_SEARCH_CONSTANTS['group_key']:
            fieldnames.append(key)

    return fieldnames


def load_grid_config(config_path: str) -> Dict[str, Any]:
    """加载网格搜索配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def apply_param_overrides(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """应用参数覆盖到配置

    Args:
        config: 基础配置字典
        params: 参数覆盖字典

    Returns:
        更新后的配置字典
    """
    for key, value in params.items():
        if '.' in key:
            # 处理嵌套参数（如 model.type, hp.batch_size）
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            # 直接参数
            config[key] = value

    return config


# ======================
# 核心实验执行函数（统一版本）
# ======================

def run_single_experiment(params: Dict[str, Any], exp_id: str, config_path: str,
                         grid_search_dir: Optional[str] = None) -> Dict[str, Any]:
    """运行单个实验（统一版本 - 自动适配单卡/多卡）

    Args:
        params: 实验参数覆盖字典
        exp_id: 实验ID（如 "001", "002"）
        config_path: 配置文件路径
        grid_search_dir: 网格搜索目录（用于保存视频级别指标等文件）

    Returns:
        实验结果字典，包含以下字段：
        - success: 是否成功
        - exp_name: 实验名称
        - params: 实验参数
        - best_accuracy: 最佳准确率
        - final_accuracy: 最终准确率
        - trained_epochs: 训练轮数
        - error: 错误信息（如果失败）
        - error_type: 错误类型（如果失败）
    """
    exp_name = f"grid_{exp_id}"

    try:
        # 导入训练函数和GPU配置函数
        from src.trainers.base_trainer import run_training
        from src.utils.config_parser import setup_gpu_config

        # 加载基础配置
        config = load_grid_config(config_path)

        # 应用参数覆盖
        config = apply_param_overrides(config, params)

        # 将 grid_search_dir 添加到配置中
        if grid_search_dir:
            config['grid_search_dir'] = grid_search_dir

        # 配置GPU环境
        setup_gpu_config(config)

        # 直接调用训练函数（Accelerator 会自动处理单卡/多卡）
        result = run_training(config, exp_name)

        # 添加参数信息到结果中
        result["params"] = params

        return result

    except Exception as e:
        print(f"❌ 实验 {exp_name} 失败: {type(e).__name__}: {str(e)}")
        return {
            "success": False,
            "exp_name": exp_name,
            "params": params,
            "best_accuracy": 0.0,
            "final_accuracy": 0.0,
            "trained_epochs": 0,
            "error": str(e),
            "error_type": type(e).__name__
        }


# ======================
# 主网格搜索函数
# ======================

def run_grid_search(args):
    """运行网格搜索（串行执行所有实验）

    Args:
        args: 命令行参数

    Returns:
        退出码（0表示成功，1表示失败）
    """
    # 加载配置并生成参数组合
    config = load_grid_config(args.config)
    combinations = generate_combinations(config)

    if len(combinations) > args.max_experiments:
        combinations = combinations[:args.max_experiments]

    # 准备CSV文件 - 根据任务类型创建对应目录
    task_tag = config.get('task', {}).get('tag', '')
    dataset_type = config.get('data', {}).get('type', '')

    # 根据任务类型确定子目录名
    if 'multilabel' in task_tag.lower() or 'multilabel' in dataset_type.lower():
        if 'neonatal' in dataset_type.lower():
            task_subdir = "neonatal_multilabel"
        else:
            task_subdir = "multilabel_classification"
    elif 'video' in task_tag.lower():
        task_subdir = "video_classification"
    elif 'image' in task_tag.lower():
        task_subdir = "image_classification"
    elif 'text' in task_tag.lower():
        task_subdir = "text_classification"
    else:
        task_subdir = dataset_type.replace('_', '_').lower() or "general"

    results_dir = os.path.join("runs", task_subdir)

    # 创建增强的网格搜索文件夹结构
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_search_dir = os.path.join(results_dir, f"grid_search_{timestamp}")

    # 主结果文件路径
    csv_filepath = os.path.join(grid_search_dir, "grid_search_results.csv")
    details_filepath = os.path.join(grid_search_dir, "grid_search_details.csv")

    # 获取CSV字段名
    all_params = [params for params in combinations]
    fieldnames = get_csv_fieldnames(all_params)

    # 创建增强的结果管理器
    results_manager = ExperimentResultsManager(
        csv_filepath=csv_filepath,
        details_filepath=details_filepath,
        grid_search_dir=grid_search_dir
    )

    # 初始化CSV文件
    if args.save_results:
        os.makedirs(grid_search_dir, exist_ok=True)
        results_manager.initialize_csv_file(fieldnames)
    else:
        results_manager.initialize_csv_file(fieldnames)

    print(f"🚀 开始网格搜索，共 {len(combinations)} 个实验")
    print(f"📊 使用配置文件: {args.config}")
    print(f"📁 网格搜索目录: {grid_search_dir}")
    print(f"💾 主结果文件: {csv_filepath}")
    print(f"📋 详情表文件: {details_filepath}")

    # 处理data_percentage参数
    data_percentage = args.data_percentage if args.data_percentage is not None else 1.0

    if args.data_percentage is not None:
        print(f"🎯 全局参数覆盖: data_percentage={args.data_percentage}")
    else:
        print(f"🎯 使用默认data_percentage: {data_percentage}")

    print("=" * 60)

    # 串行执行所有实验
    results = []
    successful = 0

    for i, params in enumerate(combinations, 1):
        # 将命令行参数添加到实验参数中
        experiment_params = params.copy()
        experiment_params['hp.data_percentage'] = data_percentage
        experiment_params['data_percentage'] = data_percentage

        # 运行单个实验（统一版本 - 自动适配单卡/多卡）
        result = run_single_experiment(
            experiment_params,
            f"{i:03d}",
            args.config,
            grid_search_dir
        )

        results.append(result)
        if result["success"]:
            successful += 1

        # 实时写入CSV
        if args.save_results:
            print(f"💾 写入实验结果到CSV: {result.get('exp_name', 'unknown')}")
            results_manager.append_result_to_csv(result)

        # 实时显示最佳结果
        if successful > 0:
            current_best = max([r for r in results if r["success"]], key=lambda x: x["best_accuracy"])
            print(f"🏆 当前最佳: {current_best['exp_name']} - {current_best['best_accuracy']:.2f}%")

    # 总结
    print("=" * 60)
    print(f"📈 网格搜索完成！")
    print(f"✅ 成功实验数量: {successful}/{len(combinations)}")

    if successful > 0:
        successful_results = [r for r in results if r["success"]]
        best_result = max(successful_results, key=lambda x: x["best_accuracy"])

        print(f"🏆 最佳实验结果:")
        print(f"实验名称: {best_result['exp_name']}, 最佳准确率: {best_result['best_accuracy']:.2f}%, "
              f"最终准确率: {best_result['final_accuracy']:.2f}%")

        # 按最佳精度排序前n组结果
        top_results = sorted(successful_results, key=lambda x: x["best_accuracy"], reverse=True)[:args.top_n]

        print(f"📊 前{args.top_n}名实验结果:")
        for i, r in enumerate(top_results, 1):
            print(f"{i}. {r['exp_name']} - {r['best_accuracy']:.2f}% - {r['params']}")

    if args.save_results:
        print(f"💾 主结果已实时保存到: {csv_filepath}")
        print(f"📋 详情表已实时保存到: {details_filepath}")
        print(f"📁 单实验文件已保存到: {results_manager.experiments_dir}")

    return 0 if successful > 0 else 1


# ======================
# 主函数
# ======================

def main():
    """主函数

    启动方式：
    - 单卡模式：python scripts/grid_search_unified.py --config ...
    - 多卡模式：accelerate launch scripts/grid_search_unified.py --config ...

    Accelerator 会自动检测启动方式并适配单卡/多卡模式。
    """
    args, _ = parse_arguments(mode="grid_search")
    return run_grid_search(args)


if __name__ == "__main__":
    sys.exit(main())

