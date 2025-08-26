"""网格搜索启动脚本

实现超参数网格搜索，支持进程内调用和多种参数组合策略。
主要功能：参数组合生成、实验执行、结果收集和CSV报告生成。
"""
import itertools
import subprocess
import yaml
import os
import sys
import csv
import json
import random
import fcntl
import hashlib
from typing import Dict, List, Any, Optional, Tuple

from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments

# ============================================================================
# 模块级常量配置
# ============================================================================

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

# ============================================================================
# 参数组合生成器类
# ============================================================================

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

            # 设置默认组名
            base_params[self.constants['group_key']] = 'default'

            # 合并固定参数
            result_params = {**fixed, **base_params}
            return [result_params] if result_params else [{}]

    def _generate_combinations_by_groups(self, groups_config: Dict[str, Any],
                                       fixed: Dict[str, Any],
                                       models_to_train: List[str]) -> List[Dict[str, Any]]:
        """分组式参数组合生成器 - 支持组内模型-batch_size智能配对

        Args:
            groups_config: 分组配置字典
            fixed: 固定参数字典
            models_to_train: 要训练的模型列表

        Returns:
            所有组合的参数列表
        """
        all_combinations = []
        total_groups = len(groups_config)

        print(f"🎯 发现 {total_groups} 个模型组:")
        for group_name in groups_config.keys():
            group_models = _as_list(groups_config[group_name].get(self.constants['model_type_key'], []))
            print(f"   - {group_name}: {group_models}")

        for group_name, group_params in groups_config.items():
            print(f"\n🔧 处理模型组: {group_name}")

            # 第1步：解析组配置
            group_models, group_batch_sizes = self._parse_group_config(group_params)

            # 第2步：处理模型-batch_size配对逻辑
            model_batch_dict = self._handle_model_batch_pairing(group_models, group_batch_sizes, group_name)

            # 第3步：过滤启用的模型
            enabled_models = self._filter_enabled_models(group_models, models_to_train, group_name)
            if not enabled_models:
                continue

            # 第4步：生成参数组合
            group_combinations = self._generate_parameter_combinations(
                enabled_models, group_params, fixed, group_name, model_batch_dict
            )
            all_combinations.extend(group_combinations)

            # 第5步：打印组合统计信息
            self._print_group_statistics(group_name, group_combinations, group_params,
                                       enabled_models, model_batch_dict)

        print(f"\n🎉 分组式搜索总计生成 {len(all_combinations)} 个组合")
        return all_combinations

    def _parse_group_config(self, group_params: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
        """解析组配置

        从组参数中提取模型列表和batch_size列表。

        Args:
            group_params: 组参数字典

        Returns:
            Tuple[模型列表, batch_size列表]
        """
        group_models = _as_list(group_params.get(self.constants['model_type_key'], []))
        group_batch_sizes = _as_list(group_params.get(self.constants['batch_size_key'], []))

        print(f"   📋 组内配置:")
        print(f"      {self.constants['model_type_key']}: {group_models} (长度: {len(group_models)})")
        print(f"      {self.constants['batch_size_key']}: {group_batch_sizes} (长度: {len(group_batch_sizes)})")

        return group_models, group_batch_sizes

    def _handle_model_batch_pairing(self, group_models: List[str], group_batch_sizes: List[Any],
                                   group_name: str) -> Optional[Dict[str, Any]]:
        """处理模型-batch_size配对逻辑

        根据模型和batch_size的数量关系，决定配对策略。

        Args:
            group_models: 模型列表
            group_batch_sizes: batch_size列表
            group_name: 组名称

        Returns:
            模型-batch_size配对字典，如果需要独立处理则返回None
        """
        if group_batch_sizes:
            if len(group_batch_sizes) == 1:
                # 情况1：batch_size长度=1，扩充到与model.type一致
                group_batch_sizes = group_batch_sizes * len(group_models)
                print(f"   🔄 扩充batch_size: {group_batch_sizes} (扩充到与model.type长度一致)")
                # 创建一对一配对字典
                model_batch_dict = dict(zip(group_models, group_batch_sizes))
            elif len(group_batch_sizes) == len(group_models):
                # 情况2：batch_size长度=model.type长度，按顺序配对
                print(f"   ✅ 长度匹配，将按顺序配对")
                model_batch_dict = dict(zip(group_models, group_batch_sizes))
            else:
                # 情况3：batch_size长度≠1且≠model.type长度，作为独立参数处理
                print(f"   🔄 batch_size作为独立参数处理，将与模型进行笛卡尔积组合")
                model_batch_dict = None  # 标记为独立参数处理
        else:
            # 没有batch_size配置，所有模型使用默认值
            model_batch_dict = {model: None for model in group_models}
            print(f"   📊 无batch_size配置，使用默认值")

        return model_batch_dict

    def _filter_enabled_models(self, group_models: List[str], models_to_train: List[str],
                              group_name: str) -> List[str]:
        """过滤启用的模型

        根据models_to_train配置过滤出需要训练的模型。

        Args:
            group_models: 组内所有模型
            models_to_train: 要训练的模型列表
            group_name: 组名称

        Returns:
            启用的模型列表
        """
        if models_to_train:
            enabled_models = [model for model in group_models if model in models_to_train]
            if not enabled_models:
                print(f"   ⏭️  跳过组 {group_name}：无启用的模型")
                return []
        else:
            enabled_models = group_models

        return enabled_models

    def _generate_parameter_combinations(self, enabled_models: List[str], group_params: Dict[str, Any],
                                       fixed: Dict[str, Any], group_name: str,
                                       model_batch_dict: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成参数组合

        根据模型-batch_size配对策略生成所有参数组合。

        Args:
            enabled_models: 启用的模型列表
            group_params: 组参数字典
            fixed: 固定参数字典
            group_name: 组名称
            model_batch_dict: 模型-batch_size配对字典，None表示独立处理

        Returns:
            参数组合列表
        """
        combinations = []

        if model_batch_dict is not None:
            # 有模型-batch_size配对的情况
            enabled_pairs = {model: batch_size for model, batch_size in model_batch_dict.items()
                            if model in enabled_models}
            print(f"   🎯 启用的模型配对: {enabled_pairs}")

            # 获取其他参数（排除model.type和hp.batch_size）
            other_params = {k: v for k, v in group_params.items()
                           if k not in self.constants['excluded_params']}

            # 生成组合
            if not other_params:
                # 只有模型-batch_size配对，无其他参数
                for model, batch_size in enabled_pairs.items():
                    combo = {**fixed, self.constants['model_type_key']: model, self.constants['group_key']: group_name}
                    if batch_size is not None:
                        combo[self.constants['batch_size_key']] = batch_size
                    combinations.append(combo)
            else:
                # 有其他参数，进行笛卡尔积组合
                param_items = [(k, _as_list(v)) for k, v in other_params.items() if _as_list(v)]
                if param_items:
                    param_keys, param_values_lists = zip(*param_items)
                    for model, batch_size in enabled_pairs.items():
                        for param_combo in itertools.product(*param_values_lists):
                            combo = {
                                **fixed,
                                self.constants['model_type_key']: model,
                                self.constants['group_key']: group_name
                            }
                            if batch_size is not None:
                                combo[self.constants['batch_size_key']] = batch_size
                            combo.update(dict(zip(param_keys, param_combo)))
                            combinations.append(combo)
                else:
                    # 其他参数都为空
                    for model, batch_size in enabled_pairs.items():
                        combo = {**fixed, self.constants['model_type_key']: model, self.constants['group_key']: group_name}
                        if batch_size is not None:
                            combo[self.constants['batch_size_key']] = batch_size
                        combinations.append(combo)
        else:
            # batch_size作为独立参数，与模型进行笛卡尔积组合
            all_params = {k: v for k, v in group_params.items() if k != self.constants['model_type_key']}
            param_items = [(k, _as_list(v)) for k, v in all_params.items() if _as_list(v)]

            if param_items:
                param_keys, param_values_lists = zip(*param_items)
                for model in enabled_models:
                    for param_combo in itertools.product(*param_values_lists):
                        combo = {
                            **fixed,
                            self.constants['model_type_key']: model,
                            self.constants['group_key']: group_name
                        }
                        combo.update(dict(zip(param_keys, param_combo)))
                        combinations.append(combo)
            else:
                # 无其他参数
                for model in enabled_models:
                    combo = {**fixed, self.constants['model_type_key']: model, self.constants['group_key']: group_name}
                    combinations.append(combo)

        return combinations

    def _print_group_statistics(self, group_name: str, group_combinations: List[Dict[str, Any]],
                               group_params: Dict[str, Any], enabled_models: List[str],
                               model_batch_dict: Optional[Dict[str, Any]]) -> None:
        """打印组合统计信息

        Args:
            group_name: 组名称
            group_combinations: 组合列表
            group_params: 组参数字典
            enabled_models: 启用的模型列表
            model_batch_dict: 模型-batch_size配对字典
        """
        group_count = len(group_combinations)

        # 计算组合数量的分解
        if model_batch_dict is not None:
            # 有模型-batch_size配对的情况
            enabled_pairs = {model: batch_size for model, batch_size in model_batch_dict.items()
                            if model in enabled_models}
            model_count = len(enabled_pairs)
            other_params = {k: v for k, v in group_params.items()
                           if k not in self.constants['excluded_params']}
            param_items = [(k, _as_list(v)) for k, v in other_params.items() if _as_list(v)]
            if param_items:
                param_counts = [len(_as_list(v)) for _, v in other_params.items() if _as_list(v)]
                other_count = 1
                for count in param_counts:
                    other_count *= count
                print(f"   ✅ 组 {group_name} 生成 {group_count} 个组合 ({model_count}模型 × {other_count}参数组合)")
            else:
                print(f"   ✅ 组 {group_name} 生成 {group_count} 个组合 ({model_count}模型)")
        else:
            # batch_size作为独立参数的情况
            model_count = len(enabled_models)
            all_params = {k: v for k, v in group_params.items() if k != self.constants['model_type_key']}
            param_items = [(k, _as_list(v)) for k, v in all_params.items() if _as_list(v)]
            if param_items:
                param_counts = [len(_as_list(v)) for _, v in all_params.items() if _as_list(v)]
                other_count = 1
                for count in param_counts:
                    other_count *= count
                print(f"   ✅ 组 {group_name} 生成 {group_count} 个组合 ({model_count}模型 × {other_count}参数组合)")
            else:
                print(f"   ✅ 组 {group_name} 生成 {group_count} 个组合 ({model_count}模型)")


# ============================================================================
# 实验结果管理器类
# ============================================================================

class ExperimentResultsManager:
    """实验结果管理器

    负责管理CSV文件的创建、写入和字段名生成等操作。
    """

    def __init__(self, csv_filepath: str, details_filepath: str = None, grid_search_dir: str = None):
        """初始化增强的实验结果管理器

        Args:
            csv_filepath: 主结果CSV文件路径
            details_filepath: 详情CSV文件路径（可选）
            grid_search_dir: 网格搜索根目录路径（可选）
        """
        self.csv_filepath = csv_filepath
        self.details_filepath = details_filepath
        self.grid_search_dir = grid_search_dir
        self.fieldnames = None
        self.details_fieldnames = None
        self.constants = GRID_SEARCH_CONSTANTS

        # 创建增强的文件夹结构
        if self.grid_search_dir:
            self.experiments_dir = os.path.join(self.grid_search_dir, "experiments")
            os.makedirs(self.experiments_dir, exist_ok=True)
            print(f"📁 创建实验文件夹结构: {self.experiments_dir}")

    def get_csv_fieldnames(self, all_params: List[Dict[str, Any]]) -> List[str]:
        """获取CSV文件的字段名列表

        Args:
            all_params: 所有参数组合列表

        Returns:
            CSV字段名列表
        """
        param_keys = sorted({k for params in all_params for k in params.keys()})

        # 合并所有参数键，去重并排序，排除冗余参数
        all_param_keys = sorted(set(param_keys + self.constants['common_runtime_params']) - set(self.constants['excluded_csv_params']))

        # 将model.type移到第3列，group移到第4列，其他参数按原顺序排列
        other_param_keys = [k for k in all_param_keys if k not in [self.constants['model_type_key'], self.constants['group_key']]]

        fieldnames = self.constants['csv_base_columns'] + other_param_keys
        self.fieldnames = fieldnames
        return fieldnames

    def initialize_csv_file(self, fieldnames: List[str]) -> None:
        """初始化CSV文件，写入表头

        Args:
            fieldnames: CSV字段名列表
        """
        results_dir = os.path.dirname(self.csv_filepath)
        os.makedirs(results_dir, exist_ok=True)

        with open(self.csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        self.fieldnames = fieldnames

        # 如果指定了详情文件，也初始化详情表
        if self.details_filepath:
            self.initialize_details_csv()

    def initialize_details_csv(self) -> None:
        """初始化详情CSV文件"""
        if not self.details_filepath:
            return

        # 增强的详情表字段名
        self.details_fieldnames = [
            'exp_name', 'config_hash', 'epoch', '类别名称', '精确率', '召回率',
            'F1分数', '准确率', '正样本', '负样本', 'gamma', 'alpha', 'pos_weight',
            'learning_rate', 'loss_name', 'model_type', 'batch_size'
        ]

        details_dir = os.path.dirname(self.details_filepath)
        os.makedirs(details_dir, exist_ok=True)

        with open(self.details_filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.details_fieldnames)
            writer.writeheader()

        print(f"📋 初始化详情表: {self.details_filepath}")

    def append_result_to_csv(self, result: Dict[str, Any]) -> None:
        """实时追加单个结果到CSV文件（线程安全）

        Args:
            result: 实验结果
        """
        if not self.fieldnames:
            raise ValueError("CSV字段名未初始化，请先调用initialize_csv_file")

        try:
            # 准备行数据
            row = {
                "exp_name": result.get("exp_name"),
                "success": result.get("success"),
                "trained_epochs": result.get("trained_epochs", 0),
            }

            # 添加多标签分类指标
            multilabel_metrics = result.get("multilabel_metrics", {})
            if multilabel_metrics:
                # 最佳指标
                best_metrics = multilabel_metrics.get("best", {})
                row.update({
                    "best_macro_accuracy": best_metrics.get("macro_accuracy"),
                    "best_micro_accuracy": best_metrics.get("micro_accuracy"),
                    "best_weighted_accuracy": best_metrics.get("weighted_accuracy"),
                    "best_macro_f1": best_metrics.get("macro_f1"),
                    "best_micro_f1": best_metrics.get("micro_f1"),
                    "best_weighted_f1": best_metrics.get("weighted_f1"),
                    "best_macro_precision": best_metrics.get("macro_precision"),
                    "best_macro_recall": best_metrics.get("macro_recall"),
                })

                # 最终指标
                final_metrics = multilabel_metrics.get("final", {})
                row.update({
                    "final_macro_accuracy": final_metrics.get("macro_accuracy"),
                    "final_micro_accuracy": final_metrics.get("micro_accuracy"),
                    "final_weighted_accuracy": final_metrics.get("weighted_accuracy"),
                    "final_macro_f1": final_metrics.get("macro_f1"),
                    "final_micro_f1": final_metrics.get("micro_f1"),
                    "final_weighted_f1": final_metrics.get("weighted_f1"),
                })

            # 传统字段（向后兼容）
            row.update({
                "best_accuracy": result.get("best_accuracy"),
                "final_accuracy": result.get("final_accuracy"),
            })

            row.update(result.get("params", {}))

            # 只写入fieldnames中存在的字段，忽略额外字段
            filtered_row = {k: v for k, v in row.items() if k in self.fieldnames}

            # 检查是否有缺失的必需字段
            missing_fields = [k for k in self.fieldnames if k not in row]
            if missing_fields:
                print(f"⚠️  缺失字段: {missing_fields}，将使用空值填充")
                for field in missing_fields:
                    filtered_row[field] = ""

            # 使用文件锁确保线程安全
            with open(self.csv_filepath, "a", newline="", encoding="utf-8") as csvfile:
                # 获取文件锁
                fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)

                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow(filtered_row)
                csvfile.flush()  # 强制刷新缓冲区

                # 释放文件锁
                fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)

            print(f"✅ CSV写入成功: {result.get('exp_name', 'unknown')}")

            # 如果有额外字段，给出提示
            extra_fields = [k for k in row.keys() if k not in self.fieldnames]
            if extra_fields:
                print(f"ℹ️  忽略额外字段: {extra_fields}")

            # 增强功能：保存详情表和单实验文件
            if result.get('success', False):
                self._save_enhanced_experiment_data(result)

        except Exception as e:
            print(f"⚠️  写入CSV失败: {e}")
            print(f"   文件路径: {self.csv_filepath}")
            print(f"   当前字段名: {self.fieldnames}")
            print(f"   行数据键: {list(row.keys()) if 'row' in locals() else 'N/A'}")
            print(f"   结果数据: {result}")

    def _save_enhanced_experiment_data(self, result: Dict[str, Any]) -> None:
        """保存增强的实验数据（详情表 + 单实验文件）

        Args:
            result: 实验结果数据
        """
        exp_name = result.get('exp_name', 'unknown')

        try:
            # 1. 保存到详情表
            if self.details_filepath and self.details_fieldnames:
                self._append_to_details_csv(result)

            # 2. 保存单实验文件
            if self.experiments_dir:
                self._save_individual_experiment_files(result)

        except Exception as e:
            print(f"⚠️ 保存增强实验数据失败 ({exp_name}): {e}")

    def _append_to_details_csv(self, result: Dict[str, Any]) -> None:
        """追加详细指标到详情CSV文件

        Args:
            result: 包含detailed_metrics的实验结果
        """
        try:
            # 获取详细指标
            detailed_metrics = result.get('detailed_metrics', {})
            if not detailed_metrics:
                print(f"⚠️ 实验 {result.get('exp_name')} 缺少详细指标数据")
                return

            exp_name = result.get('exp_name', '')
            params = result.get('params', {})

            # 生成配置哈希
            config_hash = self._generate_config_hash(params)

            # 获取训练参数
            gamma = params.get('loss.params.gamma', params.get('gamma', ''))
            alpha = params.get('loss.params.alpha', params.get('alpha', ''))
            pos_weight = params.get('loss.params.pos_weight', params.get('pos_weight', ''))
            learning_rate = params.get('hp.learning_rate', params.get('learning_rate', ''))
            loss_name = params.get('loss.name', '')
            model_type = params.get('model.type', '')
            batch_size = params.get('hp.batch_size', params.get('batch_size', ''))

            # 获取最佳epoch
            best_epoch = detailed_metrics.get('epoch', result.get('trained_epochs', 0))

            # 准备详情表数据行
            rows = []

            # 1. 添加各个类别的指标
            class_metrics = detailed_metrics.get('class_metrics', {})
            for class_name, metrics in class_metrics.items():
                row = {
                    'exp_name': exp_name,
                    'config_hash': config_hash,
                    'epoch': best_epoch,
                    '类别名称': class_name,
                    '精确率': round(metrics.get('precision', 0), 4),
                    '召回率': round(metrics.get('recall', 0), 4),
                    'F1分数': round(metrics.get('f1', 0), 4),
                    '准确率': round(metrics.get('accuracy', 0), 4),
                    '正样本': metrics.get('pos_samples', 0),
                    '负样本': metrics.get('neg_samples', 0),
                    'gamma': gamma,
                    'alpha': alpha,
                    'pos_weight': pos_weight,
                    'learning_rate': learning_rate,
                    'loss_name': loss_name,
                    'model_type': model_type,
                    'batch_size': batch_size
                }
                rows.append(row)

            # 2. 添加平均指标（作为特殊类别）
            avg_metrics = [
                ('🎯加权平均', detailed_metrics.get('weighted_avg', {})),
                ('📊宏平均', detailed_metrics.get('macro_avg', {})),
                ('📈微平均', detailed_metrics.get('micro_avg', {}))
            ]

            for avg_name, avg_data in avg_metrics:
                if avg_data:
                    row = {
                        'exp_name': exp_name,
                        'config_hash': config_hash,
                        'epoch': best_epoch,
                        '类别名称': avg_name,
                        '精确率': round(avg_data.get('precision', 0), 4),
                        '召回率': round(avg_data.get('recall', 0), 4),
                        'F1分数': round(avg_data.get('f1', 0), 4),
                        '准确率': round(avg_data.get('accuracy', 0), 4),
                        '正样本': '',  # 平均指标不显示样本数
                        '负样本': '',
                        'gamma': gamma,
                        'alpha': alpha,
                        'pos_weight': pos_weight,
                        'learning_rate': learning_rate,
                        'loss_name': loss_name,
                        'model_type': model_type,
                        'batch_size': batch_size
                    }
                    rows.append(row)

            # 批量写入详情表
            if rows:
                with open(self.details_filepath, "a", newline="", encoding="utf-8") as csvfile:
                    fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
                    writer = csv.DictWriter(csvfile, fieldnames=self.details_fieldnames)
                    writer.writerows(rows)
                    csvfile.flush()
                    fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)

                print(f"📊 已保存 {len(rows)} 条详细指标到详情表 ({exp_name})")

        except Exception as e:
            print(f"⚠️ 写入详情表失败: {e}")

    def _save_individual_experiment_files(self, result: Dict[str, Any]) -> None:
        """保存单个实验的文件

        Args:
            result: 实验结果数据
        """
        exp_name = result.get('exp_name', 'unknown')

        try:
            # 创建单实验文件夹
            exp_dir = os.path.join(self.experiments_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            # 1. 保存实验配置
            config_file = os.path.join(exp_dir, "config.yaml")
            config_data = {
                'exp_name': exp_name,
                'parameters': result.get('params', {}),
                'success': result.get('success', False),
                'trained_epochs': result.get('trained_epochs', 0),
                'timestamp': datetime.now().isoformat()
            }

            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

            # 2. 复制训练过程中生成的逐epoch指标文件
            self._copy_epoch_metrics_files(exp_dir, result)

            # 3. 保存类别指标历史（如果有详细指标）
            detailed_metrics = result.get('detailed_metrics', {})
            if detailed_metrics and 'class_metrics' in detailed_metrics:
                self._save_class_metrics_history(exp_dir, detailed_metrics)

            # 4. 保存最佳指标汇总
            if detailed_metrics:
                self._save_best_metrics_summary(exp_dir, detailed_metrics)

            print(f"📁 已保存单实验文件: {exp_dir}")

        except Exception as e:
            print(f"⚠️ 保存单实验文件失败 ({exp_name}): {e}")

    def _save_class_metrics_history(self, exp_dir: str, detailed_metrics: Dict[str, Any]) -> None:
        """保存类别指标历史文件（现在只保存最佳epoch的指标，与best_metrics_summary.csv功能类似）

        注意：此方法现在主要用于向后兼容，实际的逐epoch指标记录由训练器中的
        train_metrics_history.csv和test_metrics_history.csv文件处理
        """
        import pandas as pd

        class_metrics = detailed_metrics.get('class_metrics', {})
        epoch = detailed_metrics.get('epoch', 0)

        rows = []
        for class_name, metrics in class_metrics.items():
            row = {
                'epoch': epoch,
                'class_name': class_name,
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'accuracy': metrics.get('accuracy', 0),
                'pos_samples': metrics.get('pos_samples', 0),
                'neg_samples': metrics.get('neg_samples', 0)
            }
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            csv_file = os.path.join(exp_dir, "class_metrics_history.csv")
            df.to_csv(csv_file, index=False, encoding='utf-8')

    def _copy_epoch_metrics_files(self, exp_dir: str, result: Dict[str, Any]) -> None:
        """复制训练过程中生成的逐epoch指标文件到单实验文件夹

        Args:
            exp_dir: 单实验文件夹路径
            result: 实验结果数据
        """
        import shutil

        # 获取原始指标文件的路径（从训练器的输出目录）
        detailed_metrics = result.get('detailed_metrics', {})
        if not detailed_metrics:
            return

        # 尝试从config中获取任务输出目录
        config = result.get('config', {})
        task_config = config.get('task', {})
        task_tag = task_config.get('tag', '')
        dataset_type = config.get('data', {}).get('type', '')

        # 构建原始输出目录路径
        if 'multilabel' in dataset_type.lower() or 'multilabel' in task_tag.lower():
            from src.trainers.base_trainer import get_task_output_dir
            source_dir = get_task_output_dir(task_tag, dataset_type)

            # 需要复制的文件列表
            files_to_copy = [
                'train_metrics_history.csv',  # 训练集逐epoch指标
                'test_metrics_history.csv',   # 测试集逐epoch指标
                'class_metrics_history.csv'   # 原有的类别指标历史（现在记录每个epoch）
            ]

            for filename in files_to_copy:
                source_file = os.path.join(source_dir, filename)
                target_file = os.path.join(exp_dir, filename)

                if os.path.exists(source_file):
                    try:
                        shutil.copy2(source_file, target_file)
                        print(f"📋 已复制指标文件: {filename}")
                    except Exception as e:
                        print(f"⚠️ 复制指标文件失败 ({filename}): {e}")
                else:
                    print(f"⚠️ 指标文件不存在: {source_file}")

    def _save_best_metrics_summary(self, exp_dir: str, detailed_metrics: Dict[str, Any]) -> None:
        """保存最佳指标汇总文件"""
        import pandas as pd

        # 准备汇总数据
        summary_data = []

        # 添加各类别指标
        class_metrics = detailed_metrics.get('class_metrics', {})
        for class_name, metrics in class_metrics.items():
            summary_data.append({
                '类别名称': class_name,
                '精确率': f"{metrics.get('precision', 0):.4f}",
                '召回率': f"{metrics.get('recall', 0):.4f}",
                'F1分数': f"{metrics.get('f1', 0):.4f}",
                '准确率': f"{metrics.get('accuracy', 0):.4f}",
                '正样本数': metrics.get('pos_samples', 0),
                '负样本数': metrics.get('neg_samples', 0)
            })

        # 添加平均指标
        avg_metrics = [
            ('🎯加权平均', detailed_metrics.get('weighted_avg', {})),
            ('📊宏平均', detailed_metrics.get('macro_avg', {})),
            ('📈微平均', detailed_metrics.get('micro_avg', {}))
        ]

        for avg_name, avg_data in avg_metrics:
            if avg_data:
                summary_data.append({
                    '类别名称': avg_name,
                    '精确率': f"{avg_data.get('precision', 0):.4f}",
                    '召回率': f"{avg_data.get('recall', 0):.4f}",
                    'F1分数': f"{avg_data.get('f1', 0):.4f}",
                    '准确率': f"{avg_data.get('accuracy', 0):.4f}",
                    '正样本数': '',
                    '负样本数': ''
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = os.path.join(exp_dir, "best_metrics_summary.csv")
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    def _generate_config_hash(self, params: Dict[str, Any]) -> str:
        """生成参数配置的哈希值"""
        # 提取关键参数用于生成哈希
        key_params = {
            'model_type': params.get('model.type', ''),
            'loss_name': params.get('loss.name', ''),
            'gamma': params.get('loss.params.gamma', params.get('gamma', '')),
            'alpha': params.get('loss.params.alpha', params.get('alpha', '')),
            'pos_weight': params.get('loss.params.pos_weight', params.get('pos_weight', '')),
            'learning_rate': params.get('hp.learning_rate', params.get('learning_rate', '')),
            'batch_size': params.get('hp.batch_size', params.get('batch_size', ''))
        }

        # 生成哈希
        config_str = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


def load_grid_config(path: str = "config/grid.yaml") -> Dict[str, Any]:
    """加载网格搜索配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_list(v: Any) -> List[Any]:
    """将输入转换为列表格式

    Args:
        v: 任意类型的参数值

    Returns:
        统一格式化后的列表
    """
    if v is None:
        return []
    return v if isinstance(v, (list, tuple)) else [v]

def generate_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    分组式参数组合生成器

    设计逻辑：
    1. 从YAML中获取groups配置，每组有自己的模型和超参数范围
    2. 为每组内的参数进行笛卡尔积组合
    3. 根据models_to_train过滤启用的模型
    4. 避免无意义的模型-参数组合，节省算力

    Args:
        config: 网格搜索配置

    Returns:
        参数组合列表，每个字典代表一组实验参数
    """
    generator = ParameterCombinationGenerator(config)
    return generator.generate_combinations()



# ============================================================================
# 向后兼容的函数接口
# ============================================================================

def get_csv_fieldnames(all_params: List[Dict[str, Any]]) -> List[str]:
    """获取CSV文件的字段名列表（向后兼容接口）"""
    # 创建临时的结果管理器来生成字段名
    temp_manager = ExperimentResultsManager("")
    return temp_manager.get_csv_fieldnames(all_params)


def initialize_csv_file(filepath: str, fieldnames: List[str]) -> None:
    """初始化CSV文件，写入表头（向后兼容接口）"""
    manager = ExperimentResultsManager(filepath)
    manager.initialize_csv_file(fieldnames)


def append_result_to_csv(result: Dict[str, Any], filepath: str, fieldnames: List[str], experiment_id: int = None) -> None:
    """实时追加单个结果到CSV文件（向后兼容接口）"""
    manager = ExperimentResultsManager(filepath)
    manager.fieldnames = fieldnames  # 设置字段名
    manager.append_result_to_csv(result)





def save_results_to_csv(results: List[Dict[str, Any]], filename: str) -> Optional[str]:
    """保存实验结果到CSV文件（兼容旧接口）

    Args:
        results: 实验结果列表
        filename: CSV文件名

    Returns:
        保存的文件路径，如果无结果则返回None
    """
    if not results:
        print("⚠️  无结果数据，跳过CSV保存")
        return None

    # 确保目录存在
    os.makedirs("runs", exist_ok=True)
    filepath = f"runs/{filename}"

    # 创建结果管理器
    manager = ExperimentResultsManager(filepath)

    # 获取字段名并初始化CSV文件
    fieldnames = manager.get_csv_fieldnames([r.get("params", {}) for r in results])
    manager.initialize_csv_file(fieldnames)

    # 写入所有结果
    for result in results:
        manager.append_result_to_csv(result)

    print(f"📊 实验结果已保存到: {filepath}")
    print(f"   总实验数: {len(results)}")
    print(f"   成功实验: {sum(1 for r in results if r.get('success', False))}")
    print(f"   失败实验: {sum(1 for r in results if not r.get('success', False))}")

    return filepath


def apply_param_overrides(config, params):
    """应用参数覆盖到配置字典

    Args:
        config (dict): 基础配置字典
        params (dict): 参数覆盖字典，支持嵌套路径

    Returns:
        dict: 应用覆盖后的配置字典
    """
    import copy
    config = copy.deepcopy(config)
    
    for k, v in (params or {}).items():
        keys = k.split('.')
        target = config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = v
    
    return config


def run_single_experiment_in_process(params, exp_id, config_path):
    """进程内调用方式运行单个实验（单卡训练）"""
    exp_name = f"grid_{exp_id}"
    
    try:
        # 导入训练函数
        from src.trainers.base_trainer import run_training
        
        # 加载基础配置
        config = load_grid_config(config_path)
        
        # 应用参数覆盖
        config = apply_param_overrides(config, params)
        
        # 直接调用训练函数
        result = run_training(config, exp_name)
        
        # 添加参数信息到结果中
        result["params"] = params
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "exp_name": exp_name,
            "params": params,
            "best_accuracy": 0.0,
            "final_accuracy": 0.0,
            "trained_epochs": 0,
            "error": str(e)
        }


def run_single_experiment_subprocess(params, exp_id, use_multi_gpu, config_path):
    """子进程方式运行单个实验（多卡训练）"""
    exp_name = f"grid_{exp_id}"
    
    # 创建临时结果文件用于进程间通信
    temp_result_file = f"/tmp/grid_result_{exp_id}_{random.randint(1000,9999)}.json"
    
    # 组装命令
    if use_multi_gpu:
        import torch  # 局部导入，仅在需要时使用
        cmd = ["accelerate", "launch", "--multi_gpu", "--num_processes", str(torch.cuda.device_count())]
    else:
        cmd = [sys.executable, "-u"]
    
    # 添加训练脚本和基础参数
    cmd.extend(["scripts/train.py", "--config", config_path, "--exp_name", exp_name])
    cmd.extend(["--result_file", temp_result_file])  # 新增：指定结果文件
    
    # 添加参数覆盖（排除group参数，它只用于记录）
    for k, v in (params or {}).items():
        if k != "group":  # group参数不传递给训练脚本
            cmd.extend([f"--{k}", str(v)])

    # 清理环境变量并设置唯一端口
    env = os.environ.copy()
    for k in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(k, None)
    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(20000 + random.randint(0, 10000))

    # 启动子进程
    process = subprocess.Popen(cmd, env=env)
    try:
        rc = process.wait()
    except KeyboardInterrupt:
        print(f"捕获到中断信号，正在终止子进程 {process.pid}...")
        process.terminate()
        process.wait()
        raise

    success = (rc == 0)
    
    # 读取结果文件
    try:
        if os.path.exists(temp_result_file):
            with open(temp_result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            os.remove(temp_result_file)  # 清理临时文件
            
            # 确保结果包含必要的字段
            result["params"] = params
            result["success"] = result.get("success", success)
            result["exp_name"] = result.get("exp_name", exp_name)
            
            # 确保accuracy字段不为None
            if result.get("best_accuracy") is None:
                result["best_accuracy"] = 0.0
            if result.get("final_accuracy") is None:
                result["final_accuracy"] = 0.0
                
            return result
        else:
            print(f"结果文件不存在: {temp_result_file}")
    except Exception as e:
        print(f"读取结果文件失败: {e}")
        if os.path.exists(temp_result_file):
            try:
                with open(temp_result_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"文件内容: {content[:200]}...")  # 显示前200个字符
            except:
                pass
    
    # 回退：返回默认结果
    return {
        "success": success,
        "exp_name": exp_name,
        "params": params,
        "best_accuracy": 0.0,
        "final_accuracy": 0.0,
        "trained_epochs": 0,
        "error": "Failed to read result file" if success else "Training process failed"
    }


def run_single_experiment(params, exp_id, use_multi_gpu=False, config_path="config/grid.yaml"):
    """运行单个实验

    Args:
        params (dict): 实验参数覆盖
        exp_id (str): 实验ID
        use_multi_gpu (bool): 是否使用多GPU
        config_path (str): 配置文件路径

    Returns:
        dict: 实验结果字典
    """
    exp_name = f"grid_{exp_id}"

    # 实验信息将在训练器中的SwanLab启动后显示

    if use_multi_gpu:
        # 多卡训练：使用子进程方式
        result = run_single_experiment_subprocess(params, exp_id, use_multi_gpu, config_path)
    else:
        # 单卡训练：使用进程内调用方式
        result = run_single_experiment_in_process(params, exp_id, config_path)
    
    print(f"✅ 实验 {exp_name} 完成，最佳: {result['best_accuracy']:.2f}% | 最终: {result['final_accuracy']:.2f}%")
    
    return result


def run_grid_search(args):
    """运行网格搜索"""
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
        # 默认使用数据集类型作为子目录名
        task_subdir = dataset_type.replace('_', '_').lower() or "general"

    results_dir = os.path.join("runs", task_subdir)

    if args.results_file:
        # 使用命令行指定的文件名
        results_filename = args.results_file
    else:
        # 使用默认的时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"grid_search_results_{timestamp}.csv"
    csv_filepath = os.path.join(results_dir, results_filename)

    # 创建增强的网格搜索文件夹结构
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_search_dir = os.path.join(results_dir, f"grid_search_{timestamp}")

    # 移动主结果文件到网格搜索目录
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
        # 不保存结果时也需要初始化
        results_manager.initialize_csv_file(fieldnames)

    print(f"🚀 开始网格搜索，共 {len(combinations)} 个实验")
    print(f"📊 使用配置文件: {args.config}")
    print(f"📁 网格搜索目录: {grid_search_dir}")
    print(f"💾 主结果文件: {csv_filepath}")
    print(f"📋 详情表文件: {details_filepath}")
    
    # 处理data_percentage参数：如果未指定则使用默认值1.0
    data_percentage = args.data_percentage if args.data_percentage is not None else 1.0

    # 显示全局参数覆盖
    if args.data_percentage is not None:
        print(f"🎯 全局参数覆盖: data_percentage={args.data_percentage}")
    else:
        print(f"🎯 使用默认data_percentage: {data_percentage}")

    print("=" * 60)

    results = []
    successful = 0

    for i, params in enumerate(combinations, 1):
        exp_name = f"grid_{i:03d}"

        print(f"📊 准备实验 {i}/{len(combinations)}")

        # 将命令行参数添加到实验参数中
        experiment_params = params.copy()
        # 始终添加data_percentage参数，确保CSV记录完整
        experiment_params['data_percentage'] = data_percentage

        result = run_single_experiment(
            experiment_params, f"{i:03d}",
            use_multi_gpu=args.multi_gpu,
            config_path=args.config,
        )

        results.append(result)
        if result["success"]:
            successful += 1
            
        # 实时写入CSV（包括增强功能）
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
        # 找到“最佳准确率”最高的实验结果
        best_result = max(successful_results, key=lambda x: x["best_accuracy"])

        print(f"🏆 最佳实验结果:")
        print(f"实验名称: {best_result['exp_name']}, 最佳准确率: {best_result['best_accuracy']:.2f}%, 最终准确率: {best_result['final_accuracy']:.2f}%")
        
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


def main():
    """主函数：调度器始终单进程，不进入 Accelerate 环境"""
    args, _ = parse_arguments(mode="grid_search")
    return run_grid_search(args)


if __name__ == "__main__":
    sys.exit(main())