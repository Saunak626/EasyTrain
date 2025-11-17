"""实验结果管理器

负责管理网格搜索的结果文件，包括主结果CSV、详情CSV和单实验文件。
"""

import csv
import json
import fcntl
import os
from typing import Dict, List, Any

from .grid_search_generator import GROUP_KEY


# ======================
# CSV 配置常量
# ======================

# CSV 基础列（包含所有可能的指标列）
CSV_BASE_COLUMNS = [
    'exp_name', 'model.type', 'group', 'success', 'trained_epochs',
    # 🎯 多标签分类关键指标（优先显示）
    'best_weighted_f1', 'best_weighted_accuracy', 'best_macro_accuracy', 'best_micro_accuracy',
    'best_macro_f1', 'best_micro_f1', 'best_macro_precision', 'best_macro_recall',
    'final_weighted_f1', 'final_weighted_accuracy', 'final_macro_accuracy', 'final_micro_accuracy',
    'final_macro_f1', 'final_micro_f1',
    # 传统字段（向后兼容）
    'best_accuracy', 'final_accuracy'
]

# 常见运行时参数（会添加到 CSV 列中）
COMMON_RUNTIME_PARAMS = [
    'data_percentage',
    'optimizer.name', 'scheduler.name', 'loss.name'
]

# CSV 中排除的参数（不显示在 CSV 中）
EXCLUDED_CSV_PARAMS = ['epochs', 'batch_size', 'learning_rate']


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
        exp_name = result.get("exp_name", "unknown")
        exp_filepath = os.path.join(self.experiments_dir, f"{exp_name}.json")

        with open(exp_filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


# ======================
# CSV字段名生成函数
# ======================

def get_csv_fieldnames(all_params: List[Dict[str, Any]]) -> List[str]:
    """获取CSV文件的字段名列表

    Args:
        all_params: 所有参数组合列表

    Returns:
        CSV字段名列表
    """
    # 收集所有参数键
    param_keys = set()
    for params in all_params:
        param_keys.update(params.keys())

    # 排除不需要在CSV中显示的参数
    param_keys = {k for k in param_keys if k not in EXCLUDED_CSV_PARAMS}

    # 组合字段名：基础列 + 运行时参数 + 其他参数
    fieldnames = CSV_BASE_COLUMNS.copy()

    # 添加常见运行时参数
    for param in COMMON_RUNTIME_PARAMS:
        if param not in fieldnames:
            fieldnames.append(param)

    # 添加其他参数
    for key in sorted(param_keys):
        if key not in fieldnames and key != GROUP_KEY:
            fieldnames.append(key)

    return fieldnames

