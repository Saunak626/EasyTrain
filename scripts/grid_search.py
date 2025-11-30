"""网格搜索启动脚本

实现超参数网格搜索，支持进程内调用和多种参数组合策略。
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_parser import parse_arguments

# 网格搜索常量
GRID_SEARCH_CONSTANTS = {
    'model_type_key': 'model.type',
    'batch_size_key': 'hp.batch_size',
    'group_key': 'group',
    'excluded_params': ['model.type', 'hp.batch_size'],
    'csv_base_columns': [
        'exp_name', 'model.type', 'group', 'success', 'trained_epochs',
        'best_weighted_f1', 'best_weighted_accuracy', 'best_macro_accuracy', 'best_micro_accuracy',
        'best_macro_f1', 'best_micro_f1', 'best_macro_precision', 'best_macro_recall',
        'final_weighted_f1', 'final_weighted_accuracy', 'final_macro_accuracy', 'final_micro_accuracy',
        'final_macro_f1', 'final_micro_f1', 'best_accuracy', 'final_accuracy'
    ],
    'common_runtime_params': ['data_percentage', 'optimizer.name', 'scheduler.name', 'loss.name'],
    'excluded_csv_params': ['epochs', 'batch_size', 'learning_rate']
}


def _as_list(v: Any) -> List[Any]:
    """将输入转换为列表格式"""
    if v is None:
        return []
    return v if isinstance(v, (list, tuple)) else [v]


def load_grid_config(path: str = "config/grid.yaml") -> Dict[str, Any]:
    """加载网格搜索配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ParameterCombinationGenerator:
    """参数组合生成器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.constants = GRID_SEARCH_CONSTANTS

    def generate_combinations(self) -> List[Dict[str, Any]]:
        """生成参数组合"""
        gs = (self.config or {}).get("grid_search", {}) or {}
        fixed = gs.get("fixed", {}) or {}
        models_to_train = self.config.get("models_to_train", [])

        if "groups" in gs and gs["groups"]:
            return self._generate_combinations_by_groups(gs["groups"], fixed, models_to_train)
        else:
            # 无groups配置，从基础配置提取
            base_params = {}
            if 'model' in self.config and 'type' in self.config['model']:
                base_params[self.constants['model_type_key']] = self.config['model']['type']
            if 'optimizer' in self.config and 'name' in self.config['optimizer']:
                base_params['optimizer.name'] = self.config['optimizer']['name']
            if 'scheduler' in self.config and 'name' in self.config['scheduler']:
                base_params['scheduler.name'] = self.config['scheduler']['name']
            if 'loss' in self.config and 'name' in self.config['loss']:
                base_params['loss.name'] = self.config['loss']['name']
            base_params[self.constants['group_key']] = 'default'
            result_params = {**fixed, **base_params}
            return [result_params] if result_params else [{}]

    def _generate_combinations_by_groups(self, groups_config: Dict[str, Any],
                                        fixed: Dict[str, Any],
                                        models_to_train: List[str]) -> List[Dict[str, Any]]:
        """分组式参数组合生成"""
        all_combinations = []

        for group_name, group_params in groups_config.items():
            group_models = _as_list(group_params.get(self.constants['model_type_key'], []))
            group_batch_sizes = _as_list(group_params.get(self.constants['batch_size_key'], []))

            # 处理模型-batch_size配对
            model_batch_dict = self._handle_model_batch_pairing(group_models, group_batch_sizes)

            # 过滤启用的模型
            if models_to_train:
                enabled_models = [m for m in group_models if m in models_to_train]
                if not enabled_models:
                    continue
            else:
                enabled_models = group_models

            # 生成组合
            group_combinations = self._generate_parameter_combinations(
                enabled_models, group_params, fixed, group_name, model_batch_dict
            )
            all_combinations.extend(group_combinations)

        print(f"总计生成 {len(all_combinations)} 个参数组合")
        return all_combinations

    def _handle_model_batch_pairing(self, group_models: List[str], 
                                    group_batch_sizes: List[Any]) -> Optional[Dict[str, Any]]:
        """处理模型-batch_size配对"""
        if not group_batch_sizes:
            return {model: None for model in group_models}
        
        if len(group_batch_sizes) == 1:
            group_batch_sizes = group_batch_sizes * len(group_models)
            return dict(zip(group_models, group_batch_sizes))
        elif len(group_batch_sizes) == len(group_models):
            return dict(zip(group_models, group_batch_sizes))
        else:
            return None  # 独立参数处理

    def _generate_parameter_combinations(self, enabled_models: List[str], 
                                        group_params: Dict[str, Any],
                                        fixed: Dict[str, Any], group_name: str,
                                        model_batch_dict: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成参数组合"""
        combinations = []

        if model_batch_dict is not None:
            enabled_pairs = {m: b for m, b in model_batch_dict.items() if m in enabled_models}
            other_params = {k: v for k, v in group_params.items() 
                          if k not in self.constants['excluded_params']}

            if not other_params:
                for model, batch_size in enabled_pairs.items():
                    combo = {**fixed, self.constants['model_type_key']: model, 
                            self.constants['group_key']: group_name}
                    if batch_size is not None:
                        combo[self.constants['batch_size_key']] = batch_size
                    combinations.append(combo)
            else:
                param_items = [(k, _as_list(v)) for k, v in other_params.items() if _as_list(v)]
                if param_items:
                    param_keys, param_values_lists = zip(*param_items)
                    for model, batch_size in enabled_pairs.items():
                        for param_combo in itertools.product(*param_values_lists):
                            combo = {**fixed, self.constants['model_type_key']: model,
                                    self.constants['group_key']: group_name}
                            if batch_size is not None:
                                combo[self.constants['batch_size_key']] = batch_size
                            combo.update(dict(zip(param_keys, param_combo)))
                            combinations.append(combo)
                else:
                    for model, batch_size in enabled_pairs.items():
                        combo = {**fixed, self.constants['model_type_key']: model,
                                self.constants['group_key']: group_name}
                        if batch_size is not None:
                            combo[self.constants['batch_size_key']] = batch_size
                        combinations.append(combo)
        else:
            all_params = {k: v for k, v in group_params.items() 
                         if k != self.constants['model_type_key']}
            param_items = [(k, _as_list(v)) for k, v in all_params.items() if _as_list(v)]

            if param_items:
                param_keys, param_values_lists = zip(*param_items)
                for model in enabled_models:
                    for param_combo in itertools.product(*param_values_lists):
                        combo = {**fixed, self.constants['model_type_key']: model,
                                self.constants['group_key']: group_name}
                        combo.update(dict(zip(param_keys, param_combo)))
                        combinations.append(combo)
            else:
                for model in enabled_models:
                    combo = {**fixed, self.constants['model_type_key']: model,
                            self.constants['group_key']: group_name}
                    combinations.append(combo)

        return combinations


class ExperimentResultsManager:
    """实验结果管理器"""

    def __init__(self, csv_filepath: str, details_filepath: str = None, grid_search_dir: str = None):
        self.csv_filepath = csv_filepath
        self.details_filepath = details_filepath
        self.grid_search_dir = grid_search_dir
        self.fieldnames = None
        self.details_fieldnames = None
        self.constants = GRID_SEARCH_CONSTANTS

        if self.grid_search_dir:
            self.experiments_dir = os.path.join(self.grid_search_dir, "experiments")
            os.makedirs(self.experiments_dir, exist_ok=True)

    def get_csv_fieldnames(self, all_params: List[Dict[str, Any]]) -> List[str]:
        """获取CSV字段名列表"""
        param_keys = sorted({k for params in all_params for k in params.keys()})
        all_param_keys = sorted(set(param_keys + self.constants['common_runtime_params']) 
                               - set(self.constants['excluded_csv_params']))
        other_param_keys = [k for k in all_param_keys 
                          if k not in [self.constants['model_type_key'], self.constants['group_key']]]
        fieldnames = self.constants['csv_base_columns'] + other_param_keys
        self.fieldnames = fieldnames
        return fieldnames

    def initialize_csv_file(self, fieldnames: List[str]) -> None:
        """初始化CSV文件"""
        results_dir = os.path.dirname(self.csv_filepath)
        os.makedirs(results_dir, exist_ok=True)

        with open(self.csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        self.fieldnames = fieldnames

        if self.details_filepath:
            self._initialize_details_csv()

    def _initialize_details_csv(self) -> None:
        """初始化详情CSV文件"""
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

    def append_result_to_csv(self, result: Dict[str, Any]) -> None:
        """追加结果到CSV文件"""
        if not self.fieldnames:
            raise ValueError("CSV字段名未初始化")

        row = {
            "exp_name": result.get("exp_name"),
            "success": result.get("success"),
            "trained_epochs": result.get("trained_epochs", 0),
        }

        # 添加多标签分类指标
        multilabel_metrics = result.get("multilabel_metrics", {})
        if multilabel_metrics:
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
            final_metrics = multilabel_metrics.get("final", {})
            row.update({
                "final_macro_accuracy": final_metrics.get("macro_accuracy"),
                "final_micro_accuracy": final_metrics.get("micro_accuracy"),
                "final_weighted_accuracy": final_metrics.get("weighted_accuracy"),
                "final_macro_f1": final_metrics.get("macro_f1"),
                "final_micro_f1": final_metrics.get("micro_f1"),
                "final_weighted_f1": final_metrics.get("weighted_f1"),
            })

        row.update({
            "best_accuracy": result.get("best_accuracy"),
            "final_accuracy": result.get("final_accuracy"),
        })
        row.update(result.get("params", {}))

        filtered_row = {k: v for k, v in row.items() if k in self.fieldnames}
        for field in self.fieldnames:
            if field not in filtered_row:
                filtered_row[field] = ""

        with open(self.csv_filepath, "a", newline="", encoding="utf-8") as csvfile:
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(filtered_row)
            csvfile.flush()
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)

        if result.get('success', False):
            self._save_enhanced_experiment_data(result)

    def _save_enhanced_experiment_data(self, result: Dict[str, Any]) -> None:
        """保存增强的实验数据"""
        if self.details_filepath and self.details_fieldnames:
            self._append_to_details_csv(result)
        if self.experiments_dir:
            self._save_individual_experiment_files(result)

    def _append_to_details_csv(self, result: Dict[str, Any]) -> None:
        """追加详细指标到详情CSV"""
        detailed_metrics = result.get('detailed_metrics', {})
        if not detailed_metrics:
            return

        exp_name = result.get('exp_name', '')
        params = result.get('params', {})
        config_hash = self._generate_config_hash(params)
        best_epoch = detailed_metrics.get('epoch', result.get('trained_epochs', 0))

        gamma = params.get('loss.params.gamma', params.get('gamma', ''))
        alpha = params.get('loss.params.alpha', params.get('alpha', ''))
        pos_weight = params.get('loss.params.pos_weight', params.get('pos_weight', ''))
        learning_rate = params.get('hp.learning_rate', params.get('learning_rate', ''))
        loss_name = params.get('loss.name', '')
        model_type = params.get('model.type', '')
        batch_size = params.get('hp.batch_size', params.get('batch_size', ''))

        rows = []
        class_metrics = detailed_metrics.get('class_metrics', {})
        for class_name, metrics in class_metrics.items():
            rows.append({
                'exp_name': exp_name, 'config_hash': config_hash, 'epoch': best_epoch,
                '类别名称': class_name,
                '精确率': round(metrics.get('precision', 0), 4),
                '召回率': round(metrics.get('recall', 0), 4),
                'F1分数': round(metrics.get('f1', 0), 4),
                '准确率': round(metrics.get('accuracy', 0), 4),
                '正样本': metrics.get('pos_samples', 0),
                '负样本': metrics.get('neg_samples', 0),
                'gamma': gamma, 'alpha': alpha, 'pos_weight': pos_weight,
                'learning_rate': learning_rate, 'loss_name': loss_name,
                'model_type': model_type, 'batch_size': batch_size
            })

        avg_metrics = [
            ('加权平均', detailed_metrics.get('weighted_avg', {})),
            ('宏平均', detailed_metrics.get('macro_avg', {})),
            ('微平均', detailed_metrics.get('micro_avg', {}))
        ]
        for avg_name, avg_data in avg_metrics:
            if avg_data:
                rows.append({
                    'exp_name': exp_name, 'config_hash': config_hash, 'epoch': best_epoch,
                    '类别名称': avg_name,
                    '精确率': round(avg_data.get('precision', 0), 4),
                    '召回率': round(avg_data.get('recall', 0), 4),
                    'F1分数': round(avg_data.get('f1', 0), 4),
                    '准确率': round(avg_data.get('accuracy', 0), 4),
                    '正样本': '', '负样本': '',
                    'gamma': gamma, 'alpha': alpha, 'pos_weight': pos_weight,
                    'learning_rate': learning_rate, 'loss_name': loss_name,
                    'model_type': model_type, 'batch_size': batch_size
                })

        if rows:
            with open(self.details_filepath, "a", newline="", encoding="utf-8") as csvfile:
                fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
                writer = csv.DictWriter(csvfile, fieldnames=self.details_fieldnames)
                writer.writerows(rows)
                csvfile.flush()
                fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)

    def _save_individual_experiment_files(self, result: Dict[str, Any]) -> None:
        """保存单个实验文件"""
        exp_name = result.get('exp_name', 'unknown')
        exp_dir = os.path.join(self.experiments_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        # 保存实验配置
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

        # 保存指标
        detailed_metrics = result.get('detailed_metrics', {})
        if detailed_metrics and 'class_metrics' in detailed_metrics:
            self._save_class_metrics_history(exp_dir, detailed_metrics)
        if detailed_metrics:
            self._save_best_metrics_summary(exp_dir, detailed_metrics)

    def _save_class_metrics_history(self, exp_dir: str, detailed_metrics: Dict[str, Any]) -> None:
        """保存类别指标历史"""
        import pandas as pd
        class_metrics = detailed_metrics.get('class_metrics', {})
        epoch = detailed_metrics.get('epoch', 0)

        rows = [{'epoch': epoch, 'class_name': cn, 
                 'precision': m.get('precision', 0), 'recall': m.get('recall', 0),
                 'f1': m.get('f1', 0), 'accuracy': m.get('accuracy', 0),
                 'pos_samples': m.get('pos_samples', 0), 'neg_samples': m.get('neg_samples', 0)}
                for cn, m in class_metrics.items()]

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(exp_dir, "class_metrics_history.csv"), index=False, encoding='utf-8')

    def _save_best_metrics_summary(self, exp_dir: str, detailed_metrics: Dict[str, Any]) -> None:
        """保存最佳指标汇总"""
        import pandas as pd
        summary_data = []

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

        avg_metrics = [
            ('加权平均', detailed_metrics.get('weighted_avg', {})),
            ('宏平均', detailed_metrics.get('macro_avg', {})),
            ('微平均', detailed_metrics.get('micro_avg', {}))
        ]
        for avg_name, avg_data in avg_metrics:
            if avg_data:
                summary_data.append({
                    '类别名称': avg_name,
                    '精确率': f"{avg_data.get('precision', 0):.4f}",
                    '召回率': f"{avg_data.get('recall', 0):.4f}",
                    'F1分数': f"{avg_data.get('f1', 0):.4f}",
                    '准确率': f"{avg_data.get('accuracy', 0):.4f}",
                    '正样本数': '', '负样本数': ''
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(os.path.join(exp_dir, "best_metrics_summary.csv"), index=False, encoding='utf-8-sig')

    def _generate_config_hash(self, params: Dict[str, Any]) -> str:
        """生成配置哈希值"""
        key_params = {
            'model_type': params.get('model.type', ''),
            'loss_name': params.get('loss.name', ''),
            'gamma': params.get('loss.params.gamma', params.get('gamma', '')),
            'alpha': params.get('loss.params.alpha', params.get('alpha', '')),
            'pos_weight': params.get('loss.params.pos_weight', params.get('pos_weight', '')),
            'learning_rate': params.get('hp.learning_rate', params.get('learning_rate', '')),
            'batch_size': params.get('hp.batch_size', params.get('batch_size', ''))
        }
        config_str = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


def generate_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """生成参数组合"""
    generator = ParameterCombinationGenerator(config)
    return generator.generate_combinations()


def get_csv_fieldnames(all_params: List[Dict[str, Any]]) -> List[str]:
    """获取CSV字段名列表"""
    temp_manager = ExperimentResultsManager("")
    return temp_manager.get_csv_fieldnames(all_params)


def apply_param_overrides(config, params):
    """应用参数覆盖到配置字典"""
    import copy
    config = copy.deepcopy(config)
    for k, v in (params or {}).items():
        keys = k.split('.')
        target = config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = v
    return config


def run_single_experiment_in_process(params, exp_id, config_path, grid_search_dir=None):
    """进程内运行单个实验"""
    exp_name = f"grid_{exp_id}"
    try:
        from src.trainers.base_trainer import run_training
        from src.utils.config_parser import setup_gpu_config

        config = load_grid_config(config_path)
        config = apply_param_overrides(config, params)
        if grid_search_dir:
            config['grid_search_dir'] = grid_search_dir

        setup_gpu_config(config)
        result = run_training(config, exp_name)
        result["params"] = params
        return result
    except Exception as e:
        print(f"实验 {exp_name} 失败: {e}")
        return {
            "success": False, "exp_name": exp_name, "params": params,
            "best_accuracy": 0.0, "final_accuracy": 0.0, "trained_epochs": 0,
            "error": str(e)
        }


def run_single_experiment_subprocess(params, exp_id, use_multi_gpu, config_path):
    """子进程运行单个实验"""
    exp_name = f"grid_{exp_id}"
    temp_result_file = f"/tmp/grid_result_{exp_id}_{random.randint(1000,9999)}.json"

    if use_multi_gpu:
        import torch
        cmd = ["accelerate", "launch", "--multi_gpu", "--num_processes", str(torch.cuda.device_count())]
    else:
        cmd = [sys.executable, "-u"]

    cmd.extend(["scripts/train.py", "--config", config_path, "--exp_name", exp_name])
    cmd.extend(["--result_file", temp_result_file])

    for k, v in (params or {}).items():
        if k != "group":
            cmd.extend([f"--{k}", str(v)])

    env = os.environ.copy()
    for k in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(k, None)
    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(20000 + random.randint(0, 10000))

    process = subprocess.Popen(cmd, env=env)
    try:
        rc = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise

    success = (rc == 0)

    if os.path.exists(temp_result_file):
        with open(temp_result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        os.remove(temp_result_file)
        result["params"] = params
        result["success"] = result.get("success", success)
        result["exp_name"] = result.get("exp_name", exp_name)
        if result.get("best_accuracy") is None:
            result["best_accuracy"] = 0.0
        if result.get("final_accuracy") is None:
            result["final_accuracy"] = 0.0
        return result

    return {
        "success": success, "exp_name": exp_name, "params": params,
        "best_accuracy": 0.0, "final_accuracy": 0.0, "trained_epochs": 0,
        "error": "Failed to read result file" if success else "Training failed"
    }


def run_single_experiment(params, exp_id, use_multi_gpu=False, config_path="config/grid.yaml", grid_search_dir=None):
    """运行单个实验"""
    if use_multi_gpu:
        return run_single_experiment_subprocess(params, exp_id, use_multi_gpu, config_path)
    else:
        return run_single_experiment_in_process(params, exp_id, config_path, grid_search_dir)


def run_grid_search(args):
    """运行网格搜索"""
    config = load_grid_config(args.config)
    combinations = generate_combinations(config)

    if len(combinations) > args.max_experiments:
        combinations = combinations[:args.max_experiments]

    # 确定输出目录
    task_tag = config.get('task', {}).get('tag', '')
    dataset_type = config.get('data', {}).get('type', '')

    if 'multilabel' in task_tag.lower() or 'multilabel' in dataset_type.lower():
        task_subdir = "neonatal_multilabel" if 'neonatal' in dataset_type.lower() else "multilabel_classification"
    elif 'video' in task_tag.lower():
        task_subdir = "video_classification"
    elif 'image' in task_tag.lower():
        task_subdir = "image_classification"
    else:
        task_subdir = dataset_type.lower() or "general"

    results_dir = os.path.join("runs", task_subdir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_search_dir = os.path.join(results_dir, f"grid_search_{timestamp}")
    csv_filepath = os.path.join(grid_search_dir, "grid_search_results.csv")
    details_filepath = os.path.join(grid_search_dir, "grid_search_details.csv")

    fieldnames = get_csv_fieldnames(combinations)
    results_manager = ExperimentResultsManager(csv_filepath, details_filepath, grid_search_dir)

    os.makedirs(grid_search_dir, exist_ok=True)
    results_manager.initialize_csv_file(fieldnames)

    print(f"开始网格搜索: {len(combinations)} 个实验")
    print(f"配置: {args.config}")
    print(f"输出目录: {grid_search_dir}")

    data_percentage = args.data_percentage if args.data_percentage is not None else 1.0
    results = []
    successful = 0

    for i, params in enumerate(combinations, 1):
        experiment_params = params.copy()
        experiment_params['hp.data_percentage'] = data_percentage
        experiment_params['data_percentage'] = data_percentage

        result = run_single_experiment(
            experiment_params, f"{i:03d}",
            use_multi_gpu=args.multi_gpu,
            config_path=args.config,
            grid_search_dir=grid_search_dir
        )

        results.append(result)
        if result["success"]:
            successful += 1

        if args.save_results:
            results_manager.append_result_to_csv(result)

        if successful > 0:
            current_best = max([r for r in results if r["success"]], key=lambda x: x["best_accuracy"])
            print(f"[{i}/{len(combinations)}] 当前最佳: {current_best['exp_name']} - {current_best['best_accuracy']:.2f}%")

    # 总结
    print(f"\n网格搜索完成! 成功: {successful}/{len(combinations)}")
    if successful > 0:
        best_result = max([r for r in results if r["success"]], key=lambda x: x["best_accuracy"])
        print(f"最佳: {best_result['exp_name']} - {best_result['best_accuracy']:.2f}%")

    return 0 if successful > 0 else 1


def main():
    """主函数"""
    args, _ = parse_arguments(mode="grid_search")
    return run_grid_search(args)


if __name__ == "__main__":
    sys.exit(main())
