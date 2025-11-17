"""网格搜索统一脚本

核心优化：
- 一次启动，串行执行所有实验
- 所有实验在同一进程中顺序运行
- 无需为每个实验创建子进程

启动方式：
- 单卡：python scripts/grid_search_unified.py --config ...
- 多卡：accelerate launch scripts/grid_search_unified.py --config ...

"""
import yaml
import os
import sys

from typing import Dict, Any, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments
from src.utils.grid_search_generator import generate_combinations
from src.utils.experiment_results import ExperimentResultsManager, get_csv_fieldnames


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


# =========================
# 核心实验执行函数（统一版本）
# =========================

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


# ======================
#     主网格搜索函数
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
#         主函数
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
