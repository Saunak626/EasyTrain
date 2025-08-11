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
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments

def load_grid_config(path="config/grid.yaml"):
    """加载网格搜索配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_list(v):
    """将输入转换为列表格式

    Args:
        v: 任意类型的参数值

    Returns:
        list: 统一格式化后的列表
    """
    if v is None:
        return []
    return v if isinstance(v, (list, tuple)) else [v]

def generate_combinations(config):
    """生成参数组合列表

    支持标准笛卡尔积和智能配对两种模式。
    当model.type和hp.batch_size数组长度相同时，按位置配对。
    支持通过models_to_train参数过滤要训练的模型。

    Args:
        config (dict): 网格搜索配置

    Returns:
        list[dict]: 参数组合列表
    """
    gs = (config or {}).get("grid_search", {}) or {}
    fixed = gs.get("fixed", {}) or {}
    grid = gs.get("grid", {}) or {}

    if not grid:
        return [fixed] if fixed else [{}]

    # 获取模型选择列表，如果未配置则使用grid中的所有模型
    models_to_train = config.get("models_to_train", [])

    # 检测智能配对模式
    model_types = _as_list(grid.get("model.type", []))

    # 如果配置了models_to_train，则过滤模型列表
    if models_to_train:
        model_types = [model for model in model_types if model in models_to_train]
        print(f"🎯 根据models_to_train配置，将训练以下模型: {model_types}")

    batch_sizes = _as_list(grid.get("hp.batch_size", []))

    # 配对模式：两个数组长度相同时按位置配对
    if (len(model_types) > 1 and len(batch_sizes) > 1 and
        len(model_types) == len(batch_sizes)):

        model_batch_pairs = list(zip(model_types, batch_sizes))
        other_grid = {k: v for k, v in grid.items()
                     if k not in ["model.type", "hp.batch_size"]}

        if not other_grid:
            return [{**fixed, "model.type": model_type, "hp.batch_size": batch_size}
                   for model_type, batch_size in model_batch_pairs]
        else:
            other_valid_items = [(k, _as_list(v)) for k, v in other_grid.items() if _as_list(v)]
            if other_valid_items:
                other_keys, other_values_lists = zip(*other_valid_items)
                combinations = []
                for model_type, batch_size in model_batch_pairs:
                    for other_combo in itertools.product(*other_values_lists):
                        combo = {**fixed, "model.type": model_type, "hp.batch_size": batch_size}
                        combo.update(dict(zip(other_keys, other_combo)))
                        combinations.append(combo)
                return combinations
            else:
                return [{**fixed, "model.type": model_type, "hp.batch_size": batch_size}
                       for model_type, batch_size in model_batch_pairs]

    # 标准笛卡尔积模式
    valid_items = [(k, _as_list(v)) for k, v in grid.items() if _as_list(v)]

    if not valid_items:
        return [fixed] if fixed else [{}]

    keys, values_lists = zip(*valid_items)
    return [{**fixed, **dict(zip(keys, combo))}
            for combo in itertools.product(*values_lists)]

def save_results_to_csv(results, filename):
    """保存实验结果到CSV文件

    Args:
        results (list[dict]): 实验结果列表
        filename (str): CSV文件名

    Returns:
        str|None: 保存的文件路径
    """
    if not results:
        return None

    results_dir = "runs"
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    param_keys = sorted({k for r in results for k in r.get("params", {}).keys()})

    fieldnames = [
        "experiment_id", "exp_name", "success",
        "best_accuracy", "final_accuracy"
    ] + param_keys

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            row = {
                "experiment_id": f"{i:03d}",
                "exp_name": r.get("exp_name"),
                "success": r.get("success"),
                "best_accuracy": r.get("best_accuracy"),
                "final_accuracy": r.get("final_accuracy"),
            }
            row.update(r.get("params", {}))
            writer.writerow(row)

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
    
    # 添加参数覆盖
    for k, v in (params or {}).items():
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

    print(f"{'='*60}")
    print(f"🚀 开始实验 {exp_id}: {exp_name}")
    print(f"📋 参数: {params}")
    print(f"🎯 多卡训练: {'是' if use_multi_gpu else '否'}")
    print(f"{'='*60}")

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

    print(f"🚀 开始网格搜索，共 {len(combinations)} 个实验")
    print(f"📊 使用配置文件: {args.config}")
    print("=" * 60)

    results = []
    successful = 0

    for i, params in enumerate(combinations, 1):
        print(f"📊 准备实验 {i}/{len(combinations)}")

        result = run_single_experiment(
            params, f"{i:03d}",
            use_multi_gpu=args.multi_gpu,
            config_path=args.config,
        )

        results.append(result)
        if result["success"]:
            successful += 1

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"grid_search_results_{timestamp}.csv"
        saved_filepath = save_results_to_csv(results, results_filename)
        if saved_filepath:
            print(f"💾 结果已保存到: {saved_filepath}")

    return 0 if successful > 0 else 1


def main():
    """主函数：调度器始终单进程，不进入 Accelerate 环境"""
    args, _ = parse_arguments(mode="grid_search")
    return run_grid_search(args)


if __name__ == "__main__":
    sys.exit(main())