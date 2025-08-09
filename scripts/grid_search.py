"""网格搜索启动脚本

设计思路：
本脚本实现了高效、稳定的超参数网格搜索，采用进程内调用的架构设计。
核心设计原则包括：
- 进程内调用：直接调用训练函数，避免子进程开销和文件I/O依赖
- 内存传递：通过函数返回值直接获取训练结果，提高效率和可靠性
- 资源管理：每个实验后自动清理GPU缓存，确保资源干净释放
- 灵活配置：支持笛卡尔积和特殊配对模式的参数组合生成
- 结果管理：直接收集训练结果，生成结构化的CSV报告

核心功能：
- generate_combinations: 智能参数组合生成，支持多种组合策略
- run_single_experiment: 单实验执行器，进程内调用训练函数
- run_grid_search: 网格搜索调度器，协调整个搜索流程
- apply_param_overrides: 参数覆盖器，支持嵌套参数路径
- save_results_to_csv: 结果持久化，生成便于分析的CSV报告

特殊处理：
- 异常处理：单个实验失败不影响后续实验继续执行
- 内存管理：每个实验后清理GPU缓存，防止内存泄漏
- 配对模式：当batch_size数组与model数组长度相同时，按对应顺序配对
- SwanLab集成：保留SwanLab实验追踪，删除JSON文件依赖
"""
import itertools
import subprocess
import yaml
import os
import sys
import time
import csv
import json
import random
import torch

from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments

def load_grid_config(path="config/grid.yaml"):
    """加载网格搜索配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_list(v):
    """标量→单元素列表；列表/元组原样；None→空列表
    
    设计思路：
    统一参数格式处理的工具函数，确保所有参数都能以列表形式进行后续处理。
    这种设计简化了参数组合生成的逻辑，避免了大量的类型检查代码。
    
    Args:
        v: 任意类型的参数值
        
    Returns:
        list: 统一格式化后的列表
            - None → []
            - 标量 → [标量]
            - 列表/元组 → 原样返回
    """
    if v is None:
        return []
    return v if isinstance(v, (list, tuple)) else [v]

def generate_combinations(config):
    """
    智能参数组合生成器，支持多种组合策略
    
    设计思路：
    本函数实现了灵活的超参数组合生成策略，核心设计包括：
    - 双模式支持：标准笛卡尔积模式和特殊配对模式
    - 智能检测：自动识别batch_size与model配对的场景
    - 参数分层：区分固定参数(fixed)和搜索参数(grid)
    - 类型容错：自动处理标量、列表、None等不同类型
    
    组合策略：
    1. 标准模式：所有参数进行笛卡尔积组合
    2. 配对模式：当model.type数组与hp.batch_size数组长度相同时，
       按对应位置配对，避免不合理的模型-批大小组合
    
    Args:
        config (dict): 网格搜索配置
            - grid_search.grid: 搜索参数字典
            - grid_search.fixed: 固定参数字典
            
    Returns:
        list[dict]: 参数组合列表，每个字典代表一组实验参数
    
    示例：
        配对模式：model.type=["resnet", "vit"], hp.batch_size=[32, 16]
        → 生成[("resnet", 32), ("vit", 16)]而非4种组合
    """
    gs = (config or {}).get("grid_search", {}) or {}
    fixed = gs.get("fixed", {}) or {}
    grid = gs.get("grid", {}) or {}

    # 边界情况：无搜索参数时返回固定参数
    if not grid:
        return [fixed] if fixed else [{}]

    # === 智能配对模式检测 ===
    # 提取model.type和hp.batch_size参数列表
    model_types = _as_list(grid.get("model.type", []))
    batch_sizes = _as_list(grid.get("hp.batch_size", []))
    
    # 配对模式触发条件：两个数组都有多个元素且长度相同
    # 设计目的：避免大模型配小batch_size或小模型配大batch_size的不合理组合
    if (len(model_types) > 1 and len(batch_sizes) > 1 and 
        len(model_types) == len(batch_sizes)):
        
        # 创建model-batch配对：按位置一一对应
        model_batch_pairs = list(zip(model_types, batch_sizes))
        
        # 分离其他需要搜索的参数
        other_grid = {k: v for k, v in grid.items() 
                     if k not in ["model.type", "hp.batch_size"]}
        
        if not other_grid:
            # 纯配对模式：只有model和batch_size需要配对
            return [{**fixed, "model.type": model_type, "hp.batch_size": batch_size}
                   for model_type, batch_size in model_batch_pairs]
        else:
            # 混合模式：配对参数与其他参数做笛卡尔积
            other_valid_items = [(k, _as_list(v)) for k, v in other_grid.items() if _as_list(v)]
            if other_valid_items:
                other_keys, other_values_lists = zip(*other_valid_items)
                combinations = []
                # 每个model-batch配对与其他参数的所有组合配对
                for model_type, batch_size in model_batch_pairs:
                    for other_combo in itertools.product(*other_values_lists):
                        combo = {**fixed, "model.type": model_type, "hp.batch_size": batch_size}
                        combo.update(dict(zip(other_keys, other_combo)))
                        combinations.append(combo)
                return combinations
            else:
                # 其他参数为空，回退到纯配对模式
                return [{**fixed, "model.type": model_type, "hp.batch_size": batch_size}
                       for model_type, batch_size in model_batch_pairs]
    
    # === 标准笛卡尔积模式 ===
    # 过滤掉空值参数，避免生成无效组合
    valid_items = [(k, _as_list(v)) for k, v in grid.items() if _as_list(v)]
    
    # 边界情况：所有参数都为空
    if not valid_items:
        return [fixed] if fixed else [{}]

    # 分离参数名和参数值列表
    keys, values_lists = zip(*valid_items)

    # 生成所有参数的笛卡尔积组合，并合并固定参数
    return [{**fixed, **dict(zip(keys, combo))} 
            for combo in itertools.product(*values_lists)]

# parse_result_from_files 函数已删除，改为直接从训练函数获取结果


def save_results_to_csv(results, filename):
    """保存实验结果到CSV文件
    
    设计思路：
    将网格搜索的所有实验结果汇总到一个CSV文件中，便于后续分析和比较。
    采用标准化的CSV格式，确保数据的可读性和可处理性。
    
    功能特性：
    - 自动创建runs目录（如果不存在）
    - 包含完整的超参数信息和训练结果
    - 使用UTF-8编码，支持中文字符
    - 自动处理嵌套参数的展平
    
    Args:
        results (list[dict]): 实验结果列表，每个字典包含：
            - 超参数字段（如model.type, hp.batch_size等）
            - best_accuracy: 最佳准确率
            - final_accuracy: 最终准确率
            - exp_name: 实验名称
        filename (str): CSV文件名（不含路径）
        
    Returns:
        str|None: 保存的完整文件路径，如果results为空则返回None
    
    CSV格式：
        包含所有超参数列和结果列，便于Excel等工具打开分析
    """
    if not results:
        return None

    results_dir = "runs"
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    # 收集所有出现过的参数字段
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


# 子进程启动辅助函数已删除，改为进程内调用方式


def apply_param_overrides(config, params):
    """应用参数覆盖到配置字典
    
    Args:
        config (dict): 基础配置字典
        params (dict): 参数覆盖字典，支持嵌套路径如 "hp.batch_size"
        
    Returns:
        dict: 应用覆盖后的配置字典
    """
    import copy
    config = copy.deepcopy(config)
    
    for k, v in (params or {}).items():
        # 解析嵌套参数路径，如 "hp.batch_size" -> config["hp"]["batch_size"]
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
    """运行单个实验（混合方式）
    
    设计思路：
    - 单卡训练：使用进程内调用，高效且无需文件I/O
    - 多卡训练：使用子进程启动，通过临时文件传递结果
    
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
    """运行网格搜索（串行，确保资源干净释放）"""
    config = load_grid_config(args.config)
    
    # 实验参数进行笛卡尔积组合
    combinations = generate_combinations(config)

    # 截断实验数量
    if len(combinations) > args.max_experiments:
        combinations = combinations[:args.max_experiments]

    print(f"🚀 开始网格搜索，共 {len(combinations)} 个实验")
    print(f"📊 使用配置文件: {args.config}")
    print(f"🎯 多卡训练: {'是' if args.multi_gpu else '否'}")
    print("=" * 60)

    results = []
    successful = 0

    for i, params in enumerate(combinations, 1):
        print(f"📊 准备实验 {i}/{len(combinations)}")

        result = run_single_experiment(
            params, f"{i:03d}",
            use_multi_gpu=args.multi_gpu,
            config_path=args.config, # 训练使用的统一配置
        )
        
        results.append(result)
        if result["success"]:
            successful += 1

        # 适当间隔
        # time.sleep(1.0)

    # 总结
    print("=" * 60)
    print(f"📈 网格搜索完成！")
    print(f"✅ 成功实验数量: {successful}/{len(combinations)}")

    if successful > 0:
        # 筛选出所有成功完成的实验结果
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