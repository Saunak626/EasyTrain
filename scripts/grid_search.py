"""网格搜索启动脚本

设计思路：
本脚本采用调度器-执行器分离的架构设计，实现了高效、稳定的超参数网格搜索。
核心设计原则包括：
- 进程隔离：调度器保持单进程运行，每个实验独立启动子进程，确保资源干净释放
- 多模式支持：支持单卡/CPU和多卡分布式训练，自动选择合适的启动方式
- 环境隔离：为每个实验设置独立的分布式环境变量，避免进程间串扰
- 灵活配置：支持笛卡尔积和特殊配对模式的参数组合生成
- 结果管理：自动解析实验结果，生成结构化的CSV报告

核心功能：
- generate_combinations: 智能参数组合生成，支持多种组合策略
- run_single_experiment: 单实验执行器，处理进程启动和结果收集
- run_grid_search: 网格搜索调度器，协调整个搜索流程
- parse_result_from_files: 结果解析器，从多种格式中提取训练指标
- save_results_to_csv: 结果持久化，生成便于分析的CSV报告

特殊处理：
- 端口管理：为每个实验分配唯一MASTER_PORT，避免分布式训练冲突
- 环境清理：清理父进程的分布式环境变量，确保子进程环境干净
- 中断处理：优雅处理Ctrl+C中断，确保子进程正确终止
- 配对模式：当batch_size数组与model数组长度相同时，按对应顺序配对
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

def parse_result_from_files(exp_name):
    """从结构化文件中解析最终结果
    
    设计思路：
    实现多层次的结果解析策略，确保在不同情况下都能获取到有效的训练结果。
    采用优先级回退机制，提高结果解析的鲁棒性。
    
    解析策略：
    1. 优先级1：result.json - 包含完整的最终结果摘要
    2. 优先级2：metrics.jsonl - 逐行解析训练过程中的指标
    3. 回退：返回默认值(0.0, 0.0)
    
    Args:
        exp_name (str): 实验名称，用于构建结果文件路径
        
    Returns:
        tuple[float, float]: (最佳准确率, 最终准确率)
            - 最佳准确率：训练过程中达到的最高验证准确率
            - 最终准确率：训练结束时的验证准确率
    
    文件格式：
        - result.json: {"best_accuracy": float, "final_accuracy": float}
        - metrics.jsonl: 每行一个JSON对象，包含"val_acc"字段
    """
    result_dir = os.path.join("runs", exp_name)
    final_json = os.path.join(result_dir, "result.json")
    metrics_path = os.path.join(result_dir, "metrics.jsonl")

    # 优先读取 result.json
    try:
        if os.path.exists(final_json):
            with open(final_json, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                best_accuracy = float(data.get("best_accuracy", 0.0))
                final_accuracy = float(data.get("final_accuracy", best_accuracy))
                return best_accuracy, final_accuracy
    except Exception:
        pass

    # 回退：扫描 metrics.jsonl
    try:
        if os.path.exists(metrics_path):
            last_val, best_val = None, 0.0
            with open(metrics_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        va = json.loads(line).get("val_acc")
                        if isinstance(va, (int, float)):
                            last_val = float(va)
                            best_val = max(best_val, last_val)
                    except Exception:
                        continue
            if last_val is not None:
                return best_val, last_val
    except Exception:
        pass

    return 0.0, 0.0


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
            - experiment_name: 实验名称
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
        "experiment_id", "experiment_name", "success",
        "best_accuracy", "final_accuracy"
    ] + param_keys

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            row = {
                "experiment_id": f"{i:03d}",
                "experiment_name": r.get("exp_name"),
                "success": r.get("success"),
                "best_accuracy": r.get("best_accuracy"),
                "final_accuracy": r.get("final_accuracy"),
            }
            row.update(r.get("params", {}))
            writer.writerow(row)

    return filepath


# ================= 多卡子进程启动辅助函数 =================

def _clean_env_for_child():
    """
    清理父进程的分布式环境变量
    
    设计思路：
    在网格搜索场景下，调度器进程可能已经设置了分布式相关的环境变量。
    如果子进程继承这些变量，可能导致分布式训练初始化失败或连接错误。
    因此需要为每个子实验提供干净的环境。
    
    清理的环境变量：
    - LOCAL_RANK: 本地进程排名
    - RANK: 全局进程排名  
    - WORLD_SIZE: 总进程数
    - MASTER_ADDR: 主节点地址
    - MASTER_PORT: 主节点端口
    
    Returns:
        dict: 清理后的环境变量字典，可直接用于subprocess
    
    使用场景：
        每次启动新的训练子进程时调用，确保环境隔离
    """
    env = os.environ.copy()
    for k in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(k, None)
    return env


def _unique_master_port(base=20000, span=10000):
    """为每个实验分配唯一端口
    
    设计思路：
    在并发或连续运行多个分布式训练实验时，如果使用相同的MASTER_PORT，
    会导致NCCL连接冲突和通信异常。通过随机分配端口避免此问题。
    
    Args:
        base (int): 端口范围起始值，默认20000
        span (int): 端口范围大小，默认10000
        
    Returns:
        str: 随机生成的端口号字符串
        
    端口范围：
        [base, base+span)，默认为[20000, 30000)
        避开常用端口，减少冲突概率
    """
    return str(base + random.randint(0, span))


def _infer_num_procs() -> int:
    env_ids = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if env_ids:
        return max(1, len([x for x in env_ids.split(",") if x.strip() != ""]))
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def run_single_experiment(params, exp_id, use_multi_gpu=False, config_path="config/grid.yaml"): #, accelerate_args=""):
    """运行单个实验（每个实验独立的进程/进程组）"""
    exp_name = f"grid_{exp_id}"

    # 组装基础命令
    if use_multi_gpu:
        cmd = ["accelerate", "launch", "--multi_gpu", "--num_processes", str(torch.cuda.device_count())]
    else:
        cmd = [sys.executable, "-u"]
    
    # 添加训练脚本和基础参数
    cmd.extend(["scripts/train.py", "--config", config_path, "--experiment_name", exp_name])
    
    # 添加参数覆盖
    for k, v in (params or {}).items():
        cmd.extend([f"--{k}", str(v)])

    print(f"\n{'='*60}")
    print(f"🚀 开始实验 {exp_id}: {exp_name}")
    print(f"📋 参数: {params}")
    print(f"{'='*60}")

    # 清理父环境 + 为本实验设置唯一端口
    env = _clean_env_for_child()
    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = _unique_master_port()

    # 直接继承 TTY，保留 tqdm 一行刷新
    process = subprocess.Popen(cmd, env=env)
    try:
        # 等待子进程结束
        rc = process.wait()
    except KeyboardInterrupt:
        # 捕获到 Ctrl+C (KeyboardInterrupt)
        print(f"\n捕获到中断信号(Ctrl+C)，正在终止子进程 {process.pid}...")
        process.terminate()  # 发送 SIGTERM 信号，请求子进程终止
        process.wait()       # 等待子进程完全退出
        print("子进程已终止。")
        raise                # 重新抛出异常，以确保整个网格搜索脚本停止

    success = (rc == 0)

    # 从文件解析结果
    best_accuracy, final_accuracy = parse_result_from_files(exp_name)
    if success:
        if best_accuracy == 0.0 and final_accuracy == 0.0:
            print(f"⚠️  {exp_name} 结束，但未找到结果文件。请检查 runs/{exp_name}/。")
        else:
            print(f"✅ 实验 {exp_name} 完成，最佳: {best_accuracy:.2f}% | 最终: {final_accuracy:.2f}%")
    else:
        print(f"❌ 实验 {exp_name} 失败（返回码 {rc}）。请查看控制台与 runs/{exp_name}/。")

    return {
        "success": success,
        "exp_name": exp_name,
        "params": params,
        "best_accuracy": best_accuracy,
        "final_accuracy": final_accuracy,
    }


def run_grid_search(args):
    """运行网格搜索（串行，确保资源干净释放）"""
    config = load_grid_config(args.config)
    
    # 实验参数进行笛卡尔积组合
    combinations = generate_combinations(config)

    # 截断实验数量
    if len(combinations) > args.max_experiments:
        combinations = combinations[:args.max_experiments]

    print(f"\n🚀 开始网格搜索，共 {len(combinations)} 个实验")
    print(f"📊 使用配置文件: {args.config}")
    print(f"🎯 多卡训练: {'是' if args.multi_gpu else '否'}")
    print("=" * 60)

    results = []
    successful = 0

    for i, params in enumerate(combinations, 1):
        print(f"\n📊 准备实验 {i}/{len(combinations)}")

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
    print("\n" + "=" * 60)
    print(f"📈 网格搜索完成！")
    print(f"✅ 成功实验数量: {successful}/{len(combinations)}")

    if successful > 0:
        # 筛选出所有成功完成的实验结果
        successful_results = [r for r in results if r["success"]]
        # 找到“最佳准确率”最高的实验结果
        best_result = max(successful_results, key=lambda x: x["best_accuracy"])

        print(f"\n🏆 最佳实验结果:")
        print(f"实验名称: {best_result['exp_name']}, 最佳准确率: {best_result['best_accuracy']:.2f}%, 最终准确率: {best_result['final_accuracy']:.2f}%")
        
        # 按最佳精度排序前n组结果
        top_results = sorted(successful_results, key=lambda x: x["best_accuracy"], reverse=True)[:args.top_n]
        
        print(f"\n📊 前{args.top_n}名实验结果:")
        for i, r in enumerate(top_results, 1):
            print(f"{i}. {r['exp_name']} - {r['best_accuracy']:.2f}% - {r['params']}")

    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"grid_search_results_{timestamp}.csv"
        saved_filepath = save_results_to_csv(results, results_filename)
        if saved_filepath:
            print(f"\n💾 结果已保存到: {saved_filepath}")

    return 0 if successful > 0 else 1


def main():
    """主函数：调度器始终单进程，不进入 Accelerate 环境"""
    args, _ = parse_arguments(mode="grid_search")
    return run_grid_search(args)


if __name__ == "__main__":
    sys.exit(main())