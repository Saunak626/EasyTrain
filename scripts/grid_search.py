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
import pandas as pd
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
    """
    分组式参数组合生成器

    设计逻辑：
    1. 从YAML中获取groups配置，每组有自己的模型和超参数范围
    2. 为每组内的参数进行笛卡尔积组合
    3. 根据models_to_train过滤启用的模型
    4. 避免无意义的模型-参数组合，节省算力

    Args:
        config (dict): 网格搜索配置

    Returns:
        list[dict]: 参数组合列表，每个字典代表一组实验参数
    """
    gs = (config or {}).get("grid_search", {}) or {}
    fixed = gs.get("fixed", {}) or {}
    models_to_train = config.get("models_to_train", [])

    # === 分组式配置处理 ===
    if "groups" in gs and gs["groups"]:
        print(f"📋 使用分组式网格搜索配置")
        return _generate_combinations_by_groups(gs["groups"], fixed, models_to_train)

    # === 边界情况：无搜索参数 ===
    else:
        print(f"⚠️  未找到groups配置，返回固定参数")
        return [fixed] if fixed else [{}]


def _generate_combinations_by_groups(groups_config, fixed, models_to_train):
    """分组式参数组合生成器 - 支持组内模型-batch_size智能配对"""
    all_combinations = []
    total_groups = len(groups_config)
    
    print(f"🎯 发现 {total_groups} 个模型组:")
    for group_name in groups_config.keys():
        group_models = _as_list(groups_config[group_name].get("model.type", []))
        print(f"   - {group_name}: {group_models}")
    
    for group_name, group_params in groups_config.items():
        print(f"\n🔧 处理模型组: {group_name}")
        
        # === 第1步：获取组内的模型和batch_size ===
        group_models = _as_list(group_params.get("model.type", []))
        group_batch_sizes = _as_list(group_params.get("hp.batch_size", []))
        
        print(f"   📋 组内配置:")
        print(f"      model.type: {group_models} (长度: {len(group_models)})")
        print(f"      hp.batch_size: {group_batch_sizes} (长度: {len(group_batch_sizes)})")
        
        # === 第2步：处理模型-batch_size配对逻辑 ===
        if group_batch_sizes:
            if len(group_batch_sizes) == 1:
                # 情况1：batch_size长度=1，扩充到与model.type一致
                group_batch_sizes = group_batch_sizes * len(group_models)
                print(f"   🔄 扩充batch_size: {group_batch_sizes} (扩充到与model.type长度一致)")
                # 创建一对一配对字典
                model_batch_dict = dict(zip(group_models, group_batch_sizes))
                print(f"   📊 模型-batch_size配对字典: {model_batch_dict}")
            elif len(group_batch_sizes) == len(group_models):
                # 情况2：batch_size长度=model.type长度，按顺序配对
                print(f"   ✅ 长度匹配，将按顺序配对")
                model_batch_dict = dict(zip(group_models, group_batch_sizes))
                print(f"   📊 模型-batch_size配对字典: {model_batch_dict}")
            else:
                # 情况3：batch_size长度≠1且≠model.type长度，作为独立参数处理
                print(f"   🔄 batch_size作为独立参数处理，将与模型进行笛卡尔积组合")
                model_batch_dict = None  # 标记为独立参数处理
        else:
            # 没有batch_size配置，所有模型使用默认值
            model_batch_dict = {model: None for model in group_models}
            print(f"   📊 无batch_size配置，使用默认值")
        
        # === 第4步：根据models_to_train过滤模型 ===
        if models_to_train:
            enabled_models = [model for model in group_models if model in models_to_train]
            if not enabled_models:
                print(f"   ⏭️  跳过组 {group_name}：无启用的模型")
                continue
            print(f"   🎯 启用的模型: {enabled_models}")
        else:
            enabled_models = group_models
            print(f"   🎯 使用所有模型: {enabled_models}")
        
        # === 第5步：处理参数组合 ===
        if model_batch_dict is not None:
            # 有模型-batch_size配对的情况
            enabled_pairs = {model: batch_size for model, batch_size in model_batch_dict.items() 
                            if model in enabled_models}
            print(f"   🎯 启用的模型配对: {enabled_pairs}")
            
            # 获取其他参数（排除model.type和hp.batch_size）
            other_params = {k: v for k, v in group_params.items() 
                           if k not in ["model.type", "hp.batch_size"]}
            
            # 生成组合
            if not other_params:
                # 只有模型-batch_size配对，无其他参数
                for model, batch_size in enabled_pairs.items():
                    combo = {**fixed, "model.type": model, "group": group_name}
                    if batch_size is not None:
                        combo["hp.batch_size"] = batch_size
                    all_combinations.append(combo)
            else:
                # 有其他参数，进行笛卡尔积组合
                param_items = [(k, _as_list(v)) for k, v in other_params.items() if _as_list(v)]
                if param_items:
                    param_keys, param_values_lists = zip(*param_items)
                    for model, batch_size in enabled_pairs.items():
                        for param_combo in itertools.product(*param_values_lists):
                            combo = {
                                **fixed,
                                "model.type": model,
                                "group": group_name
                            }
                            if batch_size is not None:
                                combo["hp.batch_size"] = batch_size
                            combo.update(dict(zip(param_keys, param_combo)))
                            all_combinations.append(combo)
                else:
                    # 其他参数都为空
                    for model, batch_size in enabled_pairs.items():
                        combo = {**fixed, "model.type": model, "group": group_name}
                        if batch_size is not None:
                            combo["hp.batch_size"] = batch_size
                        all_combinations.append(combo)
        else:
            # batch_size作为独立参数，与模型进行笛卡尔积组合
            all_params = {k: v for k, v in group_params.items() if k != "model.type"}
            param_items = [(k, _as_list(v)) for k, v in all_params.items() if _as_list(v)]
            
            if param_items:
                param_keys, param_values_lists = zip(*param_items)
                for model in enabled_models:
                    for param_combo in itertools.product(*param_values_lists):
                        combo = {
                            **fixed,
                            "model.type": model,
                            "group": group_name
                        }
                        combo.update(dict(zip(param_keys, param_combo)))
                        all_combinations.append(combo)
            else:
                # 无其他参数
                for model in enabled_models:
                    combo = {**fixed, "model.type": model, "group": group_name}
                    all_combinations.append(combo)
        
        # 计算当前组的组合数量和计算过程
        group_combinations = len([c for c in all_combinations if c.get("group") == group_name])

        # 计算组合数量的分解
        if model_batch_dict is not None:
            # 有模型-batch_size配对的情况
            model_count = len(enabled_pairs)
            other_params = {k: v for k, v in group_params.items()
                           if k not in ["model.type", "hp.batch_size"]}
            param_items = [(k, _as_list(v)) for k, v in other_params.items() if _as_list(v)]
            if param_items:
                param_counts = [len(_as_list(v)) for _, v in other_params.items() if _as_list(v)]
                other_count = 1
                for count in param_counts:
                    other_count *= count
                print(f"   ✅ 组 {group_name} 生成 {group_combinations} 个组合 ({model_count}模型 × {other_count}参数组合)")
            else:
                print(f"   ✅ 组 {group_name} 生成 {group_combinations} 个组合 ({model_count}模型)")
        else:
            # batch_size作为独立参数的情况
            model_count = len(enabled_models)
            all_params = {k: v for k, v in group_params.items() if k != "model.type"}
            param_items = [(k, _as_list(v)) for k, v in all_params.items() if _as_list(v)]
            if param_items:
                param_counts = [len(_as_list(v)) for _, v in all_params.items() if _as_list(v)]
                other_count = 1
                for count in param_counts:
                    other_count *= count
                print(f"   ✅ 组 {group_name} 生成 {group_combinations} 个组合 ({model_count}模型 × {other_count}参数组合)")
            else:
                print(f"   ✅ 组 {group_name} 生成 {group_combinations} 个组合 ({model_count}模型)")
    
    print(f"\n🎉 分组式搜索总计生成 {len(all_combinations)} 个组合")
    return all_combinations



def get_csv_fieldnames(all_params):
    """获取CSV文件的字段名列表"""
    param_keys = sorted({k for params in all_params for k in params.keys()})
    
    # 将model.type移到第3列，group移到第4列，其他参数按原顺序排列
    other_param_keys = [k for k in param_keys if k not in ["model.type", "group"]]
    
    fieldnames = [
        "experiment_id", "exp_name", "model.type", "group", "success",
        "best_accuracy", "final_accuracy"
    ] + other_param_keys
    
    return fieldnames


def initialize_csv_file(filepath, fieldnames):
    """初始化CSV文件，写入表头"""
    results_dir = os.path.dirname(filepath)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def append_result_to_csv(result, filepath, fieldnames, experiment_id):
    """实时追加单个结果到CSV文件（线程安全）
    
    Args:
        result (dict): 实验结果
        filepath (str): CSV文件路径
        fieldnames (list): CSV字段名列表
        experiment_id (int): 实验ID
    """
    try:
        # 使用文件锁确保线程安全
        with open(filepath, "a", newline="", encoding="utf-8") as csvfile:
            # 获取文件锁
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            row = {
                "experiment_id": f"{experiment_id:03d}",
                "exp_name": result.get("exp_name"),
                "success": result.get("success"),
                "best_accuracy": result.get("best_accuracy"),
                "final_accuracy": result.get("final_accuracy"),
            }
            row.update(result.get("params", {}))
            
            writer.writerow(row)
            
            # 释放文件锁
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)
            
    except Exception as e:
        print(f"⚠️  写入CSV失败: {e}")


def load_completed_experiments(filepath):
    """加载已完成的实验，支持断点续传
    
    Args:
        filepath (str): CSV文件路径
        
    Returns:
        set: 已完成的实验名称集合
    """
    if not os.path.exists(filepath):
        return set()
    
    try:
        df = pd.read_csv(filepath)
        completed_experiments = set(df['exp_name'].tolist())
        print(f"🔄 发现已完成的实验: {len(completed_experiments)} 个")
        return completed_experiments
    except Exception as e:
        print(f"⚠️  读取已完成实验失败: {e}")
        return set()


def save_results_to_csv(results, filename):
    """保存实验结果到CSV文件（兼容旧接口）

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

    # 获取所有参数的字段名
    all_params = [r.get("params", {}) for r in results]
    fieldnames = get_csv_fieldnames(all_params)

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

    # 准备CSV文件
    results_dir = "runs"
    if args.results_file:
        # 使用命令行指定的文件名
        results_filename = args.results_file
    else:
        # 使用默认的时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"grid_search_results_{timestamp}.csv"
    csv_filepath = os.path.join(results_dir, results_filename)
    
    # 获取CSV字段名
    all_params = [params for params in combinations]
    fieldnames = get_csv_fieldnames(all_params)
    
    # 断点续传：检查已完成的实验
    completed_experiments = set()
    if args.save_results:
        os.makedirs(results_dir, exist_ok=True)
        
        # 如果用户指定了结果文件，优先检查该文件
        if args.results_file and os.path.exists(csv_filepath):
            completed_experiments = load_completed_experiments(csv_filepath)
            if completed_experiments:
                print(f"🔄 断点续传: 使用指定的结果文件 {results_filename}")
        else:
            # 否则检查是否有其他结果文件存在（用于断点续传）
            existing_files = [f for f in os.listdir(results_dir) if f.startswith("grid_search_results_") and f.endswith(".csv")]
            if existing_files and not args.results_file:
                latest_file = max(existing_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
                latest_filepath = os.path.join(results_dir, latest_file)
                completed_experiments = load_completed_experiments(latest_filepath)
                
                if completed_experiments:
                    # 使用已存在的文件继续写入
                    csv_filepath = latest_filepath
                    results_filename = latest_file
                    print(f"🔄 断点续传: 使用已存在的结果文件 {latest_file}")
        
        # 如果没有找到已完成的实验，初始化新的CSV文件
        if not completed_experiments:
            initialize_csv_file(csv_filepath, fieldnames)
    else:
        # 不保存结果时也需要初始化
        initialize_csv_file(csv_filepath, fieldnames)

    print(f"🚀 开始网格搜索，共 {len(combinations)} 个实验")
    print(f"📊 使用配置文件: {args.config}")
    print(f"💾 结果文件: {csv_filepath}")
    
    # 显示全局参数覆盖
    if args.data_percentage is not None:
        print(f"🎯 全局参数覆盖: data_percentage={args.data_percentage}")
    
    print("=" * 60)

    results = []
    successful = 0
    skipped = 0

    for i, params in enumerate(combinations, 1):
        exp_name = f"grid_{i:03d}"
        
        # 断点续传：跳过已完成的实验
        if exp_name in completed_experiments:
            print(f"⏭️  跳过已完成的实验 {i}/{len(combinations)}: {exp_name}")
            skipped += 1
            continue
            
        print(f"📊 准备实验 {i}/{len(combinations)}")
        
        # 将命令行参数添加到实验参数中
        experiment_params = params.copy()
        if args.data_percentage is not None:
            experiment_params['data_percentage'] = args.data_percentage

        result = run_single_experiment(
            experiment_params, f"{i:03d}",
            use_multi_gpu=args.multi_gpu,
            config_path=args.config,
        )

        results.append(result)
        if result["success"]:
            successful += 1
            
        # 实时写入CSV
        if args.save_results:
            append_result_to_csv(result, csv_filepath, fieldnames, i)
            
        # 实时显示最佳结果
        if successful > 0:
            current_best = max([r for r in results if r["success"]], key=lambda x: x["best_accuracy"])
            print(f"🏆 当前最佳: {current_best['exp_name']} - {current_best['best_accuracy']:.2f}%")

    # 总结
    print("=" * 60)
    print(f"📈 网格搜索完成！")
    print(f"✅ 成功实验数量: {successful}/{len(combinations)}")
    if skipped > 0:
        print(f"⏭️  跳过已完成实验: {skipped} 个")

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
        print(f"💾 结果已实时保存到: {csv_filepath}")

    return 0 if successful > 0 else 1


def main():
    """主函数：调度器始终单进程，不进入 Accelerate 环境"""
    args, _ = parse_arguments(mode="grid_search")
    return run_grid_search(args)


if __name__ == "__main__":
    sys.exit(main())