"""网格搜索启动脚本
支持参数网格搜索和预训练模型搜索（方案 1：子进程继承 TTY）
"""

import itertools
import subprocess
import yaml
import os
import sys
import time
import csv
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments

def is_accelerate_environment():
    """检测是否已在accelerate环境中"""
    return os.environ.get('ACCELERATE_USE_DEEPSPEED') is not None or \
           os.environ.get('LOCAL_RANK') is not None or \
           os.environ.get('WORLD_SIZE') is not None

def launch_with_accelerate():
    """使用accelerate launch重新启动当前脚本"""
    # 获取当前脚本的所有参数，但移除--multi_gpu
    current_args = [arg for arg in sys.argv[1:] if arg != '--multi_gpu']
    
    # 构建accelerate launch命令
    cmd = ['accelerate', 'launch', sys.argv[0]] + current_args
    
    print(f"🚀 启动多卡网格搜索: {' '.join(cmd)}")
    print("-" * 50)
    
    # 执行accelerate launch命令
    result = subprocess.run(cmd)
    return result.returncode


# ----------------------------- 工具函数 -----------------------------

def load_grid_config(path="config/grid.yaml"):
    """加载网格搜索配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_list(v):
    """把标量包装成单元素列表；列表则原样返回；None 则返回空列表"""
    if v is None:
        return []
    return v if isinstance(v, (list, tuple)) else [v]


def generate_combinations(config):
    """
    生成所有参数组合（更通用）
    支持三种写法（优先级从高到低）：
    1) grid_search.combinations: 直接给出组合列表（list[dict]），将被原样返回
    2) grid_search.grid: dict[str, list|scalar]，做笛卡尔积；标量会当作单元素列表
    3) 若两者都没有，返回 [{}]

    另外：
    - grid_search.fixed: dict，固定参数，会并入每个组合
    - 兼容 'model_type' → 'model_name'
    """
    gs = (config or {}).get("grid_search", {})
    fixed = gs.get("fixed", {}) or {}

    # 写法 1：直接枚举好的组合
    combos = gs.get("combinations")
    if isinstance(combos, list) and all(isinstance(x, dict) for x in combos) and combos:
        results = []
        for d in combos:
            merged = {**fixed, **d}
            if "model_type" in merged and "model_name" not in merged:
                merged["model_name"] = merged.pop("model_type")
            results.append(merged)
        return results

    # 写法 2：笛卡尔积
    grid = gs.get("grid", {})
    if grid:
        keys = list(grid.keys())
        values_lists = [ _as_list(grid[k]) for k in keys ]
        if any(len(v) == 0 for v in values_lists):
            print("警告: grid 中存在空列表，已跳过空值键。")
            keys = [k for k, vals in zip(keys, values_lists) if len(vals) > 0]
            values_lists = [vals for vals in values_lists if len(vals) > 0]

        combos = []
        for combo in itertools.product(*values_lists) if values_lists else [()]:
            param_set = dict(zip(keys, combo))
            param_set = {**fixed, **param_set}
            if "model_type" in param_set and "model_name" not in param_set:
                param_set["model_name"] = param_set.pop("model_type")
            combos.append(param_set)

        if not combos:
            print("警告: 网格为空，使用固定参数。")
            combos = [{**fixed}]
        return combos

    # 默认返回一个空组合（只有 fixed）
    return [{**fixed}] if fixed else [{}]


def parse_result_from_files(exp_name):
    """从结构化文件中解析最终结果（优先 result.json，回退 metrics.jsonl）"""
    result_dir = os.path.join("runs", exp_name)
    final_json = os.path.join(result_dir, "result.json")
    metrics_path = os.path.join(result_dir, "metrics.jsonl")

    best_accuracy = 0.0
    final_accuracy = 0.0

    # 优先读取 result.json
    if os.path.exists(final_json):
        try:
            with open(final_json, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                best_accuracy = float(data.get("best_accuracy", 0.0))
                final_accuracy = float(data.get("final_accuracy", best_accuracy))
                return best_accuracy, final_accuracy
        except Exception:
            pass

    # 回退：扫描 metrics.jsonl，取最大 val_acc 作为 best，最后一条 val_acc 作为 final
    if os.path.exists(metrics_path):
        try:
            last_val = None
            best_val = 0.0
            with open(metrics_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        va = rec.get("val_acc")
                        if isinstance(va, (int, float)):
                            last_val = float(va)
                            if last_val > best_val:
                                best_val = last_val
                    except Exception:
                        continue
            if last_val is not None:
                return best_val, last_val
        except Exception:
            pass

    return best_accuracy, final_accuracy


def save_results_to_csv(results, filename):
    """保存实验结果到CSV文件（统一保存到runs目录）"""
    if not results:
        return None

    # 统一使用runs目录，避免重复
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


# ----------------------------- 核心逻辑 -----------------------------

def run_single_experiment(params, exp_id, use_multi_gpu=False):
    """运行单个实验（子进程继承 TTY，不捕获输出）"""
    exp_name = f"grid_{exp_id}"

    # 构建训练命令（不再传 --is_grid_search，避免 base_trainer 禁用 tqdm）
    cmd = [
        sys.executable, "-u",
        "scripts/train.py",
        "--config", "config/unified.yaml",
        "--experiment_name", exp_name,
    ]

    # 添加参数
    for k, v in (params or {}).items():
        cmd.extend([f"--{k}", str(v)])

    # 注意：不再传递--multi_gpu参数给子进程
    # 因为如果需要多卡训练，父进程已经通过accelerate launch启动了

    print(f"\n{'='*60}")
    print(f"🚀 开始实验 {exp_id}: {exp_name}")
    print(f"📋 参数: {params}")
    print(f"{'='*60}")

    # 让子进程直接继承父进程终端（TTY），以便 tqdm 同行刷新
    process = subprocess.Popen(cmd)
    rc = process.wait()
    success = (rc == 0)

    # 解析训练结果（仅读文件）
    best_accuracy, final_accuracy = parse_result_from_files(exp_name)

    if success:
        if best_accuracy == 0.0 and final_accuracy == 0.0:
            print(f"⚠️  {exp_name} 结束，但未找到结果文件。请检查 runs/{exp_name}/ 是否生成 result.json 或 metrics.jsonl。")
        else:
            print(f"✅ 实验 {exp_name} 完成，最佳准确率: {best_accuracy:.2f}% | 最终: {final_accuracy:.2f}%")
    else:
        print(f"❌ 实验 {exp_name} 失败（返回码 {rc}）。建议查看 runs/{exp_name}/ 及控制台输出定位问题。")

    return {
        "success": success,
        "exp_name": exp_name,
        "params": params,
        "best_accuracy": best_accuracy,
        "final_accuracy": final_accuracy,
    }


def run_grid_search(args):
    """运行网格搜索"""
    config = load_grid_config(args.config)
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
        result = run_single_experiment(params, f"{i:03d}", args.multi_gpu)
        results.append(result)
        if result["success"]:
            successful += 1
        # 给出视觉分隔
        time.sleep(0.5)

    # 总结
    print("\n" + "=" * 60)
    print(f"📈 网格搜索完成！")
    print(f"✅ 成功实验: {successful}/{len(combinations)}")

    if successful > 0:
        successful_results = [r for r in results if r["success"]]
        best_result = max(successful_results, key=lambda x: x["best_accuracy"])

        print(f"\n🏆 最佳实验结果:")
        print(f"   实验名称: {best_result['exp_name']}")
        print(f"   最佳准确率: {best_result['best_accuracy']:.2f}%")
        print(f"   最终准确率: {best_result['final_accuracy']:.2f}%")
        print(f"   最优参数:")
        for key, value in best_result["params"].items():
            print(f"     {key}: {value}")

        top_results = sorted(successful_results, key=lambda x: x["best_accuracy"], reverse=True)[:3]
        print(f"\n📊 前3名实验结果:")
        for i, r in enumerate(top_results, 1):
            print(f"   {i}. {r['exp_name']} - {r['best_accuracy']:.2f}% - {r['params']}")

    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"grid_search_results_{timestamp}.csv"
        saved_filepath = save_results_to_csv(results, results_filename)
        if saved_filepath:
            print(f"\n💾 结果已保存到: {saved_filepath}")

    return 0 if successful > 0 else 1


def main():
    """主函数"""
    args, _ = parse_arguments(mode="grid_search")
    
    # 检查是否需要启动多卡训练
    if args.multi_gpu and not is_accelerate_environment():
        # 如果指定了多卡训练但不在accelerate环境中，重新启动
        return launch_with_accelerate()
    
    return run_grid_search(args)


if __name__ == "__main__":
    sys.exit(main())
