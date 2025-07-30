"""
网格搜索启动脚本
- 调度器保持单进程运行
- 每个实验独立用 accelerate 启动（多卡）或 python 启动（单卡/CPU）
- 为每次实验设置唯一 MASTER_PORT，清理分布式环境变量，避免进程间串扰
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
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments


# ----------------------------- 工具函数 -----------------------------

def load_grid_config(path="config/grid.yaml"):
    """加载网格搜索配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_list(v):
    """标量→单元素列表；列表/元组原样；None→空列表"""
    if v is None:
        return []
    return v if isinstance(v, (list, tuple)) else [v]

def generate_combinations(config):
    """
    只支持笛卡尔积：
      - grid_search.grid: dict[str, list|scalar]，标量会当作单元素列表
      - grid_search.fixed: dict，固定参数并入每个组合
    """
    gs = (config or {}).get("grid_search", {}) or {}
    fixed = gs.get("fixed", {}) or {}
    grid = gs.get("grid", {}) or {}

    if not grid:
        return [fixed] if fixed else [{}]

    # 过滤空列表的键，避免生成空组合
    valid_items = [(k, _as_list(v)) for k, v in grid.items() if _as_list(v)]
    
    if not valid_items:
        return [fixed] if fixed else [{}]

    keys, values_lists = zip(*valid_items)

    # 笛卡尔积生成组合，并合并 fixed
    return [{**fixed, **dict(zip(keys, combo))} 
            for combo in itertools.product(*values_lists)]


def parse_result_from_files(exp_name):
    """从结构化文件中解析最终结果（优先 result.json，回退 metrics.jsonl）"""
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
    """保存实验结果到CSV文件（统一保存到 runs/ 目录）"""
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


# ----------------------------- 多卡子进程启动辅助 -----------------------------

def _clean_env_for_child():
    """
    清理父进程里可能遗留的分布式环境变量，
    防止子训练误以为自己加入了某个现存的 DDP 组。
    """
    env = os.environ.copy()
    for k in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(k, None)
    return env


def _unique_master_port(base=20000, span=10000):
    """为每个实验分配唯一端口，避免端口复用导致 NCCL 连接异常"""
    return str(base + random.randint(0, span))


def _infer_num_procs() -> int:
    """根据 gpu_ids 或实际设备数推断进程数"""
    # if gpu_ids:
    #     return max(1, len([x for x in gpu_ids.split(",") if x.strip() != ""]))
    env_ids = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if env_ids:
        return max(1, len([x for x in env_ids.split(",") if x.strip() != ""]))
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


# ----------------------------- 核心逻辑 -----------------------------

def run_single_experiment(params, exp_id, use_multi_gpu=False, config_path="config/grid.yaml",
                        accelerate_args=""):
    """运行单个实验（每个实验独立的进程/进程组）"""
    exp_name = f"grid_{exp_id}"

    # 组装基础命令
    if use_multi_gpu:
        cmd = ["accelerate", "launch", "--multi_gpu", "--num_processes", str(_infer_num_procs())]
        if accelerate_args:
            cmd.extend(accelerate_args.split())
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
    rc = process.wait()
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
            config_path="config/grid.yaml",     # 训练使用的统一配置
            # gpu_ids=args.gpu_ids,
            accelerate_args=(args.accelerate_args or "")
        )
        results.append(result)
        if result["success"]:
            successful += 1

        # 适当间隔，便于观察分隔，也给系统时间释放端口/句柄
        time.sleep(0.5 if not args.multi_gpu else 1.0)

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
        print(f"   最优参数: {best_result['params']}")

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
    """主函数：调度器始终单进程，不进入 Accelerate 环境"""
    args, _ = parse_arguments(mode="grid_search")
    return run_grid_search(args)


if __name__ == "__main__":
    sys.exit(main())
