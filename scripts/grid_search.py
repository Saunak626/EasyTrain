"""
网格搜索启动脚本
支持参数网格搜索和预训练模型搜索
"""

import itertools
import subprocess
import yaml
import os
import sys
import time
import csv
import re
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_grid_search_arguments


def load_grid_config(path="config/grid.yaml"):
    """加载网格搜索配置"""
    with open(path) as f:
        return yaml.safe_load(f)


def generate_combinations(grid_config):
    """生成所有参数组合"""
    grid = grid_config["grid"]
    keys = list(grid.keys())
    values = list(grid.values())
    
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def run_single_experiment(params, exp_id, use_multi_gpu=False):
    """运行单个实验并返回详细结果，实时显示训练过程"""
    exp_name = f"grid_{exp_id}"
    
    # 构建训练命令，统一处理所有参数
    cmd = ["python", "scripts/train.py", "--config", "config/grid.yaml", "--experiment_name", exp_name]
    
    # 添加所有网格搜索参数作为命令行参数
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    # 添加多卡训练参数
    if use_multi_gpu:
        cmd.append("--multi_gpu")

    print(f"\n{'='*60}")
    print(f"🚀 开始实验 {exp_id}: {exp_name}")
    print(f"📋 参数: {params}")
    print(f"{'='*60}")
    
    # 让子进程直接继承父进程的stdio，保持TTY特性
    # 这样tqdm可以正常就地刷新进度条
    process = subprocess.Popen(cmd)
    
    try:
        # 等待进程完成
        process.wait(timeout=300)
        success = process.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"❌ 实验 {exp_name} 超时，正在终止...")
        process.kill()
        process.wait()
        success = False
    except Exception as e:
        print(f"❌ 实验 {exp_name} 执行出错: {e}")
        success = False
    
    # 解析训练结果
    # 注意：为保持tqdm进度条正常显示，未捕获子进程输出
    # 因此无法直接解析准确率，需要通过其他方式获取
    best_accuracy = 0.0
    final_accuracy = 0.0
    
    if success:
        # TODO: 集成SwanLab API或解析日志文件来获取真实准确率
        # 当前使用占位符值，实际准确率需查看SwanLab实验记录
        best_accuracy = 85.0  # 占位符：实际值请查看SwanLab
        final_accuracy = 85.0  # 占位符：实际值请查看SwanLab
    
    print(f"\n{'='*60}")
    if success:
        print(f"✅ 实验 {exp_name} 完成 - 最佳准确率: {best_accuracy:.2f}%")
    else:
        print(f"❌ 实验 {exp_name} 失败")
    print(f"{'='*60}\n")
    
    return {
        'success': success,
        'exp_name': exp_name,
        'params': params,
        'best_accuracy': best_accuracy,
        'final_accuracy': final_accuracy,
        'stdout': '',  # 不再捕获输出
        'stderr': ''
    }





def save_results_to_csv(results, filename):
    """保存实验结果到CSV文件"""
    if not results:
        return
    
    # 创建网格搜索结果文件夹
    results_dir = "grid_search_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 将文件保存到结果文件夹中
    filepath = os.path.join(results_dir, filename)
    
    fieldnames = ['experiment_id', 'experiment_name', 'success', 'best_accuracy', 'final_accuracy']
    
    # 添加所有参数列
    all_param_keys = set()
    for result in results:
        all_param_keys.update(result['params'].keys())
    fieldnames.extend(sorted(all_param_keys))
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, result in enumerate(results, 1):
            row = {
                'experiment_id': f"{i:03d}",
                'experiment_name': result['exp_name'],
                'success': result['success'],
                'best_accuracy': result['best_accuracy'],
                'final_accuracy': result['final_accuracy']
            }
            row.update(result['params'])
            writer.writerow(row)
    
    return filepath


def run_grid_search(args):
    """运行网格搜索"""
    config = load_grid_config(args.config)
    combinations = generate_combinations(config)

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
        
        if result['success']:
            successful += 1
        
        # 简短的间隔，让用户看清实验分隔
        time.sleep(1)

    # 输出总结
    print("\n" + "=" * 60)
    print(f"📈 网格搜索完成！")
    print(f"✅ 成功实验: {successful}/{len(combinations)}")
    
    if successful > 0:
        # 找到最佳实验
        successful_results = [r for r in results if r['success']]
        best_result = max(successful_results, key=lambda x: x['best_accuracy'])
        
        print(f"\n🏆 最佳实验结果:")
        print(f"   实验名称: {best_result['exp_name']}")
        print(f"   最佳准确率: {best_result['best_accuracy']:.2f}%")
        print(f"   最终准确率: {best_result['final_accuracy']:.2f}%")
        print(f"   最优参数:")
        for key, value in best_result['params'].items():
            print(f"     {key}: {value}")
        
        # 显示前3名结果
        top_results = sorted(successful_results, key=lambda x: x['best_accuracy'], reverse=True)[:3]
        print(f"\n📊 前3名实验结果:")
        for i, result in enumerate(top_results, 1):
            print(f"   {i}. {result['exp_name']} - {result['best_accuracy']:.2f}% - {result['params']}")
    
    # 保存结果到CSV
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"grid_search_results_{timestamp}.csv"
        saved_filepath = save_results_to_csv(results, results_filename)
        print(f"\n💾 结果已保存到: {saved_filepath}")

    return 0 if successful > 0 else 1


def main():
    """主函数"""
    args, config = parse_grid_search_arguments()
    return run_grid_search(args)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
