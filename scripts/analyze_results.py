#!/usr/bin/env python3
"""
网格搜索结果分析和可视化工具

功能：
1. 统一管理和分析多个网格搜索结果文件
2. 生成多标签分类的性能对比报告
3. 提供模型推荐和最佳配置建议
4. 创建可视化图表和热力图
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class GridSearchResultsAnalyzer:
    """网格搜索结果分析器"""
    
    def __init__(self, results_dir: str = "runs"):
        """初始化分析器
        
        Args:
            results_dir: 结果文件根目录
        """
        self.results_dir = Path(results_dir)
        self.all_results = []
        self.task_results = {}
        
    def discover_result_files(self) -> Dict[str, List[Path]]:
        """发现所有结果文件
        
        Returns:
            按任务类型分组的结果文件字典
        """
        task_files = {}
        
        # 遍历runs目录下的所有子目录
        for task_dir in self.results_dir.iterdir():
            if task_dir.is_dir():
                task_name = task_dir.name
                csv_files = list(task_dir.glob("grid_search_results_*.csv"))
                
                if csv_files:
                    task_files[task_name] = csv_files
                    print(f"📁 发现任务 '{task_name}': {len(csv_files)} 个结果文件")
        
        return task_files
    
    def load_all_results(self) -> None:
        """加载所有结果文件"""
        task_files = self.discover_result_files()
        
        for task_name, csv_files in task_files.items():
            task_results = []
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['task_type'] = task_name
                    df['result_file'] = csv_file.name
                    task_results.append(df)
                    print(f"✅ 加载 {csv_file.name}: {len(df)} 个实验")
                except Exception as e:
                    print(f"❌ 加载失败 {csv_file}: {e}")
            
            if task_results:
                combined_df = pd.concat(task_results, ignore_index=True)
                self.task_results[task_name] = combined_df
                self.all_results.append(combined_df)
        
        if self.all_results:
            self.combined_results = pd.concat(self.all_results, ignore_index=True)
            print(f"\n📊 总计加载 {len(self.combined_results)} 个实验结果")
        else:
            print("⚠️ 未找到任何结果文件")
    
    def analyze_multilabel_performance(self, task_name: str) -> Dict[str, Any]:
        """分析多标签分类性能
        
        Args:
            task_name: 任务名称
            
        Returns:
            分析结果字典
        """
        if task_name not in self.task_results:
            return {}
        
        df = self.task_results[task_name]
        
        # 过滤成功的实验
        successful_df = df[df['success'] == True].copy()
        
        if len(successful_df) == 0:
            return {"error": "没有成功的实验"}
        
        analysis = {
            "total_experiments": len(df),
            "successful_experiments": len(successful_df),
            "success_rate": len(successful_df) / len(df) * 100,
        }
        
        # 多标签指标分析
        multilabel_metrics = [
            'best_macro_accuracy', 'best_micro_accuracy', 'best_weighted_accuracy',
            'best_macro_f1', 'best_micro_f1', 'best_weighted_f1'
        ]
        
        for metric in multilabel_metrics:
            if metric in successful_df.columns:
                values = successful_df[metric].dropna()
                if len(values) > 0:
                    analysis[f"{metric}_stats"] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "median": float(values.median())
                    }
        
        # 找到最佳模型
        if 'best_weighted_f1' in successful_df.columns:
            # 过滤掉NaN值
            valid_f1_df = successful_df.dropna(subset=['best_weighted_f1'])
            if len(valid_f1_df) > 0:
                best_idx = valid_f1_df['best_weighted_f1'].idxmax()
                analysis["best_model"] = {
                    "exp_name": valid_f1_df.loc[best_idx, 'exp_name'],
                    "model_type": valid_f1_df.loc[best_idx, 'model.type'],
                    "weighted_f1": valid_f1_df.loc[best_idx, 'best_weighted_f1'],
                    "macro_accuracy": valid_f1_df.loc[best_idx, 'best_macro_accuracy'] if 'best_macro_accuracy' in valid_f1_df.columns else None,
                    "micro_accuracy": valid_f1_df.loc[best_idx, 'best_micro_accuracy'] if 'best_micro_accuracy' in valid_f1_df.columns else None,
                    "weighted_accuracy": valid_f1_df.loc[best_idx, 'best_weighted_accuracy'] if 'best_weighted_accuracy' in valid_f1_df.columns else None
                }
        
        return analysis
    
    def generate_performance_comparison(self, task_name: str, output_dir: str) -> None:
        """生成性能对比图表
        
        Args:
            task_name: 任务名称
            output_dir: 输出目录
        """
        if task_name not in self.task_results:
            return
        
        df = self.task_results[task_name]
        successful_df = df[df['success'] == True].copy()
        
        if len(successful_df) == 0:
            print(f"⚠️ 任务 {task_name} 没有成功的实验")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 模型性能对比图
        if 'model.type' in successful_df.columns:
            plt.figure(figsize=(12, 8))
            
            # 准备数据
            metrics_to_plot = ['best_macro_f1', 'best_micro_f1', 'best_weighted_f1']
            available_metrics = [m for m in metrics_to_plot if m in successful_df.columns]
            
            if available_metrics:
                model_performance = successful_df.groupby('model.type')[available_metrics].mean()
                
                # 绘制条形图
                ax = model_performance.plot(kind='bar', figsize=(12, 6))
                plt.title(f'{task_name} - 模型性能对比 (F1分数)', fontsize=14, fontweight='bold')
                plt.xlabel('模型类型', fontsize=12)
                plt.ylabel('F1分数', fontsize=12)
                plt.legend(['宏平均F1', '微平均F1', '加权平均F1'])
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # 保存图表
                plt.savefig(os.path.join(output_dir, f'{task_name}_model_f1_comparison.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. 准确率对比图
        accuracy_metrics = ['best_macro_accuracy', 'best_micro_accuracy', 'best_weighted_accuracy']
        available_acc_metrics = [m for m in accuracy_metrics if m in successful_df.columns]
        
        if available_acc_metrics and 'model.type' in successful_df.columns:
            plt.figure(figsize=(12, 6))
            
            model_accuracy = successful_df.groupby('model.type')[available_acc_metrics].mean()
            
            ax = model_accuracy.plot(kind='bar', figsize=(12, 6))
            plt.title(f'{task_name} - 模型准确率对比', fontsize=14, fontweight='bold')
            plt.xlabel('模型类型', fontsize=12)
            plt.ylabel('准确率', fontsize=12)
            plt.legend(['宏平均准确率', '微平均准确率', '加权平均准确率'])
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f'{task_name}_model_accuracy_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 性能对比图表已保存到: {output_dir}")
    
    def generate_summary_report(self, output_file: str) -> None:
        """生成汇总报告
        
        Args:
            output_file: 输出文件路径
        """
        report = {
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "total_tasks": len(self.task_results),
            "tasks": {}
        }
        
        for task_name in self.task_results.keys():
            analysis = self.analyze_multilabel_performance(task_name)
            report["tasks"][task_name] = analysis
        
        # 保存JSON报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📋 汇总报告已保存到: {output_file}")
        
        # 打印简要报告
        print("\n" + "="*60)
        print("📊 网格搜索结果汇总报告")
        print("="*60)
        
        for task_name, analysis in report["tasks"].items():
            if "error" not in analysis:
                print(f"\n🎯 任务: {task_name}")
                print(f"  总实验数: {analysis['total_experiments']}")
                print(f"  成功实验数: {analysis['successful_experiments']}")
                print(f"  成功率: {analysis['success_rate']:.1f}%")
                
                if "best_model" in analysis:
                    best = analysis["best_model"]
                    print(f"  🏆 最佳模型: {best['model_type']}")
                    print(f"    加权F1: {best['weighted_f1']:.4f}")
                    print(f"    加权准确率: {best['weighted_accuracy']:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="网格搜索结果分析工具")
    parser.add_argument("--results_dir", type=str, default="runs", 
                       help="结果文件根目录")
    parser.add_argument("--output_dir", type=str, default="analysis_output", 
                       help="分析结果输出目录")
    parser.add_argument("--task", type=str, default=None, 
                       help="指定分析的任务类型，不指定则分析所有任务")
    
    args = parser.parse_args()
    
    print("🔍 开始分析网格搜索结果...")
    
    # 创建分析器
    analyzer = GridSearchResultsAnalyzer(args.results_dir)
    
    # 加载所有结果
    analyzer.load_all_results()
    
    if not analyzer.task_results:
        print("❌ 未找到任何结果文件")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成汇总报告
    report_file = os.path.join(args.output_dir, "grid_search_summary_report.json")
    analyzer.generate_summary_report(report_file)
    
    # 为每个任务生成性能对比图表
    if args.task:
        if args.task in analyzer.task_results:
            analyzer.generate_performance_comparison(args.task, args.output_dir)
        else:
            print(f"❌ 未找到任务 '{args.task}' 的结果")
    else:
        for task_name in analyzer.task_results.keys():
            analyzer.generate_performance_comparison(task_name, args.output_dir)
    
    print(f"\n✅ 分析完成！结果保存在: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
