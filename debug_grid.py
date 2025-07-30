#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from scripts.ngrid_search import load_config, generate_combinations, apply_params_to_config

def debug_grid_params():
    config = load_config('test_grid_params.yaml')
    combinations = generate_combinations(config)
    
    print(f"生成的参数组合数量: {len(combinations)}")
    print("="*60)
    
    for i, params in enumerate(combinations, 1):
        print(f"\n实验 {i}:")
        print(f"原始参数: {params}")
        
        # 应用参数到配置
        experiment_config = apply_params_to_config(config, params)
        
        print(f"应用后的模型类型: {experiment_config.get('model', {}).get('name', 'N/A')}")
        print(f"应用后的超参数: {experiment_config.get('hyperparameters', {})}")
        print("-"*40)

if __name__ == "__main__":
    debug_grid_params()