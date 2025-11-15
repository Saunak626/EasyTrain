"""
单实验训练启动脚本：
1. 单独运行单个训练实验。通过统一的配置解析器获取训练参数，支持命令行和配置文件
2. 作为网格搜索的子进程被调用

与grid_search.py的关系：
- train.py: 执行器，负责单个实验的具体执行
- grid_search.py: 调度器，负责多实验的协调和管理
"""

import sys
import os
import subprocess

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments  # 参数解析器
from src.trainers.base_trainer import run_training   # 核心训练函数


def _should_launch_with_accelerate(args, config):
    """判断是否需要通过Accelerate重新拉起进程"""
    multi_gpu_enabled = bool(args.multi_gpu or config.get('multi_gpu', {}).get('enabled'))
    already_launched = (
        os.environ.get("ACCELERATE_PROCESS_INDEX") is not None or
        os.environ.get("LOCAL_RANK") is not None
    )
    return multi_gpu_enabled and not already_launched

def main():
    """单实验训练主函数

    解析配置参数，执行训练，并可选地保存结果到文件。

    Returns:
        int: 退出码，0表示成功
    """
    args, config = parse_arguments(mode='single_experiment')

    if _should_launch_with_accelerate(args, config):
        gpu_ids = str(config.get('gpu', {}).get('device_ids', os.environ.get('CUDA_VISIBLE_DEVICES', '0')))
        visible_ids = [gid.strip() for gid in gpu_ids.split(',') if gid.strip()]
        num_gpus = max(1, len(visible_ids))

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(visible_ids)

        cmd = [
            "accelerate", "launch", "--multi_gpu", "--num_processes", str(num_gpus),
            os.path.abspath(__file__)
        ]
        cmd.extend(sys.argv[1:])

        return subprocess.call(cmd, env=env)

    exp_name = config['training']['exp_name']
    result = run_training(config, exp_name)

    # 保存结果到文件（用于网格搜索）
    if hasattr(args, 'result_file') and args.result_file:
        try:
            import json
            with open(args.result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"写入结果文件失败: {e}")
            return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
    
