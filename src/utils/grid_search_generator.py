"""网格搜索参数组合生成器

负责处理网格搜索的参数组合生成逻辑，支持分组式配置和模型-batch_size智能配对。
"""

import itertools
from typing import Dict, List, Any, Tuple

# ======================
# 模块级常量
# ======================

# 参数键名常量
MODEL_TYPE_KEY = 'model.type'
BATCH_SIZE_KEY = 'hp.batch_size'
GROUP_KEY = 'group'

# 参数组合生成时排除的参数（这些参数会单独处理）
EXCLUDED_PARAMS = [MODEL_TYPE_KEY, BATCH_SIZE_KEY]


# ======================
# 参数组合生成器类
# ======================

class ParameterCombinationGenerator:
    """参数组合生成器
    
    负责处理网格搜索的参数组合生成逻辑，支持分组式配置和模型-batch_size智能配对。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化参数组合生成器

        Args:
            config: 网格搜索配置字典
        """
        self.config = config
    
    def generate_combinations(self) -> List[Dict[str, Any]]:
        """生成参数组合的主入口函数
        
        Returns:
            参数组合列表，每个字典代表一组实验参数
        """
        gs = (self.config or {}).get("grid_search", {}) or {}
        fixed = gs.get("fixed", {}) or {}
        models_to_train = self.config.get("models_to_train", [])
        
        # 分组式配置处理
        if "groups" in gs and gs["groups"]:
            print(f"📋 使用分组式网格搜索配置")
            return self._generate_combinations_by_groups(gs["groups"], fixed, models_to_train)
        
        # 边界情况：无搜索参数，从基础配置中提取信息
        else:
            print(f"⚠️  未找到groups配置，从基础配置中提取参数")
            base_params = {}
            
            # 从基础配置中提取模型类型
            if 'model' in self.config and 'type' in self.config['model']:
                base_params[MODEL_TYPE_KEY] = self.config['model']['type']

            # 从基础配置中提取其他参数
            if 'optimizer' in self.config and 'name' in self.config['optimizer']:
                base_params['optimizer.name'] = self.config['optimizer']['name']

            if 'scheduler' in self.config and 'name' in self.config['scheduler']:
                base_params['scheduler.name'] = self.config['scheduler']['name']

            if 'loss' in self.config and 'name' in self.config['loss']:
                base_params['loss.name'] = self.config['loss']['name']

            # 从hp中提取batch_size
            if 'hp' in self.config and 'batch_size' in self.config['hp']:
                base_params[BATCH_SIZE_KEY] = self.config['hp']['batch_size']
            
            # 合并固定参数
            base_params.update(fixed)
            
            return [base_params] if base_params else []
    
    def _generate_combinations_by_groups(self, groups: Dict[str, Any], fixed: Dict[str, Any],
                                        models_to_train: List[str]) -> List[Dict[str, Any]]:
        """按分组生成参数组合
        
        Args:
            groups: 分组配置字典
            fixed: 固定参数字典
            models_to_train: 要训练的模型列表（用于过滤）
        
        Returns:
            参数组合列表
        """
        all_combinations = []
        
        for group_name, group_config in groups.items():
            print(f"\n🔍 处理分组: {group_name}")
            group_combinations = self._generate_group_combinations(group_name, group_config, fixed, models_to_train)
            all_combinations.extend(group_combinations)
            print(f"   ✅ 生成 {len(group_combinations)} 个组合")
        
        print(f"\n📊 总共生成 {len(all_combinations)} 个参数组合")
        return all_combinations
    
    def _generate_group_combinations(self, group_name: str, group_config: Dict[str, Any],
                                    fixed: Dict[str, Any], models_to_train: List[str]) -> List[Dict[str, Any]]:
        """生成单个分组的参数组合
        
        Args:
            group_name: 分组名称
            group_config: 分组配置
            fixed: 固定参数
            models_to_train: 要训练的模型列表

        Returns:
            该分组的参数组合列表
        """
        model_type_key = MODEL_TYPE_KEY
        batch_size_key = BATCH_SIZE_KEY
        
        # 提取模型列表和batch_size列表
        models = group_config.get(model_type_key, [])
        batch_sizes = group_config.get(batch_size_key, [])

        # 确保models和batch_sizes是列表
        if not isinstance(models, list):
            models = [models]
        if not isinstance(batch_sizes, list):
            batch_sizes = [batch_sizes]

        # 过滤模型（如果指定了models_to_train）
        if models_to_train:
            original_models = models
            models = [m for m in models if m in models_to_train]
            if len(models) < len(original_models):
                filtered_out = [m for m in original_models if m not in models]
                print(f"   🔧 过滤模型: {filtered_out} (不在models_to_train中)")

        # 如果没有模型，返回空列表
        if not models:
            print(f"   ⚠️  分组 {group_name} 没有可用模型，跳过")
            return []

        # 如果没有batch_size，使用默认值
        if not batch_sizes:
            batch_sizes = [32]
            print(f"   ⚠️  未指定batch_size，使用默认值: {batch_sizes}")

        # 智能配对模型和batch_size
        print(f"   🎯 模型列表: {models}")
        print(f"   🎯 batch_size列表: {batch_sizes}")
        model_batch_pairs = self._pair_models_with_batch_sizes(models, batch_sizes)

        # 提取其他参数（排除model.type和hp.batch_size）
        other_params = {}
        for key, value in group_config.items():
            if key not in EXCLUDED_PARAMS:
                other_params[key] = value

        # 生成其他参数的笛卡尔积
        if other_params:
            # 确保所有值都是列表
            for key in other_params:
                if not isinstance(other_params[key], list):
                    other_params[key] = [other_params[key]]

            # 生成笛卡尔积
            keys = list(other_params.keys())
            values = [other_params[k] for k in keys]
            other_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        else:
            other_combinations = [{}]

        # 组合：(模型, batch_size) × 其他参数
        group_combinations = []
        for (model, batch_size), other_combo in itertools.product(model_batch_pairs, other_combinations):
            combo = {
                model_type_key: model,
                batch_size_key: batch_size,
                GROUP_KEY: group_name
            }
            combo.update(other_combo)
            combo.update(fixed)
            group_combinations.append(combo)

        return group_combinations

    def _pair_models_with_batch_sizes(self, models: List[str], batch_sizes: List[int]) -> List[Tuple[str, int]]:
        """智能配对模型和batch_size

        Args:
            models: 模型列表
            batch_sizes: batch_size列表

        Returns:
            (模型, batch_size) 配对列表
        """
        if len(batch_sizes) == 1:
            # 情况1：只有一个batch_size，所有模型使用相同的batch_size
            return [(m, batch_sizes[0]) for m in models]

        elif len(batch_sizes) == len(models):
            # 情况2：batch_size数量与模型数量相同，按顺序配对
            print(f"   ✅ 长度匹配，将按顺序配对")
            return list(zip(models, batch_sizes))

        else:
            # 情况3：长度不匹配，扩充batch_size列表
            print(f"   🔄 扩充batch_size: {batch_sizes} (扩充到与model.type长度一致)")
            expanded_batch_sizes = []
            for i in range(len(models)):
                expanded_batch_sizes.append(batch_sizes[i % len(batch_sizes)])
            print(f"   🔄 扩充后: {expanded_batch_sizes}")

            # 打印配对结果
            pairs = list(zip(models, expanded_batch_sizes))
            print(f"   🎯 启用的模型配对: {dict(pairs)}")
            return pairs


# ======================
# 便捷函数
# ======================

def generate_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """生成参数组合的便捷函数

    Args:
        config: 网格搜索配置字典

    Returns:
        参数组合列表
    """
    generator = ParameterCombinationGenerator(config)
    return generator.generate_combinations()

