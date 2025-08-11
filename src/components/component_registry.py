"""统一组件注册表

提供统一的组件注册、创建和管理功能，整合损失函数、优化器、调度器、模型、数据集的管理。
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR, ExponentialLR
import timm
from torchvision import models

# 导入项目内部组件
from src.losses.image_loss import FocalLoss
from src.models.model_registry import create_model_unified
from src.datasets.dataloader_factory import create_dataloaders


class ComponentRegistry:
    """统一的组件注册表"""
    
    def __init__(self):
        self.registry = {
            'losses': {},
            'optimizers': {},
            'schedulers': {},
            'models': {},
            'datasets': {}
        }
        self._initialize_default_components()
    
    def _initialize_default_components(self):
        """初始化默认组件"""
        # 注册损失函数
        self.register_loss('crossentropy', nn.CrossEntropyLoss, {
            'weight': None,
            'ignore_index': -100,
            'reduction': 'mean',
            'label_smoothing': 0.0
        })
        
        self.register_loss('focal', FocalLoss, {
            'alpha': 1.0,
            'gamma': 2.0,
            'reduction': 'mean'
        })
        
        self.register_loss('mse', nn.MSELoss, {
            'reduction': 'mean'
        })
        
        # 注册优化器
        self.register_optimizer('adam', Adam, {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0,
            'amsgrad': False
        })
        
        self.register_optimizer('sgd', SGD, {
            'lr': 0.01,
            'momentum': 0,
            'dampening': 0,
            'weight_decay': 0,
            'nesterov': False
        })
        
        self.register_optimizer('adamw', AdamW, {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0.01,
            'amsgrad': False
        })
        
        # 注册调度器
        self.register_scheduler('cosine', CosineAnnealingLR, {
            'T_max': 50,
            'eta_min': 0,
            'last_epoch': -1
        })
        
        self.register_scheduler('onecycle', OneCycleLR, {
            'max_lr': 0.1,
            'total_steps': None,
            'epochs': None,
            'steps_per_epoch': None,
            'pct_start': 0.3,
            'anneal_strategy': 'cos',
            'cycle_momentum': True,
            'base_momentum': 0.85,
            'max_momentum': 0.95,
            'div_factor': 25.0,
            'final_div_factor': 10000.0,
            'three_phase': False,
            'last_epoch': -1
        })
        
        self.register_scheduler('step', StepLR, {
            'step_size': 30,
            'gamma': 0.1,
            'last_epoch': -1
        })
        
        self.register_scheduler('exponential', ExponentialLR, {
            'gamma': 0.95,
            'last_epoch': -1
        })
    
    def register_loss(self, name, loss_class, default_params=None):
        """注册损失函数"""
        self.registry['losses'][name] = {
            'class': loss_class,
            'default_params': default_params or {},
            'type': 'loss'
        }
    
    def register_optimizer(self, name, optimizer_class, default_params=None):
        """注册优化器"""
        self.registry['optimizers'][name] = {
            'class': optimizer_class,
            'default_params': default_params or {},
            'type': 'optimizer'
        }
    
    def register_scheduler(self, name, scheduler_class, default_params=None):
        """注册调度器"""
        self.registry['schedulers'][name] = {
            'class': scheduler_class,
            'default_params': default_params or {},
            'type': 'scheduler'
        }
    
    def register_model(self, name, model_info):
        """注册模型"""
        self.registry['models'][name] = model_info
    
    def register_dataset(self, name, dataset_info):
        """注册数据集"""
        self.registry['datasets'][name] = dataset_info
    
    def create_component(self, component_type, component_name, **kwargs):
        """统一的组件创建接口
        
        Args:
            component_type (str): 组件类型 ('losses', 'optimizers', 'schedulers', 'models', 'datasets')
            component_name (str): 组件名称
            **kwargs: 组件参数
            
        Returns:
            创建的组件实例
        """
        if component_type not in self.registry:
            raise ValueError(f"不支持的组件类型: {component_type}")
        
        if component_name not in self.registry[component_type]:
            supported = list(self.registry[component_type].keys())
            raise ValueError(f"不支持的{component_type}: {component_name}。支持的选项: {supported}")
        
        component_info = self.registry[component_type][component_name]
        
        # 合并默认参数和用户参数
        params = component_info['default_params'].copy()
        params.update(kwargs)
        
        # 特殊处理不同类型的组件
        if component_type == 'models':
            # 模型创建使用现有的统一接口
            return create_model_unified(component_name, **params)
        elif component_type == 'datasets':
            # 数据集创建使用现有的工厂函数
            return create_dataloaders(component_name, **params)
        else:
            # 其他组件直接实例化
            return component_info['class'](**params)
    
    def create_loss(self, loss_name, **kwargs):
        """创建损失函数"""
        return self.create_component('losses', loss_name, **kwargs)
    
    def create_optimizer(self, optimizer_name, model_parameters, learning_rate, **kwargs):
        """创建优化器

        Args:
            optimizer_name (str): 优化器名称
            model_parameters: 模型参数
            learning_rate (float): 学习率
            **kwargs: 其他优化器参数
        """
        if optimizer_name not in self.registry['optimizers']:
            supported = list(self.registry['optimizers'].keys())
            raise ValueError(f"不支持的优化器: {optimizer_name}。支持的选项: {supported}")

        component_info = self.registry['optimizers'][optimizer_name]

        # 合并默认参数和用户参数
        params = component_info['default_params'].copy()
        params.update(kwargs)
        params['lr'] = learning_rate

        # 优化器需要模型参数作为第一个参数
        return component_info['class'](model_parameters, **params)
    
    def create_scheduler(self, scheduler_name, optimizer, **kwargs):
        """创建学习率调度器
        
        Args:
            scheduler_name (str): 调度器名称
            optimizer: 优化器实例
            **kwargs: 调度器参数
        """
        scheduler_class = self.registry['schedulers'][scheduler_name]['class']
        params = self.registry['schedulers'][scheduler_name]['default_params'].copy()
        params.update(kwargs)
        
        return scheduler_class(optimizer, **params)
    
    def get_supported_components(self, component_type):
        """获取支持的组件列表"""
        if component_type not in self.registry:
            return []
        return list(self.registry[component_type].keys())
    
    def get_component_info(self, component_type, component_name):
        """获取组件信息"""
        if component_type not in self.registry:
            raise ValueError(f"不支持的组件类型: {component_type}")
        
        if component_name not in self.registry[component_type]:
            raise ValueError(f"不支持的{component_type}: {component_name}")
        
        return self.registry[component_type][component_name]
    
    def validate_component_params(self, component_type, component_name, params):
        """验证组件参数"""
        component_info = self.get_component_info(component_type, component_name)
        default_params = component_info['default_params']
        
        # 检查未知参数
        unknown_params = set(params.keys()) - set(default_params.keys())
        if unknown_params:
            print(f"警告: {component_type}.{component_name} 包含未知参数: {unknown_params}")
            print(f"支持的参数: {list(default_params.keys())}")
        
        return True


# 全局组件注册表实例
COMPONENT_REGISTRY = ComponentRegistry()


def create_component(component_type, component_name, **kwargs):
    """全局组件创建函数"""
    return COMPONENT_REGISTRY.create_component(component_type, component_name, **kwargs)


def get_supported_components(component_type):
    """获取支持的组件列表"""
    return COMPONENT_REGISTRY.get_supported_components(component_type)
