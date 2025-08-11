"""统一组件管理模块

提供统一的组件注册、创建和管理功能。
"""

from .component_registry import (
    ComponentRegistry,
    COMPONENT_REGISTRY,
    create_component,
    get_supported_components
)

__all__ = [
    'ComponentRegistry',
    'COMPONENT_REGISTRY', 
    'create_component',
    'get_supported_components'
]
