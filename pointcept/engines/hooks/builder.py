"""
Hook Builder

This module provides the registry and builder functionality for hooks in the
Pointcept framework. It enables dynamic registration and construction of hooks
for customizing training and testing workflows.

The hook builder uses a registry pattern to allow:
- Dynamic registration of new hook types
- Configuration-based hook instantiation
- Flexible hook composition
- Ordered hook execution

Features:
- Registry-based hook management
- Dynamic hook registration
- Configuration-based hook building
- Support for custom hooks
- Ordered hook construction
- Type checking and validation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry


HOOKS = Registry("hooks")


def build_hooks(cfg):
    """Build a list of hooks from configuration.
    
    This function constructs hook instances based on their configuration
    dictionaries. The hooks are built in the order they appear in the config,
    which determines their execution order in the training/testing workflow.
    
    Args:
        cfg (list[dict]): List of hook configurations. Each dict should contain:
            - type: The registered name of the hook (required)
            - args: Arguments specific to the hook type (optional)
            - priority: Execution priority (optional, default=NORMAL)
    
    Returns:
        list[HookBase]: List of constructed hook instances in execution order
    
    Example:
        >>> cfg = [
        ...     dict(type='CheckpointHook', interval=1000),
        ...     dict(type='IterationTimer'),
        ...     dict(type='InformationWriter')
        ... ]
        >>> hooks = build_hooks(cfg)
    
    Note:
        - Hooks are executed in the order they are built
        - Each hook must be registered in the HOOKS registry
        - Invalid hook configurations will raise exceptions
    """
    hooks = []
    for hook_cfg in cfg:
        hooks.append(HOOKS.build(hook_cfg))
    return hooks
