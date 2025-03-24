"""
Optimizer Module

This module provides functionality for building and configuring optimizers for model training.
It supports various PyTorch optimizers and allows for flexible parameter group configuration.

Features:
- Registry-based optimizer management
- Support for parameter groups with different learning rates
- Built-in support for SGD, Adam, and AdamW optimizers
- Flexible parameter group configuration based on parameter name matching

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry

OPTIMIZERS = Registry("optimizers")

# Register PyTorch optimizers
OPTIMIZERS.register_module(
    module=torch.optim.SGD,
    name="SGD")

OPTIMIZERS.register_module(
    module=torch.optim.Adam,
    name="Adam")

OPTIMIZERS.register_module(
    module=torch.optim.AdamW,
    name="AdamW")


def build_optimizer(cfg, model, param_dicts=None):
    """Build an optimizer for model parameters with support for parameter groups.

    This function creates an optimizer instance based on the provided configuration and model.
    It supports creating parameter groups with different learning rates, momentum, and weight decay
    based on parameter name matching.

    Args:
        cfg: Configuration object containing optimizer settings with the following fields:
            - type: Optimizer type (e.g., 'SGD', 'Adam', 'AdamW')
            - lr: Base learning rate
            - Other optimizer-specific parameters (e.g., momentum, weight_decay)
        model (nn.Module): Model whose parameters need to be optimized.
        param_dicts (list, optional): List of parameter group dictionaries. Each dict can contain:
            - keyword: String to match parameter names
            - lr: Learning rate for the group
            - momentum: Momentum for the group
            - weight_decay: Weight decay for the group

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.

    Example:
        >>> cfg = Config(type='Adam', lr=0.001)
        >>> param_dicts = [
        ...     dict(keyword='backbone', lr=0.0001),
        ...     dict(keyword='head', lr=0.001)
        ... ]
        >>> optimizer = build_optimizer(cfg, model, param_dicts)

    Notes:
        - If param_dicts is None, all model parameters use the same settings from cfg.
        - Parameters matching no keywords go to the default group (group 0).
        - Parameters matching a keyword go to the corresponding group.
        - The function logs parameter grouping information through the root logger.
    """
    if param_dicts is None:
        cfg.params = model.parameters()
    else:
        cfg.params = [dict(names=[], params=[], lr=cfg.lr)]
        for i in range(len(param_dicts)):
            param_group = dict(names=[], params=[])
            if "lr" in param_dicts[i].keys():
                param_group["lr"] = param_dicts[i].lr
            if "momentum" in param_dicts[i].keys():
                param_group["momentum"] = param_dicts[i].momentum
            if "weight_decay" in param_dicts[i].keys():
                param_group["weight_decay"] = param_dicts[i].weight_decay
            cfg.params.append(param_group)

        for n, p in model.named_parameters():
            flag = False
            for i in range(len(param_dicts)):
                if param_dicts[i].keyword in n:
                    cfg.params[i + 1]["names"].append(n)
                    cfg.params[i + 1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                cfg.params[0]["names"].append(n)
                cfg.params[0]["params"].append(p)

        logger = get_root_logger()
        for i in range(len(cfg.params)):
            param_names = cfg.params[i].pop("names")
            message = ""
            for key in cfg.params[i].keys():
                if key != "params":
                    message += f" {key}: {cfg.params[i][key]};"
            logger.info(f"Params Group {i+1} -{message} Params: {param_names}.")
    return OPTIMIZERS.build(cfg=cfg)
