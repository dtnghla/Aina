"""
Scheduler Module

This module provides various learning rate schedulers for optimizing deep learning models.
Each scheduler adjusts the learning rate during training according to different strategies.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch.optim.lr_scheduler as lr_scheduler
from .registry import Registry

SCHEDULERS = Registry("schedulers")


@SCHEDULERS.register_module()
class MultiStepLR(lr_scheduler.MultiStepLR):
    """Multi-step learning rate scheduler with fixed decay at specified milestones.
    
    This scheduler decreases the learning rate by gamma at each milestone, where milestones
    are specified as fractions of the total training steps.
    
    Args:
        optimizer: Wrapped optimizer.
        milestones (list): List of milestone fractions where learning rate decay occurs.
        total_steps (int): Total number of training steps.
        gamma (float, optional): Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool, optional): If True, prints a message to stdout for each update. Default: False.
    """
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        last_epoch=-1,
        verbose=False,
    ):
        super().__init__(
            optimizer=optimizer,
            milestones=[rate * total_steps for rate in milestones],
            gamma=gamma,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class MultiStepWithWarmupLR(lr_scheduler.LambdaLR):
    """Multi-step learning rate scheduler with warmup phase.
    
    This scheduler starts with a warmup phase where the learning rate gradually increases,
    followed by step-wise decay at specified milestones.
    
    Args:
        optimizer: Wrapped optimizer.
        milestones (list): List of milestone fractions where learning rate decay occurs.
        total_steps (int): Total number of training steps.
        gamma (float, optional): Multiplicative factor of learning rate decay. Default: 0.1.
        warmup_rate (float, optional): Fraction of total steps used for warmup. Default: 0.05.
        warmup_scale (float, optional): Initial learning rate scale during warmup. Default: 1e-6.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool, optional): If True, prints a message to stdout for each update. Default: False.
    """
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        warmup_rate=0.05,
        warmup_scale=1e-6,
        last_epoch=-1,
        verbose=False,
    ):
        milestones = [rate * total_steps for rate in milestones]

        def multi_step_with_warmup(s):
            factor = 1.0
            for i in range(len(milestones)):
                if s < milestones[i]:
                    break
                factor *= gamma

            if s <= warmup_rate * total_steps:
                warmup_coefficient = 1 - (1 - s / warmup_rate / total_steps) * (
                    1 - warmup_scale
                )
            else:
                warmup_coefficient = 1.0
            return warmup_coefficient * factor

        super().__init__(
            optimizer=optimizer,
            lr_lambda=multi_step_with_warmup,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class PolyLR(lr_scheduler.LambdaLR):
    """Polynomial learning rate scheduler.
    
    Decays the learning rate polynomially based on the current step:
    lr = initial_lr * (1 - step/total_steps)^power
    
    Args:
        optimizer: Wrapped optimizer.
        total_steps (int): Total number of training steps.
        power (float, optional): Power of polynomial decay. Default: 0.9.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool, optional): If True, prints a message to stdout for each update. Default: False.
    """
    def __init__(self, optimizer, total_steps, power=0.9, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: (1 - s / (total_steps + 1)) ** power,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class ExpLR(lr_scheduler.LambdaLR):
    """Exponential learning rate scheduler.
    
    Decays the learning rate exponentially based on the current step:
    lr = initial_lr * gamma^(step/total_steps)
    
    Args:
        optimizer: Wrapped optimizer.
        total_steps (int): Total number of training steps.
        gamma (float, optional): Multiplicative factor of learning rate decay. Default: 0.9.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool, optional): If True, prints a message to stdout for each update. Default: False.
    """
    def __init__(self, optimizer, total_steps, gamma=0.9, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: gamma ** (s / total_steps),
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    """Cosine annealing learning rate scheduler.
    
    Decays the learning rate following a cosine curve from the initial lr to eta_min.
    
    Args:
        optimizer: Wrapped optimizer.
        total_steps (int): Total number of training steps.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool, optional): If True, prints a message to stdout for each update. Default: False.
    """
    def __init__(self, optimizer, total_steps, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class OneCycleLR(lr_scheduler.OneCycleLR):
    """One Cycle learning rate scheduler.
    
    Implementation of the One Cycle learning rate policy, which consists of two phases:
    1. Increasing learning rate from initial_lr to max_lr
    2. Decreasing learning rate from max_lr back to initial_lr
    
    Args:
        optimizer: Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundary in the cycle.
        total_steps (int): Total number of training steps.
        pct_start (float, optional): Percentage of cycle spent increasing lr. Default: 0.3.
        anneal_strategy (str, optional): {'cos', 'linear'}. Specifies the annealing strategy. Default: 'cos'.
        cycle_momentum (bool, optional): If True, momentum is cycled inversely to learning rate. Default: True.
        base_momentum (float, optional): Lower momentum boundary. Default: 0.85.
        max_momentum (float, optional): Upper momentum boundary. Default: 0.95.
        div_factor (float, optional): Determines the initial learning rate. Default: 25.0.
        final_div_factor (float, optional): Determines the minimum learning rate. Default: 1e4.
        three_phase (bool, optional): If True, use a third phase of the schedule to annihilate the learning rate. Default: False.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool, optional): If True, prints a message to stdout for each update. Default: False.
    """
    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
        verbose=False,
    ):
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
            verbose=verbose,
        )


def build_scheduler(cfg, optimizer):
    """Build a learning rate scheduler from config.
    
    Args:
        cfg: Configuration object containing scheduler parameters.
        optimizer: Optimizer to wrap with the scheduler.
    
    Returns:
        A learning rate scheduler instance.
    """
    cfg.optimizer = optimizer
    return SCHEDULERS.build(cfg=cfg)
