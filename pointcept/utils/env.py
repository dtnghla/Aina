"""
Environment Utils

This module provides utilities for managing the runtime environment in deep learning applications,
particularly focusing on reproducibility and random seed management across different libraries
and components.

Features:
- Random seed generation and management
- Consistent seed setting across multiple libraries (Python, NumPy, PyTorch)
- CUDA and CUDNN configuration for reproducibility
- Process-safe random seed generation

Dependencies:
- random: Python's built-in random number generator
- numpy: For numerical computations
- torch: PyTorch deep learning framework
- torch.backends.cudnn: For CUDA configuration
- datetime: For timestamp-based seed generation
- os: For system-level operations

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datetime import datetime


def get_random_seed():
    """
    Generate a random seed based on process ID, current time, and system entropy.
    
    This function creates a unique random seed by combining:
    - Process ID
    - Current microsecond timestamp
    - Random bytes from system entropy pool
    
    Returns:
        int: A random seed suitable for reproducibility purposes.
    """
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed


def set_seed(seed=None):
    """
    Set random seeds for reproducibility across multiple libraries.
    
    This function sets consistent random seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDNN backend
    - Python hash seed
    
    Args:
        seed (int, optional): Seed to use. If None, generates a random seed using get_random_seed().
    
    Note:
        Also sets cudnn.benchmark=False and cudnn.deterministic=True for reproducibility.
    """
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
