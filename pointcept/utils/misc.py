"""
Misc Utils

This module provides various utility functions and classes for common operations
in the Pointcept framework, including metrics calculation, directory management,
and module importing.

Features:
- Average value tracking with AverageMeter
- Intersection and union calculations for segmentation metrics
- Directory creation utilities
- Network port management
- Type checking and validation
- Dynamic module importing
- Placeholder class for testing

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import warnings
from collections import abc
import numpy as np
import torch
from importlib import import_module


class AverageMeter(object):
    """Computes and stores the average and current value of metrics.
    
    This class provides a simple way to track the running average of a value.
    It's particularly useful for monitoring training metrics like loss values
    or accuracy.
    
    Attributes:
        val (float): Last value added
        avg (float): Running average of all values
        sum (float): Sum of all values
        count (int): Number of values added
    
    Example:
        >>> meter = AverageMeter()
        >>> meter.update(2.0)
        >>> meter.update(4.0)
        >>> print(meter.avg)  # Output: 3.0
    """

    def __init__(self):
        """Initialize the meter with zero values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset all values to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the meter with a new value.
        
        Args:
            val (float): Value to add to the meter
            n (int, optional): Number of items this value represents.
                Default: 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_and_union(output, target, K, ignore_index=-1):
    """Calculate intersection and union for segmentation metrics.
    
    This function computes the intersection and union between predicted and
    target segmentation masks for K classes. It supports various input shapes
    and handles ignored labels.
    
    Args:
        output (numpy.ndarray): Prediction array of shape N, N*L, or N*H*W
        target (numpy.ndarray): Ground truth array of same shape as output
        K (int): Number of classes
        ignore_index (int, optional): Label to ignore in calculations.
            Default: -1.
    
    Returns:
        tuple: (area_intersection, area_union, area_target) where each is
            a numpy array of length K containing the respective areas.
    
    Note:
        All input values should be in range [0, K-1], except for ignored labels.
    """
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    """GPU version of intersection and union calculation.
    
    This function provides the same functionality as intersection_and_union
    but operates on GPU tensors for improved performance.
    
    Args:
        output (torch.Tensor): Prediction tensor of shape N, N*L, or N*H*W
        target (torch.Tensor): Ground truth tensor of same shape as output
        k (int): Number of classes
        ignore_index (int, optional): Label to ignore in calculations.
            Default: -1.
    
    Returns:
        tuple: (area_intersection, area_union, area_target) where each is
            a torch.Tensor of length k containing the respective areas.
    
    Note:
        All input values should be in range [0, k-1], except for ignored labels.
    """
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def make_dirs(dir_name):
    """Create a directory if it doesn't exist.
    
    This is a wrapper around os.makedirs that handles the case where the
    directory already exists.
    
    Args:
        dir_name (str): Path to the directory to create
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def find_free_port():
    """Find a free port on the local machine.
    
    This function attempts to find an available network port by binding to
    port 0 and letting the OS assign a free port number.
    
    Returns:
        int: An available port number
    
    Note:
        There is a small chance the port could be taken by another process
        between the time it's found and when it's actually used.
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def is_seq_of(seq, expected_type, seq_type=None):
    """Check if an object is a sequence of a specific type.
    
    This function verifies that an object is a sequence (list, tuple, etc.)
    and that all its elements are of the expected type.
    
    Args:
        seq: The sequence to check
        expected_type (type): Expected type of the sequence elements
        seq_type (type, optional): Expected type of the sequence itself.
            If None, any sequence type is accepted. Default: None.
    
    Returns:
        bool: True if seq is a sequence of expected_type elements,
            False otherwise.
    
    Example:
        >>> is_seq_of([1, 2, 3], int)  # True
        >>> is_seq_of([1, '2', 3], int)  # False
        >>> is_seq_of([1, 2, 3], int, tuple)  # False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):
    """Check if an object is a string.
    
    This is a simple wrapper around isinstance(x, str) for compatibility
    and clarity.
    
    Args:
        x: Object to check
    
    Returns:
        bool: True if x is a string, False otherwise
    
    Note:
        This method is kept for backward compatibility, though it's less
        relevant now that Python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from their string names.
    
    This function provides a way to dynamically import modules using their
    string names. It supports both single module imports and lists of modules.
    
    Args:
        imports (str or list): Module or list of modules to import
        allow_failed_imports (bool, optional): If True, failed imports return
            None instead of raising ImportError. Default: False.
    
    Returns:
        module or list: Imported module(s) or None for failed imports when
            allow_failed_imports is True.
    
    Raises:
        TypeError: If imports is not a string or list
        ImportError: If a module fails to import and allow_failed_imports
            is False
    
    Example:
        >>> os_path = import_modules_from_strings('os.path')
        >>> os_path.join('a', 'b')  # '/a/b'
        >>> modules = import_modules_from_strings(['os', 'sys'])
        >>> isinstance(modules, list)  # True
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class DummyClass:
    """Empty class for testing and placeholder purposes.
    
    This class serves as a minimal placeholder when a class instance is
    needed but no actual functionality is required.
    """
    
    def __init__(self):
        """Initialize an empty DummyClass instance."""
        pass
