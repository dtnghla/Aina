"""
Data Cache Utils

This module provides utilities for creating and managing shared memory arrays and dictionaries.
It uses SharedArray and ShareableList to enable efficient data sharing between processes,
which is particularly useful for multiprocessing scenarios in deep learning applications.

Features:
- Shared memory array creation and management
- Dictionary-based shared memory management
- Process-safe data sharing
- Automatic memory cleanup

Dependencies:
- SharedArray: For shared memory array management
- multiprocessing.shared_memory: For ShareableList (Python >= 3.8)
- numpy: For array operations

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import SharedArray

try:
    from multiprocessing.shared_memory import ShareableList
except ImportError:
    import warnings
    warnings.warn("Please update python version >= 3.8 to enable shared_memory")
import numpy as np


def shared_array(name, var=None):
    """Create or attach to a shared memory array.
    
    This function either creates a new shared memory array initialized with the content
    of var, or attaches to an existing shared memory array with the given name.
    The shared array is stored in the system's shared memory (/dev/shm/).
    
    Args:
        name (str): Name of the shared memory array. Will be prefixed with 'shm://'
            when creating the shared memory segment.
        var (numpy.ndarray, optional): Array to share. If provided, creates a new shared
            array with var's content. If None, attaches to an existing shared array.
    
    Returns:
        numpy.ndarray: Shared memory array. If var is provided, creates a new shared array
            with var's content and returns it. If var is None, returns a copy of the
            existing shared array.
    
    Example:
        >>> # Create a shared array
        >>> x = np.array([1, 2, 3])
        >>> shared_x = shared_array('my_array', x)
        >>> # Later, in another process
        >>> y = shared_array('my_array')  # Retrieves the shared array
    
    Notes:
        - The shared array is made read-only after creation to prevent accidental modifications
        - When attaching to an existing array, a copy is returned to ensure memory safety
        - The shared memory persists until explicitly deleted or system reboot
    """
    if var is not None:
        # check exist
        if os.path.exists(f"/dev/shm/{name}"):
            return SharedArray.attach(f"shm://{name}")
        # create shared_array
        data = SharedArray.create(f"shm://{name}", var.shape, dtype=var.dtype)
        data[...] = var[...]
        data.flags.writeable = False
    else:
        data = SharedArray.attach(f"shm://{name}").copy()
    return data


def shared_dict(name, var=None):
    """Create or attach to a shared memory dictionary of arrays.
    
    This function enables sharing a dictionary of numpy arrays between processes using
    shared memory. It uses ShareableList to store dictionary keys and creates individual
    shared memory arrays for each numpy array in the dictionary.
    
    Args:
        name (str): Name of the shared memory dictionary. Must not contain periods ('.')
            as they are used as separators in the internal naming scheme.
        var (dict, optional): Dictionary of arrays to share. If provided, creates new shared
            arrays for each numpy array in var. If None, attaches to existing dictionary.
    
    Returns:
        dict: Dictionary containing shared memory arrays. If var is provided, creates new
            shared arrays for each numpy array in var. If var is None, returns a dictionary
            with copies of existing shared arrays.
    
    Example:
        >>> # Create a shared dictionary
        >>> d = {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}
        >>> shared_d = shared_dict('my_dict', d)
        >>> # Later, in another process
        >>> d2 = shared_dict('my_dict')  # Retrieves the shared dictionary
    
    Notes:
        - Only numpy arrays in the dictionary are shared; other types are ignored
        - Dictionary keys are stored using ShareableList (requires Python >= 3.8)
        - Each array is stored as a separate shared memory segment
        - The name parameter must not contain periods
        - The shared memory persists until explicitly deleted or system reboot
    """
    name = str(name)
    assert "." not in name  # '.' is used as sep flag
    data = {}
    if var is not None:
        assert isinstance(var, dict)
        keys = var.keys()
        # current version only cache np.array
        keys_valid = []
        for key in keys:
            if isinstance(var[key], np.ndarray):
                keys_valid.append(key)
        keys = keys_valid

        ShareableList(sequence=keys, name=name + ".keys")
        for key in keys:
            if isinstance(var[key], np.ndarray):
                data[key] = shared_array(name=f"{name}.{key}", var=var[key])
    else:
        keys = list(ShareableList(name=name + ".keys"))
        for key in keys:
            data[key] = shared_array(name=f"{name}.{key}")
    return data
