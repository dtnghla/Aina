# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
Modified from detectron2(https://github.com/facebookresearch/detectron2)

Copyright (c) Xiaoyang Wu (xiaoyang.wu@connect.hku.hk). All Rights Reserved.
Please cite our work if you use any part of the code.
"""

import functools
import numpy as np
import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""


def get_world_size() -> int:
    """
    Get the size of the world (total number of distributed processes).

    Returns:
        int: Number of processes in the distributed training. Returns 1 if distributed
        training is not initialized.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Get the rank of the current process in the world.

    Returns:
        int: Global rank of current process. Returns 0 if distributed training
        is not initialized.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Get the rank of current process within the local machine.

    Returns:
        int: Local rank of current process within its machine. Returns 0 if
        distributed training is not initialized.

    Raises:
        AssertionError: If local process group is not created.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert (
        _LOCAL_PROCESS_GROUP is not None
    ), "Local process group is not created! Please use launch() to spawn processes!"
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Get the size of the local process group (number of processes per machine).

    Returns:
        int: Number of processes on the local machine. Returns 1 if distributed
        training is not initialized.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0).

    Returns:
        bool: True if current process is the main process (rank 0).
    """
    return get_rank() == 0


def synchronize():
    """
    Synchronize all processes in distributed training.

    This function creates a barrier that all processes must reach before proceeding.
    It's useful for ensuring all processes are at the same point in training.

    Note:
        - Does nothing if distributed training is not initialized
        - Uses device-specific barrier for NCCL backend
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Get or create a process group based on gloo backend.

    Returns:
        ProcessGroup: A process group using gloo backend that contains all ranks.
        Result is cached for efficiency.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Gather data from all processes in a way that works with any picklable data.

    Args:
        data: Any picklable object to be gathered from all processes.
        group (ProcessGroup, optional): Process group to gather from. Defaults to
            a global gloo group to reduce GPU memory usage.

    Returns:
        list: List of gathered data from each rank.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = (
            _get_global_gloo_group()
        )  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def gather(data, dst=0, group=None):
    """
    Gather data from all processes to a specific destination rank.

    Args:
        data: Any picklable object to be gathered.
        dst (int, optional): Destination rank to gather to. Defaults to 0.
        group (ProcessGroup, optional): Process group to gather from. Defaults to
            a global gloo group.

    Returns:
        list: On destination rank, returns list of gathered data from each rank.
        On other ranks, returns empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Generate a random seed that is shared across all distributed processes.

    This is useful when all workers need to use the same random seed for
    reproducibility in distributed training.

    Returns:
        int: A random seed that is the same across all workers.

    Note:
        All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce dictionary values from all processes to process with rank 0.

    Args:
        input_dict (dict): Dictionary where values are scalar CUDA tensors to be reduced.
        average (bool, optional): Whether to average or sum the values. Defaults to True.

    Returns:
        dict: On rank 0, dictionary with same keys as input_dict but with reduced values.
        Values are averaged across processes if average=True, otherwise summed.

    Note:
        - Only process with rank 0 will have the reduced values
        - All values in input_dict must be scalar CUDA tensors
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
