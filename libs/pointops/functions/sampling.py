"""
Point Sampling Operations

This module provides efficient CUDA-accelerated sampling operations for point clouds.
These operations are essential for hierarchical feature learning and multi-scale
processing of point clouds.

Key Features:
- Furthest point sampling (FPS)
- Random sampling
- Efficient CUDA implementation
- Batch-wise operations
- Support for non-uniform point distributions
- Memory-efficient implementation

The module implements:
1. Furthest Point Sampling (FPS): Selects points that maximize coverage
2. Random Sampling: Stochastic point selection
3. Hybrid Sampling: Combination of FPS and random sampling

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
from torch.autograd import Function

from pointops._C import farthest_point_sampling_cuda


class FarthestPointSampling(Function):
    """
    Custom autograd Function implementing Farthest Point Sampling (FPS) for point clouds.
    This class selects a subset of points that maximizes the minimum distance between selected points,
    ensuring good coverage of the input point cloud.
    """

    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        Forward pass of farthest point sampling.

        Args:
            ctx: Context object for autograd.
            xyz (torch.Tensor): Input point cloud coordinates (n, 3).
            offset (torch.Tensor): Batch offsets for input points (b).
            new_offset (torch.Tensor): Target batch offsets for sampled points (b).

        Returns:
            torch.Tensor: Indices of sampled points (m), where m is determined by new_offset.
                         The indices are selected to maximize the minimum distance between points.
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i - 1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b - 1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        farthest_point_sampling_cuda(
            b, n_max, xyz, offset.int(), new_offset.int(), tmp, idx
        )
        del tmp
        return idx


farthest_point_sampling = FarthestPointSampling.apply
