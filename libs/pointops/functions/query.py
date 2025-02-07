"""
Point Query Operations

This module provides efficient CUDA-accelerated query operations for point clouds,
including k-nearest neighbor (KNN) search and ball query with optional random sampling.
These operations are fundamental for point cloud feature learning and neighborhood
aggregation.

Key Features:
- K-nearest neighbor search
- Ball query with fixed radius
- Random ball query for stochastic sampling
- Batch-wise operations
- CUDA acceleration
- Memory-efficient implementation
- Support for non-uniform point distributions

The module includes three main query types:
1. KNN Query: Finds k nearest neighbors for each point
2. Ball Query: Finds points within a fixed radius
3. Random Ball Query: Randomly samples points within a radius

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
from torch.autograd import Function

from pointops._C import knn_query_cuda, random_ball_query_cuda, ball_query_cuda


class KNNQuery(Function):
    """
    Custom autograd Function implementing k-nearest neighbor query for point clouds.
    This class finds the k nearest neighbors for each point in the target point cloud
    from the source point cloud, supporting batch-wise operations and CUDA acceleration.
    """

    @staticmethod
    def forward(ctx, nsample, xyz, offset, new_xyz=None, new_offset=None):
        """
        Forward pass of k-nearest neighbor query.

        Args:
            ctx: Context object for autograd.
            nsample (int): Number of nearest neighbors to find.
            xyz (torch.Tensor): Source point cloud coordinates (n, 3).
            offset (torch.Tensor): Batch offsets for source points (b).
            new_xyz (torch.Tensor, optional): Target point cloud coordinates (m, 3).
                If None, uses xyz. Defaults to None.
            new_offset (torch.Tensor, optional): Batch offsets for target points (b).
                If None, uses offset. Defaults to None.

        Returns:
            tuple: (idx, dist) where:
                - idx (torch.Tensor): Indices of nearest neighbors (m, nsample)
                - dist (torch.Tensor): Squared distances to nearest neighbors (m, nsample)

        Note:
            - Points are queried within the same batch only
            - Output distances are squared Euclidean distances
            - If new_xyz is None, performs self-query on xyz
        """
        if new_xyz is None or new_offset is None:
            new_xyz = xyz
            new_offset = offset
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        knn_query_cuda(
            m, nsample, xyz, new_xyz, offset.int(), new_offset.int(), idx, dist2
        )
        return idx, torch.sqrt(dist2)


class RandomBallQuery(Function):
    """
    Custom autograd Function implementing random ball query for point clouds.
    This class finds points within a spherical region with random sampling,
    useful for non-deterministic neighborhood sampling in point cloud processing.
    """

    @staticmethod
    def forward(
        ctx, nsample, max_radius, min_radius, xyz, offset, new_xyz=None, new_offset=None
    ):
        """
        Forward pass of random ball query.

        Args:
            ctx: Context object for autograd.
            nsample (int): Maximum number of points to sample in each ball.
            max_radius (float): Maximum radius of the query ball.
            min_radius (float): Minimum radius of the query ball.
            xyz (torch.Tensor): Source point cloud coordinates (n, 3).
            offset (torch.Tensor): Batch offsets for source points (b).
            new_xyz (torch.Tensor, optional): Target point cloud coordinates (m, 3).
                If None, uses xyz. Defaults to None.
            new_offset (torch.Tensor, optional): Batch offsets for target points (b).
                If None, uses offset. Defaults to None.

        Returns:
            tuple: (idx, dist) where:
                - idx (torch.Tensor): Indices of randomly sampled points in balls (m, nsample)
                - dist (torch.Tensor): Distances to sampled points (m, nsample)

        Note:
            - Points are randomly sampled within the spherical shell defined by min_radius and max_radius
            - Sampling is performed independently for each query point
            - The number of returned points may be less than nsample if fewer points exist in the ball
        """
        if new_xyz is None or new_offset is None:
            new_xyz = xyz
            new_offset = offset
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        assert min_radius < max_radius

        m = new_xyz.shape[0]
        order = []
        for k in range(offset.shape[0]):
            s_k, e_k = (0, offset[0]) if k == 0 else (offset[k - 1], offset[k])
            order.append(
                torch.randperm(e_k - s_k, dtype=torch.int32, device=offset.device) + s_k
            )
        order = torch.cat(order, dim=0)
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        random_ball_query_cuda(
            m,
            nsample,
            min_radius,
            max_radius,
            order,
            xyz,
            new_xyz,
            offset.int(),
            new_offset.int(),
            idx,
            dist2,
        )
        return idx, torch.sqrt(dist2)


class BallQuery(Function):
    """
    Custom autograd Function implementing ball query for point clouds.
    This class finds all points within a spherical region around each target point,
    useful for neighborhood feature aggregation in point cloud processing.
    """

    @staticmethod
    def forward(
        ctx, nsample, max_radius, min_radius, xyz, offset, new_xyz=None, new_offset=None
    ):
        """
        Forward pass of ball query.

        Args:
            ctx: Context object for autograd.
            nsample (int): Maximum number of points to return in each ball.
            max_radius (float): Maximum radius of the query ball.
            min_radius (float): Minimum radius of the query ball.
            xyz (torch.Tensor): Source point cloud coordinates (n, 3).
            offset (torch.Tensor): Batch offsets for source points (b).
            new_xyz (torch.Tensor, optional): Target point cloud coordinates (m, 3).
                If None, uses xyz. Defaults to None.
            new_offset (torch.Tensor, optional): Batch offsets for target points (b).
                If None, uses offset. Defaults to None.

        Returns:
            tuple: (idx, dist) where:
                - idx (torch.Tensor): Indices of points in balls (m, nsample)
                - dist (torch.Tensor): Distances to points (m, nsample)

        Note:
            - Points are returned in order of increasing distance from the query point
            - The number of returned points may be less than nsample if fewer points exist in the ball
            - Points between min_radius and max_radius are included
            - Queries are performed within the same batch only
        """
        if new_xyz is None or new_offset is None:
            new_xyz = xyz
            new_offset = offset
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        assert min_radius < max_radius

        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        ball_query_cuda(
            m,
            nsample,
            min_radius,
            max_radius,
            xyz,
            new_xyz,
            offset.int(),
            new_offset.int(),
            idx,
            dist2,
        )
        return idx, torch.sqrt(dist2)


knn_query = KNNQuery.apply
ball_query = BallQuery.apply
random_ball_query = RandomBallQuery.apply
