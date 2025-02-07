import torch
from torch.autograd import Function
import pointgroup_ops_cuda


class BallQueryBatchP(Function):
    """
    Custom autograd Function implementing batch-wise ball query for point clouds.
    """

    @staticmethod
    def forward(ctx, coords, batch_idxs, batch_offsets, radius, meanActive):
        """
        Forward pass of batch-wise ball query.

        Args:
            ctx: Context object for autograd.
            coords (torch.Tensor): Point coordinates (n, 3), float.
            batch_idxs (torch.Tensor): Batch indices for each point (n), int.
            batch_offsets (torch.Tensor): Offsets for each batch (B+1), int.
            radius (float): Radius for ball query.
            meanActive (int): Initial estimate of mean number of points per ball.

        Returns:
            tuple: (idx, start_len) where:
                - idx (torch.Tensor): Indices of points within balls (nActive), int
                - start_len (torch.Tensor): Start and length information for each point (n, 2), int
        """

        n = coords.size(0)

        assert coords.is_contiguous() and coords.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert batch_offsets.is_contiguous() and batch_offsets.is_cuda

        while True:
            idx = torch.cuda.IntTensor(n * meanActive).zero_()
            start_len = torch.cuda.IntTensor(n, 2).zero_()
            nActive = pointgroup_ops_cuda.ballquery_batch_p(
                coords, batch_idxs, batch_offsets, idx, start_len, n, meanActive, radius
            )
            if nActive <= n * meanActive:
                break
            meanActive = int(nActive // n + 1)
        idx = idx[:nActive]

        return idx, start_len

    @staticmethod
    def backward(ctx, a=None, b=None):
        """
        Backward pass of batch-wise ball query (no gradients needed).

        Args:
            ctx: Context object for autograd.
            a: First gradient from downstream operations (unused).
            b: Second gradient from downstream operations (unused).

        Returns:
            None: No gradients to propagate.
        """
        return None, None, None


ballquery_batch_p = BallQueryBatchP.apply


class Clustering:
    def __init__(
        self,
        ignored_labels,
        class_mapping,
        thresh=0.03,
        closed_points=300,
        min_points=50,
        propose_points=100,
        score_func=torch.max,
    ) -> None:
        """
        Initialize the Clustering class for point cloud instance segmentation.

        Args:
            ignored_labels (list): List of labels to ignore during clustering.
            class_mapping (torch.Tensor): Mapping from input labels to output class indices.
            thresh (float, optional): Distance threshold for clustering. Defaults to 0.03.
            closed_points (int, optional): Number of closed points to consider. Defaults to 300.
            min_points (int, optional): Minimum points required for a valid cluster. Defaults to 50.
            propose_points (int, optional): Minimum points required for a proposal. Defaults to 100.
            score_func (callable, optional): Function to compute cluster scores. Defaults to torch.max.
        """
        self.ignored_labels = ignored_labels
        self.thresh = thresh
        self.closed_points = closed_points
        self.min_points = min_points
        self.class_mapping = class_mapping
        self.propose_points = propose_points
        self.score_func = score_func

    def cluster(self, vertices, scores):
        """
        Perform clustering on point cloud vertices based on semantic scores.

        Args:
            vertices (torch.Tensor): Point cloud vertices (N, 3).
            scores (torch.Tensor): Semantic scores for each vertex.

        Returns:
            tuple: (proposals_pred, labels) where:
                - proposals_pred (torch.Tensor): Binary mask for each proposal (nProposal, N)
                - labels (torch.Tensor): Semantic labels for each proposal
        """
        labels = torch.max(scores, 1)[1]  # (N) long, cuda
        proposals_idx, proposals_offset = self.cluster_(vertices, labels)

        ## debug
        # import ipdb; ipdb.set_trace()
        # colors = np.array(create_color_palette())[labels.cpu()]
        # write_triangle_mesh(vertices, colors, None, 'semantics.ply')

        # scatter
        proposals_pred = torch.zeros(
            (proposals_offset.shape[0] - 1, vertices.shape[0]), dtype=torch.int
        )  # (nProposal, N), int, cuda
        proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        labels = labels[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]

        proposals_pointnum = proposals_pred.sum(1)
        npoint_mask = proposals_pointnum > self.propose_points

        proposals_pred = proposals_pred[npoint_mask]
        labels = labels[npoint_mask]
        return proposals_pred, labels

    def cluster_(self, vertices, labels):
        """
        Internal clustering function that handles the core clustering logic.

        Args:
            vertices (torch.Tensor): Point cloud vertices (N, 3).
            labels (torch.Tensor): Semantic labels for each vertex (N).

        Returns:
            tuple: (proposals_idx, proposals_offset) where:
                - proposals_idx (torch.Tensor): Indices for each proposal point (nPoints, 2)
                - proposals_offset (torch.Tensor): Offsets for each proposal
        """
        batch_idxs = torch.zeros_like(labels)

        mask_non_ignored = torch.ones_like(labels).bool()
        for ignored_label in self.ignored_labels:
            mask_non_ignored = mask_non_ignored & (
                self.class_mapping[labels] != ignored_label
            )
        object_idxs = mask_non_ignored.nonzero().view(-1)

        vertices_ = vertices[object_idxs].float()
        labels_ = labels[object_idxs].int()

        if vertices_.numel() == 0:
            return torch.zeros((0, 2)).int(), torch.zeros(1).int()

        batch_idxs_ = batch_idxs[object_idxs].int()
        batch_offsets_ = torch.FloatTensor([0, object_idxs.shape[0]]).int().cuda()

        idx, start_len = ballquery_batch_p(
            vertices_, batch_idxs_, batch_offsets_, self.thresh, self.closed_points
        )
        proposals_idx, proposals_offset = bfs_cluster(
            labels_.cpu(), idx.cpu(), start_len.cpu(), self.min_points
        )
        proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

        return proposals_idx, proposals_offset

    def get_instances(self, vertices, scores):
        """
        Get instance segmentation results from clustered points.

        Args:
            vertices (torch.Tensor): Point cloud vertices (N, 3).
            scores (torch.Tensor): Semantic scores for each vertex.

        Returns:
            dict: Dictionary containing instance information where each key is a proposal_id and
                 values contain:
                 - 'conf': Confidence score
                 - 'label_id': Semantic label ID
                 - 'pred_mask': Binary mask for the instance
        """
        proposals_pred, labels = self.cluster(vertices, scores)
        instances = {}
        for proposal_id in range(len(proposals_pred)):
            clusters_i = proposals_pred[proposal_id]
            score = scores[clusters_i.bool(), labels[proposal_id]]
            score = self.score_func(score)
            instances[proposal_id] = {}
            instances[proposal_id]["conf"] = score.cpu().numpy()
            instances[proposal_id]["label_id"] = self.class_mapping.cpu()[
                labels[proposal_id]
            ]
            instances[proposal_id]["pred_mask"] = clusters_i.cpu().numpy()
        return instances


class BFSCluster(Function):
    """
    Custom autograd Function implementing Breadth-First Search clustering for point clouds.
    """

    @staticmethod
    def forward(ctx, semantic_label, ball_query_idxs, start_len, threshold):
        """
        Forward pass of BFS clustering.

        Args:
            ctx: Context object for autograd.
            semantic_label (torch.Tensor): Semantic labels for each point (N).
            ball_query_idxs (torch.Tensor): Ball query indices (nActive).
            start_len (torch.Tensor): Start and length information for each point (N, 2).
            threshold (int): Minimum number of points required for a valid cluster.

        Returns:
            tuple: (cluster_idxs, cluster_offsets) where:
                - cluster_idxs (torch.Tensor): Indices for cluster points (sumNPoint, 2)
                - cluster_offsets (torch.Tensor): Offsets for each cluster (nCluster + 1)
        """
        N = start_len.size(0)

        assert semantic_label.is_contiguous()
        assert ball_query_idxs.is_contiguous()
        assert start_len.is_contiguous()

        cluster_idxs = semantic_label.new()
        cluster_offsets = semantic_label.new()

        pointgroup_ops_cuda.bfs_cluster(
            semantic_label,
            ball_query_idxs,
            start_len,
            cluster_idxs,
            cluster_offsets,
            N,
            threshold,
        )

        return cluster_idxs, cluster_offsets

    @staticmethod
    def backward(ctx, a=None):
        """
        Backward pass of BFS clustering (no gradients needed).

        Args:
            ctx: Context object for autograd.
            a: Gradient from downstream operations (unused).

        Returns:
            None: No gradients to propagate.
        """
        return None


bfs_cluster = BFSCluster.apply
