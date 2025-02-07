"""
PointGroup for instance segmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Chengyao Wang
Please cite our work if the code is helpful to you.
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pointgroup_ops import ballquery_batch_p, bfs_cluster
except ImportError:
    ballquery_batch_p, bfs_cluster = None, None

from pointcept.models.utils import offset2batch, batch2offset

from pointcept.models.builder import MODELS, build_model


@MODELS.register_module("PG-v1m1")
class PointGroup(nn.Module):
    """
    PointGroup model for instance segmentation in point clouds.
    
    This model implements the PointGroup architecture for point cloud instance segmentation.
    It uses a backbone network for feature extraction, followed by semantic segmentation
    and offset prediction heads. The model supports clustering-based instance generation.
    
    Args:
        backbone (dict): Configuration for backbone network
        backbone_out_channels (int, optional): Output channels from backbone. Defaults to 64
        semantic_num_classes (int, optional): Number of semantic classes. Defaults to 20
        semantic_ignore_index (int, optional): Index to ignore in semantic loss. Defaults to -1
        segment_ignore_index (tuple, optional): Indices to ignore in segmentation. Defaults to (-1, 0, 1)
        instance_ignore_index (int, optional): Index to ignore in instance loss. Defaults to -1
        cluster_thresh (float, optional): Clustering threshold. Defaults to 1.5
        cluster_closed_points (int, optional): Number of closed points for clustering. Defaults to 300
        cluster_propose_points (int, optional): Number of points to propose clusters. Defaults to 100
        cluster_min_points (int, optional): Minimum points per cluster. Defaults to 50
        voxel_size (float, optional): Size of voxels for clustering. Defaults to 0.02
    
    Attributes:
        semantic_num_classes (int): Number of semantic classes
        segment_ignore_index (tuple): Indices to ignore in segmentation
        semantic_ignore_index (int): Index to ignore in semantic loss
        instance_ignore_index (int): Index to ignore in instance loss
        cluster_thresh (float): Clustering threshold
        cluster_closed_points (int): Number of closed points for clustering
        cluster_propose_points (int): Number of points to propose clusters
        cluster_min_points (int): Minimum points per cluster
        voxel_size (float): Size of voxels for clustering
        backbone (nn.Module): Backbone network
        bias_head (nn.Sequential): Head for offset prediction
        seg_head (nn.Linear): Head for semantic segmentation
        ce_criteria (nn.CrossEntropyLoss): Cross entropy loss for semantic segmentation
    """

    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        semantic_num_classes=20,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        cluster_thresh=1.5,
        cluster_closed_points=300,
        cluster_propose_points=100,
        cluster_min_points=50,
        voxel_size=0.02,
    ):
        """
        Initialize PointGroup model.

        Args:
            backbone (dict): Configuration for backbone network
            backbone_out_channels (int, optional): Output channels from backbone. Defaults to 64
            semantic_num_classes (int, optional): Number of semantic classes. Defaults to 20
            semantic_ignore_index (int, optional): Index to ignore in semantic loss. Defaults to -1
            segment_ignore_index (tuple, optional): Indices to ignore in segmentation. Defaults to (-1, 0, 1)
            instance_ignore_index (int, optional): Index to ignore in instance loss. Defaults to -1
            cluster_thresh (float, optional): Clustering threshold. Defaults to 1.5
            cluster_closed_points (int, optional): Number of closed points for clustering. Defaults to 300
            cluster_propose_points (int, optional): Number of points to propose clusters. Defaults to 100
            cluster_min_points (int, optional): Minimum points per cluster. Defaults to 50
            voxel_size (float, optional): Size of voxels for clustering. Defaults to 0.02
        """
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.semantic_num_classes = semantic_num_classes
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index = semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.voxel_size = voxel_size
        self.backbone = build_model(backbone)
        self.bias_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm_fn(backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 3),
        )
        self.seg_head = nn.Linear(backbone_out_channels, semantic_num_classes)
        self.ce_criteria = torch.nn.CrossEntropyLoss(ignore_index=semantic_ignore_index)

    def forward(self, data_dict):
        """
        Forward pass of the PointGroup model.
        
        Args:
            data_dict (dict): Input dictionary containing:
                - coord (torch.Tensor): Point coordinates
                - segment (torch.Tensor): Semantic segmentation labels
                - instance (torch.Tensor): Instance segmentation labels
                - instance_centroid (torch.Tensor): Instance centroid coordinates
                - offset (torch.Tensor): Batch offsets
                
        Returns:
            dict: Dictionary containing:
                During training:
                    - loss (torch.Tensor): Total loss
                    - seg_loss (torch.Tensor): Semantic segmentation loss
                    - bias_l1_loss (torch.Tensor): L1 loss for offset prediction
                    - bias_cosine_loss (torch.Tensor): Cosine loss for offset prediction
                During inference:
                    - pred_scores (torch.Tensor): Instance confidence scores
                    - pred_masks (torch.Tensor): Instance segmentation masks
                    - pred_classes (torch.Tensor): Instance semantic classes
        """
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        instance_centroid = data_dict["instance_centroid"]
        offset = data_dict["offset"]

        feat = self.backbone(data_dict)
        bias_pred = self.bias_head(feat)
        logit_pred = self.seg_head(feat)

        # compute loss
        seg_loss = self.ce_criteria(logit_pred, segment)

        mask = (instance != self.instance_ignore_index).float()
        bias_gt = instance_centroid - coord
        bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
        bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

        bias_pred_norm = bias_pred / (
            torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
        )
        bias_gt_norm = bias_gt / (torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8)
        cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
        bias_cosine_loss = torch.sum(cosine_similarity * mask) / (
            torch.sum(mask) + 1e-8
        )

        loss = seg_loss + bias_l1_loss + bias_cosine_loss
        return_dict = dict(
            loss=loss,
            seg_loss=seg_loss,
            bias_l1_loss=bias_l1_loss,
            bias_cosine_loss=bias_cosine_loss,
        )

        if not self.training:
            # Predict instance centers and segment points
            center_pred = coord + bias_pred
            center_pred /= self.voxel_size
            logit_pred = F.softmax(logit_pred, dim=-1)
            segment_pred = torch.max(logit_pred, 1)[1]  # [n]
            
            # Create mask for valid segments (excluding ignore indices)
            mask = (
                ~torch.concat(
                    [
                        (segment_pred == index).unsqueeze(-1)
                        for index in self.segment_ignore_index
                    ],
                    dim=1,
                )
                .sum(-1)
                .bool()
            )

            # Handle empty mask case
            if mask.sum() == 0:
                proposals_idx = torch.zeros(0).int()
                proposals_offset = torch.zeros(1).int()
            else:
                # Filter predictions by mask
                center_pred_ = center_pred[mask]
                segment_pred_ = segment_pred[mask]

                # Prepare batch information for clustering
                batch_ = offset2batch(offset)[mask]
                offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
                
                # Perform ball query clustering
                idx, start_len = ballquery_batch_p(
                    center_pred_,
                    batch_.int(),
                    offset_.int(),
                    self.cluster_thresh,
                    self.cluster_closed_points,
                )
                
                # Perform breadth-first-search clustering
                proposals_idx, proposals_offset = bfs_cluster(
                    segment_pred_.int().cpu(),
                    idx.cpu(),
                    start_len.cpu(),
                    self.cluster_min_points,
                )
                proposals_idx[:, 1] = (
                    mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
                )

            # Generate instance proposals
            proposals_pred = torch.zeros(
                (proposals_offset.shape[0] - 1, center_pred.shape[0]), dtype=torch.int
            )
            proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
            instance_pred = segment_pred[
                proposals_idx[:, 1][proposals_offset[:-1].long()].long()
            ]
            
            # Filter proposals by point count
            proposals_point_num = proposals_pred.sum(1)
            proposals_mask = proposals_point_num > self.cluster_propose_points
            proposals_pred = proposals_pred[proposals_mask]
            instance_pred = instance_pred[proposals_mask]

            # Calculate confidence scores for each proposal
            pred_scores = []
            pred_classes = []
            pred_masks = proposals_pred.detach().cpu()
            for proposal_id in range(len(proposals_pred)):
                segment_ = proposals_pred[proposal_id]
                confidence_ = logit_pred[
                    segment_.bool(), instance_pred[proposal_id]
                ].mean()
                object_ = instance_pred[proposal_id]
                pred_scores.append(confidence_)
                pred_classes.append(object_)
                
            # Stack predictions if any proposals exist
            if len(pred_scores) > 0:
                pred_scores = torch.stack(pred_scores).cpu()
                pred_classes = torch.stack(pred_classes).cpu()
            else:
                pred_scores = torch.tensor([])
                pred_classes = torch.tensor([])

            # Add predictions to return dictionary
            return_dict["pred_scores"] = pred_scores
            return_dict["pred_masks"] = pred_masks
            return_dict["pred_classes"] = pred_classes
        return return_dict
