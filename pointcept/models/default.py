import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    """
    Default segmentation model for point cloud segmentation.
    
    This model consists of a backbone network and segmentation criteria (loss functions).
    It supports both training and inference modes.
    """

    def __init__(self, backbone=None, criteria=None):
        """
        Initialize the DefaultSegmentor.

        Args:
            backbone (dict, optional): Configuration for the backbone network.
            criteria (dict, optional): Configuration for the loss functions.
        """
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        """
        Forward pass of the segmentation model.

        Args:
            input_dict (dict): Input dictionary containing:
                - Point cloud data
                - 'segment' labels during training/evaluation
                - Optional 'condition' for PPT (Point Prompt Training)

        Returns:
            dict: Dictionary containing:
                - 'loss' during training
                - 'loss' and 'seg_logits' during evaluation
                - 'seg_logits' during testing
        """
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    """
    Enhanced version of the default segmentation model.
    
    This version includes a dedicated segmentation head and supports the Point data structure.
    It provides better modularity and flexibility compared to DefaultSegmentor.
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        """
        Initialize the DefaultSegmentorV2.

        Args:
            num_classes (int): Number of segmentation classes.
            backbone_out_channels (int): Number of output channels from backbone.
            backbone (dict, optional): Configuration for the backbone network.
            criteria (dict, optional): Configuration for the loss functions.
        """
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        """
        Forward pass of the enhanced segmentation model.

        Args:
            input_dict (dict): Input dictionary containing point cloud data and optional labels.

        Returns:
            dict: Dictionary containing:
                - 'loss' during training
                - 'loss' and 'seg_logits' during evaluation
                - 'seg_logits' during testing
        """
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    """
    Default classification model for point cloud classification.
    
    This model consists of a backbone network, a classification head, and criteria (loss functions).
    It supports both training and inference modes.
    """

    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        """
        Initialize the DefaultClassifier.

        Args:
            backbone (dict, optional): Configuration for the backbone network.
            criteria (dict, optional): Configuration for the loss functions.
            num_classes (int, optional): Number of classes for classification. Defaults to 40.
            backbone_embed_dim (int, optional): Embedding dimension from backbone. Defaults to 256.
        """
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        """
        Forward pass of the classification model.

        Args:
            input_dict (dict): Input dictionary containing:
                - Point cloud data
                - Optional 'category' labels during training/evaluation

        Returns:
            dict: Dictionary containing:
                - 'loss' during training
                - 'loss' and 'cls_logits' during evaluation
                - 'cls_logits' during testing
        """
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
