"""
Context-aware Classifier for Semantic Segmentation

Author: Zhuotao Tian, Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model


@MODELS.register_module("CAC-v1m1")
class CACSegmentor(nn.Module):
    """
    Context-aware Classifier for Semantic Segmentation.
    
    This model implements a context-aware classifier that leverages both local and global
    context for point cloud semantic segmentation. It consists of a backbone network,
    a segmentation head, and multiple projection layers for feature enhancement.
    
    Args:
        num_classes (int): Number of semantic classes
        backbone_out_channels (int): Output channel dimension from backbone
        backbone (dict, optional): Configuration for backbone network
        criteria (dict, optional): Configuration for loss functions
        cos_temp (float, optional): Temperature for cosine similarity. Defaults to 15
        main_weight (float, optional): Weight for main loss. Defaults to 1
        pre_weight (float, optional): Weight for pre-training loss. Defaults to 1
        pre_self_weight (float, optional): Weight for self pre-training loss. Defaults to 1
        kl_weight (float, optional): Weight for KL divergence loss. Defaults to 1
        conf_thresh (float, optional): Confidence threshold. Defaults to 0
        detach_pre_logits (bool, optional): Whether to detach pre-training logits. Defaults to False
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        cos_temp=15,
        main_weight=1,
        pre_weight=1,
        pre_self_weight=1,
        kl_weight=1,
        conf_thresh=0,
        detach_pre_logits=False,
    ):
        """
        Initialize the Context-aware Classifier.

        Args:
            num_classes (int): Number of semantic classes
            backbone_out_channels (int): Output channel dimension from backbone
            backbone (dict, optional): Configuration for backbone network
            criteria (dict, optional): Configuration for loss functions
            cos_temp (float, optional): Temperature for cosine similarity. Defaults to 15
            main_weight (float, optional): Weight for main loss. Defaults to 1
            pre_weight (float, optional): Weight for pre-training loss. Defaults to 1
            pre_self_weight (float, optional): Weight for self pre-training loss. Defaults to 1
            kl_weight (float, optional): Weight for KL divergence loss. Defaults to 1
            conf_thresh (float, optional): Confidence threshold. Defaults to 0
            detach_pre_logits (bool, optional): Whether to detach pre-training logits. Defaults to False
        """
        super().__init__()
        self.num_classes = num_classes
        self.cos_temp = cos_temp
        self.main_weight = main_weight
        self.pre_weight = pre_weight
        self.pre_self_weight = pre_self_weight
        self.kl_weight = kl_weight
        self.conf_thresh = conf_thresh
        self.detach_pre_logits = detach_pre_logits

        # backbone
        self.backbone = build_model(backbone)
        # heads
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.proj = nn.Sequential(
            nn.Linear(backbone_out_channels * 2, backbone_out_channels * 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels * 2, backbone_out_channels),
        )
        self.apd_proj = nn.Sequential(
            nn.Linear(backbone_out_channels * 2, backbone_out_channels * 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels * 2, backbone_out_channels),
        )
        self.feat_proj_layer = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels, bias=False),
            nn.BatchNorm1d(backbone_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels, backbone_out_channels),
        )
        # Criteria
        self.criteria = build_criteria(criteria)

    @staticmethod
    def get_pred(x, proto):
        """
        Get prediction logits using cosine similarity between features and prototypes.
        
        Args:
            x (torch.Tensor): Input features of shape [N, C]
            proto (torch.Tensor): Prototype features of shape [num_classes, C]
            
        Returns:
            torch.Tensor: Prediction logits of shape [N, num_classes]
        """
        # x: [n,c]; proto: [cls, c]
        x = F.normalize(x, 2, 1)
        proto = F.normalize(proto, 2, 1)
        pred = x @ proto.permute(1, 0)  # [n,c] x [c, cls] -> [n, cls]
        return pred

    def get_adaptive_perspective(self, feat, target, new_proto, proto):
        """
        Get predictions from an adaptive perspective by combining class-specific prototypes.
        
        Args:
            feat (torch.Tensor): Input features of shape [N, C]
            target (torch.Tensor): Ground truth labels of shape [N]
            new_proto (torch.Tensor): New prototype features of shape [num_classes, C]
            proto (torch.Tensor): Original prototype features of shape [num_classes, C]
            
        Returns:
            torch.Tensor: Prediction logits from adaptive perspective
        """
        raw_feat = feat.clone()
        # target: [n]
        # feat: [n,c]
        # proto: [cls, c]
        unique_y = list(target.unique())
        if -1 in unique_y:
            unique_y.remove(-1)
        target = target.unsqueeze(-1)  # [n, 1]

        for tmp_y in unique_y:
            tmp_mask = (target == tmp_y).float()
            tmp_proto = (feat * tmp_mask).sum(0) / (tmp_mask.sum(0) + 1e-4)  # c
            onehot_vec = torch.zeros(new_proto.shape[0], 1).cuda()  # cls, 1
            onehot_vec[tmp_y.long()] = 1
            new_proto = (
                new_proto * (1 - onehot_vec) + tmp_proto.unsqueeze(0) * onehot_vec
            )

        new_proto = torch.cat([new_proto, proto], -1)
        new_proto = self.apd_proj(new_proto)
        raw_feat = self.feat_proj_layer(raw_feat)
        pred = self.get_pred(raw_feat, new_proto)
        return pred

    def post_refine_proto_batch(self, feat, pred, proto, offset=None):
        """
        Refine prototypes using batch-wise processing.
        
        Args:
            feat (torch.Tensor): Input features of shape [N, C]
            pred (torch.Tensor): Initial predictions of shape [N, num_classes]
            proto (torch.Tensor): Prototype features of shape [num_classes, C]
            offset (torch.Tensor, optional): Batch offsets. Defaults to None
            
        Returns:
            torch.Tensor: Refined prediction logits
        """
        # x: [n, c]; pred: [n, cls]; proto: [cls, c]
        pred_list = []
        x = feat
        raw_x = x.clone()
        if self.detach_pre_logits:
            pred = pred.detach()
        raw_pred = pred.clone()

        if offset is None:
            raw_x = x.clone()
            n, n_cls = pred.shape[:]
            pred = pred.view(n, n_cls)
            pred = F.softmax(pred, 1).permute(1, 0)  # [n, cls] -> [cls, n]
            if self.conf_thresh > 0:
                max_pred = (
                    (pred.max(0)[0] >= self.conf_thresh).float().unsqueeze(0)
                )  # 1, n
                pred = pred * max_pred
            pred_proto = (pred / (pred.sum(-1).unsqueeze(-1) + 1e-7)) @ raw_x  # cls, c

            pred_proto = torch.cat([pred_proto, proto], -1)  # cls, 2c
            pred_proto = self.proj(pred_proto)
            raw_x = self.feat_proj_layer(raw_x)
            new_pred = self.get_pred(raw_x, pred_proto)
        else:
            for i in range(len(offset)):
                if i == 0:
                    start = 0
                    end = offset[i]
                else:
                    start, end = offset[i - 1], offset[i]
                tmp_x = raw_x[start:end]
                pred = raw_pred[start:end]
                n, n_cls = pred.shape[:]
                pred = pred.view(n, n_cls)
                pred = F.softmax(pred, 1).permute(1, 0)  # [n, cls] -> [cls, n]
                if self.conf_thresh > 0:
                    max_pred = (
                        (pred.max(0)[0] >= self.conf_thresh).float().unsqueeze(0)
                    )  # 1, n
                    pred = pred * max_pred
                pred_proto = (
                    pred / (pred.sum(-1).unsqueeze(-1) + 1e-7)
                ) @ tmp_x  # cls, c

                pred_proto = torch.cat([pred_proto, proto], -1)  # cls, 2c
                pred_proto = self.proj(pred_proto)
                tmp_x = self.feat_proj_layer(tmp_x)
                new_pred = self.get_pred(tmp_x, pred_proto)
                pred_list.append(new_pred)
            new_pred = torch.cat(pred_list, 0)
        return new_pred

    @staticmethod
    def get_distill_loss(pred, soft, target, smoothness=0.5, eps=0):
        """
        Calculate knowledge distillation loss between predictions and soft targets.
        
        Args:
            pred (torch.Tensor): Model predictions of shape [N, num_classes]
            soft (torch.Tensor): Soft targets of shape [N, num_classes]
            target (torch.Tensor): Ground truth labels of shape [N]
            smoothness (float, optional): Label smoothing factor. Defaults to 0.5
            eps (float, optional): Small epsilon for numerical stability. Defaults to 0
            
        Returns:
            torch.Tensor: Knowledge distillation loss
        """
        n, c = soft.shape[:]
        soft = soft.detach()
        target = target.unsqueeze(-1)  # n, 1
        onehot = target.view(-1, 1)  # n, 1
        ignore_mask = (onehot == -1).float()
        sm_soft = F.softmax(soft / 1, 1)  # n, c

        onehot = onehot * (1 - ignore_mask)
        onehot = torch.zeros(n, c).cuda().scatter_(1, onehot.long(), 1)  # n, c
        smoothed_label = smoothness * sm_soft + (1 - smoothness) * onehot
        if eps > 0:
            smoothed_label = smoothed_label * (1 - eps) + (1 - smoothed_label) * eps / (
                smoothed_label.shape[1] - 1
            )

        loss = torch.mul(-1 * F.log_softmax(pred, dim=1), smoothed_label)  # b, n, h, w
        loss = loss.sum(1)

        sm_soft = F.softmax(soft / 1, 1)  # n, c
        entropy_mask = -1 * (sm_soft * torch.log(sm_soft + 1e-4)).sum(1)

        # for class-wise entropy estimation
        target = target.squeeze(-1)
        unique_classes = list(target.unique())
        if -1 in unique_classes:
            unique_classes.remove(-1)
        valid_mask = (target != -1).float()
        entropy_mask = entropy_mask * valid_mask
        loss_list = []
        weight_list = []
        for tmp_y in unique_classes:
            tmp_mask = (target == tmp_y).float().squeeze()
            tmp_entropy_mask = entropy_mask * tmp_mask
            class_weight = 1
            tmp_loss = (loss * tmp_entropy_mask).sum() / (tmp_entropy_mask.sum() + 1e-4)
            loss_list.append(class_weight * tmp_loss)
            weight_list.append(class_weight)

        if len(weight_list) > 0:
            loss = sum(loss_list) / (sum(weight_list) + 1e-4)
        else:
            loss = torch.zeros(1).cuda().mean()
        return loss

    def forward(self, data_dict):
        """
        Forward pass of the model.
        
        Args:
            data_dict (dict): Input data dictionary containing:
                - offset: Batch offsets
                - segment: Ground truth segmentation labels (training only)
                
        Returns:
            dict: Dictionary containing loss terms during training or predictions during inference
        """
        offset = data_dict["offset"]
        point = self.backbone(data_dict)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)

        if self.training:
            target = data_dict["segment"]
            pre_logits = seg_logits.clone()
            refine_logits = (
                self.post_refine_proto_batch(
                    feat=feat,
                    pred=seg_logits,
                    proto=self.seg_head.weight.squeeze(),
                    offset=offset,
                )
                * self.cos_temp
            )

            cac_pred = (
                self.get_adaptive_perspective(
                    feat=feat,
                    target=target,
                    new_proto=self.seg_head.weight.detach().data.squeeze(),
                    proto=self.seg_head.weight.squeeze(),
                )
                * self.cos_temp
            )

            seg_loss = self.criteria(refine_logits, target) * self.main_weight
            pre_loss = self.criteria(cac_pred, target) * self.pre_weight
            pre_self_loss = self.criteria(pre_logits, target) * self.pre_self_weight
            kl_loss = (
                self.get_distill_loss(
                    pred=refine_logits, soft=cac_pred.detach(), target=target
                )
                * self.kl_weight
            )
            loss = seg_loss + pre_loss + pre_self_loss + kl_loss
            return dict(
                loss=loss,
                seg_loss=seg_loss,
                pre_loss=pre_loss,
                pre_self_loss=pre_self_loss,
                kl_loss=kl_loss,
            )

        elif "segment" in data_dict.keys():
            refine_logits = (
                self.post_refine_proto_batch(
                    feat=feat,
                    pred=seg_logits,
                    proto=self.seg_head.weight.squeeze(),
                    offset=offset,
                )
                * self.cos_temp
            )

            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=refine_logits)

        else:
            refine_logits = (
                self.post_refine_proto_batch(
                    feat=feat,
                    pred=seg_logits,
                    proto=self.seg_head.weight.squeeze(),
                    offset=offset,
                )
                * self.cos_temp
            )
            return dict(seg_logits=refine_logits)
