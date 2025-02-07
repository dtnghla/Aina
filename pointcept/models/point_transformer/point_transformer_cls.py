"""
Point Transformer V1 for Object Classification

Might be a bit different from the original paper

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn

from .point_transformer_seg import TransitionDown, Bottleneck
from pointcept.models.builder import MODELS


class PointTransformerCls(nn.Module):
    """
    Point Transformer V1 for Object Classification.
    
    This model implements a hierarchical point transformer architecture for point cloud
    classification. It consists of multiple encoder blocks that progressively downsample
    the point cloud while learning hierarchical features.
    
    Args:
        block (nn.Module): Basic block module for transformer
        blocks (list): Number of blocks in each encoder stage
        in_channels (int, optional): Number of input channels. Defaults to 6
        num_classes (int, optional): Number of output classes. Defaults to 40
    
    Attributes:
        in_channels (int): Number of input channels
        in_planes (int): Input feature dimension
        enc1-5 (nn.Module): Encoder stages 1-5
        cls (nn.Sequential): Classification head
    """

    def __init__(self, block, blocks, in_channels=6, num_classes=40):
        """
        Initialize Point Transformer for classification.

        Args:
            block (nn.Module): Basic block module for transformer
            blocks (list): Number of blocks in each encoder stage
            in_channels (int, optional): Number of input channels. Defaults to 6
            num_classes (int, optional): Number of output classes. Defaults to 40
        """
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(
            block,
            planes[0],
            blocks[0],
            share_planes,
            stride=stride[0],
            nsample=nsample[0],
        )  # N/1
        self.enc2 = self._make_enc(
            block,
            planes[1],
            blocks[1],
            share_planes,
            stride=stride[1],
            nsample=nsample[1],
        )  # N/4
        self.enc3 = self._make_enc(
            block,
            planes[2],
            blocks[2],
            share_planes,
            stride=stride[2],
            nsample=nsample[2],
        )  # N/16
        self.enc4 = self._make_enc(
            block,
            planes[3],
            blocks[3],
            share_planes,
            stride=stride[3],
            nsample=nsample[3],
        )  # N/64
        self.enc5 = self._make_enc(
            block,
            planes[4],
            blocks[4],
            share_planes,
            stride=stride[4],
            nsample=nsample[4],
        )  # N/256
        self.cls = nn.Sequential(
            nn.Linear(planes[4], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        """
        Create an encoder stage with multiple transformer blocks.
        
        Args:
            block (nn.Module): Basic block module for transformer
            planes (int): Number of output channels
            blocks (int): Number of blocks in this stage
            share_planes (int, optional): Number of shared planes. Defaults to 8
            stride (int, optional): Stride for downsampling. Defaults to 1
            nsample (int, optional): Number of samples in local neighborhood. Defaults to 16
            
        Returns:
            nn.Sequential: Sequential container of encoder blocks
        """
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, data_dict):
        """
        Forward pass of the Point Transformer.
        
        Args:
            data_dict (dict): Input dictionary containing:
                - coord (torch.Tensor): Point coordinates
                - feat (torch.Tensor): Point features
                - offset (torch.Tensor): Batch offsets
                
        Returns:
            torch.Tensor: Classification logits of shape [B, num_classes]
        """
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()
        x0 = p0 if self.in_channels == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x = []
        for i in range(o5.shape[0]):
            if i == 0:
                s_i, e_i, cnt = 0, o5[0], o5[0]
            else:
                s_i, e_i, cnt = o5[i - 1], o5[i], o5[i] - o5[i - 1]
            x_b = x5[s_i:e_i, :].sum(0, True) / cnt
            x.append(x_b)
        x = torch.cat(x, 0)
        x = self.cls(x)
        return x


@MODELS.register_module("PointTransformer-Cls26")
class PointTransformerCls26(PointTransformerCls):
    """
    Point Transformer with 26 layers for classification.
    
    This variant uses 1 block in each encoder stage, resulting in a total of 26 layers.
    
    Args:
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(self, **kwargs):
        super(PointTransformerCls26, self).__init__(
            Bottleneck, [1, 1, 1, 1, 1], **kwargs
        )


@MODELS.register_module("PointTransformer-Cls38")
class PointTransformerCls38(PointTransformerCls):
    """
    Point Transformer with 38 layers for classification.
    
    This variant uses 2 blocks in each encoder stage (except first), resulting in a total of 38 layers.
    
    Args:
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(self, **kwargs):
        super(PointTransformerCls38, self).__init__(
            Bottleneck, [1, 2, 2, 2, 2], **kwargs
        )


@MODELS.register_module("PointTransformer-Cls50")
class PointTransformerCls50(PointTransformerCls):
    """
    Point Transformer with 50 layers for classification.
    
    This variant uses varying numbers of blocks in each encoder stage,
    resulting in a total of 50 layers.
    
    Args:
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(self, **kwargs):
        super(PointTransformerCls50, self).__init__(
            Bottleneck, [1, 2, 3, 5, 2], **kwargs
        )
