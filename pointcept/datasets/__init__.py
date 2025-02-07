"""
Dataset System

This module provides a comprehensive dataset management system for point cloud data,
supporting various types of datasets including indoor scenes, outdoor scenes, and
object-centric datasets.

Features:
- Modular dataset architecture
- Support for multiple dataset types:
  - Indoor Scenes: S3DIS, ScanNet, ScanNet++, ARKitScenes, Structure3D
  - Outdoor Scenes: SemanticKITTI, NuScenes, Waymo
  - Objects: ModelNet, ShapeNetPart
- Efficient data loading and caching
- Flexible data transformations
- Multi-dataset training support
- Custom dataset creation

Components:
- DefaultDataset: Base dataset class with standard point cloud data handling
- ConcatDataset: Dataset concatenation for multi-dataset training
- Dataset Builder: Registry-based dataset construction
- Data Collation: Specialized collate functions for point cloud batching

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from .defaults import DefaultDataset, ConcatDataset
from .builder import build_dataset
from .utils import point_collate_fn, collate_fn

# indoor scene
from .s3dis import S3DISDataset
from .scannet import ScanNetDataset, ScanNet200Dataset
from .scannetpp import ScanNetPPDataset
from .scannet_pair import ScanNetPairDataset
from .arkitscenes import ArkitScenesDataset
from .structure3d import Structured3DDataset

# outdoor scene
from .semantic_kitti import SemanticKITTIDataset
from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset

# object
from .modelnet import ModelNetDataset
from .shapenet_part import ShapeNetPartDataset

# dataloader
from .dataloader import MultiDatasetDataloader

from .laz_dataset import LAZDataset

