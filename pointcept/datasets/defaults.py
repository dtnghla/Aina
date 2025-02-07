"""
Default Datasets

This module provides base dataset classes for the Pointcept framework, including
the DefaultDataset for standard point cloud data handling and ConcatDataset for
combining multiple datasets.

The module implements:
- Base dataset functionality for point cloud data
- Standard data loading and preprocessing
- Data caching mechanisms
- Test-time augmentation support
- Multi-dataset concatenation

Features:
- Flexible data loading
- Automatic data type conversion
- Cache support for faster loading
- Test/train mode switching
- Data augmentation pipeline
- Multi-dataset handling

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class DefaultDataset(Dataset):
    """Base dataset class for point cloud data.
    
    This class provides standard functionality for loading and processing point
    cloud data, including support for various point attributes, data augmentation,
    and test-time processing.
    
    Attributes:
        VALID_ASSETS (list): Valid point cloud attributes (coord, color, etc.)
        data_root (str): Root directory for dataset
        split (str): Dataset split (train/val/test)
        transform (Compose): Data transformation pipeline
        cache (bool): Whether to cache data in memory
        ignore_index (int): Label value to ignore in evaluation
        loop (int): Number of times to repeat dataset per epoch
        test_mode (bool): Whether in test mode
        test_cfg (dict): Test-time configuration
        
    Args:
        split (str): Dataset split name. Default: 'train'
        data_root (str): Dataset root directory. Default: 'data/dataset'
        transform (list): List of data transformations. Default: None
        test_mode (bool): Whether in test mode. Default: False
        test_cfg (dict): Test-time configuration. Default: None
        cache (bool): Whether to cache data. Default: False
        ignore_index (int): Index to ignore. Default: -1
        loop (int): Dataset repeat factor. Default: 1
    """
    VALID_ASSETS = [
        "coord",  # Point coordinates (N, 3)
        "color",  # RGB colors (N, 3)
        "normal",  # Normal vectors (N, 3)
        "strength",  # Intensity values (N, 1)
        "segment",  # Semantic labels (N,)
        "instance",  # Instance labels (N,)
        "pose",  # Pose information
    ]

    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
    ):
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        """Get list of data files.
        
        Returns:
            list: List of data file paths
        """
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        """Load data from disk or cache.
        
        Args:
            idx (int): Data index
            
        Returns:
            dict: Dictionary containing point cloud data and attributes
        """
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def prepare_train_data(self, idx):
        """Prepare data for training.
        
        Args:
            idx (int): Data index
            
        Returns:
            dict: Processed data dictionary
        """
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        """Prepare data for testing.
        
        This method handles test-time augmentation and sliding window inference
        if configured.
        
        Args:
            idx (int): Data index
            
        Returns:
            dict: Dictionary containing processed fragments and metadata
        """
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class ConcatDataset(Dataset):
    """Dataset class for concatenating multiple datasets.
    
    This class enables training on multiple datasets simultaneously by combining
    them into a single dataset. It maintains the original dataset properties
    while providing a unified interface.
    
    Attributes:
        datasets (list): List of dataset instances
        loop (int): Number of times to repeat the combined dataset
        data_list (list): Combined list of data indices
        
    Args:
        datasets (list[dict]): List of dataset configurations
        loop (int): Dataset repeat factor. Default: 1
        
    Example:
        >>> cfg = [
        ...     dict(type='ScanNetDataset', data_root='data/scannet'),
        ...     dict(type='S3DISDataset', data_root='data/s3dis')
        ... ]
        >>> dataset = ConcatDataset(cfg)
    """
    def __init__(self, datasets, loop=1):
        super(ConcatDataset, self).__init__()
        self.datasets = [build_dataset(dataset) for dataset in datasets]
        self.loop = loop
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in the concat set.".format(
                len(self.data_list), self.loop
            )
        )

    def get_data_list(self):
        data_list = []
        for i in range(len(self.datasets)):
            data_list.extend(
                zip(
                    np.ones(len(self.datasets[i])) * i, np.arange(len(self.datasets[i]))
                )
            )
        return data_list

    def get_data(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx][data_idx]

    def get_data_name(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx].get_data_name(data_idx)

    def __getitem__(self, idx):
        return self.get_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
