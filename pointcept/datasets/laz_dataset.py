import os
import glob
import numpy as np
import torch
import time
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS

import laspy


@DATASETS.register_module("LAZDataset")
class LAZDataset(Dataset):
    def __init__(self,
                 data_root,
                 split,
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 loop=1):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.loop = loop if not test_mode else 1
        self.cache = {}  # Cache for loaded data

        if self.test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = None

        # Get all LAZ files in split directory
        split_dir = os.path.join(data_root, split)
        self.data_list = [f for f in os.listdir(split_dir) if f.endswith('.laz')]
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data(self, idx):
        data_path = os.path.join(self.data_root, self.split, self.data_list[idx])
        logger = get_root_logger()
        
        # Check if data is already cached
        if idx in self.cache:
            logger.info(f"Using cached data for index {idx}")
            return deepcopy(self.cache[idx])
        
        logger.info(f"Loading data from {data_path}")
        
        try:
            # Load LAZ file
            las = laspy.read(data_path)
            coord = np.vstack((las.x, las.y, las.z)).transpose()
            logger.info(f"Loaded coordinates with shape {coord.shape}")
            
            # Get color information and normalize to [0, 1]
            color = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
            logger.info(f"Loaded colors with shape {color.shape}")
            
            # Get segmentation and instance IDs
            segment = las.semantic_id.astype(np.int64) - 1  # Convert to 0-based indexing
            instance = las.instance_id.astype(np.int64)
            logger.info(f"Loaded segment and instance with shapes {segment.shape}, {instance.shape}")
            
            # Calculate instance centroids
            instance_ids = np.unique(instance)
            instance_centroids = np.zeros_like(coord)
            for instance_id in instance_ids:
                if instance_id == -1:  # Skip background points
                    continue
                mask = instance == instance_id
                centroid = coord[mask].mean(axis=0)
                instance_centroids[mask] = centroid
            logger.info(f"Calculated instance centroids with shape {instance_centroids.shape}")

            data_dict = dict(
                coord=coord,
                color=color,
                segment=segment,
                instance=instance,
                instance_centroid=instance_centroids,
                name=self.get_data_name(idx)
            )
            
            # Cache the loaded data
            self.cache[idx] = data_dict
            return deepcopy(data_dict)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def get_data_name(self, idx):
        return os.path.splitext(self.data_list[idx])[0]

    def prepare_train_data(self, idx):
        logger = get_root_logger()
        try:
            # Load data
            logger.info(f"Preparing training data for index {idx}")
            data_dict = self.get_data(idx)
            
            # Check required keys before transform
            required_keys = ["coord", "color"]
            for key in required_keys:
                if key not in data_dict:
                    logger.error(f"Required key '{key}' not found in data_dict before transforms")
                    logger.error(f"Available keys: {list(data_dict.keys())}")
                    raise KeyError(f"Required key '{key}' not found in data_dict before transforms")
            
            # Transform data with timing and validation
            logger.info("Applying transforms")
            if isinstance(self.transform.transforms, Sequence):
                for t in self.transform.transforms:
                    start_time = time.time()
                    data_dict = t(data_dict)
                    end_time = time.time()
                    logger.info(f"Transform {t.__class__.__name__} took {end_time - start_time:.3f} seconds")
                    
                    # Validate keys after each transform
                    for key in required_keys:
                        if key not in data_dict:
                            logger.error(f"Required key '{key}' was removed by transform {t.__class__.__name__}")
                            logger.error(f"Available keys after transform: {list(data_dict.keys())}")
                            raise KeyError(f"Required key '{key}' was removed by transform {t.__class__.__name__}")
            else:
                start_time = time.time()
                data_dict = self.transform(data_dict)
                end_time = time.time()
                logger.info(f"Transform took {end_time - start_time:.3f} seconds")
                
                # Validate keys after transform
                for key in required_keys:
                    if key not in data_dict:
                        logger.error(f"Required key '{key}' was removed by transform")
                        logger.error(f"Available keys after transform: {list(data_dict.keys())}")
                        raise KeyError(f"Required key '{key}' was removed by transform")
            
            # Combine coordinate and color features
            start_time = time.time()
            data_dict["feat"] = torch.cat([data_dict["coord"], data_dict["color"]], dim=1)
            end_time = time.time()
            logger.info(f"Feature concatenation took {end_time - start_time:.3f} seconds")
            
            # Add batch index
            data_dict["offset"] = torch.tensor([data_dict["coord"].shape[0]], dtype=torch.int32)
            logger.info(f"Prepared data with {data_dict['coord'].shape[0]} points")
            return data_dict
        except Exception as e:
            logger.error(f"Error in prepare_train_data: {str(e)}")
            raise

    def prepare_test_data(self, idx):
        logger = get_root_logger()
        try:
            logger.info(f"Preparing test data for index {idx}")
            data_dict = self.get_data(idx)
            
            # Check required keys before transform
            required_keys = ["coord", "color"]
            for key in required_keys:
                if key not in data_dict:
                    logger.error(f"Required key '{key}' not found in data_dict before transforms")
                    logger.error(f"Available keys: {list(data_dict.keys())}")
                    raise KeyError(f"Required key '{key}' not found in data_dict before transforms")
            
            # Voxelize if needed with timing
            if self.test_voxelize:
                start_time = time.time()
                data_dict = self.test_voxelize(data_dict)
                end_time = time.time()
                logger.info(f"Voxelization took {end_time - start_time:.3f} seconds")
                # Validate keys after voxelization
                for key in required_keys:
                    if key not in data_dict:
                        logger.error(f"Required key '{key}' was removed by voxelization")
                        logger.error(f"Available keys after voxelization: {list(data_dict.keys())}")
                        raise KeyError(f"Required key '{key}' was removed by voxelization")
            
            # Crop if needed with timing
            if self.test_crop:
                start_time = time.time()
                data_dict = self.test_crop(data_dict)
                end_time = time.time()
                logger.info(f"Cropping took {end_time - start_time:.3f} seconds")
                # Validate keys after cropping
                for key in required_keys:
                    if key not in data_dict:
                        logger.error(f"Required key '{key}' was removed by cropping")
                        logger.error(f"Available keys after cropping: {list(data_dict.keys())}")
                        raise KeyError(f"Required key '{key}' was removed by cropping")
            
            # Post transform with timing
            if self.post_transform:
                if isinstance(self.post_transform.transforms, Sequence):
                    for t in self.post_transform.transforms:
                        start_time = time.time()
                        data_dict = t(data_dict)
                        end_time = time.time()
                        logger.info(f"Post transform {t.__class__.__name__} took {end_time - start_time:.3f} seconds")
                        # Validate keys after each transform
                        for key in required_keys:
                            if key not in data_dict:
                                logger.error(f"Required key '{key}' was removed by post transform {t.__class__.__name__}")
                                logger.error(f"Available keys after transform: {list(data_dict.keys())}")
                                raise KeyError(f"Required key '{key}' was removed by post transform {t.__class__.__name__}")
                else:
                    start_time = time.time()
                    data_dict = self.post_transform(data_dict)
                    end_time = time.time()
                    logger.info(f"Post transform took {end_time - start_time:.3f} seconds")
                    # Validate keys after transform
                    for key in required_keys:
                        if key not in data_dict:
                            logger.error(f"Required key '{key}' was removed by post transform")
                            logger.error(f"Available keys after transform: {list(data_dict.keys())}")
                            raise KeyError(f"Required key '{key}' was removed by post transform")
            
            # Combine coordinate and color features with timing
            start_time = time.time()
            data_dict["feat"] = torch.cat([data_dict["coord"], data_dict["color"]], dim=1)
            end_time = time.time()
            logger.info(f"Feature concatenation took {end_time - start_time:.3f} seconds")
            
            # Add batch index
            data_dict["offset"] = torch.tensor([data_dict["coord"].shape[0]], dtype=torch.int32)
            logger.info(f"Prepared test data with {data_dict['coord'].shape[0]} points")
            return data_dict
        except Exception as e:
            logger.error(f"Error in prepare_test_data: {str(e)}")
            raise

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx % len(self.data_list))
        return self.prepare_train_data(idx % len(self.data_list))

    def __len__(self):
        return len(self.data_list) * self.loop 