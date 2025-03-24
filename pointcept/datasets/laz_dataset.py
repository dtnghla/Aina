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

        # Get all LAZ files in split directory
        split_dir = os.path.join(data_root, split)
        self.data_list = [f for f in os.listdir(split_dir) if f.endswith('.laz')]
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )
        
        # Set up test-specific transforms if in test mode
        if self.test_mode and self.test_cfg is not None:
            self.test_voxelize = None
            self.test_crop = None
            self.post_transform = None
            
            # Handle test transforms with proper dict to Compose conversion
            if hasattr(self.test_cfg, 'voxelize'):
                if isinstance(self.test_cfg.voxelize, list):
                    self.test_voxelize = Compose(self.test_cfg.voxelize)
                elif isinstance(self.test_cfg.voxelize, dict):
                    self.test_voxelize = Compose([self.test_cfg.voxelize])
            
            if hasattr(self.test_cfg, 'crop'):
                if isinstance(self.test_cfg.crop, list):
                    self.test_crop = Compose(self.test_cfg.crop)
                elif isinstance(self.test_cfg.crop, dict):
                    self.test_crop = Compose([self.test_cfg.crop])
            
            if hasattr(self.test_cfg, 'post_transform'):
                if isinstance(self.test_cfg.post_transform, list):
                    self.post_transform = Compose(self.test_cfg.post_transform)
                elif isinstance(self.test_cfg.post_transform, dict):
                    self.post_transform = Compose([self.test_cfg.post_transform])
        else:
            self.test_voxelize = None
            self.test_crop = None
            self.post_transform = None
            
        # Find and set GridSample as test_voxelize if not explicitly provided
        if self.test_mode and self.test_voxelize is None:
            # Check if GridSample is in the transform list
            if isinstance(self.transform.transforms, Sequence):
                for t in self.transform.transforms:
                    if t.__class__.__name__ == "GridSample" and getattr(t, "mode", None) == "test":
                        logger.info("Found GridSample in transform, using as test_voxelize")
                        self.test_voxelize = Compose([t])
                        break
            
        # Set default post_transform if not provided
        if self.test_mode and self.post_transform is None:
            # Find ToTensor and Collect in the transform list
            collect_transform = None
            to_tensor_transform = None
            
            if isinstance(self.transform.transforms, Sequence):
                for t in self.transform.transforms:
                    if t.__class__.__name__ == "ToTensor":
                        to_tensor_transform = t
                    elif t.__class__.__name__ == "Collect":
                        collect_transform = t
            
            transforms = []
            if to_tensor_transform is not None:
                transforms.append(to_tensor_transform)
            if collect_transform is not None:
                transforms.append(collect_transform)
            
            if transforms:
                logger.info("Creating post_transform from existing transforms")
                self.post_transform = Compose(transforms)
            else:
                logger.info("Creating default post_transform")
                self.post_transform = Compose([
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "color", "segment", "instance"),
                        feat_keys=("color",),
                    )
                ])

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
            try:
                color = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
                logger.info(f"Loaded colors with shape {color.shape}")
            except AttributeError:
                # If color information is missing, create placeholder data (all zeros)
                logger.warning(f"Color information missing in {data_path}, using zeros")
                color = np.zeros((coord.shape[0], 3), dtype=np.float32)
            
            # Get segmentation and instance IDs
            try:
                segment = las.semantic_id.astype(np.int64) - 1  # Convert to 0-based indexing
                instance = las.instance_id.astype(np.int64)
                logger.info(f"Loaded segment and instance with shapes {segment.shape}, {instance.shape}")
            except AttributeError:
                # If segmentation or instance information is missing, create placeholder data
                logger.warning(f"Segmentation or instance information missing in {data_path}, using zeros")
                segment = np.zeros(coord.shape[0], dtype=np.int64)
                instance = np.zeros(coord.shape[0], dtype=np.int64) - 1  # -1 for background
            
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
            # Load data
            logger.info(f"Preparing test data for index {idx}")
            data_dict = self.get_data(idx)
            
            # Create result dictionary
            result_dict = {}
            
            # Check required keys before transform
            required_keys = ["coord", "color"]
            for key in required_keys:
                if key not in data_dict:
                    logger.error(f"Required key '{key}' not found in data_dict before transforms")
                    logger.error(f"Available keys: {list(data_dict.keys())}")
                    raise KeyError(f"Required key '{key}' not found in data_dict before transforms")
            
            # Apply main transform
            logger.info("Applying main transforms")
            data_dict = self.transform(data_dict)
            logger.info(f"After main transform, keys: {list(data_dict.keys())}")
            
            # Create data_dict_list for fragmentation
            data_dict_list = [data_dict]
            
            # Prepare fragment_list following DefaultDataset pattern
            fragment_list = []
            
            # Apply test_voxelize (GridSample in test mode)
            if self.test_voxelize is not None:
                logger.info("Applying test voxelization")
                for data in data_dict_list:
                    start_time = time.time()
                    data_part_list = self.test_voxelize(data)
                    end_time = time.time()
                    logger.info(f"Test voxelization took {end_time - start_time:.3f} seconds")
                    for data_part in data_part_list:
                        if self.test_crop is not None:
                            data_crops = self.test_crop(data_part)
                            fragment_list.extend(data_crops)
                        else:
                            fragment_list.append(data_part)
            else:
                # If no test_voxelize, just add index and use as is
                data_dict["index"] = np.arange(data_dict["coord"].shape[0])
                if self.test_crop is not None:
                    logger.info("Applying test crop")
                    data_crops = self.test_crop(data_dict)
                    fragment_list.extend(data_crops)
                else:
                    fragment_list.append(data_dict)
            
            # Apply post_transform to each fragment
            logger.info(f"Applying post-transform to {len(fragment_list)} fragments")
            for i in range(len(fragment_list)):
                if self.post_transform is not None:
                    start_time = time.time()
                    fragment_list[i] = self.post_transform(fragment_list[i])
                    end_time = time.time()
                    logger.info(f"Post-transform for fragment {i} took {end_time - start_time:.3f} seconds")
                
                # Ensure offset is added for each fragment
                if "offset" not in fragment_list[i]:
                    fragment_list[i]["offset"] = torch.tensor([fragment_list[i]["coord"].shape[0]], dtype=torch.int32)
                
                logger.info(f"Prepared fragment {i} with {fragment_list[i]['coord'].shape[0]} points")
            
            # Add fragment_list and other necessary data to result_dict
            result_dict["fragment_list"] = fragment_list
            result_dict["segment"] = data_dict["segment"]
            result_dict["name"] = self.get_data_name(idx)
            
            logger.info(f"Returning {len(result_dict['fragment_list'])} fragments for testing")
            return result_dict
        except Exception as e:
            logger.error(f"Error in prepare_test_data: {str(e)}")
            raise

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx % len(self.data_list))
        return self.prepare_train_data(idx % len(self.data_list))

    def __len__(self):
        return len(self.data_list) * self.loop 