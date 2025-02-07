"""
ModelNet40 Dataset

get sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape)
at "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import pointops
import torch
from torch.utils.data import Dataset
from copy import deepcopy


from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class ModelNetDataset(Dataset):
    """
    Dataset class for ModelNet40 point cloud classification.
    
    This dataset provides sampled point clouds from ModelNet40, including XYZ coordinates
    and normal vectors (10k points per shape). The dataset can be downloaded from:
    https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
    
    Args:
        split (str, optional): Dataset split ('train' or 'test'). Defaults to 'train'
        data_root (str, optional): Root directory of dataset. Defaults to 'data/modelnet40'
        class_names (list, optional): List of class names to use
        transform (list, optional): List of data transformations to apply
        num_points (int, optional): Number of points per sample. Defaults to 8192
        uniform_sampling (bool, optional): Whether to use uniform sampling. Defaults to True
        save_record (bool, optional): Whether to save dataset records. Defaults to True
        test_mode (bool, optional): Whether in test mode. Defaults to False
        test_cfg (dict, optional): Test-time configuration
        loop (int, optional): Number of times to repeat dataset. Defaults to 1
    
    Attributes:
        data_root (str): Root directory of dataset
        class_names (dict): Mapping from class names to indices
        split (str): Dataset split
        num_point (int): Number of points per sample
        uniform_sampling (bool): Whether using uniform sampling
        transform (Compose): Data transformation pipeline
        loop (int): Dataset repeat factor
        test_mode (bool): Whether in test mode
        test_cfg (dict): Test-time configuration
        data_list (list): List of data samples
    """

    def __init__(
        self,
        split="train",
        data_root="data/modelnet40",
        class_names=None,
        transform=None,
        num_points=8192,
        uniform_sampling=True,
        save_record=True,
        test_mode=False,
        test_cfg=None,
        loop=1,
    ):
        """
        Initialize the ModelNet40 dataset.

        Args:
            split (str, optional): Dataset split ('train' or 'test'). Defaults to 'train'
            data_root (str, optional): Root directory of dataset. Defaults to 'data/modelnet40'
            class_names (list, optional): List of class names to use
            transform (list, optional): List of data transformations to apply
            num_points (int, optional): Number of points per sample. Defaults to 8192
            uniform_sampling (bool, optional): Whether to use uniform sampling. Defaults to True
            save_record (bool, optional): Whether to save dataset records. Defaults to True
            test_mode (bool, optional): Whether in test mode. Defaults to False
            test_cfg (dict, optional): Test-time configuration
            loop (int, optional): Number of times to repeat dataset. Defaults to 1
        """
        super().__init__()
        self.data_root = data_root
        self.class_names = dict(zip(class_names, range(len(class_names))))
        self.split = split
        self.num_point = num_points
        self.uniform_sampling = uniform_sampling
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        if test_mode:
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

        # check, prepare record
        record_name = f"modelnet40_{self.split}"
        if num_points is not None:
            record_name += f"_{num_points}points"
            if uniform_sampling:
                record_name += "_uniform"
        record_path = os.path.join(self.data_root, f"{record_name}.pth")
        if os.path.isfile(record_path):
            logger.info(f"Loading record: {record_name} ...")
            self.data = torch.load(record_path)
        else:
            logger.info(f"Preparing record: {record_name} ...")
            self.data = {}
            for idx in range(len(self.data_list)):
                data_name = self.data_list[idx]
                logger.info(f"Parsing data [{idx}/{len(self.data_list)}]: {data_name}")
                self.data[data_name] = self.get_data(idx)
            if save_record:
                torch.save(self.data, record_path)

    def get_data(self, idx):
        """
        Get data sample by index.
        
        Args:
            idx (int): Index of the data sample
            
        Returns:
            dict: Data dictionary containing:
                - coord (np.ndarray): Point coordinates
                - normal (np.ndarray): Point normal vectors
                - category (np.ndarray): Object category label
        """
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]
        if data_name in self.data.keys():
            return self.data[data_name]
        else:
            data_shape = "_".join(data_name.split("_")[0:-1])
            data_path = os.path.join(
                self.data_root, data_shape, self.data_list[data_idx] + ".txt"
            )
            data = np.loadtxt(data_path, delimiter=",").astype(np.float32)
            if self.num_point is not None:
                if self.uniform_sampling:
                    with torch.no_grad():
                        mask = pointops.farthest_point_sampling(
                            torch.tensor(data).float().cuda(),
                            torch.tensor([len(data)]).long().cuda(),
                            torch.tensor([self.num_point]).long().cuda(),
                        )
                    data = data[mask.cpu()]
                else:
                    data = data[: self.num_point]
            coord, normal = data[:, 0:3], data[:, 3:6]
            category = np.array([self.class_names[data_shape]])
            return dict(coord=coord, normal=normal, category=category)

    def get_data_list(self):
        """
        Get list of all data samples.
        
        Returns:
            np.ndarray: Array of data sample names
        """
        assert isinstance(self.split, str)
        split_path = os.path.join(
            self.data_root, "modelnet40_{}.txt".format(self.split)
        )
        data_list = np.loadtxt(split_path, dtype="str")
        return data_list

    def get_data_name(self, idx):
        """
        Get name of data sample by index.
        
        Args:
            idx (int): Index of the data sample
            
        Returns:
            str: Name of the data sample
        """
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        """
        Get data sample by index.
        
        Args:
            idx (int): Index of the data sample
            
        Returns:
            dict: Data dictionary with transformed data in test or train mode
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        """
        Get total number of samples.
        
        Returns:
            int: Number of samples times loop factor
        """
        return len(self.data_list) * self.loop

    def prepare_train_data(self, idx):
        """
        Prepare data sample for training.
        
        Args:
            idx (int): Index of the data sample
            
        Returns:
            dict: Transformed data dictionary for training
        """
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        """
        Prepare data sample for testing.
        
        Args:
            idx (int): Index of the data sample
            
        Returns:
            dict: Transformed data dictionary for testing with multiple augmentations
        """
        assert idx < len(self.data_list)
        data_dict = self.get_data(idx)
        category = data_dict.pop("category")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        for i in range(len(data_dict_list)):
            data_dict_list[i] = self.post_transform(data_dict_list[i])
        data_dict = dict(
            voting_list=data_dict_list,
            category=category,
            name=self.get_data_name(idx),
        )
        return data_dict
