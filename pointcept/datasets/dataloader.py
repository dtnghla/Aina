"""
Multi-Dataset Dataloader

This module provides functionality for loading data from multiple datasets
simultaneously during training. It enables training on multiple datasets with
different sampling ratios and batch sizes.

Features:
- Support for multiple datasets with different characteristics
- Configurable sampling ratios between datasets
- Distributed training support
- Automatic worker initialization
- Memory-efficient data loading
- Mixed batch sampling

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
import weakref
import torch
import torch.utils.data

import pointcept.utils.comm as comm
from pointcept.datasets.utils import point_collate_fn
from pointcept.datasets import ConcatDataset
from pointcept.utils.env import set_seed


class MultiDatasetDummySampler:
    """Dummy sampler for multi-dataset loading.
    
    This sampler provides compatibility with distributed training by
    forwarding epoch information to the underlying dataset samplers.
    It acts as a proxy between the dataloader and individual dataset samplers.
    
    Attributes:
        dataloader: Weak reference to the parent dataloader
    """
    def __init__(self):
        self.dataloader = None

    def set_epoch(self, epoch):
        """Set the epoch for distributed sampling.
        
        Args:
            epoch (int): Current epoch number
            
        Note:
            This method is called by the training loop in distributed settings
            to ensure proper shuffling of data across epochs.
        """
        if comm.get_world_size() > 1:
            for dataloader in self.dataloader.dataloaders:
                dataloader.sampler.set_epoch(epoch)
        return


class MultiDatasetDataloader:
    """Dataloader for handling multiple datasets simultaneously.
    
    This dataloader enables training on multiple datasets with different sampling
    ratios. It ensures that batches from each dataset are properly mixed according
    to specified ratios while maintaining memory efficiency.
    
    The overall length is determined by the main dataset (first dataset) and the
    loop parameter of the concat dataset. Batches are sampled from datasets
    according to their specified ratios.
    
    Attributes:
        datasets (list): List of dataset instances
        ratios (list): Sampling ratios for each dataset
        dataloaders (list): List of individual dataset dataloaders
        sampler: Dummy sampler for distributed training compatibility
        
    Args:
        concat_dataset (ConcatDataset): Combined dataset instance
        batch_size_per_gpu (int): Batch size for each GPU
        num_worker_per_gpu (int): Number of workers per GPU
        mix_prob (float): Probability of mixing batches. Default: 0
        seed (int, optional): Random seed for workers. Default: None
        
    Example:
        >>> concat_dataset = ConcatDataset([dataset1, dataset2])
        >>> dataloader = MultiDatasetDataloader(
        ...     concat_dataset,
        ...     batch_size_per_gpu=32,
        ...     num_worker_per_gpu=4
        ... )
    """

    def __init__(
        self,
        concat_dataset: ConcatDataset,
        batch_size_per_gpu: int,
        num_worker_per_gpu: int,
        mix_prob=0,
        seed=None,
    ):
        self.datasets = concat_dataset.datasets
        self.ratios = [dataset.loop for dataset in self.datasets]
        # reset data loop, original loop serve as ratios
        for dataset in self.datasets:
            dataset.loop = 1
        # determine union training epoch by main dataset
        self.datasets[0].loop = concat_dataset.loop
        # build sub-dataloaders
        num_workers = num_worker_per_gpu // len(self.datasets)
        self.dataloaders = []
        for dataset_id, dataset in enumerate(self.datasets):
            if comm.get_world_size() > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = None

            init_fn = (
                partial(
                    self._worker_init_fn,
                    dataset_id=dataset_id,
                    num_workers=num_workers,
                    num_datasets=len(self.datasets),
                    rank=comm.get_rank(),
                    seed=seed,
                )
                if seed is not None
                else None
            )
            self.dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size_per_gpu,
                    shuffle=(sampler is None),
                    num_workers=num_worker_per_gpu,
                    sampler=sampler,
                    collate_fn=partial(point_collate_fn, mix_prob=mix_prob),
                    pin_memory=True,
                    worker_init_fn=init_fn,
                    drop_last=True,
                    persistent_workers=True,
                )
            )
        self.sampler = MultiDatasetDummySampler()
        self.sampler.dataloader = weakref.proxy(self)

    def __iter__(self):
        """Create an iterator over the datasets.
        
        This method implements the iteration logic that samples batches from
        different datasets according to their ratios. It ensures that the
        sampling follows the specified ratios while handling dataset exhaustion.
        
        Yields:
            batch: A batch of data from one of the datasets
        """
        iterator = [iter(dataloader) for dataloader in self.dataloaders]
        while True:
            for i in range(len(self.ratios)):
                for _ in range(self.ratios[i]):
                    try:
                        batch = next(iterator[i])
                    except StopIteration:
                        if i == 0:
                            return
                        else:
                            iterator[i] = iter(self.dataloaders[i])
                            batch = next(iterator[i])
                    yield batch

    def __len__(self):
        """Calculate the total length of the combined dataloader.
        
        The length is determined by the main dataset's length and the sampling
        ratios of all datasets.
        
        Returns:
            int: Total number of batches per epoch
        """
        main_data_loader_length = len(self.dataloaders[0])
        return (
            main_data_loader_length // self.ratios[0] * sum(self.ratios)
            + main_data_loader_length % self.ratios[0]
        )

    @staticmethod
    def _worker_init_fn(worker_id, num_workers, dataset_id, num_datasets, rank, seed):
        """Initialize workers with appropriate random seeds.
        
        This method ensures that each worker has a unique random seed that is
        consistent across epochs but different between workers and datasets.
        
        Args:
            worker_id (int): ID of the current worker
            num_workers (int): Total number of workers
            dataset_id (int): ID of the current dataset
            num_datasets (int): Total number of datasets
            rank (int): Rank of the current process
            seed (int): Base random seed
        """
        worker_seed = (
            num_workers * num_datasets * rank
            + num_workers * dataset_id
            + worker_id
            + seed
        )
        set_seed(worker_seed)
