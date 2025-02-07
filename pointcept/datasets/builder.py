"""
Dataset Builder

This module provides the registry and builder functionality for datasets in the
Pointcept framework. It enables dynamic registration and construction of datasets
for various point cloud processing tasks.

The dataset builder uses a registry pattern to allow:
- Dynamic registration of new dataset types
- Configuration-based dataset instantiation
- Flexible dataset composition
- Support for custom datasets

Features:
- Registry-based dataset management
- Dynamic dataset registration
- Configuration-based dataset building
- Support for custom datasets
- Type checking and validation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

DATASETS = Registry("datasets")


def build_dataset(cfg):
    """Build a dataset from configuration.
    
    This function constructs dataset instances based on their configuration
    dictionaries. The datasets are built using the registry pattern, which
    allows for dynamic registration and instantiation.
    
    Args:
        cfg (dict): Dataset configuration containing:
            - type: The registered name of the dataset (required)
            - args: Arguments specific to the dataset type
    
    Returns:
        Dataset: Constructed dataset instance
    
    Example:
        >>> cfg = dict(
        ...     type='ScanNetDataset',
        ...     data_root='data/scannet',
        ...     split='train',
        ...     transform=[...]
        ... )
        >>> dataset = build_dataset(cfg)
    
    Note:
        - Each dataset must be registered in the DATASETS registry
        - Invalid dataset configurations will raise exceptions
    """
    return DATASETS.build(cfg)
