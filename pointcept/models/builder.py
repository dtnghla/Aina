"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

MODELS = Registry("models")
MODULES = Registry("modules")


def build_model(cfg):
    """
    Build a model from configuration.

    Args:
        cfg (dict): Configuration dictionary containing model specifications.
                   Must include 'type' key specifying the model type.

    Returns:
        nn.Module: Instantiated model based on the configuration.
    """
    return MODELS.build(cfg)
