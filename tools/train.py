"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    """
    Main worker function for training.
    
    This function sets up the training environment and runs the training procedure
    using the specified trainer from the configuration.

    Args:
        cfg: Configuration object containing training settings and parameters.
    """
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    """
    Main entry point for the training script.
    
    This function:
    1. Parses command line arguments
    2. Loads and parses the configuration file
    3. Launches the training process with specified distributed computing settings
    
    The training can be run on multiple GPUs and machines based on the provided
    command line arguments.
    """
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
