"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    """
    Main worker function for testing.
    
    This function sets up the testing environment and runs the test procedure
    using the specified tester from the configuration.

    Args:
        cfg: Configuration object containing test settings and parameters.
    """
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    """
    Main entry point for the testing script.
    
    This function:
    1. Parses command line arguments
    2. Loads and parses the configuration file
    3. Launches the testing process with specified distributed computing settings
    
    The testing can be run on multiple GPUs and machines based on the provided
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
