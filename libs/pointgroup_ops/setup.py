import os
from sys import argv
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)


def _argparse(pattern, argv, is_flag=True, is_list=False):
    """
    Parse command line arguments based on a pattern.
    
    Args:
        pattern (str): The pattern to search for in the arguments.
        argv (list): List of command line arguments.
        is_flag (bool, optional): If True, treats the pattern as a flag argument. Defaults to True.
        is_list (bool, optional): If True, parses the value as a comma-separated list. Defaults to False.
    
    Returns:
        tuple: A pair containing:
            - If is_flag=True: (bool, list) where bool indicates if pattern was found
            - If is_flag=False and is_list=True: (list, list) where first list contains parsed values
            - If is_flag=False and is_list=False: (str, list) where str is the parsed value
            The second element is always the modified argv list with parsed arguments removed.
    """
    if is_flag:
        found = pattern in argv
        if found:
            argv.remove(pattern)
        return found, argv
    else:
        arr = [arg for arg in argv if pattern == arg.split("=")[0]]
        if is_list:
            if len(arr) == 0:  # not found
                return False, argv
            else:
                assert "=" in arr[0], f"{arr[0]} requires a value."
                argv.remove(arr[0])
                val = arr[0].split("=")[1]
                if "," in val:
                    return val.split(","), argv
                else:
                    return [val], argv
        else:
            if len(arr) == 0:  # not found
                return False, argv
            else:
                assert "=" in arr[0], f"{arr[0]} requires a value."
                argv.remove(arr[0])
                return arr[0].split("=")[1], argv


INCLUDE_DIRS, argv = _argparse("--include_dirs", argv, False, is_list=True)
include_dirs = []
if not (INCLUDE_DIRS is False):
    include_dirs += INCLUDE_DIRS

setup(
    name="pointgroup_ops",
    packages=["pointgroup_ops"],
    package_dir={"pointgroup_ops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointgroup_ops_cuda",
            sources=["src/bfs_cluster.cpp", "src/bfs_cluster_kernel.cu"],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    include_dirs=[*include_dirs],
    cmdclass={"build_ext": BuildExtension},
)
