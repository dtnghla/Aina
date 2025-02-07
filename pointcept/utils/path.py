"""
Path Utils

This module provides utilities for file and directory path operations,
with support for both string paths and pathlib.Path objects. It includes
functions for file operations, directory scanning, and version control
system root finding.

Features:
- Path type checking and validation
- Safe file opening and existence checking
- Directory creation and symlink management
- Recursive directory scanning with filtering
- VCS root directory detection

Note:
    Originally from OpenMMLab. Modified for use in Pointcept.
"""

import os
import os.path as osp
from pathlib import Path

from .misc import is_str


def is_filepath(x):
    """Check if the input is a valid file path.
    
    Args:
        x: Object to check
    
    Returns:
        bool: True if x is a string or Path object, False otherwise
    
    Example:
        >>> is_filepath('path/to/file.txt')  # True
        >>> is_filepath(Path('file.txt'))    # True
        >>> is_filepath(123)                 # False
    """
    return is_str(x) or isinstance(x, Path)


def fopen(filepath, *args, **kwargs):
    """Open a file with support for both string paths and Path objects.
    
    This function provides a unified interface for opening files, handling
    both string paths and pathlib.Path objects.
    
    Args:
        filepath (str or Path): Path to the file to open
        *args: Additional positional arguments passed to open()
        **kwargs: Additional keyword arguments passed to open()
    
    Returns:
        file object: The opened file object
    
    Raises:
        ValueError: If filepath is neither a string nor a Path object
    
    Example:
        >>> with fopen('file.txt', 'r') as f:
        ...     content = f.read()
    """
    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)
    raise ValueError("`filepath` should be a string or a Path")


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    """Check if a file exists.
    
    Args:
        filename (str): Path to the file to check
        msg_tmpl (str, optional): Message template for the error.
            Default: 'file "{}" does not exist'
    
    Raises:
        FileNotFoundError: If the file does not exist
    
    Example:
        >>> check_file_exist('existing.txt')  # OK
        >>> check_file_exist('missing.txt')   # Raises FileNotFoundError
    """
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    """Create a directory if it does not exist.
    
    Args:
        dir_name (str): Path to the directory to create
        mode (int, optional): Permission mode for the directory.
            Default: 0o777
    
    Note:
        - Empty directory names are ignored
        - User home directory symbols (e.g., ~) are expanded
        - Existing directories are not modified
    
    Example:
        >>> mkdir_or_exist('~/new_dir')  # Creates directory in home
        >>> mkdir_or_exist('')           # Does nothing
    """
    if dir_name == "":
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    """Create a symbolic link, optionally overwriting existing links.
    
    Args:
        src (str): Path to the source file/directory
        dst (str): Path where the symbolic link should be created
        overwrite (bool, optional): Whether to overwrite existing links.
            Default: True
        **kwargs: Additional arguments passed to os.symlink()
    
    Note:
        If overwrite is True and a file exists at dst, it will be removed
        before creating the new link.
    
    Example:
        >>> symlink('source.txt', 'link.txt')  # Creates symbolic link
        >>> symlink('source.txt', 'link.txt', overwrite=False)  # Raises error if exists
    """
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    """Scan a directory to find files with specific suffixes.
    
    This function scans a directory and yields paths to files that match
    the specified suffix(es). It supports recursive scanning and case
    sensitivity options.
    
    Args:
        dir_path (str or Path): Path to the directory to scan
        suffix (str or tuple[str], optional): File suffix(es) to match.
            If None, all files are included. Default: None
        recursive (bool, optional): Whether to scan subdirectories.
            Default: False
        case_sensitive (bool, optional): Whether to match suffixes case-sensitively.
            Default: True
    
    Yields:
        str: Relative paths to matching files
    
    Raises:
        TypeError: If dir_path is not a string or Path object
        TypeError: If suffix is not a string or tuple of strings
    
    Example:
        >>> for file in scandir('data', '.txt', recursive=True):
        ...     print(file)  # Prints paths to all .txt files
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = (
            suffix.lower()
            if isinstance(suffix, str)
            else tuple(item.lower() for item in suffix)
        )

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        """Internal recursive scanning function.
        
        Args:
            dir_path (str): Current directory path
            suffix (str or tuple): File suffix(es) to match
            recursive (bool): Whether to scan subdirectories
            case_sensitive (bool): Whether to match case-sensitively
        
        Yields:
            str: Relative paths to matching files
        """
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive, case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)


def find_vcs_root(path, markers=(".git",)):
    """Find the root directory of a version control system repository.
    
    This function searches upward from the given path to find a directory
    containing one of the specified VCS marker files/directories.
    
    Args:
        path (str): Starting path (file or directory)
        markers (tuple[str], optional): VCS directory/file markers to look for.
            Default: (".git",)
    
    Returns:
        str or None: Path to the VCS root directory, or None if not found
    
    Example:
        >>> find_vcs_root('src/module/file.py')  # Returns git repo root
        >>> find_vcs_root('/non/vcs/path')       # Returns None
    
    Note:
        - If path is a file, starts search from its parent directory
        - Stops searching when reaching the filesystem root
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None
