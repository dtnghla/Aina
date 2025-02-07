"""
Logger Utils

This module provides a comprehensive logging system for the Pointcept framework,
with support for distributed training, colorful output, and multiple output destinations.
Modified from mmcv to provide enhanced logging capabilities.

Features:
- Hierarchical logger management
- Distributed training support with rank-aware logging
- Colorful console output with warning and error highlighting
- File-based logging with configurable modes
- Silent mode support
- API usage tracking

Components:
- _ColorfulFormatter: Custom formatter for colored console output
- get_logger: Main function for obtaining configured loggers
- print_log: Unified interface for logging messages
- get_root_logger: Convenience function for root logger access
- _log_api_usage: Internal function for API usage tracking

Modified from mmcv
Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import logging
import torch
import torch.distributed as dist

from termcolor import colored

logger_initialized = {}
root_status = 0


class _ColorfulFormatter(logging.Formatter):
    """Custom logging formatter that adds color to console output.
    
    This formatter adds color highlighting to warning and error messages,
    making them more visible in the console output. It uses the termcolor
    package for ANSI color formatting.
    
    Args:
        *args: Arguments passed to logging.Formatter
        **kwargs: Keyword arguments passed to logging.Formatter.
            Must include 'root_name' which is used as the logger name prefix.
    
    Note:
        - Warnings are displayed in blinking red
        - Errors are displayed in blinking, underlined red
        - Regular messages are displayed without color modification
    """

    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        """Format the log message with appropriate color if needed.
        
        Args:
            record (LogRecord): The log record to format.
        
        Returns:
            str: The formatted log message, potentially with color codes.
        """
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="a", color=False):
    """Initialize and get a logger by name.
    
    This function provides a centralized way to obtain loggers with consistent
    configuration. It supports both console and file output, with special handling
    for distributed training environments.
    
    Args:
        name (str): Logger name. Used for hierarchical logger organization.
        log_file (str, optional): Path to the log file. If provided, enables
            file-based logging for rank 0 process.
        log_level (int, optional): Logging level (e.g., logging.INFO). Only
            affects rank 0 process; others use ERROR level. Default: logging.INFO.
        file_mode (str, optional): File opening mode ('a' for append, 'w' for
            write). Default: 'a'.
        color (bool, optional): Whether to enable colored console output.
            Default: False.
    
    Returns:
        logging.Logger: Configured logger instance.
    
    Note:
        - In distributed training, only rank 0 process logs at the specified level
        - Child loggers inherit parent logger's configuration
        - Loggers are cached to avoid duplicate initialization
    
    Example:
        >>> logger = get_logger('train', 'train.log', logging.INFO)
        >>> logger.info('Starting training...')
    """
    logger = logging.getLogger(name)

    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    logger.propagate = False

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    plain_formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    )
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
        )
    else:
        formatter = plain_formatter
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message with flexible output options.
    
    This function provides a unified interface for logging messages through
    different channels (print, logger, or silent).
    
    Args:
        msg (str): The message to log.
        logger (Union[logging.Logger, str, None]): The logger to use. Special values:
            - None: Use print() function
            - "silent": Suppress output
            - str: Use as name to get logger
            - logging.Logger: Use directly
        level (int, optional): Logging level when using a logger.
            Default: logging.INFO.
    
    Raises:
        TypeError: If logger is not one of the supported types.
    
    Example:
        >>> print_log('Starting epoch 1', 'train')  # Uses 'train' logger
        >>> print_log('Debug info', None)  # Uses print()
        >>> print_log('Silent message', 'silent')  # No output
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, "
            f'"silent" or None, but got {type(logger)}'
        )


def get_root_logger(log_file=None, log_level=logging.INFO, file_mode="a"):
    """Get the root logger for the Pointcept framework.
    
    This is a convenience function that returns a logger named 'pointcept'
    with the specified configuration. It's typically used as the main logger
    for the entire framework.
    
    Args:
        log_file (str, optional): Path to the log file. If provided, enables
            file-based logging for rank 0 process.
        log_level (int, optional): Logging level (e.g., logging.INFO). Only
            affects rank 0 process; others use ERROR level. Default: logging.INFO.
        file_mode (str, optional): File opening mode ('a' for append, 'w' for
            write). Default: 'a'.
    
    Returns:
        logging.Logger: The root logger for Pointcept.
    
    Example:
        >>> logger = get_root_logger('train.log')
        >>> logger.info('Pointcept framework initialized')
    """
    logger = get_logger(
        name="pointcept", log_file=log_file, log_level=log_level, file_mode=file_mode
    )
    return logger


def _log_api_usage(identifier: str):
    """Log the usage of Pointcept API components.
    
    Internal function used to track the usage of different Pointcept components
    for monitoring and improvement purposes.
    
    Args:
        identifier (str): The identifier of the API component being used.
    
    Note:
        This function uses PyTorch's internal API usage tracking mechanism
        and ensures each identifier is only logged once per process.
    """
    torch._C._log_api_usage_once("pointcept." + identifier)
