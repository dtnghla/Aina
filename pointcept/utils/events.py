"""
Events Utils

This module provides a comprehensive event logging and storage system for tracking training metrics,
visualizations, and other data during model training. It is modified from Detectron2's event system
to provide robust logging capabilities.

Features:
- Centralized event storage with context management
- Multiple output formats (JSON, TensorBoard)
- Support for scalars, images, and histograms
- Smoothing and statistics calculation
- ETA estimation and progress tracking
- Exception handling and logging

Components:
- EventStorage: Central storage for all events and metrics
- EventWriter: Base class for different output formats
- JSONWriter: Writes metrics to JSON files
- TensorboardXWriter: Writes to TensorBoard
- CommonMetricPrinter: Prints metrics to console
- HistoryBuffer: Stores historical values with statistics
- ExceptionWriter: Handles and logs exceptions

Modified from Detectron2
Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import datetime
import json
import logging
import os
import time
import torch
import numpy as np
import traceback
import sys

from typing import List, Optional, Tuple
from collections import defaultdict
from contextlib import contextmanager

__all__ = [
    "get_event_storage",
    "JSONWriter",
    "TensorboardXWriter",
    "CommonMetricPrinter",
    "EventStorage",
    "ExceptionWriter",
]

_CURRENT_STORAGE_STACK = []


def get_event_storage():
    """Get the current EventStorage instance.
    
    This function provides access to the active EventStorage instance. It must be called
    within an EventStorage context manager.
    
    Returns:
        EventStorage: The currently active EventStorage instance.
    
    Raises:
        AssertionError: If called outside an EventStorage context.
    
    Example:
        >>> with EventStorage() as storage:
        ...     storage = get_event_storage()
        ...     storage.put_scalar("loss", 0.5)
    """
    assert len(
        _CURRENT_STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]


class EventWriter:
    """Base class for writers that process events from EventStorage.
    
    This is an abstract base class that defines the interface for event writers.
    Subclasses should implement the write() method to process events in their
    specific format.
    """

    def write(self):
        """Process and write events from the current storage.
        
        This method should be implemented by subclasses to write events in their
        specific format (e.g., JSON, TensorBoard).
        """
        raise NotImplementedError

    def close(self):
        """Clean up resources used by the writer.
        
        This method should be implemented by subclasses if they need to perform
        cleanup operations when the writer is no longer needed.
        """
        pass


class JSONWriter(EventWriter):
    """Write scalars to a JSON file.
    
    This writer saves metrics as one JSON object per line for easy parsing.
    Each line contains metrics for a single iteration, including the iteration
    number and all scalar values recorded for that iteration.
    
    The writer supports median smoothing of values over a specified window size
    for metrics that have smoothing_hint=True.
    
    Args:
        json_file (str): Path to the output JSON file. New data will be appended
            if the file exists.
        window_size (int): Window size for median smoothing of scalars with
            smoothing_hint=True. Default: 20.
    
    Example JSON format:
        {"iteration": 1, "loss": 1.0, "accuracy": 0.8}
        {"iteration": 2, "loss": 0.9, "accuracy": 0.85}
    """
    
    def __init__(self, json_file, window_size=20):
        self._file_handle = open(json_file, "a")
        self._window_size = window_size
        self._last_write = -1

    def write(self):
        """Write all new scalar values to the JSON file.
        
        Retrieves all scalar values from the current storage that haven't been
        written yet and writes them as JSON objects, one per line.
        """
        storage = get_event_storage()
        to_save = defaultdict(dict)

        for k, (v, iter) in storage.latest_with_smoothing_hint(
            self._window_size
        ).items():
            # keep scalars that have not been written
            if iter <= self._last_write:
                continue
            to_save[iter][k] = v
        if len(to_save):
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)

        for itr, scalars_per_iter in to_save.items():
            scalars_per_iter["iteration"] = itr
            self._file_handle.write(json.dumps(scalars_per_iter, sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        """Close the JSON file handle."""
        self._file_handle.close()


class TensorboardXWriter(EventWriter):
    """Write scalars, images, and histograms to TensorBoard.
    
    This writer supports writing multiple types of data to TensorBoard:
    - Scalar values (with optional smoothing)
    - Images
    - Histograms
    
    Args:
        log_dir (str): Directory where TensorBoard events will be written
        window_size (int): Window size for median smoothing of scalars.
            Default: 20.
        **kwargs: Additional arguments passed to torch.utils.tensorboard.SummaryWriter
    
    Note:
        This writer requires the torch.utils.tensorboard package to be installed.
    """
    
    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        self._window_size = window_size
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir, **kwargs)
        self._last_write = -1

    def write(self):
        """Write all pending data to TensorBoard.
        
        This includes:
        - Scalar values (with smoothing if enabled)
        - Images stored in storage._vis_data
        - Histograms stored in storage._histograms
        """
        storage = get_event_storage()
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(
            self._window_size
        ).items():
            if iter > self._last_write:
                self._writer.add_scalar(k, v, iter)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write

        # storage.put_{image,histogram} is only meant to be used by
        # tensorboard writer. So we access its internal fields directly from here.
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                self._writer.add_image(img_name, img, step_num)
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                self._writer.add_histogram_raw(**params)
            storage.clear_histograms()

    def close(self):
        """Close the TensorBoard writer."""
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()


class CommonMetricPrinter(EventWriter):
    """Print common training metrics to the terminal.
    
    This printer outputs common training metrics including:
    - Iteration time
    - ETA (estimated time of arrival)
    - Memory usage
    - All losses
    - Learning rate
    
    It applies median smoothing over a window of values for stability.
    
    Args:
        max_iter (int, optional): Maximum number of iterations. Used to compute ETA.
        window_size (int): Window size for smoothing. Default: 20.
    """

    def __init__(self, max_iter: Optional[int] = None, window_size: int = 20):
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
            window_size (int): the losses will be median-smoothed by this window size
        """
        self.logger = logging.getLogger(__name__)
        self._max_iter = max_iter
        self._window_size = window_size
        self._last_write = (
            None  # (step, time) of last call to write(). Used to compute ETA
        )

    def _get_eta(self, storage) -> Optional[str]:
        if self._max_iter is None:
            return ""
        iteration = storage.iter
        try:
            eta_seconds = storage.history("time").median(1000) * (
                self._max_iter - iteration - 1
            )
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            return str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            # estimate eta on our own - more noisy
            eta_string = None
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())
            return eta_string

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None
        try:
            iter_time = storage.history("time").global_avg()
        except KeyError:
            iter_time = None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        eta_string = self._get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            " {eta}iter: {iter}  {losses}  {time}{data_time}lr: {lr}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                losses="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(self._window_size))
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),
                time=(
                    "time: {:.4f}  ".format(iter_time) if iter_time is not None else ""
                ),
                data_time=(
                    "data_time: {:.4f}  ".format(data_time)
                    if data_time is not None
                    else ""
                ),
                lr=lr,
                memory=(
                    "max_mem: {:.0f}M".format(max_mem_mb)
                    if max_mem_mb is not None
                    else ""
                ),
            )
        )


class EventStorage:
    """Central storage for training metrics and events.
    
    This class provides a centralized storage for training metrics, including:
    - Scalar values (losses, learning rate, etc.)
    - Images for visualization
    - Histograms
    
    It supports:
    - Automatic smoothing of values
    - History tracking with statistics (median, average)
    - Nested name scopes for metric organization
    - Context management for safe storage access
    
    Args:
        start_iter (int): Starting iteration number. Default: 0.
    
    Example:
        >>> with EventStorage() as storage:
        ...     storage.put_scalar("loss", 0.5)
        ...     with storage.name_scope("validation"):
        ...         storage.put_scalar("accuracy", 0.95)
    """

    def __init__(self, start_iter=0):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._history = defaultdict(AverageMeter)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter
        self._current_prefix = ""
        self._vis_data = []
        self._histograms = []

    # def put_image(self, img_name, img_tensor):
    #     """
    #     Add an `img_tensor` associated with `img_name`, to be shown on
    #     tensorboard.
    #     Args:
    #         img_name (str): The name of the image to put into tensorboard.
    #         img_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
    #             Tensor of shape `[channel, height, width]` where `channel` is
    #             3. The image format should be RGB. The elements in img_tensor
    #             can either have values in [0, 1] (float32) or [0, 255] (uint8).
    #             The `img_tensor` will be visualized in tensorboard.
    #     """
    #     self._vis_data.append((img_name, img_tensor, self._iter))

    def put_scalar(self, name, value, n=1, smoothing_hint=False):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.
        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.
                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """
        name = self._current_prefix + name
        history = self._history[name]
        history.update(value, n)
        self._latest_scalars[name] = (value, self._iter)

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert (
                existing_hint == smoothing_hint
            ), "Scalar {} was put with a different smoothing_hint!".format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint

    # def put_scalars(self, *, smoothing_hint=True, **kwargs):
    #     """
    #     Put multiple scalars from keyword arguments.
    #     Examples:
    #         storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
    #     """
    #     for k, v in kwargs.items():
    #         self.put_scalar(k, v, smoothing_hint=smoothing_hint)
    #
    # def put_histogram(self, hist_name, hist_tensor, bins=1000):
    #     """
    #     Create a histogram from a tensor.
    #     Args:
    #         hist_name (str): The name of the histogram to put into tensorboard.
    #         hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
    #             into a histogram.
    #         bins (int): Number of histogram bins.
    #     """
    #     ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()
    #
    #     # Create a histogram with PyTorch
    #     hist_counts = torch.histc(hist_tensor, bins=bins)
    #     hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32)
    #
    #     # Parameter for the add_histogram_raw function of SummaryWriter
    #     hist_params = dict(
    #         tag=hist_name,
    #         min=ht_min,
    #         max=ht_max,
    #         num=len(hist_tensor),
    #         sum=float(hist_tensor.sum()),
    #         sum_squares=float(torch.sum(hist_tensor**2)),
    #         bucket_limits=hist_edges[1:].tolist(),
    #         bucket_counts=hist_counts.tolist(),
    #         global_step=self._iter,
    #     )
    #     self._histograms.append(hist_params)

    def history(self, name):
        """
        Returns:
            AverageMeter: the history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[str -> (float, int)]: mapping from the name of each scalar to the most
                recent value and the iteration number its added.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.
        This provides a default behavior that other writers can use.
        """
        result = {}
        for k, (v, itr) in self._latest_scalars.items():
            result[k] = (
                self._history[k].median(window_size) if self._smoothing_hints[k] else v,
                itr,
            )
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """
        User should either: (1) Call this function to increment storage.iter when needed. Or
        (2) Set `storage.iter` to the correct iteration number before each iteration.
        The storage will then be able to associate the new data with an iteration number.
        """
        self._iter += 1

    @property
    def iter(self):
        """
        Returns:
            int: The current iteration number. When used together with a trainer,
                this is ensured to be the same as trainer.iter.
        """
        return self._iter

    @iter.setter
    def iter(self, val):
        self._iter = int(val)

    @property
    def iteration(self):
        # for backward compatibility
        return self._iter

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self
        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix

    def clear_images(self):
        """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        """
        self._vis_data = []

    def clear_histograms(self):
        """
        Delete all the stored histograms for visualization.
        This should be called after histograms are written to tensorboard.
        """
        self._histograms = []

    def reset_history(self, name):
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        ret.reset()

    def reset_histories(self):
        for name in self._history.keys():
            self._history[name].reset()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.total += val * n
        self.count += n
        self.avg = self.total / self.count


class HistoryBuffer:
    """Buffer for storing historical values with statistical operations.
    
    This class maintains a buffer of historical values and provides various
    statistical operations like median, average, and global average.
    
    Args:
        max_length (int): Maximum number of values to store. Default: 1000000.
    
    Methods:
        update(value, iteration): Add a new value to the buffer
        median(window_size): Get median of recent values
        avg(window_size): Get average of recent values
        global_avg(): Get average of all values
        latest(): Get most recent value
        values(): Get all stored values and iterations
    """

    def __init__(self, max_length: int = 1000000) -> None:
        """
        Args:
            max_length: maximal number of values that can be stored in the
                buffer. When the capacity of the buffer is exhausted, old
                values will be removed.
        """
        self._max_length: int = max_length
        self._data: List[Tuple[float, float]] = []  # (value, iteration) pairs
        self._count: int = 0
        self._global_avg: float = 0

    def update(self, value: float, iteration: Optional[float] = None) -> None:
        """
        Add a new scalar value produced at certain iteration. If the length
        of the buffer exceeds self._max_length, the oldest element will be
        removed from the buffer.
        """
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self) -> float:
        """
        Return the latest scalar value added to the buffer.
        """
        return self._data[-1][0]

    def median(self, window_size: int) -> float:
        """
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int) -> float:
        """
        Return the mean of the latest `window_size` values in the buffer.
        """
        return np.mean([x[0] for x in self._data[-window_size:]])

    def global_avg(self) -> float:
        """
        Return the mean of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    def values(self) -> List[Tuple[float, float]]:
        """
        Returns:
            list[(number, iteration)]: content of the current buffer.
        """
        return self._data


class ExceptionWriter:
    """Context manager for handling and logging exceptions.
    
    This class provides a context manager that catches and logs any exceptions
    that occur within its scope. It's useful for debugging and error tracking
    during training.
    
    Example:
        >>> with ExceptionWriter():
        ...     # Code that might raise an exception
        ...     raise ValueError("Example error")
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            tb = traceback.format_exception(exc_type, exc_val, exc_tb)
            formatted_tb_str = "".join(tb)
            self.logger.error(formatted_tb_str)
            sys.exit(1)  # This prevents double logging the error to the console
