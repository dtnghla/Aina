# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# -*- coding: utf-8 -*-

from time import perf_counter
from typing import Optional


class Timer:
    """
    A timer class for measuring elapsed time with pause/resume functionality.
    
    This timer provides high-precision timing using performance counter,
    with support for pausing and resuming the timer, and calculating
    average time between starts/pauses.

    Example:
        >>> timer = Timer()
        >>> # Do some work
        >>> elapsed = timer.seconds()
        >>> timer.pause()
        >>> # Timer is paused
        >>> timer.resume()
        >>> avg_time = timer.avg_seconds()
    """

    def __init__(self) -> None:
        """
        Initialize the timer and start it immediately.
        """
        self.reset()

    def reset(self) -> None:
        """
        Reset the timer to its initial state.
        
        This method:
        - Sets the start time to current time
        - Clears any paused state
        - Resets the total paused time to 0
        - Resets the count of starts to 1
        """
        self._start = perf_counter()
        self._paused: Optional[float] = None
        self._total_paused = 0
        self._count_start = 1

    def pause(self) -> None:
        """
        Pause the timer.
        
        Raises:
            ValueError: If the timer is already paused.
        """
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")
        self._paused = perf_counter()

    def is_paused(self) -> bool:
        """
        Check if the timer is currently paused.

        Returns:
            bool: True if the timer is paused, False otherwise.
        """
        return self._paused is not None

    def resume(self) -> None:
        """
        Resume the timer from a paused state.
        
        This method:
        - Calculates the duration of the pause
        - Adds it to total paused time
        - Increments the start counter
        
        Raises:
            ValueError: If the timer is not paused.
        """
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")
        self._total_paused += perf_counter() - self._paused
        self._paused = None
        self._count_start += 1

    def seconds(self) -> float:
        """
        Get the total elapsed time in seconds, excluding paused time.
        
        Returns:
            float: Total number of seconds since start/reset, minus any time
                  spent in a paused state.
        """
        if self._paused is not None:
            end_time: float = self._paused  # type: ignore
        else:
            end_time = perf_counter()
        return end_time - self._start - self._total_paused

    def avg_seconds(self) -> float:
        """
        Get the average time in seconds between starts/resets and pauses.
        
        Returns:
            float: Average number of seconds between each start/reset and pause.
                  Calculated as total active time divided by number of starts.
        """
        return self.seconds() / self._count_start
