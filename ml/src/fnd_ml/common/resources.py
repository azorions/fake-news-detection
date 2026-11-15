"""Memory check for long commands, so they stop with a clear message instead of being killed."""

from __future__ import annotations

import threading
from typing import Self

import psutil


class InsufficientMemory(RuntimeError):
    """The machine does not have the free memory a command needs."""


def free_ram_gb() -> float:
    """Physical memory that can be given to a new process right now, in GiB."""
    return float(psutil.virtual_memory().available) / 2**30


def require_free_ram(gb: float, what: str) -> None:
    """Raise ``InsufficientMemory`` unless about ``gb`` GiB of memory is available."""
    available = free_ram_gb()
    if available < gb:
        raise InsufficientMemory(
            f"{what} needs about {gb:.1f} GB of free memory, but only {available:.1f} GB is "
            "available. Close other programs and run it again, or pass --force to try anyway."
        )


class PeakMemory:
    """Context manager: how far this process's resident memory rose above its level on entry.

    A thread samples the resident set size, so a spike shorter than ``interval`` seconds can be
    missed. Memory the process already held on entry and reuses is not counted.
    """

    def __init__(self, interval: float = 0.02) -> None:
        self._interval = interval
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._start = self._peak = 0

    def _sample(self) -> None:
        self._peak = max(self._peak, self._process.memory_info().rss)

    def _watch(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def __enter__(self) -> Self:
        self._start = self._peak = self._process.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    @property
    def peak_mb(self) -> float:
        return (self._peak - self._start) / 2**20
