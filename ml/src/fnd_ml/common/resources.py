"""What long commands need from the machine: enough free memory, and that it stays awake."""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Self

import psutil

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class InsufficientMemory(RuntimeError):
    """The machine does not have the free memory a command needs."""


def free_ram_gb() -> float:
    """Physical memory that can be given to a new process right now, in GiB."""
    return float(psutil.virtual_memory().available) / 2**30


def peak_ram_mb() -> float:
    """The most physical memory this process has held at one time, in MiB."""
    if sys.platform == "win32":
        return float(psutil.Process().memory_info().peak_wset) / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, else KiB


def require_free_ram(gb: float, what: str) -> None:
    """Raise ``InsufficientMemory`` unless about ``gb`` GiB of memory is available."""
    available = free_ram_gb()
    if available < gb:
        raise InsufficientMemory(
            f"{what} needs about {gb:.1f} GB of free memory, but only {available:.1f} GB is "
            "available. Close other programs and run it again, or pass --force to try anyway."
        )


def _set_execution_state(flags: int) -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(flags)


@contextmanager
def stay_awake() -> Iterator[None]:
    """Ask Windows not to go to sleep while a long command runs; does nothing elsewhere.

    The request belongs to this process and ends with it. The display may still turn off.
    """
    _set_execution_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        _set_execution_state(_ES_CONTINUOUS)


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
