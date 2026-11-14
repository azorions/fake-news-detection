"""Memory check for long commands, so they stop with a clear message instead of being killed."""

from __future__ import annotations

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
