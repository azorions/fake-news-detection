import time

import numpy as np
import pytest

from fnd_ml.common import resources
from fnd_ml.common.resources import (
    InsufficientMemory,
    PeakMemory,
    free_ram_gb,
    require_free_ram,
)


def test_free_ram_is_a_positive_number_of_gigabytes() -> None:
    assert 0 < free_ram_gb() < 4096


def test_enough_memory_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "free_ram_gb", lambda: 8.0)

    require_free_ram(3.0, "The data build")


def test_too_little_memory_names_both_numbers_and_the_way_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "free_ram_gb", lambda: 0.4)

    with pytest.raises(InsufficientMemory) as error:
        require_free_ram(3.0, "The data build")

    message = str(error.value)
    assert "The data build" in message
    assert "3.0 GB" in message and "0.4 GB" in message
    assert "--force" in message


def test_peak_memory_sees_memory_that_was_used_and_released_inside_the_block() -> None:
    with PeakMemory(interval=0.005) as memory:
        block = np.ones(64 * 2**20 // 8)  # 64 MB of float64, every page written
        time.sleep(0.05)
        del block

    assert 48 <= memory.peak_mb < 512


def test_peak_memory_is_about_zero_when_nothing_is_allocated() -> None:
    with PeakMemory() as memory:
        pass

    assert 0 <= memory.peak_mb < 16


def test_stay_awake_asks_for_the_whole_block_and_lets_go_afterwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(resources, "_set_execution_state", calls.append)

    with resources.stay_awake():
        assert calls == [0x80000001]  # continuous + system required

    assert calls == [0x80000001, 0x80000000]


def test_stay_awake_lets_go_when_the_command_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(resources, "_set_execution_state", calls.append)

    with pytest.raises(RuntimeError), resources.stay_awake():
        raise RuntimeError("the run died")

    assert calls[-1] == 0x80000000


def test_stay_awake_works_on_this_machine() -> None:
    with resources.stay_awake():
        pass
