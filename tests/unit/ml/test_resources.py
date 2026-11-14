import pytest

from fnd_ml.common import resources
from fnd_ml.common.resources import InsufficientMemory, free_ram_gb, require_free_ram


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
