import pytest

from fnd_ml.common.truncation import InputSpec, build_input, head_tail, pad_batch


def test_head_tail_keeps_both_ends() -> None:
    ids, cut = head_tail(list(range(1000)), 100)

    assert cut
    assert ids[:75] == list(range(75))
    assert ids[75:] == list(range(975, 1000))


def test_short_body_is_untouched() -> None:
    assert head_tail([1, 2, 3], 100) == ([1, 2, 3], False)


def test_a_body_that_fits_exactly_is_untouched() -> None:
    assert head_tail(list(range(100)), 100) == (list(range(100)), False)


def test_head_share_moves_the_cut() -> None:
    ids, cut = head_tail(list(range(1000)), 10, head_share=0.5)

    assert cut
    assert ids == [0, 1, 2, 3, 4, 995, 996, 997, 998, 999]


def test_a_budget_of_one_keeps_the_first_token() -> None:
    assert head_tail([4, 5, 6], 1) == ([4], True)


def test_no_budget_keeps_nothing() -> None:
    assert head_tail([4, 5, 6], 0) == ([], True)


def test_build_input_layout_and_length() -> None:
    ids, cut = build_input([7] * 100, list(range(100, 2000)), cls_id=1, sep_id=2)

    assert len(ids) == 512
    assert cut
    assert ids[0] == 1
    assert ids[1:65] == [7] * 64
    assert ids[65] == 2
    assert ids[-1] == 2
    # 512 - [CLS] - 64 title tokens - 2 x [SEP] leaves 445 for the body: 333 from its start, 112 from its end.
    assert ids[66 : 66 + 333] == list(range(100, 433))
    assert ids[66 + 333 : -1] == list(range(1888, 2000))


def test_no_title_gives_cls_body_sep() -> None:
    assert build_input([], [5, 6, 7], cls_id=1, sep_id=2) == ([1, 5, 6, 7, 2], False)


def test_a_short_title_and_body_are_joined_whole() -> None:
    assert build_input([8, 9], [5, 6, 7], cls_id=1, sep_id=2) == ([1, 8, 9, 2, 5, 6, 7, 2], False)


def test_a_title_that_is_cut_is_reported() -> None:
    ids, cut = build_input([7] * 65, [5], cls_id=1, sep_id=2)

    assert cut
    assert ids == [1] + [7] * 64 + [2, 5, 2]


@pytest.mark.parametrize("title", [0, 3, 64, 200])
@pytest.mark.parametrize("body", [0, 1, 40, 41, 5000])
def test_the_input_never_exceeds_max_length(title: int, body: int) -> None:
    ids, cut = build_input([7] * title, [9] * body, cls_id=1, sep_id=2, max_length=48, max_title=8)

    assert len(ids) <= 48
    kept_title = min(title, 8)
    room = 48 - 2 - (kept_title + 1 if kept_title else 0)
    assert cut == (title > 8 or body > room)
    assert ids.count(9) == min(body, room)


def test_an_input_spec_builds_with_its_own_settings() -> None:
    spec = InputSpec(
        max_length=10, max_title_tokens=2, head_share=0.5, cls_id=1, sep_id=2, pad_id=0
    )

    ids, cut = spec.build([7, 7, 7], list(range(100, 120)))

    # [CLS] 7 7 [SEP] leaves 5 for the body: 2 from its start, 3 from its end, then [SEP]
    assert ids == [1, 7, 7, 2, 100, 101, 117, 118, 119, 2]
    assert cut


def test_pad_batch_pads_to_the_longest_row_and_masks_the_padding() -> None:
    input_ids, mask = pad_batch([[5, 6, 7], [8]], pad_id=0)

    assert input_ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert input_ids.dtype == mask.dtype == "int64"
