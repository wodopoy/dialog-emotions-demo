import pandas as pd
import pytest

from dialog_emo_demo.schema import (
    DEFAULT_DATA_PATH,
    DialogDataError,
    EMOTION_GROUPS,
    filter_by_sender,
    load_dialog_csv,
    sender_choices,
    validate_dialog_frame,
)


def test_loads_synthetic_dialog_contract() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH)

    assert len(frame) == 20
    assert list(frame.columns) == [
        "turn_index",
        "timestamp",
        "sender",
        "text",
        *EMOTION_GROUPS,
    ]
    assert frame.loc[:, EMOTION_GROUPS].min().min() >= 0
    assert frame.loc[:, EMOTION_GROUPS].max().max() <= 1
    assert frame.loc[:, EMOTION_GROUPS].sum(axis=1).sub(1).abs().max() < 1e-4


def test_rejects_missing_required_column() -> None:
    frame = pd.DataFrame({"turn_index": [0]})

    with pytest.raises(DialogDataError, match="Missing required columns"):
        validate_dialog_frame(frame)


def test_rejects_probability_outside_unit_range() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH)
    frame.loc[0, "joy"] = 1.4

    with pytest.raises(DialogDataError, match="joy"):
        validate_dialog_frame(frame)


def test_rejects_probabilities_that_do_not_sum_to_one() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH)
    frame.loc[0, "joy"] = 0.0

    with pytest.raises(DialogDataError, match="sum to 1"):
        validate_dialog_frame(frame)


def test_sender_choices_and_filtering() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH)

    assert sender_choices(frame) == ["Все", "Аня", "Илья", "Марина"]
    filtered = filter_by_sender(frame, "Аня")

    assert set(filtered["sender"]) == {"Аня"}
    assert len(filtered) > 1
