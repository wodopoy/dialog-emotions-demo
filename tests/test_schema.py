import pandas as pd
import pytest

from dialog_emo_demo.schema import (
    DEFAULT_DATA_PATH,
    DialogDataError,
    EMOTION_GROUPS,
    filter_by_sender,
    load_dialog_csv,
    prepare_dialog_frame,
    sender_choices,
    validate_dialog_frame,
)


def test_loads_synthetic_dialog_contract() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH)

    assert len(frame) > 0
    assert list(frame.columns) == [
        "turn_index",
        "timestamp",
        "sender",
        "text",
        *EMOTION_GROUPS,
    ]
    assert frame.loc[:, EMOTION_GROUPS].min().min() >= 0
    assert frame.loc[:, EMOTION_GROUPS].max().max() <= 1
    assert frame.loc[:, EMOTION_GROUPS].sum(axis=1).sub(1).abs().max() < 1e-3


def test_rejects_missing_required_column() -> None:
    frame = pd.DataFrame({"turn_index": [0]})

    with pytest.raises(DialogDataError, match="Missing required columns"):
        validate_dialog_frame(frame)


def test_accepts_logits_and_softmaxes_them() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH)
    frame.loc[0, list(EMOTION_GROUPS)] = [3.0, 1.0, -2.0, -1.0, 0.5, 0.0]

    prepared = prepare_dialog_frame(frame)

    assert prepared.loc[0, list(EMOTION_GROUPS)].sum() == pytest.approx(1.0)
    assert prepared.loc[0, "joy"] > prepared.loc[0, "warmth"]


def test_temperature_changes_distribution_sharpness() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH)
    frame.loc[0, list(EMOTION_GROUPS)] = [3.0, 1.0, -2.0, -1.0, 0.5, 0.0]

    cold = prepare_dialog_frame(frame, temperature=0.5)
    hot = prepare_dialog_frame(frame, temperature=5.0)

    assert cold.loc[0, "joy"] > hot.loc[0, "joy"]


def test_sender_choices_and_filtering() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH)

    choices = sender_choices(frame)
    target_sender = choices[1]
    assert choices[0] == "Все"
    assert len(choices) > 1

    filtered = filter_by_sender(frame, target_sender)

    assert set(filtered["sender"]) == {target_sender}
    assert len(filtered) > 1
