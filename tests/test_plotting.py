import pandas as pd

from dialog_emo_demo.plotting import build_emotion_figure, build_slice_figure, smooth_emotions
from dialog_emo_demo.schema import EMOTION_GROUPS


def test_smooth_emotions_uses_trailing_window() -> None:
    frame = pd.DataFrame(
        {
            emotion: [0.0, 0.5, 1.0]
            for emotion in EMOTION_GROUPS
        }
    )

    smoothed = smooth_emotions(frame, window=2)

    assert smoothed["joy"].tolist() == [0.0, 0.25, 0.75]


def test_build_emotion_figure_has_one_trace_per_group() -> None:
    frame = pd.DataFrame(
        {
            "turn_index": [0, 1],
            "timestamp": ["2026-05-20 18:00", "2026-05-20 18:03"],
            "sender": ["Аня", "Илья"],
            "text": ["Привет", "Привет"],
            "joy": [0.2, 0.3],
            "warmth": [0.3, 0.2],
            "sadness": [0.1, 0.1],
            "anger": [0.0, 0.1],
            "anxiety": [0.3, 0.2],
            "neutral": [0.1, 0.1],
        }
    )

    figure = build_emotion_figure(frame, window=1)

    assert len(figure.data) == len(EMOTION_GROUPS)
    assert figure.layout.hovermode == "x unified"
    low, high = figure.layout.yaxis.range
    assert low == 0
    assert 0.3 < high <= 1.0


def test_build_area_figure_uses_stacked_traces() -> None:
    frame = pd.DataFrame(
        {
            "turn_index": [0],
            "timestamp": ["2026-05-20 18:00"],
            "sender": ["Аня"],
            "text": ["Привет"],
            "joy": [0.2],
            "warmth": [0.3],
            "sadness": [0.1],
            "anger": [0.0],
            "anxiety": [0.3],
            "neutral": [0.1],
        }
    )

    figure = build_emotion_figure(frame, window=1, graph_mode="Площади")

    assert figure.layout.hovermode == "x unified"
    assert all(trace.stackgroup == "emotion" for trace in figure.data)


def test_build_slice_figure_has_one_bar_trace() -> None:
    frame = pd.DataFrame(
        {
            "turn_index": [0],
            "timestamp": ["2026-05-20 18:00"],
            "sender": ["Аня"],
            "text": ["Привет"],
            "joy": [0.2],
            "warmth": [0.3],
            "sadness": [0.1],
            "anger": [0.0],
            "anxiety": [0.3],
            "neutral": [0.1],
        }
    )

    figure = build_slice_figure(frame, turn_index=0)

    assert len(figure.data) == 1
    assert len(figure.data[0].x) == len(EMOTION_GROUPS)
