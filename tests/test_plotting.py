import pandas as pd

from dialog_emo_demo.plotting import build_emotion_figure, smooth_emotions
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
            "warmth": [0.4, 0.5],
            "sadness": [0.1, 0.1],
            "anger": [0.0, 0.2],
            "anxiety": [0.3, 0.2],
            "neutral": [0.6, 0.7],
        }
    )

    figure = build_emotion_figure(frame, window=1)

    assert len(figure.data) == len(EMOTION_GROUPS)
    assert figure.layout.yaxis.range == (0, 1)
