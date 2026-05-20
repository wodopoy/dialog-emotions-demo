from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dialog_emo_demo.schema import EMOTION_GROUPS

EMOTION_COLORS = {
    "joy": "#f5b700",
    "warmth": "#2ca25f",
    "sadness": "#4c78a8",
    "anger": "#d1495b",
    "anxiety": "#8e63ce",
    "neutral": "#6b7280",
}

EMOTION_LABELS = {
    "joy": "Радость",
    "warmth": "Тепло",
    "sadness": "Грусть",
    "anger": "Злость",
    "anxiety": "Тревога",
    "neutral": "Нейтральность",
}


def smooth_emotions(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    window = max(int(window), 1)
    smoothed = frame.loc[:, EMOTION_GROUPS].rolling(window=window, min_periods=1).mean()
    return smoothed.reset_index(drop=True)


def build_emotion_figure(frame: pd.DataFrame, window: int) -> go.Figure:
    smoothed = smooth_emotions(frame, window)
    figure = go.Figure()

    hover_text = [
        f"{row.sender}<br>{row.timestamp}<br>{_shorten(row.text)}"
        for row in frame.itertuples(index=False)
    ]

    for emotion in EMOTION_GROUPS:
        figure.add_trace(
            go.Scatter(
                x=frame["turn_index"],
                y=smoothed[emotion],
                mode="lines+markers",
                name=EMOTION_LABELS[emotion],
                line={"color": EMOTION_COLORS[emotion], "width": 3},
                marker={"size": 7},
                customdata=hover_text,
                hovertemplate="%{customdata}<br>"
                + f"{EMOTION_LABELS[emotion]}: "
                + "%{y:.2f}<extra></extra>",
            )
        )

    figure.update_layout(
        margin={"l": 44, "r": 20, "t": 24, "b": 44},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fbfbf8",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        xaxis_title="Сообщение",
        yaxis_title="Интенсивность",
        yaxis={"range": [0, 1], "tickformat": ".0%"},
        font={"family": "Inter, system-ui, sans-serif", "size": 13},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e9e7df")
    figure.update_yaxes(showgrid=True, gridcolor="#e9e7df")
    return figure


def _shorten(text: str, limit: int = 96) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}..."
