from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dialog_emo_demo.schema import EMOTION_GROUPS

EMOTION_COLORS = {
    "joy": "#ffb000",
    "warmth": "#00a86b",
    "sadness": "#1e88e5",
    "anger": "#ff3b30",
    "anxiety": "#9b5cff",
    "neutral": "#4b5563",
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

    for emotion in EMOTION_GROUPS:
        figure.add_trace(
            go.Scatter(
                x=frame["turn_index"],
                y=smoothed[emotion],
                mode="lines+markers",
                name=EMOTION_LABELS[emotion],
                line={"color": EMOTION_COLORS[emotion], "width": 3.4},
                marker={
                    "size": 8,
                    "color": EMOTION_COLORS[emotion],
                    "line": {"color": "#ffffff", "width": 1.5},
                },
                hovertemplate=f"<b>{EMOTION_LABELS[emotion]}</b><br>%{{y:.2f}}<extra></extra>",
            )
        )

    figure.update_layout(
        margin={"l": 52, "r": 20, "t": 14, "b": 48},
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffaf0",
        hovermode="closest",
        hoverdistance=18,
        spikedistance=-1,
        hoverlabel={
            "bgcolor": "#111827",
            "bordercolor": "#111827",
            "font": {"color": "#ffffff", "size": 14},
        },
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
    figure.update_xaxes(showgrid=True, gridcolor="#efe7d2", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#efe7d2", zeroline=False)
    return figure
