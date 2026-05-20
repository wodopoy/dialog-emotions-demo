from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dialog_emo_demo.schema import EMOTION_GROUPS

GRAPH_MODES = ("Линии", "Площади")
FOCUS_ALL = "Все эмоции"

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

FOCUS_CHOICES = (FOCUS_ALL, *[EMOTION_LABELS[emotion] for emotion in EMOTION_GROUPS])


def smooth_emotions(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    window = max(int(window), 1)
    smoothed = frame.loc[:, EMOTION_GROUPS].rolling(window=window, min_periods=1).mean()
    return smoothed.reset_index(drop=True)


def build_emotion_figure(
    frame: pd.DataFrame,
    window: int,
    graph_mode: str = "Линии",
    focus_label: str = FOCUS_ALL,
) -> go.Figure:
    smoothed = smooth_emotions(frame, window)
    figure = go.Figure()
    focus_emotion = _emotion_from_label(focus_label)
    is_area = graph_mode == "Площади"

    for emotion in EMOTION_GROUPS:
        focused = focus_emotion in (None, emotion)
        opacity = 1 if focused else 0.22
        line_width = 4.2 if focused and focus_emotion else 2.3 if is_area else 3.2
        figure.add_trace(
            go.Scatter(
                x=frame["turn_index"],
                y=smoothed[emotion],
                mode="lines" if is_area else "lines+markers",
                name=EMOTION_LABELS[emotion],
                stackgroup="emotion" if is_area else None,
                groupnorm=None,
                fillcolor=_rgba(EMOTION_COLORS[emotion], 0.62 if focused else 0.16),
                line={"color": EMOTION_COLORS[emotion], "width": line_width},
                marker={
                    "size": 8 if focused else 5,
                    "color": EMOTION_COLORS[emotion],
                    "line": {"color": "#ffffff", "width": 1.5},
                },
                opacity=opacity,
                hovertemplate=f"<b>{EMOTION_LABELS[emotion]}</b><br>%{{y:.2f}}<extra></extra>",
            )
        )

    figure.update_layout(
        margin={"l": 52, "r": 20, "t": 14, "b": 48},
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffaf0",
        hovermode="x unified",
        hoverdistance=18,
        spikedistance=0,
        hoverlabel={
            "bgcolor": "rgba(17, 24, 39, 0.72)",
            "bordercolor": "rgba(17, 24, 39, 0.72)",
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
        yaxis_title="Доля",
        yaxis={"range": [0, 1], "tickformat": ".0%"},
        font={"family": "Inter, system-ui, sans-serif", "size": 13},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#efe7d2", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#efe7d2", zeroline=False)
    return figure


def build_slice_figure(frame: pd.DataFrame, turn_index: int) -> go.Figure:
    if frame.empty:
        return _empty_slice_figure()

    row = _nearest_row(frame, turn_index)
    values = [float(getattr(row, emotion)) for emotion in EMOTION_GROUPS]
    labels = [EMOTION_LABELS[emotion] for emotion in EMOTION_GROUPS]
    colors = [EMOTION_COLORS[emotion] for emotion in EMOTION_GROUPS]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors},
            text=[f"{value:.2f}" for value in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title={
            "text": f"Срез #{int(row.turn_index)} · {row.sender}",
            "x": 0,
            "font": {"size": 14},
        },
        margin={"l": 92, "r": 34, "t": 34, "b": 28},
        height=210,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis={"range": [0, 1], "tickformat": ".0%", "title": None},
        yaxis={"title": None, "autorange": "reversed"},
        font={"family": "Inter, system-ui, sans-serif", "size": 12},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#eee9dc", zeroline=False)
    figure.update_yaxes(showgrid=False)
    return figure


def max_turn_index(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame["turn_index"].max())


def _nearest_row(frame: pd.DataFrame, turn_index: int):
    if frame.empty:
        raise ValueError("Cannot build a slice for an empty frame")
    distances = (frame["turn_index"] - int(turn_index)).abs()
    return frame.loc[distances.idxmin()]


def _empty_slice_figure() -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title={"text": "Срез недоступен", "x": 0, "font": {"size": 14}},
        height=240,
        margin={"l": 92, "r": 34, "t": 34, "b": 28},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def _emotion_from_label(label: str) -> str | None:
    for emotion, emotion_label in EMOTION_LABELS.items():
        if label == emotion_label:
            return emotion
    return None


def _rgba(hex_color: str, alpha: float) -> str:
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"
