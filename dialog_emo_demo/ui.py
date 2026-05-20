from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from dialog_emo_demo.plotting import (
    FOCUS_ALL,
    FOCUS_CHOICES,
    GRAPH_MODES,
    EMOTION_COLORS,
    EMOTION_LABELS,
    build_emotion_figure,
    build_slice_figure,
    max_turn_index,
)
from dialog_emo_demo.schema import (
    DEFAULT_DATA_PATH,
    EMOTION_GROUPS,
    DialogDataError,
    filter_by_sender,
    load_dialog_csv,
    sender_choices,
)

CSS = """
.gradio-container {
    max-width: none !important;
    background: #f8f7f2;
}
.main-title h1 {
    margin-bottom: 4px;
    letter-spacing: 0;
}
.main-title p {
    color: #5f6368;
    margin-top: 0;
}
.demo-shell {
    min-height: calc(100vh - 120px);
    gap: 18px;
}
.plot-panel,
.side-panel {
    background: #ffffff;
    border: 1px solid #e3e0d7;
    border-radius: 8px;
    padding: 14px;
}
.plot-panel {
    box-shadow: 0 10px 30px rgba(37, 40, 35, 0.06);
}
.plot-controls {
    align-items: end;
    gap: 10px;
}
.slice-panel {
    margin: 4px 0 10px;
}
.side-panel {
    position: relative;
    min-height: 78vh;
}
.side-controls {
    align-items: end;
    gap: 10px;
    margin-bottom: 12px;
}
.side-controls .wrap {
    gap: 10px;
}
.upload-corner {
    position: absolute;
    right: 14px;
    bottom: 14px;
    z-index: 4;
    display: flex;
    justify-content: flex-end;
    width: fit-content !important;
    margin-top: 10px;
}
.upload-corner > *,
.upload-corner button {
    flex: 0 0 auto !important;
    width: auto !important;
}
.upload-corner button {
    border-radius: 8px !important;
    padding: 6px 10px !important;
    font-size: 12px !important;
    min-height: 32px !important;
    min-width: 118px !important;
}
.message-scroll {
    height: 210px;
    max-height: 210px;
    min-height: 210px;
    overflow-y: auto;
    padding: 2px 8px 52px 2px;
}
.message-card {
    border: 1px solid #e2ded5;
    border-radius: 8px;
    background: #ffffff;
    padding: 12px 14px;
    margin-bottom: 10px;
    transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease, background 120ms ease;
}
.message-card:hover,
.message-card:focus-within {
    border-color: #ffb000;
    background: #fffdf6;
    box-shadow: 0 10px 22px rgba(37, 40, 35, 0.12);
    transform: translateY(-1px);
}
.message-meta {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    color: #625f57;
    font-size: 12px;
    margin-bottom: 8px;
}
.message-text {
    color: #1f2933;
    line-height: 1.45;
    font-size: 15px;
}
.message-details {
    max-height: 0;
    opacity: 0;
    overflow: hidden;
    margin-top: 12px;
    border-top: 1px solid #ebe7dd;
    padding-top: 0;
    transition: max-height 160ms ease, opacity 120ms ease, padding-top 120ms ease;
}
.message-card:hover .message-details,
.message-card:focus-within .message-details {
    max-height: 240px;
    opacity: 1;
    padding-top: 10px;
}
.prob-row {
    display: grid;
    grid-template-columns: 104px minmax(90px, 1fr) 42px;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
    font-size: 12px;
}
.prob-track {
    display: block;
    height: 9px;
    border-radius: 999px;
    background: #ece7dc;
    overflow: hidden;
}
.prob-fill {
    display: block;
    height: 100%;
    border-radius: 999px;
}
.empty-state {
    border: 1px dashed #cfc9bd;
    border-radius: 8px;
    color: #625f57;
    padding: 18px;
}
"""


def build_app() -> gr.Blocks:
    default_frame = load_dialog_csv(DEFAULT_DATA_PATH)

    with gr.Blocks(title="Dialog Emotion Timeline", fill_width=True) as app:
        gr.Markdown(
            "# Трекинг эмоциональной окраски диалога\n"
            "CSV-контракт отделяет интерфейс от модели: каждая строка уже содержит "
            "текст чанка, отправителя, время и распределение вероятностей по эмоциям.",
            elem_classes=["main-title"],
        )

        with gr.Row(equal_height=False, elem_classes=["demo-shell"]):
            with gr.Column(scale=7, elem_classes=["plot-panel"]):
                plot = gr.Plot(
                    value=build_emotion_figure(default_frame, window=3),
                    label="Эмоциональные траектории",
                )
                with gr.Row(elem_classes=["plot-controls"]):
                    graph_mode = gr.Radio(
                        choices=list(GRAPH_MODES),
                        value="Линии",
                        label="Вид",
                        interactive=True,
                    )
                    focus = gr.Dropdown(
                        choices=list(FOCUS_CHOICES),
                        value=FOCUS_ALL,
                        label="Фокус",
                        interactive=True,
                    )
                smoothing = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=3,
                    step=1,
                    label="Длина сглаживающего окна",
                )
                slice_turn = gr.Slider(
                    minimum=0,
                    maximum=max_turn_index(default_frame),
                    value=0,
                    step=1,
                    label="Срез сообщения",
                )

            with gr.Column(scale=5, elem_classes=["side-panel"]):
                with gr.Row(elem_classes=["side-controls"]):
                    sender = gr.Dropdown(
                        label="Отправитель",
                        choices=sender_choices(default_frame),
                        value="Все",
                        interactive=True,
                        scale=5,
                    )
                slice_plot = gr.Plot(
                    value=build_slice_figure(default_frame, turn_index=0),
                    label="Срез",
                    elem_classes=["slice-panel"],
                )
                messages = gr.HTML(
                    value=render_message_blocks(default_frame),
                    label="Сообщения",
                )
                with gr.Row(elem_classes=["upload-corner"]):
                    upload = gr.UploadButton(
                        "Загрузить CSV",
                        file_types=[".csv"],
                        type="filepath",
                        size="sm",
                        variant="secondary",
                        min_width=120,
                    )

        upload.upload(
            fn=load_uploaded_dialog,
            inputs=[upload, smoothing, graph_mode, focus, slice_turn],
            outputs=[sender, slice_turn, plot, slice_plot, messages],
        )
        sender.change(
            fn=update_view,
            inputs=[upload, sender, smoothing, graph_mode, focus, slice_turn],
            outputs=[slice_turn, plot, slice_plot, messages],
        )
        for control in (smoothing, graph_mode, focus, slice_turn):
            control.change(
                fn=update_view,
                inputs=[upload, sender, smoothing, graph_mode, focus, slice_turn],
                outputs=[slice_turn, plot, slice_plot, messages],
            )

    return app


def load_uploaded_dialog(
    uploaded_file: Any,
    smoothing_window: int,
    graph_mode: str,
    focus_label: str,
    slice_turn: int,
) -> tuple[Any, Any, Any, Any, str]:
    frame = _load_frame(uploaded_file)
    choices = sender_choices(frame)
    turn = _clamp_turn(frame, slice_turn)
    return (
        gr.update(choices=choices, value="Все"),
        _slice_slider_update(frame, turn),
        build_emotion_figure(frame, smoothing_window, graph_mode, focus_label),
        build_slice_figure(frame, turn),
        render_message_blocks(frame),
    )


def update_view(
    uploaded_file: Any,
    sender: str,
    smoothing_window: int,
    graph_mode: str,
    focus_label: str,
    slice_turn: int,
) -> tuple[Any, Any, Any, str]:
    frame = filter_by_sender(_load_frame(uploaded_file), sender)
    turn = _clamp_turn(frame, slice_turn)
    return (
        _slice_slider_update(frame, turn),
        build_emotion_figure(frame, smoothing_window, graph_mode, focus_label),
        build_slice_figure(frame, turn),
        render_message_blocks(frame),
    )


def render_message_blocks(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<div class="empty-state">Для выбранного отправителя нет сообщений.</div>'

    cards = []
    for row in frame.itertuples(index=False):
        cards.append(_render_message_card(row))
    return f'<div class="message-scroll">{"".join(cards)}</div>'


def _render_message_card(row: Any) -> str:
    details = "".join(_render_probability(row, emotion) for emotion in EMOTION_GROUPS)
    return f"""
    <article class="message-card" tabindex="0">
        <div class="message-meta">
            <strong>{escape(row.sender)}</strong>
            <span>{escape(row.timestamp)} · #{int(row.turn_index)}</span>
        </div>
        <div class="message-text">{escape(row.text)}</div>
        <div class="message-details">{details}</div>
    </article>
    """


def _render_probability(row: Any, emotion: str) -> str:
    value = float(getattr(row, emotion))
    percent = round(value * 100)
    color = EMOTION_COLORS[emotion]
    label = EMOTION_LABELS[emotion]
    return f"""
    <div class="prob-row">
        <span>{escape(label)}</span>
        <span class="prob-track">
            <span class="prob-fill" style="width: {percent}%; background: {color};"></span>
        </span>
        <span>{value:.2f}</span>
    </div>
    """


def _load_frame(uploaded_file: Any) -> pd.DataFrame:
    path = _uploaded_path(uploaded_file) or DEFAULT_DATA_PATH
    try:
        return load_dialog_csv(path)
    except (DialogDataError, ValueError, FileNotFoundError) as error:
        raise gr.Error(f"Не удалось прочитать CSV: {error}") from error


def _clamp_turn(frame: pd.DataFrame, turn_index: int) -> int:
    if frame.empty:
        return 0
    minimum = int(frame["turn_index"].min())
    maximum = int(frame["turn_index"].max())
    return min(max(int(turn_index), minimum), maximum)


def _slice_slider_update(frame: pd.DataFrame, turn_index: int) -> Any:
    if frame.empty:
        return gr.update(minimum=0, maximum=0, value=0)
    return gr.update(
        minimum=int(frame["turn_index"].min()),
        maximum=int(frame["turn_index"].max()),
        value=turn_index,
    )


def _uploaded_path(uploaded_file: Any) -> Path | None:
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)
    name = getattr(uploaded_file, "name", None)
    return Path(name) if name else None
