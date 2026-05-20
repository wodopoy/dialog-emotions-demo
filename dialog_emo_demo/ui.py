from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from dialog_emo_demo.plotting import EMOTION_COLORS, EMOTION_LABELS, build_emotion_figure
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
}
.demo-shell {
    min-height: calc(100vh - 120px);
}
.message-scroll {
    max-height: 72vh;
    overflow-y: auto;
    padding-right: 8px;
}
.message-card {
    border: 1px solid #dedbd2;
    border-radius: 8px;
    background: #fffefa;
    padding: 12px 14px;
    margin-bottom: 10px;
    transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
}
.message-card:hover,
.message-card:focus-within {
    border-color: #2f6f6d;
    box-shadow: 0 8px 22px rgba(31, 42, 38, 0.10);
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
    display: none;
    margin-top: 12px;
    border-top: 1px solid #ebe7dd;
    padding-top: 10px;
}
.message-card:hover .message-details,
.message-card:focus-within .message-details {
    display: block;
}
.prob-row {
    display: grid;
    grid-template-columns: 112px 1fr 44px;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
    font-size: 12px;
}
.prob-track {
    height: 8px;
    border-radius: 999px;
    background: #ece8df;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
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
            "текст чанка, отправителя, время и шесть независимых вероятностей."
        )

        with gr.Row(equal_height=False, elem_classes=["demo-shell"]):
            with gr.Column(scale=7):
                plot = gr.Plot(
                    value=build_emotion_figure(default_frame, window=3),
                    label="Эмоциональные траектории",
                )
                smoothing = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=3,
                    step=1,
                    label="Длина сглаживающего окна",
                )

            with gr.Column(scale=5):
                upload = gr.File(label="CSV с диалогом", file_types=[".csv"])
                sender = gr.Dropdown(
                    label="Отправитель",
                    choices=sender_choices(default_frame),
                    value="Все",
                    interactive=True,
                )
                messages = gr.HTML(
                    value=render_message_blocks(default_frame),
                    label="Сообщения",
                )

        upload.change(
            fn=load_uploaded_dialog,
            inputs=[upload, smoothing],
            outputs=[sender, plot, messages],
        )
        sender.change(
            fn=update_view,
            inputs=[upload, sender, smoothing],
            outputs=[plot, messages],
        )
        smoothing.change(
            fn=update_view,
            inputs=[upload, sender, smoothing],
            outputs=[plot, messages],
        )

    return app


def load_uploaded_dialog(uploaded_file: Any, smoothing_window: int) -> tuple[Any, Any, str]:
    frame = _load_frame(uploaded_file)
    choices = sender_choices(frame)
    return (
        gr.update(choices=choices, value="Все"),
        build_emotion_figure(frame, smoothing_window),
        render_message_blocks(frame),
    )


def update_view(uploaded_file: Any, sender: str, smoothing_window: int) -> tuple[Any, str]:
    frame = filter_by_sender(_load_frame(uploaded_file), sender)
    return build_emotion_figure(frame, smoothing_window), render_message_blocks(frame)


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


def _uploaded_path(uploaded_file: Any) -> Path | None:
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)
    name = getattr(uploaded_file, "name", None)
    return Path(name) if name else None
