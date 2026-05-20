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
.summary-card {
    --block-background-fill: #fff7dc;
    --background-fill-secondary: #fff7dc;
    border: 1px solid #ffd57a;
    border-radius: 8px;
    background: #fff7dc;
    padding: 12px 14px 10px;
    margin: 0 0 12px;
    box-shadow: 0 8px 20px rgba(92, 66, 0, 0.08);
}
.summary-card .block,
.summary-card .form,
.summary-card .wrap,
.summary-card .prose {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
.summary-card .block {
    padding: 0 !important;
}
.summary-head {
    align-items: center;
    background: #fff7dc !important;
    gap: 10px;
    margin-bottom: 8px;
}
.summary-title {
    color: #6f5200;
    font-size: 13px;
    font-weight: 700;
    line-height: 1.2;
    white-space: nowrap;
}
.summary-title p {
    color: #6f5200;
    font-size: 12px;
    font-weight: 700;
    margin: 0;
    white-space: nowrap;
}
.sender-compact {
    flex: 1 1 auto;
    max-width: 210px;
}
.sender-compact input,
.sender-compact [role="combobox"] {
    background: #ffffff !important;
    border: 1px solid #f0ca70 !important;
    border-radius: 8px !important;
    min-height: 34px !important;
}
.slice-card .message-meta {
    color: #6f5200;
}
.slice-card {
    background: #fff7dc;
}
.slice-details {
    display: grid;
    grid-template-columns: 1fr 1fr;
    column-gap: 14px;
    margin-top: 10px;
    border-top: 1px solid #f3d48b;
    padding-top: 8px;
}
.slice-details .prob-row {
    grid-template-columns: 92px minmax(58px, 1fr) 34px;
    gap: 6px;
    margin: 4px 0;
}
.slice-details .prob-track {
    height: 8px;
}
.side-panel {
    position: relative;
    min-height: 78vh;
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
    height: calc(100vh - 326px);
    max-height: calc(100vh - 326px);
    min-height: 390px;
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

            with gr.Column(scale=5, elem_classes=["side-panel"]):
                with gr.Group(elem_classes=["summary-card"]):
                    with gr.Row(elem_classes=["summary-head"]):
                        stats_title = gr.HTML(
                            value=render_stats_title("Все"),
                            elem_classes=["summary-title"],
                        )
                        sender = gr.Dropdown(
                            choices=sender_choices(default_frame),
                            value="Все",
                            interactive=True,
                            show_label=False,
                            container=False,
                            min_width=140,
                            elem_classes=["sender-compact"],
                        )
                    stats_view = gr.HTML(value=render_sender_stats(default_frame))
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
            inputs=[upload, smoothing, graph_mode, focus],
            outputs=[sender, plot, stats_title, stats_view, messages],
        )
        sender.change(
            fn=update_view,
            inputs=[upload, sender, smoothing, graph_mode, focus],
            outputs=[plot, stats_title, stats_view, messages],
        )
        for control in (smoothing, graph_mode, focus):
            control.change(
                fn=update_view,
                inputs=[upload, sender, smoothing, graph_mode, focus],
                outputs=[plot, stats_title, stats_view, messages],
            )

    return app


def load_uploaded_dialog(
    uploaded_file: Any,
    smoothing_window: int,
    graph_mode: str,
    focus_label: str,
) -> tuple[Any, Any, str, str, str]:
    frame = _load_frame(uploaded_file)
    choices = sender_choices(frame)
    return (
        gr.update(choices=choices, value="Все"),
        build_emotion_figure(frame, smoothing_window, graph_mode, focus_label),
        render_stats_title("Все"),
        render_sender_stats(frame),
        render_message_blocks(frame),
    )


def update_view(
    uploaded_file: Any,
    sender: str,
    smoothing_window: int,
    graph_mode: str,
    focus_label: str,
) -> tuple[Any, str, str, str]:
    frame = filter_by_sender(_load_frame(uploaded_file), sender)
    return (
        build_emotion_figure(frame, smoothing_window, graph_mode, focus_label),
        render_stats_title(sender or "Все"),
        render_sender_stats(frame),
        render_message_blocks(frame),
    )


def render_stats_title(sender: str) -> str:
    return f"Статистика: {escape(sender or 'Все')}"


def render_sender_stats(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<div class="slice-card">Нет сообщений для статистики.</div>'

    averages = frame.loc[:, EMOTION_GROUPS].mean()
    details = "".join(
        _render_probability_value(emotion, float(averages[emotion]))
        for emotion in EMOTION_GROUPS
    )
    count = len(frame)
    return f"""
    <section class="slice-card">
        <div class="message-meta">
            <strong>{count} сообщений</strong>
            <span>среднее распределение</span>
        </div>
        <div class="slice-details">{details}</div>
    </section>
    """


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
    return _render_probability_value(emotion, value)


def _render_probability_value(emotion: str, value: float) -> str:
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
