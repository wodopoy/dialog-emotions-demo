from gradio import Blocks

from app import demo
from dialog_emo_demo.schema import DEFAULT_DATA_PATH, EMOTION_GROUPS, load_dialog_csv
from dialog_emo_demo.ui import render_message_blocks, render_sender_stats


def test_app_builds() -> None:
    assert isinstance(demo, Blocks)


def test_message_cards_include_emotion_bars() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH).head(1)

    html = render_message_blocks(frame)

    assert html.count('class="prob-row"') == len(EMOTION_GROUPS)
    assert "message-details" in html


def test_sender_stats_card_aggregates_emotion_bars() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH)

    html = render_sender_stats(frame)

    assert "slice-card" in html
    assert html.count('class="prob-row"') == len(EMOTION_GROUPS)
    assert "сообщений" in html
