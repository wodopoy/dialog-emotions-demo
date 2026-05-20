from gradio import Blocks

from app import demo
from dialog_emo_demo.schema import DEFAULT_DATA_PATH, EMOTION_GROUPS, load_dialog_csv
from dialog_emo_demo.ui import render_message_blocks, render_slice_card


def test_app_builds() -> None:
    assert isinstance(demo, Blocks)


def test_message_cards_include_emotion_bars() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH).head(1)

    html = render_message_blocks(frame)

    assert html.count('class="prob-row"') == len(EMOTION_GROUPS)
    assert "message-details" in html


def test_slice_card_is_rendered_like_a_fixed_message_summary() -> None:
    frame = load_dialog_csv(DEFAULT_DATA_PATH)

    html = render_slice_card(frame, turn_index=int(frame.iloc[0]["turn_index"]))

    assert "slice-card" in html
    assert "Срез сообщения" in html
    assert html.count('class="prob-row"') == len(EMOTION_GROUPS)
