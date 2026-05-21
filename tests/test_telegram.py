from __future__ import annotations

import json

import pytest

from dialog_emo_demo.telegram import load_telegram_export


@pytest.fixture
def export_path(tmp_path):
    payload = {
        "name": "Test",
        "type": "personal_chat",
        "id": 1,
        "messages": [
            {"id": 1, "type": "service", "date": "2026-05-20T18:00:00", "action": "create_group", "text": ""},
            {"id": 2, "type": "message", "date": "2026-05-20T18:01:00", "from": "Аня", "text": "Привет!"},
            {"id": 3, "type": "message", "date": "2026-05-20T18:02:00", "from": "Аня",
             "media_type": "sticker", "text": ""},
            {"id": 4, "type": "message", "date": "2026-05-20T18:03:00", "from": "Илья",
             "media_type": "voice_message", "text": ""},
            {"id": 5, "type": "message", "date": "2026-05-20T18:04:00", "from": "Илья",
             "media_type": "video_message", "text": ""},
            {"id": 6, "type": "message", "date": "2026-05-20T18:05:00", "from": "Илья",
             "text": ["Привет ", {"type": "mention", "text": "@anya"}, " 👍"]},
            {"id": 7, "type": "message", "date": "2026-05-20T18:06:00", "from": "Аня",
             "text": [{"type": "custom_emoji", "text": "🔥", "document_id": "1"}]},
            {"id": 8, "type": "message", "date": "2026-05-20T18:07:00", "from": "Аня",
             "media_type": "photo", "text": "Лови фотку"},
            {"id": 9, "type": "message", "date": "2026-05-20T18:08:00", "from": "Аня", "text": "   "},
        ],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_drops_service_and_unwanted_media(export_path):
    frame = load_telegram_export(export_path)
    assert list(frame["text"]) == [
        "Привет!",
        "Привет @anya 👍",
        "🔥",
        "Лови фотку",
    ]


def test_turn_index_is_dense(export_path):
    frame = load_telegram_export(export_path)
    assert list(frame["turn_index"]) == [0, 1, 2, 3]


def test_keeps_sender_and_timestamp(export_path):
    frame = load_telegram_export(export_path)
    assert list(frame["sender"]) == ["Аня", "Илья", "Аня", "Аня"]
    assert frame["timestamp"].iloc[0] == "2026-05-20T18:01:00"
