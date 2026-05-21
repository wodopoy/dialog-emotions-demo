from __future__ import annotations

import json

import pandas as pd
import pytest

from dialog_emo_demo.models import EmotionModel
from dialog_emo_demo.pipeline import MODEL_REGISTRY, score_dialog, write_scored_csv
from dialog_emo_demo.schema import EMOTION_GROUPS, load_dialog_csv
from dialog_emo_demo.telegram import load_telegram_export


def test_uniform_pipeline_end_to_end(tmp_path):
    payload = {
        "messages": [
            {"id": 1, "type": "message", "date": "2026-05-20T18:01:00", "from": "A", "text": "привет"},
            {"id": 2, "type": "message", "date": "2026-05-20T18:02:00", "from": "B", "text": "ага"},
        ]
    }
    tg_path = tmp_path / "result.json"
    tg_path.write_text(json.dumps(payload), encoding="utf-8")

    frame = load_telegram_export(tg_path)
    scored = score_dialog(frame, MODEL_REGISTRY["uniform"]())
    csv_path = write_scored_csv(scored, tmp_path / "out.csv")

    loaded = load_dialog_csv(csv_path)
    assert len(loaded) == 2
    for emotion in EMOTION_GROUPS:
        assert loaded[emotion].iloc[0] == pytest.approx(1 / 6)


def test_score_dialog_softmaxes_logit_outputs():
    class LogitModel(EmotionModel):
        def predict_proba(self, text):
            return {emotion: float(i) for i, emotion in enumerate(EMOTION_GROUPS)}

    frame = pd.DataFrame(
        {"turn_index": [0], "timestamp": ["t"], "sender": ["A"], "text": ["x"]}
    )
    scored = score_dialog(frame, LogitModel())
    row_sum = sum(float(scored[e].iloc[0]) for e in EMOTION_GROUPS)
    assert row_sum == pytest.approx(1.0)
    assert float(scored["neutral"].iloc[0]) > float(scored["joy"].iloc[0])


def test_score_dialog_keeps_valid_probabilities():
    class ProbaModel(EmotionModel):
        def predict_proba(self, text):
            return {
                "joy": 0.5, "warmth": 0.5, "sadness": 0.0,
                "anger": 0.0, "anxiety": 0.0, "neutral": 0.0,
            }

    frame = pd.DataFrame(
        {"turn_index": [0], "timestamp": ["t"], "sender": ["A"], "text": ["x"]}
    )
    scored = score_dialog(frame, ProbaModel())
    assert float(scored["joy"].iloc[0]) == pytest.approx(0.5)
    assert float(scored["sadness"].iloc[0]) == pytest.approx(0.0)
