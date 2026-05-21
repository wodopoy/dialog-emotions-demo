from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import pandas as pd

from dialog_emo_demo.models import EmotionModel
from dialog_emo_demo.schema import (
    EMOTION_GROUPS,
    PROBABILITY_SUM_TOLERANCE,
    REQUIRED_COLUMNS,
    validate_dialog_frame,
)


def score_dialog(frame: pd.DataFrame, model: EmotionModel) -> pd.DataFrame:
    """Run `model.predict_proba` on every row's `text` and attach six emotion
    columns. If the model already returns probabilities (sum ≈ 1, all ≥ 0)
    we keep them as-is; otherwise we softmax."""
    columns: dict[str, list[float]] = {emotion: [] for emotion in EMOTION_GROUPS}
    for text in frame["text"].fillna("").astype(str):
        probabilities = _to_probabilities(model.predict_proba(text))
        for emotion in EMOTION_GROUPS:
            columns[emotion].append(probabilities[emotion])
    out = frame.copy()
    for emotion in EMOTION_GROUPS:
        out[emotion] = columns[emotion]
    return out


def write_scored_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    validated = validate_dialog_frame(frame.loc[:, list(REQUIRED_COLUMNS)])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output, index=False)
    return output


def _to_probabilities(scores: dict[str, float]) -> dict[str, float]:
    missing = [e for e in EMOTION_GROUPS if e not in scores]
    if missing:
        raise ValueError(f"Model output missing emotions: {missing}")
    values = {e: float(scores[e]) for e in EMOTION_GROUPS}
    total = sum(values.values())
    looks_like_proba = (
        all(v >= 0 for v in values.values())
        and total > 0
        and abs(total - 1.0) <= PROBABILITY_SUM_TOLERANCE
    )
    if looks_like_proba:
        return values
    return _softmax(values)


def _softmax(values: dict[str, float]) -> dict[str, float]:
    shift = max(values.values())
    exps = {e: math.exp(v - shift) for e, v in values.items()}
    total = sum(exps.values())
    return {e: exps[e] / total for e in EMOTION_GROUPS}


class _UniformModel(EmotionModel):
    def predict_proba(self, text: str) -> dict[str, float]:
        p = 1.0 / len(EMOTION_GROUPS)
        return {emotion: p for emotion in EMOTION_GROUPS}


MODEL_REGISTRY: dict[str, Callable[[], EmotionModel]] = {
    "uniform": _UniformModel,
}
