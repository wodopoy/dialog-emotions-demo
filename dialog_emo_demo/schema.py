from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

EMOTION_GROUPS = ("joy", "warmth", "sadness", "anger", "anxiety", "neutral")
PROBABILITY_SUM_TOLERANCE = 1e-3

REQUIRED_COLUMNS = (
    "turn_index",
    "timestamp",
    "sender",
    "text",
    *EMOTION_GROUPS,
)

DEFAULT_DATA_PATH = Path(
    os.environ.get("DIALOG_CSV")
    or Path(__file__).resolve().parent.parent / "data" / "synthetic_dialog.csv"
)


class DialogDataError(ValueError):
    """Raised when a dialogue CSV does not match the expected contract."""


def load_raw_dialog_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_dialog_frame(frame)


def load_dialog_csv(path: str | Path, temperature: float = 1.0) -> pd.DataFrame:
    return prepare_dialog_frame(load_raw_dialog_csv(path), temperature=temperature)


def validate_dialog_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DialogDataError(f"Missing required columns: {', '.join(missing)}")

    cleaned = frame.loc[:, REQUIRED_COLUMNS].copy()
    if cleaned.empty:
        raise DialogDataError("CSV must contain at least one dialogue row")

    cleaned["turn_index"] = pd.to_numeric(cleaned["turn_index"], errors="raise").astype(int)
    for column in ("timestamp", "sender", "text"):
        cleaned[column] = cleaned[column].fillna("").astype(str)

    for column in EMOTION_GROUPS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
        if not np.isfinite(cleaned[column]).all():
            raise DialogDataError(f"Column {column} must contain finite numeric values")

    return cleaned.sort_values("turn_index", kind="stable").reset_index(drop=True)


def prepare_dialog_frame(frame: pd.DataFrame, temperature: float = 1.0) -> pd.DataFrame:
    cleaned = validate_dialog_frame(frame)
    temperature = float(temperature)
    if temperature <= 0:
        raise DialogDataError("Temperature must be positive")

    scores = cleaned.loc[:, EMOTION_GROUPS].astype(float)
    row_sums = scores.sum(axis=1)
    probability_rows = (
        scores.ge(0).all(axis=1)
        & scores.le(1).all(axis=1)
        & ((row_sums - 1).abs() <= PROBABILITY_SUM_TOLERANCE)
    )

    logits = scores.copy()
    if probability_rows.any():
        logits.loc[probability_rows, :] = np.log(
            scores.loc[probability_rows, :].clip(lower=1e-12)
        )

    scaled = logits / temperature
    shifted = scaled.sub(scaled.max(axis=1), axis=0)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores.div(exp_scores.sum(axis=1), axis=0)

    prepared = cleaned.copy()
    prepared.loc[:, EMOTION_GROUPS] = probabilities.loc[:, EMOTION_GROUPS]
    return prepared


def sender_choices(frame: pd.DataFrame) -> list[str]:
    senders = sorted(sender for sender in frame["sender"].dropna().unique() if str(sender).strip())
    return ["Все", *senders]


def filter_by_sender(frame: pd.DataFrame, sender: str) -> pd.DataFrame:
    if not sender or sender == "Все":
        return frame
    return frame[frame["sender"] == sender].reset_index(drop=True)
