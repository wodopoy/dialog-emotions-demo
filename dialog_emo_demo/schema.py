from __future__ import annotations

from pathlib import Path

import pandas as pd

EMOTION_GROUPS = ("joy", "warmth", "sadness", "anger", "anxiety", "neutral")

REQUIRED_COLUMNS = (
    "turn_index",
    "timestamp",
    "sender",
    "text",
    *EMOTION_GROUPS,
)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_dialog.csv"


class DialogDataError(ValueError):
    """Raised when a dialogue CSV does not match the expected contract."""


def load_dialog_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_dialog_frame(frame)


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
        out_of_range = ~cleaned[column].between(0, 1)
        if out_of_range.any():
            raise DialogDataError(f"Column {column} must contain values in [0, 1]")

    return cleaned.sort_values("turn_index", kind="stable").reset_index(drop=True)


def sender_choices(frame: pd.DataFrame) -> list[str]:
    senders = sorted(sender for sender in frame["sender"].dropna().unique() if str(sender).strip())
    return ["Все", *senders]


def filter_by_sender(frame: pd.DataFrame, sender: str) -> pd.DataFrame:
    if not sender or sender == "Все":
        return frame
    return frame[frame["sender"] == sender].reset_index(drop=True)
